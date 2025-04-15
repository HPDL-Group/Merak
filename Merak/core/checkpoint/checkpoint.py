# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com), Yck (eyichenke@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/v2.6/megatron/checkpointing.py

import torch
import torch.nn as nn
import random
import sys
import os
import numpy as np
import datetime
import json
import shutil
import torch.distributed as dist

from collections import OrderedDict
from typing import Optional, Tuple, Callable, Union, List, Dict
from dataclasses import asdict
from pathlib import Path

from Merak import print_rank_0
from .utils import (
    ensure_directory_exists,
    unwrap_model,
    same_layer_idx,
    get_zero_param_shapes,
    PeftType,
    ShardedSafetensorLoader
)
from .. import mpu
from ..recompute import get_rng_tracker

_CHECKPOINT_VERSION = None
TRANSFORMERS_MODEL_NAME = ["pytorch_model.bin", "adapter_model.bin"]

def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, \
            "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


def get_rank_ckpt_name(peft_id: str, zero_file: bool = False) -> str:
    dp_rank = f"{mpu.get_data_parallel_rank():05d}"
    pp_rank = f"{mpu.get_pipe_parallel_rank():03d}"
    mp_rank = f"{mpu.get_model_parallel_rank():02d}"

    if mpu.get_model_parallel_world_size() == 1 and \
       mpu.get_pipe_parallel_world_size() == 1:

        if zero_file:
            return f"zero_dp_rank_{dp_rank}_opt_states.pt"
        else:
            return f"{peft_id}model_optim.pt"
    elif mpu.get_pipe_parallel_world_size() == 1 and \
         mpu.get_model_parallel_world_size() != 1:

        if zero_file:
            return f"zero_dp_rank_{dp_rank}_mp_rank_{mp_rank}_opt_states.pt"
        else:
            return f"mp_rank_{mp_rank}/{peft_id}partial_model_optim.pt"
    else:
        if zero_file:
            return f"zero_dp_rank_{dp_rank}_mp_rank_{mp_rank}_pp_rank_{pp_rank}_opt_states.pt"
        else:
            return f"mp_rank_{mp_rank}_pp_rank_{pp_rank}/{peft_id}partial_model_optim.pt"

def get_checkpoint_name(
        checkpoints_path: str, iteration: int, args, release: bool = False,
        complete: bool = False, best: Optional[bool] = None, peft: bool = False
    ) -> Tuple[str, str]:
    """A unified checkpoint name."""
    if release:
        directory = 'release'
    elif best is not None:
        directory = 'best_model'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    if peft:
        peft_id = 'lora_'
    else:
        peft_id = ''
    # Use both the tensor and pipeline MP rank.
    model_ckpt_name = os.path.join(
        checkpoints_path,
        directory,
        get_rank_ckpt_name(peft_id)
    )
    opt_ckpt_name = os.path.join(
        checkpoints_path,
        directory,
        get_rank_ckpt_name(peft_id, zero_file=True)
    )
    if not args.zero_stage == 1:
        opt_ckpt_name = None
    return model_ckpt_name, opt_ckpt_name

def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 
                        'latest_checkpointed_iteration.txt')


def get_best_checkpoint_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'best_model_loss.txt')


def read_metadata(tracker_filename: str) -> Tuple[int, bool]:
    # Read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    # Get the max iteration retrieved across the ranks.
    if torch.cuda.is_available():
        iters = torch.LongTensor([iteration]).cuda()
    else:
        iters = torch.LongTensor([iteration])
    torch.distributed.all_reduce(iters, op=torch.distributed.ReduceOp.MAX)
    max_iter = iters[0].item()

    # We should now have all the same iteration.
    # If not, print a warning and chose the maximum
    # iteration across all ranks.
    if iteration != max_iter:
        print('WARNING: on rank {} found iteration {} in the '
              'metadata while max iteration across the ranks '
              'is {}, replacing it with max iteration.'.format(
                  mpu.get_pipe_parallel_rank(), iteration, max_iter), flush=True)
    return max_iter, release


def save_checkpoint(
        iteration: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        args,
        **kwargs):
    """Save a model checkpoint."""
    dtime = datetime.datetime.now().strftime('%Y-%m-%d')

    best_model = kwargs.get("best_model", None)
    peft_config = kwargs.get("peft_config", None)

    if best_model is not None:
        save_path = args.output_dir+'/best_ckpt'
        tracker_filename = get_best_checkpoint_filename(save_path)

        if not torch.distributed.is_initialized() or \
            torch.distributed.get_rank()==0:
            if os.path.isfile(tracker_filename):
                with open(tracker_filename, 'r') as f:
                    metastring = f.read().strip()
                    saved_best = eval(metastring)
                if best_model > saved_best:
                    sig = torch.tensor(0)
                    torch.distributed.broadcast(sig, src=0)
                    return

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            with open(tracker_filename, 'w') as f:
                f.write(str(best_model))

            sig = torch.tensor(1)
            torch.distributed.broadcast(sig, src=0)
        else:
            sig = torch.tensor(0)
            torch.distributed.broadcast(sig, src=0)
            if sig == 0:
                return

        print_rank_0('saving best model with loss {} checkpoint at \
                     iteration {:7d} to {}'.format(
            best_model, iteration, save_path))
    else:
        save_path = args.output_dir+'/{time}_ckpt'.format(time=dtime)

        print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
            iteration, save_path))

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)
    checkpoint_name, zero_ckpt_name = get_checkpoint_name(
                                        save_path,
                                        iteration, args,
                                        best=best_model,
                                        peft=peft_config is not None
                                    )

    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0:
        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        state_dict['iteration'] = iteration
        state_dict['model'] = model.state_dict()
        if kwargs is not None:
            for k, v in kwargs.items():
                state_dict[k] = v

        # Optimizer stuff.
        if not args.zero_stage == 1 and not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict['random_rng_state'] = random.getstate()
            state_dict['np_rng_state'] = np.random.get_state()
            state_dict['torch_rng_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
            state_dict['rng_tracker_states'] \
                = get_rng_tracker().get_states()


        # Save.
        ensure_directory_exists(checkpoint_name)
        if args.lora_config is not None or peft_config is not None:
            peft = True
            peft_state_dict = get_peft_model_state_dict(model, 
                                                        peft_config)

            output_path = os.path.join(
                os.path.dirname(checkpoint_name),
                "lora_config.json"
            )
            output_dict = asdict(peft_config)
            # converting set type to list
            for key, value in output_dict.items():
                if isinstance(value, set):
                    output_dict[key] = list(value)

            # save config
            with open(output_path, "w") as writer:
                writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

        else:
            peft = False
        torch.save(
            state_dict if not peft else peft_state_dict,
            checkpoint_name,
            _use_new_zipfile_serialization=False
        )

    if args.zero_stage == 1:
        try:
            ensure_directory_exists(zero_ckpt_name)
        except:
                print_rank_0(f"Failed creating zero checkpoint to {save_path}")
        dist.barrier()
            # Save zero checkpoint
        zero_sd = dict(
            optimizer_state_dict=optimizer.state_dict(),
            param_shapes=get_zero_param_shapes(optimizer, model)
        )
        torch.save(zero_sd, zero_ckpt_name)
        print_rank_0('zero checkpoint saved {}'.format(zero_ckpt_name))   
  

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f'successfully saved checkpoint at iteration {iteration} to {save_path}'
    )

    # And update the latest iteration
    if not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(save_path)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        args,
        verbose: bool = False,
        strict: bool = True
    ) -> Tuple[Union[int, OrderedDict, Optional[OrderedDict]]]:
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """

    load_dir = args.resume_from_checkpoint
    model = unwrap_model(model)

    # get transformers checkpoint file
    checkpoint_name = detect_checkpoint_format(load_dir)

    if checkpoint_name is None:
        # Read the tracker file and set the iteration.
        tracker_filename = get_checkpoint_tracker_filename(load_dir)

        # If no tracker file, return iretation zero.
        if not os.path.isfile(tracker_filename):
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('will not load any checkpoints and will start from '
                        'random')
            return 0, None, None
        # Otherwise, read the tracker file and either set the iteration or
        # mark it as a release checkpoint.
        iteration, release = read_metadata(tracker_filename)
        checkpoint_name, zero_ckpt_name = get_checkpoint_name(
                                            load_dir,
                                            iteration,
                                            args,
                                            release
                                        )
    elif checkpoint_name:
        checkpoint_name = checkpoint_name
        iteration, release = 0, True
    else:
        return 0, None, None

    if args.zero_stage == 1 and not release:
        if os.path.isfile(zero_ckpt_name):
            opt_state_dict = torch.load(zero_ckpt_name, map_location='cpu')
    else:
        opt_state_dict = None

    print_rank_0(f' loading checkpoint from {args.resume_from_checkpoint} '
                 f'at iteration {iteration}')

    # Load the checkpoint.
    if mpu.get_data_parallel_rank() == 0:
        print(checkpoint_name)
    try:
        if "safetensor" not in checkpoint_name:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        else:
            state_dict = ShardedSafetensorLoader(
                model, args.resume_from_checkpoint
            ).get_state_dict()
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    # check merak or transformer model
    state_dict, strict = check_state_dict(model, state_dict, args, verbose)

    # set checkpoint version
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))


    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0(
                    'A metadata file exists but unable to load '
                    'iteration from checkpoint {}, exiting'.format(
                        checkpoint_name
                    )
                )
                sys.exit()

    # Check arguments.
    # assert args.consumed_train_samples == 0
    # assert args.consumed_valid_samples == 0
    if 'args' in state_dict:
        checkpoint_args = state_dict['args']
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        # update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if 'model' in state_dict.keys():
        model.load_state_dict(state_dict['model'], strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)


    # Fix up query/key/value matrix ordering if needed
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    # fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                if args.zero_stage is not None:
                    optimizer.load_state_dict(
                        opt_state_dict['optimizer_state_dict'],
                        load_optimizer_states=True)
                    print(f'  successfully loaded zero optimizer checkpoint \
                           from {zero_ckpt_name} '
                 f'at iteration {iteration}')
                elif args.fp16:
                    optimizer.load_state_dict(
                        state_dict['optimizer'],
                        load_optimizer_states=True)
                else:
                    optimizer.load_state_dict(state_dict['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no_load_optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(zero_ckpt_name \
                                              if args.zero_stage is not None \
                                                else checkpoint_name))
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            # Check for empty states array
            if not state_dict['rng_tracker_states']:
                raise KeyError
            get_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no_load_rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed 
    # being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f'  successfully loaded checkpoint from {args.resume_from_checkpoint} '
        f'at iteration {iteration}'
    )

    return iteration, state_dict, opt_state_dict if args.zero_stage == 1 else None


def _sorted_checkpoints(
        output_dir: Optional[str] = None,
        checkpoint_prefix: str = "ckpt"
    ) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob("*_ckpt")]

    for path in glob_checkpoints:
        if not os.listdir(path):
            print_rank_0(f"Deleting empty checkpoint dirctory [{path}]")
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
        if "best_ckpt" in path:
            continue
        for x in Path(path).glob("iter_*"):
            ordering_and_checkpoint_path.append(x)

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)

    return checkpoints_sorted


def rotate_checkpoints(args, output_dir: Optional[str] = None):
    if args.save_total_limit is None or args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(output_dir=output_dir)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
    # we don't do to allow resuming.
    save_total_limit = args.save_total_limit
    if (
        args.save_total_limit == 1
    ):
        save_total_limit = 2

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

    # Only rank 0 to delete files.
    if dist.get_rank() == 0:
        for checkpoint in checkpoints_to_be_deleted:
            print_rank_0(f"Due to the setting of save_total_limit, "
                         f"old checkpoint [{checkpoint}] are deleted.")
            shutil.rmtree(checkpoint)
    dist.barrier()

def detect_checkpoint_format(model_dir=".", peft=False):
    """
    检测模型目录的检查点格式类型
    :param model_dir: 模型文件夹路径，默认为当前目录
    :return: "pytorch_bin" | "safetensors" | "both" | "none"
    """
    # 构造完整文件路径
    pytorch_bin = os.path.join(model_dir, TRANSFORMERS_MODEL_NAME[0])
    adapter = os.path.join(model_dir, TRANSFORMERS_MODEL_NAME[1])
    safetensors = os.path.join(model_dir, "model.safetensors.index.json")

    # 检查文件存在性
    has_pytorch = os.path.isfile(pytorch_bin)
    has_adapter = os.path.isfile(adapter)
    has_safe = os.path.isfile(safetensors)

    # 判断结果
    if has_pytorch and has_safe:
        raise ValueError(
            "The path of resume_from_checkpoint has at least 2 type of checkpint file."
        )
    elif has_pytorch and not peft:
        return pytorch_bin
    elif has_adapter and peft:
        return adapter
    elif has_safe and not peft:
        return safetensors
    else:
        return None


def check_state_dict(
        model: nn.Module,
        old_state_dict: OrderedDict,
        args,
        verbose: bool = False,
        load_list: Optional[List[str]] = None
    ) -> Tuple[Union[OrderedDict, bool]]:

    load_dir = args.resume_from_checkpoint
    new_keys = list(model.state_dict().keys())
    old_keys = list(old_state_dict['model'].keys() \
                    if 'model' in old_state_dict.keys() \
                        else old_state_dict.keys())

    if new_keys == old_keys:
        return old_state_dict, True

    def convert_key(new_keys: str, old_keys: str, state_dict: OrderedDict):
        new_state_dict = {}

        for i in range(len(new_keys)):
            for j in range(len(old_keys)):
                split_mk = new_keys[i].split(".")[1:]
                split_uk = old_keys[j].split(".")
                mk = ".".join(split_mk)
                uk = ".".join(split_uk)
                min_len = min(len(split_mk), len(split_uk))
                if split_mk == split_uk[2:] or split_mk == split_uk[1:]:
                    new_state_dict[new_keys[i]] = state_dict[old_keys[j]]
                    # print(new_keys[i], old_keys[j],
                    #       mpu.get_pipe_parallel_rank())
                    break
                elif mk == uk or new_keys[i] == old_keys[j]:
                    new_state_dict[new_keys[i]] = state_dict[old_keys[j]]
                    # print(new_keys[i], old_keys[j], 
                    #       mpu.get_pipe_parallel_rank())
                    break
                elif min_len <= 3 and same_layer_idx(split_mk, split_uk) \
                    and split_mk[-2:] == split_uk[-2:]:
                    new_state_dict[new_keys[i]] = state_dict[old_keys[j]]
                    # print(new_keys[i], old_keys[j], 
                    #       mpu.get_pipe_parallel_rank())
                    break
                elif min_len > 3 and same_layer_idx(split_mk, split_uk) \
                    and split_mk[-4:] == split_uk[-4:]:
                    new_state_dict[new_keys[i]] = state_dict[old_keys[j]]
                    # print(new_keys[i], old_keys[j], 
                    #       mpu.get_pipe_parallel_rank())
                    break

        return new_state_dict

    if "model" in old_state_dict.keys():
        old_state_dict = old_state_dict['model']

    convert_state_dict = convert_key(new_keys, old_keys, old_state_dict)

    convert_keys = convert_state_dict.keys()

    missing_keys = list(set(new_keys) - set(convert_keys))
    unexcepted_keys = list(set(convert_keys) - set(new_keys))

    strict = True

    if len(missing_keys) != 0:
        # if missing keys has no lora module, do not print it.
        if load_list is not None:
            for n in load_list:
                if n not in ".".join(missing_keys):
                    verbose = False
        if verbose and mpu.get_data_parallel_rank() == 0:
            print(f'Stage {mpu.get_pipe_parallel_rank()}:'
                   'Warning! Some weights of model were not initialized from the model checkpoint.'
                  f'Missing keys: {missing_keys}')
        strict = False
    if len(unexcepted_keys) != 0:
        if verbose and mpu.get_data_parallel_rank() == 0:
            print(f'Stage {mpu.get_pipe_parallel_rank()}:'
                  f'Warning! Some weights of the model checkpoint at {load_dir} were not used.'
                  f'Unexcepted keys: {unexcepted_keys}')
        strict = False

    if 'model' in old_state_dict.keys():
        old_state_dict['model'] = convert_state_dict
        return old_state_dict, strict

    return convert_state_dict, strict


# peft
def load_peft_model_state_dict(
        model: nn.Module,
        args,
        peft_config,
        verbose: bool = False
    ) -> OrderedDict:

    load_dir = args.resume_from_checkpoint
    filename = detect_checkpoint_format(load_dir, peft=True)

    if filename is None:
        # Read the tracker file and set the iteration.
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if not os.path.isfile(tracker_filename):
            if mpu.get_data_parallel_rank() == 0:
                print(f'WARNING ! The path {args.resume_from_checkpoint} '
                      f'has no lora model files {filename}')
                return
        iteration, release = read_metadata(tracker_filename)
        filename, _ = get_checkpoint_name(load_dir, iteration,
                                          args,release=release, peft=True)
    else:
        iteration = 0

    print_rank_0(f' loading peft checkpoint from {filename}')


    peft_state_dict = torch.load(filename, map_location=torch.device("cpu"))
    load_results = set_peft_model_state_dict(model, peft_state_dict, 
                                             args, peft_config, verbose)
    return load_results


def get_peft_model_state_dict(model, peft_config, state_dict=None) -> Dict[str, torch.Tensor]:
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. 
        When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP, 
        the model should be the underlying model/unwrapped model 
        (i.e. model.module). 
        state_dict (`dict`, *optional*, defaults to `None`):
        The state dict of the model. If not provided, the state dict of the 
        model will be used.
    """
    if state_dict is None:
        state_dict = model.state_dict()
    modules_to_save = peft_config.modules_to_save
    if peft_config.peft_type == PeftType.LORA:
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = peft_config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" 
                         in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        if modules_to_save is not None:
            for ms in modules_to_save:
                for k in state_dict:
                    if ms in k:
                        to_return[k] = state_dict[k]

    else:
        raise NotImplementedError

    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, args, peft_config, 
                              verbose=False) -> OrderedDict:
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = peft_config
    state_dict = peft_model_state_dict
    adapter_name = args.adapter_name
    modules_to_save = peft_config.modules_to_save

    if config.peft_type in (PeftType.LORA,):
        peft_model_state_dict = {}
        parameter_prefix = {
            PeftType.LORA: "lora_",
        }[config.peft_type]
        modules_to_save.append(parameter_prefix)
        for k, v in state_dict.items():
            if adapter_name in k:
                adapter_name = ""
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    if adapter_name:
                        k = k.replace(suffix_to_replace, 
                                      f"{adapter_name}.{suffix_to_replace}")
                    else:
                        k = k.replace(suffix_to_replace, f"{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}" if adapter_name else f"{k}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
    else:
        raise NotImplementedError

    peft_model_state_dict, strict = check_state_dict(model, 
                                                     peft_model_state_dict, 
                                                     args, 
                                                     verbose, 
                                                     load_list=modules_to_save)

    load_result = model.load_state_dict(peft_model_state_dict, strict=strict)

    del peft_model_state_dict

    return load_result