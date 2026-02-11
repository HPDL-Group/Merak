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

import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open

from Merak import get_logger

from .. import mpu

TRANSFORMERS_MODEL_NAME = ["pytorch_model.bin", "adapter_model.bin"]
logger = get_logger("simple")

def get_checkpoint_name(
    checkpoints_path: str,
    iteration: int,
    args,
    release: bool = False,
    best: Optional[bool] = None,
    peft: bool = False
) -> Tuple[str, Optional[str]]:
    """
    Generates the checkpoint file names based on the training parameters.

    Args:
        checkpoints_path (str): Path to the checkpoint directory.
        iteration (int): Training iteration number.
        args: Training arguments.
        release (bool, optional): Whether this is a release checkpoint. Defaults to False.
        complete (bool, optional): Whether this is a complete checkpoint. Defaults to False.
        best (Optional[bool], optional): Whether this is the best checkpoint. Defaults to None.
        peft (bool, optional): Whether this is a Peft checkpoint. Defaults to False.

    Returns:
        Tuple[str, Optional[str]]: The model and optimizer checkpoint file names.
    """
    if release:
        directory = 'release'
    elif best is not None:
        directory = 'best_model'
    else:
        directory = f'iter_{iteration:07d}'

    peft_id = 'lora_' if peft else ''

    model_ckpt_name = os.path.join(
        checkpoints_path,
        directory,
        get_rank_ckpt_name(peft_id)
    )

    if args.zero_stage == 1:
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            directory,
            get_rank_ckpt_name(peft_id, zero_file=True)
        )
    else:
        zero_ckpt_name = None

    return model_ckpt_name, zero_ckpt_name

def get_rank_ckpt_name(peft_id: str, zero_file: bool = False) -> str:
    """
    Generates the checkpoint file name based on parallel ranks.

    Args:
        peft_id (str): Peft model identifier.
        zero_file (bool, optional): Whether it's a zero optimizer state file. Defaults to False.

    Returns:
        str: The checkpoint file name.
    """
    dp_rank = f"{mpu.get_data_parallel_rank():05d}"
    pp_rank = f"{mpu.get_pipe_parallel_rank():03d}"
    mp_rank = f"{mpu.get_model_parallel_rank():02d}"

    if mpu.get_model_parallel_world_size() == 1 and mpu.get_pipe_parallel_world_size() == 1:
        if zero_file:
            return f"zero_dp_rank_{dp_rank}_opt_states.pt"
        return f"{peft_id}model_optim.pt"
    if mpu.get_pipe_parallel_world_size() == 1 and mpu.get_model_parallel_world_size() != 1:
        if zero_file:
            return f"zero_dp_rank_{dp_rank}_mp_rank_{mp_rank}_opt_states.pt"
        return f"mp_rank_{mp_rank}/{peft_id}partial_model_optim.pt"

    if zero_file:
        return f"zero_dp_rank_{dp_rank}_mp_rank_{mp_rank}_pp_rank_{pp_rank}_opt_states.pt"
    return f"mp_rank_{mp_rank}_pp_rank_{pp_rank}/{peft_id}partial_model_optim.pt"

def read_metadata(tracker_filename: str) -> Tuple[int, bool]:
    '''
    Get iterations from metadata file.

    Args:
        tracker_filename (str): directory of metadata file.

    Returns:
         Tuple[int, bool]: iterations and release
    '''
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
                logger.error(
                    f'ERROR: Invalid metadata file {tracker_filename}. Exiting',
                    ranks=[0]
                )
                sys.exit()
    assert iteration > 0 or release, f'error parsing metadata file {tracker_filename}'

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
        logger.warning(
            f'WARNING: on rank {mpu.get_pipe_parallel_rank()} found '
            f'iteration {iteration} in the metadata while max iteration '
            f'across the ranks is {max_iter}, replacing it with max iteration.',
            ranks=[0]
        )
    return max_iter, release

def detect_checkpoint_format(model_dir=".", peft=False):
    """
    检测模型目录的检查点格式类型
    :param model_dir: 模型文件夹路径，默认为当前目录
    :return: "pytorch_bin" | "safetensors" | "both" | "none"
    """
    # 构造完整文件路径
    pytorch_bin = os.path.join(model_dir, TRANSFORMERS_MODEL_NAME[0])
    adapter = os.path.join(model_dir, TRANSFORMERS_MODEL_NAME[1])
    safe_index = os.path.join(model_dir, "model.safetensors.index.json")
    safe_weight = os.path.join(model_dir, "model.safetensors")

    # 检查文件存在性
    has_pytorch = os.path.isfile(pytorch_bin)
    has_adapter = os.path.isfile(adapter)
    has_safe_index = os.path.isfile(safe_index)
    has_safe_weight = os.path.isfile(safe_weight)

    # 判断结果
    if has_pytorch and not peft:
        return pytorch_bin
    if has_adapter and peft:
        return adapter
    if has_safe_index and not peft:
        return safe_index
    if has_safe_weight and not peft:
        return safe_weight
    if has_pytorch and has_safe_weight:
        raise ValueError(
            "The path of resume_from_checkpoint has at least 2 type of checkpint file."
        )
    return None

def check_state_dict(
    model: nn.Module,
    old_state_dict: OrderedDict,
    args,
    verbose: bool = False,
    load_list: Optional[List[str]] = None
) -> Tuple[Union[OrderedDict, bool]]:
    """
    Address mismatches between old checkpoints and current model states,
    and return the converted state dictionary along with a strict matching flag.
    """

    def _get_model_keys(model: nn.Module) -> List[str]:
        """Get the list of model's state dictionary keys."""
        return list(model.state_dict().keys())

    def _get_old_keys(old_state_dict: OrderedDict) -> List[str]:
        """Get the list of keys from the old state dictionary."""
        if 'model' in old_state_dict:
            return list(old_state_dict['model'].keys())
        return list(old_state_dict.keys())

    def _convert_keys(new_keys: str, old_keys: str, state_dict: OrderedDict):
        new_state_dict = {}

        for _, nk in enumerate(new_keys):
            for _, ok in enumerate(old_keys):
                split_mk = nk.split(".")[1:]
                split_uk = ok.split(".")
                mk = ".".join(split_mk)
                uk = ".".join(split_uk)
                min_len = min(len(split_mk), len(split_uk))
                if split_mk in [split_uk[2:], split_uk[1:]]:
                    new_state_dict[nk] = state_dict[ok]
                    # print(nk, ok,
                    #       mpu.get_pipe_parallel_rank())
                    break
                if mk == uk or nk == ok:
                    new_state_dict[nk] = state_dict[ok]
                    # print(nk, ok,
                    #       mpu.get_pipe_parallel_rank())
                    break
                if min_len <= 3 and same_layer_idx(split_mk, split_uk) \
                    and split_mk[-2:] == split_uk[-2:]:
                    new_state_dict[nk] = state_dict[ok]
                    # print(nk, ok,
                    #       mpu.get_pipe_parallel_rank())
                    break
                if min_len > 3 and same_layer_idx(split_mk, split_uk) \
                    and split_mk[-4:] == split_uk[-4:]:
                    new_state_dict[nk] = state_dict[ok]
                    # print(nk, ok,
                    #       mpu.get_pipe_parallel_rank())
                    break

        return new_state_dict

    # Main logic section
    new_keys = _get_model_keys(model)
    old_keys = _get_old_keys(old_state_dict)

    if new_keys == old_keys:
        return old_state_dict, True

    # Prepare the old state dictionary
    if "model" in old_state_dict:
        old_model_state = old_state_dict['model']
    else:
        old_model_state = old_state_dict

    # Perform key conversion
    new_state_dict = _convert_keys(new_keys, old_keys, old_model_state)

    # Check for missing and unexpected keys
    convert_keys = list(new_state_dict.keys())
    missing_keys = list(set(new_keys) - set(convert_keys))
    unexpected_keys = list(set(convert_keys) - set(new_keys))

    strict = True
    if load_list is not None:
        for n in load_list:
            if n not in ".".join(missing_keys):
                verbose = False
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        strict = False
        if verbose and mpu.get_data_parallel_rank() == 0:
            if len(missing_keys) > 0:
                # Filter out keys not in the load list
                logger.info(f'Stage {mpu.get_pipe_parallel_rank()}:'
                   'Warning! Some weights of model were not initialized from the model checkpoint.'
                  f'Missing keys: {missing_keys}')
            if len(unexpected_keys) > 0:
                logger.info(f'Stage {mpu.get_pipe_parallel_rank()}:'
                  f'Warning! Some weights of the model checkpoint at {args.resume_from_checkpoint} '
                  f'were not used. Unexcepted keys: {unexpected_keys}')

    if "model" in old_state_dict:
        old_state_dict['model'] = new_state_dict
    else:
        old_state_dict = new_state_dict
    return old_state_dict, strict

def ensure_directory_exists(path: str) -> None:
    """Ensures the directory exists, creating it if necessary."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path,
                        'latest_checkpointed_iteration.txt')

def get_best_checkpoint_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'best_model_loss.txt')

def unwrap_model(model: Union[nn.Module, List[nn.Module]]) -> Union[nn.Module, List[nn.Module]]:
    '''Normalize model input to either single module or list of modules.'''
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model

def same_layer_idx(list1: List[str], list2: List[str]) -> bool:
    ''''Check if two str lists contain same digits'''
    digits1 = [int(x) for x in list1 if x.isdigit()]
    digits2 = [int(x) for x in list2 if x.isdigit()]
    return digits1 == digits2


def get_zero_param_shapes(optimizer: torch.optim.Optimizer, model: nn.Module) -> List[torch.Tensor]:
    """Returns a dict of name to shape mapping, only for the flattened
    fp32 weights saved by the optimizer. the names are exactly as in 
    state_dict. The order is absolutely important, since the saved data 
    is just flattened data with no identifiers and requires reconstruction 
    in the same order it was saved.

    We can't rely on self.module.named_parameters() to get the saved 
    tensors, as some params will be missing and others unsaved and then 
    it'd be impossible to reconstruct state_dict from the flattened weights.
    """
    param_names = {param: name for name, param in model.named_parameters()}
    param_group_shapes = []
    cnt = 0
    numel = 0

    fp16_groups =  optimizer.fp16_groups

    for fp16_group in fp16_groups:
        param_shapes = OrderedDict()
        for param in fp16_group:
            cnt += 1
            numel += param.numel()
            shape = param.shape
            if param not in param_names:
                raise ValueError(
                    "failed to find optimizer param in named params"
                )
            name = param_names[param]
            param_shapes[name] = shape

        param_group_shapes.append(param_shapes)
    # if self.global_rank == 0: print(
    #     f"Total saved {numel} numels in {cnt} params")

    return param_group_shapes


def find_latest_ckpt_folder(directory):
    """Find the latest dated _ckpt folder in the directory"""
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})_ckpt$')
    latest_date = None
    latest_folder = None

    for folder in os.listdir(directory):
        match = pattern.match(folder)
        if match:
            try:
                current_date = datetime.strptime(match.group(1), '%Y-%m-%d').date()
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date
                    latest_folder = folder
            except ValueError:
                continue

    return latest_folder

def load_sharded_safetensors(
        model,
        safetensor_file: str = ".",
    ):
    '''Loader for sharded safetensor model weights with lazy loading capability.'''
    assert os.path.isfile(safetensor_file)
    safetensor_path = os.path.dirname(safetensor_file)
    safetensor_filename = os.path.basename(safetensor_file)
    cached_files: Dict[str, dict] = {}  # 缓存已加载的分片文件

    if safetensor_filename == "model.safetensors":
        # whole load
        with safe_open(safetensor_file, framework="pt") as f:
            weights = {k: f.get_tensor(k) for k in f.keys()}
    else:
        weights = {}
        state_dict_keys = list(model.state_dict().keys())

        # sharded load
        with open(safetensor_file, "r") as f:
            index = json.load(f)["weight_map"]
        is_start_model = list(index.keys())[0].split('.')[0] == 'model'
        for k in state_dict_keys:
            split_k = k.split('.')
            if is_start_model:
                split_k[0] = "model"
            else:
                del split_k[0]
            new_k = ".".join(split_k)

            if new_k not in index:
                continue
                # raise KeyError(f"权重 {new_k} 不存在于索引文件中")

            shard_file = index[new_k]
            if shard_file not in cached_files:
                # 延迟加载分片文件
                shard_path = os.path.join(safetensor_path, shard_file)
                if not os.path.isfile(shard_path):
                    raise FileNotFoundError(f"分片文件 {shard_file} 不存在")

                with safe_open(shard_path, framework="pt") as f:
                    weights[k] = f.get_tensor(new_k)

    return weights
