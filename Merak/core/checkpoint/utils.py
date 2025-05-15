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

import os
import sys
import enum
import json
from pathlib import Path
from typing import Dict
from safetensors import safe_open
from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Tuple, Callable, Union, List, Dict
from pathlib import Path

from Merak import print_rank_0
from .. import mpu

TRANSFORMERS_MODEL_NAME = ["pytorch_model.bin", "adapter_model.bin"]

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
        else:
            return f"{peft_id}model_optim.pt"
    elif mpu.get_pipe_parallel_world_size() == 1 and mpu.get_model_parallel_world_size() != 1:
        if zero_file:
            return f"zero_dp_rank_{dp_rank}_mp_rank_{mp_rank}_opt_states.pt"
        else:
            return f"mp_rank_{mp_rank}/{peft_id}partial_model_optim.pt"
    else:
        if zero_file:
            return f"zero_dp_rank_{dp_rank}_mp_rank_{mp_rank}_pp_rank_{pp_rank}_opt_states.pt"
        else:
            return f"mp_rank_{mp_rank}_pp_rank_{pp_rank}/{peft_id}partial_model_optim.pt"

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
                print(f'Stage {mpu.get_pipe_parallel_rank()}:'
                   'Warning! Some weights of model were not initialized from the model checkpoint.'
                  f'Missing keys: {missing_keys}')
            if len(unexpected_keys) > 0:
                print(f'Stage {mpu.get_pipe_parallel_rank()}:'
                  f'Warning! Some weights of the model checkpoint at {args.resume_from_checkpoint} were not used.'
                  f'Unexcepted keys: {unexpected_keys}')

    if "model" in old_state_dict:
        old_state_dict['model'] = new_state_dict
    else:
        old_state_dict = new_state_dict
    return old_state_dict, strict

def ensure_directory_exists(path: str) -> None:
    """Ensures the directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path,
                        'latest_checkpointed_iteration.txt')

def get_best_checkpoint_filename(checkpoints_path: str) -> str:
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'best_model_loss.txt')

def unwrap_model(model: Union[nn.Module, List[nn.Module]]) -> List[nn.Module]:
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


def ensure_directory_exists(filename: str):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def same_layer_idx(list1: List[int], list2: List[int]) -> bool:
    digits1 = set([int(x) for x in list1 if x.isdigit()])
    digits2 = set([int(x) for x in list2 if x.isdigit()])
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
                    f"failed to find optimizer param in named params")
            name = param_names[param]
            param_shapes[name] = shape

        param_group_shapes.append(param_shapes)
    # if self.global_rank == 0: print(
    #     f"Total saved {numel} numels in {cnt} params")

    return param_group_shapes


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LOHA = "LOHA"

class ShardedSafetensorLoader:
    def __init__(
            self,
            model,
            model_dir: str = ".",
            index_file: str = "model.safetensors.index.json"
        ):
        self.model = model
        self.model_dir = Path(model_dir)
        self.index = self._load_index(index_file)
        self.cached_files: Dict[str, dict] = {}  # 缓存已加载的分片文件

    def _load_index(self, index_file: str) -> dict:
        """加载分片索引文件"""
        index_path = self.model_dir / index_file
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        with open(index_path, "r") as f:
            return json.load(f)["weight_map"]  # 提取权重映射表

    def _get_tensor(self, layer_name: str) -> torch.Tensor:
        """按层名获取张量"""
        if layer_name not in self.index:
            raise KeyError(f"权重 {layer_name} 不存在于索引文件中")

        shard_file = self.index[layer_name]
        if shard_file not in self.cached_files:
            # 延迟加载分片文件
            shard_path = self.model_dir / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"分片文件 {shard_file} 不存在")

            with safe_open(shard_path, framework="pt") as f:
                self.cached_files[shard_file] = {k: f.get_tensor(k) for k in f.keys()}

        return self.cached_files[shard_file][layer_name]

    def get_state_dict(self):
        weights = {}
        state_dict_keys = list(self.model.state_dict().keys())
        for k in state_dict_keys:
            split_k = k.split('.')
            del split_k[0]
            new_k = ".".join(split_k)
            try:
                weights[k] = self._get_tensor(new_k)
            except KeyError:
                continue
        return weights
