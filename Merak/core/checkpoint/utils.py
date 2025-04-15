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
import enum
import json
from pathlib import Path
from typing import Dict
from safetensors import safe_open
from collections import OrderedDict
from typing import List, Union

import torch
import torch.nn as nn

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
