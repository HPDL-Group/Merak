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


