# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/model/utils.py
# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/utils.py

import math
import torch
from typing import Callable, Union

from torch.nn import LayerNorm 
# from transformers.utils.fx import _generate_supported_model_classes
from transformers.utils.fx import _generate_supported_model_class_names


def init_method_normal(sigma: float) -> Callable:
    """Init method based on N(0, sigma)."""
    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma: float, num_layers: Union[int, dict]):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    if isinstance(num_layers, dict):
        num_layers = sum(list(num_layers.values()))
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def tp_overlapping_available(model_class: Callable) -> bool:
    SUPPORTED_MODEL_NAMES = ['gpt2']
    for model_name in SUPPORTED_MODEL_NAMES:
        if model_class.__name__ in \
            _generate_supported_model_class_names(model_name):
            return True
    return False

def get_params_for_weight_decay_optimization(module: torch.nn.Module):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and 'bias' not in n ])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and 'bias' in n ])

    return weight_decay_params, no_weight_decay_params

def reset_module_tensor(
        module: torch.nn.Module,
        tensor_name: str,
        device: torch.device,
        value: torch.Tensor
    ):
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)
    param_cls = type(module._parameters[tensor_name])
    with torch.no_grad():
        new_value = value.to(device)
        if is_buffer:
            module._buffers[tensor_name] = new_value
        else:
            new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
            module._parameters[tensor_name] = new_value

