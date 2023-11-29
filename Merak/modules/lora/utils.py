# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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

# The code here are adapted from https://github.com/huggingface/peft/blob/v0.0.2/src/peft/tuners/lora.py

import warnings

import torch
import torch.nn as nn
from .layer import Linear, LoraLayer, MergedLinear


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if old_module.bias is not None:
        new_module.bias = old_module.bias

def _find_and_replace(model, config):
    kwargs = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "fan_in_fan_out": config.fan_in_fan_out,
        "merge_weights": config.merge_weights,
    }
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if any(key.endswith(target_key) for target_key in config.target_modules):
            parent, target, target_name = _get_submodules(model, key)
            bias = target.bias is not None
            if isinstance(target, torch.nn.Linear) and config.enable_lora is None:
                new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
            elif config.enable_lora is not None:
                kwargs.update({"enable_lora": config.enable_lora})
                # if isinstance(target, Conv1D):
                if "Conv1D" in str(type(target)):
                    in_features, out_features = target.weight.shape
                else:
                    in_features, out_features = target.in_features, target.out_features
                    if kwargs["fan_in_fan_out"]:
                        warnings.warn(
                            "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                            "Setting fan_in_fan_out to False."
                        )
                        kwargs["fan_in_fan_out"] = False
                new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
            _replace_module(parent, target_name, new_module, target)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoraLayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError