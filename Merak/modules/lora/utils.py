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
# The code here are adapted from https://github.com/huggingface/peft/blob/v0.6.0/src/peft/tuners/lora.py

import warnings

import torch
import torch.nn as nn
from .layer import Linear, LoraLayer, Embedding, Conv2d
from transformers.modeling_utils import Conv1D


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer
        elif hasattr(child, "quant_linear_module"):
            child = child.quant_linear_module

        # TODO: layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

def _find_and_replace(model, adapter_name, config):

    kwargs = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "fan_in_fan_out": config.fan_in_fan_out,
    }
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if any(key.endswith(target_key) for target_key in config.target_modules):
            parent, target, target_name = _get_submodules(model, key)
            bias = target.bias is not None
            if isinstance(target, torch.nn.Embedding):
                embedding_kwargs = kwargs.copy()
                embedding_kwargs.pop("fan_in_fan_out", None)
                in_features, out_features = target.num_embeddings, target.embedding_dim
                new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
            elif isinstance(target, torch.nn.Conv2d):
                out_channels, in_channels = target.weight.size()[:2]
                kernel_size = target.weight.size()[2:]
                stride = target.stride
                padding = target.padding
                new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
            else:
                if isinstance(target, torch.nn.Linear):
                    in_features, out_features = target.in_features, target.out_features
                    if kwargs["fan_in_fan_out"]:
                        warnings.warn(
                            "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                            "Setting fan_in_fan_out to False."
                        )
                        kwargs["fan_in_fan_out"] = config.fan_in_fan_out = False
                elif isinstance(target, Conv1D):
                    in_features, out_features = (
                        target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                    )
                    kwargs["is_target_conv_1d_layer"] = True
                    if not kwargs["fan_in_fan_out"]:
                        warnings.warn(
                            "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                            "Setting fan_in_fan_out to True."
                        )
                        kwargs["fan_in_fan_out"] = config.fan_in_fan_out = True
                else:
                    raise ValueError(
                        f"Target module {target} is not supported. Currently, only the following modules are supported: "
                        "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
                    )
                new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

            _replace_module(parent, target_name, new_module, target)
        

def mark_only_lora_as_trainable(model: nn.Module, config) -> None:
    bias = config.bias
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        if config.modules_to_save is not None:
            for module_to_save in config.modules_to_save:
                if module_to_save in n:
                    p.requires_grad = True
        # print(n, p.requires_grad)
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