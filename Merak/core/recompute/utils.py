# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Any

def move_to_device(item: Any, device: torch.device):
    """
    Move tensor onto device. Works on individual tensors, and tensors contained/
    nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device

    Returns:
        None
    """
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item, list):
        return [move_to_device(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    else:
        return item