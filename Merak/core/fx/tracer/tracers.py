# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
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

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/utils/fx.py

import torch


class LayerProxyTracer(torch.fx.Tracer):
    """Tracer with an extended set of leaf nn.Modules."""

    def __init__(self, leaf_modules):
        super().__init__()
        self.leaf_modules = leaf_modules

    def is_manual_leaf_module(self, m):
        for i in self.leaf_modules:
            if isinstance(m, i):
                return True
        return False

    def is_leaf_module(self, m: torch.nn.Module, model_qualified_name: str) -> bool:
        return super().is_leaf_module(
            m, model_qualified_name
        ) or self.is_manual_leaf_module(m)
