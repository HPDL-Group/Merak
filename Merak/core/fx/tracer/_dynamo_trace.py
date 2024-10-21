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

import torch

from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from typing import Dict, List, Union, Tuple

def dynamo_trace(
        module: torch.nn.Module,
        dummy_inputs: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor], List[torch.Tensor]]
    ) -> List[GraphModule]:
    if isinstance(dummy_inputs, dict):
        inputs = dummy_inputs.values()
    elif isinstance(dummy_inputs, (tuple, list)):
        inputs = tuple(dummy_inputs)
    else:
        raise TypeError("Type of dummy inputs must be list, tuple or dict")
    layers = []

    @compatibility(is_backward_compatible=True)
    def custom_backend(gm, ex):
        layers.append(gm)
        return gm.forward

    mod = torch.compile(module, backend=custom_backend)
    outputs = mod(*inputs)

    assert len(layers) == 1, (
        'torch dynamo produces multiple subgraphs due to guard fail, '
        'current split method do not support multi inputs'
    )

    return layers[0]