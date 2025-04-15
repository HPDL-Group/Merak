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
        inputs = tuple(dummy_inputs.values())
    elif isinstance(dummy_inputs, (tuple, list)):
        inputs = tuple(dummy_inputs)
    else:
        raise TypeError("Type of dummy inputs must be list, tuple or dict")

    try:
        ep = torch.export.export_for_training(
            module,
            inputs,
        )
    except Exception as e:
        raise RuntimeError(
            "It seems that we cannot capture your model as a full graph. "
            "Typical reasons include graph breaks, data/shape-dependent "
            "control flow, or missing meta kernels for custom operators. "
            "You can use our manual pipeline interfaces, or try to fix the "
            "graph breaks, see https://pytorch.org/docs/stable/export.html"
        ) from e

    traced = ep.module()

    # Deduplicate `get_attr` nodes that refer to the same parameter . Downstream code for moving
    # parameters relies on the invariant that parameter accesses happen once. This is not necessarily
    # the case (especially with custom tracers), so fix that up here.
    get_attr_nodes: Dict[str, torch.fx.Node] = {}
    for node in traced.graph.nodes:  # type: ignore[union-attr]
        if node.op == "get_attr":
            get_attr_nodes.setdefault(node.target, node)

            if get_attr_nodes[node.target] != node:
                node.replace_all_uses_with(get_attr_nodes[node.target])
                traced.graph.erase_node(node)  # type: ignore[operator, union-attr]

    return traced