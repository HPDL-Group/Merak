# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Yck (eyichenke@gmail.com)
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

from torch.fx.graph_module import GraphModule
from typing import Dict, List, Set, Any, Optional, Tuple

from Merak.merak_args import get_args

from .farthest_deps import farthest_deps_split
from .layer_config import layer_config_split
from .nearest_deps import nearest_deps_split


def _shard_model_transformers(
        traced_graph_module: GraphModule,
        model: torch.nn.Module,
        shard_count=3,
    ) -> Tuple[List[GraphModule], Dict[str, int]]:
    """Utility used to shard a model using torch.fx.

    This function traces the model twice in an attempt to identify the
    right cutpoints and then shard the model. In the first pass we calculate
    the number of parameters as we are tracing the graph and mark nodes at
    which we might want to create a new module. In the second pass we
    modify the graph by inserting placeholders and output nodes to 
    essentially shard the graph.

    We don't support skip connections between shards. This means that all
    input and output is self contained within a given shard. A node from
    shard 1 cannot be an input to a node from shard 3. We expect all inputs
    to a given shard to be coming from the last node in the previous shard.
    This means that we may not be able to shard models by the specified
    `shard_count` mentioned by the user.

    Args:
        model (nn.Module): Model to be sharded as specified by the device 
        count.

        shard_count (int): Number of shards that we want to split the model 
        into.

    """
    args = get_args()

    # a experience users number threshold, a node has more user than this
    # threshold indicate the node is needed in multiple stages and
    # could be transmitted between stages
    output_node_threshold = 5
    output_nodes_count = {}
    for node in traced_graph_module.graph.nodes:
        if len(list(node.users)) > output_node_threshold:
            output_nodes_count[node.name] = len(list(node.users))

    if args.split_method == 'farthest_min_deps':
        module_list, func_inputs = farthest_deps_split(traced_graph_module, 
                                                       model, shard_count, 
                                                       output_nodes_count)
    elif args.split_method == 'layer_split':
        module_list, func_inputs = layer_config_split(traced_graph_module, model)
    elif args.split_method == 'nearest_min_deps':
        module_list, func_inputs = nearest_deps_split(traced_graph_module, model)
    else:
        assert args.split_method in ['farthest_min_deps', 'layer_split', 'nearest_min_deps']
    return module_list, func_inputs