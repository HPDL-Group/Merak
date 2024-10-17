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

from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
from typing import Dict, List, Set, Any, Optional, Type, Tuple

from .split_module import split_module, Partition

def nearest_dependency_mapping(
        m: GraphModule,
        partition_name: str,
        partitions: Partition
    ) -> Dict[str, Partition]:
    min_deps_partitions: Dict[str, Partition] = {}
    new_partition_name = 0
    cur_partition_name = 0
    new_cost = 0
    cur_cost = 0

    # recreate partitions by min nearnest data dependancy cost
    while cur_partition_name <= int(partition_name):
        if new_cost > cur_cost:
            new_partition_name += 1
        cur_partition = partitions[str(cur_partition_name)]
        new_partition = min_deps_partitions.get(str(new_partition_name))
        if new_partition is None:
            min_deps_partitions[
                str(new_partition_name)
            ] = new_partition = Partition(str(new_partition_name))
        if len(new_partition.inputs) == 0 and new_partition_name != 0:
            new_partition.inputs = cur_partition.inputs

        next_partition_name = cur_partition_name + 1
        next_partition = partitions[
            str(next_partition_name)
        ] if next_partition_name <= int(partition_name) else None

        if next_partition is not None:
            node_names = cur_partition.node_names + next_partition.node_names
            new_partition.node_names += node_names
            new_partition.outputs = next_partition.outputs
            cur_cost = len(new_partition.inputs) + len(cur_partition.outputs)
            new_cost = len(new_partition.inputs) + len(next_partition.outputs)
        else:
            new_partition.node_names += cur_partition.node_names
            new_partition.outputs = cur_partition.outputs
            break
        cur_partition_name += 2

    # re-label partition index per node after recreate min dependancy partitions
    partition_count = 0
    for node in m.graph.nodes:
        if partition_count <= new_partition_name:
            cur_partition = min_deps_partitions[str(partition_count)]
        else:
            break
        if node.name not in cur_partition.node_names:
            partition_count += 1
            if partition_count > new_partition_name:
                break
            cur_partition = min_deps_partitions[str(partition_count)]
        if hasattr(node, '_fx_partition'):
            node._fx_partition = cur_partition.name
    return min_deps_partitions


def inplace_patch(
        gm: GraphModule,
        node: Node,
        nodes_so_far: List[str],
        node_name_to_shard_id: Dict[str, int],
        shard_id: int
    ):
    # inplace operation process should hold in same shard, 
    # since Output 1 of CheckpointFunctionBackward is a view 
    # and its base or another view of its base has been modified inplace. 
    # This view is the output of a function that returns multiple views. 
    # Such functions do not allow the output views to be modified inplace. 
    for arg in node.args:
        if hasattr(gm, node.target):
            mod = getattr(gm, node.target)
        else:
            submod = gm
            prefix = node.target.split('.')
            for item in prefix:
                mod = getattr(submod, item, None)
                submod = mod
        if hasattr(mod, 'inplace'):
            inplace_node_name = arg.name
            inplace_shard_id = node_name_to_shard_id[inplace_node_name]
            if shard_id > inplace_shard_id:
                for node_name in reversed(nodes_so_far):
                    pre_node_shard_id = node_name_to_shard_id[node_name]
                    if pre_node_shard_id > inplace_shard_id:
                        node_name_to_shard_id[node_name] = inplace_shard_id
                    if node_name == inplace_node_name:
                        break
                shard_id = inplace_shard_id


def avgnode_split_pass(gm: torch.fx.GraphModule, shard_nums: int) -> Dict[str, int]:
    """
    In avgnode_split_pass, simply split graph by node number.
    """
    node_name_to_shard_id: Dict[str, int] = {}
    nodes_so_far: List[str] = []
    mod_graph = gm.graph
    avg_num_node = len(mod_graph.nodes) // shard_nums
    accumulate_num_node = 0
    shard_id = 0

    for node in mod_graph.nodes:
        if node.op == 'call_module':
            inplace_patch(gm, node, nodes_so_far, node_name_to_shard_id, shard_id)
        if node.op == 'placeholder':
            node_name_to_shard_id[node.name] = 0
            nodes_so_far.append(node.name)
        if node.op in ['get_attr', 'call_function', 'call_method', 'call_module']:
            node_name_to_shard_id[node.name] = shard_id
            accumulate_num_node += 1
            nodes_so_far.append(node.name)
        elif node.op == 'output':
            nodes_so_far.append(node.name)
            break
        if accumulate_num_node >= avg_num_node:
            accumulate_num_node = 0
            shard_id += 1
    return node_name_to_shard_id

def nearest_deps_split(
        traced_graph_module: GraphModule,
        model: torch.nn.Module
    ) -> Tuple[List[GraphModule], Dict[str, int]]:
    # mapping node name to shard id
    node_name_to_shard_id = avgnode_split_pass(
        traced_graph_module, len(traced_graph_module.graph.nodes)
    )
    
    # split graph
    module_list, func_inputs = split_module(
        traced_graph_module,
        model,
        node_name_to_shard_id,
        nearest_dependency_mapping
    )
    return module_list, func_inputs