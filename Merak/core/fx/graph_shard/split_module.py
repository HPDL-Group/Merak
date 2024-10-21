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

# Parts of the code here are adapted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/fx/passes/split_module.py

import torch

from collections import ChainMap
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from collections import ChainMap
from typing import Any, Callable, Dict, List, Optional, Tuple

from Merak.merak_args import get_args


@compatibility(is_backward_compatible=True)
class Partition:
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/split_module.py
    """

    def __init__(self, name: str):
        self.name: str = name
        self.node_names: List[str] = []
        self.inputs: Dict[str, None] = {}
        self.outputs: Dict[str, None] = {}
        self.partitions_dependent_on: Dict[str, None] = {}
        self.partition_dependents: Dict[str, None] = {}
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.environment: Dict[torch.fx.node.Node, torch.fx.node.Node] = {}
        self.targets: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"name: {self.name},\n" \
            f" nodes: {self.node_names},\n" \
            f" inputs: {self.inputs},\n" \
            f" outputs: {self.outputs},\n" \
            f" partitions dependent on: {self.partitions_dependent_on},\n" \
            f" partition dependents: {self.partition_dependents}"


# Creates subgraphs out of main graph
@compatibility(is_backward_compatible=True)
def split_module(
        m: GraphModule,
        root_m: torch.nn.Module,
        split_callback: Callable[[torch.fx.node.Node], int],
        merge_nearest=None,
    ) -> Tuple[List[GraphModule], Dict[str, int]]:
    """
    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/fx/passes/split_module.py
    Creates subgraphs out of main graph
    Args:
        m (GraphModule): Graph module to split
        root_m (torch.nn.Module): root nn module. Not currently used. Included
            because the root nn module is usually transformed via
            torch.fx._symbolic_trace.symbolic_trace (see example below)
        split_callback (Callable[[torch.fx.node.Node], int]): Callable function
            that maps a given Node instance to a numeric partition identifier.
            split_module will use this function as the policy for which operations
            appear in which partitions in the output Module.
    Returns:
        GraphModule: the module after split.
    Example:
        This is a sample setup:
            import torch
            from torch.fx.symbolic_trace import symbolic_trace
            from torch.fx.graph_module import GraphModule
            from torch.fx.node import Node
            from colossalai.fx.passes.split_module import split_module
            class MyModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 4))
                    self.linear = torch.nn.Linear(4, 5)
                def forward(self, x, y):
                    z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                    w = self.linear(y).clamp(min=0.0, max=1.0)
                    return z + w
            # symbolically trace model
            my_module = MyModule()
            my_module_traced = symbolic_trace(my_module)
            # random mod partitioning
            partition_counter = 0
            NPARTITIONS = 3
            def mod_partition(node: Node):
                global partition_counter
                partition = partition_counter % NPARTITIONS
                partition_counter = (partition_counter + 1) % NPARTITIONS
                return partition
            # split module in module with submodules
            module_with_submodules = split_module(
                my_module_traced, my_module, mod_partition
            )
        Output looks like this. Original graph is broken into partitions
            > print(module_with_submodules)
            GraphModule(
                (submod_0): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_1): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_2): GraphModule()
            )
            def forward(self, x, y):
                param = self.param
                submod_0 = self.submod_0(x, param, y);  x = param = y = None
                getitem = submod_0[0]
                getitem_1 = submod_0[1];  submod_0 = None
                submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None
                getitem_2 = submod_1[0]
                getitem_3 = submod_1[1];  submod_1 = None
                submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
                return submod_2
        Output of split module is the same as output of input traced module.
        This is an example within a test setting:
            > orig_out = my_module_traced(x, y)
            > submodules_out = module_with_submodules(x, y)
            > self.assertEqual(orig_out, submodules_out)
            True
    """
    partitions: Dict[str, Partition] = {}
    orig_nodes: Dict[str, torch.fx.node.Node] = {}
    func_inputs: Dict[Node, int] = {}
    args = get_args()
    submod_list = []
    min_deps_partitions: Dict[str, Partition] = {}

    def record_cross_partition_use(def_node: torch.fx.node.Node,
                                   use_node: Optional[torch.fx.node.Node], output=False):

        def_partition_name = getattr(def_node, '_fx_partition', None)
        use_partition_name = getattr(use_node, '_fx_partition', None)

        if def_partition_name != use_partition_name:
            if def_partition_name is not None:
                def_partition = partitions[def_partition_name]
                def_partition.outputs.setdefault(def_node.name)
                if use_partition_name is not None:
                    def_partition.partition_dependents.setdefault(use_partition_name)

            if use_partition_name is not None:
                use_partition = partitions[use_partition_name]
                use_partition.inputs.setdefault(def_node.name)
                if def_partition_name is not None:
                    use_partition.partitions_dependent_on.setdefault(def_partition_name)

                    # calculate if cross partition node exist
                    cross_patition_length = int(use_partition_name) - int(def_partition_name)
                    # cross partition node exist
                    if cross_patition_length > 1:
                        prev_partition_idx = int(use_partition_name) - 1
                        prev_partition = partitions[str(prev_partition_idx)]

                        if prev_partition is not None:
                            # set correct cross partition nodes cross partition nodes should be
                            # transmitted from the most beginning partition, which produce these
                            # nodes, to the currenet partition that need these nodes.
                            most_beginning_idx = int(def_partition_name) - 1
                            cur_need_idx = int(use_partition_name)
                            for idx in range(cur_need_idx, most_beginning_idx, -1):
                                if idx > most_beginning_idx + 1:
                                    partitions[str(idx)].inputs.setdefault(def_node.name)
                                if idx < cur_need_idx:
                                    partitions[str(idx)].outputs.setdefault(def_node.name)
        if output:
            # set the correct output for the partition before the last partition,
            # in case this partition holds the node that need to be emitted by the last partition,
            # and the nodes in the last partition do not have any data dependency with this node.
            # So we need to transfer this node to the last partition to make it produce correct
            # output.
            if use_partition_name is not None:
                use_partition = partitions[use_partition_name]
                use_partition.outputs.setdefault(def_node.name)
            # set the correct output for the last partition.
            if def_partition_name is not None:
                def_partition = partitions[def_partition_name]
                def_partition.outputs.setdefault(def_node.name)

    # split nodes into partitions
    for node in m.graph.nodes:
        if args.trace_method == 'dynamo' and not args.use_cpu:
            if 'device' in node.kwargs.keys():
                change_kwargs = node.kwargs.copy()
                change_kwargs['device'] = torch.device('cuda')
                node.kwargs = change_kwargs
        orig_nodes[node.name] = node

        if node.op == 'output':
            # torch.fx.graph.map_arg(node.args[0], lambda n: record_cross_partition_use(n, None))
            torch.fx.graph.map_arg(
                node.args[0],
                lambda n: record_cross_partition_use(n, node.prev, output=True)
            )
            continue

        partition_name = str(split_callback[node.name]) \
            if type(split_callback) == dict else str(split_callback(node))

        if node.op == "placeholder":
            func_inputs[node.name] = int(partition_name)

        # add node to partitions
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.node_names.append(node.name)
        node._fx_partition = partition_name

        torch.fx.graph.map_arg(
            node.args,
            lambda def_node: record_cross_partition_use(def_node, node)
        )
        torch.fx.graph.map_arg(
            node.kwargs,
            lambda def_node: record_cross_partition_use(def_node, node)
        )    # noqa: B950

    # find partitions with no dependencies
    root_partitions: List[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.partitions_dependent_on):
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions: List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].partition_dependents:
            partitions[dependent].partitions_dependent_on.pop(root_partition)
            if not partitions[dependent].partitions_dependent_on:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    if merge_nearest is not None:
        min_deps_partitions = merge_nearest(m, partition_name, partitions)
    else:
        min_deps_partitions = partitions

    # add placeholders to min_deps_partitions
    for partition_name, partition in min_deps_partitions.items():
        # partition = min_deps_partitions[partition_name]
        for input in partition.inputs:
            placeholder = partition.graph.placeholder(input)
            partition.environment[orig_nodes[input]] = placeholder

    # Transform nodes and collect targets for partition's submodule
    for node in m.graph.nodes:
        if hasattr(node, '_fx_partition'):
            partition = min_deps_partitions[node._fx_partition]

            # swap out old graph nodes in kw/args with references to new nodes in this submodule
            environment = partition.environment
            
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(node.kwargs, lambda n: environment[n])

            if node.op not in ['call_module', 'get_attr']:
                target = node.target
            else:
                target_atoms = node.target.split('.')
                target_attr = m
                for atom in target_atoms:
                    if not hasattr(target_attr, atom):
                        raise RuntimeError(f'Operator target {node.target} not found!')
                    target_attr = getattr(target_attr, atom)
                # target = target_atoms[-1]
                target = '.'.join(target_atoms)
                partition.targets[target] = target_attr

            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)
            new_node = partition.graph.create_node(op=node.op,
                                                   target=target,
                                                   args=gathered_args,
                                                   kwargs=gathered_kwargs,
                                                   name=node.name)

            # new_node = partition.graph.node_copy(node, lambda n: environment[n])
            partition.environment[node] = new_node

    all_new_node_map = {}
    for partition_name, partition in min_deps_partitions.items():
        all_new_node_map = dict(ChainMap(partition.environment, all_new_node_map))

    for partition_name, partition in min_deps_partitions.items():
        try:
            output_vals = tuple(
                partition.environment[orig_nodes[name]] for name in partition.outputs
            )
        except:
            # cross partition data denpandency exist
            output_vals = tuple(
                all_new_node_map[orig_nodes[name]] for name in partition.outputs
            )

        output_vals = output_vals[0] \
            if len(output_vals) == 1 else output_vals # type: ignore[assignment]

        partition.graph.output(output_vals)

        submod = torch.fx.graph_module.GraphModule(
            partition.targets if args.trace_method == 'dynamo' else root_m, partition.graph)

        submod_list.append(submod)

    return submod_list, func_inputs
