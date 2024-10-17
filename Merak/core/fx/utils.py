# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
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

__all__ = ['_load_cache', '_save_cache', '_split_attr_values']

import torch
import os
import torch.distributed as dist

from torch.nn import Module
from torch.fx.graph_module import GraphModule
from typing import Dict, List, Set, Any, Optional, Type, Tuple

from Merak.core import mpu
from Merak.merak_args import MerakArguments

from ..tensor_parallel.mp_attrs import (
    DEFAULT_MP_ATTR,
    set_tp_layer_lists,
    get_tp_attr_list
)
from ..tensor_parallel.mp_mapping import get_mp_layer_lists


def _load_cache(args: MerakArguments) -> Tuple[List[GraphModule], Dict[str, int]]:
    assert args.cache_name is not None
    if os.path.isfile(f'{args.cache_name}_graph0_cache.pt'):
        cache_dir = os.path.split(f'{args.cache_name}_graph0_cache.pt')[0]
        result_len = len([n for n in os.listdir(cache_dir) if 'graph' in n])
        result = []
        for i in range(result_len):
            graph_slice = torch.load(f'{args.cache_name}_graph{i}_cache.pt')
            result.append(graph_slice)
            del graph_slice
        input_to_shard = torch.load(f'{args.cache_name}_input_cache.pt')

        return result, input_to_shard

def _save_cache(result: List[GraphModule], input_to_shard: Dict[str, int], args: MerakArguments):
    if dist.get_rank() == 0:
        file_path = os.path.abspath(os.path.dirname(args.cache_name))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for idx, graph in enumerate(result):
            # graph.to_folder(f'{args.cache_name}', f'module_{idx}')
            torch.save(graph, f'{args.cache_name}_graph{idx}_cache.pt')
        torch.save(input_to_shard, f'{args.cache_name}_input_cache.pt')
        dist.barrier()
    else:
        dist.barrier()

def _split_attr_values(model: Module, gm: GraphModule, merak_args: MerakArguments) -> GraphModule:
    '''
    Replace constant node to real value.
    '''

    new_graph = torch.fx.Graph()
    new_graph2 = torch.fx.Graph()
    value_remap = {}

    mp_layer_lists = get_mp_layer_lists(model.__class__)
    if mp_layer_lists is not None:
        set_tp_layer_lists(**mp_layer_lists)
    mp_attr_list = get_tp_attr_list()

    if not mp_attr_list:
        mp_attr_list = DEFAULT_MP_ATTR

    split_attr = {'gpt': 3*merak_args.hidden_size}
    def _set_model_mp_attr(model):
        for n, module in model.named_children():
            for attr in mp_attr_list:
                if hasattr(module, attr):
                    old_attr = getattr(module, attr)
                    split_attr[attr] = old_attr
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                _set_model_mp_attr(module)
    _set_model_mp_attr(model)

    for node in gm.graph.nodes:
        new_args = []

        for arg in node.args:
            if isinstance(arg, int):
                if arg == merak_args.hidden_size or \
                   arg in merak_args.num_heads or \
                   arg in split_attr.values():
                    arg = int(arg / mpu.get_model_parallel_world_size())
            if isinstance(arg, (tuple, list)):
                value_list = []
                for value in arg:
                    if isinstance(value, int):
                        if value == merak_args.hidden_size or \
                           value in merak_args.num_heads or \
                           value in split_attr.values():
                            value = int(value / mpu.get_model_parallel_world_size())
                    value_list.append(value)
                arg = tuple(value_list)
            new_args.append(arg)

        with new_graph.inserting_after():
            # print(node.target, type(node.target))
            new_node = new_graph.create_node(op=node.op, args=tuple(new_args),
                                             kwargs=node.kwargs, name=node.name,
                                             target=node.target)
            value_remap[node.name] = new_node

    for node in new_graph.nodes:
        try:
            if node.name != "output":
                new_node = new_graph2.node_copy(node, lambda x: value_remap[x.name])
                value_remap[node.name] = new_node
            else:
                new_node = new_graph2.node_copy(node, lambda x: value_remap[x.name])
                new_graph2.lint()
                break
        except KeyError as e:
            print(f"The Node {node.name} missing {e}, "
                  f"the node's args: {node.args}, kwargs: {node.kwargs}")
            exit()

    gm.graph = new_graph2
    gm.recompile()
    return gm