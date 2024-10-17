# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), Yck (eyichenke@gmail.com)
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

# Parts of the code here are adapted from https://github.com/facebookresearch/fairscale/blob/86c62cc9c989178e629823c2fd4a3cc11723e92b/fairscale/experimental/nn/auto_shard.py

import torch

from copy import deepcopy
from torch.fx.graph_module import GraphModule
from typing import Dict, List, Set, Any, Optional, Type, Tuple

from Merak import print_rank_0

from .split_module import split_module
from .utils import _snake_case, _create_shard_to_param_count


SHARD_METHOD = 'exclude_emb'
# SHARD_METHOD = 'param_uniform'

def farthest_deps_mapping(
        traced_graph_module: GraphModule,
        shard_count: int,
        node_user_count=None
    ) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """Utility used to trace a graph and identify shard cutpoints."""

    node_name_to_shard_id: Dict[str, int] = {}
    shard_id = 0
    nodes_so_far = []
    param_count: Dict[str, int] = {}
    shard_to_param_count = {}
    excluded_count = 0
    init_node_user_count = deepcopy(node_user_count)
    
    # Find the total number of params in the model and
    # the number of params per shard we are aiming for.
    for name, module in traced_graph_module.named_modules():
        name = _snake_case(name).replace(".", "_")
        if SHARD_METHOD == 'exclude_emb':
            if isinstance(module, torch.nn.Embedding):
                param_count[name] = 0
                excluded_count += sum([x.numel() for x in module.parameters()])
                continue
            ## test only
            # if isinstance(module, torch.nn.Embedding):
            #     param_count[name] = 1024*1024
            #     continue
            # if isinstance(module, torch.nn.Linear):
            #     param_count[name] = 1024*1024
            #     continue
        param_count[name] = sum([x.numel() for x in module.parameters()])

    for name, p in traced_graph_module.named_parameters():
        name = _snake_case(name).replace(".", "_")
        param_count[name] = p.numel()

    # print_rank_0(param_count)
    print_rank_0(f"Total number of params are {param_count['']}")
    per_shard_param = (param_count[""]-excluded_count) // shard_count
    print_rank_0(f"Per shard param count {per_shard_param}")
    print_rank_0(f"Node count {len(traced_graph_module.graph.nodes)}")


    if SHARD_METHOD == 'exclude_emb':
        for name, module in traced_graph_module.named_modules():
            name = _snake_case(name).replace(".", "_")
            if isinstance(module, torch.nn.Embedding):
                param_count[name] = per_shard_param + 1


    func_inputs = {}
    shard_output = {-1: None, 0: None}
    extra_output = {}

    for node in traced_graph_module.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            func_inputs[node.name] = 0
        elif node.op in ["get_attr", "call_function", "call_method", "call_module"]:

            min_shard_id = shard_id
            min_node_name = ""
            # For each of the args of a given node, find the arg that is not the
            # last node we traversed. This is to help us find skip connections
            # across shards.
            # 如果args里面有非上一个node输出的输入, 找到args里面shard_id最小的, 
            # 记录shard_id和名字
            # print(node.name, node.op, node.args)
            for arg in node.args:
                # If the node has args that are inputs to the forward function, 
                # they may not have explicit names.

                if not hasattr(arg, "name"): 
                    continue

                # 如果args有需记录user的node, 将user数减一
                if arg.name in node_user_count:
                    node_user_count[arg.name] -= 1

                # if a node is inplace OP, it should stay with its input
                if node.op == 'call_module':
                    try:
                        if hasattr(traced_graph_module, node.target):
                            mod = getattr(traced_graph_module, node.target)
                        else:
                            submod = traced_graph_module
                            prefix = node.target.split('.')
                            for item in prefix:
                                mod = getattr(submod, item, None)
                                submod = mod
                        if hasattr(mod, 'inplace'):
                            min_node_name = arg.name
                            min_shard_id = node_name_to_shard_id[min_node_name]
                            continue
                    except:
                        pass

                
                # 如果args里面有某一个shrad的输出
                if arg.name in shard_output.values():
                    # 如果args里面有上一个shrad输出, 跳过
                    if arg.name == shard_output[shard_id-1]:
                        continue
                    # 不是上一个shard的输出改node最小shard_id为此shard+1
                    # print_rank_0([node.name, 'has ', arg.name, 
                    # node_name_to_shard_id[arg.name]])
                    if node_name_to_shard_id[arg.name] + 1 < min_shard_id:
                        min_shard_id = node_name_to_shard_id[arg.name] + 1
                        min_node_name = arg.name
                    continue

                # 记录inputs的使用情况
                if arg.name in func_inputs:
                    if func_inputs[arg.name] == 0:
                        # the first node to use this input
                        func_inputs[arg.name] = node.name
                        continue
                    else:
                        input_arg_id = node_name_to_shard_id[func_inputs[arg.name]]
                        if input_arg_id < min_shard_id:
                            min_shard_id = input_arg_id
                            min_node_name = func_inputs[arg.name]
                        continue

                if arg.name in node_name_to_shard_id and arg.name != nodes_so_far[-1]:
                    if node_name_to_shard_id[arg.name] < min_shard_id and arg.name \
                        not in node_user_count:
                        min_shard_id = node_name_to_shard_id[arg.name]
                        # print_rank_0(['because of ', arg.name, 
                        # node_name_to_shard_id[arg.name]])
                        min_node_name = arg.name

            # If there is an input that is not from the previous shard,
            # we collapse all the shards in between to be part of 1 shard.
            # and update the param count per shard accordingly.
            # 从args内shard_id最小的输入到当前node, 需要划分到同一个shard内, 
            # 当前shard_id也改为此shard_id

            if min_shard_id < shard_id:
                for node_name in reversed(nodes_so_far):
                    if node_name_to_shard_id[node_name] > min_shard_id:
                        # print_rank_0(['reset', node_name, 
                        # node_name_to_shard_id[node_name], min_shard_id])
                        node_name_to_shard_id[node_name] = min_shard_id
                    if node_name == min_node_name:
                        break
                shard_id = min_shard_id

                shard_to_param_count = \
                    _create_shard_to_param_count(param_count, node_name_to_shard_id)                
            # Update state that is tracking node -> shard id and shard id -> param count.
            node_name_to_shard_id[node.name] = shard_id
            # print_rank_0([node.name, node_name_to_shard_id[node.name]])
            nodes_so_far.append(node.name)

            shard_to_param_count = \
                _create_shard_to_param_count(param_count, node_name_to_shard_id)
            # print_rank_0([node.name, shard_to_param_count])
            # If we have gone over the number of params per shard count that we want to
            # achieve, we should add a new shard.
            # The shard_id may not have been updated in the map if we are at a node 
            # that does not have params.
            if shard_id in shard_to_param_count and \
                shard_to_param_count[shard_id] > per_shard_param \
                    and "embedding" not in node.name:
                shard_output[shard_id] = node.name
                reset_keys = [ i for i in shard_output.keys() if i > shard_id]
                for key in reset_keys:
                    shard_output.pop(key)

                # 如果需记录user的node中, node已产生且还有user未使用, 
                # 则需要成为这个shard的output
                extra_output[shard_id] = [k for k in node_user_count \
                    if node_user_count[k]>0 and k in node_name_to_shard_id]
    
                reset_keys = [ i for i in extra_output.keys() if i > shard_id]
                for key in reset_keys:
                    extra_output.pop(key)
                shard_id += 1
                # print_rank_0(['output', shard_output])
                # print_rank_0([extra_output, node_user_count])
        elif node.op == "output":
            break
        # print_rank_0([node.name, len(nodes_so_far)])    
    # print_rank_0(func_inputs)
    # print_rank_0([ node_name_to_shard_id[name] 
    # for name in shard_output.values() if name is not None])
    for k, v in func_inputs.items():
        func_inputs[k] = node_name_to_shard_id[v]
    # print_rank_0(func_inputs)
    return node_name_to_shard_id, shard_output, func_inputs, extra_output


def farthest_deps_split(
        traced_graph_module: GraphModule,
        model: torch.nn.Module,
        shard_count: int,
        output_nodes_count=None
    ) -> Tuple[List[GraphModule], Dict[str, int]]:
    # mapping node name to shard id
    node_name_to_shard_id, shard_output, func_inputs, extra_output = \
            farthest_deps_mapping(traced_graph_module, shard_count, output_nodes_count)
    
    # split graph
    module_list, func_inputs = split_module(traced_graph_module, model, node_name_to_shard_id)
    return module_list, func_inputs
