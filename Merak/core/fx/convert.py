# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com), Yck(eyichenke@gmail.com)
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

import gc
import torch
import torch.nn as nn

from torch.nn import Module
from torch.fx.graph_module import GraphModule
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.utils.fx import _SUPPORTED_MODELS as tf_suported_models
from typing import Dict, List, Set, Any, Optional, Tuple

from Merak.merak_args import MerakArguments

from .tracer.utils import _generate_dummy_input
from .tracer import (
    LayerProxyTracer,
    tf_symbolic_trace,
    dynamo_trace
)
from .graph_shard import _shard_model_transformers
from .utils import (
    _load_cache, _save_cache,
    _split_attr_values,
    )

def convert_to_sequential(
        model: Module,
        args: MerakArguments,
        extra_leaf_modules: Tuple[Module] = ()
    ) -> Tuple[Module, List[GraphModule], Dict[str, int]]:
    
    if args.cache_sharding:
        result, input_to_shard = _load_cache(args)
        return model, result, input_to_shard

    assert args.trace_method in ['fx', 'dynamo']

    if args.trace_method == 'fx':
        new_suported_models = tf_suported_models + (args.trace_model,)
        extra_leaf_modules = (T5LayerNorm,) + extra_leaf_modules

        if model.__class__.__name__ in new_suported_models:

            traced = tf_symbolic_trace(
                model,
                input_names=args.input_names,
                leaf_modules=extra_leaf_modules
            )
            # 用于修改trace后的常数
            traced = _split_attr_values(model, traced, args)
            dummy_inputs = args.input_names
        else:
            if isinstance(extra_leaf_modules, list):
                extra_leaf_modules = tuple(extra_leaf_modules)
            elif isinstance(extra_leaf_modules, nn.Module):
                extra_leaf_modules = tuple([extra_leaf_modules])
            else:
                assert isinstance(extra_leaf_modules, tuple), 'leaf_modules should be tuple'
            
            leaf_modules = extra_leaf_modules
            tracer = LayerProxyTracer(leaf_modules)
            traced_graph = tracer.trace(model)
            traced = torch.fx.GraphModule(model, traced_graph)
            dummy_inputs = _generate_dummy_input(args, model)
    elif args.trace_method == 'dynamo':
        traced, dummy_inputs = dynamo_trace(model, args)

    ## test code
    # print_rank_0(traced.graph)
    # if torch.distributed.get_rank() == 0:
    #   traced.graph.print_tabular()
    # print_rank_0(traced.code)
    # print_rank_0(traced)
    # print_rank_0(model)

    # shard GraphModule to List[GraphModule]
    result, input_to_shard = _shard_model_transformers(
            traced, model, args.shard_count
        )

    del traced
    gc.collect()
    if not args.use_cpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    result[0].dummy_inputs = dummy_inputs

    if args.cache_sharding:
        _save_cache(result, input_to_shard, args)

    # # test code
    # for idx, i in enumerate(result):
    #     if torch.distributed.get_rank() == 0:
    #         print(f"==================={idx}======================")
    #         i.graph.print_tabular()
    #     print_rank_0(i.code)

    return model, result, input_to_shard

def add_inputs_to_shards(gm: GraphModule, inputs: List[str]) -> GraphModule:
    # for input_name in inputs:
    add_outputs = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder' and node.next.op != 'placeholder' and add_outputs == []:
            with gm.graph.inserting_after(node):
                for input_name in reversed(inputs):
                    pl_node = gm.graph.create_node("placeholder", input_name)
                    add_outputs.append(pl_node)
        elif node.op == "output":
            with gm.graph.inserting_after(node):
                node_inputs = tuple(node.args) if len(node.args) == 1 else tuple(node.args[0])
                added_output = node_inputs + tuple(reversed(add_outputs))
                gm.graph.create_node(op='output', target='output', args=(added_output,))
                # gm.graph.output(added_output)
            gm.graph.erase_node(node)
            break
    gm.recompile()
    return gm
