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
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed
import torch.nn as nn
from tabulate import tabulate
from torch.fx.graph_module import GraphModule
from torch.nn import Module
from transformers import Conv1D
from transformers.utils.fx import _SUPPORTED_MODELS as tf_supported_models

from Merak import get_logger

from .. import mpu
from .graph_shard import _shard_model_transformers
from .tracer import LayerProxyTracer, dynamo_trace, tf_symbolic_trace
from .tracer.utils import _generate_dummy_input
from .utils import _load_cache, _save_cache, _split_attr_values


def convert_to_sequential(
    model: Module,
    args: Any,
    dummy_inputs: Optional[Dict[str, torch.Tensor]] = None,
    extra_leaf_modules: Tuple[Module] = (),
) -> Tuple[Module, List[GraphModule], Dict[str, int]]:
    """
    Convert a PyTorch model to a sequential model for efficient sharding.

    Args:
        model: Input PyTorch model to convert.
        args: MerakArguments containing configuration for conversion.
        dummy_inputs: Optional dummy inputs for model tracing.
        extra_leaf_modules: Additional leaf modules to consider during tracing.

    Returns:
        Tuple containing:
            - Original model
            - List of sharded GraphModule instances
            - Dictionary containing input sharding information
    """
    if args.fp16:
        model = model.half()
    if args.bf16:
        model = model.bfloat16()
    logger = get_logger("simple")
    if args.cache_sharding:
        # Load cached sharding results if available
        result, input_to_shard = _load_cache(args)
        return model, result, input_to_shard

    # Validate tracing method
    assert args.trace_method in [
        "fx",
        "dynamo",
    ], f"Unsupported trace method: {args.trace_method}"

    # Initialize supported models
    supported_models = tf_supported_models + (args.trace_model,)

    # Configure extra leaf modules
    extra_leaf_modules = (Conv1D,) + extra_leaf_modules

    if model.__class__.__name__ in supported_models:
        # Generate dummy inputs if not provided
        dummy_inputs = _generate_dummy_input(args, model)

        if args.trace_method == "fx":
            # Create symbolic trace using Transformers-fx tracing
            traced = tf_symbolic_trace(
                model,
                input_names=args.input_names,
                leaf_modules=extra_leaf_modules,
            )
            # Split attribute values for optimization
            traced = _split_attr_values(model, traced, args)
    else:
        # Fallback tracing for non-Transformers models
        if args.trace_method == "fx":
            if isinstance(extra_leaf_modules, list):
                extra_leaf_modules = tuple(extra_leaf_modules)
            elif isinstance(extra_leaf_modules, nn.Module):
                extra_leaf_modules = tuple([extra_leaf_modules])
            else:
                assert isinstance(
                    extra_leaf_modules, tuple
                ), "leaf_modules must be a tuple"

            leaf_modules = extra_leaf_modules
            tracer = LayerProxyTracer(leaf_modules)
            traced_graph = tracer.trace(model)
            traced = torch.fx.GraphModule(model, traced_graph)

    if args.trace_method == "dynamo":
        assert (
            mpu.get_model_parallel_world_size() == 1
        ), "Dynamo tracing not supported with tensor parallelism"
        traced = dynamo_trace(model, dummy_inputs)

    # Debugging: Print graph details
    logger.debug(traced.graph, ranks=[0])
    node_specs = [
        [n.op, n.name, n.target, n.args, n.kwargs] for n in traced.graph.nodes
    ]
    msg = tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"])
    logger.debug(msg, ranks=[0])
    logger.debug(traced.code, ranks=[0])
    logger.debug(traced, ranks=[0])
    logger.debug(model, ranks=[0])

    # Shard the traced GraphModule
    result, input_to_shard = _shard_model_transformers(traced, model, args)

    # Clean up memory
    del traced
    gc.collect()
    if not args.use_cpu:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Store dummy inputs for future use
    result[0].dummy_inputs = dummy_inputs

    # Save caching results if enabled
    if args.cache_sharding:
        _save_cache(result, input_to_shard, args)

    # Debugging: Print final graph details
    for idx, i in enumerate(result):
        logger.debug(f"==================={idx}======================", ranks=[0])
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in i.graph.nodes]
        msg = tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        logger.debug(msg, ranks=[0])
        logger.debug(i.code, ranks=[0])

    return model, result, input_to_shard


def add_inputs_to_shards(gm: GraphModule, inputs: List[str]) -> GraphModule:
    """
    Add input placeholders to the sharded GraphModule.

    Args:
        gm: GraphModule to modify.
        inputs: List of input names to add as placeholders.

    Returns:
        Modified GraphModule with additional input placeholders.
    """
    add_outputs = []
    for node in gm.graph.nodes:
        if (
            node.op == "placeholder"
            and node.next.op != "placeholder"
            and add_outputs == []
        ):
            with gm.graph.inserting_after(node):
                for input_name in reversed(inputs):
                    pl_node = gm.graph.create_node("placeholder", input_name)
                    add_outputs.append(pl_node)
        elif node.op == "output":
            with gm.graph.inserting_after(node):
                node_inputs = (
                    tuple(node.args) if len(node.args) == 1 else tuple(node.args[0])
                )
                added_output = node_inputs + tuple(reversed(add_outputs))
                gm.graph.create_node(op="output", target="output", args=(added_output,))
                # gm.graph.output(added_output)
            gm.graph.erase_node(node)
            break
    gm.recompile()
    return gm
