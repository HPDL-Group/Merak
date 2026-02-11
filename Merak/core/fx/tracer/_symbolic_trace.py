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

import copy
import inspect
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.fx import Graph
from transformers import logging
from transformers.utils.fx import HFTracer

from Merak import get_logger

logger = get_logger("simple")


class MerakTracer(HFTracer):
    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        super().__init__()
        self.leaf_modules = autowrap_functions

    def is_manual_leaf_module(self, m):
        for i in self.leaf_modules:
            if isinstance(m, i):
                return True
        return False

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if hasattr(self, "_stateless_mod_instanciation_depends_on_proxies"):
            return (
                (not self._stateless_mod_instanciation_depends_on_proxies(m))
                and super().is_leaf_module(m, module_qualified_name)
                or self.is_manual_leaf_module(m)
            )
        return (
            (not self._stateless_mod_instantiation_depends_on_proxies(m))
            and super().is_leaf_module(m, module_qualified_name)
            or self.is_manual_leaf_module(m)
        )


def tf_symbolic_trace(
    model: torch.nn.Module,
    input_names: List[str] = None,
    tracer_cls=MerakTracer,
    leaf_modules=(),
) -> Graph:
    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
        disable_check (`bool`, *optional*, defaults to `False`):
            If `True`, no check is done before trying to trace the model, this is mostly usesul for debugging purposes.
        tracer_cls (`Type[HFTracer]`, *optional*, defaults to `HFTracer`):
            The tracer class to use for instantiating the tracer. If unset, `HFTracer` is used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    if "past_key_values" in input_names and not getattr(
        model.config, "use_cache", False
    ):
        logger.warning(
            "`past_key_values` were specified as input names, but model.config.use_cache = False, this might lead to "
            "unexpected behavior."
        )
    if "past_key_values" not in input_names and getattr(
        model.config, "use_cache", False
    ):
        logger.warning(
            "`past_key_values` were not specified as input names, but model.config.use_cache = True. Setting "
            "model.config.use_cache = False."
        )
        model.config.use_cache = False

    # Tracing.
    tracer = tracer_cls(autowrap_functions=leaf_modules)
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    traced.config = model.config
    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return traced


def get_concrete_args(
    model: torch.nn.Module, input_names: List[str]
) -> Dict[str, None]:
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = (
            input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        )
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    return {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
