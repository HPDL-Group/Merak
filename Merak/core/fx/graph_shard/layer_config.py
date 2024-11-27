# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com), Yck (eyichenke@gmail.com)
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

# Parts of the code here are adapted from https://github.com/SymbioticLab/Oobleck/blob/sosp23-artifact/oobleck/module/sharding.py

import torch
import torch.fx
import torch.distributed as dist
import warnings

from itertools import chain
from torch.fx.node import Node
from collections import defaultdict
from transformers import PretrainedConfig
from torch.fx.graph_module import GraphModule
from typing import Dict, List, Set, Any, Optional, Type, Tuple

from Merak.merak_args import get_args
from .split_module import split_module


def get_split_points(config: Type[PretrainedConfig]) -> List[str]:
    args = get_args()
    split_points = []

    if "gpt2" == config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"transformer.h.{i}")
        split_points.append("transformer.ln_f")
    elif "bert" == config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"bert.encoder.layer.{i}")
        # split_points.append("cls")
    elif "t5" in config.model_type:
        for i in range(config.num_layers):
            split_points.append(f"encoder.block.{i}")
        for i in range(config.num_decoder_layers):
            split_points.append(f"decoder.block.{i}")
        split_points.append("lm_head")
    # Sharding for the Google's HuggingFace ViT model
    # e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
    elif "vit" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"vit.encoder.layer.{i}")
        split_points.append("vit.layernorm")
    # Sharding for the Microsoft's HuggingFace ResNet model
    # e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
    elif "resnet" in config.model_type:
        for i, depth in enumerate(config.depths):
            for j in range(depth):
                split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")
        split_points.append("resnet.pooler")
    elif "dinov2" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"dinov2.encoder.layer.{i}")
        split_points.append("dinov2.layernorm")
        split_points.append("classifier")
    elif "trocr" in config.model_type:
        for i in range(config.decoder_layers):
            split_points.append(f"model.decoder.layers.{i}")
    elif "lxmert" in config.model_type:
        split_points.append(f"lxmert.encoder.visn_fc")
        for i in range(config.l_layers):
            split_points.append(f"lxmert.encoder.layer.{i}")
        for i in range(config.x_layers):
            split_points.append(f"lxmert.encoder.x_layers.{i}")
        for i in range(config.r_layers):
            split_points.append(f"lxmert.encoder.r_layers.{i}")
        # split_points.append(f"xmert.pooler.dense")
    elif "altclip" == config.model_type:
        # split_points.append(f"logit_scale")
        for i in range(config.text_config.num_hidden_layers):
            split_points.append(f"text_model.roberta.encoder.layer.{i}")
        # split_points.append(f"text_model.pre_LN")
        # split_points.append(f"vision_model.pre_layrnorm")
        for i in range(config.vision_config.num_hidden_layers):
            split_points.append(f"vision_model.encoder.layers.{i}")
        # split_points.append(f"vision_model.post_layernorm")
        # split_points.append(f"visual_projection")
        # split_points.append(f"text_projection")
    elif "clip" == config.model_type:
        split_points.append(f"logit_scale")
        for i in range(config.text_config.num_hidden_layers):
            split_points.append(f"text_model.encoder.layers.{i}")
        split_points.append(f"text_model.final_layer_norm")
        split_points.append(f"vision_model.pre_layrnorm")
        for i in range(config.vision_config.num_hidden_layers):
            split_points.append(f"vision_model.encoder.layers.{i}")
        split_points.append(f"vision_model.post_layernorm")
        split_points.append(f"visual_projection")
        split_points.append(f"text_projection")
    elif "deberta" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"deberta.encoder.layer.{i}")
        split_points.append(f"cls.predictions")
    elif "gptj" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"transformer.h.{i}")
        split_points.append("lm_head")
    elif "m2m" in config.model_type:
        for i in range(config.encoder_layers):
            split_points.append(f"model.encoder.layers.{i}")
        split_points.append("model.encoder.layer_norm")
        for i in range(config.decoder_layers):
            split_points.append(f"model.decoder.layers.{i}")
        split_points.append("model.decoder.layer_norm")
    elif "mobilebert" in config.model_type:
        split_points.append("mobilebert.embeddings")
        for i in range(config.num_hidden_layers):
            split_points.append(f"mobilebert.encoder.layer.{i}")
        split_points.append("cls.predictions")
    elif "mt5" in config.model_type:
        for i in range(config.num_layers):
            split_points.append(f"encoder.block.{i}")
        split_points.append("encoder.final_layer_norm")
        for i in range(config.num_decoder_layers):
            split_points.append(f"decoder.block.{i}")
        split_points.append("decoder.final_layer_norm")
        split_points.append("lm_head")
    elif "pegasus" in config.model_type:
        for i in range(config.encoder_layers):
            split_points.append(f"model.encoder.layers.{i}")
        split_points.append("model.encoder.layer_norm")
        for i in range(config.decoder_layers):
            split_points.append(f"model.decoder.layers.{i}")
        split_points.append("model.decoder.layer_norm")
    elif "wav2vec2" in config.model_type:
        for i in range(len(config.conv_dim)):
            split_points.append(f"wav2vec2.feature_extractor.conv_layers.{i}")
        # split_points.append("wav2vec2.feature_projection.layer_norm")
        # split_points.append("wav2vec2.feature_projection.projection")
        # split_points.append("wav2vec2.encoder.layer_norm")
        for i in range(config.num_hidden_layers):
            split_points.append(f"wav2vec2.encoder.layers.{i}")
        split_points.append("lm_head")
    elif "distilbert" == config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"distilbert.transformer.layer.{i}")
        split_points.append("vocab_transform")
    elif "marian" in config.model_type:
        for i in range(config.encoder_layers):
            split_points.append(f"model.encoder.layers.{i}.final_layer_norm")
        for i in range(config.decoder_layers):
            split_points.append(f"model.decoder.layers.{i}.final_layer_norm")
        # split_points.append("lm_head")
    elif "opt" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"model.decoder.layers.{i}")
        # split_points.append("lm_head")
    elif "swin" in config.model_type:
        for i, depth in enumerate(config.depths):
            for j in range(depth):
                split_points.append(f"swin.encoder.layers.{i}.blocks.{j}.attention")
    elif "llama" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"model.layers.{i}")
    elif "qwen" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"model.layers.{i}")
        split_points.append("lm_head")
    elif args.custom_split_points is not None:
        assert isinstance(args.custom_split_points, list)
        split_points = args.custom_split_points

    assert (
        split_points
    ), f"Split points is empty. Check your model {config.model_type} is supported."

    return split_points


def layer_config_mapping(
        traced: torch.fx.GraphModule,
        split_points: List[str]
    ) -> Tuple[Dict[str, int], Dict[int, List[str]], Dict[str, int]]:
    """Analyze the given traced module and split it to subgraphs.
    While partitioning, it also finds additioanl required inputs and outputs
    so that they are added.

    Args:
        traced (torch.fx.GraphModule): A traced graph module to be split.
    """

    node_name_to_shard_id: Dict[str, int] = {}
    shard_id_to_node: Dict[int, List[Node]] = defaultdict(list)
    shard_id = 0

    nodes_so_far: List[str] = []
    # func_inputs: List[str] = []
    extra_output: Dict[int, Node] = {}

    func_inputs: Dict[Node, int] = {}
    shard_output = {-1: None, 0: None}

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)
            # func_inputs.append(node)
            func_inputs[node.name] = shard_id
        elif node.op in [
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)

            point = next(
                filter(lambda p: p in node.next.name, split_points), None
            )
            if point:
                # Record outputs that should be used later, so that it can be added
                # in return of this shard
                outputs = []
                nodes = list(chain(*shard_id_to_node.values()))
                for node in nodes:
                    for user in node.users.keys():
                        if user.name not in node_name_to_shard_id:
                            outputs.append(node.name)

                extra_output[shard_id] = list(dict.fromkeys(outputs).keys())

                # If the current node is in the next shard, we increase shard count.
                shard_output[shard_id] = point
                shard_id += 1
                split_points.remove(point)

        elif node.op == "output":
            break

    # assert len(split_points) == 0, f"Sharding is not complete. {split_points} not sharded"
    if len(split_points) != 0 and dist.get_rank() == 0:
        warnings.warn(f"Sharding is not complete. {split_points} not sharded", Warning)

    return node_name_to_shard_id, extra_output, func_inputs, shard_output

def layer_config_split(
        traced_graph_module: GraphModule,
        model: torch.nn.Module
    ) -> Tuple[List[GraphModule], Dict[str, int]]:
    # mapping node name to shard id 
    split_points = get_split_points(model.config)
    split_points = [p.replace(".", "_") for p in split_points]
    node_name_to_shard_id, extra_output, func_inputs, shard_output = \
        layer_config_mapping(traced_graph_module, split_points)
    
    # split graph
    module_list, func_inputs = split_module(traced_graph_module, model, node_name_to_shard_id)
    return module_list, func_inputs