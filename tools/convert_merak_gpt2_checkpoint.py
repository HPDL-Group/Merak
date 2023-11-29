# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#

import argparse
import os
import re
import zipfile
from math import ceil, floor
import json

import torch

from transformers import AutoTokenizer, GPT2Config


####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

merak_to_merak = {
        "attn.c_attn": "attn.c_attn.col_linear",
        "attn.c_proj": "attn.c_proj.row_linear",
        "mlp.c_fc": "mlp.c_fc.col_linear",
        "mlp.c_proj": "mlp.c_proj.row_linear",
    }

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.col_linear.weight",
    "attn.c_attn.col_linear.bias",
    "attn.c_proj.row_linear.weight",
    "mlp.c_fc.col_linear.weight",
    "mlp.c_fc.col_linear.bias",
    "mlp.c_proj.row_linear.weight",
]

def partition_uniform(num_items, num_parts, use_ceil=True):
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    if use_ceil:
        chunksize = ceil(num_items / num_parts)
    else:
        chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
    return parts

def partition_layers(args, config):
    num_layers = config.num_hidden_layers + 3
    if args.partition_method == 'uniform':
        parts = partition_uniform(num_items=num_layers,
                                                num_parts=args.merak_pp, 
                                                use_ceil=True)
    elif args.partition_method == 'uniform_floor':
        parts = partition_uniform(num_items=num_layers,
                                                num_parts=args.merak_pp, 
                                                use_ceil=False)
    elif args.partition_method == 'custom':
        parts = [int(i) for i in args.custom_partition.split(",")]
    else:
        raise NotImplementedError(f'Partitioning method {method} not implemented.')
    return parts

####################################################################################################


def convert_megatron_checkpoint(args, input_state_dict, config):
    # args.target_tp, args.target_pp
    # The converted output model.
    output_state_dict = {}

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        # from pprint import pprint
        # pprint(vars(ds_args))

        config.vocab_size = ds_args.padded_vocab_size
        config.n_positions = ds_args.max_position_embeddings
        config.n_embd = ds_args.hidden_size
        config.n_layer = ds_args.num_layers
        config.n_head = ds_args.num_attention_heads
        config.n_inner = ds_args.ffn_hidden_size
        # pprint(config)

    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head #// args.target_tp
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # Set Merak modules
    module_num = 0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict[f"module.{module_num}.transformer.wte.weight"] = word_embeddings
    module_num += 1

    # The position embeddings.
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    # Read the causal mask dimension (seqlen). [max_sequence_length, hidden_size]
    n_positions = pos_embeddings.size(0)
    assert (
        n_positions == config.n_positions
    ), f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
    # Store the position embeddings.
    output_state_dict[f"module.{module_num}.transformer.wpe.weight"] = pos_embeddings
    module_num += 1

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_merak = {
        "attention.dense": ".attn.c_proj.row_linear.",
        "self_attention.dense": ".attn.c_proj.row_linear.",
        "mlp.dense_h_to_4h": ".mlp.c_fc.col_linear.",
        "mlp.dense_4h_to_h": ".mlp.c_proj.row_linear.",
    }

    pre_layer_idx = 0

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        if pre_layer_idx != layer_idx:
            module_num += 1
            pre_layer_idx = layer_idx

        # The name of the layer.
        layer_name = f"module.{module_num}.transformer.h.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):

            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Transpose the QKV matrix.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":

            # Insert a tensor of 1x1xDxD bias.
            causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
                1, 1, n_positions, n_positions
            )
            output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # Insert a "dummy" tensor for masked_bias.
            masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
            out_val = out_val.transpose(0, 1).contiguous()
            # Store.
            output_state_dict[layer_name + ".attn.c_attn.col_linear.weight"] = out_val

        # Transpose the bias.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":

            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Store. No change of shape.
            output_state_dict[layer_name + ".attn.c_attn.col_linear.bias"] = out_val

        # Transpose the weights.
        elif weight_or_bias == "weight":

            out_name = megatron_to_merak[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":

            out_name = megatron_to_merak[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val
    module_num += 1
    # # DEBUG.
    # assert config.n_layer == layer_idx + 1

    # The final layernorm.
    if "final_layernorm.weight" in transformer.keys():
        output_state_dict[f"module.{module_num}.transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
        output_state_dict[f"module.{module_num}.transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    if "word_embeddings_for_head" in model.keys():
        output_state_dict[f"module.{module_num}.lm_head.weight"] = model["word_embeddings_for_head"]["weight"]

    # It should be done!
    # print(config.n_layer)
    # for n, p in output_state_dict.items():
    #     print(n, p.size())
    return output_state_dict

####################################################################################################

def _vocab_size_with_padding(orig_vocab_size, tp_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    if tp_size == 1:
        return orig_vocab_size
    if orig_vocab_size % tp_size == 0:
        return orig_vocab_size
    after = orig_vocab_size
    make_vocab_size_divisible_by = 128
    multiple = make_vocab_size_divisible_by * \
        tp_size
    while (after % multiple) != 0:
        after += 1
    print(' > padded vocab (size: {}) with {} dummy tokens '
            '(new size: {})'.format(
                orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

def merak_shard(args, ds_or_merak_args, input_state_dict, config):
    # for n, p in input_state_dict.items():
    #     print(n, p.size())
    if hasattr(ds_or_merak_args, "curriculum_scheduler"):
        del ds_or_merak_args.curriculum_scheduler
    tp_size = args.merak_tp
    pp_size = args.merak_pp
    full_checkpoint = tp_size == 1 and pp_size == 1

    release_dir = os.path.join(args.output_folder, "release") if not full_checkpoint else args.output_folder
    os.makedirs(release_dir, exist_ok=True)

    # save dummy optim state dict
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,
        "param_groups": [
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],
            }
        ],
    }


    print("Converting")
    output_state_dict = []
    for i in range(tp_size):
        output_state_dict.append({})
        output_state_dict[i]["model"] = {}

    # Embedding layer
    print("converting embedding layer")
    pos_embedding = input_state_dict["module.1.transformer.wpe.weight"]
    word_embedding = input_state_dict["module.0.transformer.wte.weight"]
    orig_vocab_size = config.vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, tp_size)
    # setattr(margs, "padded_vocab_size", padded_vocab_size)
    # Cut out extra padding we don't need
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = word_embedding[0:padded_vocab_size, :]
    # Expanding embedding to larger size by replicating final entry
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size
        full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
    # Same size!
    else:
        full_word_embed = word_embedding

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, tp_size, dim=0)


    # Transformer layers
    print("converting transformer layers")
    if config.num_hidden_layers % tp_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
            f" ({tp_size})"
        )
    num_layers = config.num_hidden_layers + 3
    if num_layers < pp_size:
        raise ValueError(
            f"Number of layers ({num_layers}) must be divisible by number of pipeline parallelism"
            f" ({pp_size})"
        )
    # parts = partition_uniform(num_layers, pp_size, use_ceil=True)
    parts = partition_layers(args, config)

    layer_re = re.compile(r"module\.(\d+)\.transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head
    print(parts)
    for pp_rank in range(pp_size):
        num_layers_per_pp = parts[pp_rank+1] - parts[pp_rank]
        layer_offset = parts[pp_rank]  #pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(tp_size):
                output_state_dict.append({})
                output_state_dict[i]["model"] = {}

        if pp_rank == 0:
            for i in range(tp_size):
                output_state_dict[i]["model"]["module.0.transformer.wte.weight"] = out_word_embed[i]
                output_state_dict[i]["model"]["module.1.transformer.wpe.weight"] = pos_embedding

        for layer in range(num_layers_per_pp):
            if pp_rank == 0:
                pp_layer_id = layer + layer_offset
                pp_module_id = pp_layer_id + 2
                if pp_module_id >= num_layers_per_pp:
                    continue
            else:
                pp_layer_id = layer + layer_offset - 2
                pp_module_id = pp_layer_id + 2
            layers_to_copy = [
                layer_name
                for layer_name in input_state_dict.keys()
                if layer_name.startswith(f"module.{pp_module_id}.transformer.h.{pp_layer_id}.")
            ]

            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break

                # module index of the layer.
                module_idx = int(m.group(1))
                # The index of the layer.
                layer_idx = int(m.group(2))
                # The name of the operation.
                op_name = m.group(3)
                # Is it a weight or a bias?
                weight_or_bias = m.group(4)

                params = input_state_dict[layer_name]

                if tp_size > 1 and op_name in merak_to_merak:
                    new_op_name = merak_to_merak[op_name]
                    layer_name = layer_name.replace(op_name, new_op_name)
                    op_name = new_op_name

                if op_name + "." + weight_or_bias in tensor_parallel_params:

                    dim = 1 if op_name in ["attn.c_proj.row_linear", "mlp.c_proj.row_linear"] else 0
                    # print(layer_name, params.size(), dim)
                    if weight_or_bias == "weight" or ( op_name.startswith("attn.c_attn") and weight_or_bias == "weight"):
                        params = params.transpose(0, 1)

                    if op_name.startswith("attn.c_attn") and weight_or_bias == "weight":
                        params = transformers_to_megatron_fix_query_key_value_ordering(
                            params,
                            3.0,
                            3,
                            heads,
                            hidden_size_per_head,
                        )
                    elif op_name.startswith("attn.c_attn") and weight_or_bias == "bias":
                        params = transformers_to_megatron_fix_query_key_value_ordering(
                            params,
                            3.0,
                            3,
                            heads,
                            hidden_size_per_head,
                        )

                    assert params.size()[dim] % tp_size == 0
                    params = torch.chunk(params, tp_size, dim=dim)

                    if op_name.startswith("attn.c_attn") and weight_or_bias == "weight":
                        params = list(params)
                        for i in range(tp_size):
                            params[i] = fix_query_key_value_ordering(
                                    params[i],
                                    3.0,
                                    3,
                                    heads // tp_size,
                                    hidden_size_per_head,
                                )
                        # params[i] = params[i].transpose(0, 1).contiguous()
                    elif op_name.startswith("attn.c_attn") and weight_or_bias == "bias":
                        params = list(params)
                        for i in range(tp_size):
                            params[i] = fix_query_key_value_ordering(
                                    params[i],
                                    3.0,
                                    3,
                                    heads // tp_size,
                                    hidden_size_per_head
                                )

                    # print(layer_name, params[0].size(), dim)
                # else:
                #     print(op_name + "." + weight_or_bias)

                for i in range(tp_size):
                    output_state_dict[i]["model"][layer_name] = (
                        params[i] if (op_name + "." + weight_or_bias in tensor_parallel_params) else params
                    )
            # print(layer_name)

        if pp_rank == pp_size - 1:
            # handle final layernorm
            for weight_or_bias in ["weight", "bias"]:
                layer_name = f"module.{parts[pp_size]-1}.transformer.ln_f.{weight_or_bias}"
                # layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(tp_size):
                    output_state_dict[i]["model"][layer_name] = input_state_dict[layer_name]
            for i in range(tp_size):
                layer_name = f"module.{parts[pp_size]-1}.lm_head.weight"
                output_state_dict[i]["model"][layer_name] = out_word_embed[i] #input_state_dict[layer_name]
        # for i in range(tp_size):
        #     print("\n\n")
        #     for n, p in output_state_dict[i]["model"].items():
        #         print(n, p.size(), i, pp_rank)

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(tp_size):
            with open(args.output_folder + "/latest_checkpointed_iteration.txt", 'w') as f:
                f.write("release")
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            if not full_checkpoint:
                checkpoint_dir = (
                    f"mp_rank_{tp_rank:02d}"
                    if pp_size == 1
                    else f"mp_rank_{tp_rank:02d}_pp_rank_{pp_rank:03d}"
                )
            else:
                checkpoint_dir = 'iter_{:07d}_all_dp'.format(0)
            # if args.use_distributed_optimizer:
            #     checkpoint_name = "model_rng.pt"
            # else:
            checkpoint_name =  "model_optim.pt" if full_checkpoint else "partial_model_optim.pt"
            output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict["optimizer"]
            if ds_or_merak_args:
                output_state_dict[tp_rank]["args"] = ds_or_merak_args
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            # if args.print_checkpoint_structure:
            #     print(
            #         f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
            #         f" {pp_rank}:"
            #     )
            #     recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)

    # return

####################################################################################################

def get_parallel_degree(dirlist):
    if "pp_rank" in dirlist[0]:
        degree_re = re.compile(r"mp_rank\_([0-9_.]+)\_([a-z]+)\_([a-z]+)\_([0-9_.]+)")
    else:
        degree_re = re.compile(r"mp_rank\_([a-z0-9_]+)")

    pp_list = []
    tp_list = []
    for sub_dir in dirlist:
        degree = degree_re.match(sub_dir)
        tp = degree.group(1)
        tp_list.append(tp)
        if "pp_rank" in sub_dir:
            pp = degree.group(4)
            pp_list.append(pp)
    tp_size = int(max(tp_list)) + 1
    if pp_list:
        pp_size = int(max(pp_list)) + 1
    else:
        pp_size = 1
    return tp_size, pp_size
        
def get_merak_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from Merak checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        if tp_size == 1 and pp_size == 1:
            sub_dir_name = ""
        else:
            sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_pp_rank_{pp_rank:03d}"
        for checkpoint_name in ["partial_model_optim.pt", "model_optim.pt"]:
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
            if os.path.isfile(checkpoint_path):
                break
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def convert_checkpoint_from_split_to_one(args):
    """
    Convert Merak checkpoint to Merak checkpoint. This handles Merak checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using Merak checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_pp_rank_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    if "model_optim.pt" in sub_dirs:
        rank0_checkpoint_path = os.path.join(args.load_path, "model_optim.pt")
    print(f"Loading Merak checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    merak_args = state_dict.get("args", None)
    if args.json_path and merak_args is None:
        import json
        with open(args.json_path, "r")as f:
            config_dict = json.load(f)
            merak_args = argparse.Namespace(**config_dict)

    if merak_args is None:
        raise ValueError(
            "Merak checkpoint does not contain arguments. This utility only supports Merak checkpoints"
            " containing all the Merak arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Merak checkpoint along with all the Merak"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    # if merak_args is not None:
        # if merak_args.bias_gelu_fusion:
        #     activation_function = "gelu_fast"
        # elif merak_args.openai_gelu:
        #     activation_function = "gelu_new"
        # else:
    activation_function = "gelu"
    # else:
    #     # in the very early days this used to be "gelu_new"
    #     activation_function = "gelu_new"
    # vocab_size = (
    #     merak_args.padded_vocab_size
    #     if getattr(merak_args, "orig_vocab_size", None) is None
    #     else merak_args.orig_vocab_size
    # )
    if hasattr(merak_args, "orig_vocab_size"):
        vocab_size = merak_args.orig_vocab_size
    elif hasattr(merak_args, "padded_vocab_size"):
        vocab_size = merak_args.padded_vocab_size
    elif hasattr(merak_args, "vocab_size"):
        vocab_size = merak_args.vocab_size
    else:
        raise ValueError("Couldn't find vocab size in merak args, please set vocab_size")


    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=merak_args.max_position_embeddings
                    if getattr(merak_args, "n_positions", None) is None
                    else merak_args.n_positions,   #max_position_embeddings,
        n_embd=merak_args.hidden_size
                    if getattr(merak_args, "n_embd", None) is None
                    else merak_args.n_embd,
        n_layer=merak_args.num_layers
                    if getattr(merak_args, "n_layer", None) is None
                    else merak_args.n_layer,
        n_head=merak_args.num_attention_heads
                    if getattr(merak_args, "n_head", None) is None
                    else merak_args.n_head,
        n_inner=merak_args.ffn_hidden_size
                    if getattr(merak_args, "n_inner", None) is None
                    else merak_args.n_inner,
        activation_function=activation_function,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
        architectures=["GPT2LMHeadModel"],
    )

    if args.convert_type == "merak" and "model_optim.pt" in sub_dirs:
        return state_dict["model"], config, merak_args

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)

    # Get parallel degree
    if "model_optim.pt" in sub_dirs:
        tp_size,  pp_size = (1, 1)
    else:
        tp_size,  pp_size = get_parallel_degree(sub_dirs)
    args.merak_tp = tp_size
    args.merak_pp = pp_size
    dtype = torch.float32
    # The regex to extract layer names.
    # layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    layer_re = re.compile(r"module\.(\d+)\.transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_merak_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the position embeddings.
    position_embeddings = tp_state_dicts[0]["model"]["module.1.transformer.wpe.weight"]

    

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            tp_state_dicts[tp_rank]["model"]["module.0.transformer.wte.weight"]
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict["module.0.transformer.wte.weight"] = word_embeddings
    output_state_dict["module.1.transformer.wpe.weight"] = position_embeddings.to(dtype)

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head
    n_positions = config.n_positions
    # num_layers = config.num_hidden_layers // pp_size
    # parts = partition_uniform(config.num_hidden_layers, pp_size, use_ceil=True)/
    parts = partition_layers(args, config)
    print(parts)

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_merak_sharded_states(args, tp_size, pp_size, pp_rank)
    
        count = 0
        for key, val in tp_state_dicts[0]["model"].items():
            # Pass embedding layer
            if pp_rank == 0 and count <= 1:
                count += 1
                continue
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break

            # The index of the module
            module_idx = int(m.group(1))
            # The index of the layer.
            layer_idx = int(m.group(2))
            # The name of the operation.
            op_name = m.group(3)
            # Is it a weight or a bias?
            weight_or_bias = m.group(4)

            # The name of the layer.
            layer_name = f"module.{module_idx}.transformer.h.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["attn.c_proj.row_linear", "mlp.c_proj.row_linear"] else 0
                params = torch.cat(
                    [val]
                    + [
                        tp_state_dicts[tp_rank]['model'][key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.startswith("ln"):
                # ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
                output_state_dict[key] = params

            # Transpose the QKV matrix.
            elif op_name == "attn.c_attn" and weight_or_bias == "weight":
                output_state_dict[key] = params

            # Transpose the bias.
            elif op_name == "attn.c_attn" and weight_or_bias == "bias":
                output_state_dict[key] = params

            # Transpose the weights.
            elif weight_or_bias == "weight":
                output_state_dict[key] = params

            # Copy the bias.
            elif weight_or_bias == "bias":
                output_state_dict[key] = params

        # if config.n_layer != (layer_idx + 1):
        #     raise ValueError(f"Expected {config.n_layer} layers but found {layer_idx + 1}")

        # The final layernorm.
        if pp_rank + 1 == pp_size:
            print("Converting final layernorm")
            module_idx += 1
            for key, val in tp_state_dicts[0]["model"].items():
                if "ln_f" in key:
                    output_state_dict[key] = val
                    output_state_dict[key] = val
                if "lm_head" in key:
                    # For LM head, transformers' wants the matrix to weight embeddings.
                    print("Converting LM head")
                    output_state_dict[f"module.{module_idx}.lm_head.weight"] = word_embeddings

    # It should be done!
    print("Conversion from Merak multiple checkpoints to single checkpoint is done!")
    # for k, v in output_state_dict.items():
    #     print(k, v.size())
    # os._exit(0)
    return output_state_dict, config, merak_args

####################################################################################################
def split_model_name(key):
    layer_re = re.compile(r"module\.(\d+)\.([a-z0-9_.]+)")
    match = layer_re.match(key)
    return match.group(2)

def convert_checkpoint_from_merak_to_transformers(args):
    input_state_dict, config, merak_args = convert_checkpoint_from_split_to_one(args)

    output_state_dict = {}

    for k, v in input_state_dict.items():
        new_k = split_model_name(k)
        output_state_dict[new_k] = v
        
    
    # for k, v in output_state_dict.items():
    #     print(k, v.size())
    # os._exit(0)

    # Store the config to file.
    output_config_file = os.path.join(args.save_path, "config.json")
    os.makedirs(args.save_path, exist_ok=True)
    config.model_type = "gpt2"
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        f.write(json.dumps(config.to_dict(), sort_keys=True , indent=4))

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(args.save_path, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)

####################################################################################################

def convert_transformers_checkpoint(args, input_state_dict, config):
    # args.target_tp, args.target_pp
    # The converted output model.
    output_state_dict = {}

    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head #// args.target_tp

    # Set Merak modules
    module_num = 0

    # The embedding.
    try:
        output_state_dict[f"module.{module_num}.transformer.wte.weight"] = input_state_dict["transformer.wte.weight"]
        transformer = "transformer."
    except KeyError:
        output_state_dict[f"module.{module_num}.transformer.wte.weight"] = input_state_dict["wte.weight"]
        transformer = ""
    module_num += 1
    output_state_dict[f"module.{module_num}.transformer.wpe.weight"] = input_state_dict[f"{transformer}wpe.weight"]
    module_num += 1

    # The regex to extract layer names.
    layer_re = re.compile(f"{transformer}h\.(\d+)\.([a-z0-9_.]+)")
    pre_layer_idx = 0

    has_masked_bias = False
    for key in input_state_dict.keys():
        if ".attn.masked_bias" in key:
            has_masked_bias = True
            break
    # Extract the layers.
    for key, val in input_state_dict.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            continue

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # # Is it a weight or a bias?
        # weight_or_bias = m.group(3)

        if pre_layer_idx != layer_idx:
            module_num += 1
            pre_layer_idx = layer_idx

        # The name of the layer.
        layer_name = f"module.{module_num}.transformer.h.{layer_idx}.{op_name}"
        if ".attn.bias" in layer_name:
            output_state_dict[layer_name] = val
            if not has_masked_bias:
                masked_bias = torch.tensor(-1e4, dtype=torch.float16)
                output_state_dict[layer_name.replace("bias", "masked_bias")] = masked_bias
        else:
            output_state_dict[layer_name] = val
    module_num += 1
    # # DEBUG.
    # assert config.n_layer == layer_idx + 1

    # The final layernorm.
    output_state_dict[f"module.{module_num}.transformer.ln_f.weight"] = input_state_dict[f"{transformer}ln_f.weight"]
    output_state_dict[f"module.{module_num}.transformer.ln_f.bias"] = input_state_dict[f"{transformer}ln_f.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    if transformer == "transformer.":
        output_state_dict[f"module.{module_num}.lm_head.weight"] = input_state_dict["lm_head.weight"]
    else:
        output_state_dict[f"module.{module_num}.lm_head.weight"] = input_state_dict["wte.weight"]

    # It should be done!
    # print(config.n_layer)
    # for n, p in output_state_dict["model"].items():
    #     print(n, p.size())
    # os._exit(0)
    return output_state_dict

####################################################################################################

def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        type=str,
        help="Path to the checkpoint folder (iter000xxx or release folder)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Save checkpoint folder path",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        help="load json file",
    )
    parser.add_argument(
        "--convert_type",
        default="",
        type=str,
        choices=["merak", "transformers", "transformers_to_merak"],
        required=True,
        help="Choose the convert type, merak to merak or merak to transformers.",
    )
    parser.add_argument('--merak_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--merak_pp', default=1, type=int, help='Target PP degree')
    parser.add_argument('--partition_method', default="uniform", type=str, 
                choices=['uniform','uniform_floor','custom'],
                help='Possible choices are the pipeline layer partion strategy as strings, Defaults to uniform.')
    parser.add_argument('--custom_partition', default=None, type=str, 
                help='Customize the partition size of the model. Length of list is pipeline_world_size + 1.'
                'Example: [0, 6, 12, 18, 26, ..., last_layer_idx]')
    args = parser.parse_args()

    # Convert.
    print("Converting")
    if args.convert_type == "merak":
        args.output_folder = args.save_path
        output_state_dict, config, merak_args = convert_checkpoint_from_split_to_one(args)
        merak_shard(args, merak_args, output_state_dict, config)
    elif args.convert_type == "transformers":
        convert_checkpoint_from_merak_to_transformers(args)
    elif args.convert_type == "transformers_to_merak":
        args.output_folder = args.save_path
        input_state_dict = torch.load(args.load_path, map_location='cpu')
        with open(args.json_path, "r") as f:
            config_kwarg = json.load(f)
        config = GPT2Config(
            **config_kwarg
        )
        output_state_dict = convert_transformers_checkpoint(args, input_state_dict, config)
        merak_shard(args, None, output_state_dict, config)
    else:
        raise ValueError("Input type of converting error")


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
