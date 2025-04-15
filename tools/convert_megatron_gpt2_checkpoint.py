####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
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

####################################################################################################

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

# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py


import argparse
import os
import re
import zipfile
import sys

import torch

from transformers import AutoTokenizer, GPT2Config
import torch.distributed as dist

####################################################################################################

USE_MERAK = False
USE_MEGATRON = False

try:
    import Merak
    from Merak import mpu, print_rank_0
    USE_MERAK = True
except:
    pass

try:
    import megatron
    from megatron import mpu, print_rank_0
    USE_MEGATRON = True
except:
    pass

if not USE_MEGATRON and not USE_MERAK: 
    raise ImportError("Megatron or Merak not found, please install Megatron or Merak")

if  USE_MEGATRON and  USE_MERAK: 
    USE_MERAK = True 
    USE_MEGATRON = False

def init_model_parallel():
    if USE_MERAK:
        pipe_rank = mpu.get_pipe_parallel_rank()
        pipe_size = mpu.get_pipe_parallel_world_size()
        pipe_group = mpu.get_pipe_parallel_group()
        model_rank = mpu.get_model_parallel_rank()
        model_size = mpu.get_model_parallel_world_size()
        prev_rank = mpu.get_pipeline_model_parallel_prev_rank()
        next_rank = mpu.get_pipeline_model_parallel_next_rank()
        global_rank_of_stage_0 = Merak.get_grid().stage_to_global(0)
    elif USE_MEGATRON:
        pipe_rank = mpu.get_pipeline_model_parallel_rank()
        pipe_size = mpu.get_pipeline_model_parallel_world_size()
        pipe_group = mpu.get_pipeline_model_parallel_group()
        model_rank = mpu.get_tensor_model_parallel_rank()
        model_size = mpu.get_tensor_model_parallel_world_size()
        prev_rank = mpu.get_pipeline_model_parallel_prev_rank()
        next_rank = mpu.get_pipeline_model_parallel_next_rank()
        global_rank_of_stage_0 = mpu.get_pipeline_model_parallel_first_rank()
    else:
        raise ValueError("Megatron or Merak not found, please install Megatron or Merak")

        
    return (pipe_rank,  pipe_size,  pipe_group,  model_rank,  model_size,  prev_rank,  next_rank, global_rank_of_stage_0)

####################################################################################################


def recursive_print(name, val, path_to_output_checkpoint, parallel_args, spaces=0):
    # Format the message.
    # For printing model structure
    pipe_rank, pipe_size, pipe_group, model_rank, model_size, prev_rank, next_rank, global_rank_of_stage_0 = parallel_args

    if name is None:
        msg = None
    else:
        # fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        fmt = "{:" + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], path_to_output_checkpoint, parallel_args, spaces + 2)
    elif isinstance(val, torch.Tensor):
        with open(f"{path_to_output_checkpoint}/pp{pipe_rank}_tp{model_rank}.txt", "w") as f:
            f.write(f"{msg},{val.size()} \n")
        # print(msg, ":", val.size())
    else:
        with open(f"{path_to_output_checkpoint}/pp{pipe_rank}_tp{model_rank}.txt", "w") as f:
            f.write(f"{msg},{val} \n ")
        # print(msg, ":", val)


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

####################################################################################################

def init_p2p():
    next_rank = None
    prev_rank = None
    num_stages = Merak.get_grid().pipe_parallel_size
    stage_id = Merak.get_grid().get_stage_id()
    prev_stage = stage_id - 1
    next_stage = stage_id + 1
    if 0 <= next_stage < num_stages:
        next_rank = Merak.get_grid().stage_to_global(stage_id=next_stage)
    if 0 <= prev_stage < num_stages:
        prev_rank = Merak.get_grid().stage_to_global(stage_id=prev_stage)
    mpu.initialize.set_pipeline_model_parallel_next_rank(next_rank)
    mpu.initialize.set_pipeline_model_parallel_prev_rank(prev_rank)

####################################################################################################

def cal_layers_num(lm, layer_re):
    fl_layers = 1
    if 'embedding' in lm.keys():
        embed_layers = len(lm['embedding'].keys())
    else:
        embed_layers = 0
    if "final_layernorm.weight" in lm['encoder'].keys():
        include_fl_layers = int(layer_re.match(list(lm['encoder'].keys())[-3]).group(1)) + 1
        if len(lm.keys()) == 1:
            layers_num = include_fl_layers + fl_layers
        elif len(lm.keys()) == 2:
            layers_num = include_fl_layers + fl_layers + embed_layers
        else:
            raise KeyError(f"The language_model has error numbers of {len(lm.keys())} keys")
    else:
        normal_layers = int(layer_re.match(list(lm['encoder'].keys())[-1]).group(1)) + 1
        if len(lm.keys()) == 1:
            layers_num = normal_layers
        elif len(lm.keys()) == 2:
            layers_num = normal_layers + embed_layers
        else:
            raise KeyError(f"The language_model has error numbers of {len(lm.keys())} keys")
    return layers_num

####################################################################################################


def convert_megatron_checkpoint(args, input_state_dict, config, parallel_args):
    # set device
    if torch.cuda.is_available() and args.distributed_backend == "nccl":
        device = f"cuda:{args.local_rank}"
    else:
        device = "cpu"

    pipe_rank, pipe_size, pipe_group, model_rank, model_size, prev_rank, next_rank, global_rank_of_stage_0 = parallel_args

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
    hidden_size_per_head = config.n_embd // config.n_head // model_size
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]

    # get numbers of layers
    layers_num_tensor = torch.Tensor([0]).to(device)
    layers_num_tensor_var = torch.Tensor([0]).to(device)
    if pipe_rank == 0:
        layers_num = 0
        first_layer_idx = 0
    else:
        dist.recv(layers_num_tensor, prev_rank, group=pipe_group)
        layers_num = int(layers_num_tensor.item())
        print(f"current stage {pipe_rank}, rank {dist.get_rank()}, recv from {prev_rank}, num layers {layers_num}")
        first_layer_idx = layers_num - 2 # exclude embedding layers
    
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    if 1 <= len(model.keys()) <= 2 :
        layers_num = cal_layers_num(lm, layer_re)
    # elif len(model.keys()) == 2:
    #     layers_num = cal_layers_num(lm, layer_re)
        # layers_num += 1
    else:
        raise ValueError("The model not megatron standard checkpoint")

    if pipe_rank != pipe_size - 1:
        print(f"current stage {pipe_rank}, rank {dist.get_rank()}, send to {next_rank}, tensor {layers_num_tensor}")
        dist.send(layers_num_tensor + layers_num, next_rank, group=pipe_group)
    total_layers_num = layers_num_tensor.item() + layers_num

    # get numbers of layer list
    layers_num_tensor_var = layers_num_tensor_var+total_layers_num
    layers_num_tensor_var_list = [torch.zeros_like(layers_num_tensor_var) for _ in range(pipe_size)]
    layers_num_tensor_var_list[pipe_rank] = layers_num_tensor_var
    torch.distributed.all_gather(layers_num_tensor_var_list, layers_num_tensor_var, group=pipe_group)
    
    dist.barrier()

    # set Merak module number
    module_num_tensor = torch.Tensor([0]).to(device)
    if pipe_rank == 0:
        module_num = 0
        module_num_tensor = module_num_tensor + layers_num
        dist.send(module_num_tensor, next_rank, group=pipe_group)
    else:
        dist.recv(module_num_tensor, prev_rank, group=pipe_group)
        module_num = int(module_num_tensor.item())
        if pipe_rank != pipe_size - 1:
            dist.send(module_num_tensor + layers_num, next_rank, group=pipe_group)
    del module_num_tensor

    # print(total_layers_num, pipe_rank, model_rank, "number of layers")
    # print(module_num, pipe_rank, model_rank, "first index of modules")

    # The embeddings.
    # Normally, only first model has embedding.
    n_positions_tensor = torch.zeros(1).to(device)
    if "embedding" in lm.keys():
        embeddings = lm["embedding"]

        # The word embeddings.
        word_embeddings = embeddings["word_embeddings"]["weight"]
        # Truncate the embedding table to vocab_size rows.
        word_embeddings = word_embeddings[: config.vocab_size, :]
        output_state_dict[f"module.{module_num}.transformer.wte.weight"] = word_embeddings
        module_num += 1

        # _tensor_constant0
        # constant_tensor = torch.ones((1, 1024), dtype=torch.float32)
        # output_state_dict[f"module.{module_num}._tensor_constant0"] = constant_tensor

        # The position embeddings.
        pos_embeddings = embeddings["position_embeddings"]["weight"]
        # Read the causal mask dimension (seqlen). [max_sequence_length, hidden_size]
        n_positions = pos_embeddings.size(0)
        if n_positions != config.n_positions:
            raise ValueError(
                f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
            )

        # Copies n_positions from src to all other processes.
        if pipe_size > 1:
            dist.broadcast(n_positions_tensor+n_positions, src=global_rank_of_stage_0)

        # Store the position embeddings.
        output_state_dict[f"module.{module_num}.transformer.wpe.weight"] = pos_embeddings
        module_num += 1
    else:
        dist.broadcast(n_positions_tensor, src=global_rank_of_stage_0)
        n_positions = int(n_positions_tensor.item())
    del n_positions_tensor

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_merak = {
        "attention.dense": ".self_attention.c_proj.row_linear.",
        "self_attention.dense": ".self_attention.c_proj.row_linear.",
        "mlp.dense_h_to_4h": ".mlp.c_fc.col_linear.",
        "mlp.dense_4h_to_h": ".mlp.c_proj.row_linear.",
    }

    # Record previous layer index
    pre_layer_idx = first_layer_idx
    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1)) + first_layer_idx

        if layer_idx != pre_layer_idx:
            module_num += 1
            pre_layer_idx = layer_idx

        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"transformer.h.{layer_idx}"

        if args.overlap_level == 0:
            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm"):
                ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
                output_state_dict[f"module.{module_num}" + "." +layer_name + "." + ln_name + "." + weight_or_bias] = val

            # Transpose the QKV matrix.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "weight":
                # Insert a tensor of 1x1xDxD bias.
                causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float32)).view(
                    1, 1, n_positions, n_positions
                )
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".attn.bias"] = causal_mask

                # Insert a "dummy" tensor for masked_bias.
                masked_bias = torch.tensor(-1e4, dtype=torch.float32)
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".attn.masked_bias"] = masked_bias

                out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
                # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
                out_val = out_val #.transpose(0, 1).contiguous()
                # Store.
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".attn.c_attn.col_linear.weight"] = out_val

            # Transpose the bias.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "bias":
                out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
                # Store. No change of shape.
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".attn.c_attn.col_linear.bias"] = out_val

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_merak[op_name]
                output_state_dict[f"module.{module_num}" + "." + layer_name + out_name + "weight"] = val #.transpose(0, 1)

            # Copy the bias.
            elif weight_or_bias == "bias":
                out_name = megatron_to_merak[op_name]
                output_state_dict[f"module.{module_num}" + "." + layer_name + out_name + "bias"] = val
        elif args.overlap_level == 3:
            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm"):
                ln_name = op_name
                output_state_dict[f"module.{module_num}" + "." +layer_name + "." + ln_name + "." + weight_or_bias] = val

            # Transpose the QKV matrix.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "weight":
                # Insert a tensor of 1x1xDxD bias.
                causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float32)).view(
                    1, 1, n_positions, n_positions
                )
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".self_attention.bias"] = causal_mask

                # Insert a "dummy" tensor for masked_bias.
                masked_bias = torch.tensor(-1e4, dtype=torch.float32)
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".self_attention.masked_bias"] = masked_bias

                out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
                # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
                out_val = out_val #.transpose(0, 1).contiguous()
                # Store.
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".self_attention.c_attn.col_linear.weight"] = out_val

            # Transpose the bias.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "bias":
                out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
                # Store. No change of shape.
                output_state_dict[f"module.{module_num}" + "." + layer_name + ".self_attention.c_attn.col_linear.bias"] = out_val

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_merak[op_name]
                output_state_dict[f"module.{module_num}" + "." + layer_name + out_name + "weight"] = val #.transpose(0, 1)

            # Copy the bias.
            elif weight_or_bias == "bias":
                out_name = megatron_to_merak[op_name]
                output_state_dict[f"module.{module_num}" + "." + layer_name + out_name + "bias"] = val
    module_num += 1

    # DEBUG.
    # assert config.n_layer < layer_idx + 1

    # The final layernorm.
    if "final_layernorm.weight" in transformer.keys():
        output_state_dict[f"module.{module_num}.transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
        output_state_dict[f"module.{module_num}.transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    if "word_embeddings_for_head" in model.keys():
        output_state_dict[f"module.{module_num}.lm_head.weight"] = model["word_embeddings_for_head"]["weight"]

    # print(module_num, pipe_rank, model_rank, "end index of modules")
    dist.barrier()

    # It should be done!
    return output_state_dict, layers_num_tensor_var_list


####################################################################################################

def get_megatron_checkpoint_name(checkpoints_path, parallel_args,
                        release=False, complete=False):
    """A unified checkpoint name."""
    pipe_rank, pipe_size, pipe_group, model_rank, model_size, prev_rank, next_rank, global_rank_of_stage_0 = parallel_args

    iteration = 0
    with open(checkpoints_path + "/latest_checkpointed_iteration.txt", 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                sys.exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(checkpoints_path + "latest_checkpointed_iteration.txt")

    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # Use both the tensor and pipeline MP rank.
    if pipe_size == 1:
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}'.format(
                                model_rank),
                            'model_optim_rng.pt')
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}_{:03d}'.format(
                            model_rank,
                            pipe_rank),
                        'model_optim_rng.pt')

def check_checkpoint_path(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

def get_merak_checkpoint_name(checkpoints_path, parallel_args,
                        release=False, complete=False):
    """A unified checkpoint name."""
    pipe_rank, pipe_size, pipe_group, model_rank, model_size, prev_rank, next_rank, global_rank_of_stage_0 = parallel_args

    iteration = 0
    release = True
    if not os.path.exists(checkpoints_path) and dist.get_rank() == 0:
        os.makedirs(checkpoints_path)
        with open(checkpoints_path + "/latest_checkpointed_iteration.txt", 'w') as f:
            f.write("release")  
    dist.barrier()

    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    # Use both the tensor and pipeline MP rank.
    if model_size == 1 and pipe_size == 1:
        check_checkpoint_path(os.path.join(checkpoints_path, 'iter_{:07d}_all_dp'.format(iteration)))
        return os.path.join(checkpoints_path,
                            'iter_{:07d}_all_dp'.format(
                                iteration,
                            ),
                            'model_optim.pt')

    elif pipe_size == 1 and model_size != 1:
        check_checkpoint_path(os.path.join(checkpoints_path, directory,
                              'mp_rank_{:02d}'.format(model_rank)))
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}'.format(
                                model_rank),
                            'partial_model_optim.pt')

    else:
        check_checkpoint_path(os.path.join(checkpoints_path, directory,
                                'mp_rank_{:02d}_pp_rank_{:03d}'.format(
                                model_rank,
                                pipe_rank)))
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}_pp_rank_{:03d}'.format(
                                model_rank,
                                pipe_rank),
                            'partial_model_optim.pt')

####################################################################################################

def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file" # (.zip archive , directory or direct .pt file)",
    )
    parser.add_argument(
        "--path_to_output_checkpoint",
        type=str,
        help="Path to the output checkpoint file" # (.zip archive , directory or direct .pt file)",
    )
    parser.add_argument('--dp', default=-1, type=int, 
                         help='dimension of data parallelism', required=True)
    parser.add_argument('--tp', default=-1, type=int, 
                         help='dimension of tensor model parallelism', required=True)
    parser.add_argument('--pp', default=-1, type=int, 
                         help='dimension of pipeline model parallelism', required=True)
    parser.add_argument('--overlap_level', default=0, type=int, choices=[1, 3],
                         help='set Merak overlap level')
    parser.add_argument('--local_rank', default=-1, type=int, 
                         help='local rank')
    parser.add_argument('--distributed_backend', default='nccl',
                       choices=['nccl', 'mpi'])
    args = parser.parse_args()

    # init Merak or Megatron
    dp = int(args.dp)
    tp = int(args.tp) 
    pp = int(args.pp)
    if USE_MERAK:
        Merak.init(pp, tp, dp, backend=args.distributed_backend)
    elif USE_MEGATRON:
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=dp*tp*pp, rank=int(os.getenv('RANK', '0')))
        mpu.initialize_model_parallel(tensor_model_parallel_size_=tp, pipeline_model_parallel_size_=pp)
    assert dp * tp * pp >= 1, "Please set the correct dimension of parallelism"

    if pp > 1 and USE_MERAK:
        init_p2p()

    parallel_args = init_model_parallel()
    pipe_rank, pipe_size, pipe_group, model_rank, model_size, prev_rank, next_rank, global_rank_of_stage_0 = parallel_args

    
    #     print(f"current stage is {pipe_rank}, rank {dist.get_rank()}, next {next_rank}, prev {prev_rank}")
    # os._exit(0)

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint)

    # create output directory
    if not os.path.exists(args.path_to_output_checkpoint) and dist.get_rank() == 0:
        os.makedirs(args.path_to_output_checkpoint)
    dist.barrier()

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    if dist.get_rank() == 0:
        print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    elif os.path.isdir(args.path_to_checkpoint):
        checkpoint_name = get_megatron_checkpoint_name(args.path_to_checkpoint, parallel_args)
        input_state_dict = torch.load(checkpoint_name , map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    ds_args = input_state_dict.get("args", None)

    # Read the config, or default to the model released by NVIDIA.
    if args.config_file == "":
        if ds_args is not None:
            if ds_args.bias_gelu_fusion:
                activation_function = "gelu_fast"
            elif ds_args.openai_gelu:
                activation_function = "gelu_new"
            else:
                activation_function = "gelu"
        else:
            # in the very early days this used to be "gelu_new"
            activation_function = "gelu_new"

        # Spell out all parameters in case the defaults change.
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
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
            bos_token_id=50256,
            eos_token_id=50256,
        )
    else:
        config = GPT2Config.from_json_file(args.config_file)

    config.architectures = ["GPT2LMHeadModel"]

    # Convert.
    if dist.get_rank() == 0:
        print("Converting")
    output_state_dict, layers_num_tensor_var_list = convert_megatron_checkpoint(args, input_state_dict, config, parallel_args)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict, args.path_to_output_checkpoint, parallel_args)

    # # Add tokenizer class info to config
    # # see https://github.com/huggingface/transformers/issues/13906)
    # if ds_args is not None:
    #     tokenizer_type = ds_args.tokenizer_type
    #     if tokenizer_type == "GPT2BPETokenizer":
    #         tokenizer_model_name = "gpt2"
    #     elif tokenizer_type == "PretrainedFromHF":
    #         tokenizer_model_name = ds_args.tokenizer_name_or_path
    #     else:
    #         raise ValueError(f"Unrecognized tokenizer_type {tokenizer_type}")
    # else:
    #     tokenizer_model_name = "gpt2"

    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    # tokenizer_class = type(tokenizer).__name__
    # config.tokenizer_class = tokenizer_class

    # Store the config to file.
    if dist.get_rank() == 0:
        print("Saving config")
        config.save_pretrained(basename)

    # # Save tokenizer based on args
    # print(f"Adding {tokenizer_class} tokenizer files")
    # tokenizer.save_pretrained(basename)

    # Store the state_dict to file.
    merak_checkpoint_name = get_merak_checkpoint_name(args.path_to_output_checkpoint, parallel_args)
    # output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{merak_checkpoint_name}"')
    torch.save(output_state_dict, merak_checkpoint_name)

    if dist.get_rank() == 0:
        layer_list = [0]
        for i in range(pipe_size):
            layer_list.append(layers_num_tensor_var_list[i].int().item())
        print("megatron partition model layer index is ",layer_list)
        print("If you want to load this model in Merak, please set args","'",f"--partition_method custom --custom_partition {','.join(str(e) for e in layer_list)}","'")
    dist.barrier()


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################