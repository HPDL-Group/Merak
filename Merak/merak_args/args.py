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

import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Any, Union, List, Optional
import torch
from transformers import TrainingArguments, PretrainedConfig
from transformers.file_utils import (
    cached_property,
    torch_only_method,
)


_GLOBAL_ARGS = None

@dataclass
class MerakArguments(TrainingArguments):
    """
    MerakArguments inherits from transformers.TrainingArguments (https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer) 
    extending the arguments we need in 3D parallelism.
    Using [`HfArgumentParser`] we can turn this class into [argparse](https://docs.python.org/3/library/argparse#module-argparse) 
    arguments that can be specified on the command line.
    Parameters:
    -   train_schedule (str, Optional,  defaults to '1f1b') -- Some possible choices are the pipe schedules as strings: '1f1b', 'ds_default','last_no_recompute_1f1b', 'shifted_critical_path'.
    -   partition_method (str, Optional, defaults to 'uniform') -- Possible choices are the pipeline layer partion strategy as strings: 
        'uniform','uniform_floor','parameters'.
    -   split_method(str, Optional, defaults to 'farthest_min_deps') -- Possible choices are graph partion method as strings: 'farthest_min_deps','layer_split','nearest_min_deps'.
    -   custom_split_points(List(str), defaults to None) -- Create split points for layer_split method, default is None.
    -   trace_method(str, defaults to 'fx') -- None refers to no tracing, 'fx' refers to use torch.fx for tracing, 'dynamo' refers to use torch._dynamo for tracing.
    -   trace_model(str, Optional, defaults to '') -- Add new trace module. example: --trace_model 'Qwen2ForCausalLM'.
    -   init_method_std (float, defaults to 0.02) -- Standard deviation of the zero mean normal distribution used for tp weight initialization in Megatron
    -   activation_checkpointing (bool, defaults to True) -- Whether to use activation checkpointing. 
    -   checkpoint_num_layers (int, defaults to 1) -- Chunk size (number of layers) for checkpointing.
    -   input_names (List[str], Optional, defaults to None) -- The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead. 
        Example: ['input_ids', 'attention_mask', 'token_type_ids']
    -   num_layers (int, Optional, defaults to None) -- Number of hidden layers in the Transformer, will try to get this in model config.
    -   seq_length (int, Optional, defaults to None) -- The maximum sequence length that this model might ever be used with, will try to get this in model config.
    -   num_heads (int, Optional, defaults to None) -- The number of heads that this model might ever be used with, will try to get this in model config. Defaults to None.
    -   wall_clock_breakdown (bool, defaults to False) -- Whether to log detail time spend on each rank. 
    -   shard_count (int, Optional, defaults to None) -- Number of shards that model needs to be break, will be training_args.num_layers*2 if not set.
    -   prescale_gradients (bool, defaults to False) -- Whether to enable gradient prescaling.
    -   gradient_predivide_factor (float, defaults to 1.0) -- Gradient predivide factor in gradient prescaling.
    -   zero_allow_untested_optimizer (bool, defaults to False) -- Whether to allow wrap untested optimizer. The untested optimizer does not guarantee the correctness of training.
    -   zero_stage (float, defaults to 1) -- Stage of zero optimization.
    -   zero_allgather_bucket_size (float, defaults to 500000000) -- The bucket size per communication in optimzier step.
    -   zero_reduce_bucket_size (float, defaults to 500000000) -- The bucket size per communication in gradients reduce.
    -   save (bool, defaults to False) -- Whether to save checkpoint.
    -   finetune (bool, defaults to False) -- Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.
    -   no_save_rng (bool, defaults to False) -- Do not save current rng state.
    -   no_save_optim (bool, defaults to False) -- Do not save current optimizer.
    -   no_load_rng (bool, defaults to False) -- Do not load current optimizer.
    -   no_load_optim (bool, defaults to False) -- Do not load current optimizer.
    -   split_inputs (bool, defaults to False) -- Whether to split input data.
    -   parallel_vocab (bool, defaults to False) -- Whether to parallel vocabulary when TMP > 1.
    -   sequence_parallel (bool, defaults to False) -- Whether to use sequence parallel when TMP > 1.
    -   sequence_dim (int, defaults to 1) -- Sequence length dimension in hidden states.
    -   dealloc_pipeoutput (bool, defaults to False) -- Whether to dealloc pipeline sended activation output.
    -   activation_checkpoint_ratio (float, Optional, defaults to None) -- activation checkpoint ratio of first stage, in range(0,1). Default to None.
    -   tp_overlapping_level (float, Optional, defaults to 0) -- "Possible tensor parallelism communication overlapping level from 0 to 3."
        "0 refers to no overlapping; 1 refers to only overlap within linear function;"
        "2 refers to overlap within transformer blocks, requires rewrite transformer blocks;"
        "3 refers to overlap between transformer blocks, requires rewrite transformer model.",
        "choices": [0,1,2,3]
    -   loss_scale (float, defaults to 0) -- 'loss_scale is a fp16 parameter representing the loss scaling value for FP16 training.'
        'The default value of 0.0 results in dynamic loss scaling, '
        'otherwise the value will be used for static fixed loss scaling.'
    -   initial_scale_power (int, defaults to 32) -- 'initial_scale_power is a fp16 parameter representing the power of the initial dynamic loss scale value.'
        'The actual loss scale is computed as 2^initial_scale_power.'
    -   loss_scale_window (int, defaults to 1000) -- 'loss_scale_window is a fp16 parameter representing the window over which to raise/lower the dynamic loss scale value.'
    -   hysteresis (int, defaults to 2) -- 'hysteresis is a fp16 parameter representing the delay shift in dynamic loss scaling.'
    -   min_loss_scale (int, defaults to 1) -- 'min_loss_scale is a fp16 parameter representing the minimum dynamic loss scale value.'
    -   custom_partition (float, defaults to None) -- 'Customize the partition size of the model. Length of list is pipeline_world_size + 1.
        'Example: [0, 6, 12, 18, 26, ..., last_layer_idx]', Default to None.
    -   no_tie_modules (bool, defaults to False) -- Whether to set tie modules.
    -   save_total_limit (int, defaults to -1) -- Limit the max numbers of checkpoints.
    -   eval_iters (int, defaults to None) -- Number of iterations to run for evaluationvalidation/test for.
    -   text_generation (bool, Optional, defaults to False) -- Whether to do text generate.
    -   out_seq_length (int, Optional, defaults to 1024) -- The maximum sequence length that this model's output. Defaults to 1024.
    -   temperature (float, Optional, defaults to 0.9) -- Sampling temperature.
    -   lora_config (str, Optional, defaults to None) -- Set lora config path.
    -   adapter_name (str, Optional, defaults to default) -- The name of the adapter to be injected, if not provided, the default adapter name is used ('default').
    """

    train_schedule: str = field(
        default="1f1b",
        metadata={
            "help": "Possible choices are the pipe schedules as strings:"
                    "`1f1b`, `ds_default`, `pre_recompute_1f1b`, "
                    "`ds_default`,  `last_no_recompute_1f1b`, "
                    "`full_critical_path_1f1b, shifted_critical_path`,"
                    " Defaults to `1f1b`.",
        },
    )
    partition_method: str = field(
        default="uniform_floor",
        metadata={
            "help": "Possible choices are the pipeline layer partion strategy "
                    "as strings: 'uniform','uniform_floor', 'parameters'."
                    "Defaults to 'uniform'.",
        },
    )
    split_method: str = field(
        default="min_deps",
        metadata={
            "help": "Possible choices are graph partion method "
                    "as strings: 'farthest_min_deps','layer_split',"
                    "'nearest_min_deps'.",
        },
    )
    custom_split_points: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Create split points for layer_split method, "
                    "default is None",
        },
    )
    trace_method: str = field(
        default='fx',
        metadata={
            "help": "None refers to no tracing;"
                    "'fx' refers to use torch.fx for tracing;"
                    "'dynamo' refers to use torch._dynamo for tracing."
        },
    )
    trace_model: Optional[str] = field(
        default='',
        metadata={
            "help": "Add new trace module. example: --trace_model 'Qwen2ForCausalLM'"
        },
    )
    init_method_std: float = field(
        default=0.02, 
        metadata={
            "help": "Standard deviation of the zero mean normal distribution"
                    "used for TP weight initialization in Megatron."
                    "Defaults to 0.02."
        },
    )
    activation_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Whether to use activation checkpointing."
                    "Defaults to True."
        },
    )
    checkpoint_num_layers: int = field(
        default=1, 
        metadata={
            "help": "chunk size (number of layers) for checkpointing."
                    "0 means disable activation checkpoint."
                    "Defaults to 1"
        },
    )
    input_names: Optional[List[str]] = field(
        default=None, 
        metadata={
            "help": "The names of the inputs of the traced model."
                    "If unset, model.dummy_inputs().keys() are used instead."
                    "Example: ['input_ids', 'attention_mask', 'token_type_ids']"
        },
    )
    num_layers: Optional[int] = field(
        default=None, 
        metadata={
            "help": "Number of hidden layers in the Transformer,"
                    "will try to get or eval this in model config."
                    "Defaults to None."
        },
    )
    seq_length: Optional[int] = field(
        default=None, 
        metadata={
            "help": "The maximum sequence length that this model might ever be "
                    "used with, will try to get this in model config."
                    "Defaults to None."
        },
    )
    hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "The hidden size that this model might ever be "
                    "used with, will try to get this in model config."
                    "Defaults to None."
        },
    )
    num_heads: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of heads that this model might ever be "
                    "used with, will try to get this in model config."
                    "Defaults to None."
        },
    )
    wall_clock_breakdown: bool = field(
        default=True,
        metadata={
            "help": "Whether to log detail time spend on each rank."
                    "Defaults to False"
        },
    )
    shard_count: Optional[int] = field(
        default=None, 
        metadata={
            "help": "Number of shards that model needs to be break."
                    "It will be training_args.num_layers*2 if not set."
                    "Defaults to None."
        },
    )
    prescale_gradients: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable gradient prescaling."
                    "Defaults to False"
        },
    )
    gradient_predivide_factor: float = field(
        default=1.0, 
        metadata={
            "help": "Gradient predivide factor in gradient prescaling."
                    "Defaults to 1"
        },
    )
    cache_sharding: bool = field(
        default=False,
        metadata={
            "help": "Whether to cache the partitioned graphs of model with"
                    "microbatch size."
                    "Defaults to False"
        },
    )
    cache_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Set the cache name of partitioned graphs,"
                    "must be setted when cache_sharding is True."
                    "Defaults to None"
        },
    )
    return_logits: bool = field(
        default=False,
        metadata={
            "help": "Whether to return logits and labels in evaluation."
                    "Defaults to False"
        },
    )
    # zero arguments
    zero_allow_untested_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow zero to wrap untested optimizer."
            "Defaults to False"
        },
    )
    zero_stage: float = field(
        default=None,
        metadata={
            "help": "Set the stage of zero optimizer."
                    "Only support zero stage 1 currently."
                    "Defaults to None."
        },
    )
    zero_allgather_bucket_size: float = field(
        default=500000000,
        metadata={
            "help": "Set the all gather partition size for zero stage 1"
                    "optimization."
                    "Defaults to 500000000."
        },
    )
    zero_reduce_bucket_size: float = field(
        default=500000000,
        metadata={
            "help": "Set the reduce bucket size(max_elems_per_comm) for zero "
                    "stage 1 optimization."
                    "Defaults to 500000000."
        },
    )
    # checkpoint arguments
    save: bool = field(
        default=False,
        metadata={
            "help": "Whether to save checkpoints."
                    "Defaults to False."
        },
    )
    finetune: bool = field(
        default=False,
        metadata={
            "help": "Load model for finetuning. Do not load optimizer "
                    "or rng state from checkpoint and set iteration to 0."
                    "Defaults to False."
        },
    )
    no_save_rng: bool = field(
        default=False,
        metadata={
            "help": "Do not save current rng state"
                    "Defaults to False."
        },
    )
    no_save_optim: bool = field(
        default=False,
        metadata={
            "help": "Do not save current optimizer"
                    "Defaults to False."
        },
    )
    no_load_optim: bool = field(
        default=False,
        metadata={
            "help": "Do not load optimizer when loading checkpoint."
                    "Defaults to False."
        },
    )
    no_load_rng: bool = field(
        default=False,
        metadata={
            "help": "Do not load rng state when loading checkpoint."
                    "Defaults to False."
        },
    )

    # split input
    split_inputs: bool = field(
        default=False,
        metadata={
            "help": "Whether to split input data"
                    "Defaults to False."
        },
    )
    parallel_vocab: bool = field(
        default=False,
        metadata={
            "help": "Whether to parallel vocabulary when TMP > 1"
                    "Defaults to False."
        },
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sequence parallel when TMP > 1"
                    "Defaults to False."
        },
    )
    sequence_dim: int = field(
        default=1,
        metadata={
            "help": "Sequence length dimension in hidden states"
                    "Defaults to False."
        },
    )
    dealloc_pipeoutput: bool = field(
        default=False,
        metadata={
            "help": "Whether to dealloc pipeline sended activation output"
                    "Defaults to False."
        },
    )
    activation_checkpoint_ratio: Optional[List[str]] = field(
        default=None, 
        metadata={
            "help": "activation checkpoint ratio of first stage,"
                    "in range(0,1) for each pipeline stage."
                    "Default to None."
        },
    )
    tp_overlapping_level: int = field(
        default=0,
        metadata={
            "help": "Possible tensor parallelism communication overlapping "
                    "level from 0 to 3. 0 refers to no overlapping;"
                    "1 refers to only overlap within linear function;"
                    "2 refers to overlap within transformer blocks, requires "
                    "rewrite transformer blocks; 3 refers to overlap between "
                    "transformer blocks, requires rewrite transformer model."
                    "Default to 0.",
            "choices": [0,1,2,3],
        },
    )

    # amp
    half_precision_backend: str = field(
        default=None,
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["cuda_amp", "apex", "cpu_amp"],
        },
    )
    loss_scale: float = field(
        default=0.,
        metadata={
            "help": "loss_scale is a fp16 parameter representing the loss "
                    "scaling value for FP16 training. The default value of 0.0 "
                    "results in dynamic loss scaling, otherwise the value will "
                    "be used for static fixed loss scaling."
                    "Default to 0.",
        },
    )
    initial_scale_power: int = field(
        default=32,
        metadata={
            "help": "initial_scale_power is a fp16 parameter representing the "
                    "power of the initial dynamic loss scale value.The actual "
                    "loss scale is computed as 2^initial_scale_power."
                    "Default to 32.",
        },
    )
    loss_scale_window: int = field(
        default=1000,
        metadata={
            "help": "loss_scale_window is a fp16 parameter representing the "
                    "window over which to raise/lower the dynamic loss scale "
                    "value."
                    "Default to 1000.",
        },
    )
    hysteresis: int = field(
        default=2,
        metadata={
            "help": "hysteresis is a fp16 parameter representing the delay "
                    "shift in dynamic loss scaling."
                    "Default to 2.",
        },
    )
    min_loss_scale: int = field(
        default=1,
        metadata={
            "help": "min_loss_scale is a fp16 parameter representing the "
                    "minimum dynamic loss scale value."
                    "Default to 1.",
        },
    )

    custom_partition: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Customize the partition size of the model. Length of list "
                    "is pipeline_world_size + 1."
                    "Example: [0, 6, 12, 18, 26, ..., last_layer_idx]"
                    "Default to None.",
        },
    )
    no_tie_modules: bool = field(
        default=False,
        metadata={
            "help": "Whether to set tie modules."
            "Default to False."
        },
    )
    save_total_limit: int = field(
        default=-1,
        metadata={
            "help": 'Limit the max numbers of checkpoints.'
            "Default to -1.",
        },
    )
    eval_iters: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of iterations to run for evaluation/validation/test"
                    "Default to -1.",
        },
    )

    # text generation
    text_generation: bool = field(
        default=False,
        metadata={
            "help": "Whether to do text generate"
                    "Default to False.",
        },
    )
    out_seq_length: int = field(
        default=1024,
        metadata={
            "help": "The maximum sequence length that this model's output."
                    "Defaults to 1024."
        },
    )
    temperature: float = field(
        default=0.9,
        metadata={
            "help": "Sampling temperature"
                    "Defaults to 0.9."
        },
    )

    # peft
    lora_config: str = field(
        default=None,
        metadata={
            "help": "Set lora config path"
                    "Defaults to None."
        },
    )
    adapter_name: str = field(
        default="default",
        metadata={
            "help": "The name of the adapter to be injected, if not provided, "
                    "the default adapter name is used ('default')."
                    "Defaults to 'default'."
        },
    )
    def get_lora_config(self):
        with open(self.lora_config, "r") as f:
            lora_kwargs = json.load(f)
        if "auto_mapping" in lora_kwargs.keys():
            lora_kwargs.pop("auto_mapping")
        return lora_kwargs


    @cached_property
    @torch_only_method
    def _setup_devices(self) -> "torch.device":
        
        if self.use_cpu:
            device = torch.device('cpu')
        else:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            assert self.local_rank != -1, \
                'only support distributed training/evaluation for now'
            device = torch.device("cuda", self.local_rank)

            if device.type == "cuda":
                torch.cuda.set_device(device)

        return device

def manual_set_args(args: MerakArguments):
    global _GLOBAL_ARGS
    # some args for megatron
    if hasattr(args,'fp16') and args.fp16:
        args.params_dtype = torch.half
    else:
        args.params_dtype = torch.float32
    args.use_cpu_initialization = True if args.use_cpu else False
    _GLOBAL_ARGS = args


def get_args() -> MerakArguments:
    """Return arguments."""
    assert _GLOBAL_ARGS is not None, '{} is not initialized.'.format('args')
    return _GLOBAL_ARGS


args_dict = {
    "seq_length": ['seq_length', 'max_position_embeddings', 'n_positions', 'embed_dim',
                   'max_target_positions', 'num_conv_pos_embeddings'],
    "num_heads": ['num_attention_heads', 'n_head', 'num_heads'],
    "hidden_size": ['hidden_size', 'dim', 'n_embd', 'd_model', 'hidden_sizes'],
    "num_layers": ['num_hidden_layers', 'n_layers', 'num_layers'],
}


def mergeargs(training_args: MerakArguments, model_config: Union[PretrainedConfig, dict]):
    training_args.DDP_impl = 'local'
    if not training_args.input_names:
        training_args.input_names = None

    if hasattr(model_config, 'vision_config'):
        training_args.num_layers = model_config.text_config.num_hidden_layers + \
                                        model_config.vision_config.num_hidden_layers
        training_args.num_heads = [model_config.text_config.num_attention_heads,
                                   model_config.vision_config.num_attention_heads]
    if hasattr(model_config, 'decoder_layers'):
        if hasattr(model_config, 'encoder_layers'):
            training_args.num_layers = model_config.decoder_layers + \
                                        model_config.encoder_layers
            training_args.num_heads = [model_config.decoder_attention_heads,
                                   model_config.encoder_attention_heads]
        else:
            training_args.num_layers = model_config.decoder_layers
            training_args.num_heads = model_config.decoder_attention_heads

        if hasattr(model_config, 'num_conv_layers'):
            training_args.num_layers += model_config.num_conv_layers

    if hasattr(model_config, 'depths'):
        training_args.num_layers = sum(model_config.depths)
    if hasattr(model_config, 'true_hidden_size'):
        training_args.hidden_size = model_config.true_hidden_size

    for n, name_list in args_dict.items():
        if getattr(training_args, n) is None:
            if hasattr(model_config, 'text_config'):
                model_config = model_config.text_config
            for name in name_list:
                if hasattr(model_config, name):
                    values = getattr(model_config, name)
                    setattr(training_args, n, values)
                    break
    if not isinstance(training_args.num_heads, list):
        training_args.num_heads = [training_args.num_heads]

    # assert training_args.num_layers is not None, 'num_layers should be set.'
    # assert training_args.hidden_size is not None, 'hidden_size should be set.'

    if training_args.shard_count is None:
        if isinstance(training_args.num_layers, dict):
            training_args.shard_count = sum(list(training_args.num_layers.values())) * 2 + 4
        else:
            training_args.shard_count = training_args.num_layers*2 + 4

    if training_args.train_schedule == 'shifted_critical_path':
        training_args.train_schedule = 'full_critical_path_1f1b'

    if training_args.activation_checkpointing == False:
        training_args.checkpoint_num_layers = 0

    if torch.distributed.get_rank()==0 and training_args.wall_clock_breakdown:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(training_args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots,
                                                getattr(training_args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)