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
import torch
import Merak

from Merak import MerakArguments, MerakTrainer, print_rank_0, init_empty_weights
from Merak.core import mpu
from config import load_config

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import (
    default_data_collator,
    set_seed,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaConfig
)
from Merak.utils.datasets import DynamicGenDataset


# Add custom command-line arguments
def parse_option(parser):
    parser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. gpt2)')
    return parser


def main():
    # Initialize Merak distributed training environment
    # pp: pipeline parallelism, tp: tensor parallelism, dp: data parallelism
    pp = 4
    tp = 2
    dp = 1
    Merak.init(pp, tp, dp)
    torch.backends.cuda.matmul.allow_tf32 = True

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if tp > 1:
        from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists

        col_para_list = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                         'gate_proj', 'up_proj']
        row_para_list = ['self_attn.o_proj', 'down_proj']
        tp_attr_list=['num_heads', 'num_key_value_heads']  
        # manully set tp attribute
        set_tp_layer_lists(col_para_list=col_para_list, row_para_list=row_para_list, tp_attr_list=tp_attr_list)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = LlamaConfig(
            **config_kwarg
        )
    config._attn_implementation="eager"

    # meta init model
    with init_empty_weights():
        model = LlamaForCausalLM(config)
    model.generation_config.pad_token_id=-100

    # Create a fake dataset for training
    train_dataset = DynamicGenDataset(
        model.config, mode="text_only", dataset_size=1e6
    )
        
    # Initialize our Trainer
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        leaf_modules=(LlamaAttention,)
    )
        
    trainer.train()

if __name__ == "__main__":
    main()
