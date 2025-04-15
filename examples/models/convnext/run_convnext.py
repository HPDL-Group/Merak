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

import Merak
import torch

from Merak import MerakArguments, MerakTrainer, init_empty_weights
from config import load_config
from transformers import (
    set_seed,
    HfArgumentParser,
    ConvNextForImageClassification,
    ConvNextConfig,
)
from Merak.utils.datasets import DynamicGenDataset


# Add custom command-line arguments
def parse_option(parser):
    parser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. gpt2)')
    return parser


def main():
    # Initialize Merak distributed training environment
    # pp: pipeline parallelism, tp: tensor parallelism, dp: data parallelism
    pp = 2
    tp = 2
    dp = 1
    Merak.init(pp, tp, dp)

    if tp > 1:
        from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists

        col_para_list = ['query', 'key', 'value']
        row_para_list = ['attention.dense']
        tp_attr_list = ['num_attention_heads', 'all_head_size']

        # manully set tp attribute for swin model
        set_tp_layer_lists(col_para_list=col_para_list, row_para_list=row_para_list, 
                           tp_attr_list=tp_attr_list)


    # Parse training and model arguments
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Load model configuration from custom config file
    config_kwarg = load_config(args.model_name)
    config = ConvNextConfig(**config_kwarg)

    # Initialize model
    with init_empty_weights():
        model = ConvNextForImageClassification(config)

    # Create a fake dataset for training
    train_dataset = DynamicGenDataset(
        model.config, mode="vision_only", dataset_size=1e6
    )

    # Initialize trainer with model, training arguments and dataset
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()