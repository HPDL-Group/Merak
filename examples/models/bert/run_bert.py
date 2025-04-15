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

from Merak import MerakArguments, MerakTrainer, init_empty_weights
from config import load_config
from transformers import (
    set_seed,
    BertForMaskedLM,
    BertForPreTraining,
    BertConfig,
    HfArgumentParser,
)
from Merak.utils.datasets import DynamicGenDataset


def parse_option(parser):
    # easy config modification
    parser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. bert)')
    return parser


def main():
    # init dist
    pp = 2
    tp = 2
    dp = 1
    Merak.init(pp, tp, dp)
    
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = BertConfig(**config_kwarg)


    with init_empty_weights():
        if args.model_name == 'bert-large-uncased':
            model = BertForPreTraining(config)
        elif args.model_name == 'bert-large' or args.model_name == 'bert-base-uncased':
            model = BertForMaskedLM(config)

    # Create a fake dataset for training
    train_dataset = DynamicGenDataset(
        model.config, mode="text_only", dataset_size=1e6
    )


    # using our distributed trainer        
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
    )

    # Training
    trainer.train()

if __name__ == "__main__":
    main()
