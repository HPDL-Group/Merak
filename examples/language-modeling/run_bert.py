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

# using our distributed trainer
import Merak
from Merak import MerakArguments, MerakTrainer, print_rank_0
from utils import create_tokenizer, load_data, preprocessing_datasets
from config import load_config

from transformers import (
    DataCollatorForLanguageModeling,
    set_seed,
    BertForMaskedLM,
    BertConfig,
    HfArgumentParser,
)

import torch

def parse_option(parser):
    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')
    parser.add_argument('--dataset-name', type=str, help='name of dataset from the datasets package')
    parser.add_argument('--model-name', type=str, help='gpt2 or t5-base')
    parser.add_argument('--validation-split-percentage', type=int, default=5, help='split data for validation')

    return parser

def main():
    # init dist
    pp = 2
    tp = 1
    dp = 2
    Merak.init(pp, tp, dp)
    
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # create dataset
    raw_datasets = load_data(args.data_files, args.cache_dir, args.validation_split_percentage)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = BertConfig(**config_kwarg)

    # create tokenizer
    tokenizer = create_tokenizer(args.cache_dir, args.model_name, config)

    model = BertForMaskedLM(config)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    train_dataset, eval_dataset = preprocessing_datasets(model, raw_datasets,
                                                         tokenizer, args.model_name)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
    )

    # using our distributed trainer        
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
    )

    # Training
    trainer.train()


if __name__ == "__main__":
    main()
