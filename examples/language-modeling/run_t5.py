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
from utils import create_tokenizer, Prepare_data
from config import load_config

from transformers import (
    default_data_collator,
    set_seed,
    HfArgumentParser,
    T5ForConditionalGeneration,
    T5Config,
)


def parse_option(parser):
    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')
    parser.add_argument('--dataset-name', type=str, help='name of dataset from the datasets package')
    parser.add_argument('--model-name', type=str, help='t5-base')
    parser.add_argument('--validation-split-percentage', type=int, default=5, help='split data for validation')

    return parser

def main():
    # init dist
    pp = 2
    tp = 1
    dp = 2
    Merak.init(pp, tp, dp)

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwarg = load_config(args.model_name)
    config = T5Config(
            **config_kwarg
        )

    # create tokenizer
    tokenizer = create_tokenizer(args.cache_dir, args.model_name, config)
    
    # create dataset
    dataset = Prepare_data(tokenizer, input_length=512, output_length=512)

    # create model
    model = T5ForConditionalGeneration(config)

    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset, 
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()
