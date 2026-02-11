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

import enum
import random

import numpy as np
import torch
from config import load_config
from transformers import GPT2Config, GPT2LMHeadModel, HfArgumentParser, set_seed
from utils import create_tokenizer

# using our distributed trainer
import Merak
from Merak import MerakArguments
from Merak.inference import text_generation_pipeline


def parse_option(parser):
    # easy config modification
    parser.add_argument("--cache-dir", type=str, help="where to save cache")
    parser.add_argument("--model-name", type=str, help="gpt2")

    return parser


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


def main():
    # init dist
    pp = 4
    tp = 1
    dp = 1
    Merak.init(pp, tp, dp)

    torch.cuda.set_device("cuda:0")

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwarg = load_config(args.model_name)
    config = GPT2Config(**config_kwarg)

    # create tokenizer
    tokenizer = create_tokenizer(args.cache_dir, "IDEA-CCNL/Wenzhong-GPT2-110M", config)

    # create model
    model = GPT2LMHeadModel(config)
    tokenizer.eod = model.config.eos_token_id

    # Initialize our Trainer)
    pipeline = text_generation_pipeline(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    # Training
    pipeline.generate_samples_interactive()


if __name__ == "__main__":
    main()
