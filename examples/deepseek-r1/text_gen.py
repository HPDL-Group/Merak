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

import enum
import os
import random

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)

import Merak
from Merak import MerakArguments, MerakTrainer, init_empty_weights
from Merak.inference import text_generation_pipeline


def parse_option(parser):
    # Add custom command-line arguments
    parser.add_argument("--cache-dir", type=str, help="Directory for cache storage")
    parser.add_argument("--model-name", type=str, help="Model name (e.g., deepseekr1)")
    parser.add_argument("--model_path", type=str, help="Path to model directory")

    return parser


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


def main():
    # Initialize distributed environment
    pp = int(os.environ["PP"])  # Pipeline parallelism
    tp = int(os.environ["TP"])  # Tensor parallelism
    dp = int(os.environ["DP"])  # Data parallelism
    Merak.init(pp, tp, dp)

    # Merge HuggingFace and custom arguments
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set random seeds for reproducibility
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    # torch.cuda.manual_seed(training_args.seed)

    set_seed(training_args.seed)

    # Load model path
    model_name = args.model_path
    # Example: model_name = "/gf3/models/DeepSeek-R1-Distill-Qwen-32B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # # (Optional) Create data collator for dynamic padding
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # Load model configuration
    print("\nLoading configuration...")
    config_path = model_name + "/config.json"
    config = AutoConfig.from_pretrained(config_path)
    # config._attn_implementation="eager"  # Optional: Force attention implementation

    # Load model with empty weights (for memory-efficient loading)
    print("\nLoading model...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Load tokenizer again (in case needed after model config)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load config again (may be redundant)
    print("\nLoading configuration...")
    config_path = model_name + "/config.json"
    config = AutoConfig.from_pretrained(config_path)

    # Load model again with empty weights (may be redundant)
    print("\nLoading model...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Set tokenizer's end-of-document token ID
    tokenizer.eod = model.config.eos_token_id

    # Initialize text generation pipeline
    pipeline = text_generation_pipeline(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    # Start interactive generation loop
    pipeline.generate_samples_interactive()


if __name__ == "__main__":
    main()
