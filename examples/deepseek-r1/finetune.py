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

import torch
import Merak
import os
import pandas as pd

from Merak import MerakArguments, MerakTrainer, print_rank_0, init_empty_weights
from datasets import Dataset, load_dataset

from transformers.optimization import AdamW
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    HfArgumentParser,
    DataCollatorForSeq2Seq
)

# Add custom command-line arguments
def parse_option(parser):
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--model_path', type=str, help='path to model')
    return parser


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def process_data(data: dict, tokenizer, max_seq_length):
    """
    Tokenizes and processes a single data sample into input_ids, attention_mask, and labels.
    """
    conversation = data["conversation"]
    input_ids, attention_mask, labels = [], [], []

    for i, conv in enumerate(conversation):
        human_text = conv["input"].strip()
        assistant_text = conv["output"].strip()

        input_text = "input:" + human_text + "\n\noutput:"

        input_tokenizer = tokenizer(
            input_text,
            add_special_tokens=False,
            truncation=True,
            padding=False, 
            return_tensors=None,
        )
        output_tokenizer = tokenizer(
            assistant_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token ID!")
        
        input_ids += (
            input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
        )
        attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
        labels += (
            [-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
        )

    # Padding
    seq_length = len(input_ids)
    if seq_length < max_seq_length:
        pad_length = max_seq_length - seq_length
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        if tokenizer.padding_side == "right":  
            input_ids.extend([pad_token_id] * pad_length)
            attention_mask.extend([1] * pad_length)
            labels.extend([pad_token_id] * pad_length)
        else: 
            input_ids = [pad_token_id] * pad_length + input_ids
            attention_mask = [1] * pad_length + attention_mask
            labels = [pad_token_id] * pad_length + labels

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # Initialize distributed parallelism
    pp = int(os.environ['PP'])  # Pipeline parallelism
    tp = int(os.environ['TP'])  # Tensor parallelism
    dp = int(os.environ['DP'])  # Data parallelism
    
    Merak.init(pp, tp, dp)

    # Merge HuggingFace and custom arguments
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    model_name = args.model_path
    train_file = args.data_path

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load and preprocess dataset
    data = pd.read_json(train_file)
    train_ds = Dataset.from_pandas(data)

    train_dataset = train_ds.map(process_data,
                            fn_kwargs={"tokenizer": tokenizer, "max_seq_length": training_args.seq_length},
                            remove_columns=train_ds.column_names)

    # Create data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # Load model config
    print("\nLoading configuration...")
    config_path = model_name + "/config.json"
    config = AutoConfig.from_pretrained(config_path)
    # config._attn_implementation = "eager"  # Optional: specify attention backend

    # Load model weights (empty init for sharded loading)
    print("\nLoading model...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.enable_input_require_grads()


    class DeepSeekR1Trainer(MerakTrainer):
        """
        Custom Trainer that defines optimizer creation logic.
        """

        def create_optimizer(self, module):
            def get_parameter_names(model, forbidden_layer_types):
                """
                Returns parameter names not inside specified layer types (e.g., LayerNorm).
                """
                result = []
                for name, child in model.named_children():
                    result += [
                        f"{name}.{n}"
                        for n in get_parameter_names(child, forbidden_layer_types)
                        if not isinstance(child, tuple(forbidden_layer_types))
                    ]
                # Include parameters directly on the model (not part of submodules)
                result += list(model._parameters.keys())
                return result

            decay_parameters = get_parameter_names(module, [torch.nn.LayerNorm])
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in module.named_parameters() 
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in module.named_parameters() 
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            adam_kwargs = {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=training_args.learning_rate, **adam_kwargs
            )
            return optimizer

    # Initialize trainers
    trainer = DeepSeekR1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        leaf_modules=(LlamaAttention,),
        data_collator=data_collator
    )
        
    trainer.train()


if __name__ == "__main__":
    main()