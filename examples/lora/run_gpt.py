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

import Merak
from Merak import MerakArguments, MerakTrainer, print_rank_0, mpu
from config import load_config
import os

from transformers import (
    default_data_collator,
    set_seed,
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Config,
)
import torch
import collections



def parse_option(parser):
    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')
    parser.add_argument('--model-name', type=str, help='gpt2')
    parser.add_argument('--hidden-size', type=int, default=None, help='set hidden size')
    parser.add_argument('--num-heads', type=int, default=None, help='set number of heads')
    parser.add_argument('--layers', type=int, default=None, help='set number of layers')
    parser.add_argument('--fake-data', default=False, type=bool, help='using fake data')

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
    config = GPT2Config(
            **config_kwarg
        )

    model = GPT2LMHeadModel(config)

    train_dataset, eval_dataset = (None, None)

    class GPT2Trainer(MerakTrainer):
        def create_optimizer(self):
            def get_parameter_names(model, forbidden_layer_types):
                """
                Returns the names of the model parameters that are not inside a forbidden layer.
                """
                result = []
                for name, child in model.named_children():
                    result += [
                        f"{name}.{n}"
                        for n in get_parameter_names(child, forbidden_layer_types)
                        if not isinstance(child, tuple(forbidden_layer_types))
                    ]
                # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
                result += list(model._parameters.keys())
                return result
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])

            # Delete the parameters that no grad
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

        def get_train_dataloader(self):
            return range(training_args.max_steps * training_args.per_device_train_batch_size)
        def _prepare_inputs(self, data):
            input_ids = torch.randint(0, 1000, [training_args.per_device_train_batch_size, training_args.seq_length]).long() #.chunk(mpu.get_model_parallel_world_size(), \
            labels = torch.randint(0, 1000, [training_args.per_device_train_batch_size, training_args.seq_length]).long()
            return (input_ids, labels)


    # Initialize our Trainer
    trainer = GPT2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )



    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()
