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

# Test command:
# yhrun -N 1 -n 1 -p 3090 torchrun --nproc-per-node=4 test_merak_trainer.py --output_dir './output' --logging_steps 1 --wall_clock_breakdown true

# using our distributed trainer
import os
import Merak
import torch

from Merak import MerakTrainer, MerakArguments, init_empty_weights

from transformers import (
    default_data_collator,
    set_seed,
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Config,
)

def parse_option(parser):
    # easy config modification
    parser.add_argument('--cache-dir', type=str, help='where to save cache')
    parser.add_argument('--model-name', type=str, help='gpt2')
    return parser


def main():

    # init dist
    pp = 4
    tp = 1
    dp = 1
    Merak.init(pp, tp, dp)

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load data
    config = GPT2Config(n_layer=4,
                        vocab_size=50344,
                        reorder_and_upcast_attn=False, use_cache=False)

    # create model
    with init_empty_weights():
        model = GPT2LMHeadModel(config)

    # Preprocessing the datasets.
    train_dataset = {k: v for k, v in enumerate(range(1000))}
    eval_dataset = {k: v for k, v in enumerate(range(1000))}

    class TestTrainer(MerakTrainer):
        def prepare_data(self, data):
            input_ids = torch.randint(
                0, 1000,
                [
                    training_args.per_device_train_batch_size,
                    training_args.seq_length
                ]
            ).long().cuda()
            labels = torch.randint(
                0, 1000,
                [
                    training_args.per_device_train_batch_size,
                    training_args.seq_length
                ]
            ).long().cuda()
            if pp > 1 or tp > 1:
                return (input_ids, labels)
            else:
                return {'input_ids': input_ids, 'labels': labels}
        
        def get_loss_fn(self):
            criterion = torch.nn.CrossEntropyLoss()
            def loss_fn(outputs, labels):
                if pp == 1 and tp == 1:
                    outputs = outputs['logits']
                    labels = labels['labels']
                elif pp > 1 or tp > 1:
                    if isinstance(outputs, dict):
                        outputs = outputs['logits']
                    else:
                        outputs = outputs[0] \
                            if isinstance(outputs, tuple) else outputs
                    labels = labels[0] \
                        if isinstance(labels, tuple) else labels
                loss = criterion(outputs.view(-1, outputs.size(-1)), 
                                 labels.view(-1))
                return loss
            return loss_fn

    # Initialize our Trainer)
    trainer = TestTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Training
    trainer.train()


if __name__ == "__main__":
    main()
