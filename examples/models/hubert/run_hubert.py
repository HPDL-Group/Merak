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

from Merak import MerakArguments, MerakTrainer, init_empty_weights
from config import load_config
from typing import Union

from transformers import (
    set_seed,
    HubertForCTC,
    HubertConfig,
    HfArgumentParser,
)
from Merak.utils.datasets import DynamicGenDataset


def _get_feat_extract_output_lengths(config, input_lengths: Union[torch.LongTensor, int]):
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    return input_lengths

# Add custom command-line arguments
def parse_option(parser):
    parser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. gpt2)')
    return parser


def main():
    # Initialize Merak distributed training environment
    # pp: pipeline parallelism, tp: tensor parallelism, dp: data parallelism
    pp = 4
    tp = 1
    dp = 1
    Merak.init(pp, tp, dp)
    
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = HubertConfig(**config_kwarg)

    with init_empty_weights():
        model = HubertForCTC(config)

    # Create a fake dataset for training
    eval_datasets = DynamicGenDataset(
        model.config, mode="speech", dataset_size=1e6, seq_length=training_args.seq_length
    )

    #eval for tracing
    model.eval()

    class HubertTrainer(MerakTrainer):

        def get_loss_fn(self):
            def loss_fn(outputs, labels):
                if isinstance(labels, tuple):
                    labels = labels[0]
                if labels.max() >= config.vocab_size:
                        raise ValueError(f"Label values must be <= vocab_size: {config.vocab_size}")

                # retrieve loss input_lengths from attention_mask
                attention_mask = None
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones([training_args.per_device_eval_batch_size, self.args.seq_length], dtype=torch.long)
                )
                input_lengths = _get_feat_extract_output_lengths(
                    config, attention_mask.sum(-1)
                ).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = torch.nn.functional.log_softmax(
                    outputs, dim=-1, dtype=torch.float32
                ).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = torch.nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=config.pad_token_id,
                        reduction=config.ctc_loss_reduction,
                        zero_infinity=config.ctc_zero_infinity,
                    )
                return loss
            return loss_fn

    # using our distributed trainer        
    trainer = HubertTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_datasets,
    )

    # Training
    trainer.evaluation()


if __name__ == "__main__":
    main()
