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
    AltCLIPModel,
    AltCLIPConfig,
)
from Merak.utils.datasets import DynamicGenDataset


def parse_option(parser):
    # easy config modification
    parser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. bert)')
    return parser


def main():
    # init dist
    pp = 4
    tp = 1
    dp = 1
    Merak.init(pp, tp, dp)


    if tp > 1:
        from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists

        col_para_list = ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'intermediate.dense', 'fc1']
        row_para_list = ['output.dense', 'out_proj', 'fc2']
        tp_attr_list = ['num_heads']

        # manully set tp attribute for swin model
        set_tp_layer_lists(col_para_list=col_para_list, row_para_list=row_para_list, 
                           tp_attr_list=tp_attr_list)
    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = AltCLIPConfig(**config_kwarg)


    with init_empty_weights():
        model = AltCLIPModel(config)

    model.config._attn_implementation = 'eager'

    # Create a fake dataset for training
    train_dataset = DynamicGenDataset(
        model.config, mode="multimodal", dataset_size=1e6
    )


    # define custom loss function
    def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
        def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(
                logits, torch.arange(len(logits), device=logits.device)
            )
        caption_loss = contrastive_loss(similarity)
        image_loss = contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    class MyTrainer(MerakTrainer):  

        def get_loss_fn(self):
            def loss_fn(outputs, labels):
                loss = clip_loss(outputs[1])
                return loss
            return loss_fn


    # using our distributed trainer        
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
    )

    # Training
    trainer.train()

if __name__ == "__main__":
    main()
