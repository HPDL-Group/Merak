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

import os

import torchvision
from config import get_config
from data import build_loader
from transformers import HfArgumentParser

import Merak
from Merak import MerakArguments, MerakTrainer


def parse_option(parser):
    group = parser.add_argument_group(
        "Torchvision model training and evaluation script"
    )
    group.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    group.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="path to data folder",
    )

    return parser


def main(config):
    dataset_train, dataset_val, _, _, _ = build_loader(config)

    model = torchvision.models.resnet152()

    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
    )
    trainer.train()


if __name__ == "__main__":
    pp = 2
    tp = 1
    dp = 2
    Merak.init(pp, tp, dp)

    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # using data config from swin transformer
    config = get_config(args)
    main(config)
