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
from Merak import MerakArguments
from Merak import MerakTrainer

import os
from transformers import (
    HfArgumentParser
)

from config import get_config
from models import build_model
from models.swin_transformer import window_reverse
from timm.models.layers import DropPath
from data import build_loader


def parse_option(parser):
    group = parser.add_argument_group('Swin Transformer training and evaluation script')
    group.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    group.add_argument('--data_path', type=str, default=None, help='path to data folder', )

    return parser


def main(config):
    dataset_train, dataset_val, _, _, _ = build_loader(config)

    model = build_model(config)
    
    leaf = ((window_reverse, DropPath))
    
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train, 
        eval_dataset=dataset_val, 
        leaf_modules=leaf,
    )
    trainer.train()


if __name__ == '__main__':
    pp = 2
    tp = 1
    dp = 2
    Merak.init(pp, tp, dp)

    if tp > 1:
        from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists
        col_para_list = ['qkv', 'fc1']
        row_para_list = ['proj', 'fc2']
        weight_change_list = [('relative_position_bias_table', 1)]
        tp_attr_list = ['num_heads']

        # manully set tp attribute for swin model
        set_tp_layer_lists(col_para_list=col_para_list, row_para_list=row_para_list, 
            weight_change_list=weight_change_list, tp_attr_list=tp_attr_list)

    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    config = get_config(args)

    path = os.path.join("./config.json")
    with open(path, "w") as f:
        f.write(config.dump())

    main(config)
