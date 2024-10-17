# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Yck(eyichenke@gmail.com)
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

# test command
# yhrun -N 1 -n 1 -p 3090 torchrun --nproc-per-node=4 test_module.py --output_dir ./output

import torch
import Merak

from Merak import get_topo, get_grid, MerakArguments
from Merak.core.pipeline import PipelineModule
from Merak.core.recompute import checkpoint as checkpoint_func
from Merak.core.fx import convert_to_sequential
from Merak.merak_args import mergeargs

from transformers import (
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Config,
)

def main():
    # config_kwarg = load_config("layoutlm-base-uncased")
    Merak.init(4, 1, 1)

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    training_args = hfparser.parse_args_into_dataclasses()[0]

    config = GPT2Config()

    # create model
    model = GPT2LMHeadModel(config).cuda()

    mergeargs(training_args, model.config)

    model, layers, input_to_shard = convert_to_sequential(model, training_args)
    del model

    pipe_model = PipelineModule(
                layers=layers,
                args=training_args,
                loss_fn=torch.nn.CrossEntropyLoss(),
                topology=get_topo(),
                communicaiton_grid=get_grid(), 
                activation_checkpoint_func=checkpoint_func, 
                tie_dims=set(),
                input_to_shard_dic=input_to_shard,
                )
    
    print(pipe_model.partitions())


if __name__ == "__main__":
    main()