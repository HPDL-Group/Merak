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

# Test command:
# yhrun -p 3090 -N 1 -n 1 torchrun --nproc-per-node=4 test_symbolic_trace.py --output_dir ./

import Merak

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    HfArgumentParser,
)

from Merak import MerakArguments
from Merak.merak_args import mergeargs, manual_set_args
from Merak.core.fx.tracer import symbolic_trace

def main():
    # init dist
    pp = 4
    tp = 1
    dp = 1
    Merak.init(pp, tp, dp)

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    training_args = hfparser.parse_args_into_dataclasses()[0]

    config = GPT2Config()

    # create model
    model = GPT2LMHeadModel(config)

    # set args
    mergeargs(training_args, model.config)
    manual_set_args(training_args)

    # trace model 
    traced, dummy_inputs = symbolic_trace(
                            model,
                            input_names = training_args.input_names,
                            batch_size = training_args.per_device_train_batch_size,
                            sequence_length = training_args.seq_length,
                            )

    print(['==dummy_inputs==', dummy_inputs])
    print(['==traced==', traced])


if __name__ == "__main__":
    main()
