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
# yhrun -p 3090 -N 1 -n 1 torchrun --nproc-per-node=4 test_annote_nodes.py --output_dir ./

import time
import Merak

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    HfArgumentParser,
)

from Merak import MerakArguments
from Merak.merak_args import mergeargs, manual_set_args
from Merak.core.fx.tracer import symbolic_trace
from Merak.core.fx.graph_shard import _annote_nodes, oan

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
    # traced, dummy_inputs = tf_symbolic_trace(
    #                         model,
    #                         input_names = training_args.input_names)

    traced, dummy_inputs = symbolic_trace(
    model,
    input_names = 'input_ids',
    batch_size = 1,
    sequence_length = 1024)
    
    # a experience users number threshold, a node has more user than this 
    # threshold indicate the node is needed in multiple stages and could 
    # be transmitted between stages
    output_node_threshold = 5
    output_nodes_count = {}
    for node in traced.graph.nodes:
        if len(list(node.users)) > output_node_threshold:
            output_nodes_count[node.name] = len(list(node.users))

    # annote nodes
    start_time = time.time()
    node_name_to_shard_id, shard_output, func_inputs, extra_output = \
        _annote_nodes(traced, training_args.shard_count, output_nodes_count)
    end_time = time.time()
    time_consume_0 = end_time - start_time
    
    start_time = time.time()
    _node_name_to_shard_id, _shard_output, _func_inputs, _extra_output = \
    oan(traced, training_args.shard_count, output_nodes_count)
    end_time = time.time()
    time_consume_1 = end_time - start_time

    correctness = True
    if node_name_to_shard_id != _node_name_to_shard_id:
        correctness = False
        print('==node_name_to_shard_id != _node_name_to_shard_id==')
    if shard_output != _shard_output:
        correctness = False
        print('==shard_output != _shard_output==')
    if func_inputs != _func_inputs:
        correctness = False
        print('==func_inputs != _func_inputs==')
    if extra_output != _extra_output:
        correctness = False
        print('==extra_output != _extra_output')

    speed_up = ((time_consume_0 - time_consume_1)/time_consume_0) * 100

    print('==correctness==', correctness, '==speed_up==', speed_up)

        
    # print('==node_name_to_shard_id==', node_name_to_shard_id)
    # print('==shard_output==', shard_output)
    # print('==func_inputs==', func_inputs)
    # print('==extra_output==', extra_output)


if __name__ == "__main__":
    main()