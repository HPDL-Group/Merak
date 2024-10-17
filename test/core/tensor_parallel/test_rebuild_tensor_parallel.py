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
# yhrun -N 1 -n 1 -p 3090 torchrun --nproc-per-node=4 test_rebuild_tensor_parallel.py --output_dir ./output --parallel_vocab true --sequence_parallel true

import torch
import Merak

from transformers import (
    HfArgumentParser,
    GPT2LMHeadModel,
    GPT2Config,
)

from Merak import get_topo, get_grid, MerakArguments
from Merak.core.fx import convert_to_sequential
from Merak.core.pipeline import PipelineModule
from Merak.core.recompute import checkpoint as checkpoint_func
from Merak.core.tensor_parallel import (ModuleRebuild, Conv1DProxy, 
                                        LinearProxy, EmbeddingProxy)
from Merak.merak_args import mergeargs, manual_set_args

def _get_emb_dim(model):
    emb_dim = set()
    def add_dim(emb_dim, m):
        if isinstance(m, (EmbeddingProxy, LinearProxy)):
            emb_dim.add(m.weight_shape)
        else:
            emb_dim.add(m.weight.shape)
    if model.config.tie_word_embeddings:
        for m in model.modules():
            try:
                if hasattr(m, 'get_input_embeddings') and \
                    m.get_input_embeddings() is not None:
                    add_dim(emb_dim, m.get_input_embeddings())
                if hasattr(m, 'get_output_embeddings') and \
                    m.get_output_embeddings() is not None:
                    add_dim(emb_dim, m.get_output_embeddings())
            except AttributeError:
                continue
    elif hasattr(model, 'get_input_embeddings'):
        add_dim(emb_dim, model.get_input_embeddings())
    elif hasattr(model, 'get_output_embeddings'):
        add_dim(emb_dim, model.get_output_embeddings())

    return emb_dim

def main():
    # init dist
    pp = 1
    tp = 4
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

    # init tensor parallel builder
    rebuild_module = ModuleRebuild(training_args, model)

    # get embedding dim
    emb_dim = _get_emb_dim(model)

    # fx trace and split module 
    _model, layers, input_to_shard = convert_to_sequential(model, training_args)
    del _model

    # build pipe module
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
    
    # rebuild tensor parallel module 
    module = rebuild_module.recover_module(pipe_model)
    module = rebuild_module.vocab_parallel(pipe_model,
                                           emb_dim=emb_dim)

    print('==rebuild_tensor_parallel_model==', module)


if __name__ == "__main__":
    main()