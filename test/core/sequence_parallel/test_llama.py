# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: oys
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
# yhrun -N 4 -n 4 -p 3090 torchrun --nproc-per-node=4 test_llama.py --output_dir ./output --parallel_vocab true --sequence_parallel true


import torch
import Merak
from Merak import MerakArguments, MerakTrainer, init_empty_weights
from Merak.utils.datasets import DynamicGenDataset

from transformers import (
    set_seed,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaConfig,
)

def get_config():
    config = {"architectures": ["LlamaForCausalLM"], 
            "bos_token_id": 0, 
            "eos_token_id": 1, 
            "hidden_act": "silu", 
            "hidden_size": 4096, 
            "intermediate_size": 11008, 
            "initializer_range": 0.02, 
            "max_sequence_length": 2048, 
            "model_type": "llama", 
            "num_attention_heads": 32, 
            "num_hidden_layers": 32, 
            "pad_token_id": -1, 
            "rms_norm_eps": 1e-06, 
            "vocab_size": 32000,
            "return_dict": False,
            "use_cache": False,
            }
    return config
            
def main(pp, dp, sp, tp):
    Merak.init(pp, dp, sp, tp)
    
    hfparser = HfArgumentParser(MerakArguments)
    hfparser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. gpt2)')
    training_args, args = hfparser.parse_args_into_dataclasses()
    training_args.sequence_parallel=True if sp > 1 else False
    set_seed(training_args.seed)

    # set model config
    config_kwarg = get_config()
    config = LlamaConfig(
            **config_kwarg
        )
    
    with init_empty_weights():
        model = LlamaForCausalLM(config)
    train_dataset = DynamicGenDataset(
        model.config, mode="text_only", dataset_size=1e6
    )

    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()

if __name__ == '__main__':
    main(pp=8, dp=1, sp=2, tp=1)