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

from transformers.models.swin.modeling_swin import SwinDropPath
from transformers import (
    set_seed,
    SwinForImageClassification,
    SwinConfig,
    HfArgumentParser,
)
from Merak.utils.datasets import DynamicGenDataset

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
    
    # merege args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    if tp > 1:
        from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists

        col_para_list = ['query', 'key', 'value', 'intermediate.dense']
        row_para_list = ['output.dense']
        weight_change_list = [('relative_position_bias_table', 1)]
        tp_attr_list = ['num_attention_heads']

        # manully set tp attribute for swin model
        set_tp_layer_lists(col_para_list=col_para_list, row_para_list=row_para_list, 
            weight_change_list=weight_change_list, tp_attr_list=tp_attr_list)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set model config
    config = SwinConfig()
    config.return_dict = False
    config.use_cache = False
    config.num_attention_heads = [4, 6, 12, 24]

    # meta init
    with init_empty_weights():
        model = SwinForImageClassification(config)

    # Create a fake dataset for training
    train_dataset = DynamicGenDataset(
        model.config, mode="vision_only", dataset_size=1e6
    )


    class MyTrainer(MerakTrainer):
        # define custom loss function
        def get_loss_fn(self):
            def loss_fn(outputs, labels):
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(labels, tuple):
                    labels = labels[0]
                if model.config.problem_type is None:
                    if model.config.num_labels == 1:
                        model.config.problem_type = "regression"
                    elif model.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        model.config.problem_type = "single_label_classification"
                    else:
                        model.config.problem_type = "multi_label_classification"
                if model.config.problem_type == "regression":
                    loss_fct = torch.nn.MSELoss()
                    if model.config.num_labels == 1:
                        loss = loss_fct(outputs.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(outputs, labels)
                elif model.config.problem_type == "single_label_classification":
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(outputs.view(-1, model.config.num_labels), labels.view(-1))
                elif model.config.problem_type == "multi_label_classification":
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(outputs, labels) 
                return loss
            return loss_fn


    # using our distributed trainer        
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, 
        leaf_modules=(SwinDropPath,)
    )

    # Training
    trainer.train()


if __name__ == "__main__":
    main()
