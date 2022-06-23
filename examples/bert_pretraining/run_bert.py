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

# using our distributed trainer
import Merak
from Merak import MerakArguments, MerakTrainer
from config import load_config

from transformers import (
    set_seed,
    BertForPreTraining,
    BertConfig,
    HfArgumentParser,
)
from bert_data import WorkerInitObj, HDF5Dataset
import torch

def parse_option(parser):
    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')
    parser.add_argument('--dataset-name', type=str, help='name of dataset from the datasets package')
    parser.add_argument('--model-name', type=str, help='gpt2 or t5-base')
    parser.add_argument('--validation-split-percentage', type=int, default=5, help='split data for validation')

    return parser

def main():
    # init dist
    pp = 2
    tp = 1
    dp = 2
    Merak.init(pp, tp, dp)

    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # create dataset
    train_dataset = HDF5Dataset(args, training_args.seed, max_pred_length=80)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = BertConfig(**config_kwarg)

    model = BertForPreTraining(config)

    class BertTrainer(MerakTrainer):
        def get_train_dataloader(self):
            train_data = self.train_dataset
            train_sampler = self._get_train_sampler()
            worker_init = WorkerInitObj(self.args.seed + self.args.local_rank)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler,
                                        num_workers=self.args.dataloader_num_workers, worker_init_fn=worker_init,
                                        pin_memory=True)
            return train_dataloader
        def get_loss_fn(self, criterion):
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
            def loss_fn(outputs, labels):
                prediction_scores, seq_relationship_score = outputs
                mlm_labels, next_sentence_label = labels
                masked_lm_loss = criterion(prediction_scores.view(-1, prediction_scores.size(-1)), mlm_labels.view(-1))
                next_sentence_loss = criterion(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                return total_loss
            return loss_fn
    # using our distributed trainer
    trainer = BertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()
