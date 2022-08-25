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
from Merak import MerakArguments, MerakTrainer, mpu
from Merak.utils.checkpoint import load_checkpoint, save_checkpoint

import os
from transformers import (
    set_seed,
    BertForPreTraining,
    BertConfig,
    HfArgumentParser,
)

from config import load_config
from bert_data import WorkerInitObj, HDF5Dataset
import torch
from lamb import Lamb
from schedulers import PolyWarmUpScheduler
try:
    from apex import amp
    from apex.optimizers import FusedLAMB
except:
    pass


def parse_option(parser):
    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')
    parser.add_argument('--dataset-name', type=str, help='name of dataset from the datasets package')
    parser.add_argument('--model-name', type=str, help='gpt2 or t5-base')
    parser.add_argument('--validation-split-percentage', type=int, default=5, help='split data for validation')
    parser.add_argument('--max-pred-length', type=int, default=20, help='max sequence length')
    parser.add_argument('--warmup-proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for.')

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
    train_dataset = HDF5Dataset(args, training_args, json_file="./json_dir_2/data_info_{length}_{size}_{rank}.json".format(length=args.max_pred_length, size=mpu.get_data_parallel_world_size(),
                                                                                                             rank=mpu.get_data_parallel_rank()), max_pred_length=args.max_pred_length)

    # set model config
    config_kwarg = load_config(args.model_name)
    config = BertConfig(**config_kwarg)

    model = BertForPreTraining(config)

    class BertTrainer(MerakTrainer):
        def get_train_dataloader(self):
            train_data = self.train_dataset
            train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
            worker_init = WorkerInitObj(self.args.seed + mpu.get_data_parallel_rank())
            train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size = self.args.per_device_train_batch_size,
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

        def create_optimizer(self):
            opt_model = self.model
            param_optimizer = list(opt_model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            self.optimizer = Lamb(optimizer_grouped_parameters, lr=self.args.learning_rate, betas=(.9, .999), adam=False)
            # self.optimizer = FusedLAMB(optimizer_grouped_parameters, lr=self.args.learning_rate)
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer):
            self.lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=self.args.max_steps)
            return self.lr_scheduler

        def load_from_checkpoint(self, resume_from_checkpoint):
            if os.path.exists(resume_from_checkpoint):
                iteration, state_dict = load_checkpoint(self.pipe_model, self.optimizer, self.lr_scheduler, self.args)

                from apex import amp
                keys = list(state_dict['optimizer']['state'].keys())
                # #Override hyperparameters from previous checkpoint
                for key in keys:
                    state_dict['optimizer']['state'][key]['step'] = 0
                for iter, item in enumerate(state_dict['optimizer']['param_groups']):
                    state_dict['optimizer']['param_groups'][iter]['step'] = 0
                    state_dict['optimizer']['param_groups'][iter]['t_total'] = self.args.max_steps
                    state_dict['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                    state_dict['optimizer']['param_groups'][iter]['lr'] = self.args.learning_rate
                self.optimizer.load_state_dict(state_dict['optimizer'])
                if 'master params' in state_dict.keys():
                    for param, saved_param in zip(amp.master_params(self.optimizer), state_dict['master params']):
                        param.data.copy_(saved_param.data)

                if "file_idx" in state_dict.keys():
                    self.train_dataset.file_idx = state_dict["file_idx"]

                del state_dict
                if self.args.max_steps > 0 and iteration > self.args.max_steps:
                    self.state.global_step = 0
                    self.pipe_model.global_steps = 0
                else:
                    self.state.global_step = iteration
                    self.pipe_model.global_steps = iteration
            else:
                raise ValueError("Cannot find checkpoint files")
            return iteration

        def save_to_checkpoint(self):
            kwargs = {"file_idx": self.train_dataset.file_idx}
            if self.args.half_precision_backend == "apex":
                kwargs['master params'] = list(amp.master_params(self.optimizer))
            save_checkpoint(self.state.global_step, self.pipe_model, self.optimizer,
                            self.lr_scheduler, self.args, **kwargs)


    # using our distributed trainer
    trainer = BertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()
