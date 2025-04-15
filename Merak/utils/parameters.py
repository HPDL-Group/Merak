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

import torch
import math
from ..merak_args import MerakArguments
from ..core import mpu
from ..core.printer import logger

__all__ = ['BaseParams']

class BaseParams:

    def __init__(self, args: MerakArguments):
        self.args: MerakArguments = args

        self.mbs: int = args.per_device_train_batch_size
        self.eval_mbs: int = args.per_device_eval_batch_size
        self.gas: int = args.gradient_accumulation_steps
        self.total_batch_size: int = self.mbs * self.gas * \
            mpu.get_data_parallel_world_size()
        self.eval_total_batch_size: int = self.eval_mbs * \
            mpu.get_data_parallel_world_size()

        self.max_steps: int = args.max_steps
        self.train_epochs: int = math.ceil(args.num_train_epochs)
        self.num_steps_per_epoch: int = None
        self.num_train_samples: int = None
        self.num_eval_samples: int = None

        self.eval_steps: int = args.eval_steps

        self.micro_steps: int = 0
        self.global_steps: int = 0
        self.global_eval_steps: int = 0
        self.step: int = 0
        self.eval_step: int = 0

    def train(self, dataloader: torch.utils.data.DataLoader):
        if dataloader is None:
            return
        total_steps = math.ceil(
                self.train_epochs * len(dataloader) // (self.total_batch_size)
            )
        if self.args.max_steps < 0:
            self.max_steps = total_steps
        elif self.args.max_steps > total_steps:
            self.train_epochs = (
                self.args.max_steps * self.total_batch_size
            ) // len(dataloader)

        self.num_train_samples = self.max_steps * self.total_batch_size
        self.num_steps_per_epoch = self.max_steps // self.train_epochs

    def eval(self, dataloader: torch.utils.data.DataLoader):
        if dataloader is None:
            return
        if self.args.eval_steps is None:
            self.eval_steps = len(dataloader) // self.gas
        self.num_eval_samples = self.eval_steps * self.eval_total_batch_size

    def resume(self, iteration: int) -> int:
        # load checkpoint
        epochs_trained: int = 0
        max_steps = self.max_steps
        num_steps_per_epoch = self.num_steps_per_epoch

        if iteration > 0:
            if self.args.max_steps > 0:
                if iteration < self.args.max_steps:
                    epochs_trained = int(iteration/num_steps_per_epoch)
            else:
                if iteration < max_steps:
                    epochs_trained = int(iteration/num_steps_per_epoch)

            if self.args.max_steps > 0 and iteration > self.args.max_steps:
                self.global_steps = 0
                self.global_steps = 0
            elif iteration > max_steps:
                self.global_steps = 0
            else:
                self.global_steps = iteration

        return epochs_trained

    def should_break(self) -> bool:
        return self.global_steps % self.num_steps_per_epoch == 0

    def logging(self, eval=False):
        if eval:
            samples = self.num_eval_samples
            epochs = 1
            mbs = self.eval_mbs
            gas = self.gas
            total_batch_size = self.eval_total_batch_size
            max_steps = self.eval_steps
            strings = "Evaluation"
        else:
            samples = self.num_train_samples
            epochs = self.train_epochs
            mbs = self.mbs
            gas = self.gas
            total_batch_size = self.total_batch_size
            max_steps = self.max_steps
            strings = "Training"

        if torch.distributed.get_rank() == 0:
            logger.info(f"***** Running {strings} *****")
            logger.info(f"  Num examples = {samples}")
            logger.info(f"  Num Epochs = {epochs}")
            logger.info(f"  Instantaneous micro batch size per device = {mbs}")
            logger.info(f"  Gradient Accumulation steps = {gas}")
            logger.info(f"  Total batch size (w. parallel, distributed & accumulation) "
                        f"= {total_batch_size}")
            logger.info(f"  Total optimization steps = {max_steps}")
