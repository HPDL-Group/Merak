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
import torch.distributed as dist
from torch.distributed.distributed_c10d import get_global_rank
from typing import Union, Tuple, Dict

from .fp16_optimizer import FP16_Optimizer
from .. import mpu
from .. import zero
from ..printer import logger, log_dist
from ...merak_args import MerakArguments

try:
    import apex
    from apex import amp
except ImportError:
    pass

class MixedPrecisionConfig:

    def __init__(self, args: MerakArguments):
        self.args = args
        self.module = None
        self.optimizer = None
        self.device = None
        self.enable_backward_allreduce = True

        self.global_rank = dist.get_rank()
        self.data_parallel_group = mpu.get_data_parallel_group()
        self.broadcast_src_rank = get_global_rank(
            mpu.get_data_parallel_group(), 0
        )

    def set_half(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.args.fp16 or self.args.half_precision_backend == "apex":
            module = module.half()
        return module

    def configure(
            self,
            optimizer: torch.optim.Optimizer,
            module: torch.nn.Module,
            device: torch.device
        ) -> Tuple[Union[torch.nn.Module, torch.optim.Optimizer]]:
        self.optimizer = optimizer
        self.device = device
        if self.args.fp16:
            self._configure_fp16_model(module)
        else:
            self.module = module.float().to(self.device)
        if optimizer is not None:
            self._configure_optimizer()
        return self.module, self.optimizer


    def _configure_optimizer(self):
        if self.global_rank == 0:
            logger.info("Basic Optimizer = {}".format(
                self.optimizer.__class__.__name__))

        if self.args.half_precision_backend == "apex":
            amp_params = {
                "enabled": True,
                "opt_level": self.args.fp16_opt_level,
            }
            if self.global_rank == 0:
                logger.info(f"Initializing AMP with these params: {amp_params}")
            try:
                logger.info(f"Initializing Apex amp from: {amp.__path__}")
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError(
                  "Unable to import apex/amp, please make sure it is installed")
            self.module, self.optimizer = amp.initialize(
                self.module, self.optimizer, **amp_params
            )
            self.optimizer.zero_grad()
        elif self.args.fp16:
            self.optimizer = self.configure_fp16_optimizer(self.optimizer) 
        else:
            raise ValueError("Please set --fp16 or --half_precision_backend")

        self.quantizer = None

    def _configure_fp16_model(self, model: torch.nn.Module):
        self.module = model
        self.module.half()
        self.module.to(self.device)

    def configure_fp16_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        dynamic_loss_args = self.dynamic_loss_args()
        log_dist("Creating fp16 unfused optimizer with dynamic loss scale",
                    ranks=[0])
        optimizer = FP16_Optimizer(
            optimizer,
            deepspeed=self,
            static_loss_scale=self.args.loss_scale,
            dynamic_loss_scale=True,
            dynamic_loss_args=dynamic_loss_args,
            mpu=mpu,
            clip_grad=self.args.max_grad_norm,
            fused_lamb_legacy=optimizer.__class__.__name__ == "lamb",
        )

        return optimizer

    def dynamic_loss_args(self) -> Dict[str, int]:
        loss_scale_args = {
                'INITIAL_LOSS_SCALE': 2 ** self.args.initial_scale_power,
                'SCALE_WINDOW': self.args.loss_scale_window,
                'DELAYED_SHIFT': 2,
                'MIN_LOSS_SCALE': self.args.min_loss_scale,
            }
        return loss_scale_args
