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
from typing import Union, Tuple, Dict, Optional
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
    """
   Mixin for mixed precision training configurations.
    Handles the setup and configuration of models and optimizers for different
    precision modes (FP16, FP32, etc.).
    """

    def __init__(self, args: MerakArguments):
        """
        Initialize the mixed precision configuration.

        Args:
            args (MerakArguments): Training arguments.
        """
        self.args = args
        self.module = None
        self.optimizer = None
        self.device = None
        self.enable_backward_allreduce = True
        self.global_rank = dist.get_rank()
        self.data_parallel_group = mpu.get_data_parallel_group()
        self.broadcast_src_rank = dist.get_global_rank(self.data_parallel_group, 0)

    def configure(
        self,
        optimizer: torch.optim.Optimizer,
        module: torch.nn.Module,
        device: torch.device
    ) -> Tuple[Union[torch.nn.Module, torch.optim.Optimizer]]:
        """
        Configure the model and optimizer for mixed precision training.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to configure.
            module (torch.nn.Module): The model to configure.
            device (torch.device): The device to move the model to.

        Returns:
            Tuple: A tuple containing the configured model and optimizer.
        """
        self.optimizer = optimizer
        self.device = device

        if self.args.fp16:
            self._configure_fp16_model(module)
        elif self.args.bf16 and self.args.half_precision_backend != "cuda_amp":
            self._configure_bf16_model(module)
        else:
            self.module = module.float().to(self.device)

        if optimizer is not None:
            self._configure_optimizer()

        return self.module, self.optimizer

    def _configure_optimizer(self):
        """
        Configure the optimizer for mixed precision training.
        """
        if self.global_rank == 0:
            logger.info(f"Basic Optimizer: {self.optimizer.__class__.__name__}")

        if self.args.half_precision_backend == "apex":
            self._configure_apex_amp()
        elif self.args.fp16:
            self.optimizer = self.configure_fp16_optimizer(self.optimizer)
        elif self.args.bf16:
            pass
        else:
            raise ValueError("Please set --fp16 or --bf16 or --half_precision_backend")

        self.quantizer = None

    def _configure_apex_amp(self):
        """
        Configure the model and optimizer using APEX's automatic mixed precision.
        """
        amp_params = {
            "enabled": True,
            "opt_level": self.args.fp16_opt_level,
        }

        if self.global_rank == 0:
            logger.info(f"Initializing APEX AMP with parameters: {amp_params}")

        try:
            self.module, self.optimizer = amp.initialize(
                self.module, self.optimizer, **amp_params
            )
        except Exception as e:
            raise RuntimeError(f"Unable to initialize APEX AMP: {str(e)}")

        if self.global_rank == 0:
            logger.info(f"Successfully initialized APEX AMP from: {amp.__path__}")

        self.optimizer.zero_grad()

    def _configure_bf16_model(self, model: torch.nn.Module):
        """
        Configure the model for BF16 training.

        Args:
            model (torch.nn.Module): The model to configure.
        """
        self.module = model
        self.module.bfloat16()
        self.module.to(self.device)

    def _configure_fp16_model(self, model: torch.nn.Module):
        """
        Configure the model for FP16 training.

        Args:
            model (torch.nn.Module): The model to configure.
        """
        self.module = model
        self.module.half()
        self.module.to(self.device)

    def configure_fp16_optimizer(
        self,
        optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """
        Configure the optimizer for FP16 training.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to configure.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        dynamic_loss_args = self.dynamic_loss_args()
        log_dist("Creating FP16 unfused optimizer with dynamic loss scale",
                 ranks=[0])

        configured_optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=self.args.loss_scale,
            dynamic_loss_scale=True,
            dynamic_loss_args=dynamic_loss_args,
            clip_grad=self.args.max_grad_norm,
        )

        return configured_optimizer

    def dynamic_loss_args(self) -> Dict[str, int]:
        """
        Generate dynamic loss scale arguments.

        Returns:
            Dict[str, int]: A dictionary containing loss scale parameters.
        """
        return {
            'INITIAL_LOSS_SCALE': 2 ** self.args.initial_scale_power,
            'SCALE_WINDOW': self.args.loss_scale_window,
            'DELAYED_SHIFT': 2,
            'MIN_LOSS_SCALE': self.args.min_loss_scale,
        }