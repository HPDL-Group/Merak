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

from typing import Any, Dict, Tuple, Union

import torch

from Merak import get_logger

from .amp_optimizer import HalfPrecisionOptimizer

try:
    import apex
except ImportError:
    pass


class MixedPrecisionConfig:
    """
    Mixin for mixed precision training configurations.
    Handles the setup and configuration of models and optimizers for different
    precision modes (FP16, FP32, etc.).
    """

    def __init__(self, args: Any):
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
        self.logger = get_logger("simple")

    def configure(
        self,
        optimizer: torch.optim.Optimizer,
        module: torch.nn.Module,
        device: torch.device,
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
        self.module = module
        base_optimizer = None

        if optimizer is not None:
            base_optimizer = self.optimizer(self.module)
            self.optimizer = base_optimizer
            if not self.args.zero_stage == 1:
                self._configure_optimizer()

        return self.optimizer, base_optimizer

    def _configure_optimizer(self):
        """
        Configure the optimizer for mixed precision training.
        """
        self.logger.info(
            f"Basic Optimizer: {self.optimizer.__class__.__name__}", ranks=[0]
        )

        if self.args.half_precision_backend == "apex":
            self._configure_apex_amp()
        elif self.args.fp16 or self.args.bf16:
            self.optimizer = self.configure_fp16_or_bf16_optimizer(self.optimizer)
        else:
            pass
            # raise ValueError(
            #     "Please set --fp16 or --bf16 or --half_precision_backend")

    def _configure_apex_amp(self):
        """
        Configure the model and optimizer using APEX's automatic mixed precision.
        """
        amp_params = {
            "enabled": True,
            "opt_level": self.args.fp16_opt_level,
        }

        self.logger.info(
            f"Initializing APEX AMP with parameters: {amp_params}", ranks=[0]
        )

        self.module, self.optimizer = apex.amp.initialize(
            self.module, self.optimizer, **amp_params
        )

        self.logger.info(
            f"Successfully initialized APEX AMP from: {apex.amp.__path__}", ranks=[0]
        )

        self.optimizer.zero_grad()

    def configure_fp16_or_bf16_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """
        Configure the optimizer for FP16 training.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to configure.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        dynamic_loss_args = self.dynamic_loss_args()
        self.logger.info(
            "Creating FP16 unfused optimizer with dynamic loss scale", ranks=[0]
        )

        configured_optimizer = HalfPrecisionOptimizer(
            optimizer,
            self.args,
            static_loss_scale=self.args.loss_scale if not self.args.bf16 else 1,
            dynamic_loss_scale=True if not self.args.bf16 else False,
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
            "INITIAL_LOSS_SCALE": 2**self.args.initial_scale_power,
            "SCALE_WINDOW": self.args.loss_scale_window,
            "DELAYED_SHIFT": 2,
            "MIN_LOSS_SCALE": self.args.min_loss_scale,
        }
