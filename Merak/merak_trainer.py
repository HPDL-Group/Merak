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

import gc
import logging
import math
import os
import socket
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, Adam
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from . import get_logger
from .core import PipelineEngine, PipelineModule, mpu
from .core.checkpoint import (
    CheckpointLoader,
    CheckpointSaver,
    rotate_checkpoints,
    RcharaCheckpointSaver,
)
from .core.fx.tracer.utils import _generate_dummy_input
from .core.zero import configure_zero_optimizer
from .merak_args import MerakArguments, manual_set_args, mergeargs
from .utils import BaseParams, MegatronPretrainingRandomSampler, WorkerInitObj

from process_monitor.procguard.dist_monitor import get_monitor
from .CheckpointSync.optim import OptimizerWrapper
from .CheckpointSync.recover import Recover

class MerakTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: MerakArguments,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        leaf_modules: Tuple[nn.Module] = (),
    ):
        """
        MerakTrainer is a simple training and eval loop for PyTorch,
        optimized for Merak.

        Args:
        -   model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
                The model to train, evaluate or use for predictions.
                If not provided, a `model_init` must be passed.
        -   args ([`TrainingArguments` and `MerakArguments`], *optional*):
                The arguments to tweak for training. Will default to a basic
                instance of [`TrainingArguments` and `MerakArguments`] with the
                `output_dir` set to a directory named *tmp_trainer* in
                the current directory if not provided.
        -   train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.
                IterableDataset`, *optional*):
                The dataset to use for training. If it is an `datasets.
                Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed.

                Note that if it's a `torch.utils.data.IterableDataset` with
                some randomization and you are training in
                a distributed fashion, your iterable dataset should either use
                a internal attribute `generator` that
                is a `torch.Generator` for the randomization that must be
                identical on all processes (and the Trainer
                will manually set the seed of this `generator` at each epoch)
                or have a `set_epoch()` method that
                internally sets the seed of the RNGs used.
        -   eval_dataset (`torch.utils.data.Dataset`, *optional*):
                The dataset to use for evaluation. If it is an `datasets.
                Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed.
        -   optimizer (`torch.optim.optimizer`, *optional*):
        -   data_collator (`Callable`, *optional*):
                The function to use to form a batch from a list of elements of
                `train_dataset` or `eval_dataset`.
        -   leaf_modules ([`nn.Module`], *optional*):
                List of nn.Module can not be trace, add to leaf modules

        """

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = data_collator
        self.leaf_modules = leaf_modules

        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.engine = None
        self.input_to_stage_dic = None
        self.dummy_inputs = args.input_names
        self._already_load = False

        self.logger = get_logger("simple")
        if hasattr(self.model, "config"):
            self.model_config = self.model.config
            mergeargs(self.args, self.model_config)
        else:
            mergeargs(self.args, None)
        manual_set_args(self.args)

        self.peft_config = None

        self.tuning_params = BaseParams(self.args)

        if dist.get_rank() == 0:
            self.summary_writer = self.get_summary_writer()
        else:
            self.summary_writer = None

        # init pipeline module
        self.loss_fn = (
            self.get_loss_fn()
            if not self.args.parallel_vocab
            else self.get_vocab_loss_fn()
        )
        self.create_pipeline_module()
        self._configure_checkpoints()

        # init training parameters
        self.create_dataloader()
        self.tuning_params.train(self.train_dataset)
        if self.eval_dataset is not None:
            self.tuning_params.eval(self.eval_dataset)
        self.get_dummy_inputs()

    def _configure_checkpoints(self):
        if not self.args.ada_checkpoint:
            self._save_checkpoint = CheckpointSaver().save_checkpoint
        else:
            self._save_checkpoint = RcharaCheckpointSaver().save_checkpoint

        self._rotate_checkpoints = rotate_checkpoints
        self._load_checkpoint = CheckpointLoader().load_checkpoint

    def create_pipeline_module(self):
        if self.args.trace_method in ["fx", "dynamo"]:
            self.model = PipelineModule(
                model=self.model,
                args=self.args,
                loss_fn=self.loss_fn,
                leaf_modules=self.leaf_modules,
                dummy_inputs=self.dummy_inputs,
            )

            self.input_to_stage_dic = self.model.input_to_stage_dic
        else:
            dummy_input = _generate_dummy_input(self.args, self.model)
            self.input_to_stage_dic = {0: list(dummy_input.keys())}
            self.args.no_tie_modules = True
            setattr(self.model, "input_to_stage_dic", self.input_to_stage_dic)
            setattr(self.model, "_grid", self.args.get_communication_grid())
            assert (
                mpu.get_pipe_parallel_world_size() * mpu.get_model_parallel_world_size()
                == 1
            )
        dist.barrier()

        if mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0:
            if hasattr(self.model, "stage_id"):
                for i in range(mpu.get_pipe_parallel_world_size()):
                    if i == self.model.stage_id:
                        self.logger.debug(
                            [dist.get_rank(), self.model.stage_id, self.model]
                        )
            else:
                self.logger.debug([dist.get_rank(), self.model])

    def get_train_sampler(self) -> MegatronPretrainingRandomSampler:
        return MegatronPretrainingRandomSampler(
            total_samples=len(self.train_dataset),
            # Set random seed according to be consumed examples,
            # but currently not supported
            consumed_samples=0,
            micro_batch_size=self.args.per_device_train_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )

    def create_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None and self.eval_dataset is None:
            raise ValueError(
                "MerakTrainer: training requires a train_dataset/eval_dataset."
            )
        if self.train_dataset is not None:
            train_dataset = self.train_dataset

            worker_init = WorkerInitObj(self.args.seed + mpu.get_data_parallel_rank())
            train_sampler = self.get_train_sampler()

            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=self.collate_fn,
                worker_init_fn=(
                    worker_init if self.args.dataloader_num_workers > 0 else None
                ),
            )
            if mpu.get_pipe_parallel_rank() not in self.input_to_stage_dic.keys():
                self.train_dataloader = range(len(self.train_dataloader))

        if self.eval_dataset is not None:

            eval_dataset = self.eval_dataset

            self.eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=self.collate_fn,
            )
            if mpu.get_pipe_parallel_rank() not in self.input_to_stage_dic.keys():
                self.eval_dataloader = range(len(self.eval_dataloader))

    def create_optimizer(self, model: PipelineModule) -> torch.optim.Optimizer:

        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.pytorch_tp:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for p in model.parameters()
                        if isinstance(p, DTensor) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for p in model.parameters()
                        if not isinstance(p, DTensor) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler:

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.get_warmup_steps(self.tuning_params.max_steps),
            num_training_steps=self.tuning_params.max_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )
        return self.lr_scheduler

    def get_dummy_inputs(self):
        if self.dummy_inputs is None:
            self.iter_dataloader = iter(
                self.train_dataloader
                if self.train_dataloader is not None
                else self.eval_dataloader
            )
            one_batch = next(self.iter_dataloader)
            try:
                if not isinstance(one_batch[-1][0], torch.Tensor):
                    one_batch = one_batch[:-2]
                else:
                    one_batch.pop(-1)
            except:
                pass
            self.dummy_inputs = one_batch
            del self.iter_dataloader, self.train_dataloader
            gc.collect()
            self.create_dataloader()

    def get_loss_fn(self) -> Callable:
        criterion = nn.CrossEntropyLoss()

        def loss_fn(
            outputs: Union[torch.Tensor, tuple], labels: Union[torch.Tensor, tuple]
        ) -> torch.Tensor:
            if not isinstance(self.model, PipelineModule):
                if isinstance(outputs, dict):
                    outputs = outputs["logits"]
                else:
                    outputs = outputs[1]
                labels = labels["labels"]
            else:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(labels, tuple):
                    labels = labels[0]

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            return loss

        return loss_fn

    def get_vocab_loss_fn(self) -> Callable:
        def vocab_loss_fn(
            outputs: Union[torch.Tensor, tuple], labels: Union[torch.Tensor, tuple]
        ) -> torch.Tensor:
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(labels, tuple):
                labels = labels[0]
            losses = mpu.vocab_parallel_cross_entropy(outputs, labels)
            # loss mask??
            loss = torch.mean(losses.view(-1))
            return loss

        return vocab_loss_fn

    def prepare_data(self, data: Union[tuple, list, dict]) -> Union[tuple, list]:
        if not isinstance(data, (tuple, list)):
            if hasattr(data, "data"):
                data = data.data
            if isinstance(data, dict):
                if (
                    mpu.get_pipe_parallel_world_size() == 1
                    and mpu.get_model_parallel_world_size() == 1
                    and self.args.trace_method == "None"
                ):
                    return data
                inputs_list = []
                for key, val in self.input_to_stage_dic.items():
                    for d in list(data.keys()):
                        for i in val:
                            if d in i:
                                inputs_list.append(data.pop(d))
                                break
                inputs_list += list(data.values())
                return tuple(inputs_list)
            else:
                raise NotImplementedError("only support data in tuple, list or dict")
        else:
            return data

    def get_summary_writer(self, name: str = "MerakJobName") -> SummaryWriter:
        base_dir = self.args.output_dir
        date_format = "%b%d_%H-%M-%S_"
        date = datetime.now().strftime(date_format)
        hostname = socket.gethostname()
        log_dir = os.path.join(base_dir, "runs", date + hostname)

        os.makedirs(log_dir, exist_ok=True)

        return SummaryWriter(log_dir=log_dir)

    def init_engine(self, eval: bool = False):
        self.engine = PipelineEngine(
            self.model,
            self.args,
            optimizer=self.create_optimizer if not eval else None,
            lr_scheduler=self.create_lr_scheduler if not eval else None,
            tuning_params=self.tuning_params,
            loss_fn=self.loss_fn,
        )

    def train(self):
        self.init_engine()
        self.engine.batch_fn = self.prepare_data

        monitor = get_monitor()
        recover = Recover(monitor, self.engine.module, self.engine.optimizer, self.logger)
        self.engine.optimizer = OptimizerWrapper(recover, optim=self.engine.optimizer)

        if self.args.resume_from_checkpoint:
            iteration = self.load_checkpoint(self.args.resume_from_checkpoint)
            epochs_trained = self.tuning_params.resume(iteration)
            if hasattr(self.train_dataloader, "batch_sampler"):
                self.train_dataloader.batch_sampler.consumed_samples = (
                    iteration * self.tuning_params.total_batch_size
                )
        else:
            iteration = 0
            epochs_trained = 0

        if self.args.lora_config:
            self.peft_config = self.engine.module._add_lora_layers(self.model_config)
            self.engine._configure_optimizer(force=True)
            _ = self.load_checkpoint(
                self.args.resume_from_checkpoint, peft_config=self.peft_config
            )

        self.tuning_params.logging()

        for epoch in range(epochs_trained, math.ceil(self.args.num_train_epochs)):
            if dist.get_rank() == 0:
                self.logger.info(
                    f"\nEpoch: {epoch+1}/{math.ceil(self.args.num_train_epochs)}",
                    ranks=[0],
                )
            # if hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            #     self.train_dataloader.batch_sampler.set_epoch(epoch)
            for step in range(self.tuning_params.num_steps_per_epoch):
                self.tuning_params.step = step + 1
                self.tuning_params.global_steps += 1

                recover.start_recover()
                loss = self.engine.train_batch(self.train_dataloader)

                if self.summary_writer is not None:
                    self.summary_events = [
                        (
                            f"Train/loss",
                            loss.mean().item(),
                            self.tuning_params.global_steps,
                        ),
                        (
                            f"Train/learning_rate",
                            self.engine.optimizer.param_groups[0]["lr"].real,
                            self.tuning_params.global_steps,
                        ),
                    ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                if (
                    self.args.output_dir
                    and self.args.save
                    and self.tuning_params.global_steps % self.args.save_steps == 0
                ):
                    self.save_checkpoint()
                    self._rotate_checkpoints(self.args, self.args.output_dir)

                if self.tuning_params.should_break():
                    break

            if self.args.output_dir and self.args.save:
                self.save_checkpoint()
                self._rotate_checkpoints(self.args, self.args.output_dir)
            if self.args.do_eval:
                self.evaluation()

    def evaluation(self):
        assert (
            self.eval_dataloader is not None
        ), "The eval_dataloader is None, Please check eval_dataset"
        if self.engine is None:
            self.init_engine(eval=True)

        if self.args.resume_from_checkpoint and not self._already_load:
            iteration = self.load_checkpoint(self.args.resume_from_checkpoint)

        # eval_iterator = iter(eval_dataloader)
        self.logger.info("\nStart Evaluation...", ranks=[0])
        self.tuning_params.logging(eval=True)
        for step in range(self.tuning_params.eval_steps):
            self.tuning_params.eval_step += 1
            self.tuning_params.global_eval_steps += 1

            loss, logits, labels = self.engine.eval_batch(
                batch_fn=self.prepare_data, data_iter=self.eval_dataloader
            )

        if self.summary_writer is not None:
            self.summary_events = [
                (
                    f"Eval/loss",
                    loss.mean().item(),
                    self.tuning_params.global_eval_steps,
                ),
            ]
            for event in self.summary_events:  # write_summary_events
                self.summary_writer.add_scalar(event[0], event[1], event[2])
        dist.barrier()

    def save_checkpoint(self, best_model: Any = None):
        kwargs = {
            "best_model": best_model,
            "args": self.args,
            "peft_config": self.peft_config,
        }
        self._save_checkpoint(
            self.tuning_params.global_steps,
            self.engine.module,
            self.engine.optimizer,
            self.engine.lr_scheduler,
            **kwargs,
        )

    def load_checkpoint(
        self, resume_from_checkpoint: str, peft_config: Optional[Callable] = None
    ) -> int:
        if os.path.exists(resume_from_checkpoint):
            load_result, state_dict, opt_state_dict = self._load_checkpoint(
                self.engine.module,
                self.engine.optimizer,
                self.engine.lr_scheduler,
                self.args,
                peft_config,
                verbose=True,
            )

            if os.path.isfile(self.args.zero_reshard):
                # load consolidate opt state dict
                opt_state_dict = torch.load(
                    self.args.zero_reshard, map_location="cpu", weights_only=False
                )
                opt_state_dict["param_groups"] = self.engine.optimizer.param_groups
                self.engine.optimizer.load_state_dict(opt_state_dict)

                # reshard zero optimizer with new dp axies
                dynamic_loss_args = (
                    self.engine.amp_engine.dynamic_loss_args()
                    if self.args.fp16 or self.args.bf16
                    else None
                )
                self.engine.optimizer = configure_zero_optimizer(
                    self.engine.optimizer, self.args, dynamic_loss_args
                )

                # load loss scaler
                self.engine.optimizer.load_state_dict(opt_state_dict)

            del state_dict, opt_state_dict
            self._already_load = True
            return load_result

        else:
            raise ValueError(
                f"Cannot find checkpoint files in {self.args.resume_from_checkpoint}"
            )
