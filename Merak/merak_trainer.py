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

import os
import socket
import math
import torch
import torchvision
from datetime import datetime
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Any, Dict, Tuple, Optional, Callable, Union

from Merak import print_rank_0
from .utils import WorkerInitObj, MegatronPretrainingRandomSampler
from .core import (
    mpu,
    PipelineEngine,
    checkpoint,
    PipelineModule
)
from .core.recompute import checkpoint as checkpoint_func
from .core.fx.tracer.utils import _generate_dummy_input
from .utils import BaseParams
from .merak_args import MerakArguments, mergeargs, manual_set_args
from .initialize import get_grid, get_topo

from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import AdamW, Adafactor, get_scheduler

class MerakTrainer:
    def __init__(
            self,
            model: nn.Module,
            args: MerakArguments,
            train_dataset: Optional[Any] = None,
            eval_dataset: Optional[Any] = None,
            data_collator: Optional[Callable] = None,
            leaf_modules: Tuple[nn.Module] = ()
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

        super().__init__()

        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.train_engine = None
        self.eval_engine = None
        self.input_to_stage_dic = None

        self._save_checkpoint = checkpoint.save_checkpoint
        self._rotate_checkpoints = checkpoint.rotate_checkpoints
        self._load_checkpoint = checkpoint.load_checkpoint
        self._load_peft_model_state_dict = checkpoint.load_peft_model_state_dict
        self.engine = PipelineEngine

        if hasattr(self.model, "config"):
            self.model_config = self.model.config
            mergeargs(self.args, self.model_config)
        # else:
        #     mergeargs(self.args, self.model)
        manual_set_args(self.args)

        self.peft_config = None

        self.train_params = BaseParams(self.args)
        self.eval_params = BaseParams(self.args)

        if dist.get_rank() == 0:
            self.summary_writer = self.get_summary_writer()
        else:
            self.summary_writer = None

        # init training parameters
        self.create_dataloader()
        # self.optimizer = self.create_optimizer(self.model)
        self.train_params.train(self.train_dataloader)
        if self.eval_dataloader is not None:
            self.eval_params.eval(self.eval_dataloader)
        # self.lr_scheduler = self.create_lr_scheduler()
        self.loss_fn = self.get_loss_fn() \
            if not self.args.parallel_vocab else self.get_vocab_loss_fn()

        self.create_pipeline_module()

    def create_pipeline_module(self):
        if self.args.trace_method in ['fx', 'dynamo']:
            self.model = PipelineModule(
                model=self.model,
                args=self.args,
                loss_fn=self.loss_fn,
                topology=get_topo(),
                communicaiton_grid=get_grid(),
                activation_checkpoint_func=checkpoint_func,
                leaf_modules=self.leaf_modules
            )

            self.input_to_stage_dic = self.model.input_to_stage_dic
        else:
            dummy_input = _generate_dummy_input(self.args, self.model)
            self.input_to_stage_dic = {0: list(dummy_input.keys())}
            self.args.no_tie_modules = True
            setattr(self.model, 'input_to_stage_dic', self.input_to_stage_dic)
            setattr(self.model, '_grid', get_grid())
            assert mpu.get_pipe_parallel_world_size() * mpu.get_model_parallel_world_size() == 1

        if mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0:
            if hasattr(self.model, 'stage_id'):
                print(dist.get_rank(), self.model.stage_id, self.model)
            else:
                print(dist.get_rank(), self.model)

    def get_train_sampler(self) -> MegatronPretrainingRandomSampler:
        return MegatronPretrainingRandomSampler(
            total_samples=len(self.train_dataset),
            # Set random seed according to be consumed examples,
            # but currently not supported
            consumed_samples=0,
            micro_batch_size=self.args.per_device_train_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())

    def create_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is not None:
            train_dataset = self.train_dataset

            worker_init = WorkerInitObj(self.args.seed + \
                                        mpu.get_data_parallel_rank())
            train_sampler = self.get_train_sampler()

            self.train_dataloader =  torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=self.collate_fn,
                worker_init_fn=worker_init \
                    if self.args.dataloader_num_workers > 0 else None
            )
        else:
            raise ValueError("MerakTrainer: training requires a train_dataset.")
        
        if self.eval_dataset is not None:

            eval_dataset = self.eval_dataset

            self.eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=self.collate_fn,
            )

    def create_optimizer(self, model: PipelineModule) -> torch.optim.Optimizer:
        
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters \
                            if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() \
                           if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() \
                           if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False,
                                "relative_step": False}
        else:
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                  **optimizer_kwargs)

        return self.optimizer

    def create_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer
        ) -> torch.optim.lr_scheduler:

        self.lr_scheduler = get_scheduler(
                       self.args.lr_scheduler_type,
                       optimizer=optimizer,
                       num_warmup_steps=self.args.get_warmup_steps(
                                             self.train_params.max_steps),
                       num_training_steps=self.train_params.max_steps,
                       scheduler_specific_kwargs=self.args.lr_scheduler_kwargs
                       )
        return self.lr_scheduler

    def get_loss_fn(self) -> Callable:
        criterion = nn.CrossEntropyLoss()
        def loss_fn(
            outputs: Union[torch.Tensor, tuple],
            labels: Union[torch.Tensor, tuple]
            ) -> torch.Tensor:
            if not isinstance(self.model, PipelineModule):
                if isinstance(outputs, dict):
                    outputs = outputs['logits']
                else:
                    outputs = outputs[1]
                labels = labels['labels']
            else:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(labels, tuple):
                    labels = labels[0]

            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             labels.view(-1))
            return loss
        return loss_fn

    def get_vocab_loss_fn(self) -> Callable:
        def vocab_loss_fn(
            outputs: Union[torch.Tensor, tuple],
            labels: Union[torch.Tensor, tuple]
            ) -> torch.Tensor:
            if isinstance(outputs, tuple):
                outputs = outputs[0][0]
            if isinstance(labels, tuple):
                labels = labels[0]
            losses = mpu.vocab_parallel_cross_entropy(outputs, labels)
            # loss mask??
            loss = torch.mean(losses.view(-1))
            return loss
        return vocab_loss_fn

    def prepare_data(
        self,
        data: Union[tuple, list, dict]
        ) -> Union[tuple, list]:
        if not isinstance(data, (tuple, list)):
            if hasattr(data, "data"):
                data = data.data
            if isinstance(data, dict):
                if mpu.get_pipe_parallel_world_size() == 1 and \
                   mpu.get_model_parallel_world_size() == 1 and \
                   self.args.trace_method == 'None':
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
                raise NotImplementedError('only support data in tuple, list or dict')
        else:
            return data

    def get_summary_writer(self, name: str = "MerakJobName") -> SummaryWriter:
        base_dir = self.args.output_dir
        date_format = "%b%d_%H-%M-%S_"
        date = datetime.now().strftime(date_format)
        hostname = socket.gethostname()
        log_dir = os.path.join(base_dir, "runs", date+hostname)

        os.makedirs(log_dir, exist_ok=True)

        return SummaryWriter(log_dir=log_dir)

    def train(self):
        self.train_engine = self.engine(
            self.model,
            self.args,
            optimizer=self.create_optimizer,
            lr_scheduler=self.create_lr_scheduler,
            tuning_params=self.train_params,
            dataloader=self.train_dataloader,
            loss_fn=self.loss_fn,
        )
        self.train_engine.batch_fn = self.prepare_data

        if self.args.resume_from_checkpoint:
            iteration = self.load_checkpoint(self.args.resume_from_checkpoint)
            epochs_trained = self.train_params.resume(iteration)
        else:
            iteration = 0
            epochs_trained = 0

        if self.args.lora_config:
            peft_config = self.train_engine.module._add_lora_layers(self.model_config)
            _ = self.load_checkpoint(self.args.resume_from_checkpoint, peft_config=peft_config)

        for epoch in range(epochs_trained, math.ceil(self.args.num_train_epochs)):
            if dist.get_rank() == 0:
                print(f"\nEpoch: {epoch+1}/{math.ceil(self.args.num_train_epochs)}")
            # if hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            #     self.train_dataloader.batch_sampler.set_epoch(epoch)
            for step in range(self.train_params.num_steps_per_epoch):
                self.train_params.step = step + 1
                self.train_params.global_steps += 1

                loss = self.train_engine.train_batch(self.train_dataloader)

                if self.summary_writer is not None:
                    self.summary_events = [
                        (f'Train/loss',
                         loss.mean().item(),
                         self.train_params.global_steps),
                        (f'Train/learning_rate',
                         self.train_engine.optimizer.param_groups[0]['lr'].real,
                         self.train_params.global_steps)
                    ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0],
                                                       event[1],
                                                       event[2])
                if self.args.output_dir and self.args.save and \
                   self.train_params.global_steps % self.args.save_steps == 0:
                    self.save_checkpoint()
                    self._rotate_checkpoints(self.args, self.args.output_dir)

            self.train_engine.reset_dataiterator(self.train_dataloader)
            if self.args.do_eval:
                self.evaluation()
            if self.args.output_dir and self.args.save:
                self.save_checkpoint()
                self._rotate_checkpoints(self.args, self.args.output_dir)

    def evaluation(self):
        assert self.eval_dataloader is not None, \
        "The eval_dataloader is None, Please check eval_dataset"
        if self.eval_engine is None:
            self.eval_engine = self.engine(
                self.train_engine.module,
                self.args,
                optimizer=None,
                lr_scheduler=None,
                tuning_params=self.eval_params,
                dataloader=self.eval_dataloader,
                loss_fn=self.loss_fn
            )
            self.eval_engine.return_logits = True

        if self.args.resume_from_checkpoint:
            iteration = self.load_checkpoint(self.args.resume_from_checkpoint)

        # eval_iterator = iter(eval_dataloader)
        print_rank_0("\nStart Evaluation...")
        for step in range(self.eval_params.eval_steps):
            self.eval_params.step += 1
            self.eval_params.global_steps += 1

            loss, logits, labels = self.eval_engine.eval_batch(
                batch_fn=self.prepare_data
            )

        if self.summary_writer is not None:
            self.summary_events = [
                (f'Eval/loss',
                loss.mean().item(),
                self.eval_params.global_steps),
            ]
            for event in self.summary_events:  # write_summary_events
                self.summary_writer.add_scalar(event[0],
                                               event[1],
                                               event[2])
        self.eval_engine.reset_dataiterator(self.eval_dataloader)

    def save_checkpoint(self, best_model: Any = None):
        peft_config = self.train_engine.peft_config \
            if self.train_engine is not None else self.eval_engine.peft_config

        kwargs = {"best_model": best_model,
                  "args": self.args,
                  "peft_config": peft_config
                  }
        self._save_checkpoint(self.train_params.global_steps,
                              self.train_engine.module,
                              self.optimizer,
                              self.lr_scheduler,
                              **kwargs)

    def load_checkpoint(
        self,
        resume_from_checkpoint: str,
        peft_config: Optional[Callable] = None
        ) -> int:
        if os.path.exists(resume_from_checkpoint):
            if peft_config:
                load_results = checkpoint.load_peft_model_state_dict(
                    self.model,
                    self.args,
                    peft_config,
                    verbose=True
                )
                return load_results
            else:
                iteration, state_dict, opt_state_dict = self._load_checkpoint(
                        self.train_engine.module,
                        self.optimizer,
                        self.lr_scheduler,
                        self.args,
                        verbose=True
                )
                del state_dict, opt_state_dict
                return iteration

        else:
            raise ValueError("Cannot find checkpoint files")
        
