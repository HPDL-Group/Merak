# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com), Swli (lucasleesw9@gmail.com)
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

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/trainer.py

import torch
from transformers.trainer_utils import (
    TrainOutput,
    set_seed,
    speed_metrics,
)

from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.utils import logging
import collections
import math
import os
import warnings
import sys
import time
from . import mpu, print_rank_0

from .modules.lora.config import TaskType, LoraConfig, _prepare_lora_config

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


logger = logging.get_logger(__name__)


def train(
    self,
    resume_from_checkpoint = None,
    trial= None,
    ignore_keys_for_eval = None,
    **kwargs,
):
    """
    Main training entry point.

    Args:
        resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
            If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
            :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
            `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
            training will resume from the model/optimizer/scheduler states loaded here.
        trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
            The trial run or the hyperparameter dictionary for hyperparameter search.
        ignore_keys_for_eval (:obj:`List[str]`, `optional`)
            A list of keys in the output of your model (if it is a dictionary) that should be ignored when
            gathering predictions for evaluation during the training.
        kwargs:
            Additional keyword arguments used to hide deprecated arguments
    """
    resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

    # memory metrics - must set up as early as possible
    self._memory_tracker.start()

    args = self.args

    self.is_in_train = True

    if "model_path" in kwargs:
        resume_from_checkpoint = kwargs.pop("model_path")
        warnings.warn(
            "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
            "instead.",
            FutureWarning,
        )
    if len(kwargs) > 0:
        raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
    # This might change the seed so needs to run first.
    self._hp_search_setup(trial)

    # Model re-init
    model_reloaded = False
    if self.model_init is not None:
        # Seed must be set before instantiating the model when using model_init.
        set_seed(args.seed)
        self.model = self.call_model_init(trial)
        model_reloaded = True
        # Reinitializes optimizer and scheduler
        self.optimizer, self.lr_scheduler = None, None

    # If model was re-initialized, put it on the right device and update self.model_wrapped
    if model_reloaded:
        if self.place_model_on_device:
            self._move_model_to_device(self.model, args.device)
        self.model_wrapped = self.model

    # Keeping track whether we can can len() on the dataset or not
    train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps #* args.world_size
    if train_dataset_is_sized:
        if args.max_steps > 0:
            max_steps = args.max_steps
        else:
            max_steps = math.ceil(args.num_train_epochs * len(self.train_dataset) // (total_train_batch_size * mpu.get_data_parallel_world_size()))
    else:
        # see __init__. max_steps is set when the dataset has no __len__
        max_steps = args.max_steps

    self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    self.state = TrainerState()
    self.state.is_hyper_param_search = trial is not None

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        self.model.gradient_checkpointing_enable()

    model = self._wrap_model(self.model_wrapped)

    # for the rest of this function `model` is the outside model, whether it was wrapped or not
    if model is not self.model:
        self.model_wrapped = model

    # Data loader and number of training steps
    self.len_dataset = None
    self._get_iter_dataloader()

    # Setting up training control variables:
    # number of training epochs: num_train_epochs
    # number of training steps per epoch: num_update_steps_per_epoch
    # total number of training steps to execute: max_steps
    if train_dataset_is_sized:
        if self.iter_dataloader is not None:
            num_update_steps_per_epoch = len(self.iter_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                if self.train_dataset is not None:
                    num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            assert self.len_dataset is not None, "Length of datasets or dataloader could not be None"
            num_update_steps_per_epoch = self.len_dataset // ( args.per_device_train_batch_size * args.gradient_accumulation_steps)
            if args.max_steps < 0:
                max_steps = math.ceil(args.num_train_epochs * len(self.train_dataset) // 
                                        (args.train_batch_size * args.gradient_accumulation_steps * mpu.get_data_parallel_world_size()))
            num_train_epochs = max_steps // num_update_steps_per_epoch 
    else:
        # see __init__. max_steps is set when the dataset has no __len__
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_train_epochs = 1
        num_update_steps_per_epoch = max_steps
        num_train_samples = args.max_steps * total_train_batch_size
    num_steps_per_epoch = max_steps // num_train_epochs

    # load checkpoint
    epochs_trained = 0
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        iteration = self.load_from_checkpoint(resume_from_checkpoint)
        if self.args.max_steps > 0:
            if iteration < self.args.max_steps:
                epochs_trained = int(iteration/num_steps_per_epoch)
        else:
            if iteration < max_steps:
                epochs_trained = int(iteration/num_steps_per_epoch)

        if self.args.max_steps > 0 and iteration > self.args.max_steps:
            self.state.global_step = 0
            self.pipe_model.global_steps = 0
        elif iteration > max_steps:
            self.state.global_step = 0
            self.pipe_model.global_steps = 0
        else:
            self.state.global_step = iteration
            self.pipe_model.global_steps = iteration

    if self.args.lora_config:
        assert self.mp == 1, "Don't support tensor parallelism for lora, currently"
        peft_config = LoraConfig(**self.args.get_lora_config())
        peft_config = _prepare_lora_config(peft_config, self.model.config.to_dict())
        self.pipe_model.loramodel(peft_config)
        self.peft_config = peft_config

        if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
            load_results = self.load_from_checkpoint(resume_from_checkpoint, peft=True)

        # reset optimizer
        self.create_optimizer()
        self.create_scheduler(num_training_steps=max_steps, optimizer=self.optimizer)
        self.pipe_model._configure_optimizer(self.optimizer, None)
        self.pipe_model.lr_scheduler = self.lr_scheduler

        self.pipe_model.print_trainable_parameters()
    else:
        self.peft_config = None


    # Train!
    if self.iter_dataloader is not None and torch.distributed.get_rank()==0:
        num_examples = (
            self.num_examples(self.iter_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

    self.state.epoch = 0
    start_time = time.time()
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None

    # Update the references
    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    self.callback_handler.train_dataloader = self.iter_dataloader
    self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
    # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    # to set this after the load.
    self.state.max_steps = max_steps
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()

    # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    tr_loss = torch.tensor(0.0).to(args.device)
    # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()

    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    if not args.ignore_data_skip:
        for epoch in range(epochs_trained):
            # We just need to begin an iteration to create the randomization of the sampler.
            for _ in self.iter_dataloader:
                break

    for epoch in range(epochs_trained, num_train_epochs):
        self._reset_dataloader(epoch)
        print_rank_0("\nCurrent processing of training epoch (%d/%d)\n" % (epoch + 1, num_train_epochs))

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None


        epoch_iterator = range(num_steps_per_epoch)

        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
        

        for step, _ in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.update(1)
                continue
            elif steps_trained_progress_bar is not None:
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None

            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

            tr_loss_step = self.training_step(self.iter_dataloader)

            if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                # if loss is nan or inf simply add the average of previous logged losses
                tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
            else:
                tr_loss += tr_loss_step

            self.state.global_step += 1
            self.state.epoch = epoch + (step + 1) / num_steps_per_epoch
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                
            if self.args.output_dir and self.args.save and self.state.global_step % self.args.save_steps == 0:
                self.save_to_checkpoint()
                self._rotate_checkpoints(self.args.output_dir)


            if self.control.should_epoch_stop or self.control.should_training_stop:
                break


        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        if self.args.output_dir and self.args.save:
            self.save_to_checkpoint()
            self._rotate_checkpoints(self.args.output_dir)

        if self.control.should_training_stop:
            break

    if args.past_index and hasattr(self, "_past"):
        # Clean the state at the end of training
        delattr(self, "_past")

    # add remaining tr_loss
    self._total_loss_scalar += tr_loss.item()
    train_loss = self._total_loss_scalar / self.state.global_step
    if self.train_dataset is not None:
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    else:
        metrics = None

    return TrainOutput(self.state.global_step, train_loss, metrics)
