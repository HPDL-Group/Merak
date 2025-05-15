# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com), Yck (eyichenke@gmail.com)
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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/v2.6/megatron/checkpointing.py

import torch
import torch.nn as nn
import random
import sys
import os
import numpy as np
import datetime
import json
import shutil
import torch.distributed as dist

from collections import OrderedDict
from typing import Optional, Tuple, Callable, Union, List, Dict
from dataclasses import asdict
from pathlib import Path

from Merak import print_rank_0
from .utils import (
    ensure_directory_exists,
    unwrap_model,
    get_zero_param_shapes,
    get_checkpoint_name,
    ensure_directory_exists,
    get_best_checkpoint_filename,
    get_checkpoint_tracker_filename,
    read_metadata,
    detect_checkpoint_format,
    check_state_dict,
    PeftType,
    ShardedSafetensorLoader
)
from .. import mpu
from ..recompute import get_rng_tracker

class VersionManager:
    _CHECKPOINT_VERSION = None

    def __init__(self):
        pass

    @classmethod
    def set_checkpoint_version(cls, value):
        if cls._CHECKPOINT_VERSION is not None:
            assert cls._CHECKPOINT_VERSION == value, \
                "checkpoint versions do not match"
        cls._CHECKPOINT_VERSION = value

    @classmethod
    def get_checkpoint_version(cls):
        return cls._CHECKPOINT_VERSION

class CheckpointSaver:
    """Handles saving model checkpoints during training.

    This class manages the process of saving model checkpoints, including handling
    models in distributed environments and saving optimizer and learning rate
    scheduler states.
    """

    def __init__(self):
        """Initialize CheckpointSaver with training arguments.
        """
        self.args = None
        self.dtime = datetime.datetime.now().strftime('%Y-%m-%d')
        self.iteration = None
        self.save_path = None
        self.saved_best_model = None
        self.peft_config = None
        self.higher_better = None

    def save_checkpoint(
            self,
            iteration: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler,
            **kwargs):
        """Save a model checkpoint.

        This function saves the current state of the model, optimizer, and learning rate scheduler.
        It also handles special cases for saving LoRA configurations and Zero-Redundancy Optimizer (ZeRO) states.

        Args:
            iteration: Current training iteration.
            model: Training model to save.
            optimizer: Optimizer to save.
            lr_scheduler: Learning rate scheduler to save.
            **kwargs: Additional keyword arguments, including 'best_model' for tracking and 'peft_config' for LoRA.
        """
        try:
            # Extract optional arguments
            self.iteration = iteration
            self.peft_config = kwargs.get("peft_config", None)
            self.args = kwargs.get("args", None)
            self.higher_better = kwargs.get("higher_better", None)
            best_model = kwargs.get("best_model", None)

            # Prepare checkpoint file paths and directories
            checkpoint_name, zero_ckpt_name = self._prepare_save_path(iteration, best_model)
            if checkpoint_name == None and zero_ckpt_name == None:
                return

            if self.peft_config is not None:
                self._save_lora_config(checkpoint_name)

            # Prepare state dictionaries for saving
            model_state = self._prepare_model_state_dict(iteration, model, optimizer, lr_scheduler, **kwargs)
            lora_state = self._prepare_lora_state_dict(model)  # LoRA state if applicable
            zero_state = self._prepare_zero_optim_state_dict(model, optimizer)

            # Save the state dictionaries to disk
            self._save_state_dict(model_state, checkpoint_name, zero_ckpt_name, lora_state, zero_state, **kwargs)

        except Exception as e:
            print(f"Error occurred during checkpoint saving: {str(e)}")
            raise

    def _prepare_save_path(self, iteration: int = None, auc_score: Optional[float] = None) -> Tuple[str, str]:
        """Prepare and validate the checkpoint save path.

        Args:
            iteration: Current training iteration for naming.
            auc_score: Best model's metric score for tracking.
        Returns:
            Tuple containing the checkpoint save path and Zero-Redundancy Optimizer path.
        """
        # Determine whether this is a best model checkpoint
        if auc_score is not None:
            save_path = os.path.join(self.args.output_dir, 'best_ckpt')
            is_best = self._update_best_model(auc_score)
            if not is_best:
                return None, None
        else:
            # Date-based save directory.
            save_path = os.path.join(self.args.output_dir, f'{self.dtime}_ckpt')
        self.save_path = save_path

        # Generate checkpoint filenames
        checkpoint_name, zero_ckpt_name = get_checkpoint_name(
            save_path,
            iteration,
            self.args,
            best=auc_score,
            peft=self.peft_config is not None
        )

        # Ensure the save path exists
        if  mpu.get_data_parallel_rank() == 0:
            # Only dp rank 0 needs to create directories and save files
            ensure_directory_exists(checkpoint_name)

        if self.args.zero_stage == 1:
            with self.args.main_process_first():
                ensure_directory_exists(zero_ckpt_name)

        return checkpoint_name, zero_ckpt_name

    def _save_tracker_file(self):
        """Save a metadata file to track checkpoint iteration.

        Args:
            save_path: Path to save the tracker file.
            iteration: Current iteration number to save.
        """
        if dist.get_rank() == 0:  # Only save once in distributed environments.
            tracker_filename = get_checkpoint_tracker_filename(self.save_path)
            with open(tracker_filename, 'w') as f:
                f.write(str(self.iteration))

    def _prepare_model_state_dict(
            self,
            iteration: int,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: Callable,
            **kwargs) -> Dict:
        """Prepare the state dictionary for checkpoint saving.

        Args:
            iteration: Current iteration.
            model: Model to save.
            optimizer: Optimizer to save.
            lr_scheduler: Learning rate scheduler to save.
            **kwargs: Additional keyword arguments.
        Returns:
            Comprehensive state dictionary containing model, optimizer, and training states.
        """
        if  mpu.get_data_parallel_rank() == 0:
            state_dict = {}
            # Save training arguments and metadata
            state_dict['args'] = self.args
            state_dict['checkpoint_version'] = 3.0  # Versioning for backward compatibility.
            state_dict['iteration'] = iteration

            # Save model state
            state_dict['model'] = unwrap_model(model).state_dict()  # Unwrap for distributed models

            # Save optimizer state if configured
            if self.args.zero_stage != 1 and not self.args.no_save_optim:
                if optimizer is not None:
                    state_dict['optimizer'] = optimizer.state_dict()

            # Save learning rate scheduler state if applicable
            if lr_scheduler is not None:
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            # Save RNG states for reproducibility if enabled
            if not self.args.no_save_rng:
                state_dict['random_rng_state'] = random.getstate()
                state_dict['np_rng_state'] = np.random.get_state()
                state_dict['torch_rng_state'] = torch.get_rng_state()
                if torch.cuda.is_available():
                    state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
                state_dict['rng_tracker_states'] = get_rng_tracker().get_states()

            return state_dict
        return {}

    def _prepare_zero_optim_state_dict(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict:
        """Prepare optimizer states for Zero-Redundancy Optimization (ZeRO).

        Args:
            model: Model with parameters to save.
            optimizer: Optimizer to save states for.
        Returns:
            Dictionary containing optimizer states and parameter shapes.
        """
        if self.args.zero_stage == 1:
            return dict(
                optimizer_state_dict=optimizer.state_dict(),
                param_shapes=get_zero_param_shapes(optimizer, model)
            )

    def _prepare_lora_state_dict(self, model: nn.Module) -> Dict:
        """Prepare LoRA adapter state dictionary for saving.

        Args:
            model: Model containing LoRA adapters.
        Returns:
            State dictionary containing LoRA parameters.
        """
        if mpu.get_data_parallel_rank() == 0 and self.peft_config is not None:
            return self.get_peft_model_state_dict(model)
        return {}

    def get_peft_model_state_dict(self, model, state_dict=None) -> Dict[str, torch.Tensor]:
        """
        Extract the state dictionary of the Peft model.

        Args:
            model: The Peft model.
            state_dict (`dict`, optional): The state dictionary to extract from. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: The state dictionary containing Peft parameters.
        """
        if state_dict is None:
            state_dict = model.state_dict()

        modules_to_save = self.peft_config.modules_to_save

        if self.peft_config.peft_type == PeftType.LORA:
            # Handle different bias configurations for LoRA
            bias = self.peft_config.bias
            to_return = OrderedDict()

            if bias == "none":
                to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
            elif bias == "all":
                to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
            elif bias == "lora_only":
                for k in state_dict:
                    if "lora_" in k:
                        to_return[k] = state_dict[k]
                        bias_name = k.split("lora_")[0] + "bias"
                        if bias_name in state_dict:
                            to_return[bias_name] = state_dict[bias_name]
            else:
                raise NotImplementedError("Unsupported bias configuration.")

            # Include specified modules to save
            if modules_to_save:
                for module in modules_to_save:
                    for k in state_dict:
                        if module in k:
                            to_return[k] = state_dict[k]

        else:
            raise NotImplementedError("Unsupported Peft type.")

        return to_return

    def _save_lora_config(self, checkpoint_name: str):
        """Save LoRA configuration file.

        Args:
            checkpoint_name: Path where the configuration will be saved adjacent to the checkpoint.
        """
        config_path = os.path.join(
            os.path.dirname(checkpoint_name),
            "lora_config.json"
        )
        output_dict = asdict(self.peft_config)
        # Convert sets to lists for JSON serialization
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)
        # Save to file
        with open(config_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    def _save_state_dict(
            self,
            state_dict: Dict,
            checkpoint_name: str,
            zero_ckpt_name: Optional[str] = None,
            peft_state_dict: Optional[Dict] = None,
            zero_state_dict: Optional[Dict] = None,
            **kwargs):
        """Save the state dictionary to the specified path.

        Args:
            state_dict: State dictionary to save.
            checkpoint_name: Path to save the main checkpoint.
            zero_ckpt_name: Path for saving Zero-Redundancy Optimizer states.
            peft_state_dict: LoRA state dictionary to save if applicable.
            zero_state_dict: Zero Optimizer state dictionary.
            **kwargs: Additional keyword arguments.
        """
        try:
            # Save main state dictionary
            if mpu.get_data_parallel_rank() == 0:  # Only primary process saves the checkpoint.
                # Use LoRA state if available, otherwise use regular state.
                save_state = peft_state_dict if self.peft_config is not None else state_dict
                torch.save(
                    save_state,
                    checkpoint_name,
                    _use_new_zipfile_serialization=False
                )

            # Save Zero Optimizer states if required
            if self.args.zero_stage == 1:
                dist.barrier()
                torch.save(zero_state_dict, zero_ckpt_name)
                print_rank_0('zero checkpoint saved {}'.format(zero_ckpt_name))

            dist.barrier()  # Ensure all processes are synchronized

            # Update the latest iteration tracker
            self._save_tracker_file()

            print_rank_0(
                f'successfully saved checkpoint at iteration {self.iteration} to {self.save_path}'
            )

        except Exception as e:
            print(f"Failed to save checkpoint: {str(e)}")
            raise

    def _save_best_score(self):
        tracker_filename = get_best_checkpoint_filename(self.save_path)
        if dist.get_rank() == 0:
            with open(tracker_filename, 'w') as f:
                f.write(str(self.saved_best_model))

    def _update_best_model(self, auc_score: float):
        """Update the best model tracker.

        Args:
            auc_score: AUC score to compare and potentially update the best model.
        """
        update = False
        if self.saved_best_model is None:
            self.saved_best_model = auc_score
            update = True
        else:
            if self.higher_better and self.saved_best_model < auc_score:
                self.saved_best_model = auc_score
                update = True
            elif not self.higher_better and self.saved_best_model > auc_score:
                self.saved_best_model = auc_score
                update = True
            else:
                print_rank_0("Current model AUC score is not better than the saved best model.")
        if update:
            self._save_best_score()
        return update


class CheckpointLoader:
    """Checkpoint loader class to handle model loading in a distributed environment."""

    def __init__(self):
        """Initialize the CheckpointLoader instance."""
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.args = None
        self.peft_config = None
        self.version_manager = VersionManager()  # Initialize version manager

    def load_checkpoint(
            self, model, optimizer, lr_scheduler, args, peft_config = None, verbose: bool = False, strict: bool = True
        ) -> tuple:
        """
        Load a model checkpoint and return the iteration.

        Args:
            model (nn.Module): Model to load the checkpoint into.
            optimizer (torch.optim.Optimizer): Optimizer to load the state from the checkpoint.
            lr_scheduler (Callable): Learning rate scheduler to load the state.
            args: Arguments namespace containing configurations.
            verbose (bool, optional): Whether to print detailed loading information. Defaults to False.
            strict (bool, optional): Whether to strictly match keys in the state_dict. Defaults to True.

        Returns:
            tuple: A tuple containing the iteration number, model state_dict, and optimizer state_dict (if applicable).
        """
        self.model = unwrap_model(model)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.peft_config = peft_config

        if self.peft_config is not None:
            return self.load_peft_model_state_dict(verbose), None, None

        try:
            # Step 1: Detect checkpoint format and path
            state_dict, iteration, opt_state_dict, release = self._detect_checkpoint(self.args.resume_from_checkpoint, verbose)

            # Step 2: Load checkpoint contents
            if self.args.zero_stage == 1 and not release:
                self._load_zero_optimizer(opt_state_dict, iteration)

            # Step 3: Load state dictionary
            if 'model' in state_dict.keys():
                model_state_dict = state_dict['model']
            else:
                model_state_dict = state_dict
            self._load_state_dict(model_state_dict, strict, verbose)

            # Step 4: Load optimizer and learning rate scheduler
            if not release and not self.args.finetune and not self.args.no_load_optim:
                self._load_optimizer(state_dict, opt_state_dict)
                self._load_learning_rate_scheduler(state_dict)

            # Step 5: Load RNG states
            if not release and not self.args.finetune and not self.args.no_load_rng:
                self._load_rng_states(state_dict)

            # Step 6: Set and check version
            self.version_manager.set_checkpoint_version(state_dict.get('checkpoint_version', 0))
            print_rank_0(f'Checkpoint version: {self.version_manager.get_checkpoint_version()}')

            print_rank_0(f'Successfully loaded checkpoint at iteration {iteration}')
            return iteration, state_dict, opt_state_dict if self.args.zero_stage == 1 else None

        except Exception as e:
            print_rank_0(f'Failed to load checkpoint: {str(e)}')
            sys.exit(1)

    def _detect_checkpoint(self, load_dir: str, verbose: bool = False) -> tuple:
        """
        Detect the checkpoint and gather necessary information.

        Args:
            load_dir (str): Path to the checkpoint directory.
            verbose (bool, optional): Whether to print detailed information. Defaults to False.

        Returns:
            tuple: Contains state dictionary, iteration number, optimizer state dictionary (if any), and release flag.
        """
        checkpoint_name = detect_checkpoint_format(load_dir)
        if checkpoint_name is None:
            # If checkpoint not detected, read metadata file
            tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if not os.path.isfile(tracker_filename):
                print_rank_0('Warning: Could not find metadata file. Starting from random initialization...')
                return 0, None, None, False
            iteration, release = read_metadata(tracker_filename)
            checkpoint_name, zero_ckpt_name = get_checkpoint_name(
                load_dir, iteration, self.args, release=release
            )
        else:
            iteration = 0
            release = True
            zero_ckpt_name = None

        if verbose and dist.get_rank() == 0:
            print(f'Loading checkpoint: {checkpoint_name}')

        if self.args.zero_stage == 1 and not release:
            if os.path.isfile(zero_ckpt_name):
                opt_state_dict = torch.load(zero_ckpt_name, map_location='cpu')
            else:
                opt_state_dict = None
        else:
            opt_state_dict = None

        try:
            if 'safetensor' not in checkpoint_name:
                state_dict = torch.load(checkpoint_name, map_location='cpu')
            else:
                state_dict = ShardedSafetensorLoader(
                    self.model, self.args.resume_from_checkpoint
                ).get_state_dict()
        except Exception as e:
            print_rank_0(f'Failed to load checkpoint: {str(e)}')
            sys.exit(1)

        return state_dict, iteration, opt_state_dict, release

    def _load_state_dict(self, state_dict: dict, strict: bool, verbose: bool) -> None:
        """
        Load the state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
            strict (bool): Whether to strictly match keys in the state_dict.
            verbose (bool): Whether to print detailed information.
        """
        state_dict, strict = check_state_dict(self.model, state_dict, self.args, verbose)
        self.model.load_state_dict(state_dict, strict=strict)

    def _load_zero_optimizer(self, zero_ckpt_name: str, iteration: int) -> None:
        """
        Load the zero optimizer checkpoint.

        Args:
            zero_ckpt_name (str): Path to the zero optimizer checkpoint.
            iteration (int): Current iteration number.
        """
        opt_state_dict = torch.load(zero_ckpt_name, map_location='cpu')
        if self.optimizer is not None:
            self.optimizer.load_state_dict(
                opt_state_dict['optimizer_state_dict'],
                load_optimizer_states=True
            )

    def _load_optimizer(self, state_dict: dict, opt_state_dict: dict) -> None:
        """
        Load the optimizer state from the state dictionary.

        Args:
            state_dict (dict): State dictionary containing optimizer states.
            opt_state_dict (dict): Optimizer state dictionary.
            verbose (bool): Whether to print detailed information.
        """
        try:
            if self.optimizer is not None:
                if self.args.zero_stage is not None:
                    self._load_zero_optimizer_from_state_dict(state_dict, opt_state_dict)
                else:
                    self._load_optimizer_from_state_dict(state_dict)
        except KeyError as e:
            print_rank_0(f'Failed to load optimizer state: {str(e)}. Consider using --no_load_optim or --finetune.')
            sys.exit(1)

    def _load_zero_optimizer_from_state_dict(self, state_dict: dict, opt_state_dict: dict) -> None:
        """
        Load zero optimizer from the state dictionary.

        Args:
            state_dict (dict): State dictionary.
            opt_state_dict (dict): Optimizer state dictionary.
        """
        if opt_state_dict is not None:
            self.optimizer.load_state_dict(
                opt_state_dict['optimizer_state_dict'],
                load_optimizer_states=True
            )
        else:
            if self.args.fp16:
                self.optimizer.load_state_dict(state_dict['optimizer'], load_optimizer_states=True)
            else:
                self.optimizer.load_state_dict(state_dict['optimizer'])

    def _load_optimizer_from_state_dict(self, state_dict: dict) -> None:
        """
        Load optimizer from the state dictionary.

        Args:
            state_dict (dict): State dictionary.
            verbose (bool): Whether to print detailed information.
        """
        if self.args.fp16:
            self.optimizer.load_state_dict(state_dict['optimizer'], load_optimizer_states=True)
        else:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def _load_learning_rate_scheduler(self, state_dict: dict) -> None:
        """
        Load learning rate scheduler state.

        Args:
            state_dict (dict): State dictionary containing scheduler states.
            verbose (bool): Whether to print detailed information.
        """
        try:
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        except KeyError as e:
            print_rank_0(f'Failed to load learning rate scheduler state: {str(e)}')
            sys.exit(1)

    def _load_rng_states(self, state_dict: dict) -> None:
        """
        Load random number generator states.

        Args:
            state_dict (dict): State dictionary containing RNG states.
            verbose (bool): Whether to print detailed information.
        """
        try:
            if not self.args.no_load_rng:
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                get_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0(f'Failed to load RNG states from checkpoint. Consider using --no_load_rng or --finetune.')
            sys.exit(1)

    def load_peft_model_state_dict(
            self,
            verbose: bool = False
        ) -> OrderedDict:
        """
        Load a Peft model's state dictionary from a checkpoint file.

        Args:
            model: The Peft model to load the state dictionary into.
            args: Configuration arguments containing the checkpoint path.
            verbose: Whether to print detailed loading information.

        Returns:
            OrderedDict: The loaded state dictionary.
        """
        load_dir = self.args.resume_from_checkpoint

        # Detect the appropriate checkpoint file
        filename = detect_checkpoint_format(load_dir, peft=True)

        if filename is None:
            # Read the tracker file to set the iteration
            tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if not os.path.isfile(tracker_filename):
                if mpu.get_data_parallel_rank() == 0:
                    print(f'WARNING: No Lora model files found at {self.args.resume_from_checkpoint}')
                return OrderedDict()

            iteration, release = read_metadata(tracker_filename)
            filename, _ = get_checkpoint_name(load_dir, iteration, self.args, release=release, peft=True)
        else:
            iteration = 0

        print_rank_0(f'Loading Peft checkpoint from {filename}')

        try:
            # Load the checkpoint file
            peft_state_dict = torch.load(
                filename,
                map_location=torch.device("cpu"),
                weights_only=False
            )
        except Exception as e:
            print_rank_0(f'Failed to load Peft checkpoint: {str(e)}')
            return OrderedDict()

        # Update the state dictionary
        load_results = self._set_peft_model_state_dict(peft_state_dict, verbose)

        return load_results

    def _set_peft_model_state_dict(self, peft_model_state_dict, verbose=False) -> OrderedDict:
        """
        Set the state dictionary of the Peft model.

        Args:
            model: The Peft model.
            peft_model_state_dict: The state dictionary to set.
            args: Configuration arguments.
            peft_config: Configuration for the Peft model.
            verbose: Whether to print detailed information.

        Returns:
            OrderedDict: The resulting state dictionary after loading.
        """
        config = self.peft_config
        adapter_name = self.args.adapter_name
        modules_to_save = self.peft_config.modules_to_save

        if config.peft_type in (PeftType.LORA,):
            parameter_prefix = "lora_"
            modules_to_save.append(parameter_prefix)
            peft_model_state = OrderedDict()

            for k, v in peft_model_state_dict.items():
                if adapter_name in k:
                    adapter_name = ""  # Clear adapter_name if not needed

                if parameter_prefix in k:
                    suffix = k.split(parameter_prefix)[1]
                    if "." in suffix:
                        suffix_to_replace = ".".join(suffix.split(".")[1:])
                        k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}" if adapter_name else suffix_to_replace)
                    else:
                        k = f"{parameter_prefix}{k}" if adapter_name else k
                peft_model_state[k] = v

            # Check and apply strict loading
            peft_model_state, strict = check_state_dict(self.model, peft_model_state, self.args, verbose, load_list=modules_to_save)

            # Update the model's state dictionary
            load_result = self.model.load_state_dict(peft_model_state, strict=strict)

            return load_result
        else:
            raise NotImplementedError("Unsupported Peft type.")


def _sorted_checkpoints(
        output_dir: Optional[str] = None,
        checkpoint_prefix: str = "ckpt"
    ) -> List[str]:
    """
    List and sort the checkpoint paths in the specified output directory.

    Args:
        output_dir (Optional[str]): Directory containing checkpoints. Defaults to None.
        checkpoint_prefix (str): Prefix for checkpoint files. Defaults to "ckpt".

    Returns:
        List[str]: A sorted list of checkpoint paths.
    """
    checkpoint_paths = []

    if output_dir:
        output_path = Path(output_dir)
        # Collect all paths matching the checkpoint pattern
        glob_pattern = output_path.glob(f"*_ckpt")
        for path in glob_pattern:
            if not path.is_dir():
                continue  # Skip non-directory paths
            # Skip 'best_ckpt' directories if present
            if "best_ckpt" in str(path):
                continue
            # Look for iteration-specific files within each checkpoint directory
            for item in path.glob("iter_*"):
                checkpoint_paths.append(str(item))

    # Sort the collected checkpoint paths
    checkpoint_paths.sort()
    return checkpoint_paths

def delete_empty_checkpoint_directories(
        output_dir: Optional[str] = None,
        checkpoint_prefix: str = "ckpt"
    ) -> None:
    """
    Delete empty checkpoint directories in the specified output directory.

    Args:
        output_dir (Optional[str]): Directory containing checkpoints. Defaults to None.
        checkpoint_prefix (str): Prefix for checkpoint directories. Defaults to "ckpt".
    """
    if output_dir:
        output_path = Path(output_dir)
        glob_pattern = output_path.glob(f"*_ckpt")
        for directory in glob_pattern:
            if directory.is_dir() and not list(directory.iterdir()):
                print(f"Deleting empty checkpoint directory: {directory}")
                try:
                    shutil.rmtree(str(directory))
                except (FileNotFoundError, PermissionError) as e:
                    print(f"Failed to delete directory {directory}: {str(e)}")

def rotate_checkpoints(
        args,
        output_dir: Optional[str] = None,
        checkpoint_prefix: str = "ckpt"
    ) -> None:
    """
    Manage checkpoints by deleting old ones according to save_total_limit.

    Args:
        args: Arguments containing checkpoint management parameters.
        output_dir (Optional[str]): Directory containing checkpoints. Defaults to None.
        checkpoint_prefix (str): Prefix for checkpoint files. Defaults to "ckpt".
    """
    if args.save_total_limit is None or args.save_total_limit <= 0:
        return

    # Delete any empty checkpoint directories
    delete_empty_checkpoint_directories(output_dir, checkpoint_prefix)

    # Retrieve sorted list of checkpoint paths
    checkpoints = _sorted_checkpoints(output_dir, checkpoint_prefix)

    # Determine the number of checkpoints to keep
    if not checkpoints:
        print("No checkpoints found.")
        return

    if len(checkpoints) <= args.save_total_limit:
        print(f"All {len(checkpoints)} checkpoints are within the limit of {args.save_total_limit}. Nothing to delete.")
        return

    # Adjust the save limit if necessary
    save_limit = args.save_total_limit
    if args.save_total_limit == 1 and args.load_best_model_at_end:
        save_limit = 2  # Keep one more to ensure we don't delete the last checkpoint

    # Calculate the number of checkpoints to delete
    num_to_delete = max(0, len(checkpoints) - save_limit)
    if num_to_delete > 0:
        checkpoints_to_delete = checkpoints[:num_to_delete]
        if dist.get_rank() == 0:
            for path in checkpoints_to_delete:
                print(f"Deleting checkpoint: {path}")
                try:
                    shutil.rmtree(path)
                except (FileNotFoundError, PermissionError) as e:
                    print(f"Failed to delete checkpoint {path}: {str(e)}")
            print(f"Successfully maintained {save_limit} most recent checkpoints.")
        dist.barrier()
    else:
        print(f"All {len(checkpoints)} checkpoints are within the limit of {args.save_total_limit}.")