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
import torch
import torchvision
import torch.distributed as dist
from torch.utils.data import DataLoader
import gc
import importlib
import warnings

from . import print_rank_0, get_grid, get_topo, get_patched_func
from . import mpu
from .mpu.initialize import _set_random_seed

from .modules.utils import get_params_for_weight_decay_optimization, init_method_normal
from .modules.module import PipelineModule
from .modules.layer_proxy import Conv1DProxy, LinearProxy, EmbeddingProxy
from .modules.mp_attrs import set_mp_attr, mp_is_setted, set_tp_layer_lists
from .modules.mp_layers import ColPara

from .runtime.utils import see_memory_usage
from .runtime.checkpointing import checkpoint as checkpoint_func
from .runtime.pipe_engine import PipelineEngine

from .utils.merak_args import mergeargs, MerakArguments, manual_set_args
from .utils.dataloader import MegatronPretrainingRandomSampler
from .utils.logging import AccMetric, log_dist
from .utils.checkpoint import save_checkpoint, load_checkpoint, load_peft_model_state_dict

from .autoshard.convert import convert_to_sequential, hf_fx_compatibility

from .train_func import train

import transformers
import datasets
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.file_utils import is_datasets_available
from transformers import Trainer
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)

from pathlib import Path
import shutil


class MerakTrainer(Trainer):
    def __init__(self, add_model=(), leaf_modules=(), loss_fn=torch.nn.CrossEntropyLoss(), **kwargs):
        """
        Class of Merak's trainer was derived with transformers.Trainer (https://huggingface.co/docs/transformers/master/en/main_classes/trainer#trainer) for convenience.
        Merak trainer extends transformers.Trainer to 3D parallelism.
        We provide some argument for user, to support tracing and loss computing

        Parameters:
        -   leaf_modules (Tuple[`torch.nn.Module`], defaults to ()) -- If a module cannot be traced by `torch.fx`, set it as leaf modules.
        -   loss_fn (`torch.nn.Module`, defaults to `torch.nn.CrossEntropyLoss()`) -- Loss function that computes loss value. Merak would not use `trainer.compute_loss`.

        """

        self.dp = mpu.get_data_parallel_world_size()
        self.pp = mpu.get_pipe_parallel_world_size()
        self.mp = mpu.get_model_parallel_world_size()

        self.leaf_modules = leaf_modules
        self.loss_fn = loss_fn

        self.add_model = add_model

        if 'args' not in kwargs:
            output_dir = "tmp_trainer"
            print_rank_0(f"No `MerakArguments` passed, using `output_dir={output_dir}`.")
            kwargs['args'] = MerakArguments(output_dir=output_dir)

        # simplely set num_layer with summation of length of Sequential class
        if kwargs['args'].num_layers is None:
            def recursive_find_sequantial(m, num_layer):
                if hasattr(m, '_modules'):
                    for k in getattr(m, '_modules'):
                        if isinstance(getattr(m,k),torch.nn.Sequential):
                            num_layer += len(getattr(m,k))
                        elif isinstance(getattr(m,k),torch.nn.Module):
                            num_layer = recursive_find_sequantial(getattr(m,k),num_layer)
                return num_layer
            n_layer = recursive_find_sequantial(kwargs['model'], 0)
            kwargs['args'].num_layers = n_layer if n_layer > 0 else None
        super().__init__(**kwargs)

        self.args.torch_ddp = False
        # Merge duplicate parameters
        if hasattr(self.model, "config"):
            mergeargs(self.args, self.model.config)
        else:
            mergeargs(self.args, self.model)

        if self.args.fp16 or self.args.half_precision_backend == "apex":
            self.model = self.model.half()

        assert dist.get_world_size() == self.pp*self.mp*self.dp, 'pp*tp*dp must equal to world size'



    def add_leaf_modules(self, leaf_modules):
        self.leaf_modules = leaf_modules

    ## used in beginning of each train/eval loop
    def _wrap_model(self, model, training=True):
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1 and self.args.DDP_impl == 'torch':
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                find_unused_parameters = not getattr(model.config, "_gradient_checkpointing", False)
            else:
                find_unused_parameters = True
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device=self.args.local_rank if self.args._n_gpu != 0 else None,
                find_unused_parameters=find_unused_parameters,
            )

        return model

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        r"""
        Merak use this function to create 3D paralllism model, please DO NOT REWRITE.
        Change optimizer and lr scheduler please rewrite 
        self.create_optimizer and self.create_scheduler instead
        """

        manual_set_args(self.args)
        _set_random_seed(self.args.seed)

        if self.mp > 1:
            if self.args.sequence_parallel:
                mpu.mappings.set_sequence_dim(self.args.sequence_dim)
            model_class = self.model.__class__
            if self.args.tp_overlapping_level > 1:
                from .modules.transformer_blocks import PipedGPT2Model, tp_overlapping_available, PipedGPT2Block
                if not tp_overlapping_available(model_class):
                    print_rank_0(f'not support tp overlapping level {self.args.tp_overlapping_level} in model {model_class}, will reset the level to 1')
                    self.args.tp_overlapping_level = 1
                else:
                    assert self.model.config.use_cache == False
                    assert self.model.config.output_attentions == False
                    assert self.model.config.output_hidden_states == False
                    assert self.model.config.add_cross_attention == False
                    self.model.transformer = PipedGPT2Model(self.model.config)


            if not mp_is_setted():
                from .modules.mp_mapping import get_mp_layer_lists
                mp_layer_lists = get_mp_layer_lists(model_class)
                if mp_layer_lists is not None:
                    set_tp_layer_lists(**mp_layer_lists)
            assert mp_is_setted(), \
            f'model {self.model.__class__.__name__} is not supported by auto tp now, should set tp attr manually with set_tp_layer_lists'
            self.model = set_mp_attr(self.model, self.mp)
        
        if self.args.wall_clock_breakdown: 
            print_rank_0(self.model)

        emb_dim = set()
        def add_dim(emb_dim, m):
            if isinstance(m, (EmbeddingProxy, LinearProxy)):
                emb_dim.add(m.weight_shape)
            else:
                emb_dim.add(m.weight.shape)
        hf_config=None
        if hasattr(self.model, 'config'):
            hf_config = self.model.config
            if self.model.config.tie_word_embeddings:
                for m in self.model.modules():
                    try:
                        if hasattr(m, 'get_input_embeddings') and m.get_input_embeddings() is not None:
                            add_dim(emb_dim, m.get_input_embeddings())
                        if hasattr(m, 'get_output_embeddings') and m.get_output_embeddings() is not None:
                            add_dim(emb_dim, m.get_output_embeddings())
                    except AttributeError:
                        continue
        elif hasattr(self.model, 'get_input_embeddings'):
            add_dim(emb_dim, self.model.get_input_embeddings())
        elif hasattr(self.model, 'get_output_embeddings'):
            add_dim(emb_dim, self.model.get_output_embeddings())
        
        # see_memory_usage('**** \n memory consumption after replacing mp', force=True)
        if self.args.input_names is None and hf_fx_compatibility(self.model): 
            self.iter_dataloader = iter(self.get_train_dataloader())
            trace_batch = next(self.iter_dataloader)
            if isinstance(trace_batch, dict):
                if 'labels' in trace_batch:
                    trace_batch.pop('labels')
                if 'label' in trace_batch:
                    trace_batch.pop('label')
                self.args.input_names=list(trace_batch.keys())
            del self.iter_dataloader

        model, model_layers, input_to_shard_dic = convert_to_sequential(self.model, self.args, self.add_model, self.leaf_modules)
        self.model_name = self.model._get_name()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if model_layers[0].dummy_inputs is None:
            self.iter_dataloader = iter(self.get_train_dataloader())
            one_batch = next(self.iter_dataloader)
            one_batch.pop(-1)
            model_layers[0].dummy_inputs = one_batch
            del self.iter_dataloader

        # see_memory_usage('**** \n memory consumption after layer shard', force=True)

        pipe_model = PipelineModule(layers=model_layers,
                                loss_fn=self.get_loss_fn(self.loss_fn),
                                topology=get_topo(),
                                communicaiton_grid=get_grid(), 
                                partition_method=self.args.partition_method,
                                activation_checkpoint_interval=self.args.checkpoint_num_layers,
                                activation_checkpoint_func=checkpoint_func, 
                                activation_checkpoint_ratio = self.args.activation_checkpoint_ratio,
                                tie_dims=emb_dim,
                                input_to_shard_dic=input_to_shard_dic,
                                sequence_parallel=self.args.sequence_parallel)

        self.input_to_stage_dic = pipe_model.input_to_stage_dic

        # print_rank_0(['*****', self.input_to_stage_dic, input_to_shard_dic])
        # see_memory_usage('**** \n memory consumption after pipe module created', force=True)


        # transformers.modeling_utils.Conv1D, torch.nn.Linear = get_patched_func()
        importlib.reload(transformers.modeling_utils)
        importlib.reload(torch.nn)

        def build_module(model, proxy_layer, init_args):
            for n, module in model.named_children():
                if isinstance(module, proxy_layer):
                    setattr(model, n, module.build(init_args, self.args.fp16))
                if len(list(module.children())) > 0:
                    ## compound module, go inside it
                    build_module(module, proxy_layer, init_args)
        
        build_module(pipe_model, Conv1DProxy, (self.args.init_method_std, self.args.num_layers))
        build_module(pipe_model, LinearProxy, (self.args.init_method_std, self.args.num_layers))
        init_method = init_method_normal(self.args.init_method_std)
        build_module(pipe_model, EmbeddingProxy, (self.args.sequence_parallel, init_method))
        
        if self.mp > 1:
            if self.args.parallel_vocab:
                # replace loss function
                pipe_model.loss_fn = self.get_vocab_loss_fn()
                # replace module to VocabParallelEmbedding and column parallel
                def replace_module(model, to_replaced, module_func, get_args, sequence_parallel):
                    for n, module in model.named_children():
                        if sequence_parallel and isinstance(module, torch.nn.LayerNorm):
                            setattr(module.weight, 'sequence_parallel', self.args.sequence_parallel)
                            setattr(module.bias, 'sequence_parallel', self.args.sequence_parallel)
                        if isinstance(module, to_replaced) and str(module.weight.shape).replace(".", "_") in pipe_model.tied_modules_keys:
                            setattr(model, n, module_func(*get_args(module)))
                        if len(list(module.children())) > 0:
                            replace_module(module, to_replaced, module_func, get_args, sequence_parallel)
                replace_module(pipe_model, torch.nn.Embedding, mpu.VocabParallelEmbedding, 
                                lambda x: (x.weight.size(0), x.weight.size(1), init_method), False)
                if self.args.sequence_parallel:
                    # input_size, output_size, bias=True, gather_output=True,init_method=init.xavier_normal_
                    replace_module(pipe_model, torch.nn.Linear, mpu.ColumnSequenceParallel, 
                                    lambda x: (x.in_features, x.out_features, (x.bias is not None),False,torch.nn.init.xavier_normal_), True)   
                else:    
                    # in_feature,out_feature,bias=bias,gather_output=gather_output,init_method=init_method    
                    replace_module(pipe_model, torch.nn.Linear, mpu.ColumnParallelLinear, 
                                    lambda x: (x.in_features, x.out_features, (x.bias is not None),False,torch.nn.init.xavier_normal_), False)                
                keys_mapping = {str(i).replace(".", "_") : f'torch_Size([{i[0]//self.mp}, {i[1]}])' for i in emb_dim}
                pipe_model.tied_modules_keys = set(keys_mapping.values())
                new_tied_dic = {keys_mapping[i] : pipe_model.tied_stage[i] for i in pipe_model.tied_stage}
                pipe_model.tied_stage = new_tied_dic

            if self.args.tp_overlapping_level > 1:
                first = True
                for n, m in pipe_model.named_modules():
                    if isinstance(m, PipedGPT2Block):
                        last = m
                        if first:
                            m.is_first_layer = True
                            first = False
                last.is_last_layer = True

        if not self.args.no_tie_modules:
            pipe_model.tie_modules()

        if self.args.wall_clock_breakdown and mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0:
            print(dist.get_rank(), pipe_model.stage_id, pipe_model)
        torch.cuda.empty_cache()
        # see_memory_usage('**** \n memory consumption after deepspeep init ', force=True)

        if self.args.wall_clock_breakdown: 
            print_rank_0(pipe_model._topo)
            
        param_groups = get_params_for_weight_decay_optimization(pipe_model)

        # Add model parallel attribute if it is not set.
        for param_group in param_groups:
            for param in param_group['params']:
                if not hasattr(param, 'model_parallel'):
                    param.model_parallel = False
        
        self.model = pipe_model
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        
        deepspeed_config = {
                            "train_micro_batch_size_per_gpu": self.args.per_device_train_batch_size,
                            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                            "steps_per_print": self.args.logging_steps,
                            "gradient_clipping": self.args.max_grad_norm,
                            "wall_clock_breakdown": self.args.wall_clock_breakdown,
                            "prescale_gradients": self.args.prescale_gradients,
                            "gradient_predivide_factor": self.args.gradient_predivide_factor,
                            }

        deepspeed_config = self.amp_config(deepspeed_config)
        deepspeed_config = self.zero_config(deepspeed_config)

        self.pipe_model = PipelineEngine(args=self.args,
                                model=pipe_model,
                                optimizer=self.optimizer,
                                lr_scheduler=self.lr_scheduler,
                                mpu=pipe_model.mpu(),
                                dist_init_required=False,
                                config=deepspeed_config,
                                train_schedule=self.args.train_schedule,
                                return_logits=self.args.return_logits)

        self.optimizer = self.pipe_model.optimizer
        self.lr_scheduler = self.pipe_model.lr_scheduler

        if self.args.split_inputs:
            self.pipe_model.batch_fn = self._prepare_split_inputs
        else:
            self.pipe_model.batch_fn = self._prepare_inputs

        self.pipe_model.input_to_stage_dic = self.input_to_stage_dic
        self.model = self.pipe_model
        self.model.config = hf_config
        
    def training_step(self, iter_dataloader):

        loss = self.pipe_model.train_batch(iter_dataloader)
        return loss.detach()


    def do_prediction(self):
        eval_dataloader = self.get_eval_dataloader()

        # keep training iterator
        train_loader =self.pipe_model.data_iterator
        # create eval iterator
        self.pipe_model.set_dataiterator(eval_dataloader)
        if self.args.eval_iters is not None:
            eval_step = self.args.eval_iters
        else:
            eval_step = len(eval_dataloader)//self.args.gradient_accumulation_steps
        # eval_iterator = iter(eval_dataloader)
        metrics = AccMetric()
        for idx in range(eval_step):
            loss, logits, labels = self.pipe_model.eval_batch()
            metrics.update('eval_loss', loss.item())
            if self.pipe_model.is_last_stage() and self.args.return_logits:
                step_metrics = self.compute_metrics(
                    transformers.trainer_utils.EvalPrediction(predictions=torch.cat(logits).cpu(), 
                    label_ids=torch.cat(labels).cpu())
                    )
                for key in step_metrics:
                    metrics.update(key, step_metrics[key])
            dist.barrier()
        # Restore the training iterator
        self.pipe_model.set_dataiterator(train_loader)
        return metrics.avg



    def _get_train_sampler(self):

        return MegatronPretrainingRandomSampler(
            total_samples=len(self.train_dataset),
            # Set random seed according to be consumed examples, but currently not supported
            consumed_samples=0,
            micro_batch_size=self.args.per_device_train_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())


    def get_train_dataloader(self):
        if self.args.text_generation:
            return range(self.args.per_device_train_batch_size * 100)

        assert self.train_dataset is not None, "Trainer: training requires a train_dataset."

        if isinstance(self.train_dataset, torchvision.datasets.folder.ImageFolder):
            self.data_collator = None

        train_dataset = self.train_dataset

        if self.args.input_names is not None and is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            self._signature_columns = self.args.input_names + ["labels", "label", "label_ids"]
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_sampler = self._get_train_sampler()

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_iter_dataloader(self):
        if self.args.split_inputs:
            from .utils.dataloader import DistributedDataset
            DD = DistributedDataset(self.pipe_model, self.train_dataset, self.input_to_stage_dic, self.data_collator, self.args)
            self.iter_dataloader = DD.get_dataloader()
            self.len_dataset = DD.len_dataset
        else:
            self.iter_dataloader = self.get_train_dataloader()

    def _reset_dataloader(self, epoch):
        if epoch == 0:
            pass
        elif epoch > 0:
            if self.iter_dataloader is not None:
                del self.iter_dataloader
            self._get_iter_dataloader()
        else:
            raise ValueError("Invalid Epoch numbers, unexpected epoch = {}".format(epoch))

        # reset random seed according to epoch
        if self.iter_dataloader and isinstance(self.iter_dataloader, DataLoader) and hasattr(self.iter_dataloader.batch_sampler, 'set_epoch'):
            self.iter_dataloader.batch_sampler.set_epoch(epoch)
        elif self.iter_dataloader: 
            if hasattr(self.iter_dataloader, 'dataset'):
                if isinstance(self.iter_dataloader.dataset, IterableDatasetShard):
                    self.iter_dataloader.dataset.set_epoch(epoch)
        
        if epoch > 0:
            self.pipe_model.reset_dataiterator(self.iter_dataloader)


    def get_eval_dataloader(self):
        if self.eval_dataset is None:
            raise ValueError("Trainer: training requires a eval_dataset.")

        if isinstance(self.eval_dataset, torchvision.datasets.folder.ImageFolder):
            self.data_collator = None
        
        eval_dataset = self.eval_dataset

        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _prepare_inputs(self, data):
        if not isinstance(data, (tuple, list)):
            if isinstance(data, dict):
                inputs = self._prepare_input(data)
                inputs_list = []
                for key, val in self.input_to_stage_dic.items():
                    for i in val:
                        inputs_list.append(inputs.pop(i))
                inputs_list += list(inputs.values())
                return tuple(inputs_list)
            else:
                raise NotImplementedError('only support data in tuple, list or dict')
        else:
            return data

    def _prepare_split_inputs(self, data):
        if data is None:
            return data
        if not isinstance(data, (tuple, list)):
            if isinstance(data, dict):
                data = self._remove_unused_data(data)
                inputs = self._prepare_input(data)
                inputs_list = []
                # 对输入数据按input_to_stage_dic进行排序
                for key, val in self.input_to_stage_dic.items():
                    if self.pipe_model.stage_id == key: 
                        for i in val:
                            inputs_list.append(inputs.pop(i))                        
                inputs_list += list(inputs.values())
                return tuple(inputs_list)
            else:
                raise NotImplementedError('only support data in tuple, list or dict')
        else:
            return data

    def _remove_unused_data(self, data):
        labels_name = ["label", "label_ids", "labels"]

        input_name = []
        if self.pipe_model.is_last_stage():
            for val in self.input_to_stage_dic.values():
                for v in val:
                    input_name.append(v)
            input_name += labels_name

        else:
            input_name = self.input_to_stage_dic[self.pipe_model.stage_id]
            
        ignore_name = list(set(list(data)) - set(input_name))
        for name in ignore_name:
            del data[name]
        return data


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if self.args.process_index == 0:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.do_prediction()
            if self.state.epoch is not None:
                metrics["epoch"] = round(self.state.epoch, 2)
            output = {**metrics, **{"step": self.state.global_step}}
            if self.pipe_model.is_last_stage() and mpu.get_data_parallel_rank() == 0:
                log_dist(output, ranks=[dist.get_rank()])

        if self.control.should_save:
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def get_loss_fn(self, trainer_criterion):
        criterion = trainer_criterion
        def loss_fn(outputs, labels):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(labels, tuple):
                labels = labels[0]
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            return loss
        return loss_fn

    def get_vocab_loss_fn(self):
        def vocab_loss_fn(outputs, labels):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(labels, tuple):
                labels = labels[0]
            losses = mpu.vocab_parallel_cross_entropy(outputs, labels)
            # loss mask??
            loss = torch.sum(losses.view(-1))
            return loss
        return vocab_loss_fn

    def amp_config(self, deepspeed_config):

        if self.args.fp16:
            self.args.half_precision_backend = "auto"
            fp16_var = {
                        "enabled": True,
                        "loss_scale": self.args.loss_scale,
                        "initial_scale_power": self.args.initial_scale_power,
                        "loss_scale_window": self.args.loss_scale_window,
                        "hysteresis": self.args.hysteresis,
                        "min_loss_scale": self.args.min_loss_scale,
            }
            deepspeed_config["fp16"] = fp16_var
        elif self.args.half_precision_backend != "auto":
            if self.args.half_precision_backend == "apex":
                amp_var = {
                        "enabled": True,
                        "opt_level": self.args.fp16_opt_level,
                        }
            elif self.args.half_precision_backend == "amp":
                amp_var = {
                        "enabled": True
                        }
            deepspeed_config["amp"] = amp_var

        return deepspeed_config

    def zero_config(self, deepspeed_config):

        if self.args.zero_optimization:
            zero_var = {
                        "enabled": True,
                        "zero_allow_untested_optimizer": self.args.zero_allow_untested_optimizer,
                        "stage": self.args.zero_stage,
                        "zero_allgather_partition_size": self.args.zero_allgather_bucket_size,
                        "zero_reduce_bucket_size": self.args.zero_reduce_bucket_size,
            }
            deepspeed_config["zero"] = zero_var
        return deepspeed_config

    def load_from_checkpoint(self, resume_from_checkpoint, peft=False):
        if os.path.exists(resume_from_checkpoint):
            if peft:
                load_results = load_peft_model_state_dict(self.pipe_model, self.args, self.peft_config)
                return load_results
            else:
                iteration, state_dict, opt_state_dict = load_checkpoint(self.pipe_model, self.optimizer, self.lr_scheduler, self.args)
            del state_dict, opt_state_dict

        else:
            raise ValueError("Cannot find checkpoint files")

        return iteration

    def save_to_checkpoint(self, best_model=None):
        kwargs = {"best_model": best_model,
                  "args": self.args,
                  "peft_config": self.peft_config
                  }
        save_checkpoint(self.state.global_step, self.pipe_model, self.optimizer, self.lr_scheduler, **kwargs)

    def _sorted_checkpoints(self, output_dir=None, checkpoint_prefix="ckpt"):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob("*_ckpt")]

        for path in glob_checkpoints:
            if not os.listdir(path):
                print_rank_0(f"Deleting empty checkpoint dirctory [{path}]")
                shutil.rmtree(path)
            if "best_ckpt" in path:
                continue
            for x in Path(path).glob("iter_*"):
                ordering_and_checkpoint_path.append(x)

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)

        return checkpoints_sorted

    def _rotate_checkpoints(self, output_dir=None):
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.args.save_total_limit == 1
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]

        # Only rank 0 to delete files.
        if dist.get_rank() == 0:
            for checkpoint in checkpoints_to_be_deleted:
                print_rank_0(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                shutil.rmtree(checkpoint)
        dist.barrier()



# monkey patch for train function
MerakTrainer.train = train




