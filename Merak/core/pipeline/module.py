# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/pipe/module.py

import collections
import os
import re as regex
from typing import Optional, List, Tuple, Callable, Union, Dict

from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_global_rank
from transformers import PretrainedConfig

# from .. import print_rank_0
from . import module_utils
from .layers_partition import LayerPartition
from .. import checkpoint
from .. import mpu
from ..recompute.checkpointing import checkpoint as checkpoint_func
from ..recompute.checkpointing import get_rng_tracker
from ..fx import add_inputs_to_shards, convert_to_sequential
from ..mpu.layers import VocabParallelEmbedding
from ..tensor_parallel import EmbeddingProxy, LinearProxy, ModuleRebuild
from ..finetuning.lora import (
    LoraConfig, _prepare_lora_config,
    mark_only_lora_as_trainable, _find_and_replace
)

from ...merak_args import MerakArguments

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result

class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule """


class PipelineModule(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            args: MerakArguments,
            topology: Callable,
            loss_fn: Optional[Callable] = None,
            seed_layers: bool = False,
            seed_fn: Optional[Callable] = None,
            base_seed: int = 1234,
            communicaiton_grid: Optional[Callable] = None,
            activation_checkpoint_func: Callable = checkpoint_func,
            leaf_modules: list = (),
        ):
        """Modules to be parallelized with pipeline parallelism.

        The key constraint that enables pipeline parallelism is the
        representation of the forward pass as a sequence of layers
        and the enforcement of a simple interface between them. The
        forward pass is implicitly defined by the module ``layers``. The key
        assumption is that the output of each layer can be directly fed as
        input to the next, like a ``torch.nn.Sequence``. The forward pass is
        implicitly:

        .. code-block:: python

            def forward(self, inputs):
                x = inputs
                for layer in self.layers:
                    x = layer(x)
                return x

        Args:
            layers (Iterable): A sequence of layers defining pipeline
            structure. Can be a ``torch.nn.Sequential`` module.
            num_stages (int, optional): The degree of pipeline parallelism. If
            not specified, ``topology`` must be provided.
            topology (``mpu.topology.ProcessTopology``, optional): Defines the
            axes of parallelism axes for training. Must be provided if
            ``num_stages`` is ``None``.
            loss_fn (callable, optional): Loss is computed ``loss = loss_fn
            (outputs, label)``
            base_seed (int, optional): [description]. Defaults to 1234.
            communicaiton_grid  (``mpu.topology.PipelineParallelGrid``,
            optional): Defines the communicators of every parallelism axes for
            training.
            partition_method (str, optional): [description]. Defaults to
            'parameters'.
            activation_checkpoint_interval (int, optional): The granularity
            activation checkpointing in terms of number of layers. 0 disables
            activation checkpointing.
            activation_checkpoint_func (callable, optional): The function to
            use for activation checkpointing. Defaults to ``runtime.
            checkpointing.checkpoint``.
            tie_dims (set(str)): torch shape string of parameters that needs to
            be tied.
            input_to_shard_dic (dict): input name to shard id mapping from
            autoshard.covert func.
        """

        super().__init__()

        self.args = args
        self.loss_fn = loss_fn
        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        self._topo = topology
        self._grid = communicaiton_grid
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f'SEED_LAYERS={self.seed_layers}'
                f'BASE_SEED={self.base_seed} SEED_FN={seed_str}'
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", None))
        assert self.local_rank != None

        self.sequence_parallel = self.args.sequence_parallel

        self.stage_id = self._topo.get_coord(self.global_rank).pipe
        self.num_stages = self._topo.get_dim('pipe')

        self.forward_funcs = []
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}
        self.tied_comms = {}

        self.micro_offset = 0
        self._local_start = 0
        self._local_stop = None

        # trace model
        model_class = model.__class__
        layers, input_to_shard_dic = self._traced_module(model, leaf_modules)

        # set tie module
        tie_dims = self._set_tie_dim(model)

        # rebuild module for tensor parallel
        rebuild_module = ModuleRebuild(self.args, layers, model_class)
        if dist.get_rank() == 0:
            print(layers)

        # Initialize partition information
        self._layer_specs = layers if isinstance(layers, list) else list(layers)
        self._num_layers = len(self._layer_specs)

        # layer partition
        self.split_method = LayerPartition(
            args, self._layer_specs,
            topology, self.global_rank
        )
        self._partition_layers(input_to_shard_dic=input_to_shard_dic)

        # set recompute config
        self.act_ckpt_interval = self.args.checkpoint_num_layers
        self.act_ckpt_func = activation_checkpoint_func
        self.act_ckpt_ratio = self.args.activation_checkpoint_ratio
        if self.act_ckpt_ratio is not None:
            if len(self.act_ckpt_ratio) == 1:
                first_ratio = 1 - float(self.act_ckpt_ratio[0])
                self.act_ckpt_ratio = [
                    1 - (
                        first_ratio * (self.num_stages - 1) / (self.num_stages - s)
                    )
                    for s in range(1,self.num_stages)
                ] + [0]
                # print_rank_0(f'activation checkpoint ratio list:'
                #              f'{self.act_ckpt_ratio}')
                if self.act_ckpt_ratio[self.stage_id] <= 0:
                    self.act_ckpt_interval = 0

            elif len(self.act_ckpt_ratio) < self.num_stages:
                last_ratio = self.act_ckpt_ratio[-1]
                self.act_ckpt_ratio += [last_ratio] * (
                                        self.num_stages - len(self.act_ckpt_ratio)
                                    )
        
        self._build(tie_dims, input_to_shard_dic)

        if model.config.model_type == "gpt2":
            assert mpu.get_model_parallel_world_size() == 1, (
                  "Currently gpt model not supported in tensor parallel"
                )

        rebuild_module.recover_module(self)
        rebuild_module.vocab_parallel(emb_dim=tie_dims)

        del model, layers

    def _traced_module(
            self,
            module: nn.Module,
            leaf_modules: List[nn.Module]
        ) -> Tuple[Union[List[torch.fx.GraphModule], Dict[str, int]]]:
        model, layers, input_to_shard = convert_to_sequential(
            module, self.args,
            extra_leaf_modules=leaf_modules,
        )
        del model
        return layers, input_to_shard

    def _add_lora_layers(
            self,
            model_config: Union[PretrainedConfig, dict]
        ):
        assert mpu.get_model_parallel_world_size() == 1, \
            "Don't support tensor parallelism for lora, currently"
        peft_config = LoraConfig(**self.args.get_lora_config())
        peft_config = _prepare_lora_config(peft_config, model_config.to_dict())
        _find_and_replace(self, adapter_name=self.args.adapter_name,
                            config=peft_config)
        mark_only_lora_as_trainable(self, peft_config)

        module_utils.print_trainable_parameters(self)
        return peft_config

    def _build(self, tie_dims: set, input_to_shard_dic: Dict[str, int]):
        specs = self._layer_specs
        self.tied_modules_keys = set(str(s).replace(".", "_") for s in tie_dims)
        self.tied_stage = collections.defaultdict(set)

        self.input_to_stage_dic = collections.defaultdict(list)
        for input_name, shard_id in input_to_shard_dic.items():
            self.input_to_stage_dic[self.stage_owner(shard_id)].append(input_name)

        for layer_idx, layer in enumerate(specs):
            for m in layer.modules():
                if hasattr(m, 'weight'):
                    if isinstance(m, (EmbeddingProxy, LinearProxy)):
                        weight_shape = m.weight_shape
                    else:
                        try:
                            weight_shape = m.weight.shape
                        except AttributeError:
                            # loss module pass
                            continue
                    if weight_shape in tie_dims:
                        self.tied_stage[
                            str(weight_shape).replace(".", "_")].add(
                                self.stage_owner(layer_idx))


        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                # Offset the random seed by the stage ID.
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    module_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            elif isinstance(layer, nn.Module):
                name = str(layer_idx)

                inputs_of_this_stage = []
                for input in self.input_to_stage_dic[self.stage_id]:
                    if layer_idx < input_to_shard_dic[input]:
                        inputs_of_this_stage.append(input)
                # print(layer_idx, inputs_of_this_stage)
                if inputs_of_this_stage:
                    layer = add_inputs_to_shards(layer, inputs_of_this_stage)
                    # print(layer.code)
                self.forward_funcs.append(layer)
                self.add_module(name, layer)

            else:
                self.forward_funcs.append(layer)
                
        num_layers = len(self.forward_funcs)
        if self.act_ckpt_ratio is not None and float(self.act_ckpt_ratio[self.stage_id]) != 1.0:
            self.checkpointable_num_layers = 0
            self.checkpointable_idx = []
            # self.no_checkpointable_idx = []
            prev_checkpointable = None
            for start_idx in range(0, num_layers):
                end_idx = start_idx + 1
                funcs = self.forward_funcs[start_idx:end_idx]
                if self._is_checkpointable(funcs):
                    if prev_checkpointable:
                        self.checkpointable_idx[-1][1] = end_idx
                    else:
                        if prev_checkpointable is None:
                            self.frist_checkpointable = True
                        self.checkpointable_idx.append([start_idx, end_idx])
                    self.checkpointable_num_layers += 1
                    prev_checkpointable = True
                else:
                    if prev_checkpointable is None or \
                        prev_checkpointable == True:
                        if prev_checkpointable is None:
                            self.frist_checkpointable = False
                        self.checkpointable_idx.append([start_idx, end_idx])
                    else:
                        self.checkpointable_idx[-1][1] = end_idx
                    prev_checkpointable = False
        # All pipeline parameters should be considered as model parallel in the 
        # context of our FP16 optimizer
        for p in self.parameters():
            p.model_parallel = True

    def _broadcast_model(self):

        def is_replicated(p):
            if hasattr(p, 'ds_status'):
                return False
            return True

        broadcast_src_rank = _get_global_rank(mpu.get_data_parallel_group(), 0)

        for p in self.parameters():
            if torch.is_tensor(p) and is_replicated(p):
                dist.broadcast(p,
                               broadcast_src_rank,
                               group=mpu.get_data_parallel_group())

    def configure_distributed_model(self, device: torch.device):

        if self.args.fp16:
            shuold_dtype = torch.half
        else:
            shuold_dtype = torch.float
        if not all(
            [param.dtype == shuold_dtype for param in self.parameters()]):
            names = [
                n for n,
                p in self.named_parameters() if p.dtype != shuold_dtype
            ]
            raise ValueError(
                f"Current model should be {str(shuold_dtype)} but the following parameters have "
                f"dtype that is not {str(shuold_dtype)}: {', '.join(names)}"
            )

        self.to(device)

        if not self.args.half_precision_backend == "apex":
            self._broadcast_model()

    def forward(
            self,
            forward_input: Union[List[torch.Tensor], Tuple[torch.Tensor]]
        ) -> Tuple[torch.Tensor]:
        # We need to offset the seed by the microbatch ID. Save it in a local 
        # var to ensure it is preserved in the closure. Otherwise checkpointed 
        # forward funcs will see a different offset.
        self.micro_offset += 1

        def exec_range_func(start: int, end: int):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:# and not isinstance(inputs[0], tuple):
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed *
                                    local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            module_utils.set_random_seed(new_seed)
                    if isinstance(inputs, tuple):
                        inputs = layer(*inputs)
                    else:
                        inputs = layer(inputs)
                return inputs

            return exec_func
        if self.sequence_parallel:
            rng_context = get_rng_tracker().fork()
        else:
            rng_context = nullcontext()
            
        with rng_context:
            if self.act_ckpt_interval == 0:
                func = exec_range_func(0, len(self.forward_funcs))
                x = func(forward_input)
            elif self.act_ckpt_ratio is not None and \
                float(self.act_ckpt_ratio[self.stage_id]) != 1.0:
                # if self.stage_id == 0:
                #     self.act_ckpt_ratio = 0.6
                ac_num_layers = int(len(self.forward_funcs) * \
                        float(self.act_ckpt_ratio[self.stage_id]))
                
                ## a naive implement
                non_ac_layers = len(self.forward_funcs) - ac_num_layers
                x = forward_input
                if not isinstance(x, tuple):
                    x = (x, )
                x = exec_range_func(0, non_ac_layers)(*x)
                if not isinstance(x, tuple):
                    x = (x, )
                x = self.act_ckpt_func(
                        exec_range_func(non_ac_layers, 
                                        len(self.forward_funcs)),
                                        *x)

                next_checkpointable = self.frist_checkpointable
                x = forward_input
                for start_idx, end_idx in self.checkpointable_idx:
                    if next_checkpointable:
                        if not isinstance(x, tuple):
                            x = (x, )
                        if ac_num_layers <= 0:
                            x = exec_range_func(start_idx, end_idx)(*x)
                        else:
                            layer_num = end_idx - start_idx
                            if ac_num_layers >= layer_num:
                                x = self.act_ckpt_func(
                                            exec_range_func(start_idx, end_idx),
                                            *x)
                            else:
                                x = self.act_ckpt_func(
                                            exec_range_func(start_idx,
                                                    start_idx+ac_num_layers),
                                                    *x)
                                if not isinstance(x, tuple):
                                    x = (x, )
                                x = exec_range_func(
                                    start_idx+ac_num_layers, end_idx
                                    )(*x)
                            ac_num_layers -=layer_num
                    else:
                        if not isinstance(x, tuple):
                            x = (x, )
                        x = exec_range_func(start_idx, end_idx)(*x)
                    next_checkpointable = not next_checkpointable
            else:
                num_layers = len(self.forward_funcs)
                x = forward_input
                for start_idx in range(0, 
                                       num_layers, 
                                       self.act_ckpt_interval):
                    end_idx = min(start_idx + \
                                  self.act_ckpt_interval,
                                num_layers)

                    funcs = self.forward_funcs[start_idx:end_idx]
                    # Since we either pass tensors or tuples of tensors without 
                    # unpacking, weneed to be careful not to double-wrap 
                    # tensors with tuple.
                    if not isinstance(x, tuple):
                        x = (x, )
                    if self._is_checkpointable(funcs) and \
                        (self.stage_id < self.num_stages-1 or \
                         end_idx !=num_layers):
                        # print('checkpoint', 
                        # self.stage_id,self.forward_funcs[start_idx:end_idx])
                        x = self.act_ckpt_func(
                            exec_range_func(start_idx,
                                            end_idx),
                            *x)
                    else:
                        # print('no checkpoint',
                        # self.stage_id,self.forward_funcs[start_idx:end_idx])
                        x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_layers(self, input_to_shard_dic: Optional[Dict[str, int]] = None):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        self.parts = self.split_method.layer_split_by_index(input_to_shard_dic)

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_bounds(start=self.parts[stage_id], 
                         stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        '''All reduce the gradients of the tied weights between tied stages'''
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm['weight_attr'])
            dist.all_reduce(weight.grad, group=comm['group'])

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm['module'],
                        comm['weight_attr']),
                src=min(comm['ranks']),
                group=comm['group'],
            )

    def _set_tie_dim(self, model: nn.Module) -> int:
        emb_dim = set()
        def add_dim(emb_dim, m):
            if isinstance(m, (EmbeddingProxy, LinearProxy)):
                emb_dim.add(m.weight_shape)
            else:
                emb_dim.add(m.weight.shape)
        if hasattr(model, 'config'):
            if model.config.tie_word_embeddings:
                for m in model.modules():
                    try:
                        if hasattr(m, 'get_input_embeddings') and \
                            m.get_input_embeddings() is not None:
                            add_dim(emb_dim, m.get_input_embeddings())
                        if hasattr(m, 'get_output_embeddings') and \
                            m.get_output_embeddings() is not None:
                            add_dim(emb_dim, m.get_output_embeddings())
                    except (AttributeError, NotImplementedError):
                        continue
        elif hasattr(model, 'get_input_embeddings'):
            add_dim(emb_dim, model.get_input_embeddings())
        elif hasattr(model, 'get_output_embeddings'):
            add_dim(emb_dim, model.get_output_embeddings())
        return emb_dim

    def tie_modules(self):
        ''' Build communication structures for tied modules. '''
        # for layer_idx, layer in enumerate(specs):

        def get_name(name):
            return str(name).replace(".", "_")

        for m in self.modules():
            if hasattr(m, 'weight'):
                if m.weight is not None:
                    if hasattr(m.weight, 'shape') and \
                        get_name(m.weight.shape) in self.tied_modules_keys:
                        if get_name(m.weight.shape) in self.tied_modules:
                            # print(f'module {m} ties to exsiting '
                            #  f'{self.tied_modules[get_name(m.weight.shape)]}')
                            m.weight = self.tied_modules[
                                get_name(m.weight.shape)
                            ].weight
                        else:
                            # print(f'module {m} needs '
                            #       f'tied to key {get_name(m.weight.shape)}')
                            self.tied_modules[get_name(m.weight.shape)] = m
                            self.tied_weight_attrs[get_name(m.weight.shape)] = \
                                'weight'


        tied_comms = {}
        if self._topo.get_dim('pipe') == 1:
            return
        for key in self.tied_modules_keys:
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(self.tied_stage[key]):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp,
                                                           model=mp))
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s,
                                                           data=dp))
                    group = dist.new_group(ranks=tied_ranks)

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                'ranks': tied_ranks,
                                'group': group,
                                'weight_attr': self.tied_weight_attrs[key],
                                'module': self.tied_modules[key],
                            }
        self.tied_comms = tied_comms
        # print(self.tied_comms)
        self._synchronize_tied_weights()


    def partitions(self) -> List[int]:
        return self.parts

    def stage_owner(self, layer_idx: int):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim('pipe')):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f'Layer {layer_idx} not owned? parts={self.parts}')

    def _set_bounds(self, start: int = None, stop: int = None):
        """Manually define the range of layers that will be built on this
        process.

        These boundaries are treated as list slices and so start is inclusive
        and stop is exclusive. The default of None for both results in all
        layers being built locally.
        """
        self._local_start = start
        self._local_stop = stop

    def topology(self) -> Callable:
        """ ProcessTopology object to query process mappings. """
        return self._topo

    def num_pipeline_stages(self) -> int:
        return self._topo.get_dim('pipe')

    def _is_checkpointable(self, funcs: nn.Module) -> bool:
        for f in funcs:
            if isinstance(f, torch.fx.GraphModule):
                for n, m in f.named_modules():
                    if isinstance(m, (torch.nn.Embedding,
                                      VocabParallelEmbedding)):
                        return False
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
