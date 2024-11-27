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


import torch
import torch.nn as nn
import torch.fx
from typing import List, Tuple, Callable

from transformers.pytorch_utils import Conv1D

from Merak import print_rank_0
from .mp_attrs import set_mp_attr, mp_is_setted, set_tp_layer_lists
from .utils import (
    tp_overlapping_available, init_method_normal,
    reset_module_tensor, scaled_init_method_normal
)
from .mp_layers import ColPara, RowPara, ConvPara
from .transformer_blocks import PipedGPT2Model, PipedGPT2Block
from .. import mpu
from ..pipeline.module_utils import set_random_seed

class ModuleRebuild:
    '''
    rebuild model to support model tensor parallel.
    '''

    def __init__(self, args, model: List[torch.fx.GraphModule], model_class: Callable):
        self.args = args
        # self.device = model.device
        if not isinstance(model, list):
            model = [model]
        self.model = model
        self.model_class = model_class #self.model.__class__
        self.mp_size = mpu.get_model_parallel_world_size()
        self.init_method = init_method_normal(self.args.init_method_std)
        
        set_random_seed(self.args.seed)
        if self.args.tp_overlapping_level > 1:
            self.overlap_init()
        # self.replace_module()
        # self.model = self.model.to(self.device)
        if self.mp_size > 1:
            self._set_attr()

    def _set_attr(self):
        if not mp_is_setted():
            from .mp_mapping import get_mp_layer_lists
            mp_layer_lists = get_mp_layer_lists(self.model_class)
            if mp_layer_lists is not None:
                set_tp_layer_lists(**mp_layer_lists)
        assert mp_is_setted(), \
        f'''
        model {self.model_class.__name__} is not supported by auto tp now,
        should set tp attr manually with set_tp_layer_lists
        '''
        for model in self.model:
            set_mp_attr(model, self.mp_size)
        # self.model = set_mp_attr(self.model, self.mp_size)

    def build_parallel_layers(self, n, module, init_args):
        if isinstance(module, nn.Linear) or n in ['c_attn', 'c_proj', 'c_fc']:
            _bias = module.bias if isinstance(module.bias, bool) else isinstance(module.bias, torch.Tensor)
            if not hasattr(module, 'mp_attr'):
                return module
            elif module.mp_attr.startswith('row'):
                module_args = [module.in_features * self.mp_size, module.out_features]
                init_method = scaled_init_method_normal(*init_args)
                if module.mp_attr == 'row_mlp':
                    return RowPara(module_args[0], module_args[1],
                                init_method, bias=_bias)
                else:
                    return RowPara(module_args[0], module_args[1],
                                init_method, bias=_bias, need_permute=False)
            elif module.mp_attr.startswith('col'):
                module_args = [module.in_features, module.out_features * self.mp_size]
                init_method = init_method_normal(init_args[0])
                if module.mp_attr == 'col_mlp':
                    return ColPara(module_args[0], module_args[1],
                                init_method, bias=_bias)
                else:
                    return ColPara(module_args[0], module_args[1],
                                init_method, bias=_bias, need_permute=False)
        elif isinstance(module, nn.Conv2d):
            module_args = module.__dict__
            if not hasattr(module, 'mp_attr'):
                return module
            return ConvPara(**module_args)
        else:
            return None

    def recover_module(self, module):
        self.model = module
        def build_module(model, init_args):
            for n, module in model.named_children():
                # if isinstance(module, layer_cls):
                parallel_layer = self.build_parallel_layers(n, module, init_args)
                if parallel_layer is not None:
                    setattr(model, n, parallel_layer)
                if len(list(module.children())) > 0:
                    # compound module, go inside it
                    build_module(module, init_args)

        build_module(self.model, (self.args.init_method_std, self.args.num_layers))

        # reset parameters if the device is meta
        for param_name, param in self.model.named_parameters():
            normal_param = torch.empty(param.shape).normal_(mean=0.0,
                                                            std=self.args.init_method_std)
            if str(param.device) == 'meta':
                lower_name = param_name.lower()
                if ('norm' in lower_name or 'ln' in lower_name) and lower_name.endswith('weight'):
                    reset_module_tensor(
                        self.model, param_name, torch.device('cpu'), torch.ones(param.shape)
                    )
                elif 'embed' in lower_name:
                    reset_module_tensor(
                        self.model, param_name, torch.device('cpu'), normal_param
                    )
                else:
                    reset_module_tensor(
                        self.model, param_name, torch.device('cpu'), torch.zeros(param.shape)
                    )
        for buffer_name, buffer in self.model.named_buffers():
            if str(buffer.device) == 'meta':
                reset_module_tensor(
                    self.model, buffer_name, torch.device('cpu'), torch.rand(buffer.shape)
                )

        # return self.model

    def sequence_parallel(self):
        if self.args.sequence_parallel:
            mpu.mappings.set_sequence_dim(self.args.sequence_dim)

        return self.model

    def vocab_parallel(self, emb_dim: int):
        if self.args.parallel_vocab:
            # replace module to VocabParallelEmbedding and column parallel
            def replace_module(model,to_replaced,
                               module_func, get_args, sequence_parallel):
                for n, module in model.named_children():
                    if sequence_parallel and \
                       isinstance(module, torch.nn.LayerNorm):
                        setattr(module.weight, 'sequence_parallel',
                                self.args.sequence_parallel)
                        setattr(module.bias, 'sequence_parallel',
                                self.args.sequence_parallel)

                    if isinstance(module, to_replaced) and \
                        str(module.weight.shape).replace(".", "_") \
                            in self.model.tied_modules_keys:
                        setattr(model, n, module_func(*get_args(module)))

                    if len(list(module.children())) > 0:
                        replace_module(module, to_replaced, 
                                       module_func, get_args, sequence_parallel)

            replace_module(
                self.model,
                torch.nn.Embedding,
                mpu.VocabParallelEmbedding,
                lambda x: (
                    x.weight.size(0),
                    x.weight.size(1),
                    self.init_method
                ),
                False
            )

            if self.args.sequence_parallel:
                # input_size, output_size, bias=True, gather_output=True,\
                # init_method=init.xavier_normal_
                replace_module(
                    self.model,
                    torch.nn.Linear,
                    mpu.ColumnSequenceParallel,
                    lambda x: (
                        x.in_features,
                        x.out_features,
                        (x.bias is not None),
                        False,
                        torch.nn.init.xavier_normal_
                    ),
                    True
                )
            else:
                # in_feature,out_feature,bias=bias,gather_output=gather_output,\
                # init_method=init_method
                replace_module(
                    self.model,
                    torch.nn.Linear,
                    mpu.ColumnParallelLinear,
                    lambda x: (
                        x.in_features,
                        x.out_features,
                        (
                            x.bias is not None
                        ),
                        False,
                        torch.nn.init.xavier_normal_
                    ),
                    False
                )
            keys_mapping = {
                str(i).replace(".", "_"):
                f'torch_Size([{i[0]//self.mp_size}, {i[1]}])'
                for i in emb_dim
            }
            self.model.tied_modules_keys = set(keys_mapping.values())
            new_tied_dic = {
                keys_mapping[i]:
                self.model.tied_stage[i]
                for i in self.model.tied_stage
            }
            self.model.tied_stage = new_tied_dic

        if self.args.tp_overlapping_level > 1:
            first = True
            for n, m in self.model.named_modules():
                if isinstance(m, PipedGPT2Block):
                    last = m
                    if first:
                        m.is_first_layer = True
                        first = False
            last.is_last_layer = True

    def overlap_init(self):
        if not tp_overlapping_available(self.model_class):
            message = (f'not support tp overlapping level '
                       f'{self.args.tp_overlapping_level} in model '
                       f'{self.model_class}, will reset the level to 1')
            print_rank_0(message)
            self.args.tp_overlapping_level = 1
        else:
            assert self.model.config.use_cache == False
            assert self.model.config.output_attentions == False
            assert self.model.config.output_hidden_states == False
            assert self.model.config.add_cross_attention == False
            self.model.transformer = PipedGPT2Model(self.model.config)