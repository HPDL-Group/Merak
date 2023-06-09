# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com)
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
import torch.distributed as dist
from .. import print_rank_0
from ..mpu import get_model_parallel_world_size
import transformers

from .utils import init_method_normal, scaled_init_method_normal
from ..utils import get_args
from .mp_layers import ColPara, RowPara, SequenceParallelEmb
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math

SHOULD_PRINT_CONV = True
SHOULD_PRINT_LINEAR = True
SHOULD_PRINT_EMB = True


class NumelParameter(nn.Parameter):
    def numel(self):
        return self.num_element()


class LinearProxy(nn.Module):
    def __init__(self, *module_args, **module_kwargs):
        self.module_args = module_args
        if len(module_kwargs) != 0:
            for k in module_kwargs:
                self.module_args += (module_kwargs[k],)
        super(LinearProxy, self).__init__()
        global SHOULD_PRINT_LINEAR
        if SHOULD_PRINT_LINEAR:
            print_rank_0('using linear proxy')
            SHOULD_PRINT_LINEAR = False
        self.in_features = self.module_args[0]
        self.out_features = self.module_args[1]
        if len(self.module_args) >= 3:
            self._bias = self.module_args[2]
        else:
            self._bias = False

        self.mp_attr = ' '
        self.mp_size = get_model_parallel_world_size()

        self.__flops__ = 0

        w = torch.empty(1, 1)
        self.weight = NumelParameter(w)
        self.weight.num_element = lambda: self.out_features*self.in_features

        if self._bias:
            self.bias = NumelParameter(torch.zeros(1))
            self.bias.num_element = lambda: self.out_features
        else:
            self.register_parameter('bias', None)
        self.weight_shape = torch.Size((self.out_features, self.in_features))


    def build(self, init_args, fp16):
        if self.mp_attr == ' ':
            if fp16:
                return torch.nn.Linear(*self.module_args).cuda().half()
            else:
                return torch.nn.Linear(*self.module_args).cuda()
        if self.mp_attr.startswith('row'):
            init_method = scaled_init_method_normal(*init_args)
            if self.mp_attr == 'row_mlp':
                return RowPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias)
            else:
                return RowPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias, need_permute=False)
        elif self.mp_attr.startswith('col'):
            init_method = init_method_normal(init_args[0])
            if self.mp_attr == 'col_mlp':
                return ColPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias)
            else:
                return ColPara(self.module_args[0], self.module_args[1], init_method, bias=self._bias, need_permute=False)
        
        raise NotImplementedError(f"Not supported model/tensor parallelism layers {self.mp_attr}")
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def forward(self, x):
        # shape = x.shape
        # shape[-1] = self.out_features
        # if len(x.shape) == 2:
        #     return x[:, :1].expand(-1, self.out_features).contiguous()
        self.__flops__ = 2 * x.numel() * self.out_features
        args = get_args()
        if args.sequence_parallel:
            if self.mp_attr.startswith('row'):
                x = x.chunk(self.mp_size, dim=args.sequence_dim)[0].contiguous()
            elif self.mp_attr.startswith('col'):
                x = torch.cat([x]*self.mp_size, dim=args.sequence_dim).contiguous()
        try:
            return x[:, :, :1].expand(-1,-1, self.out_features).contiguous()#, device=x.device)
        except IndexError:
            return x[:, :1].expand(-1, self.out_features).contiguous()


class Conv1DProxy(nn.Module):
    def __init__(self, out_features, in_features):
        self.module_args = (out_features, in_features)
        super(Conv1DProxy, self).__init__()
        global SHOULD_PRINT_CONV
        if SHOULD_PRINT_CONV:
            print_rank_0('using conv1d proxy')
            SHOULD_PRINT_CONV = False
        self.in_features = in_features
        self.out_features = out_features

        self.mp_attr = ' '
        self.mp_size = get_model_parallel_world_size()
        w = torch.empty(1, 1)
        self.weight = NumelParameter(w)
        self.weight.num_element = lambda: self.out_features*self.in_features
        self.bias = NumelParameter(torch.zeros(1))
        self.bias.num_element = lambda: self.out_features
    
    def build(self, init_args, fp16):
        if self.mp_attr == ' ':
            if fp16:
                return transformers.modeling_utils.Conv1D(*self.module_args).cuda().half()
            else:
                return transformers.modeling_utils.Conv1D(*self.module_args).cuda()
        if self.mp_attr.startswith('row'):
            init_method = scaled_init_method_normal(*init_args)
            if self.mp_attr == 'row_mlp':
                return RowPara(self.module_args[1], self.module_args[0], init_method)
            else:
                return RowPara(self.module_args[1], self.module_args[0], init_method)#, need_permute=True)
        elif self.mp_attr.startswith('col'):
            init_method = init_method_normal(init_args[0])
            if self.mp_attr == 'col_mlp':
                return ColPara(self.module_args[1], self.module_args[0], init_method)
            else:
                return ColPara(self.module_args[1], self.module_args[0], init_method)#, need_permute=True)
        
        raise NotImplementedError(f"Not supported model/tensor parallelism layers {self.mp_attr}")

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    def forward(self, x):
        self.__flops__ = 2 * x.numel() * self.out_features
        args = get_args()
        if args.sequence_parallel:
            if self.mp_attr.startswith('row'):
                x = x.chunk(self.mp_size, dim=args.sequence_dim)[0].contiguous()
            elif self.mp_attr.startswith('col'):
                x = torch.cat([x]*self.mp_size, dim=args.sequence_dim).contiguous()
            else:
                assert False, 'sequence parallel should use with tp > 1'
        return x[:, :, :1].expand(-1, -1, self.out_features)


class EmbeddingProxy(nn.Module):
    def __init__(self, *module_args, **module_kwargs):
        self.module_args = module_args
        if len(module_kwargs) != 0:
            for k in module_kwargs:
                self.module_args += (module_kwargs[k],)
        super(EmbeddingProxy, self).__init__()
        global SHOULD_PRINT_EMB
        if SHOULD_PRINT_EMB:
            print_rank_0('using embedding proxy')
            SHOULD_PRINT_EMB = False
        self.num_embeddings = self.module_args[0]
        self.embedding_dim = self.module_args[1]
        self.padding_idx = None


        self.mp_attr = ' '
        self.mp_size = get_model_parallel_world_size()
        w = torch.empty(1, 1)
        self.weight = NumelParameter(w)
        self.weight.num_element = lambda: self.num_embeddings*self.embedding_dim
        self.weight_shape = torch.Size((self.num_embeddings, self.embedding_dim))
    
    def build(self, init_args, fp16):
        sequence_parallel, init_method = init_args
        if sequence_parallel:
            return SequenceParallelEmb(*self.module_args).cuda()
        if fp16:
            module = torch.nn.Embedding(*self.module_args).cuda().half()
        else:
            module = torch.nn.Embedding(*self.module_args).cuda()
        init_method(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
        return module

    def extra_repr(self) -> str:
        return '{}, {}'.format(
            self.num_embeddings, self.embedding_dim
        )
    
    def forward(self, x):
        self.__flops__ = 0
        args = get_args()
        if args.sequence_parallel:
            try:
                x = x.float().unsqueeze(-1).expand(-1,-1, self.embedding_dim)
            except RuntimeError:
                x = x.float().unsqueeze(-1).expand(-1, self.embedding_dim)
            x = x.chunk(self.mp_size, dim=args.sequence_dim)[0].contiguous()
            return x
        if args.fp16 or args.half_precision_backend == "apex":
            try:
                x = x.half().unsqueeze(-1).expand(-1,-1, self.embedding_dim)
            except RuntimeError:
                x = x.half().unsqueeze(-1).expand(-1, self.embedding_dim)
            return x
        try:
            return x.float().unsqueeze(-1).expand(-1,-1, self.embedding_dim).contiguous()#, device=x.device)
        except RuntimeError:
            return x.float().unsqueeze(-1).expand(-1, self.embedding_dim).contiguous()
