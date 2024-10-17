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

import math
import torch
import transformers
import torch.nn as nn
from typing import Tuple, Union, Callable

from transformers.pytorch_utils import Conv1D

from Merak.merak_args import get_args
from Merak import print_rank_0
from .mp_layers import ColPara, RowPara, SequenceParallelEmb
from .utils import init_method_normal, scaled_init_method_normal
from ..mpu import get_model_parallel_world_size

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
        self.weight_shape = torch.Size((self.out_features, self.in_features))


    def build(self, init_args: Tuple[Union[float, Union[int, dict]]], fp16: bool):
        if self.mp_attr == ' ':
            if fp16:
                return torch.nn.Linear(*self.module_args).half()
            else:
                return torch.nn.Linear(*self.module_args)
        if self.mp_attr.startswith('row'):
            init_method = scaled_init_method_normal(*init_args)
            if self.mp_attr == 'row_mlp':
                return RowPara(self.module_args[0], self.module_args[1],
                               init_method, bias=self._bias)
            else:
                return RowPara(self.module_args[0], self.module_args[1],
                               init_method, bias=self._bias, need_permute=False)
        elif self.mp_attr.startswith('col'):
            init_method = init_method_normal(init_args[0])
            if self.mp_attr == 'col_mlp':
                return ColPara(self.module_args[0], self.module_args[1],
                               init_method, bias=self._bias)
            else:
                return ColPara(self.module_args[0], self.module_args[1],
                               init_method, bias=self._bias, need_permute=False)
        
        raise NotImplementedError(f"Not supported model/tensor \
                                  parallelism layers {self.mp_attr}")
        

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self._bias
        )


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
        # self.mp_size = get_model_parallel_world_size()
        # w = torch.empty(1, 1)
        # self.weight = NumelParameter(w)
        # self.weight.num_element = lambda: self.out_features*self.in_features
        # self.bias = NumelParameter(torch.zeros(1))
        # self.bias.num_element = lambda: self.out_features
    
    def build(self, init_args: Tuple[Union[float, Union[int, dict]]], fp16: bool):
        if self.mp_attr == ' ':
            if fp16:
                return Conv1D(*self.module_args).half()
            else:
                return Conv1D(*self.module_args)
        if self.mp_attr.startswith('row'):
            init_method = scaled_init_method_normal(*init_args)
            if self.mp_attr == 'row_mlp':
                return RowPara(self.module_args[1],
                               self.module_args[0], init_method)
            else:
                return RowPara(self.module_args[1], self.module_args[0],
                               init_method)#, need_permute=True)
        elif self.mp_attr.startswith('col'):
            init_method = init_method_normal(init_args[0])
            if self.mp_attr == 'col_mlp':
                return ColPara(self.module_args[1],
                               self.module_args[0], init_method)
            else:
                return ColPara(self.module_args[1], self.module_args[0],
                               init_method)#, need_permute=True)
        
        raise NotImplementedError(f"Not supported model/tensor \
                                  parallelism layers {self.mp_attr}")

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

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
        w = torch.empty(self.num_embeddings, self.embedding_dim)
        self.weight = NumelParameter(w)
        self.weight.num_element = lambda: self.num_embeddings*self.embedding_dim
        self.weight_shape = torch.Size((self.num_embeddings,
                                        self.embedding_dim))
    
    def build(self, init_args: Tuple[Union[bool, Callable]], fp16: bool):
        sequence_parallel, init_method = init_args
        self.module_args = (self.num_embeddings, self.embedding_dim)
        if sequence_parallel:
            return SequenceParallelEmb(*self.module_args)
        if fp16:
            module = torch.nn.Embedding(*self.module_args).half()
        else:
            module = torch.nn.Embedding(*self.module_args)
        init_method(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
        return module

    def extra_repr(self) -> str:
        return '{}, {}'.format(
            self.num_embeddings, self.embedding_dim
        )
