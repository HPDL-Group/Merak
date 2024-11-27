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
from typing import Optional, List, Tuple
from transformers.modeling_utils import Conv1D

MP_LAYER_LIST_SETTED = False

DEFAULT_MP_ATTR = ['num_heads', 'split_size',  # GPT
                'n_heads', 'inner_dim']  # T5

# list of linear layer for column parallel style
COL_PARA_LIST = []
# list of linear layer for row parallel style
ROW_PARA_LIST = []
# list of conv2d layer for channels parallel style
CONV_PARA_ATTR = []
# list of (layer name, tp dimension 1 or 0), 1 for column style and 0 for row style
MP_WEIGHT_LIST = []
# manual tp attr list for each layer, each will be divided by tp number
MANUAL_MP_ATTR = []
# ratio between input and output of linear layer to indicate the tp style, 
# list of (input, output, tp style)
INPUT_OUTPUT_MAPPING = []

def set_mp_attr(model: torch.nn.Module, mp: int) -> torch.nn.Module:
    if COL_PARA_LIST:
        for n, m in model.named_modules():
            for col_name in COL_PARA_LIST:
                if col_name in n and isinstance(m, nn.Linear):
                    # m.mp_attr = 'col'
                    setattr(m, 'mp_attr', 'col')
                    m.out_features //= mp
    if ROW_PARA_LIST:
        for n, m in model.named_modules():
            for row_name in ROW_PARA_LIST:
                if row_name in n and isinstance(m, nn.Linear):
                    setattr(m, 'mp_attr', 'row')
                    m.in_features //= mp
    if MP_WEIGHT_LIST:
        for n, m in model.named_modules():
            for (l_name, d) in MP_WEIGHT_LIST:
                if l_name in n:
                    if not isinstance(m, nn.Linear):

                        # for case that changes m.lname.weigth 
                        if hasattr(m, 'weight'):
                            m.weight.data = \
                                m.weight.data.chunk(mp,dim=d)[0].contiguous()
                            if hasattr(m, 'num_embeddings'):
                                m.num_embeddings = m.weight.size()[0]
                                m.embedding_dim = m.weight.size()[1]
                        else:
                            pass
       
                    # for case that changes Conv1DProxy, LinearProxy
                    else:
                        if d == 1:
                            setattr(m, 'mp_attr', 'col')
                            m.out_features //= mp
                        elif d == 0:
                            setattr(m, 'mp_attr', 'row')
                            m.in_features //= mp
                        else:
                            assert False, 'dim should be 0 or 1'

                # for case that changes m.lname
                elif hasattr(m, l_name):
                    old_attr = getattr(m, l_name)
                    # if m.lname is torch.nn.Parameter
                    if isinstance(old_attr, torch.nn.Parameter):
                        setattr(m, l_name,
                                torch.nn.Parameter(
                                    old_attr.chunk(mp, dim=d)[0].contiguous()))
                    else:
                        pass
    if CONV_PARA_ATTR:
        for n, m in model.named_modules():
            for conv_name in CONV_PARA_ATTR:
                if conv_name in n and isinstance(m, nn.Conv2d):
                    # m.mp_attr = 'col'
                    setattr(m, 'mp_attr', 'conv')

    if INPUT_OUTPUT_MAPPING:
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                for i, o, mp_style in INPUT_OUTPUT_MAPPING:
                    # in_features:out_features = i:o
                    if m.in_features * o == m.out_features * i:
                        m.mp_attr = mp_style
                        if mp_style == 'col':
                            m.out_features //= mp
                        elif mp_style == 'row':
                            m.in_features //= mp
                        break
            elif isinstance(m, Conv1D):
                for i, o, mp_style in INPUT_OUTPUT_MAPPING:
                    if m.weight.size(0) * o == m.weight.size(1) * i:
                        setattr(m, 'mp_attr', f'{mp_style}')
                        if mp_style == 'col':
                            in_features = m.weight.size(0)
                            out_features = m.weight.size(1) // mp
                            setattr(m, 'in_features', in_features)
                            setattr(m, 'out_features', out_features)
                        elif mp_style == 'row':
                            in_features = m.weight.size(0) // mp
                            out_features = m.weight.size(1)
                            setattr(m, 'in_features', in_features)
                            setattr(m, 'out_features', out_features)
                        break

    if MANUAL_MP_ATTR:
        mp_attr_list = MANUAL_MP_ATTR
    else:
        mp_attr_list = DEFAULT_MP_ATTR

    def _set_model_mp_attr(model: torch.nn.Module):
        for n, module in model.named_children():
            for attr in mp_attr_list:
                if hasattr(module, attr):
                    old_attr = getattr(module, attr)
                    assert old_attr % mp == 0, \
                        f'mp attr set error, {attr} of {n} is {old_attr}'
                    setattr(module, attr, old_attr//mp)
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                _set_model_mp_attr(module)
    _set_model_mp_attr(model)

    return model


def set_tp_layer_lists(
        col_para_list: Optional[List[str]] = None,
        row_para_list: Optional[List[str]] = None,
        conv_para_list: Optional[List[str]] = None,
        input_output_mapping: Optional[List[Tuple[int, int, str]]] = None,
        weight_change_list: Optional[List[str]] = None,
        tp_attr_list: Optional[List[str]] = None,
    ):
    """
    Set the tp feature dict for model does not have default dict. Indicates 
    the layers and attributes needs to be changed according to the tp degree. 
    Could refer `Merak.modules.mp_mapping.MP_MODEL_MAPPING` as examples.

    Parameters:

    -   col_para_list (List[str], defaults to None) 
        -- Name list of linear layer for column parallel style.
    -   row_para_list (List[str], defaults to None) 
        -- Name list of linear layer for row parallel style..
    -   input_output_mapping (List[tuple(str)], defaults to None) 
        -- Ratio between input and output of linear layer to indicate the 
           tp style, list of (input, output, 'row' or 'col')
    -   weight_change_list (List[tuple(str)], defaults to None) 
        -- List of (layer name, tp dimension), will divide the tp dimension 
           of layer name or layer_name.weight by the tp degree.
    -   tp_attr_list (List[str], defaults to None) 
        -- Manual tp attributes list for each layer, each attribute will be 
           divided by tp degree.
    """

    assert (col_para_list and row_para_list) or conv_para_list \
            or weight_change_list or input_output_mapping , \
            'should use at least one kind of tp list to set tp attr'
    if col_para_list:
        global COL_PARA_LIST
        COL_PARA_LIST = col_para_list
    if row_para_list:
        global ROW_PARA_LIST
        ROW_PARA_LIST = row_para_list
    if weight_change_list:
        global MP_WEIGHT_LIST
        MP_WEIGHT_LIST = weight_change_list
    if input_output_mapping:
        global INPUT_OUTPUT_MAPPING
        INPUT_OUTPUT_MAPPING = input_output_mapping
    if tp_attr_list:
        global MANUAL_MP_ATTR
        MANUAL_MP_ATTR = tp_attr_list
    if conv_para_list:
        global CONV_PARA_ATTR
        CONV_PARA_ATTR = conv_para_list
    global MP_LAYER_LIST_SETTED
    MP_LAYER_LIST_SETTED = True

def get_tp_attr_list() -> List[str]:
    global MANUAL_MP_ATTR
    return MANUAL_MP_ATTR

def mp_is_setted() -> bool:
    global MP_LAYER_LIST_SETTED
    return MP_LAYER_LIST_SETTED
