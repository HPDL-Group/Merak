from typing import Optional, Tuple, Dict
from torch import nn

from .modeling import MODELS_SP_CONFIG
from ...utils import init_empty_weights
from ..tensor_parallel.utils import(init_method_normal,
                                    scaled_init_method_normal)
from ..tensor_parallel.mp_attrs import set_mp_attr, mp_is_setted
from ..tensor_parallel.mp_mapping import get_mp_layer_lists
from ..tensor_parallel.mp_layers import build_layers
from ..mpu import get_model_parallel_world_size
from ...merak_args import MerakArguments


def _get_sub_module_for_replaced(raw_model : nn.Module
                                 ) -> Dict[nn.Module, nn.Module]:
    
    models_cls_name = raw_model.__class__.__name__
    
    for model_type, models_sp_info in MODELS_SP_CONFIG.items():
        if model_type in models_cls_name:
            supported_models = models_sp_info['models']
            if models_cls_name in supported_models:
                return models_sp_info['sub_module_replacement']

    return {}

def _build_tp_module(model, init_method, scaled_init_method):
    assert get_model_parallel_world_size() > 1

    set_mp_attr(model, 1)
    
    def build(model):
        for n, module in model.named_children():
            tp_layer = build_layers(n, module, 1, init_method, scaled_init_method)
            if tp_layer is not None:
                setattr(model, n, tp_layer)
            if len(list(module.children())) > 0:
                build(module)
    
    build(model)     

def replace_to_sp_module(pipe_model, raw_model : nn.Module, args : MerakArguments):
    
    raw_model_config = raw_model.config if hasattr(raw_model, "config") else None
    assert raw_model_config, \
        f'need the configuration for model {raw_model.__class__().__name__}'

    sub_module_replacement_dict = _get_sub_module_for_replaced(raw_model)
    assert sub_module_replacement_dict, \
        f'Model {raw_model.__class__().__name__} cannot enable sequence parallelism in Merak'

    # 支持SP + TP
    tp_on = get_model_parallel_world_size() > 1
    if tp_on:
        if not mp_is_setted():
            mp_layer_lists = get_mp_layer_lists(raw_model.__class__)
            if mp_layer_lists is not None:
                set_tp_layer_lists(**mp_layer_lists)

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(
            args.init_method_std, args.num_layers
        )
        
            
    def replace_module(model, config, raw_module_cls, sp_module_cls): 
        for n, module in model.named_children():
                if isinstance(module, raw_module_cls):
                    with init_empty_weights():
                        # 使用meta初始化，统一参数的初始化方式
                        layer_sp = sp_module_cls(config, module.self_attn.layer_idx)
                    if tp_on:
                        # TODO: tp ModuleRebuild可以去掉sp并行中存在的tp并行替换处理
                        _build_tp_module(layer_sp, init_method, scaled_init_method)
                    setattr(model, n, layer_sp)
                if len(list(module.children())) > 0:
                    replace_module(module, config, raw_module_cls, sp_module_cls)        
                
    for raw_module_cls, sp_module_cls in sub_module_replacement_dict.items():
        replace_module(pipe_model, raw_model_config, raw_module_cls, sp_module_cls)

def get_leaf_modules_for_sp(raw_model : nn.Module) ->Tuple[nn.Module]:
    
    sub_module_replacement_dict = _get_sub_module_for_replaced(raw_model)
    if sub_module_replacement_dict:
        return tuple(sub_module_replacement_dict.keys())
    else:
        return () 
    
    