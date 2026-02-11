# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Maintainer: daocaoren360 (ouyangshun@foxmail.com)
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

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache

# from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)

from Merak import get_logger
from Merak.core import mpu

logger = get_logger("simple")


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(
                layer
                for layer in module.modules()
                if isinstance(layer, torch.nn.Linear)
            ).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    use_top_left_mask_flag = (
        is_flash_attn_2_available() and not is_flash_attn_greater_or_equal_2_10()
    )
    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=use_top_left_mask_flag,
        target_dtype=target_dtype,
        # **kwargs,
    )

    return attn_output, None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for
    # some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal`
    # if statement instead of an inline conditional assignment in SDPA to support both
    # torch.compile's dynamic shapes and full graph options. An inline conditional
    # prevents dynamic shapes from compiling. Note that it is important to check first
    # for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


LLAMA_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,
    "flash_attention_2": flash_attention_forward,
    "sdpa": sdpa_attention_forward,
    "flex_attention": None,
}

# col_para_list = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
#                  'gate_proj', 'up_proj']
# row_para_list = ['self_attn.o_proj', 'down_proj']


class LlamaAttentionSP(nn.Module):
    """LlamaAttention for Merak sequence parallel(transformers version >= 4.47.1)"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        assert (
            self.num_heads
            % (
                mpu.get_sequence_parallel_world_size()
                * mpu.get_model_parallel_world_size()
            )
            == 0
        )
        self.num_heads_per_partition = self.num_heads // (
            mpu.get_sequence_parallel_world_size() * mpu.get_model_parallel_world_size()
        )
        self.num_key_value_heads = config.num_key_value_heads
        assert (
            self.num_key_value_heads
            % (
                mpu.get_sequence_parallel_world_size()
                * mpu.get_model_parallel_world_size()
            )
            == 0
        )
        self.num_key_value_heads_per_partition = self.num_key_value_heads // (
            mpu.get_sequence_parallel_world_size() * mpu.get_model_parallel_world_size()
        )
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.num_heads = self.num_heads // mpu.get_model_parallel_world_size()
        self.num_key_value_heads = (
            self.num_key_value_heads // mpu.get_model_parallel_world_size()
        )

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        setattr(self.q_proj, "mp_attr", "col")
        setattr(self.k_proj, "mp_attr", "col")
        setattr(self.v_proj, "mp_attr", "col")
        setattr(self.o_proj, "mp_attr", "row")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # query_states.shape (B, Ln, Hp), Ln=L//sp_world_size, Hp=H//tp_world_size
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = mpu.all_to_all_sequence_parallel_region(
            query_states
        )  # query_states.shape (B, L, Hnp)
        key_states = mpu.all_to_all_sequence_parallel_region(key_states)
        value_states = mpu.all_to_all_sequence_parallel_region(value_states)
        bsz, q_len, _ = query_states.size()
        _, k_len, _ = key_states.size()
        _, v_len, _ = value_states.size()

        query_states = query_states.view(
            bsz, q_len, self.num_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, k_len, self.num_key_value_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, v_len, self.num_key_value_heads_per_partition, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention. This warning can be removed using the argument "
                    "`attn_implementation='eager'` when loading the model."
                )
            elif self.config._attn_implementation == "flex_attention":
                logger.warning(
                    "squenece parallel does not support flex_attention, "
                    "This warning can be removed using the argument "
                    "`attn_implementation='eager'` when loading the model."
                )
            else:
                # sdpa or flash_attention
                attention_interface = LLAMA_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(
            bsz, q_len, -1
        ).contiguous()  # attn_output.shape (B, L, Hnp)
        attn_output = mpu.all_to_all_sequence_parallel_region(
            attn_output, scatter_dim=1, gather_dim=2
        )  # B, Ln, Hp

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayerSP(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttentionSP(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.is_first_layer = layer_idx == 0
        self.is_last_layer = layer_idx == (config.num_hidden_layers - 1)
        self.idx = layer_idx
        setattr(self.mlp.up_proj, "mp_attr", "col")
        self.mlp.up_proj.out_features //= mpu.get_model_parallel_world_size()
        setattr(self.mlp.gate_proj, "mp_attr", "col")
        self.mlp.gate_proj.out_features //= mpu.get_model_parallel_world_size()
        setattr(self.mlp.down_proj, "mp_attr", "row")
        self.mlp.down_proj.in_features //= mpu.get_model_parallel_world_size()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        # TODO: test past_key_value is not None for sp
        assert past_key_value is None

        if self.is_first_layer:
            hidden_states = mpu.split_for_sequence_parallel_region(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.is_last_layer:
            hidden_states = mpu.gather_for_sequence_parallel_region(hidden_states)
            if self_attn_weights is not None:
                self_attn_weights = mpu.gather_for_sequence_parallel_region(
                    self_attn_weights
                )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
