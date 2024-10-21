# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
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

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/utils/fx.py

__all__ = ['_generate_dummy_input']

import random
import torch
import functools
import inspect
import collections

from typing import List, Optional, Type, Union, Dict
from transformers.models.auto import get_values
from transformers import (
    logging,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    GPT2DoubleHeadsModel,
)

from Merak.merak_args import MerakArguments

logger = logging.get_logger(__name__)


def _generate_random_int(
        low: int = 10, 
        high: int = 20, 
        forbidden_values: Optional[List[int]] = None
) -> int:
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value

def _generate_dummy_input(args: MerakArguments, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Generates dummy input for model inference recording."""
    sequence_length = args.seq_length
    batch_size = args.per_device_train_batch_size

    encoder_sequence_length = \
    sequence_length[0] if isinstance(sequence_length, 
                                        (list, tuple)) else sequence_length
    decoder_sequence_length = (
        sequence_length[1] if isinstance(sequence_length, (list, tuple)) 
        else encoder_sequence_length
    )
    encoder_shape = [batch_size, encoder_sequence_length]
    decoder_shape = (
        [batch_size, decoder_sequence_length] if decoder_sequence_length is not None 
        else list(encoder_shape)
    )
    model_class = model.__class__
    device = 'cpu' #model.device
    # device = 'cpu'
    inputs_dict = dict()

    if args.input_names is None or args.input_names == []:
        input_names = model.dummy_inputs.keys()
    else:
        input_names = args.input_names

    sig = inspect.signature(model.forward)

    concrete_args = {p.name: p.default for p in sig.parameters.values() \
                     if p.name not in input_names}

    input_names = sig.parameters.keys() - concrete_args.keys()

    for input_name in input_names:
        if input_name in ["labels", "start_positions", "end_positions"]:
            batch_size = encoder_shape[0]
            if model_class in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = torch.ones(batch_size, dtype=torch.long,
                                                    device=device)
            elif model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = torch.zeros(batch_size, dtype=torch.long, 
                                                                device=device)
                inputs_dict["end_positions"] = torch.zeros(batch_size, dtype=torch.long, 
                                                            device=device)
            elif model_class in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = torch.zeros(batch_size, dtype=torch.long, 
                                                    device=device)
            elif model_class in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
                GPT2DoubleHeadsModel,
            ]:
                inputs_dict["labels"] = torch.zeros(decoder_shape, dtype=torch.long, 
                                                    device=device)
            elif model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(encoder_shape, dtype=torch.long, 
                                                    device=device)
            else:
                raise NotImplementedError(f"{model_class} not supported yet.")
        elif "visual_feats" in input_name:
                inputs_dict[input_name] = torch.zeros(
                    encoder_shape
                    + [
                        model.config.visual_feat_dim,
                    ],
                    dtype=torch.float,
                    device=device,
                )
        elif "visual_pos" in input_name:
            inputs_dict[input_name] = torch.zeros(
                encoder_shape
                + [
                    model.config.visual_pos_dim,
                ],
                dtype=torch.float,
                device=device,
            )
        elif "input_features" in input_name:
            inputs_dict[input_name] = torch.zeros(
                *encoder_shape,
                model.config.input_feat_per_channel,
                dtype=torch.float, device=device
            )
        elif "input_values" in input_name:
            batch_size, seq_length = encoder_shape
            # Generating big sequence length for audio inputs.
            inputs_dict[input_name] = torch.zeros(batch_size, seq_length,
                                                    dtype=torch.float, device=device)
        elif "mask" in input_name or "ids" in input_name:
            shape = encoder_shape if "decoder" not in input_name else decoder_shape
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.long, device=device)
        elif "pixel_values" in input_name:
            batch_size = encoder_shape[0]
            image_size = getattr(model.config, "image_size", None)
            if image_size is None:
                if hasattr(model.config, "vision_config"):
                    image_size = model.config.vision_config.image_size
                elif hasattr(model.config, "encoder"):
                    image_size = model.config.encoder.image_size
                else:
                    image_size = (_generate_random_int(), _generate_random_int())

            # If no num_channels is in the config, use some arbitrary value.
            num_channels = getattr(model.config, "num_channels", 3)
            if not isinstance(image_size, collections.abc.Iterable):
                image_size = (image_size, image_size)
            height, width = image_size
            inputs_dict[input_name] = torch.zeros(
                batch_size, num_channels, height, width, dtype=torch.float32, device=device
            )
        else:

            shape = encoder_shape if "decoder" not in input_name else decoder_shape
            shape += [model.config.hidden_size]
            inputs_dict[input_name] = torch.ones(shape, dtype=torch.float, device=device)

        # if args.fp16 or args.half_precision_backend == "apex":
        #     half_inputs_dict = {}
        #     for k, v in inputs_dict.items():
        #         half_inputs_dict[k] = v.half()
        #     inputs_dict = half_inputs_dict

    return inputs_dict
