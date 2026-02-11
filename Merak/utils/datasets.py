# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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
from torch.utils.data import Dataset

SUPPORTED_MODES = [
    "text_only",
    "multimodal",
    "vision_only",
    "condition",
    "speech",
    "speech2text",
    "for_qa",
]

MANDATORY_CONFIG_KEYS = [
    "seq_length",
    "max_position_embeddings",
    "n_positions",
    "embed_dim",
    "max_target_positions",
]


class DynamicGenDataset(Dataset):
    """
    A dynamic dataset for generating synthetic data for various tasks.

    Args:
        model_config: Configuration object containing model parameters.
        mode (str, optional): Mode of data generation. One of:
            - "text_only": Generate text data.
            - "multimodal": Generate multimodal data (text + images).
            - "vision_only": Generate vision data.
            - "condition": Generate data for conditional generation.
            - "speech": Generate speech data.
            - "speech2text": Generate data for speech-to-text tasks.
            - "for_qa": Generate data for question answering tasks.
        Default: "text_only".
        dataset_size (int, optional): Number of virtual samples.
            Controls the __len__ of the dataset. Default: 1000.
        image_size (int, optional): Size of generated images.
            Default: 224.
        seq_length (int, optional): Sequence length. If not provided,
            it will be inferred from model_config.
    """

    def __init__(
        self,
        model_config,
        mode="text_only",
        dataset_size=1000,
        image_size=None,
        seq_length=None,
    ):
        super().__init__()
        self._validate_mode(mode)

        # Configuration
        self.model_config = model_config
        self.dataset_size = dataset_size
        self.seq_length = seq_length
        self.image_size = image_size or 224
        self.mode = mode
        self.num_labels = getattr(model_config, "num_labels", None)
        self._infer_sequence_length(model_config)
        self._validate_sequence_length()

        # Validation
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode: {mode}. " f"Supported modes: {SUPPORTED_MODES}"
            )

    def _infer_sequence_length(self, model_config):
        """Infer sequence length from model configurations."""
        if self.seq_length is not None:
            return
        model_config = getattr(model_config, "text_config", model_config)
        for key in MANDATORY_CONFIG_KEYS:
            if hasattr(model_config, key):
                self.seq_length = getattr(model_config, key)
                break
        if self.mode != "vision_only" and self.seq_length is None:
            raise ValueError(
                "Sequence length could not be inferred and is required"
                " for non-vision only modes."
            )

    def _validate_mode(self, mode):
        """Validate the mode argument."""
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Mode '{mode}' is not supported. " f"Options: {SUPPORTED_MODES}"
            )

    def _validate_sequence_length(self):
        """Validate sequence length after inference."""
        if self.sequence_length is None and self.mode != "vision_only":
            raise ValueError("Sequence length is required for non-vision only modes.")

    @property
    def sequence_length(self):
        """Get the sequence length."""
        return self.seq_length

    def __len__(self):
        """Return the number of virtual samples."""
        return int(self.dataset_size)

    def __getitem__(self, idx):
        """Generate a synthetic data sample based on the mode."""
        try:
            if self.mode == "text_only":
                return self._generate_text_data()
            elif self.mode == "multimodal":
                return self._generate_multimodal_data()
            elif self.mode == "vision_only":
                return self._generate_vision_data()
            elif self.mode == "condition":
                return self._generate_condition_data()
            elif self.mode == "speech":
                return self._generate_speech_data()
            elif self.mode == "speech2text":
                return self._generate_speech_to_text_data()
            elif self.mode == "for_qa":
                return self._generate_qa_data()
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate data: {str(e)}")

    def _generate_text_data(self):
        """Generate text data sample."""
        input_ids = torch.randint(1, 1000, (self.sequence_length,))
        labels = torch.randint(1, 1000, (self.sequence_length,))
        return {"input_ids": input_ids, "labels": labels}

    def _generate_multimodal_data(self):
        """Generate multimodal data sample."""
        input_ids = torch.randint(1, 64, (self.sequence_length,))
        pixel_values = torch.randn(3, self.image_size, self.image_size)
        return {"input_ids": input_ids, "pixel_values": pixel_values}

    def _generate_vision_data(self):
        """Generate vision-only data sample."""
        pixel_values = torch.randn(3, self.image_size, self.image_size)
        labels = torch.randint(0, self.num_labels, (1,))
        return {"pixel_values": pixel_values, "labels": labels}

    def _generate_condition_data(self):
        """Generate condition data."""
        input_ids = torch.randint(1, 1000, (self.sequence_length,)).long()
        decoder_input_ids = torch.randint(1, 1000, (self.sequence_length,)).long()
        labels = torch.randint(1, 1000, (self.sequence_length,)).long()
        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

    def _generate_speech_data(self):
        """Generate speech data."""
        input_values = torch.randn((self.sequence_length,))
        labels = torch.randint(
            1, self.model_config.vocab_size, (self.sequence_length,)
        ).long()
        return {"input_values": input_values, "labels": labels}

    def _generate_speech_to_text_data(self):
        """Generate data for speech-to-text tasks."""
        input_features = torch.randn(
            [self.sequence_length, self.model_config.input_feat_per_channel]
        )
        decoder_input_ids = torch.randint(1, 1000, (self.sequence_length,)).long()
        labels = torch.randint(1, 1000, (self.sequence_length,)).long()
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

    def _generate_qa_data(self):
        """Generate data for question answering."""
        input_ids = torch.randint(1, 1000, (self.sequence_length,)).long()
        visual_feats = torch.randn(
            [self.sequence_length, self.model_config.visual_feat_dim]
        )
        visual_pos = torch.randn(
            [self.sequence_length, self.model_config.visual_pos_dim]
        )
        labels = torch.randint(0, 1, (1,)).long()
        return {
            "input_ids": input_ids,
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
            "labels": labels,
        }
