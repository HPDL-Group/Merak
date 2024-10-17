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

# Parts of the code here are adapted from https://github.com/huggingface/transformers/blob/v4.15.0/examples/pytorch/image-classification/run_image_classification.py

import os
from PIL import Image
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import numpy as np
import datasets
from datasets import load_dataset
from sklearn.metrics import accuracy_score

def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")

def collate_fn(examples):
    data_list = []
    for i in examples[0].keys():
        if i != 'image_file_path' and i != 'image':
            if torch.is_tensor(examples[0][i]):
                data = torch.stack([example[i] for example in examples])
            else:
                data = torch.tensor([example[i] for example in examples])
            data_list.append(data)
    inputs = list(examples[0].keys())
    if 'image_file_path' in inputs:
        inputs.remove('image_file_path')
    if 'image' in inputs:
        inputs.remove('image')
    return {j: data_list[i] for i, j in enumerate(inputs)}


# Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p, normalize=True, sample_weight=None):
    """Computes accuracy on a batch of predictions"""
    # Load the accuracy metric from the datasets package
    metric = {
            "accuracy": accuracy_score(
                p.label_ids, np.argmax(p.predictions, axis=1), 
                normalize=normalize, sample_weight=sample_weight
            ).item(),
        }
    return metric

def prepare_dataset(data_path, cache_dir):
    # Initialize our dataset and prepare it for the 'image-classification' task.
    data_files = {}
    data_files['train'] = os.path.join(data_path, "train/**")
    data_files['validation'] = os.path.join(data_path, "val/**")
    if os.path.isdir(cache_dir):
        ds = load_dataset(cache_dir)
    else:
        ds = load_dataset(
            'imagefolder',
            # data_args.dataset_config_name,
            data_files=data_files,
            cache_dir=cache_dir,
            task="image-classification",
            ignore_verifications=True,
        )
    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    _train_transforms = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        if "image" in example_batch.keys():
            example_batch["pixel_values"] = [
                _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
            ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        if "image" in example_batch.keys():
            example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    # Set the training transforms
    ds["train"].set_transform(train_transforms)

    # Set the validation transforms
    ds["validation"].set_transform(val_transforms)

    return ds