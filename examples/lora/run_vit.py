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

# using our distributed trainer
import Merak
from Merak import MerakArguments, MerakTrainer
from Merak.core import mpu
from Merak.core.printer import AccMetric

import transformers
from transformers.utils.dummy_vision_objects import ViTFeatureExtractor
from transformers.optimization import AdamW
from transformers import (
    set_seed,
    HfArgumentParser,
    ViTForImageClassification,
    ViTConfig
)

import os
import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch.distributed as dist

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    )

def parse_option(parser):

    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')

    return parser

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

config_file = {
        "_name_or_path": "google/vit-base-patch16-224-in21k",
        "architectures": [
            "ViTForImageClassification"
        ],
        "attention_probs_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 768,
        "image_size": 224,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "model_type": "vit",
        "num_attention_heads": 12,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "patch_size": 16,
        "qkv_bias": True,
        # "transformers_version": "4.13.0.dev0",
        "label2id": {},
        "id2label": {},
        "return_dict": False
        }

def main():
    # init dist
    pp = 2
    tp = 1
    dp = 2
    Merak.init(pp, tp, dp)

    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed + mpu.get_data_parallel_rank())

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    


    feature_extractor = ViTFeatureExtractor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])

    dataset = load_dataset("food101", split="train[:5000]")
    labels = dataset.features["label"].names

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    print(id2label[2])

    config = ViTConfig(**config_file)
    config.id2label = id2label
    config.label2id = label2id
    model = ViTForImageClassification(config)

    training_args.seq_length = config.patch_size ** 2
    print_trainable_parameters(model)
    

    normalize = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    train_transforms = Compose(
        [
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            normalize,
        ]
    )


    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    splits = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]
    
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    def compute_metrics(p, normalize=True, sample_weight=None):
        """Computes accuracy on a batch of predictions"""
        # Load the accuracy metric from the datasets package
        metric = {
                "accuracy": accuracy_score(
                    p.label_ids, np.argmax(p.predictions, axis=1), normalize=normalize, sample_weight=sample_weight
                ).item(),
            }
        return metric

    class ViTTrainer(MerakTrainer):

        def create_optimizer(self):
            def get_parameter_names(model, forbidden_layer_types):
                """
                Returns the names of the model parameters that are not inside a forbidden layer.
                """
                result = []
                for name, child in model.named_children():
                    result += [
                        f"{name}.{n}"
                        for n in get_parameter_names(child, forbidden_layer_types)
                        if not isinstance(child, tuple(forbidden_layer_types))
                    ]
                # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
                result += list(model._parameters.keys())
                return result
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            adam_kwargs = {
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=training_args.learning_rate, **adam_kwargs
            )
            return optimizer

        def evaluation(self):
            assert self.eval_dataloader is not None, \
            "The eval_dataloader is None, Please check eval_dataset"
            if self.eval_engine is None:
                self.eval_engine = self.engine(
                                        self.train_engine.module,
                                        self.args,
                                        optimizer=None,
                                        lr_scheduler=None,
                                        tuning_params=self.eval_params,
                                        dataloader=self.eval_dataloader,
                                        loss_fn=self.loss_fn
                                        )

            metrics = AccMetric()

            for _ in range(self.eval_params.eval_steps):
                # try:
                self.eval_params.step += 1
                self.eval_params.global_steps += 1

                loss, logits, labels = self.eval_engine.eval_batch(
                                                    batch_fn=self.prepare_data)

                if self.eval_engine.is_last_stage() and self.args.return_logits:
                    label_list = []
                    logit_list = []
                    for i in labels:
                        label_list.append(i[0])
                    for i in logits:
                        logit_list.append(i[0])
                    del labels
                    step_metrics = compute_metrics(
                        transformers.trainer_utils.EvalPrediction(
                            predictions=torch.cat(logit_list).cpu(),
                            label_ids=torch.cat(label_list).cpu()
                            )
                        )
                    for key in step_metrics:
                        metrics.update(key, step_metrics[key])
                # except StopIteration:
                #     self.eval_engine.timers('eval_batch').stop()
                #     break
            if self.summary_writer is not None:
                self.summary_events = [
                    (f'Eval/loss',
                    loss.mean().item(),
                    self.eval_params.global_steps),
                ]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(
                        event[0],
                        event[1],
                        event[2]
                    )
                avg_acc = list(metrics.avg.values())[0]
                self.summary_events = [
                    (f'Eval/Accuracy',
                    avg_acc,
                    self.eval_params.global_steps),
                ]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(
                        event[0],
                        event[1],
                        event[2]
                    )
            self.eval_engine.reset_dataiterator(self.eval_dataloader)

    # Initalize our trainer
    trainer = ViTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds, 
        eval_dataset=val_ds,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)

if __name__ == "__main__":
    main()
