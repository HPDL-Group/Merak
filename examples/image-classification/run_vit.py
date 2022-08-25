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
from Merak import MerakArguments, MerakTrainer, print_rank_0
from utils import collate_fn, prepare_dataset, compute_metrics

from transformers import (
    HfArgumentParser,
    ViTForImageClassification,
    ViTConfig
)

from transformers.utils.dummy_vision_objects import ViTFeatureExtractor


def parse_option(parser):

    # easy config modification
    parser.add_argument('--data-files', type=str, help='path to dataset')
    parser.add_argument('--cache-dir', type=str, help='where to save cache')

    return parser

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

    config = ViTConfig(num_labels=1000, return_dict=False)
    model = ViTForImageClassification(config)
    feature_extractor = ViTFeatureExtractor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])

    ds = prepare_dataset(args.data_files, args.cache_dir)

    # Initalize our trainer
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"], 
        eval_dataset=ds["validation"],
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)

if __name__ == "__main__":
    main()
