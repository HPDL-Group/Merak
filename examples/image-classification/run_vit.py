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
from transformers import HfArgumentParser, ViTConfig, ViTForImageClassification
from transformers.utils.dummy_vision_objects import ViTFeatureExtractor
from utils import collate_fn, compute_metrics, prepare_dataset

# using our distributed trainer
import Merak
from Merak import MerakArguments, MerakTrainer


def parse_option(parser):

    # easy config modification
    parser.add_argument("--data-files", type=str, help="path to dataset")
    parser.add_argument("--cache-dir", type=str, help="where to save cache")

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

    ds = prepare_dataset(args.data_files, args.cache_dir)

    class VitTrainer(MerakTrainer):
        def create_dataloader(self):
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_sampler=self.get_train_sampler(),
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=collate_fn,
            )

        def prepare_data(self, data):
            if not isinstance(data, (tuple, list)):
                if isinstance(data, dict):
                    inputs_list = []
                    for key, val in self.input_to_stage_dic.items():
                        for i in val:
                            inputs_list.append(data.pop(i))
                    inputs_list += list(data.values())
                    return tuple(inputs_list)
                else:
                    raise NotImplementedError(
                        "only support data in tuple, list or dict"
                    )
            else:
                return data

    # Initalize our trainer
    trainer = VitTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # compute_metrics=compute_metrics,
    )

    # Training
    trainer.train()


if __name__ == "__main__":
    main()
