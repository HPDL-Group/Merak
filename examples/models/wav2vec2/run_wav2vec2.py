# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com), Yck(eyichenke@gmail.com)
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
import Merak

from Merak import MerakArguments, MerakTrainer, init_empty_weights

from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers import (
    DataCollatorForLanguageModeling,
    set_seed,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    HfArgumentParser,
)
from Merak.utils.datasets import DynamicGenDataset


# Add custom command-line arguments
def parse_option(parser):
    parser.add_argument('--model_name', type=str, help='Name of the model to load (e.g. gpt2)')
    return parser


def main():
    # Initialize Merak distributed training environment
    # pp: pipeline parallelism, tp: tensor parallelism, dp: data parallelism
    pp = 4
    tp = 1
    dp = 1
    Merak.init(pp, tp, dp)
    
    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    if tp > 1:
        from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists

        col_para_list = ['q_proj', 'k_proj', 'v_proj', 'intermediate_dense']
        row_para_list = ['out_proj', 'output_dense']
        tp_attr_list = ['num_heads']

        # manully set tp attribute for swin model
        set_tp_layer_lists(col_para_list=col_para_list, row_para_list=row_para_list, 
                           tp_attr_list=tp_attr_list)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # set model config
    # config_kwarg = load_config(args.model_name)
    config_kwarg = {
        'conv_kernel': (10, 2, 2, 2, 2, 1, 1),
        'return_dict': False,
        'use_cache': False
    }
    config = Wav2Vec2Config(**config_kwarg)

    # meta init
    with init_empty_weights():
        model = Wav2Vec2ForCTC(config)
    model.eval()
    model.config._attn_implementation = 'eager'

    # Create a fake dataset for training
    eval_dataset = DynamicGenDataset(
        model.config, mode="speech", dataset_size=1e6, seq_length=training_args.seq_length
    )


    class MyTrainer(MerakTrainer):
        # define custom loss function
        def get_loss_fn(self):
            def loss_fn(outputs, labels):
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(labels, tuple):
                    labels = labels[-1]
                if labels.max() >= model.config.vocab_size:
                    raise ValueError(
                        f"Label values must be <= vocab_size: {model.config.vocab_size}"
                    )

                # retrieve loss input_lengths from attention_mask
                attention_mask = None
                input_ids = torch.randint(1, model.config.vocab_size,
                                      [training_args.per_device_eval_batch_size,
                                       self.args.seq_length]).float()
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(
                        input_ids,
                        dtype=torch.long)
                )
                input_lengths = model._get_feat_extract_output_lengths(
                    attention_mask.sum(-1)
                ).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)

                # ctc_loss doesn't support fp16
                log_probs = torch.nn.functional.log_softmax(outputs, dim=-1,
                                                            dtype=torch.float32).transpose(0, 1)

                with torch.backends.cudnn.flags(enabled=False):
                    loss = torch.nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=model.config.pad_token_id,
                        reduction=model.config.ctc_loss_reduction,
                        zero_infinity=model.config.ctc_zero_infinity,
                    )
                return loss
            return loss_fn

    # using our distributed trainer        
    trainer = MyTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset, 
        # leaf_modules=(_compute_mask_indices,)
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=data_collator,
    )

    # Training
    trainer.evaluation()


if __name__ == "__main__":
    main()
