<!---
Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.

Maintainer: TXacs (txacs1993@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Merak text generation examples

This demonstrates how to perform text generation using 3D parallelism in Merak.

```bash
torchrun --nproc_per_node=4 \
               run_gpt_text_generation.py \
               --model-name gpt2 \
               --cache-dir ./cache/gpt2-110M \
               --output_dir ./output \
               --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
               --resume_from_checkpoint output/transformers_model \
               --activation_checkpointing false --checkpoint_num_layers 0 \
               --return_logits true --no_tie_modules true --text_generation true --seed 42 \
               --split_method layer_split
```


