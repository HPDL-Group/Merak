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

## Merak lora examples

This demonstrates how to perform lora using Merak.

```bash
torchrun --nproc-per-node=4  run_vit.py \
                --per_device_train_batch_size 128 --gradient_accumulation_steps 4  \
                --cache-dir ./vit_cache \
                --output_dir ./output --remove_unused_columns False \
                --input_name "pixel_values" \
                --activation_checkpointing false --checkpoint_num_layers 0 \
                --num_train_epochs 5 --learning_rate 5e-3 --dataloader_num_workers 4 \
                --evaluation_strategy='epoch' --return_logits true \
                --save --save_steps 100 --seed 42 \
                --resume_from_checkpoint ./output/ckpt --lora_config ./lora_config.json
```


