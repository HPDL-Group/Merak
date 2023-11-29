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

This demonstrates how to perform lora using Merak.

```bash
python -um torch.distributed.launch --nproc_per_node=4 run_gpt.py \
               --model-name gpt2 \
               --output_dir ./output \
               --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
               --max_steps 20 \
               --activation_checkpointing false --checkpoint_num_layers 0 \
               --input_names "input_ids" \
               --lora_config "./adapter_config.json" \
               --no_load_optim true \
	           --fake-data true --save --save_steps 5 --resume_from_checkpoint /dat/txacs/merak-final/test-examples/peft-gpt2/output/2023-11-15_ckpt
```
