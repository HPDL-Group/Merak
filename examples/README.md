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

## Merak examples

These examples show that which model can be run with 3D parallelism in Merak. It shows that five popular models of pytorch, including GPT2, ViT, BERT, T5, Swin-transformer, running with 3D parallelism in Merak. These models show three cases of training model:

1. Model can be traced by `transformers.utils.fx` , like GPT2, T5 and BERT.
2. Model is from `transformers`, but cannot be traced by `transformers.utils.fx`, like ViT.
3. Model is not from `transformers`, but can be traced by `torch.fx`, like Swin-tranfromer.

User could make sense of Merak's mechanism by these examples, and apply it to another models. Currently, the running bash is on a machine with 4 GPUs.

```bash
python -m torch.distributed.launch --nproc_per_node=4  run_gpt.py \
                --model-name gpt2 \
                --data-files ./train_context.csv \
                --output_dir ./output \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --data_path /ssd/datasets/imagenet/pytorch
```


