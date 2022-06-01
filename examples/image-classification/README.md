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


# Example for running ViT model

This example show case of ViT model. Model is from `transformers`, but cannot be traced by `transformers.utils.fx`. We fit it in our Merak and can run it with 3D parallelism.

## Running ViT model with ImageNet for classification

---

Run it according to following bash:

```bash
python -m torch.distributed.launch --nproc_per_node=4  run_vit.py \
                --per_device_train_batch_size 4 --gradient_accumulation_steps 4  \
                --cache-dir ./vit \
                --data-files /ssd/datasets/imagenet/pytorch/ \
                --seq_length 1024 --output_dir ./output --remove_unused_columns False \
                --input_name "pixel_values"
```

Code is based on [transformers](https://github.com/huggingface/transformers/tree/master/examples/pytorch/image-classification) repository.