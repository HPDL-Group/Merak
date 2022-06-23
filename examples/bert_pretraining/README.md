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


# BERT pretraining

Here is an example code of [BERT](https://arxiv.org/abs/1810.04805) pretraining with Merak.

## Getting the data

This sample uses hdf5 data files. The preparation of the pre-training dataset is described in
[NVIDIA Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/180382499f791d38eb9e91c105d75764cd2f1cd7/PyTorch/LanguageModeling/BERT#getting-the-data) and its related [scripts](https://github.com/NVIDIA/DeepLearningExamples/tree/180382499f791d38eb9e91c105d75764cd2f1cd7/PyTorch/LanguageModeling/BERT/data).

## Training with Merak

The main modification of training scipts is related to hdf5 data loading and customized loss function.

---

Running according to following bash:

```bash
export TOKENIZERS_PARALLELISM=false
python -m torch.distributed.launch --nproc_per_node=4 \
               run_bert.py \
               --model-name bert-large \
               --data-files "/path/to/hdf5_file/" \
               --output_dir ./output \
               --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
               --logging_steps 10 \
               --input_name input_ids attention_mask token_type_ids \
               --dataloader_num_workers 2
```
