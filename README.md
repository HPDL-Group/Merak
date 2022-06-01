<!---
Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.

Maintainer: TXacs (txacs1993@gmail.com), Swli (lucasleesw9@gmail.com)

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

# Merak: 3D parallelism for everyone

Merak is a distributed deep learning training framework with automated 3D parallelism. It can automatically slice, allocate and training a DNN model, making the development of giant model fast and easy. The current version of Merak is adapted to PyTorch.

## Motivation of Merak

With the rapidly growing size of DNN models, exquisite distributed training solutions for giant model are required. However, the SOTA technology of giant model pretraining: 3D parallelism (data parallelism, tensor model parallelism, pipeline model parallelism) needs much experiences and model rewriting.

The motivation of Merak is to simplify the usage of 3D parallelism and ensure that users only need to add as little code as the popular training tool [Huggingface transformers trainer](https://huggingface.co/docs/transformers/master/en/main_classes/trainer#trainer) to achieve complicated 3D parallelism. 

## Merak Features

-   Automatic 3D parallel training

In pipeline model parallelism of Merak, we uses `torch.fx` and `transformers.utils.fx` to trace model into `GraphModule`. Then we come up with a graph shard algorithm to split traced graph evenly into a sequence of `GraphModule`. For example, in the GPT model, each attention block and mlp block will be an individual module. Next a modified pipeline runtime engine from [DeepSpeed](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/pipe) would allocate the module sequence and execute the training procedures.

As for tensor model parallelism, we use a feature dict to map the parameters into `ColumnParallelLinear` and `RowParallelLinear` in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/layers.py). We hold default feature dicts for common models in `transformers` package. In addition, users could define a feature dict through our API easily to achieve the tensor model parallelism.

-   Using as easy as single GPU training

For giant model in `transformers`: our implementation is based on `transformers.trainer` class. With a few lines of code setting of parallel degrees, training model with 3D parallelism could be as easy as single GPU training.

For model not in  `transformers`: as long as a model is traceable by `torch.fx` and trainable by `transformers.trainer`, model could trained by Merak as well.

-   Sharding a giant model in a single worker

Training, even only loading, a DNN model on a single GPU device could easily exceed the device's memory capacity nowadays. Before the model created, we create proxy layers for `torch.nn.Linear` layers. Proxy layers do not own parameters but could participate in model trace and graph shard normally. This make it possible that a single worker could store a whole giant DNN model and execute the graph sharding swiftly. 

-   Auto dataloader for 3D parallelism

When we train the model with pipeline parallelism, different stages require different data, some stages even do not load data. Naive solution that all gpu workers load the full datasets and dataloaders leads to a unsatisfied performance. So we try to make the different stages only get their needed datasets.


-   High-performance training

To ensure the training performance, Merak adopts technologies such as activation checkpointing, 1F1B pipeline training schedules and NCCL P2P communication operations. Merak has competitive performance with respect to DeepSpeed and Megatron-LM.

## Installation

To install Merak: 

```bash
git clone http://hpdl-group/Merak.git
cd Merak
pip install .
```


## How to use

To use Merak, make the following modifications to your programs:

1. Import Merak before import transformers and torch
2. Set degrees of the data parallel, tensor model parallel and pipeline model parallel; and run `Merak.init(dp, tp, pp)` to initialize Merak.
3. Set training arguments `MerakArguments`. Replacement of [transformers trainer arguments](https://huggingface.co/docs/transformers/master/en/main_classes/trainer#transformers.TrainingArguments)
4. Set `MerakTrainer`. Replacement of [transformers trainer](https://huggingface.co/docs/transformers/master/en/main_classes/trainer#trainer).

Example usage (see the Merak [examples](https://github.com/HPDL-group/merak_prerelease/examples/) directory for full training examples):

```Python
import Merak
from Merak import MerakArguments, MerakTrainer

# Init Merak with degrees of 3D parallelism.
dp = 2
tp = 2
pp = 1
Merak.init(dp, tp, pp)

# Set training args MerakArgument.
training_args = MerakArguments(
	activation_checkpointing=True
)

# Set our Trainer
trainer = MerakTrainer(
     do_train=...,
     model=...,
     args=training_args,
     train_data=...,
     eval_data=...,
)

# Do train
trainer.train()
```

For more details you could refer to our api [document](https://github.com/HPDL-Group/merak_prerelease/blob/main/docs/api_doc.md).
For more detail usage, please check [transformers](https://github.com/huggingface/transformers) tutorials and its trainer [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch).


## References

The Merak source code was based off the  [transformers trainer](https://huggingface.co/docs/transformers/master/en/main_classes/trainer#trainer), [Deepspeed](https://github.com/microsoft/DeepSpeed) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) repository.

