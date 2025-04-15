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


# Merak Parallel Training Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Merak is a high-performance distributed training framework designed for 3D parallel model training. Supports seamless parallelization across multiple GPUs/nodes.

## 🔥 Supported Models

### Population Models
|   |   |   |   |
|---|---|---|---|
| ✅ **DeepseekR1** |

### Natural Language Processing
|   |   |   |   |
|---|---|---|---|
| ✅ ALBERT | ✅ Bart | ✅ BERT | ✅ BlenderBot |
| ✅ DistilBERT | ✅ Electra | ✅ GPT-2 | ✅ GPT-J |
| ✅ LLaMA | ✅ MarianMT | ✅ mBART | ✅ mT5 | 
| ✅ Nezha | ✅ Pegasus | ✅ PLBART | ✅ T5 |
| ✅ XGLM | ✅ OPT | ✅ m2m100 | ✅ LayoutLM |

### Multimodal & Vision-Language
|   |   |   |
|---|---|---|
| ✅ CLIP | ✅ AltCLIP | ✅ TroCR |

### Computer Vision
|   |   |   |   |
|---|---|---|---|
| ✅ ConvNeXt | ✅ ResNet | ✅ Swin | ✅ DINOv2 |
| ✅ SegFormer | ✅ MobileBERT | ✅ LXMERT |
| ✅ UNet |  ✅ ViT |

### Speech Processing
|   |   |   |
|---|---|---|
| ✅ Wav2Vec2 | ✅ Speech2Text | ✅ Speech2Text2 |
| ✅ Hubert |


## ✨ Key Features
- &zwnj;**Multi-Strategy Parallelism**&zwnj;
  Tensor/Data/Pipeline Parallelism hybrid training
- &zwnj;**Memory Optimization**&zwnj;
  Zero Redundancy Optimizer (ZeRO) Stage 1
- &zwnj;**High-Performance Pipeline Parallelism Strategy**&zwnj;
  last_no_recompute_1f1b/full_critical_path_1f1b
- &zwnj;**Other Functions**&zwnj;
  Lora/Text Generation

## 📌 Key Notes
1. &zwnj;**Compatibility Levels**&zwnj;
   - ✅ Full Support: Out-of-the-box parallel strategies  
   - ⚠️ Partial Support: Requires manual configuration or has functional constraints  
