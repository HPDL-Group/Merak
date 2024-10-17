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

Merak supports dependency-based automatic graph sharding and config-based manual graph sharding.

# Graph capture
Merak employs trace tool torch.fx and torch._dynamo in Pytorch to trace model into a fx GraphModule.  
1. Please ensure pytorch is installed. Installation please refer to [pytorch repository](https://github.com/pytorch/pytorch).

2. Enable graph capture with setting `trace_method='fx'` or `trace_method='dynamo'` in `Merak.MerakArguments`.

# Graph sharding
Merak supports dependency-based automatic graph sharding and config-based manual graph sharding.  
Enable graph sharding with setting `split_method='farthest_min_deps'` or `split_method='nearest_min_deps'` or `split_method='layer_split'`  in `Merak.MerakArguments`.




