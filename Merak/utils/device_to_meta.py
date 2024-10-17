# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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

# Parts of the code here are adapted from https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/big_modeling.py

from contextlib import contextmanager

import torch
import torch.nn as nn

@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    with init_on_device(torch.device("meta")) as f:
        yield f

@contextmanager
def init_on_device(device: torch.device):

    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(
        module: nn.Module,
        name: str,
        param: torch.Tensor
        ):
        old_register_parameter(module, name, param)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            if param is not None:
                param_cls = type(module._parameters[name])
                kwargs = module._parameters[name].__dict__
                kwargs["requires_grad"] = param.requires_grad
                module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter