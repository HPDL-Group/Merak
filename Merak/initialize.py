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

import os

import torch

try:
    import torch_ft
except:
    pass


def init(pp: int = -1, tp: int = -1, dp: int = -1, sp: int = 1, backend: str = "nccl"):
    """
    Initialized the distributed communication groups, include data parallel,
    tensor model parallel and pipeline model parallel. Each parallel degree
    has it own communication group, we can ge the rank or size through mpu API.

    Parameters:
    -   dp (int) -- Parallel degree of data parallelism.
    -   tp (int) -- Parallel degree of tensor model parallelism.
    -   pp (int) -- Parallel degree of pipeline model parallelism.
    -   sp (int) -- Parallel degree of sequence model parallelism.
    """
    # Get parallelism dim from parameters or enviroments
    if dp * tp * dp < 0:
        dp = int(os.getenv("DP", "-1"))
        pp = int(os.getenv("PP", "-1"))
        tp = int(os.getenv("TP", "-1"))
        sp = int(os.getenv("SP", "1"))
    else:
        os.environ["DP"] = f"{dp}"
        os.environ["PP"] = f"{pp}"
        os.environ["TP"] = f"{tp}"
        os.environ["SP"] = f"{sp}"

    assert dp * tp * dp > 0, (
        "Please set dp, pp, tp dimension in Merak.init() or "
        "set enviroments(DP, PP, TP)"
    )
