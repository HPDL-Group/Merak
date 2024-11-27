# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
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

import torch
import gc
import psutil
from .logging import logger, log_dist
from typing import Optional, List

try:
    import torch_ft
    USE_CPU = True
except ModuleNotFoundError:
    USE_CPU = False

if USE_CPU:
    if hasattr(torch_ft.utils, "dsp_mem_used"):
        torch_memory_reserved = torch_ft.utils.dsp_mem_used
    else:
        torch_memory_reserved = None
    if hasattr(torch_ft.utils, "dsp_mem_used_peak"):
        torch_max_memory_reserved = torch_ft.utils.dsp_mem_used_peak
    else:
        torch_max_memory_reserved = None
else:
    if hasattr(torch.cuda, "memory_reserved"):
        torch_memory_reserved = torch.cuda.memory_reserved
    else:
        torch_memory_reserved = torch.cuda.memory_allocated
    if hasattr(torch.cuda, "max_memory_reserved"):
        torch_max_memory_reserved = torch.cuda.max_memory_reserved
    else:
        torch_max_memory_reserved = torch.cuda.memory_cached

peak_memory = 0

def see_memory_usage(
        message: str,
        force: bool= False,
        ram: bool = False,
        ranks: List[int] = [0],
    ):
    if not force:
        return
    if torch.distributed.is_initialized() and \
        not torch.distributed.get_rank() in ranks:
        # torch.cuda.empty_cache()
        # torch.distributed.barrier(group=group)
        return

    # python doesn't do real-time garbage collection so do it explicitly to get
    # the correct RAM reports
    gc.collect()
    global peak_memory
    if USE_CPU:
        max_ma = round(torch_max_memory_reserved() / (1024 * 1024),2)
    else:
        max_ma = round(torch.cuda.max_memory_allocated() / (1024 * 1024),2)
    peak_memory = max(peak_memory, max_ma)
    # Print message except when distributed but not rank 0
    # log_dist(message, ranks=ranks)
    log_dist(
        f"{message} MA {round(torch_memory_reserved() / (1024 * 1024),2 )} MB \
        Max_MA {max_ma} MB \
        CA {round(torch_memory_reserved() / (1024 * 1024),2)} MB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024))} MB \
        PEAK_MA {peak_memory} MB    ", ranks=ranks)

    if ram:
        vm_stats = psutil.virtual_memory()
        used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
        logger.info(
            f'CPU Virtual Memory:  used = {used_GB} GB, \
              percent = {vm_stats.percent}%')
    
    # torch.cuda.empty_cache()
    # torch.distributed.barrier(group=group)
    # get the peak memory to report correct data, so reset the counter for the
    # next call
    if not USE_CPU:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
            torch.cuda.reset_peak_memory_stats()