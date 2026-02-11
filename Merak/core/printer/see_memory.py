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

import gc
from typing import List, Optional

import psutil
import torch

from Merak import get_logger

logger = get_logger("normal")

# Check if we are using CPU-based memory tracking
try:
    import torch_ft

    USE_CPU = True
except ModuleNotFoundError:
    USE_CPU = False

# Initialize memory tracking functions based on the device (CPU or GPU)
if USE_CPU:
    fn = lambda: 0
    if hasattr(torch_ft.utils, "dsp_mem_used"):
        torch_memory_allocated = torch_ft.utils.dsp_mem_used
        torch_memory_reserved = torch_memory_allocated
    else:
        torch_memory_allocated = fn
        torch_memory_reserved = fn

    if hasattr(torch_ft.utils, "dsp_mem_used_peak"):
        torch_max_memory_allocated = torch_ft.utils.dsp_mem_used_peak
        torch_max_memory_reserved = torch_max_memory_allocated
    else:
        torch_max_memory_allocated = fn
        torch_max_memory_reserved = fn
else:
    torch_memory_allocated = torch.cuda.memory_allocated
    torch_memory_reserved = torch.cuda.memory_reserved

    torch_max_memory_allocated = torch.cuda.max_memory_allocated
    torch_max_memory_reserved = torch.cuda.max_memory_reserved


# Track the peak memory usage across all function calls
peak_memory = 0


def get_memory_info() -> dict:
    """Returns a dictionary containing memory usage statistics.

    Returns:
        dict: Contains 'current' and 'peak' memory usage in MB.
    """
    if USE_CPU:
        allocated = round(torch_memory_allocated() / (1024 * 1024), 2)
        reserved = round(torch_memory_reserved() / (1024 * 1024), 2)
        max_memory = round(torch_max_memory_reserved() / (1024 * 1024), 2)
    else:
        allocated = round(torch_memory_allocated() / (1024 * 1024), 2)
        reserved = round(torch_memory_reserved() / (1024 * 1024), 2)
        max_memory = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    return {"allocated": allocated, "reserved": reserved, "max": max_memory}


def see_memory_usage(
    message: str, force: bool = False, ram: bool = False, ranks: List[int] = [0]
):
    """Logs the memory usage at a specific point in execution.

    Args:
        message (str): Message to include in the log.
        force (bool, optional): Force memory reporting, even when not in the specified ranks. Defaults to False.
        ram (bool, optional): Include RAM usage statistics. Defaults to False.
        ranks (List[int], optional): List of process ranks to report memory for. Defaults to [0].
    """
    global peak_memory

    # Skip memory reporting if not forced and the current rank is not in the report ranks
    if not force or (
        torch.distributed.is_initialized() and torch.distributed.get_rank() not in ranks
    ):
        return

    # Force garbage collection to get accurate memory readings
    gc.collect()

    # Get the current memory usage information
    memory_info = get_memory_info()
    allocated = memory_info["allocated"]
    reserved = memory_info["reserved"]
    max_ma = memory_info["max"]

    # Update the peak memory usage
    peak_memory = max(peak_memory, max_ma)

    # Prepare the formatted log message
    log_message = (
        f"{message}                   "
        f"MA {allocated:.2f} MB       "
        f"Max_MA {max_ma:.2f} MB      "
        f"CA {reserved:.2f} MB        "
        f"Max_CA {max_ma:.2f} MB      "
        f"PEAK_MA {peak_memory:.2f} MB"
    )

    # Log the memory usage
    logger.debug(log_message, ranks=ranks)

    if ram:
        # Get RAM usage statistics
        vm_stats = psutil.virtual_memory()
        used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
        percent = vm_stats.percent

        # Log RAM usage
        logger.debug(f"CPU Virtual Memory:  used = {used_GB} GB, percent = {percent}%")

    # Reset peak memory statistics if supported
    if USE_CPU:
        if hasattr(torch_ft.utils, "reset_dsp_peak_mem_stats"):
            torch_ft.utils.reset_dsp_peak_mem_stats()
    else:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):  # For PyTorch 1.4+
            torch.cuda.reset_peak_memory_stats()
