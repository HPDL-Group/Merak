'''
Copyright 2019 The Microsoft DeepSpeed Team

Copyright NVIDIA/Megatron

Helper functions and classes from multiple sources.
'''
# https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/utils.py
import os
import psutil
import gc
from math import ceil
from math import floor
from bisect import bisect_left, bisect_right

import torch
from torch._six import inf
import torch.distributed as dist

from ..utils import logger, log_dist
from numpy import prod

# pt-1.9 deprecations
if hasattr(torch.cuda, "memory_reserved"):
    torch_memory_reserved = torch.cuda.memory_reserved
else:
    torch_memory_reserved = torch.cuda.memory_allocated
if hasattr(torch.cuda, "max_memory_reserved"):
    torch_max_memory_reserved = torch.cuda.max_memory_reserved
else:
    torch_max_memory_reserved = torch.cuda.memory_cached

peak_memory = 0

def noop_decorator(func):
    return func


def ensure_directory_exists(filename):
    """Create the directory path to ``filename`` if it does not already exist.

    Args:
        filename (str): A file path.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def move_to_device(item, device):
    """
    Move tensor onto device. Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device

    Returns:
        None
    """
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item, list):
        return [move_to_device(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    else:
        return item


def see_memory_usage(message, force=False, ram=False, ranks=[0], group=None):
    if not force:
        return
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() in ranks:
        # torch.cuda.empty_cache()
        # torch.distributed.barrier(group=group)
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()
    global peak_memory
    max_ma = round(torch.cuda.max_memory_allocated() / (1024 * 1024),2)
    peak_memory = max(peak_memory, max_ma)
    # Print message except when distributed but not rank 0
    log_dist(message, ranks=ranks)
    log_dist(
        f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024),2 )} MB \
        Max_MA {max_ma} MB \
        CA {round(torch_memory_reserved() / (1024 * 1024),2)} MB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024))} MB \
        PEAK_MA {peak_memory} MB    ", ranks=ranks)

    if ram:
        vm_stats = psutil.virtual_memory()
        used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
        logger.info(
            f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')
    
    # torch.cuda.empty_cache()
    # torch.distributed.barrier(group=group)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call
        args (tuple, optional): args to ``base``
        kwargs (dict, optional): kwargs supplied to ``base``

    Returns:
        str: A string representation of base(*args, **kwargs)
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name

