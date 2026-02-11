"""
Copyright 2019 The Microsoft DeepSpeed Team
"""

# https://github.com/microsoft/DeepSpeed/blob/85ce85dd5f4b18c0019a5121b06900e3a2c3933b/deepspeed/utils/timer.py

import time
from typing import List, Optional

import torch

from Merak import get_logger

__all__ = ["SynchronizedWallClockTimer"]


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""

    class Timer:
        """Timer."""

        def __init__(self, name: str):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, "timer has already been started"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self, reset: bool = False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if reset:
                self.elapsed_ = time.time() - self.start_time
            else:
                self.elapsed_ += time.time() - self.start_time
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset: bool = True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}
        self.logger = get_logger("detailed")
        self.log_rank = [0]

    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def set_log_rank(self, ranks=[]):
        self.log_rank = ranks

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(
            torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
        )
        max_alloc = "max_mem_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        )
        cache = "cache_allocated: {:.4f} GB".format(
            torch.cuda.memory_cached() / (1024 * 1024 * 1024)
        )
        max_cache = "max_cache_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_cached() / (1024 * 1024 * 1024)
        )
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

    def log(
        self,
        names: str,
        normalizer: float = 1.0,
        reset: bool = True,
        ranks: Optional[List[int]] = None,
    ):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = f"rank={torch.distributed.get_rank()} time (ms)"
        for name in names:
            if name in self.timers:
                elapsed_time = (
                    self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
                )
                string += " | {}: {:.2f}".format(name, elapsed_time)

        self.logger.debug(string, ranks=ranks if ranks else self.log_rank)
