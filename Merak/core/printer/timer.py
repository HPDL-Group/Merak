'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

# https://github.com/microsoft/DeepSpeed/blob/85ce85dd5f4b18c0019a5121b06900e3a2c3933b/deepspeed/utils/timer.py

import time
import torch
from typing import Optional, List
from .logging import log_dist, logger
from ...merak_args import get_args

__all__ = ['SynchronizedWallClockTimer', 'ThroughputTimer']

LOG_RANKS = [0]

def set_timer_log_rank(ranks=[0]):
    global LOG_RANKS
    LOG_RANKS = ranks

class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""
    class Timer:
        """Timer."""
        def __init__(self, name: str):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()
            self.args = get_args()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            if not self.args.use_cpu:
                torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self, reset: bool = False):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            if not self.args.use_cpu:
                torch.cuda.synchronize()
            if reset:
                self.elapsed_ = (time.time() - self.start_time)
            else:
                self.elapsed_ += (time.time() - self.start_time)
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

    def __call__(self, name: str):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def _init(self):
        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        self('forward_microstep').start()
        self('forward_microstep').stop()
        self('backward_microstep').start()
        self('backward_microstep').stop()
        self('backward_inner_microstep').start()
        self('backward_inner_microstep').stop()
        self('backward_tied_allreduce').start()
        self('backward_tied_allreduce').stop()
        self('backward_allreduce').start()
        self('backward_allreduce').stop()
        self('step_microstep').start()
        self('step_microstep').stop()

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(
            torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(torch.cuda.memory_cached() /
                                                    (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(
            torch.cuda.max_memory_cached() / (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

    def log(self,
            names: str,
            normalizer: float = 1.0,
            reset: bool = True,
            ranks: Optional[List[int]] = None
        ):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = f'rank={torch.distributed.get_rank()} time (ms)'
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(
                    reset=reset) * 1000.0 / normalizer
                string += ' | {}: {:.2f}'.format(name, elapsed_time)

        log_dist(string, ranks=ranks if ranks else LOG_RANKS)
