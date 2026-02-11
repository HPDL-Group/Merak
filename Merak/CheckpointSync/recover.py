# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from original source.
# Original: https://github.com/meta-pytorch/torchft-8ef24c055ebb495caf39fb2acdbddb8ebcebdf19/blob/main/torchft/manager.py
# Modifications Copyright (c) 2026.


import time
import torch
import socket
import concurrent
import os 
import traceback
import weakref
import torch.distributed as dist

from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from torch.distributed import ReduceOp, TCPStore
from torch.distributed.distributed_c10d import AllreduceOptions, ReduceOp, Work
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)

from .pg_transport import PGTransport
from .process_group import ProcessGroupNCCL
from .work import _DummyWork
from ._rwlock import RWLock
from .futures import future_timeout
from .utils import get_stream_context

T = TypeVar("T")
S = TypeVar("S")

class ExceptionWithTraceback(Exception):
    def __init__(self, e: Exception) -> None:
        self.original_exception = e
        self.stack_trace: str = traceback.format_exc()
        super().__init__(f"{e}\n{self.stack_trace}")

class Recover:
    def __init__(
            self, 
            monitor, 
            model,
            optimizer,
            logger,
            master_tcp=None
    ):
        self.monitor = monitor
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.master_tcp=master_tcp
        self._group_rank = int(os.environ.get("RANK", 0))
        self.store_addr = os.environ["MASTER_ADDR"]
        self.store_port = int(os.environ["MASTER_PORT"])
        self._replica_id = None,
        self._recovery_event = None
        self._healing = False
        self.allow_heal = True
        self.shrink_only = False
        self.timeout = timedelta(minutes=3)
        self._user_state_dicts: Dict[str, Callable[[], object]] = {}
        self._load_state_dict_fns: Dict[str, Callable[[object], None]] = {}
        self._future: Optional[concurrent.futures.Future] = None
        self._first_init = True
        self._pending_state_dict = None
        self.state_dict_lock = RWLock(timeout=self.timeout.total_seconds())
        self.pg = ProcessGroupNCCL(timeout=timedelta(minutes=1))
        self._executor = ThreadPoolExecutor(
            max_workers=1, 
            thread_name_prefix="recover_state"
        )
        self.ckpt_transporter = PGTransport(
            self.pg, 
            timeout=timedelta(minutes=3),
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self._pg_initialized = False
        # self._initialize_process_group()  # 延迟初始化，等到第一次调用allreduce时再初始化

        # if load_state_dict and state_dict:
        self.register_state_dict_fn("default", self.load_state_dict, self.state_dict)
    
    def _initialize_process_group(self):
        """Initialize the process group with current environment."""
        max_retries = 3
        retry_delay = 0.1  # 100ms for fast recovery
        
        self.logger.info(f"[DEBUG] _initialize_process_group called, rank={self._group_rank}, _first_init={self._first_init}")
        
        for attempt in range(max_retries):
            try:
                replica_rank = self._group_rank
                
                # 等待所有worker就绪
                active_workers = self.monitor.list_active_workers()
                replica_world_size = len(active_workers)
                
                self.logger.info(f"[DEBUG] Attempt {attempt + 1}/{max_retries}: rank={replica_rank}, active_workers={replica_world_size}")
                
                if replica_world_size == 0:
                    self.logger.warning(f"No active workers, waiting...")
                    time.sleep(retry_delay)
                    continue
                
                # 使用实际活跃worker数量而不是world_size
                if replica_world_size < 2:
                    self.logger.warning(f"Not enough active workers ({replica_world_size}), waiting...")
                    time.sleep(retry_delay)
                    continue
                
                # 如果不是第一次初始化，可能需要等待master更新端口
                if not self._first_init:
                    self.logger.info(f"[DEBUG] Waiting for port update to propagate...")
                    time.sleep(3)
                
                # 使用dist_monitor提供的TCPStore端口
                tcp_store_port = self.monitor.get_tcp_store_port()
                self.logger.info(f"[DEBUG] Using TCPStore port from manager: {tcp_store_port}")
                _store_addr = self.store_addr + f':{tcp_store_port}'
                # 所有worker使用相同的store地址，不要包含rank
                store_prefixed_addr = f"{_store_addr}/recover"

                self.logger.info(f"Initializing ProcessGroup (attempt {attempt + 1}/{max_retries}): rank={replica_rank}, world_size={replica_world_size}, store={store_prefixed_addr}")

                if torch.accelerator.is_available():
                    torch.accelerator.synchronize()

                self.logger.info(f"[DEBUG] Calling pg.configure with: store={store_prefixed_addr}, replica_id={self._replica_id}, rank={replica_rank}, world_size={replica_world_size}, first_init={self._first_init}")
                self.pg.configure(
                    store_prefixed_addr,
                    self._replica_id if self._replica_id is not None else "0",
                    replica_rank,
                    replica_world_size,
                    self._first_init
                )
                self._first_init = False
                self._pg_initialized = True
                self.logger.info(f"ProcessGroup initialized successfully: rank={replica_rank}, world_size={replica_world_size}")
                return
                
            except Exception as e:
                self.logger.exception(f"Failed to initialize process group (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to initialize process group after {max_retries} attempts")
                    self.report_error(e)
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optim"])

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
        }

    def register_state_dict_fn(
        self,
        key: str,
        load_state_dict: Callable[[T], None],
        state_dict: Callable[[], T],
    ) -> None:
        # Can't register duplicate keys
        assert key not in self._load_state_dict_fns
        assert key not in self._user_state_dicts

        self._load_state_dict_fns[key] = cast(Callable[[object], None], load_state_dict)
        self._user_state_dicts[key] = state_dict

    def _manager_state_dict(self):
        with self.state_dict_lock.r_lock():
            assert (
                len(self._user_state_dicts) > 0
            ), "user state_dict is not initialized."
            return {
                "user": {key: value() for key, value in self._user_state_dicts.items()},
                "recover": self.state_dict(),
            }

    def report_error(self, e):
        self._erroed = ExceptionWithTraceback(e)
    
    def wrap_future(
        self,
        fut: torch.futures.Future[T],
        default: T,
        timeout: Optional[timedelta] = None,
    ) -> torch.futures.Future[T]:
        """
        Wrap a Future and swallow any errors that occur and report them to the manager.

        If an error occurs, the Future will be completed with the default value.

        Args:
            fut: the Future to wrap
            default: the default value to complete the Future with if an error occurs
            timeout: the timeout for the Future, if None, the manager's timeout will be used
        """

        fut = future_timeout(fut, timeout or self.timeout)

        stream: Optional[torch.Stream] = (
            torch.accelerator.current_stream()
            if torch.accelerator.is_available()
            else None
        )

        # schedule error handling as a continuation on the Future
        def callback(
            fut: torch.futures.Future[T],
        ) -> T:
            nonlocal default, stream

            with get_stream_context(stream):
                try:
                    return fut.value()
                except Exception as e:
                    self.logger.exception(
                        f"got exception in future -- skipping remaining: {e}"
                    )
                    self.report_error(e)
                    return default

        fut = fut.then(callback)
        return fut

    def recover_state(
            self,
            allow_heal: bool = True,
            shrink_only: bool = False,
            timeout: Optional[timedelta] = None,
            curr_device=None
    ):
        """恢复后的训练同步

        确保所有rank完成checkpoint传输和加载后，继续训练。
        注意：checkpoint传输已在start_recover中通过_execute_checkpoint_transfer完成。
        """
        torch.accelerator.set_device_index(curr_device)

        recovery_stream = (
            torch.Stream() if torch.accelerator.is_available() else None
            )
        with get_stream_context(recovery_stream):
            self._recovery_event = (
                torch.accelerator.current_stream().record_event()
                if recovery_stream is not None
                else None
            )

    def start_recover(self):
        self.logger.info(f"[DEBUG] start_recover called, rank={self._group_rank}, _pg_initialized={self._pg_initialized}")
        self._errored = None
        self._healing = self.monitor.is_failover_worker()

        self.failed_ranks = self.monitor.get_failed_ranks(force_refresh=True)
        self.src_rank = self.monitor.get_recover_src_rank(force_refresh=True)
        self.logger.info(f"[DEBUG] failed_ranks={self.failed_ranks}, src_rank={self.src_rank}")
        print(f"[DEBUG] failed_ranks={self.failed_ranks}, src_rank={self.src_rank}")

        pg_is_valid = self._pg_initialized and self.pg._pg is not None
        self.logger.info(f"[DEBUG] pg_is_valid={pg_is_valid}, _pg_initialized={self._pg_initialized}, self.pg._pg={self.pg._pg}")
        print(f"[DEBUG] pg_is_valid={pg_is_valid}, _pg_initialized={self._pg_initialized}, self.pg._pg={self.pg._pg}")

        has_failed_workers = len(self.failed_ranks) > 0 and self._pg_initialized
        self.logger.info(f"[DEBUG] has_failed_workers={has_failed_workers}")
        print(f"[DEBUG] has_failed_workers={has_failed_workers}")

        if not pg_is_valid and self._pg_initialized:
            self.logger.info(f"Process group is invalid, will reinitialize")
            self._pg_initialized = False
            has_failed_workers = True

        if has_failed_workers:
            if self._pg_initialized:
                self.logger.info(f"[DEBUG] Aborting process group")
                try:
                    self.pg.abort(errored=False)
                    self._pg_initialized = False
                    self._first_init = False
                    import time
                    time.sleep(2)
                except Exception as e:
                    self.logger.warning(f"Error aborting: {e}")
            
            if self._group_rank == 0:
                new_port = self.monitor.get_tcp_store_port() + 1
                self.logger.info(f"[DEBUG] Master updating TCPStore port to {new_port}")
                self.monitor.update_tcp_store_port(new_port)

        if not self._pg_initialized:
            self.logger.info(f"[DEBUG] Process group not initialized, calling _initialize_process_group")
            self._initialize_process_group()
        else:
            self.logger.info(f"[DEBUG] Process group already initialized, skipping initialization")

        # 等待所有rank完成PG初始化
        self.logger.info(f"[DEBUG] Rank {self._group_rank}: 等待所有rank PG初始化...")
        self.monitor.wait_for_workers(timeout=60.0, check_interval=0.5)
        self.logger.info(f"[DEBUG] Rank {self._group_rank}: 所有rank PG初始化完成")

        # 执行checkpoint传输
        # self._execute_checkpoint_transfer()

        if self._future is not None:
            self._future.result()
        
        self._future = self._executor.submit(
            self.recover_state,
            allow_heal=self.allow_heal,
            shrink_only=self.shrink_only,
            timeout=self.timeout,
            curr_device=self._group_rank,
            )
        self.logger.info(f"[DEBUG] start_recover completed")

    def _execute_checkpoint_transfer(self):
        """执行checkpoint传输

        确保所有rank PG初始化完成后，由rank 0发送checkpoint，故障rank接收checkpoint。
        """
        max_step = 10
        checkpoint_metadata = '<n/a>'

        recovery_stream = (
            torch.Stream() if torch.accelerator.is_available() else None
        )
        with get_stream_context(recovery_stream):
            try:
                # Rank 0: 发送 checkpoint 给失败的 worker
                if self._group_rank == 0:
                    if len(self.failed_ranks) >= 1:
                        print(f'====Rank:{self._group_rank}:start send checkpoint===================')
                        self.ckpt_transporter.send_checkpoint(
                            dst_ranks=self.failed_ranks,
                            step=max_step,
                            state_dict=self._manager_state_dict(),
                            timeout=self.timeout
                        )
                        print(f'====Rank:{self._group_rank}:finish send checkpoint===================')

                # 故障转移 worker: 接收 checkpoint
                if self._healing:
                    print(f'====Rank:{self._group_rank}:start recv checkpoint===================')
                    self._pending_state_dict = self.ckpt_transporter.recv_checkpoint(
                        src_rank=self.src_rank,
                        metadata=checkpoint_metadata,
                        step=max_step,
                        timeout=self.timeout,
                    )
                    print(f'====Rank:{self._group_rank}:finish recv checkpoint===================')

                    print(f'====Rank:{self._group_rank}:start load checkpoint===================')
                    self.load_state_dict(self._pending_state_dict["recover"])
                    print(f'====Rank:{self._group_rank}:finish load checkpoint===================')


            except Exception as e:
                self.logger.exception(f"got exception in checkpoint transfer: {e}")
                self.report_error(e)
    
    def allreduce(
        self,
        tensor: torch.Tensor,
        should_quantize: bool = False,
        reduce_op: ReduceOp = ReduceOp.AVG,
    ) -> Work:
        self.logger.info(f"[DEBUG] allreduce called, rank={self._group_rank}, _pg_initialized={self._pg_initialized}, _pg={self.pg}")
        num_participants = self.monitor.get_world_size()

        if self._errored:
            self.logger.warning(f"[DEBUG] _errored is True, returning _DummyWork")
            return _DummyWork(tensor)
        
        assert (
            self._future is not None
        ), "must call start_recover before wait_quorum"

        self._future.result()

        # special logic for average
        pg_reduce_op = reduce_op
        if reduce_op == ReduceOp.AVG:
            if not torch.is_floating_point(tensor):
                raise ValueError(
                    "average reduce op is only supported for floating point tensors"
                )
            pg_reduce_op = ReduceOp.SUM
        # try:
            self.logger.info(f"[DEBUG] Calling self.pg.allreduce, tensor.shape={tensor.shape}, reduce_op={pg_reduce_op}")
            opts = AllreduceOptions()
            opts.reduceOp = pg_reduce_op
            work = self.pg.allreduce([tensor], opts)
            self.logger.info(f"[DEBUG] self.pg.allreduce returned, work={work}")

            def callback(
                fut: torch.futures.Future[torch.Tensor],
            ) -> torch.Tensor:
                nonlocal tensor
                if reduce_op == ReduceOp.AVG:
                    tensor /= num_participants
                return tensor
            
            recovered_work = _RecoveredWork(self, work, tensor)
            fut = recovered_work.get_future()
            fut = cast(torch.futures.Future[torch.Tensor], fut)
            fut = fut.then(callback)
            return recovered_work
        
        # except Exception as e:
        #     self.logger.exception(
        #         f"got exception in all reduce -- skipping remaining: {e}"
        #     )
        #     self.report_error(e)

        #     return _DummyWork(tensor)



class _SimpleFuture(torch.futures.Future[T]):
    """
    A simplified implementation of torch.futures.Future that wraps a value.

    This class provides a minimal Future implementation that holds a pre-determined value.
    It's primarily used as a wrapper for values in the callback chain of `_ManagedFuture`.
    Most methods raise `RuntimeError` as they're not intended to be called.

    This class is designed to be used only in specific contexts where we don't
    want to call `value()` on the underlying `Future` as that would cause the CPU to block.
    """

    def __init__(self, value: object) -> None:
        super().__init__()
        self._value = value

    def value(self) -> object:
        return self._value

    def then(
        self, callback: Callable[[torch.futures.Future[T]], S]
    ) -> torch.futures.Future[S]:
        raise NotImplementedError(
            "This future is only supposed to be used in callback chain to extract the value"
        )

    def wait(self) -> object:
        raise NotImplementedError(
            "This future is only supposed to be used in callback chain to extract the value"
        )

    def done(self) -> bool:
        raise NotImplementedError(
            "This future is only supposed to be used in callback chain to extract the value"
        )

    def add_done_callback(
        self, callback: Callable[[torch.futures.Future[T]], None]
    ) -> None:
        raise NotImplementedError(
            "This future is only supposed to be used in callback chain to extract the value"
        )

    def set_result(self, result: object) -> None:
        raise NotImplementedError(
            "This future is only supposed to be used in callback chain to extract the value"
        )

    def set_exception(self, result: object) -> None:
        raise NotImplementedError(
            "This future is only supposed to be used in callback chain to extract the value"
        )


class _RecovedFuture(torch.futures.Future[T]):
    """
    A specialized Future implementation that works alongside `_RecoveredWork`.

    This class extends torch.futures.Future to provide future chaining that is
    lazy - `then()` method simply stores the callback, which is only executed when
    `wait()` is called on `_ManagedFuture` or `_ManagedWork`

    Callback chains are implemented as a linked list of `_ManagedFuture` objects through the
    `_next` attribute. When appending a callback to the chain, it also updates the tail of the
    linked list stored in `_ManagedWork`.

    Delegates actual future operations to an internal torch.futures.Future.

    Raises RuntimeError for methods that should not be called.
    """

    def __init__(self, recovered_work: weakref.ReferenceType["_RecovedWork"]) -> None:
        super().__init__()
        # Store a weak reference to _ManagedWork to avoid reference cycles
        self._recovered_work = recovered_work

        # The underlying torch.futures.Future that this class delegates to
        self._fut: Optional[torch.futures.Future[T]] = None

        # The next future in the callback chain
        self._next: Optional[_RecovedFuture[object]] = None

        # The callback to be executed when the future is completed - this callback
        # returns the next future in the chain
        self._callback: Optional[Callable[[torch.futures.Future[T]], object]] = None

    def then(
        self,
        callback: Callable[[torch.futures.Future[T]], S],
    ) -> torch.futures.Future[S]:
        """
        Sets the callback to be executed when the future is completed.

        Since the callback returns a future, this method also creates a new future
        in the chain and also updates the tail of the chain in `_ManagedWork`.
        """
        recovered_work = self._recovered_work()
        assert recovered_work is not None, "got garbage collected"

        self._callback = callback
        self._next = _RecovedFuture[object](self._recovered_work)
        recovered_work._recovered_fut_tail = self._next
        return cast(torch.futures.Future[S], self._next)

    def wait(self) -> object:
        assert self._fut
        return self._fut.wait()

    def value(self) -> object:
        raise NotImplementedError(
            "This future is supposed to be used to create callback chain"
        )

    def done(self) -> bool:
        raise NotImplementedError(
            "This future is supposed to be used to create callback chain"
        )

    def add_done_callback(
        self, callback: Callable[[torch.futures.Future[T]], None]
    ) -> None:
        raise NotImplementedError(
            "This future is supposed to be used to create callback chain"
        )

    def set_result(self, result: object) -> None:
        raise NotImplementedError(
            "This future is supposed to be used to create callback chain"
        )

    def set_exception(self, result: object) -> None:
        raise NotImplementedError(
            "This future is supposed to be used to create callback chain"
        )


class _RecoveredWork(dist._Work):
    """
    A specialized `Work` implementation that works alongside `_ManagedFuture` to create
    callback chains lazily. The callback chain is created when `wait()`, `block_current_stream()`
    or `synchronize()` are called.
    """

    def __init__(
        self,
        recover: Recover,
        work: dist._Work,
        value: object,
    ) -> None:
        super().__init__()
        # Underlying `Work` retruned from process group operations
        self._work = work

        # Used to report errors to the manager through `wrap_future()`
        self._recover = recover

        # The value returned by the final future in the callback chain
        self._value = value

        # The head of the callback chain
        self._recovered_fut_head = _RecovedFuture[object](weakref.ref(self))

        # The tail of the callback chain
        self._recovered_fut_tail: _RecovedFuture[object] = self._recovered_fut_head

        # The stream used to created the `Work` - we ensure all operations in the future
        # callback chain are executed on this stream
        self._stream: Optional[torch.Stream] = (
            torch.accelerator.current_stream()
            if torch.accelerator.is_available()
            else None
        )

        # To ensure the future callback chain is only created once
        self._is_set_future_callback_called = False

    def _set_future_callback(
        self,
    ) -> None:
        """
        Sets up the stored future callback chain.

        This method creates a chain of callbacks for the futures in the managed work,
        ensuring that each callback is executed in the proper order and with the
        appropriate stream context. It also wraps the futures with error handling
        through the manager's `wrap_future` method.

        The method is called internally when waiting or synchronizing on the work.
        """
        if self._is_set_future_callback_called:
            return

        recovered_fut: _RecovedFuture[object] = self._recovered_fut_head
        recovered_fut._fut = self._work.get_future()
        value = self._value

        is_future_wrapped = False
        while recovered_fut._next:

            def callback(
                fut: torch.futures.Future[object],
            ) -> object:
                nonlocal recovered_fut, value
                # change the stream to avoid making the callback stream
                # dependent on process group stream running the allreduce
                with get_stream_context(self._stream):
                    # Setup stream dependency
                    fut.wait()
                    assert recovered_fut._callback
                    value = recovered_fut._callback(
                        _SimpleFuture(value),
                    )
                    return value

            assert recovered_fut._fut
            fut = recovered_fut._fut.then(callback)
            assert recovered_fut._next
            recovered_fut = recovered_fut._next
            recovered_fut._fut = fut

            if is_future_wrapped:
                continue

            recovered_fut._fut = self._recover.wrap_future(recovered_fut._fut, value)
            is_future_wrapped = True

        self._value = value
        self._is_set_future_callback_called = True

    def _assert_same_stream(self) -> None:
        """
        Asserts that the current CUDA stream is the same as the one used to create this work.

        This makes sure users of the API are aware about stream dependencies.
        """
        if self._stream is not None:
            assert self._stream == torch.accelerator.current_stream()

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        self._assert_same_stream()

        try:
            with get_stream_context(self._stream):
                self._work.wait()
                self._set_future_callback()

            with get_stream_context(self._stream):
                self._recovered_fut_tail.wait()

            return True
        except Exception as e:
            self._recover.logger.exception(f"got exception waiting for work {e}")
            self._recover.report_error(e)
            return False

    def block_current_stream(self, timeout: Optional[timedelta] = None) -> None:
        self._assert_same_stream()

        with get_stream_context(self._stream):
            self._work.block_current_stream()

        self._set_future_callback()

    def synchronize(self) -> None:
        self._assert_same_stream()

        if torch.cuda.is_available():
            self.block_current_stream()
        elif torch.xpu.is_available():
            self._set_future_callback()
        else:
            # No stream dependencies need to be set
            self._set_future_callback()

    def get_future(
        self,
    ) -> torch.futures.Future[object]:
        """
        Returns:
            The tail of the managed future chain, which represents the final
            result of all the chained operations. This future will be completed when
            all the work and its callbacks have been executed.
        """
        return self._recovered_fut_tail


