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

#!/usr/bin/env python3
"""
Worker Launcher - Independent Worker Process Launcher

This module provides the WorkerLauncher class for receiving commands from
the ProcGuard manager and executing actual script commands. It can run
on the same machine as the manager or on different machines.

Supports SLURM cluster environments and PyTorch distributed training.

Usage:
    # Standard mode
    python worker_launcher.py --worker-id <worker_id> --manager-url <url> --command <command>

    # SLURM mode (auto-generate worker_id)
    python worker_launcher.py --slurm --manager-url <url> --command <command>

    # PyTorch distributed training mode
    python worker_launcher.py --slurm --gpu-count 4 --manager-url <url> --command "python train.py"

Environment Variables:
    WORKER_ID        - Worker identifier
    MANAGER_URL      - Manager URL
    COMMAND          - Command to execute
    WORKING_DIR      - Working directory
    HEARTBEAT_INTERVAL - Heartbeat interval in seconds
    SLURM_MODE       - Enable SLURM mode
    GPU_COUNT        - Number of GPUs per node

Classes:
    SLURMEnvDetector: SLURM environment detection utilities
    WorkerLauncher: Main worker launcher class

Example:
    >>> launcher = WorkerLauncher(
    ...     worker_id="worker-0",
    ...     manager_url="http://localhost:5000",
    ...     command="python train.py",
    ...     slurm_mode=True
    ... )
    >>> launcher.run()
"""

import os
import sys
import signal
import logging
import time
import subprocess
import threading
import argparse
import requests
import socket
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SLURMEnvDetector:
    """SLURM Environment Detection Utilities.

    Provides static methods for detecting SLURM cluster environment,
    generating worker IDs, and building PyTorch distributed training
    environment variables from SLURM environment variables.

    All methods are class methods (static) and can be called without
    instantiating the class.
    """

    @staticmethod
    def detect() -> Dict[str, Any]:
        """Detect if running in a SLURM environment.

        Examines environment variables to determine if the current
        process is running under SLURM job scheduling.

        Returns:
            Dict[str, Any]: Environment information dictionary containing:
                - is_slurm: Whether running in SLURM environment
                - job_id: SLURM job ID
                - job_name: SLURM job name
                - nodelist: List of allocated nodes
                - hostname: Current hostname
                - local_rank: Local rank on this node
                - rank: Global rank across all nodes
                - world_size: Total number of tasks
                - gpu_count: Number of GPUs per node

        Example:
            >>> env_info = SLURMEnvDetector.detect()
            >>> if env_info["is_slurm"]:
            ...     print(f"Running in SLURM job {env_info['job_id']}")
        """
        env_info = {
            "is_slurm": False,
            "job_id": None,
            "job_name": None,
            "nodelist": None,
            "hostname": socket.gethostname(),
            "local_rank": 0,
            "rank": 0,
            "world_size": 1,
            "gpu_count": 1,
        }

        if "SLURM_JOB_ID" in os.environ:
            env_info["is_slurm"] = True
            env_info["job_id"] = os.environ.get("SLURM_JOB_ID")
            env_info["job_name"] = os.environ.get("SLURM_JOB_NAME")
            env_info["nodelist"] = os.environ.get("SLURM_NODELIST")
            env_info["hostname"] = os.environ.get("SLURMD_NODENAME", socket.gethostname())

            try:
                env_info["rank"] = int(os.environ.get("SLURM_PROCID", 0))
                env_info["local_rank"] = int(os.environ.get("SLURM_LOCALID", 0))
                env_info["world_size"] = int(os.environ.get("SLURM_NTASKS", 1))
            except (ValueError, TypeError):
                pass

        return env_info

    @staticmethod
    def generate_worker_id(env_info: Dict[str, Any]) -> str:
        """Generate a unique worker ID based on environment information.

        Creates a worker ID in the format 'hostname-local_rank' for
        easy identification of workers across the cluster.

        Args:
            env_info: Environment information dictionary from detect()

        Returns:
            str: Generated worker ID string

        Example:
            >>> env = SLURMEnvDetector.detect()
            >>> worker_id = SLURMEnvDetector.generate_worker_id(env)
            >>> print(worker_id)
            compute-node-0
        """
        if env_info["is_slurm"]:
            return f"{env_info['hostname']}-{env_info['local_rank']}"
        else:
            return f"{env_info['hostname']}-{env_info['local_rank']}"

    @staticmethod
    def extract_master_node(nodelist: str) -> str:
        """Extract the first node from SLURM_NODELIST as master node.

        Parses the nodelist string which may contain node ranges
        in bracket notation (e.g., "node[1-3]") and returns the
        first node name.

        Args:
            nodelist: SLURM_NODELIST environment variable value

        Returns:
            str: First node name from the nodelist, empty string if invalid

        Example:
            >>> SLURMEnvDetector.extract_master_node("compute[1-5]")
            'compute1'
        """
        if not nodelist:
            return ""

        nodes = nodelist.split(",")
        if not nodes:
            return ""

        first_node = nodes[0].strip()

        if "[" in first_node:
            import re
            prefix = first_node.split("[")[0]
            range_match = re.search(r'\[(\d+)-(\d+)\]', first_node)
            if range_match:
                start_num = range_match.group(1)
                num_len = len(start_num)
                return f"{prefix}{start_num.zfill(num_len)}"
            return prefix
        return first_node

    @staticmethod
    def build_pytorch_env(env_info: Dict[str, Any], gpu_count: int = 1) -> Dict[str, str]:
        """Build PyTorch distributed training environment variables.

        Constructs environment variables needed for PyTorch distributed
        training using SLURM environment variables. This includes
        MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, etc.

        Args:
            env_info: Environment information from detect()
            gpu_count: Number of GPUs per node

        Returns:
            Dict[str, str]: Environment variables for PyTorch distributed training:
                - LOCAL_RANK: Local rank on this node
                - RANK: Global rank
                - WORLD_SIZE: Total number of processes
                - LOCAL_WORLD_SIZE: Processes per node
                - GPU_COUNT: GPUs per node
                - MASTER_ADDR: Address of master node
                - MASTER_PORT: Port for communication
                - SLURM_JOB_ID: SLURM job ID
                - SLURM_PROCID: SLURM process ID
                - SLURM_LOCALID: SLURM local ID

        Example:
            >>> env = SLURMEnvDetector.detect()
            >>> pytorch_env = SLURMEnvDetector.build_pytorch_env(env, gpu_count=4)
            >>> os.environ.update(pytorch_env)
        """
        local_rank = env_info.get("local_rank", 0)
        rank = env_info.get("rank", 0)
        hostname = env_info.get("hostname", "localhost")
        nodelist = env_info.get("nodelist", "")

        nnodes = int(os.environ.get("SLURM_NNODES", 1))
        ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", gpu_count))

        logging.info(f"DEBUG build_pytorch_env: nnodes={nnodes}, ntasks_per_node={ntasks_per_node}, "
                      f"gpu_count={gpu_count}, local_rank={local_rank}")

        if env_info["is_slurm"]:
            world_size = nnodes * ntasks_per_node

            if rank == 0:
                master_addr = hostname
            elif nodelist:
                master_addr = SLURMEnvDetector.extract_master_node(nodelist)
                if not master_addr:
                    master_addr = hostname
            else:
                master_addr = hostname
        else:
            ntasks_per_node = gpu_count
            world_size = gpu_count
            master_addr = hostname if rank == 0 else ""

        env_vars = {
            "LOCAL_RANK": str(local_rank),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(ntasks_per_node),
            "GPU_COUNT": str(gpu_count),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
            "SLURM_PROCID": os.environ.get("SLURM_PROCID", ""),
            "SLURM_LOCALID": os.environ.get("SLURM_LOCALID", ""),
        }

        return env_vars


class WorkerLauncher:
    """Worker Launcher for Remote Process Management.

    The WorkerLauncher class manages a single worker process that connects
    to a ProcGuard manager. It handles registration, heartbeat reporting,
    command execution, log streaming, and graceful shutdown.

    The launcher supports:
    - Registration with the ProcGuard manager
    - Heartbeat reporting at configurable intervals
    - Receiving and executing commands (start, stop, restart, kill)
    - Streaming stdout/stderr logs to the manager
    - SLURM cluster integration
    - PyTorch distributed training environment setup

    Attributes:
        worker_id: Unique identifier for this worker
        manager_url: URL of the ProcGuard manager
        command: Command to execute for the worker
        working_dir: Working directory for the process
        env: Additional environment variables
        heartbeat_interval: Seconds between heartbeats
        slurm_mode: Whether running in SLURM environment
        gpu_count: Number of GPUs per node

    Example:
        >>> launcher = WorkerLauncher(
        ...     worker_id="worker-0",
        ...     manager_url="http://manager:5000",
        ...     command="python train.py",
        ...     slurm_mode=True
        ... )
        >>> launcher.run()
    """

    def __init__(
        self,
        worker_id: str,
        manager_url: str,
        command: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        heartbeat_interval: float = 5.0,
        slurm_mode: bool = False,
        gpu_count: int = 1,
        auto_start: bool = False,
    ):
        self.worker_id = worker_id
        self.manager_url = manager_url.rstrip("/")
        self.command = command
        self.working_dir = working_dir
        self.env = env or {}
        self.heartbeat_interval = heartbeat_interval
        self.slurm_mode = slurm_mode
        self.gpu_count = gpu_count
        self.auto_start = auto_start

        self.logger = logging.getLogger(f"worker_launcher.{worker_id}")

        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()

        self._status = "stopped"
        self._last_heartbeat = None
        self._start_time = None
        self._restart_count = 0

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Configure signal handlers for graceful shutdown.

        Registers handlers for SIGINT and SIGTERM to ensure clean
        shutdown when the process receives interrupt signals.
        """
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals.

        Args:
            signum: Signal number received
            frame: Current stack frame
        """
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for manager API requests.

        Returns:
            Dict[str, str]: Headers including Content-Type and Worker-ID
        """
        return {"Content-Type": "application/json", "X-Worker-ID": self.worker_id}

    def register(self) -> bool:
        """Register this worker with the ProcGuard manager.

        Sends a registration request to the manager with worker metadata
        including worker_id, command, working_dir, and hostname.

        Returns:
            bool: True if registration successful, False otherwise

        Example:
            >>> launcher.register()
            True
        """
        try:
            url = f"{self.manager_url}/api/workers/register"
            data = {
                "worker_id": self.worker_id,
                "command": self.command,
                "working_dir": self.working_dir,
                "hostname": os.environ.get("HOSTNAME", "unknown"),
            }

            response = requests.post(url, json=data, headers=self._get_headers(), timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.logger.info(f"Registered with manager: {self.manager_url}")
                    return True

            self.logger.error(f"Failed to register with manager: {response.text}")
            return False

        except requests.RequestException as e:
            self.logger.error(f"Failed to connect to manager: {e}")
            return False

    def _get_config_hash(self) -> int:
        """Calculate hash of PyTorch configuration for change detection.

        Computes an MD5 hash of the PyTorch-related environment variables
        to detect when the distributed training configuration changes.

        Returns:
            int: Hash value of the PyTorch configuration
        """
        import hashlib
        
        pytorch_vars = {
            "MASTER_ADDR",
            "MASTER_PORT",
            "WORLD_SIZE",
            "TORCH_DISTRIBUTED_BACKEND",
            "CUDA_VISIBLE_DEVICES",
            "NCCL_SOCKET_IFNAME",
        }
        
        pytorch_config = {
            k: str(v) if v is not None else "" for k, v in self.env.items() 
            if k in pytorch_vars
        }
        
        import json
        config_str = json.dumps(pytorch_config, sort_keys=True)
        hash_value = int(hashlib.md5(config_str.encode()).hexdigest(), 16)
        
        return hash_value
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to the ProcGuard manager.

        Reports current worker status, PID, restart count, and config hash.
        Receives commands and PyTorch environment updates from the manager.

        Returns:
            bool: True if heartbeat successful, False otherwise

        Example:
            >>> launcher.send_heartbeat()
            True
        """
        try:
            url = f"{self.manager_url}/api/workers/{self.worker_id}/heartbeat"
            
            config_hash = self._get_config_hash()
            
            data = {
                "status": self._status,
                "pid": self._process.pid if self._process else None,
                "restart_count": self._restart_count,
                "config_hash": config_hash,
            }
            
            headers = self._get_headers()
            
            response = requests.post(url, json=data, headers=headers, timeout=5)
            
            if response.status_code == 200:
                self._last_heartbeat = datetime.now()
                result = response.json()
                
                pytorch_env = result.get("pytorch_env")
                if pytorch_env:
                    self._failover_pytorch_env = pytorch_env
                    self._update_pytorch_env(pytorch_env)
                
                command = result.get("command")
                if command:
                    self.logger.info(f"Received command from server: '{command}'")
                    self._handle_command(command)
                
                return True

        except requests.exceptions.Timeout:
            self.logger.error(f"Heartbeat timeout to {url}")
            return False
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Heartbeat connection error to {url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Heartbeat error: {e}")
            return False

    def _handle_command(self, command: str):
        """Process a command received from the manager.

        Executes the specified command by calling the appropriate handler.

        Args:
            command: Command to execute (start, stop, restart, kill)
        """
        failover_pytorch_env = getattr(self, '_failover_pytorch_env', None)
        if failover_pytorch_env:
            self.env.update(failover_pytorch_env)
            self._failover_pytorch_env = None

        if command == "start":
            self.logger.info(f"Command 'start' received, current status: {self._status}")
            if self._process and self._process.poll() is None:
                self.logger.info("Process already running, skipping start command")
            else:
                self._start_process()
        elif command == "stop":
            self._stop_process(force=False)
        elif command == "restart":
            self._restart_process()
        elif command == "kill":
            self._stop_process(force=True)

    def _start_process(self) -> bool:
        """Start the worker process.

        Creates a subprocess to execute the configured command with
        the specified environment and working directory.

        Returns:
            bool: True if process started successfully, False otherwise
        """
        with self._lock:
            if self._process and self._process.poll() is None:
                self.logger.warning("Process already running")
                return True

            try:
                self.logger.info(f"Starting process: {self.command}")

                env = os.environ.copy()

                failover_pytorch_env = getattr(self, '_failover_pytorch_env', None)
                if failover_pytorch_env:
                    env.update(failover_pytorch_env)
                    self._failover_pytorch_env = None

                env.update(self.env)

                env["WORKER_ID"] = self.worker_id
                env["PROCGuard_WORKER_ID"] = self.worker_id
                env["WORKER_START_TIME"] = str(time.time())
                env["PROCGuard_URL"] = self.manager_url

                working_dir = self.working_dir or "."

                cmd = self.command
                if cmd.startswith("python "):
                    cmd = cmd.replace("python ", "python -u ", 1)
                elif cmd.startswith("'") or cmd.startswith('"'):
                    cmd = f"python -u -c {cmd}"
                else:
                    cmd = f"python -u {cmd}"
                
                self._process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=working_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )

                self._status = "running"
                self._start_time = time.time()
                self._restart_count += 1

                self.logger.info(f"Process started with PID: {self._process.pid}")

                self._start_output_reader()

                self._send_immediate_heartbeat()

                return True

            except Exception as e:
                self.logger.error(f"Failed to start process: {e}")
                self._status = "failed"
                return False

    def _stop_process(self, force: bool = False) -> bool:
        """Stop the worker process and its children.

        Terminates the subprocess gracefully by default. If force is True,
        sends SIGKILL instead of SIGTERM. Uses psutil if available for
        reliable process tree termination.

        Args:
            force: If True, kill immediately; otherwise, terminate gracefully

        Returns:
            bool: True if process stopped successfully, False otherwise
        """
        with self._lock:
            if not self._process:
                self.logger.warning("No process to stop")
                return True

            try:
                pid = self._process.pid
                self.logger.info(f"Stopping process (PID: {pid}, force={force})")

                if PSUTIL_AVAILABLE:
                    try:
                        process = psutil.Process(pid)
                        children = process.children(recursive=True)
                        
                        if force:
                            for child in children:
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            process.kill()
                        else:
                            process.terminate()
                            for child in children:
                                try:
                                    child.terminate()
                                except psutil.NoSuchProcess:
                                    pass

                            try:
                                process.wait(timeout=5)
                            except psutil.TimeoutExpired:
                                self.logger.warning("Process did not terminate, killing")
                                for child in children:
                                    try:
                                        child.kill()
                                    except psutil.NoSuchProcess:
                                        pass
                                process.kill()
                                process.wait(timeout=5)
                    except psutil.NoSuchProcess:
                        self.logger.warning(f"Process {pid} already gone")
                else:
                    import signal
                    
                    if force:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    else:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        try:
                            self._process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.logger.warning("Process did not terminate, killing")
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                            self._process.wait(timeout=5)

                self._status = "stopped"
                self._process = None

                self.logger.info("Process stopped")
                return True

            except Exception as e:
                self.logger.error(f"Failed to stop process: {e}")
                return False

    def _restart_process(self, new_config: Optional[Dict[str, str]] = None):
        """Restart the worker process.

        Stops the current process and starts a new one with the same
        or updated configuration.

        Args:
            new_config: Optional dict of environment variables to update
        """
        if new_config:
            self.logger.info(f"[PyTorch] Updating environment via _restart_process")
            self._update_pytorch_env(new_config)

        self.logger.info("Restarting process")
        self._stop_process(force=True)
        time.sleep(1)
        self._start_process()

    def _update_pytorch_env(self, pytorch_env: Dict[str, str]) -> bool:
        """Update PyTorch distributed training environment variables.

        Unified interface for updating PyTorch environment variables from server.
        Handles rank changes, detects config changes, and logs updates.

        Args:
            pytorch_env: Dictionary of PyTorch environment variables
                - RANK: Global rank of this worker
                - LOCAL_RANK: Local rank on this node
                - NODE_RANK: Rank of this node in the cluster
                - WORLD_SIZE: Total number of workers
                - MASTER_ADDR: Address of master node
                - MASTER_PORT: Port for master communication
                - TORCH_DISTRIBUTED_BACKEND: Distributed backend (nccl, gloo)

        Returns:
            bool: True if environment was updated, False otherwise
        """
        if not pytorch_env:
            return False

        old_rank = self.env.get("RANK")
        new_rank = pytorch_env.get("RANK")

        changed_keys = []
        for key, value in pytorch_env.items():
            old_value = self.env.get(key)
            if old_value != value:
                changed_keys.append(f"{key}: {old_value} -> {value}")
                self.env[key] = value

        if changed_keys:
            self.logger.info(f"[PyTorch] Updated environment variables:")
            for change in changed_keys:
                self.logger.info(f"  {change}")

            if old_rank is not None and new_rank is not None and old_rank != new_rank:
                self.logger.info(f"[PyTorch] RANK changed: {old_rank} -> {new_rank}")

        return len(changed_keys) > 0

    def _start_output_reader(self):
        """Start background threads to read process stdout/stderr.

        Creates daemon threads that continuously read from the process's
        stdout and stderr pipes, logging messages and forwarding them
        to the manager.
        """

        def read_output(pipe, stream_type):
            try:
                line_count = 0
                for line in iter(pipe.readline, b""):
                    if line:
                        line_count += 1
                        message = line.decode("utf-8", errors="replace").strip()
                        self.logger.info(f"[{stream_type}] {message}")
                        self._send_log(message, stream_type)
                self.logger.debug(f"{stream_type} pipe closed, total lines: {line_count}")
            except Exception as e:
                self.logger.debug(f"Error reading {stream_type}: {e}")
            finally:
                pipe.close()

        if self._process:
            if self._process.stdout:
                threading.Thread(
                    target=read_output, args=(self._process.stdout, "stdout"), daemon=True
                ).start()

            if self._process.stderr:
                threading.Thread(
                    target=read_output, args=(self._process.stderr, "stderr"), daemon=True
                ).start()
        else:
            self.logger.warning("Cannot start output reader: no process")

    def _send_log(self, message: str, level: str = "info"):
        """Send a log message to the ProcGuard manager.

        Forwards process output logs to the manager for centralized
        logging and display in the web interface.

        Args:
            message: Log message content
            level: Log level (info, warning, error, debug)
        """
        try:
            url = f"{self.manager_url}/api/workers/{self.worker_id}/logs"
            data = {"message": message, "level": level, "timestamp": datetime.now().isoformat()}

            response = requests.post(url, json=data, headers=self._get_headers(), timeout=5)
            if response.status_code != 200:
                self.logger.warning(f"Failed to send log: {response.text}")
        except requests.exceptions.ConnectionError:
            self.logger.debug(f"Cannot connect to manager for logs")
        except requests.exceptions.Timeout:
            self.logger.debug(f"Log request timeout")
        except Exception as e:
            self.logger.debug(f"Error sending log: {e}")

    def _send_immediate_heartbeat(self):
        """Send an immediate heartbeat to report running status.

        Called right after process starts to immediately update
        the manager with the running status and PID.
        """
        try:
            status = self._status or "running"
            pid = self._process.pid if self._process else None

            heartbeat_data = {
                "status": status,
                "pid": pid,
                "restart_count": self._restart_count,
            }

            url = f"{self.manager_url}/api/workers/{self.worker_id}/heartbeat"
            response = requests.post(
                url, json=heartbeat_data, headers=self._get_headers(), timeout=5
            )
            if response.status_code == 200:
                self.logger.debug(f"[Heartbeat] Immediate heartbeat sent: status={status}, pid={pid}")
            else:
                self.logger.warning(f"[Heartbeat] Failed to send immediate heartbeat: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.logger.debug(f"[Heartbeat] Cannot connect to manager for immediate heartbeat")
        except requests.exceptions.Timeout:
            self.logger.debug(f"[Heartbeat] Immediate heartbeat request timeout")
        except Exception as e:
            self.logger.debug(f"[Heartbeat] Error sending immediate heartbeat: {e}")

    def _check_process_status(self):
        """Check and update the process status.

        Polls the subprocess to determine if it has exited and updates
        the internal status accordingly. Called periodically in the main loop.
        """
        with self._lock:
            if not self._process:
                return

            poll_result = self._process.poll()
            if poll_result is not None:
                exit_code = self._process.returncode
                
                if self._status == "running":
                    self._status = "exited"
                    if exit_code == 0:
                        self.logger.info(f"Process exited normally (code: 0)")
                    else:
                        self.logger.warning(f"Process exited with code: {exit_code}")
                
                self._process = None

    def run(self):
        """Run the main worker launcher loop.

        Registers with the manager, starts the heartbeat thread, and
        enters the main event loop that handles process management
        and communication with the manager.

        Returns:
            bool: True if the loop completed normally, False on registration failure

        Example:
            >>> launcher = WorkerLauncher(...)
            >>> launcher.run()
        """
        self.logger.info(f"Worker Launcher starting for {self.worker_id}")
        self.logger.info(f"Auto-start: {self.auto_start} (等待前端手动启动)")

        if not self.register():
            self.logger.error("Failed to register with manager, exiting")
            return False

        self.logger.info("Registration successful, entering main loop")
        self._running = True

        if self.auto_start:
            self.logger.info("Auto-start enabled, launching process")
            self._start_process()

        self.logger.info(f"Starting heartbeat loop (interval={self.heartbeat_interval}s)")
        
        try:
            while self._running and not self._shutdown_event.is_set():
                self._check_process_status()

                if not self.send_heartbeat():
                    self.logger.warning("Heartbeat failed, will retry")

                time.sleep(self.heartbeat_interval)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.shutdown()

        return True

    def shutdown(self):
        """Gracefully shutdown the worker launcher.

        Stops the running process, sends unregistration request to
        the manager, and cleans up resources.
        """
        if self._running:
            self.logger.info("Shutting down Worker Launcher")
            self._running = False
            self._shutdown_event.set()

            self._stop_process(force=True)

            try:
                requests.post(
                    f"{self.manager_url}/api/workers/{self.worker_id}/unregister",
                    headers=self._get_headers(),
                    timeout=5,
                )
            except requests.RequestException:
                pass


def parse_env() -> Dict[str, str]:
    """Parse configuration from environment variables.

    Reads configuration values from environment variables for
    COMMAND, WORKING_DIR, and HEARTBEAT_INTERVAL.

    Returns:
        Dict[str, str]: Configuration dictionary with environment values

    Example:
        >>> config = parse_env()
        >>> if "COMMAND" in config:
        ...     print(f"Command: {config['COMMAND']}")
    """
    env = {}
    for key in ["COMMAND", "WORKING_DIR", "HEARTBEAT_INTERVAL"]:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def main():
    """Main entry point for the Worker Launcher.

    Parses command-line arguments, creates a WorkerLauncher instance,
    and starts the main event loop.

    Command-line Arguments:
        --worker-id: Unique worker identifier
        --manager-url: URL of the ProcGuard manager
        --command: Command to execute for the worker
        --working-dir: Working directory for the process
        --env: Environment variables in KEY=VALUE format
        --heartbeat-interval: Seconds between heartbeats (default: 5.0)
        --slurm: Enable SLURM mode for cluster deployment
        --gpu-count: Number of GPUs per node (default: 1)
        --auto-start: Automatically start the process on launch

    Example:
        $ python worker_launcher.py --worker-id worker-0 \\
            --manager-url http://manager:5000 \\
            --command "python train.py"

        $ python worker_launcher.py --slurm --gpu-count 4 \\
            --manager-url http://manager:5000 \\
            --command "python -m torch.distributed.run train.py"
    """
    parser = argparse.ArgumentParser(
        description="Worker Launcher - Independent Worker Process Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using command-line arguments
    python worker_launcher.py --worker-id worker1 --manager-url http://localhost:5000 --command "python train.py"
    
    # Using environment variables
    export WORKER_ID=worker1
    export MANAGER_URL=http://localhost:5000
    export COMMAND="python train.py"
    python worker_launcher.py
    
    # SLURM mode (auto-detect and generate worker_id)
    python worker_launcher.py --slurm --manager-url http://manager:5000 --command "python train.py"
    
    # PyTorch distributed training mode
    python worker_launcher.py --slurm --gpu-count 4 --manager-url http://manager:5000 --command "python -m torch.distributed.run train.py"
        """,
    )

    parser.add_argument("--worker-id", type=str, help="Worker identifier")
    parser.add_argument("--manager-url", type=str, help="ProcGuard manager URL")
    parser.add_argument("--command", type=str, help="Command to execute")
    parser.add_argument("--working-dir", type=str, default=None, help="Working directory")
    parser.add_argument("--env", type=str, default=None, help="Environment variables (JSON format)")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0, help="Heartbeat interval in seconds")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Enable SLURM mode, auto-detect SLURM environment and generate worker_id",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs per node (for PyTorch distributed training)",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically start command after launch (default: off, requires manual start)",
    )

    args = parser.parse_args()

    slurm_mode = args.slurm or os.environ.get("SLURM_MODE", "").lower() in ("true", "1", "yes")
    gpu_count = int(args.gpu_count or os.environ.get("GPU_COUNT", 1))
    auto_start = args.auto_start or os.environ.get("AUTO_START", "").lower() in ("true", "1", "yes")

    worker_id = args.worker_id or os.environ.get("WORKER_ID")
    manager_url = args.manager_url or os.environ.get("MANAGER_URL")
    command = args.command or os.environ.get("COMMAND")
    working_dir = args.working_dir or os.environ.get("WORKING_DIR")
    heartbeat_interval = float(args.heartbeat_interval or os.environ.get("HEARTBEAT_INTERVAL", 5.0))

    if not manager_url:
        parser.error("--manager-url 或 MANAGER_URL 环境变量必须指定")

    if not command:
        parser.error("--command or COMMAND environment variable must be specified")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if slurm_mode:
        slurm_env = SLURMEnvDetector.detect()
        worker_id = SLURMEnvDetector.generate_worker_id(slurm_env)
        logging.info(f"SLURM mode: worker_id = {worker_id}, rank = {slurm_env['rank']}/{slurm_env['world_size']}")

    if not worker_id:
        parser.error("--worker-id or WORKER_ID environment variable must be specified")

    env = {}
    if args.env:
        import json

        try:
            env = json.loads(args.env)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid env JSON format: {e}")

    if slurm_mode:
        slurm_env = SLURMEnvDetector.detect()
        slurm_vars = SLURMEnvDetector.build_pytorch_env(slurm_env, gpu_count)
        env.update(slurm_vars)
        logging.info(f"SLURM env vars: LOCAL_RANK={env.get('LOCAL_RANK')}, RANK={env.get('RANK')}, "
                     f"MASTER_ADDR={env.get('MASTER_ADDR')}, WORLD_SIZE={env.get('WORLD_SIZE')}")
    
    try:
        import requests
        response = requests.get(f"{manager_url}/api/pytorch/env", timeout=5)
        if response.status_code == 200:
            pytorch_env = response.json()
            env.update(pytorch_env)
            logging.info(f"Applied PyTorch config env vars: {pytorch_env}")
    except Exception as e:
        logging.warning(f"Could not fetch PyTorch config from manager: {e}")
    
    if slurm_mode:
        logging.info(f"Full environment variables: {env}")
        logging.info(f"Added PyTorch distributed training env vars: MASTER_ADDR={env.get('MASTER_ADDR', 'N/A')}, WORLD_SIZE={env.get('WORLD_SIZE', 'N/A')}")

    launcher = WorkerLauncher(
        worker_id=worker_id,
        manager_url=manager_url,
        command=command,
        working_dir=working_dir,
        env=env,
        heartbeat_interval=heartbeat_interval,
        slurm_mode=slurm_mode,
        gpu_count=gpu_count,
        auto_start=auto_start,
    )

    success = launcher.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
