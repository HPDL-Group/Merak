# 从 torchft 迁移到 ProcGuard 的说明文档

## 概述

本文档说明如何从 `torchft` 项目中剥离进程监控组件，构建独立的 `ProcGuard` 系统。ProcGuard 保留了 torchft 的核心监控逻辑，同时提供了更通用、解耦和可配置的架构。

## 代码边界识别

### 1. torchft 中的关键组件

在 torchft 项目中，进程监控和恢复相关的核心代码位于以下位置：

#### Manager 类 (torchft/manager.py)
- **职责**: 管理完整的容错训练循环
- **关键方法**:
  - `start_quorum()`: 计算新的 quorum 并准备新步骤
  - `wait_quorum()`: 等待 quorum 完成
  - `_async_quorum()`: 异步执行 quorum 逻辑
  - `allreduce()`: 容错的 allreduce 操作
- **监控逻辑**: 通过 heartbeat_interval 参数控制心跳间隔（默认 100ms）

#### 进程监控 (torchft/examples/slurm/runner.py)
- **职责**: 监控 SLURM 作业状态
- **关键方法**:
  - `monitor()`: 检查作业状态并重启失败的副本
- **监控策略**: 轮询检查作业状态，检测到失败时自动重启

#### 多进程管理 (torchft/multiprocessing.py)
- **职责**: 提供受监控的管道通信
- **关键类**: `_MonitoredPipe`
- **功能**: 带超时的管道通信，用于进程间通信

### 2. ProcGuard 中的对应模块

| torchft 组件 | ProcGuard 模块 | 功能映射 |
|-------------|---------------|---------|
| Manager.quorum | ProcessMonitor | 进程健康检查 |
| runner.monitor | ProcessMonitor | 轮询监控 |
| Manager._async_quorum | FailureHandler | 故障恢复逻辑 |
| - | ProcessManager | 进程生命周期管理 |
| - | StateManager | 状态持久化 |
| - | CommunicationAdapter | 通信抽象 |

## 核心逻辑提取

### 1. 进程健康检查

从 torchft 提取的核心逻辑：

```python
# torchft 中的心跳机制
heartbeat_interval: timedelta = timedelta(milliseconds=100)

# ProcGuard 中的实现
class ProcessMonitor:
    def _check_worker_with_psutil(self, worker_id: str, pid: Optional[int]):
        process = psutil.Process(pid)
        
        # 检查进程是否运行
        if not process.is_running():
            return HealthStatus.DEAD
        
        # 检查 CPU 和内存使用率
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_percent = process.memory_percent()
        
        # 僵尸进程检测
        if self.zombie_detection:
            is_zombie = self._detect_zombie_process(process, cpu_percent, memory_percent)
```

**关键改进**:
- 支持无 psutil 环境的降级处理
- 增强的僵尸进程检测
- 可配置的 CPU/内存阈值

### 2. 故障恢复流程

从 torchft 提取的恢复逻辑：

```python
# torchft 中的恢复逻辑
def _async_quorum(self, allow_heal: bool, ...):
    # 1. 计算 quorum
    quorum = self._client._quorum(...)
    
    # 2. 重新配置 ProcessGroup
    self._pg.configure(...)
    
    # 3. 恢复检查点
    if allow_heal and heal:
        self._checkpoint_transport.recv_checkpoint(...)
```

**ProcGuard 中的增强实现**:

```python
def _handle_single_failure(self, failed_worker_id: str):
    # 1. 记录失效上下文
    self._record_failure_context(failed_worker_id)
    
    # 2. 紧急停止所有 Worker
    if self._stop_all_on_failure:
        self._emergency_stop_all(recovery_context)
    
    # 3. 重新分配任务
    if self._task_reassignment:
        target_worker_id = self._reassign_tasks(failed_worker_id, recovery_context)
    
    # 4. 重启健康 Worker
    self._restart_workers(failed_worker_id, recovery_context)
```

**关键增强**:
- 添加了全局停止机制（stop_all）
- 实现了任务重新分配逻辑
- 支持可配置的恢复策略

### 3. 进程生命周期管理

ProcGuard 新增的进程管理功能：

```python
class ProcessManager:
    def start_worker(self, worker_id: str) -> bool:
        # 启动 Worker 进程
        process = self._launch_process(command, working_dir, env)
        
    def stop_worker(self, worker_id: str, force: bool = False) -> bool:
        # 优雅停止或强制终止
        
    def stop_all_workers(self, force: bool = False) -> Dict[str, bool]:
        # 停止所有 Worker（核心功能）
        
    def restart_healthy_workers(self, exclude_ids: List[str]) -> Dict[str, bool]:
        # 重启健康 Worker（核心功能）
```

## 抽象与解耦

### 1. 配置管理

**torchft**: 硬编码配置和环境变量
```python
# torchft 中的配置方式
os.environ.get("TORCHFT_MANAGER_PORT", "0")
os.environ.get("TORCHFT_LIGHTHOUSE", "http://slurm-head-node-0:29510")
```

**ProcGuard**: YAML 配置文件
```yaml
workers:
  - worker_id: "worker_0"
    command: "python train.py"
    env:
      CUDA_VISIBLE_DEVICES: "0,1"

monitoring:
  interval: 1.0
  heartbeat_timeout: 10.0

recovery:
  enable_auto_recovery: true
  stop_all_on_failure: true
```

### 2. 通信适配器

**torchft**: 直接使用 Rust 绑定的 ManagerClient
```python
from torchft._torchft import ManagerClient, ManagerServer
self._client = ManagerClient(addr, connect_timeout=connect_timeout)
```

**ProcGuard**: 抽象的通信适配器接口
```python
class CommunicationAdapter(ABC):
    @abstractmethod
    def send_command(self, worker_id: str, command: str, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def send_heartbeat(self, worker_id: str) -> bool:
        pass
```

**支持的适配器**:
- `MockCommunicationAdapter`: 用于测试和开发
- `ZMQCommunicationAdapter`: 基于 ZeroMQ 的实现
- 可扩展支持 gRPC、Redis 等

### 3. 状态管理

**torchft**: 依赖外部存储（TCPStore、Lighthouse）
```python
self._store = TCPStore(host_name=store_addr, port=store_port, ...)
```

**ProcGuard**: 内置的 JSON 状态持久化
```python
class StateManager:
    def _save_state(self):
        data = {
            'workers': {...},
            'tasks': {...},
            'task_queue': [...]
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f)
```

## 兼容性考虑

### 1. 进程识别

**torchft**: 使用 replica_id 和 group_rank
```python
replica_id = os.environ.get("REPLICA_ID")
group_rank = int(os.environ["RANK"])
```

**ProcGuard**: 使用 worker_id
```python
worker_id: str  # 唯一标识符
pid: Optional[int]  # 进程 ID
```

**迁移建议**: 在配置中映射 replica_id 到 worker_id

### 2. 心跳机制

**torchft**: 通过 ManagerServer 和 ManagerClient
```python
ManagerServer(
    replica_id=replica_id,
    heartbeat_interval=heartbeat_interval,
    ...
)
```

**ProcGuard**: 通过 ProcessMonitor 和 CommunicationAdapter
```python
ProcessMonitor(
    heartbeat_timeout=10.0,
    ...
)
communication_adapter.send_heartbeat(worker_id)
```

**迁移建议**: 保持相同的心跳间隔配置

### 3. 检查点恢复

**torchft**: 使用 CheckpointTransport
```python
self._checkpoint_transport.recv_checkpoint(
    src_rank=recover_src_replica_rank,
    metadata=checkpoint_metadata,
    ...
)
```

**ProcGuard**: 任务重新分配
```python
state_manager.reassign_tasks(failed_worker_id, target_worker_id)
```

**迁移建议**: 如果需要检查点恢复，可以实现专门的 CheckpointAdapter

## 潜在问题与解决方案

### 1. 进程组通信

**问题**: torchft 依赖 PyTorch 的 ProcessGroup 进行分布式训练

**解决方案**: ProcGuard 不直接管理 ProcessGroup，而是监控训练进程。训练进程内部继续使用 torchft 的 ProcessGroup 管理。

### 2. Lighthouse 依赖

**问题**: torchft 需要 Lighthouse 服务进行 quorum 协调

**解决方案**: ProcGuard 实现了简化的故障恢复逻辑，不依赖外部服务。如需要 quorum，可以实现 LighthouseAdapter。

### 3. 性能开销

**问题**: 额外的监控层可能影响性能

**解决方案**:
- 可配置的监控间隔
- 异步监控线程
- 最小化监控操作的影响

## 迁移步骤

### 1. 配置迁移

将 torchft 的环境变量转换为 ProcGuard 配置：

```python
# torchft 环境变量
TORCHFT_MANAGER_PORT=29500
TORCHFT_LIGHTHOUSE=http://lighthouse:29510
TORCHFT_TIMEOUT_SEC=60

# ProcGuard 配置
communication:
  adapter_type: "zmq"
  adapter_config:
    port_base: 29500

monitoring:
  interval: 1.0
  heartbeat_timeout: 10.0
```

### 2. Worker 配置迁移

```python
# torchft 启动命令
torchrun --nproc_per_node=4 train_ddp.py

# ProcGuard Worker 配置
workers:
  - worker_id: "training_worker"
    command: "torchrun --nproc_per_node=4 train_ddp.py"
    env:
      CUDA_VISIBLE_DEVICES: "0,1,2,3"
```

### 3. 集成到现有系统

```python
from procguard import ProcGuard

# 创建 ProcGuard 实例
procguard = ProcGuard("configs/procguard_config.yaml")
procguard.load_config()
procguard.setup_logging()
procguard.initialize_components()

# 启动监控
procguard.start_workers()
procguard.run()
```

## 测试建议

1. **单元测试**: 测试各个模块的独立功能
2. **集成测试**: 测试故障恢复流程
3. **性能测试**: 验证监控开销
4. **压力测试**: 模拟多个 Worker 同时故障

## 总结

ProcGuard 成功地从 torchft 中剥离了进程监控组件，提供了：

1. **更通用的架构**: 不依赖 PyTorch，可用于任何进程监控场景
2. **更好的可配置性**: 通过 YAML 配置文件灵活配置
3. **增强的故障恢复**: 支持全局停止和任务重新分配
4. **模块化设计**: 各组件职责清晰，易于扩展和维护

通过遵循本迁移指南，您可以顺利地将 torchft 的监控逻辑迁移到 ProcGuard，并获得更强大和灵活的进程监控能力。
