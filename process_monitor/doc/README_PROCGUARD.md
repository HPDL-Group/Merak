# ProcGuard

一个独立的、高可用的进程监控与恢复系统，从 `torchft` 项目中剥离而来，提供通用的进程监控、故障检测和自动恢复功能。

## 核心特性

- **进程监控**: 实时监控 Worker 进程的健康状态
- **故障检测**: 自动检测进程死亡、僵尸进程和无响应状态
- **自动恢复**: 故障时自动执行恢复链：停止所有进程 → 重新分配任务 → 重启健康进程
- **任务重分配**: 将失效进程的任务智能分配给健康 Worker
- **灵活配置**: 通过 YAML 配置文件管理所有参数
- **通信抽象**: 支持多种通信适配器（ZeroMQ、Mock 等）
- **状态持久化**: 自动保存 Worker 状态和任务分配信息

## 项目结构

```
ProcGuard/
├── procguard/
│   ├── __init__.py
│   ├── main.py                 # 主程序入口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── monitor.py          # ProcessMonitor - 进程监控器
│   │   ├── manager.py          # ProcessManager - 进程管理器
│   │   ├── handler.py          # FailureHandler - 故障恢复处理器
│   │   └── state.py            # StateManager - 状态持久化管理
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py           # ConfigLoader - 配置加载器
│   │   └── schema.py           # 配置数据模型
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py             # 通信适配器基类
│   │   ├── zmq_adapter.py      # ZeroMQ 适配器
│   │   └── mock_adapter.py     # Mock 适配器
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # 日志工具
├── configs/
│   └── example_config.yaml     # 示例配置文件
├── tests/
│   ├── __init__.py
│   ├── test_monitor.py
│   ├── test_manager.py
│   └── test_handler.py
├── requirements.txt
├── setup.py
├── README.md
└── MIGRATION_GUIDE.md        # 从 torchft 迁移指南
```

## 核心模块

### 1. ProcessMonitor（进程监控器）

负责以可配置的时间间隔轮询检查所有 Worker 进程的存活状态。

**关键功能**:
- 使用 `psutil` 进行精细的进程监控
- 检测进程死亡、僵尸状态和无响应
- 可配置的 CPU/内存阈值
- 支持无 psutil 环境的降级处理

**使用示例**:
```python
from procguard.core import ProcessMonitor

monitor = ProcessMonitor(
    worker_configs={},
    monitoring_interval=1.0,
    heartbeat_timeout=10.0,
    zombie_detection=True
)

monitor.register_worker("worker_0", {})
monitor.set_failure_callback(lambda failed_ids: print(f"Failed: {failed_ids}"))
monitor.start()
```

### 2. ProcessManager（进程管理器）

负责 Worker 进程的全生命周期管理。

**关键方法**:
- `start_worker(worker_id)`: 启动一个 Worker
- `stop_worker(worker_id, force=False)`: 停止一个 Worker
- `stop_all_workers()`: 紧急停止所有 Worker
- `restart_healthy_workers(exclude_ids)`: 重启健康 Worker
- `get_worker_status(worker_id)`: 获取进程状态

**使用示例**:
```python
from procguard.core import ProcessManager

manager = ProcessManager()
manager.register_worker_config("worker_0", {
    'command': 'python train.py',
    'working_dir': '/path/to/project',
    'env': {'CUDA_VISIBLE_DEVICES': '0'}
})

manager.start_worker("worker_0")
manager.stop_all_workers(force=True)
```

### 3. FailureHandler（故障恢复处理器）

实现核心的故障恢复逻辑链。

**恢复流程**:
1. 接收故障警报
2. 记录失效上下文
3. 触发全局停止
4. 重新分配任务
5. 发起重启

**使用示例**:
```python
from procguard.core import FailureHandler

handler = FailureHandler(
    state_manager=state_manager,
    process_manager=process_manager,
    recovery_config={
        'enable_auto_recovery': True,
        'stop_all_on_failure': True,
        'task_reassignment': True
    }
)

handler.handle_failure(['worker_0'])
```

### 4. StateManager（状态管理器）

管理 Worker 状态和任务分配的持久化。

**关键功能**:
- Worker 状态跟踪
- 任务分配管理
- JSON 格式的状态持久化
- 线程安全的状态访问

## 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/procguard.git
cd procguard
pip install -e .
```

### 使用 pip 安装

```bash
pip install procguard
```

## 快速开始

### 1. 创建配置文件

创建 `configs/procguard_config.yaml`:

```yaml
workers:
  - worker_id: "worker_0"
    command: "python train.py"
    working_dir: "/path/to/project"
    env:
      CUDA_VISIBLE_DEVICES: "0"
    max_restarts: 3
    restart_delay: 5.0

monitoring:
  interval: 1.0
  heartbeat_timeout: 10.0
  zombie_detection: true

recovery:
  enable_auto_recovery: true
  stop_all_on_failure: true
  task_reassignment: true

communication:
  adapter_type: "mock"

logging:
  level: "INFO"
  file: "logs/procguard.log"
```

### 2. 启动 ProcGuard

```bash
python -m procguard.main --config configs/procguard_config.yaml
```

或使用命令行工具：

```bash
procguard --config configs/procguard_config.yaml
```

### 3. 编程方式使用

```python
from procguard import ProcGuard

procguard = ProcGuard("configs/procguard_config.yaml")
procguard.load_config()
procguard.setup_logging()
procguard.initialize_components()
procguard.start_workers()
procguard.run()
```

## 配置说明

### Worker 配置

```yaml
workers:
  - worker_id: "unique_worker_id"    # 唯一标识符
    command: "python script.py"        # 启动命令
    working_dir: "/path/to/dir"       # 工作目录
    env:                              # 环境变量
      KEY: "value"
    pid_file: "/tmp/worker.pid"       # PID 文件路径
    worker_type: "training"            # Worker 类型
    max_restarts: 3                   # 最大重启次数
    restart_delay: 5.0                # 重启延迟（秒）
    health_check_url: "http://..."     # 健康检查 URL
```

### 监控配置

```yaml
monitoring:
  interval: 1.0              # 监控间隔（秒）
  heartbeat_timeout: 10.0     # 心跳超时（秒）
  zombie_detection: true       # 启用僵尸检测
  cpu_threshold: 0.1         # CPU 使用率阈值
  memory_threshold: 0.1      # 内存使用率阈值
```

### 恢复配置

```yaml
recovery:
  enable_auto_recovery: true   # 启用自动恢复
  stop_all_on_failure: true   # 故障时停止所有 Worker
  task_reassignment: true     # 启用任务重新分配
  recovery_timeout: 60.0      # 恢复超时（秒）
  max_recovery_attempts: 3    # 最大恢复尝试次数
```

### 通信配置

```yaml
communication:
  adapter_type: "mock"       # 适配器类型: mock, zmq
  adapter_config:             # 适配器特定配置
    host: "127.0.0.1"
    port_base: 5550
    timeout: 1000
```

## 从 torchft 迁移

详细的迁移指南请参考 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)。

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black procguard/
flake8 procguard/
```

## 贡献

欢迎贡献！请先阅读贡献指南。

## 许可证

MIT License

## 联系方式

- 项目主页: https://github.com/yourusername/procguard
- 问题反馈: https://github.com/yourusername/procguard/issues

## 致谢

本项目从 [torchft](https://github.com/pytorch/torchft) 项目中剥离了进程监控组件，感谢 torchft 团队的优秀工作。
