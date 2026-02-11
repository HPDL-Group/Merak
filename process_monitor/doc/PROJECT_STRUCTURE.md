# ProcGuard 项目结构树

```
ProcGuard/
│
├── procguard/                          # 主包目录
│   ├── __init__.py                     # 包初始化文件
│   ├── main.py                         # 主程序入口
│   │
│   ├── core/                           # 核心模块
│   │   ├── __init__.py                # 核心模块导出
│   │   ├── monitor.py                 # ProcessMonitor - 进程监控器
│   │   │   ├── HealthStatus           # 健康状态枚举
│   │   │   ├── WorkerHealthReport     # 健康报告数据类
│   │   │   └── ProcessMonitor        # 监控器主类
│   │   │
│   │   ├── manager.py                 # ProcessManager - 进程管理器
│   │   │   ├── WorkerProcessInfo     # 进程信息数据类
│   │   │   └── ProcessManager        # 管理器主类
│   │   │
│   │   ├── handler.py                # FailureHandler - 故障恢复处理器
│   │   │   ├── RecoveryStage          # 恢复阶段枚举
│   │   │   ├── RecoveryContext        # 恢复上下文数据类
│   │   │   └── FailureHandler        # 处理器主类
│   │   │
│   │   └── state.py                  # StateManager - 状态管理器
│   │       ├── WorkerStatus           # Worker 状态枚举
│   │       ├── WorkerState           # Worker 状态数据类
│   │       ├── TaskAssignment        # 任务分配数据类
│   │       └── StateManager         # 状态管理器主类
│   │
│   ├── config/                         # 配置管理模块
│   │   ├── __init__.py              # 配置模块导出
│   │   ├── schema.py                # 配置数据模型
│   │   │   ├── WorkerConfig         # Worker 配置
│   │   │   ├── MonitoringConfig     # 监控配置
│   │   │   ├── RecoveryConfig       # 恢复配置
│   │   │   ├── CommunicationConfig  # 通信配置
│   │   │   ├── LoggingConfig        # 日志配置
│   │   │   ├── StateConfig          # 状态配置
│   │   │   └── ProcGuardConfig     # 主配置类
│   │   └── loader.py               # ConfigLoader - 配置加载器
│   │
│   ├── adapters/                      # 通信适配器模块
│   │   ├── __init__.py             # 适配器模块导出
│   │   ├── base.py                 # CommunicationAdapter - 适配器基类
│   │   ├── mock_adapter.py         # MockCommunicationAdapter - Mock 适配器
│   │   └── zmq_adapter.py          # ZMQCommunicationAdapter - ZeroMQ 适配器
│   │
│   └── utils/                         # 工具模块
│       ├── __init__.py              # 工具模块导出
│       └── logger.py               # 日志工具
│           ├── setup_logging()       # 设置日志
│           └── get_logger()         # 获取日志器
│
├── configs/                           # 配置文件目录
│   └── example_config.yaml         # 示例配置文件
│
├── tests/                             # 测试目录
│   ├── __init__.py
│   ├── test_monitor.py
│   ├── test_manager.py
│   └── test_handler.py
│
├── requirements.txt                    # Python 依赖
├── setup.py                          # 安装脚本
├── README_PROCGUARD.md               # ProcGuard README
├── MIGRATION_GUIDE.md                # 从 torchft 迁移指南
└── PROJECT_STRUCTURE.md              # 本文件
```

## 文件说明

### 核心模块 (procguard/core/)

#### monitor.py - 进程监控器
- **ProcessMonitor**: 主监控类，负责轮询检查 Worker 进程状态
- **HealthStatus**: 健康状态枚举（HEALTHY, UNRESPONSIVE, DEAD, ZOMBIE, UNKNOWN）
- **WorkerHealthReport**: 健康报告数据类，包含进程状态、CPU/内存使用率等

**关键方法**:
- `register_worker()`: 注册 Worker 进行监控
- `start()`: 启动监控循环
- `stop()`: 停止监控
- `get_health_report()`: 获取 Worker 健康报告
- `get_failed_workers()`: 获取失败的 Worker 列表

#### manager.py - 进程管理器
- **ProcessManager**: 主管理类，负责 Worker 进程的全生命周期管理
- **WorkerProcessInfo**: 进程信息数据类，包含进程对象、PID、状态等

**关键方法**:
- `start_worker()`: 启动一个 Worker
- `stop_worker()`: 停止一个 Worker
- `stop_all_workers()`: 紧急停止所有 Worker（核心功能）
- `restart_worker()`: 重启一个 Worker
- `restart_healthy_workers()`: 重启健康 Worker（核心功能）
- `get_worker_status()`: 获取 Worker 状态
- `cleanup_dead_workers()`: 清理死进程

#### handler.py - 故障恢复处理器
- **FailureHandler**: 主处理器类，实现故障恢复逻辑链
- **RecoveryStage**: 恢复阶段枚举（DETECTED, STOPPING, REASSIGNING, RESTARTING, COMPLETED, FAILED）
- **RecoveryContext**: 恢复上下文数据类，记录恢复过程信息

**关键方法**:
- `handle_failure()`: 处理 Worker 故障
- `_emergency_stop_all()`: 紧急停止所有 Worker
- `_reassign_tasks()`: 重新分配任务
- `_restart_workers()`: 重启 Worker
- `get_recovery_stats()`: 获取恢复统计信息

#### state.py - 状态管理器
- **StateManager**: 主状态管理类，负责状态持久化
- **WorkerStatus**: Worker 状态枚举（UNKNOWN, STARTING, RUNNING, STOPPED, FAILED, ZOMBIE）
- **WorkerState**: Worker 状态数据类
- **TaskAssignment**: 任务分配数据类

**关键方法**:
- `register_worker()`: 注册 Worker
- `update_worker_status()`: 更新 Worker 状态
- `assign_task()`: 分配任务给 Worker
- `reassign_tasks()`: 重新分配任务
- `get_state_summary()`: 获取状态摘要

### 配置模块 (procguard/config/)

#### schema.py - 配置数据模型
定义所有配置数据类，使用 Python dataclass 实现。

#### loader.py - 配置加载器
- **ConfigLoader**: 配置加载器类，负责从 YAML 文件加载和验证配置

**关键方法**:
- `load()`: 加载配置文件
- `validate()`: 验证配置有效性

### 适配器模块 (procguard/adapters/)

#### base.py - 适配器基类
- **CommunicationAdapter**: 通信适配器抽象基类

**抽象方法**:
- `send_command()`: 发送命令
- `send_heartbeat()`: 发送心跳
- `receive_status()`: 接收状态
- `broadcast_command()`: 广播命令
- `shutdown()`: 关闭适配器

#### mock_adapter.py - Mock 适配器
用于测试和开发的 Mock 实现，不依赖外部服务。

#### zmq_adapter.py - ZeroMQ 适配器
基于 ZeroMQ 的实现，支持进程间通信。

### 工具模块 (procguard/utils/)

#### logger.py - 日志工具
- `setup_logging()`: 设置日志系统
- `get_logger()`: 获取日志器实例

### 主程序 (procguard/main.py)

- **ProcGuard**: 主程序类，协调所有组件
- `main()`: 命令行入口函数

**关键方法**:
- `load_config()`: 加载配置
- `setup_logging()`: 设置日志
- `initialize_components()`: 初始化所有组件
- `start_workers()`: 启动所有 Worker
- `run()`: 运行主循环
- `shutdown()`: 关闭系统

## 配置文件 (configs/)

### example_config.yaml
示例配置文件，展示如何配置 3 个不同类型的 Worker。

## 文档文件

- **README_PROCGUARD.md**: ProcGuard 项目说明文档
- **MIGRATION_GUIDE.md**: 从 torchft 迁移指南
- **PROJECT_STRUCTURE.md**: 本文件，项目结构说明

## 安装文件

- **requirements.txt**: Python 依赖列表
- **setup.py**: Python 包安装脚本

## 测试文件 (tests/)

- **test_monitor.py**: ProcessMonitor 测试
- **test_manager.py**: ProcessManager 测试
- **test_handler.py**: FailureHandler 测试

## 使用流程

1. 创建配置文件（参考 `configs/example_config.yaml`）
2. 运行 ProcGuard: `python -m procguard.main --config configs/your_config.yaml`
3. ProcGuard 自动启动所有 Worker 并开始监控
4. 检测到故障时自动执行恢复流程

## 扩展点

1. **通信适配器**: 继承 `CommunicationAdapter` 实现新的通信方式
2. **监控策略**: 扩展 `ProcessMonitor` 添加自定义健康检查
3. **恢复策略**: 扩展 `FailureHandler` 实现自定义恢复逻辑
4. **状态存储**: 扩展 `StateManager` 支持其他存储后端
