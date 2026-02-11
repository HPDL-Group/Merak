# Worker 启动器使用说明

## 架构概述

```
┌─────────────────────────────────────────────────────┐
│                    管理服务端                        │
│  ┌─────────────────────────────────────────────┐   │
│  │  ProcGuard 主程序                            │   │
│  │  - ProcessMonitor 进程监控                   │   │
│  │  - FailureHandler 故障恢复                   │   │
│  │  - WebMonitor Web界面                        │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         │ HTTP API                  │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  Web Server (Flask + SocketIO)              │   │
│  │  - /api/workers/register                    │   │
│  │  - /api/workers/<id>/heartbeat             │   │
│  │  - /api/workers/<id>/command               │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
                    HTTP API
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Worker  │    │ Worker  │    │ Worker  │
    │启动器 1 │    │启动器 2 │    │启动器 3 │
    │(本地/远程)│   │(本地/远程)│   │(本地/远程)│
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │子进程   │    │子进程   │    │子进程   │
    │(实际任务)│    │(实际任务)│    │(实际任务)│
    └─────────┘    └─────────┘    └─────────┘
```

## 使用方法

### 1. 启动管理服务端

```bash
# 方式一：直接运行
python -m procguard.main --config configs/procguard_config.yaml

# 方式二：使用启动脚本
chmod +x start_manager.sh
./start_manager.sh --config configs/procguard_config.yaml

# 方式三：作为守护进程
./start_manager.sh --config configs/procguard_config.yaml --daemon
```

管理服务端默认在 `http://localhost:5000` 启动。

### 2. 启动 Worker 启动器

```bash
# 方式一：使用命令行参数
chmod +x start_worker.sh
./start_worker.sh \
    --worker-id worker1 \
    --manager-url http://localhost:5000 \
    --command "python train.py --epochs 100"

# 方式二：使用环境变量
export WORKER_ID=worker1
export MANAGER_URL=http://localhost:5000
export COMMAND="python train.py --epochs 100"
export WORKING_DIR=/path/to/project
./start_worker.sh

# 方式三：直接运行 Python 脚本
python -m procguard.worker_launcher \
    --worker-id worker1 \
    --manager-url http://localhost:5000 \
    --command "python train.py"
```

### 3. 控制 Worker

启动 Worker 后，可以通过 Web 界面或 API 进行控制：

```bash
# 启动 Worker
curl -X POST http://localhost:5000/api/workers/worker1/command -H "Content-Type: application/json" -d '{"command": "start"}'

# 停止 Worker
curl -X POST http://localhost:5000/api/workers/worker1/command -H "Content-Type: application/json" -d '{"command": "stop"}'

# 重启 Worker
curl -X POST http://localhost:5000/api/workers/worker1/command -H "Content-Type: application/json" -d '{"command": "restart"}'

# 强制停止 Worker
curl -X POST http://localhost:5000/api/workers/worker1/command -H "Content-Type: application/json" -d '{"command": "kill"}'

# 查看 Worker 状态
curl http://localhost:5000/api/remote-workers

# 查看 Worker 日志
curl http://localhost:5000/api/remote-workers/worker1/logs
```

## 多机部署示例

### 服务器 A (管理端)
```bash
# 启动管理服务端
./start_manager.sh --config configs/procguard_config.yaml
```

### 服务器 B (Worker 1)
```bash
# 启动 Worker 1
./start_worker.sh \
    --worker-id worker1 \
    --manager-url http://<服务器A的IP>:5000 \
    --command "python /path/to/train.py --model resnet50"
```

### 服务器 C (Worker 2)
```bash
# 启动 Worker 2
./start_worker.sh \
    --worker-id worker2 \
    --manager-url http://<服务器A的IP>:5000 \
    --command "python /path/to/train.py --model vit"
```

## 配置示例

### 完整配置文件 (procguard_config.yaml)
```yaml
# ProcGuard 配置文件示例

# 日志配置
logging:
  level: INFO
  file: logs/procguard.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 状态管理
state:
  state_file: data/procguard_state.yaml
  auto_save: true

# 通信配置
communication:
  adapter_type: mock  # mock 或 zmq
  adapter_config:
    host: 0.0.0.0
    port_base: 5550
    timeout: 1000

# 监控配置
monitoring:
  interval: 1.0
  heartbeat_timeout: 10.0
  zombie_detection: true
  cpu_threshold: 0.1
  memory_threshold: 0.1

# 故障恢复配置
recovery:
  enable_auto_recovery: true
  stop_all_on_failure: false
  task_reassignment: false
  recovery_timeout: 30.0
  max_recovery_attempts: 3

# Web 界面配置
web:
  enabled: true
  host: 0.0.0.0
  port: 5000
```

## 功能特点

1. **解耦设计**
   - 管理服务和 Worker 完全独立
   - Worker 可以在任意机器上运行
   - 支持跨网络通信

2. **易于扩展**
   - 新增 Worker 只需启动新的启动器
   - 无需修改管理服务端代码

3. **统一管理**
   - 所有 Worker 状态集中显示
   - 支持批量操作
   - 日志集中收集

4. **故障恢复**
   - 自动检测 Worker 状态
   - 支持自动重启失败的任务

## 注意事项

1. 管理端防火墙需要开放 5000 端口
2. Worker 需要能够访问管理端的 API
3. 建议使用 `systemd` 或 `supervisor` 管理 Worker 启动器进程
4. Worker 启动器日志可以在运行目录的 `logs` 文件夹查看
