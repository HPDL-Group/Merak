# ProcGuard

一个独立的、高可用的进程监控与恢复系统，从 `torchft` 项目中剥离而来，提供通用的进程监控、故障检测和自动恢复功能。

## 核心特性

- **进程监控**: 实时监控 Worker 进程的健康状态
- **故障检测**: 自动检测进程死亡、僵尸进程和无响应状态
- **自动恢复**: 故障时自动执行恢复链
- **解耦架构**: 管理服务与 Worker 分离，支持分布式部署
- **Web 监控面板**: 实时网页监控界面，支持 WebSocket 实时推送

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动管理服务端

```bash
./start_manager.sh --config configs/procguard_config.yaml
```

### 3. 启动 Worker

```bash
./start_worker.sh \
    --worker-id worker1 \
    --manager-url http://localhost:5000 \
    --command "python examples/train_ddp.py --epochs 100"
```

### 4. 访问 Web 界面

打开浏览器访问: `http://localhost:5000`

## 示例

项目提供了多个示例程序，演示 ProcGuard 的各种使用场景：

| 示例文件 | 说明 |
|----------|------|
| [examples/train_ddp.py](examples/train_ddp.py) | 分布式训练示例，模拟分布式训练任务 |
| [examples/fake_worker.py](examples/fake_worker.py) | 模拟 Worker 示例，用于测试监控功能 |
| [examples/worker_launcher.py](examples/worker_launcher.py) | Worker 启动器示例，支持多机部署 |
| [examples/test_worker.py](examples/test_worker.py) | 简单测试 Worker |
| [examples/test_launch.py](examples/test_launch.py) | 集成测试启动脚本 |

运行示例：

```bash
# 启动管理服务
./start_manager.sh

# 启动示例 Worker
./start_worker.sh --worker-id worker1 --manager-url http://localhost:5000 --command "python examples/fake_worker.py"

# 或运行训练示例
./start_worker.sh --worker-id trainer1 --manager-url http://localhost:5000 --command "python examples/train_ddp.py --epochs 100"

## 文档

详细文档请参考 `doc/` 目录：

| 文档 | 说明 |
|------|------|
| [doc/PYTORCH_DISTRIBUTED.md](doc/PYTORCH_DISTRIBUTED.md) | PyTorch 分布式训练配置说明，包括分组管理、Rank 分配、环境变量设置 |
| [doc/WORKER_LAUNCHER.md](doc/WORKER_LAUNCHER.md) | Worker 启动器使用说明，包括架构设计、命令行参数、多机部署示例 |
| [doc/PROJECT_STRUCTURE.md](doc/PROJECT_STRUCTURE.md) | 项目结构说明，包含所有目录和文件的详细说明 |
| [doc/README_PROCGUARD.md](doc/README_PROCGUARD.md) | ProcGuard 核心功能文档，包括模块说明和配置选项 |
| [doc/MIGRATION_GUIDE.md](doc/MIGRATION_GUIDE.md) | 从 torchft 项目的迁移指南 |

## 常用命令

```bash
# 启动管理服务（前台）
./start_manager.sh

# 启动管理服务（后台）
./start_manager.sh --daemon

# 启动 Worker
./start_worker.sh --worker-id worker1 --manager-url http://localhost:5000 --command "python train.py"

# 停止管理服务
pkill -f "procguard.main"
```

## 安装

```bash
cd procguard_project
pip install -e .
pip install -r requirements.txt
```

## 许可证

MIT License
