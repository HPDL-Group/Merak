# 示例程序

本目录包含 ProcGuard 的各种示例程序，演示如何使用进程监控、分布式训练等功能。

## 示例列表

### 训练示例

| 文件 | 说明 | 使用场景 |
|------|------|----------|
| [train_ddp.py](train_ddp.py) | PyTorch 分布式训练示例 | 演示 DDP 训练环境变量读取 |
| [test_launch.py](test_launch.py) | 分布式启动测试 | 测试多节点分布式训练 |

### 测试 Worker

| 文件 | 说明 | 使用场景 |
|------|------|----------|
| [fake_worker.py](fake_worker.py) | 模拟 Worker | 测试监控和故障恢复功能 |
| [test_worker.py](简单测试 Worker | 快速验证 Worker 注册 | |
| [test_worker.py](test_worker.py) | 简单测试 Worker | 快速验证 Worker 启动和注册 |

## 快速开始

### 1. 启动管理服务

```bash
./start_manager.sh --config configs/procguard_config.yaml
```

### 2. 启动示例 Worker

```bash
# 使用 fake_worker 模拟实际工作
./start_worker.sh \
    --worker-id test1 \
    --manager-url http://localhost:5000 \
    --command "python examples/fake_worker.py"

# 或使用分布式训练示例
./start_worker.sh \
    --worker-id trainer1 \
    --manager-url http://localhost:5000 \
    --command "python examples/train_ddp.py --epochs 100"
```

### 3. 使用 Web 界面配置分布式训练

1. 访问 `http://localhost:5000`
2. 创建新分组
3. 添加 Worker 到分组
4. 点击分组上的 ⚙️ 配置按钮
5. 设置分布式参数：
   - Master 地址（如 `gn6`）
   - World Size（如 `4`）
   - Master 端口（默认 `29500`）
   - 后端（如 `nccl`）

## train_ddp.py 详解

`train_ddp.py` 是一个完整的 PyTorch 分布式训练示例，支持从环境变量自动读取配置：

```python
import torch
from examples.train_ddp import SimpleModel, setup_distributed, train, cleanup_distributed
import argparse

if __name__ == "__main__":
    # 自动从环境变量读取 RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    rank, world_size, device = setup_distributed()
    
    model = SimpleModel(input_size=100, hidden_size=200, output_size=10).to(device)
    
    args = argparse.Namespace(
        epochs=10,
        batch_size=32,
        batches_per_epoch=100,
        input_size=100,
        hidden_size=200,
        output_size=10,
        lr=0.001
    )
    
    try:
        train(rank, device, args)
    finally:
        cleanup_distributed()
```

也可以直接使用 `main()` 函数：

```python
from examples.train_ddp import main

if __name__ == "__main__":
    import sys
    sys.argv = ["train_ddp.py", "--epochs", "10", "--batch-size", "32"]
    main()
```

## 环境变量

训练脚本会自动从环境变量读取分布式配置：

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `RANK` | 当前进程排名 | `0, 1, 2, 3` |
| `WORLD_SIZE` | 总进程数 | `4` |
| `LOCAL_RANK` | 本节点内排名 | `0, 1` |
| `MASTER_ADDR` | 主节点地址 | `gn6` |
| `MASTER_PORT` | 通信端口 | `29500` |

## 分布式启动测试

使用 `test_launch.py` 测试多节点分布式训练：

```bash
# 在所有节点上运行
python examples/test_launch.py \
    --nnodes 2 \
    --node_rank 0 \
    --nproc_per_node 2 \
    --master_addr gn6 \
    --master_port 29500 \
    --command "python examples/train_ddp.py --epochs 10"
```

## 故障排除

### 问题：分布式训练无法启动

**检查项**：
1. 确认所有节点网络互通
2. 确认 Master 地址可解析
3. 确认端口未被防火墙阻止

### 问题：Rank 分配不正确

**检查项**：
1. 确认 Web 界面中的分组配置已保存
2. 确认 World Size 不大于 Worker 数量
3. 查看 Worker 日志确认环境变量正确

## 相关文档

- [doc/PYTORCH_DISTRIBUTED.md](../doc/PYTORCH_DISTRIBUTED.md) - PyTorch 分布式训练详细说明
- [doc/WORKER_LAUNCHER.md](../doc/WORKER_LAUNCHER.md) - Worker 启动器使用说明
