# PyTorch 分布式训练配置

本项目支持 PyTorch 分布式训练（DDP）的配置和管理。通过 Web 界面可以轻松配置分布式训练环境，并自动为每个 Worker 设置正确的环境变量。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                     ProcGuard 管理服务端                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Web 界面                                            │   │
│  │  - 分组管理                                          │   │
│  │  - PyTorch 配置                                      │   │
│  │  - Worker 状态监控                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           │ 心跳 + 环境变量                   │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Worker 启动器                                       │   │
│  │  - 接收分组配置                                      │   │
│  │  - 设置环境变量                                     │   │
│  │  - 启动训练进程                                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 配置项说明

### 分组配置

在 Web 界面中创建分组后，可以配置以下 PyTorch 分布式参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| **Master 地址** | 分布式训练的主节点地址 | - |
| **World Size** | 总共的进程数 | 分组中 Worker 数量 |
| **Master 端口** | NCCL 通信端口 | 29500 |
| **后端** | 分布式通信后端 | nccl |

### 环境变量

ProcGuard 会自动为每个 Worker 设置以下环境变量：

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `MASTER_ADDR` | 主节点地址 | `gn6` |
| `MASTER_PORT` | 通信端口 | `29500` |
| `WORLD_SIZE` | 总进程数 | `4` |
| `RANK` | 当前进程排名 | `0, 1, 2, 3` |
| `LOCAL_RANK` | 本节点内排名 | `0, 1` |
| `NODE_RANK` | 节点排名 | `0, 1, 2` |

## 使用流程

### 1. 启动管理服务端

```bash
python -m procguard --config configs/procguard_config.yaml
```

### 2. 启动 Worker

确保 Worker 启动器已注册到管理服务端。每个 Worker 启动器会定期发送心跳。

### 3. 创建分组并添加 Worker

在 Web 界面中：
1. 点击"新建分组"创建分组
2. 在 Worker 列表中勾选要添加到分组的 Worker
3. 点击"移动到"按钮，选择目标分组

### 4. 配置分布式参数

1. 点击分组卡片上的 ⚙️ 配置按钮
2. 填写 PyTorch 分布式配置：
   - **Master 地址**：输入主节点的名称（如 `gn6`）
   - **World Size**：输入总进程数
   - **Master 端口**：保持默认 29500 或自定义
   - **后端**：通常使用 `nccl`（GPU训练）或 `gloo`（CPU训练）
3. 点击"保存配置"

### 5. 验证配置

查看 Worker 日志，确认环境变量已正确设置：

```
[Rank 0] World Size: 4, Local Rank: 0, Master Addr: gn6
[Rank 1] World Size: 4, Local Rank: 0, Master Addr: gn6
[Rank 2] World Size: 4, Local Rank: 0, Master Addr: gn6
[Rank 3] World Size: 4, Local Rank: 0, Master Addr: gn6
```

## Rank 分配规则

ProcGuard 按照以下规则自动分配 RANK：

1. **Master 节点优先**：配置中指定的 Master 节点始终排在第一位
2. **节点排序**：其他节点按名称字母顺序排列
3. **Worker 排序**：每个节点内的 Worker 按 ID 数字排序
4. **Local Rank**：每个节点内的 Worker 从 0 开始编号

### 示例

假设有 4 个 Worker 在 4 个不同节点上：
- `gn6-0` (Master)
- `gn39-0`
- `gn40-0`
- `gn7-0`

配置 Master 地址为 `gn6`，World Size 为 4。

**分配结果**：
| Worker | RANK | LOCAL_RANK | NODE_RANK |
|--------|------|------------|-----------|
| gn6-0  | 0    | 0          | 0         |
| gn39-0 | 1    | 0          | 1         |
| gn40-0 | 2    | 0          | 2         |
| gn7-0  | 3    | 0          | 3         |

### 多 Worker 同节点示例

假设 4 个 Worker 在 2 个节点上：
- `gn6-0`, `gn6-1` (Master)
- `gn7-0`, `gn7-1`

**分配结果**：
| Worker | RANK | LOCAL_RANK | NODE_RANK |
|--------|------|------------|-----------|
| gn6-0  | 0    | 0          | 0         |
| gn6-1  | 1    | 1          | 0         |
| gn7-0  | 2    | 0          | 1         |
| gn7-1  | 3    | 1          | 1         |

## 训练脚本要求

你的训练脚本需要从环境变量读取分布式配置：

```python
import os
import torch
import torch.distributed as dist

def main():
    # 从环境变量读取配置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    
    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # 初始化分布式训练
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    
    # ... 继续训练代码 ...
```

或者使用 `train_ddp.py` 示例脚本：

```python
from examples.train_ddp import train_ddp

if __name__ == "__main__":
    train_ddp(
        model=YourModel(),
        input_size=100,
        hidden_size=200,
        output_size=10,
        batch_size=32,
        epochs=10,
        lr=0.001
    )
```

## 故障排除

### 问题：Rank 分配不正确

**检查项**：
1. 确认 Master 地址填写正确（节点名，不含 `-编号`）
2. 确认 World Size 不大于分组中的 Worker 数量
3. 查看服务端日志确认分组配置已保存

### 问题：Worker 无法连接

**检查项**：
1. 确认所有节点网络互通
2. 确认 Master 端口未被防火墙阻止
3. 确认 `MASTER_ADDR` 可以被其他节点解析

### 问题：多 GPU 训练

**配置建议**：
- 每个节点启动多个 Worker（每个 GPU 一个）
- 设置 `LOCAL_RANK` 为 GPU 编号
- 使用 `nccl` 后端

## 相关文件

- `examples/train_ddp.py` - PyTorch DDP 训练示例
- `examples/test_launch.py` - 分布式启动测试脚本
- `doc/WORKER_LAUNCHER.md` - Worker 启动器详细说明
