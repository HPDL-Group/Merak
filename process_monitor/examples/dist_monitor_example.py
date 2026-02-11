import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.dist_monitor import (
    DistMonitor,
    get_monitor,
    get_local_rank_info,
    get_local_master_info,
    WorkerState,
)


class SimpleModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=200, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ClusterMonitor:
    def __init__(self, monitor: DistMonitor):
        self._monitor = monitor
        self._my_info = monitor.get_self_info()

    def log_cluster_status(self, rank: int):
        cluster = self._monitor.get_cluster()
        active_workers = self._monitor.list_active_workers()
        all_ranks = self._monitor.list_all_ranks()

        print(f"[Rank {rank}]" + "=" * 50)
        print(f"[Rank {rank}] 集群状态:")
        print(f"[Rank {rank}]   Master: {cluster.master_addr}:{cluster.master_port}")
        print(f"[Rank {rank}]   World Size: {cluster.world_size}")
        print(f"[Rank {rank}]   Backend: {cluster.backend}")
        print(f"[Rank {rank}]   运行中 Worker: {len(active_workers)}/{cluster.world_size}")
        print(f"[Rank {rank}]   已分配 Ranks: {sorted(all_ranks)}")
        print(f"[Rank {rank}]" + "=" * 50)

    def log_all_workers(self, rank: int):
        workers = self._monitor.list_all_workers()
        print(f"[Rank {rank}] Worker 状态:")
        for worker_id, worker in sorted(workers.items(), key=lambda x: x[1].rank or 0):
            status_icon = "✓" if worker.state == WorkerState.运行中 else "✗"
            rank_str = f"Rank {worker.rank}" if worker.rank is not None else "No rank"
            pid_str = f"PID:{worker.pid}" if worker.pid else "No PID"
            print(f"[Rank {rank}]   {status_icon} {rank_str} ({worker_id}): {worker.state.value} | {pid_str}")

    def check_cluster_health(self, rank: int) -> bool:
        cluster = self._monitor.get_cluster()
        active_workers = self._monitor.list_active_workers()
        all_ranks = self._monitor.list_all_ranks()

        if cluster.world_size > 0 and len(active_workers) < cluster.world_size:
            print(f"[Rank {rank}] 警告: 运行中 worker 数量 ({len(active_workers)}) 小于 World Size ({cluster.world_size})")
            return False

        expected_ranks = set(range(cluster.world_size)) if cluster.world_size > 0 else set()
        current_ranks = set(all_ranks)
        if current_ranks != expected_ranks and expected_ranks:
            missing = expected_ranks - current_ranks
            print(f"[Rank {rank}] 警告: 缺少 Ranks: {sorted(missing)}")
            return False

        return True

    def wait_for_all_workers(self, rank: int, timeout: float = 60.0) -> bool:
        world_size = self._monitor.get_world_size()
        print(f"[Rank {rank}] 等待所有 {world_size} 个 Worker 就绪 (超时: {timeout}s)...")

        success = self._monitor.wait_for_workers(world_size, timeout)
        if success:
            active_count = len(self._monitor.list_active_workers())
            print(f"[Rank {rank}] 所有 {active_count} 个 Worker 已就绪")
        else:
            print(f"[Rank {rank}] 等待 Worker 超时")

        return success

    def get_worker_rank_info(self, worker_id: str) -> dict:
        worker = self._monitor.get_worker_by_rank(0)
        rank = self._monitor.get_rank(worker_id)
        worker_meta = self._monitor.list_all_workers().get(worker_id)

        return {
            "worker_id": worker_id,
            "rank": rank,
            "state": worker_meta.state if worker_meta else None,
            "pid": worker_meta.pid if worker_meta else None
        }

    def log_my_info(self, rank: int):
        my_info = self._monitor.get_self_info()
        if my_info:
            print(f"[Rank {rank}] 本 Worker 信息:")
            print(f"[Rank {rank}]   Worker ID: {my_info.worker_id}")
            print(f"[Rank {rank}]   Rank: {my_info.rank}")
            print(f"[Rank {rank}]   Local Rank: {my_info.local_rank}")
            print(f"[Rank {rank}]   Node Rank: {my_info.node_rank}")
            print(f"[Rank {rank}]   状态: {my_info.state.value}")
            print(f"[Rank {rank}]   PID: {my_info.pid}")
        else:
            print(f"[Rank {rank}] 无法获取本 Worker 信息")


def on_config_changed(old_config: dict, new_config: dict, rank: int, world_size_ref: list):
    old_world = old_config.get("world_size", 0) if old_config else 0
    new_world = new_config.get("world_size", 0)

    print(f"[Rank {rank}] ⚠ 配置变更检测!")
    print(f"[Rank {rank}]   World Size: {old_world} -> {new_world}")
    print(f"[Rank {rank}]   Master: {new_config.get('master_addr')}:{new_config.get('master_port')}")

    world_size_ref[0] = new_world

    if old_world != new_world:
        print(f"[Rank {rank}] ⚠ World Size 已变更，请重启进程以应用新配置")
        print(f"[Rank {rank}]    如果使用相同配置，可以继续运行")
        print(f"[Rank {rank}]    如果需要不同配置，请停止并重新启动 worker")


def setup_distributed(monitor: DistMonitor, rank: int):
    rank_info = get_local_rank_info()
    master_info = get_local_master_info()

    world_size = rank_info["world_size"]
    local_rank = rank_info["local_rank"]

    world_size_ref = [world_size]

    def config_callback(old_config, new_config):
        on_config_changed(old_config, new_config, rank, world_size_ref)

    if monitor:
        monitor.register_config_change_callback(config_callback)
        env = monitor.get_env_for_worker(monitor._worker_id)
        if env.get("MASTER_ADDR"):
            master_info["master_addr"] = env["MASTER_ADDR"]
        if env.get("MASTER_PORT"):
            master_info["master_port"] = int(env["MASTER_PORT"])
        if env.get("WORLD_SIZE"):
            world_size_ref[0] = int(env["WORLD_SIZE"])
            world_size = world_size_ref[0]
        if env.get("TORCH_DISTRIBUTED_BACKEND"):
            master_info["backend"] = env["TORCH_DISTRIBUTED_BACKEND"]

    if world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_info["master_addr"]
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_info["master_port"])

        print(f"[Rank {rank}] World Size: {world_size}, Local Rank: {local_rank}, Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

        if torch.cuda.is_available():
            actual_gpu_count = torch.cuda.device_count()
            device_index = local_rank % actual_gpu_count if actual_gpu_count > 0 else 0
            device = torch.device("cuda", device_index)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print(f"============{device}=============")
        dist.init_process_group(
            backend=backend,
            device_id=device,
            rank=rank,
            world_size=world_size,
        )

        return rank, world_size, device
    else:
        return 0, 1, torch.device("cpu")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_fake_data(batch_size=32, input_size=100, num_batches=100):
    for i in range(num_batches):
        x = torch.randn(batch_size, input_size)
        y = torch.randint(0, 10, (batch_size,))
        yield x, y


def train(rank, device, args, monitor: DistMonitor):
    cluster_monitor = ClusterMonitor(monitor)

    model = SimpleModel(
        input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size
    ).to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Rank {rank}] 初始化训练, 设备: {device}")
    print(f"[Rank {rank}] 模型结构:\n{model}")

    cluster_monitor.log_my_info(rank)

    if rank == 0:
        cluster_monitor.log_cluster_status(rank)

    global_step = 0
    start_time = time.time()

    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(
                generate_fake_data(
                    batch_size=args.batch_size,
                    input_size=args.input_size,
                    num_batches=args.batches_per_epoch,
                )
            ):

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                time.sleep(1)
                loss = criterion(outputs, y)
                loss.backward()
                time.sleep(2)

                if monitor.should_stop(force_refresh=True):
                    failed_ranks = monitor.get_failed_ranks()
                    src_rank = monitor.get_recover_src_rank()
                    print(f"[Ranks {failed_ranks}] 检测这些Worker停止")
                    print(f"[Rank {src_rank}] 可以从这些rank中恢复")
                    print(f"[Rank {rank}] 检测到其他 Worker 停止，停止训练")
                    dist.barrier()
                else:
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if batch_idx % 20 == 0 and (rank == 0 or not dist.is_initialized()):
                    active_workers = monitor.list_active_workers()
                    active_count = len(active_workers)
                    world_size = monitor.get_world_size()
                    print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, 活跃Worker: {active_count}/{world_size}")

                time.sleep(0.01)

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start

            if rank == 0 or not dist.is_initialized():
                print(f"[Rank {rank}] Epoch {epoch} 完成, Avg Loss: {avg_loss:.4f}, 耗时: {epoch_time:.2f}s")

            if rank == 0:
                cluster_monitor.log_all_workers(rank)
                cluster_monitor.check_cluster_health(rank)

    except KeyboardInterrupt:
        print(f"[Rank {rank}] 用户中断训练")
    except Exception as e:
        print(f"[Rank {rank}] 训练错误: {e}")
        raise

    total_time = time.time() - start_time
    print(f"[Rank {rank}] 训练完成, 总耗时: {total_time:.2f}s, 总步数: {global_step}")

    return model


def main():
    parser = argparse.ArgumentParser(description="分布式训练监控示例")
    parser.add_argument("--epochs", type=int, default=99, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--batches-per-epoch", type=int, default=50, help="每轮迭代次数")
    parser.add_argument("--input-size", type=int, default=100, help="输入特征大小")
    parser.add_argument("--hidden-size", type=int, default=200, help="隐藏层大小")
    parser.add_argument("--output-size", type=int, default=10, help="输出大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--monitor-url", type=str, default=None, help="ProcGuard 管理端 URL")
    parser.add_argument("--worker-id", type=str, default=None, help="本 Worker ID")

    args = parser.parse_args()

    monitor = None
    try:
        print("初始化分布式训练监控...")
        monitor = get_monitor()

        health = monitor.health_check()
        print(f"监控接口健康状态: {'正常' if health['healthy'] else '异常'}")
        print(f"  管理端可达: {health['manager_reachable']}")
        print(f"  Worker 已注册: {health['worker_registered']}")
        print(f"  Worker 运行中: {health['worker_active']}")

        rank_info = get_local_rank_info()
        rank, world_size, device = setup_distributed(monitor, rank_info["rank"])

        print(f"[Rank {rank}] 初始化完成, World Size: {world_size}, 设备: {device}")

        model = train(rank, device, args, monitor)

        if rank == 0:
            cluster = monitor.get_cluster()
            print("\n" + "=" * 60)
            print("训练完成后的集群状态:")
            if cluster:
                print(f"  Master: {cluster.master_addr}:{cluster.master_port}")
                print(f"  World Size: {cluster.world_size}")
            else:
                print("  (管理端不可达)")
            print(f"  运行中 Worker: {len(monitor.list_active_workers(force_refresh=True))}")
            print("=" * 60)

    finally:
        if monitor:
            monitor.close()
            print("监控器已关闭")

        cleanup_distributed()
        print("分布式训练清理完成")


if __name__ == "__main__":
    main()
