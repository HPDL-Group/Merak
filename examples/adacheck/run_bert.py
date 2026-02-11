import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path

from procguard.dist_monitor import (
    DistMonitor,
    get_monitor,
    get_local_rank_info,
    get_local_master_info,
    WorkerState,
)

from procguard.utils import get_logger
from Merak.CheckpointSync.ddp import DistributedDataParallel as DDP
from Merak.CheckpointSync.recover import Recover

from transformers import (
    BertLMHeadModel,
    BertConfig,
)
from Merak.utils.datasets import DynamicGenDataset


class ClusterMonitor:
    def __init__(self, monitor: DistMonitor):
        self._monitor = monitor

    def log_cluster_status(self, rank: int):
        cluster = self._monitor.get_cluster()
        active_workers = self._monitor.list_active_workers()

        print(f"[Rank {rank}] 集群状态: Master:{cluster.master_addr}:{cluster.master_port}, "
              f"WorldSize:{cluster.world_size}, 活跃Worker:{len(active_workers)}/{cluster.world_size}")

    def log_all_workers(self, rank: int):
        workers = self._monitor.list_all_workers()
        print(f"[Rank {rank}] Worker状态:")
        for worker_id, worker in sorted(workers.items(), key=lambda x: x[1].rank or 0):
            status = "✓" if worker.state == WorkerState.运行中 else "✗"
            print(f"[Rank {rank}]   {status} Rank{worker.rank}: {worker.state.value}")


def setup_distributed(monitor):
    rank_info = get_local_rank_info()
    master_info = get_local_master_info()

    rank = rank_info["rank"]
    world_size = rank_info["world_size"]
    local_rank = rank_info["local_rank"]

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", master_info["master_addr"])
        os.environ.setdefault("MASTER_PORT", str(master_info["master_port"]))

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device = torch.device("cuda", local_rank % device_count)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        return rank, world_size, device
    return 0, 1, torch.device("cpu")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _wait_for_workers(monitor: DistMonitor, max_wait: float = 60.0):
    start_time = time.time()
    expected_size = monitor.get_world_size()
    my_worker_id = monitor._worker_id

    while time.time() - start_time < max_wait:
        active_count = len(monitor.list_active_workers())
        if active_count >= expected_size:
            print(f"[Worker {my_worker_id}] 所有Worker已就绪 ({active_count}/{expected_size})")
            return
        time.sleep(0.1)

    print(f"[Worker {my_worker_id}] 等待Worker超时，继续初始化")


def train(world_size, rank, device, args, monitor: DistMonitor):
    config = BertConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=args.vocab_size,
    )
    model = BertLMHeadModel(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = get_logger('bert_train')
    recover = Recover(monitor, model, optimizer, logger)
    model = DDP(recover, model)

    dataset = DynamicGenDataset(config, mode="text_only", dataset_size=1e6)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    train_dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=train_sampler,
        )

    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(train_dataloader):
            if batch_idx >= args.batches_per_epoch:
                break

            input_ids = data["input_ids"].to(device)
            labels = data["labels"].to(device)

            recover.start_recover()
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )

            loss = outputs.loss if outputs.loss is not None else None
            if loss is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs.logits.view(-1, config.vocab_size), labels.view(-1))

            loss.backward()

            if not monitor.should_stop(force_refresh=True):
                optimizer.step()
            else:
                failed = monitor.get_failed_ranks(force_refresh=True)
                src = monitor.get_recover_src_rank(force_refresh=True)
                print(f'[Rank {rank}] 检测到失败: {failed}, 恢复源: {src}')

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if rank == 0:
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"[Rank {rank}] Epoch {epoch} 完成, Avg Loss: {avg_loss:.4f}")


    total_time = time.time() - start_time
    print(f"[Rank {rank}] 训练完成, 总耗时: {total_time:.2f}s, 总步数: {global_step}")


def main():
    parser = argparse.ArgumentParser(description="BERT分布式训练")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches-per-epoch", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    monitor = None
    try:
        print("初始化分布式训练...")
        monitor = get_monitor()
        _wait_for_workers(monitor)

        rank, world_size, device = setup_distributed(monitor)
        print(f"[Rank {rank}] 初始化完成, WorldSize:{world_size}, Device:{device}")

        if rank == 0:
            cluster_monitor = ClusterMonitor(monitor)
            cluster_monitor.log_cluster_status(rank)

        train(world_size, rank, device, args, monitor)

    finally:
        if monitor:
            monitor.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
