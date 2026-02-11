import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import argparse
import signal
import sys


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


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        print(f"[Rank {rank}] World Size: {world_size}, Local Rank: {local_rank}, Master Addr: {os.environ['MASTER_ADDR']}")
        if torch.cuda.is_available():
            actual_gpu_count = torch.cuda.device_count()
            device_index = local_rank % actual_gpu_count if actual_gpu_count > 0 else 0
            device = torch.device("cuda", device_index)
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
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


def train(rank, device, args):
    model = SimpleModel(
        input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size
    ).to(device)

    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Rank {rank}] Starting training on device: {device}")
    print(f"[Rank {rank}] Model: {model}")

    global_step = 0

    try:
        for epoch in range(args.epochs):
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
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1

                if batch_idx % 10 == 0:
                    if rank == 0 or not dist.is_initialized():
                        print(
                            f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                        )

                time.sleep(0.01)

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            if rank == 0 or not dist.is_initialized():
                print(f"[Rank {rank}] Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

    except KeyboardInterrupt:
        print(f"[Rank {rank}] Training interrupted by user")
    except Exception as e:
        print(f"[Rank {rank}] Error during training: {e}")
        raise

    print(f"[Rank {rank}] Training completed")


def main():
    parser = argparse.ArgumentParser(description="Simple DDP Training Script")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--batches-per-epoch", type=int, default=100, help="Batches per epoch")
    parser.add_argument("--input-size", type=int, default=100, help="Input feature size")
    parser.add_argument("--hidden-size", type=int, default=200, help="Hidden layer size")
    parser.add_argument("--output-size", type=int, default=10, help="Output size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    rank, world_size, device = setup_distributed()

    print(f"[Rank {rank}] Initialized with world_size={world_size}, device={device}")

    try:
        train(rank, device, args)
    finally:
        cleanup_distributed()
        print(f"[Rank {rank}] Cleanup completed")


if __name__ == "__main__":
    main()
