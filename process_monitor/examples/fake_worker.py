import os
import time
import random
import argparse
import signal
import sys


class FakeWorker:
    def __init__(self, worker_id, epochs=100, batch_size=32, sleep_interval=0.1):
        self.worker_id = worker_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self.running = True
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.loss = 1.0

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"[Worker {self.worker_id}] Received signal {signum}, shutting down...")
        self.running = False

    def _generate_fake_data(self):
        x = [random.random() for _ in range(100)]
        y = random.randint(0, 9)
        return x, y

    def _train_step(self):
        x, y = self._generate_fake_data()

        loss = self.loss * (0.9 + random.random() * 0.2)
        self.loss = max(0.1, loss * 0.99)

        return loss

    def run(self):
        print(f"[Worker {self.worker_id}] Starting training...", flush=True)
        print(
            f"[Worker {self.worker_id}] Configuration: epochs={self.epochs}, batch_size={self.batch_size}",
            flush=True,
        )
        print(f"[Worker {self.worker_id}] PID: {os.getpid()}", flush=True)

        try:
            for epoch in range(self.epochs):
                if not self.running:
                    break

                self.current_epoch = epoch
                epoch_loss = 0.0

                for batch in range(100):
                    if not self.running:
                        break

                    self.current_batch = batch
                    loss = self._train_step()
                    epoch_loss += loss
                    self.total_batches += 1

                    if batch % 10 == 0:
                        print(
                            f"[Worker {self.worker_id}] Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}",
                            flush=True,
                        )

                    time.sleep(self.sleep_interval)

                avg_loss = epoch_loss / 100
                print(
                    f"[Worker {self.worker_id}] Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}",
                    flush=True,
                )

            print(f"[Worker {self.worker_id}] Training completed successfully", flush=True)
            print(
                f"[Worker {self.worker_id}] Total batches processed: {self.total_batches}",
                flush=True,
            )
            print(f"[Worker {self.worker_id}] Final loss: {self.loss:.4f}", flush=True)

        except KeyboardInterrupt:
            print(f"[Worker {self.worker_id}] Training interrupted by user", flush=True)
        except Exception as e:
            print(f"[Worker {self.worker_id}] Error during training: {e}", flush=True)
            import traceback

            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description="Simple Fake Training Worker")
    parser.add_argument("--worker-id", type=str, default="worker", help="Worker ID")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--sleep-interval", type=float, default=0.1, help="Sleep interval between batches"
    )

    args = parser.parse_args()

    worker = FakeWorker(
        worker_id=args.worker_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sleep_interval=args.sleep_interval,
    )

    worker.run()


if __name__ == "__main__":
    main()
