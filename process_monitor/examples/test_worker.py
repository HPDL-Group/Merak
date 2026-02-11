import subprocess
import time
import signal
import sys


def test_worker(worker_id):
    cmd = [
        "python",
        "fake_worker.py",
        "--worker-id",
        worker_id,
        "--epochs",
        "2",
        "--sleep-interval",
        "0.05",
    ]
    print(f"Starting worker {worker_id}...")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        time.sleep(3)
        print(f"Worker {worker_id} is running (PID: {process.pid})")

        time.sleep(2)
        print(f"Stopping worker {worker_id}...")
        process.terminate()

        process.wait(timeout=5)
        print(f"Worker {worker_id} stopped successfully")

    except subprocess.TimeoutExpired:
        print(f"Worker {worker_id} did not stop gracefully, killing...")
        process.kill()
        process.wait()
    except Exception as e:
        print(f"Error with worker {worker_id}: {e}")
        process.kill()
        process.wait()


if __name__ == "__main__":
    print("Testing fake_worker.py...")
    print("=" * 50)

    test_worker("test_worker_0")

    print("=" * 50)
    print("Test completed!")
