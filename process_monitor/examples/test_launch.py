import subprocess
import time
import sys
import os


def test_worker_launch():
    print("Testing worker launch...")

    command = "python fake_worker.py --worker-id test_worker --epochs 1 --sleep-interval 0.1"

    try:
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        print(f"Worker launched with PID: {process.pid}")

        time.sleep(2)

        if process.poll() is None:
            print(f"Worker is still running (PID: {process.pid})")

            import psutil

            ps_process = psutil.Process(process.pid)
            print(f"Process status: {ps_process.status()}")
            print(f"Process is running: {ps_process.is_running()}")
            print(f"CPU percent: {ps_process.cpu_percent(interval=0.1)}")
            print(f"Memory percent: {ps_process.memory_percent()}")
        else:
            print(f"Worker has exited with code: {process.returncode}")
            stdout, stderr = process.communicate()
            if stdout:
                print(f"Stdout: {stdout.decode()}")
            if stderr:
                print(f"Stderr: {stderr.decode()}")

        process.terminate()
        process.wait(timeout=5)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_worker_launch()
