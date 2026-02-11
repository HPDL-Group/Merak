"""
HA failover example with two groups.

This example demonstrates:
1. Creating two worker groups
2. Adding workers to groups (4 in active, remaining in standby)
3. Starting workers in the active group
4. Associating the two groups for HA failover

Usage:
    # Terminal 1: Start ProcGuard server
    cd procguard
    python -m web.app

    # Terminal 2: Run this example
    python examples/ha_failover_example.py

    # Terminal 3-6: Start workers on different nodes
    python examples/dist_monitor_example.py --worker-id gn35-0
    python examples/dist_monitor_example.py --worker-id gn35-1
    python examples/dist_monitor_example.py --worker-id gn36-0
    python examples/dist_monitor_example.py --worker-id gn36-1

    # Terminal 7-10: Start standby workers (will be used for failover)
    python examples/dist_monitor_example.py --worker-id gn37-0
    python examples/dist_monitor_example.py --worker-id gn37-1
    python examples/dist_monitor_example.py --worker-id gn38-0
    python examples/dist_monitor_example.py --worker-id gn38-1
"""

import sys
import time
import requests
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Please set manager url !!")
BASE_URL = "http://10.107.16.90:5001"


def wait_for_server(timeout: int = 30) -> bool:
    """Wait for ProcGuard server to be available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{BASE_URL}/api/health", timeout=2)
            if response.status_code == 200:
                print("ProcGuard server is available")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("ProcGuard server not available")
    return False


def get_existing_workers() -> dict:
    """Get existing groups and their workers from the server."""
    try:
        response = requests.get(f"{BASE_URL}/api/groups", timeout=5)
        if response.status_code == 200:
            data = response.json()
            groups = data.get("groups", [])
            return {g.get("group_id"): g.get("workers", []) for g in groups}
    except requests.exceptions.RequestException:
        pass
    return {}


def cleanup_existing():
    """Clean up existing groups and associations."""
    print("\n=== Cleaning up existing groups and associations ===")

    try:
        response = requests.delete(f"{BASE_URL}/api/ha/associations", timeout=5)
        print(f"Deleted all associations: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to delete associations: {e}")

    try:
        response = requests.delete(f"{BASE_URL}/api/groups", timeout=5)
        print(f"Deleted all groups: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to delete groups: {e}")


def get_master_addr(worker_ids: list) -> str:
    """Get master address from worker IDs.

    Master addr is the hostname with the smallest number.
    E.g., ['gn34-0', 'gn39-0', 'gn35-0', 'gn37-0'] -> 'gn34'
    """
    hostnames = set()
    for worker_id in worker_ids:
        match = re.match(r"([a-z]+)(\d+)-\d+", worker_id)
        if match:
            prefix = match.group(1)
            num = int(match.group(2))
            hostnames.add((prefix, num))

    if not hostnames:
        return "localhost"

    hostnames_sorted = sorted(hostnames, key=lambda x: x[1])
    prefix, num = hostnames_sorted[0]
    return f"{prefix}{num}"


def configure_group_config(group_id: str, master_addr: str, world_size: int):
    """Configure PyTorch distributed training settings for a group."""
    print(f"\n=== Configuring PyTorch for {group_id} ===")
    print(f"  master_addr: {master_addr}")
    print(f"  world_size: {world_size}")

    response = requests.put(
        f"{BASE_URL}/api/groups/{group_id}/config",
        json={
            "master_addr": master_addr,
            "master_port": 29500,
            "world_size": world_size,
            "backend": "nccl",
        },
        timeout=5,
    )
    print(f"  Configure response: {response.status_code}")


def create_groups():
    """Create active and standby groups based on existing workers."""
    print("\n=== Creating groups ===")

    existing_workers = get_existing_workers()

    has_active = (
        "active_training" in existing_workers and existing_workers["active_training"]
    )
    has_standby = (
        "standby_backup" in existing_workers and existing_workers["standby_backup"]
    )

    if has_active and has_standby:
        print(
            f"Active group already exists with workers: {existing_workers['active_training']}"
        )
        print(
            f"Standby group already exists with workers: {existing_workers['standby_backup']}"
        )
        return existing_workers["active_training"], existing_workers["standby_backup"]

    cleanup_existing()
    existing_workers = get_existing_workers()

    default_workers = existing_workers.get("default", [])

    if not default_workers:
        print("No workers found in default group. Please start workers first.")
        print("Then run this script again to organize workers into groups.")
        return [], []

    active_workers = (
        default_workers[:4] if len(default_workers) >= 4 else default_workers
    )
    standby_workers = default_workers[4:] if len(default_workers) > 4 else []

    response = requests.post(
        f"{BASE_URL}/api/groups",
        json={
            "group_id": "active_training",
            "name": "Active Training Group",
        },
        timeout=5,
    )
    print(f"Create active_training group: {response.status_code}")

    for worker_id in active_workers:
        requests.post(
            f"{BASE_URL}/api/groups/active_training/workers/{worker_id}",
            timeout=5,
        )
    print(f"Added to active_training: {active_workers}")

    if standby_workers:
        response = requests.post(
            f"{BASE_URL}/api/groups",
            json={
                "group_id": "standby_backup",
                "name": "Standby Backup Group",
            },
            timeout=5,
        )
        print(f"Create standby_backup group: {response.status_code}")

        for worker_id in standby_workers:
            requests.post(
                f"{BASE_URL}/api/groups/standby_backup/workers/{worker_id}",
                timeout=5,
            )
        print(f"Added to standby_backup: {standby_workers}")

    return active_workers, standby_workers


def wait_for_workers_running(worker_ids: list, timeout: float = 120.0) -> bool:
    """Wait for all workers to be in running status."""
    print(f"\n=== Waiting for workers to be running ===")
    start = time.time()
    interval = 3.0

    while time.time() - start < timeout:
        all_running = True
        running_count = 0

        try:
            response = requests.get(f"{BASE_URL}/api/workers", timeout=5)
            if response.status_code == 200:
                all_workers_data = response.json()
                for worker_id in worker_ids:
                    worker_data = all_workers_data.get(worker_id, {})
                    status = worker_data.get("status", "unknown")
                    if status == "running":
                        running_count += 1
                    else:
                        all_running = False
            else:
                all_running = False
        except requests.exceptions.RequestException:
            all_running = False

        elapsed = int(time.time() - start)
        print(f"  Running: {running_count}/{len(worker_ids)} (elapsed: {elapsed}s)")

        if all_running:
            print(f"  All workers are running!")
            return True

        time.sleep(interval)

    print(f"  Timeout! Only {running_count}/{len(worker_ids)} workers are running")
    return False


def create_ha_association() -> str:
    """Create HA association between active and standby groups.

    Returns:
        str: Association ID (auto-generated)
    """
    print("\n=== Creating HA association ===")

    master_addr = "localhost"
    try:
        response = requests.get(
            f"{BASE_URL}/api/groups/active_training/config", timeout=5
        )
        if response.status_code == 200:
            config = response.json().get("config", {})
            master_addr = config.get("master_addr", "localhost")
            print(f"Using master_addr from active_training group: {master_addr}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to get active_training config: {e}")
        print(f"Using default master_addr: {master_addr}")

    response = requests.post(
        f"{BASE_URL}/api/ha/associations",
        json={
            "active_group_id": "active_training",
            "standby_group_id": "standby_backup",
            "world_size": 4,
            "failover_threshold": 1,
            "auto_failover": True,
            "master_addr": master_addr,
            "master_port": 29500,
            "backend": "nccl",
        },
        timeout=5,
    )
    print(f"Create HA association: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        association_id = data.get(
            "association_id", "assoc_active_training_standby_backup"
        )
        print(f"HA association created: {association_id}")
        return association_id
    else:
        print(f"Failed to create HA association: {response.text}")
        return "assoc_active_training_standby_backup"


def check_status(association_id: str = None):
    """Check the status of groups and association."""
    print("\n=== Checking status ===")

    try:
        response = requests.get(f"{BASE_URL}/api/groups", timeout=5)
        if response.status_code == 200:
            data = response.json()
            groups = data.get("groups", [])
            print(f"Groups: {[g.get('group_id') for g in groups]}")
            for group in groups:
                workers = group.get("workers", [])
                if workers:
                    print(f"  {group.get('group_id')}: {workers}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to get groups: {e}")

    try:
        if association_id:
            response = requests.get(
                f"{BASE_URL}/api/ha/associations/{association_id}/status", timeout=5
            )
        else:
            response = requests.get(f"{BASE_URL}/api/ha/associations", timeout=5)
        if response.status_code == 200:
            status = response.json()
            if "status" in status:
                s = status["status"]
                print(f"HA Status:")
                print(f"  State: {s.get('state')}")
                print(f"  Active Workers: {s.get('active_workers')}")
                print(f"  Standby Available: {s.get('standby_available')}")
                print(f"  Health: {s.get('health_status')}")
        elif response.status_code != 404:
            print(f"No HA association or error")
    except requests.exceptions.RequestException as e:
        print(f"Failed to get HA status: {e}")

    print("\n=== Worker Failover Status ===")
    try:
        response = requests.get(f"{BASE_URL}/api/workers/failover-workers", timeout=5)
        if response.status_code == 200:
            data = response.json()
            failover_workers = data.get("workers", [])
            print(f"Total failover workers: {data.get('count', 0)}")
            for worker in failover_workers:
                info = worker.get("failover_info", {})
                print(f"  {worker.get('worker_id')}:")
                print(f"    - Replaced: {info.get('replaced_worker_id', 'N/A')}")
                print(f"    - Association: {info.get('association_id', 'N/A')}")
                print(f"    - Failover Time: {info.get('failover_time', 'N/A')}")
                print(f"    - Status: {worker.get('status')}")
                print(f"    - PID: {worker.get('pid')}")
        else:
            print("  No failover workers found")
    except requests.exceptions.RequestException as e:
        print(f"Failed to get failover workers: {e}")

    try:
        response = requests.get(f"{BASE_URL}/api/workers", timeout=5)
        if response.status_code == 200:
            workers = response.json()
            print(f"\n=== All Workers Failover Status ===")
            failover_count = 0
            for worker_id, worker_info in workers.items():
                is_failover = worker_info.get("is_failover_worker", False)
                if is_failover:
                    failover_count += 1
                    info = worker_info.get("failover_info", {})
                    print(f"  {worker_id}: FAILOVER WORKER")
                    print(f"    - Replaced: {info.get('replaced_worker_id', 'N/A')}")
                    print(f"    - Time: {info.get('failover_time', 'N/A')}")
                else:
                    print(f"  {worker_id}: Normal worker")
            print(f"\nTotal: {len(workers)} workers, {failover_count} failover workers")
    except requests.exceptions.RequestException as e:
        print(f"Failed to get workers: {e}")


def stop_non_zero_rank_worker(worker_ids: list):
    """Stop one worker that has rank != 0 (master)."""
    if len(worker_ids) <= 1:
        print("\nOnly one worker, cannot stop master")
        return False

    non_zero_workers = []
    for wid in worker_ids:
        try:
            response = requests.get(
                f"{BASE_URL}/api/workers/{wid}/rank",
                timeout=5,
            )
            if response.status_code == 200:
                rank = response.json().get("rank", -1)
                if rank != 0:
                    non_zero_workers.append((wid, rank))
        except requests.exceptions.RequestException:
            continue

    if not non_zero_workers:
        print("\nAll workers have rank=0, cannot stop master")
        return False

    worker_to_stop, rank = non_zero_workers[0]
    print(f"\n=== Stopping worker {worker_to_stop} (rank={rank}) ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/workers/{worker_to_stop}/stop",
            json={},
            timeout=5,
        )
        print(f"  Stop {worker_to_stop}: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Failed to stop {worker_to_stop}: {e}")
        return False


def main():
    print("=" * 60)
    print("HA Failover Example - Two Group Setup")
    print("=" * 60)

    if not wait_for_server():
        return

    active_workers, standby_workers = create_groups()

    if not active_workers:
        print("\nPlease start workers first, then run this script again.")
        return

    master_addr = get_master_addr(active_workers)

    # Configure PyTorch distributed training BEFORE starting workers
    configure_group_config("active_training", master_addr, 4)

    print(f"\n=== Master Address Analysis ===")
    print(f"Active workers: {active_workers}")
    print(f"Master address (smallest hostname): {master_addr}")

    print("\n" + "=" * 60)
    print("Workers organized:")
    print(f"  Active group (running): {active_workers}")
    print(f"  Standby group (stopped): {standby_workers}")
    print("=" * 60)

    print("\n=== Starting workers in active group ===")
    for worker_id in active_workers:
        try:
            response = requests.post(
                f"{BASE_URL}/api/workers/{worker_id}/start",
                timeout=5,
            )
            print(f"  Start {worker_id}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  Failed to start {worker_id}: {e}")
        time.sleep(0.3)

    print("\n=== Waiting for active workers to be running ===")
    if not wait_for_workers_running(active_workers):
        print("\nWarning: Not all active workers are running. Continuing anyway...")

    check_status()

    print("\n=== Creating HA association ===")
    association_id = create_ha_association()

    print("\n=== Starting HA monitoring ===")

    try:
        response = requests.post(
            f"{BASE_URL}/api/ha/associations/{association_id}/monitor/start",
            timeout=5,
        )
        print(f"Start HA monitoring: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to start HA monitoring: {e}")

    print("\n" + "=" * 60)
    print("HA association is active!")
    print("If a worker in active group fails, system will:")
    print("1. Detect failure")
    print("2. Start standby worker")
    print("3. Move to active group")
    print("4. Update PyTorch config")
    print("=" * 60)

    def trigger_failover_after_delay():
        """Trigger failover test after 60 seconds."""
        import time

        print("\n[Thread] Waiting 60 seconds before triggering failover test...")
        time.sleep(60)
        print("\n[Thread] Triggering failover test now!")
        stop_non_zero_rank_worker(active_workers)

    import threading

    test_thread = threading.Thread(target=trigger_failover_after_delay, daemon=True)
    test_thread.start()

    print("\nMonitoring... (Ctrl+C to stop)")
    print(
        "Note: In 60 seconds, one worker (rank != 0) will be stopped to test failover"
    )
    try:
        while True:
            time.sleep(10)
            check_status(association_id)
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
