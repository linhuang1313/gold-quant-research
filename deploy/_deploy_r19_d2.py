"""Deploy R19 (Signal Quality & Regime Enhancement) to Server D2 (port 45630)."""
import paramiko
import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def ssh(client, cmd, timeout=30):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    print("=== Deploying R19 to D2 (45630) ===\n")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    # Check if choppy OC is still running
    procs = ssh(client, "ps aux | grep 'run_exp_choppy\\|run_round' | grep python | grep -v grep")
    if procs:
        print(f"WARNING: Experiment still running!\n{procs}")
        print("\nKilling old experiment to start R19...")
        ssh(client, "pkill -f 'run_exp_choppy' 2>/dev/null")
        ssh(client, "pkill -f 'run_round19' 2>/dev/null")
        time.sleep(2)
    else:
        print("No experiments running — ready to deploy")

    # Ensure directories
    ssh(client, f"mkdir -p {REMOTE_DIR}/scripts/experiments")
    ssh(client, f"mkdir -p {REMOTE_DIR}/logs")
    ssh(client, f"mkdir -p {REMOTE_DIR}/results/round19_results")
    ssh(client, f"mkdir -p {REMOTE_DIR}/backtest")

    # Upload files
    uploads = [
        ("experiments/run_round19.py",
         f"{REMOTE_DIR}/scripts/experiments/run_round19.py"),
        ("backtest/__init__.py", f"{REMOTE_DIR}/backtest/__init__.py"),
        ("backtest/engine.py", f"{REMOTE_DIR}/backtest/engine.py"),
        ("backtest/runner.py", f"{REMOTE_DIR}/backtest/runner.py"),
        ("backtest/stats.py", f"{REMOTE_DIR}/backtest/stats.py"),
        ("research_config.py", f"{REMOTE_DIR}/research_config.py"),
        ("indicators.py", f"{REMOTE_DIR}/indicators.py"),
    ]

    print("\nUploading files...")
    for local_rel, remote_path in uploads:
        local_path = os.path.join(LOCAL_ROOT, local_rel)
        if not os.path.exists(local_path):
            print(f"  [SKIP] {local_rel}")
            continue
        sftp.put(local_path, remote_path)
        print(f"  [OK] {local_rel} ({os.path.getsize(local_path):,} bytes)")

    # Start experiment
    print("\nStarting R19...")
    start_cmd = (
        f"cd {REMOTE_DIR} && "
        f"nohup python3 -u scripts/experiments/run_round19.py "
        f"> logs/round19.log 2>&1 &"
    )
    client.exec_command(start_cmd)
    time.sleep(5)

    # Verify
    proc = ssh(client, "ps aux | grep run_round19 | grep python | grep -v grep")
    if proc:
        pid = proc.split()[1]
        print(f"  Running! PID: {pid}")
    else:
        print("  WARNING: Process not found! Checking log...")
        log = ssh(client, f"tail -20 {REMOTE_DIR}/logs/round19.log")
        print(log)

    sftp.close()
    client.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
