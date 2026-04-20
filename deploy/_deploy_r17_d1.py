"""Deploy R17 (Capital Curve Engineering) to Server D1 (port 35258)."""
import paramiko
import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def ssh(client, cmd, timeout=30):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    print("=== Deploying R17 to D1 (35258) ===\n")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    # Kill old experiments if any
    procs = ssh(client, "ps aux | grep 'run_round\\|run_exp' | grep python | grep -v grep")
    if procs:
        print(f"Found running experiments:\n  {procs}")
        ssh(client, "pkill -f 'run_round17' 2>/dev/null")
        print("  Killed existing R17 if any")
    else:
        print("No experiments running")

    # Ensure directories
    ssh(client, f"mkdir -p {REMOTE_DIR}/scripts/experiments")
    ssh(client, f"mkdir -p {REMOTE_DIR}/logs")
    ssh(client, f"mkdir -p {REMOTE_DIR}/results/round17_results")
    ssh(client, f"mkdir -p {REMOTE_DIR}/backtest")

    # Upload files
    uploads = [
        ("experiments/run_round17.py",
         f"{REMOTE_DIR}/scripts/experiments/run_round17.py"),
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
    print("\nStarting R17...")
    start_cmd = (
        f"cd {REMOTE_DIR} && "
        f"nohup python3 -u scripts/experiments/run_round17.py "
        f"> logs/round17.log 2>&1 &"
    )
    client.exec_command(start_cmd)
    time.sleep(5)

    # Verify
    proc = ssh(client, "ps aux | grep run_round17 | grep python | grep -v grep")
    if proc:
        pid = proc.split()[1]
        print(f"  Running! PID: {pid}")
    else:
        print("  WARNING: Process not found! Checking log...")
        log = ssh(client, f"tail -20 {REMOTE_DIR}/logs/round17.log")
        print(log)

    sftp.close()
    client.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
