"""Deploy EXP-CHOPPY-OC to Server C and start it."""
import paramiko
import os
import sys

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

FILES_TO_UPLOAD = [
    ("experiments/run_exp_choppy_opportunity_cost.py",
     f"{REMOTE_DIR}/scripts/experiments/run_exp_choppy_opportunity_cost.py"),
    ("backtest/__init__.py", f"{REMOTE_DIR}/backtest/__init__.py"),
    ("backtest/engine.py", f"{REMOTE_DIR}/backtest/engine.py"),
    ("backtest/runner.py", f"{REMOTE_DIR}/backtest/runner.py"),
    ("backtest/stats.py", f"{REMOTE_DIR}/backtest/stats.py"),
    ("research_config.py", f"{REMOTE_DIR}/research_config.py"),
    ("indicators.py", f"{REMOTE_DIR}/indicators.py"),
]


def main():
    print("=== Deploying EXP-CHOPPY-OC to Server C ===\n")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    def ssh(cmd, timeout=30):
        _, o, e = client.exec_command(cmd, timeout=timeout)
        out = o.read().decode('utf-8', errors='replace').strip()
        err = e.read().decode('utf-8', errors='replace').strip()
        return out, err

    # Kill any existing experiment processes (not jupyter/tensorboard)
    print("Checking for running experiments...")
    out, _ = ssh("ps aux | grep 'run_exp\\|run_round' | grep python | grep -v grep")
    if out:
        print(f"  Found running experiments:\n  {out}")
        ssh("pkill -f 'run_exp_choppy' 2>/dev/null")
        print("  Killed existing choppy experiment if any")
    else:
        print("  No experiments running")

    # Ensure directories exist
    ssh(f"mkdir -p {REMOTE_DIR}/scripts/experiments")
    ssh(f"mkdir -p {REMOTE_DIR}/backtest")
    ssh(f"mkdir -p {REMOTE_DIR}/logs")
    ssh(f"mkdir -p {REMOTE_DIR}/results/choppy_oc_results")

    # Upload files
    print("\nUploading files...")
    for local_rel, remote_path in FILES_TO_UPLOAD:
        local_path = os.path.join(LOCAL_ROOT, local_rel)
        if not os.path.exists(local_path):
            print(f"  [SKIP] {local_rel} — not found locally")
            continue
        try:
            sftp.put(local_path, remote_path)
            size = os.path.getsize(local_path)
            print(f"  [OK] {local_rel} ({size:,} bytes)")
        except Exception as e:
            print(f"  [ERR] {local_rel}: {e}")

    # Start experiment using bash -c to properly background
    print("\nStarting experiment...")
    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u "
           f"scripts/experiments/run_exp_choppy_opportunity_cost.py "
           f"> logs/exp_choppy_oc.log 2>&1 & echo $!")
    _, o, _ = client.exec_command(f"bash -c '{cmd}'", timeout=10)
    import time
    time.sleep(2)
    try:
        pid_out = o.read(100).decode('utf-8', errors='replace').strip()
        print(f"  Background PID: {pid_out}")
    except Exception:
        print("  (nohup started, pid read timed out — normal)")
    time.sleep(3)

    # Verify
    out, _ = ssh("ps aux | grep run_exp_choppy | grep -v grep")
    if out:
        print(f"\n  Process running!")
        pid = out.split()[1]
        print(f"  PID: {pid}")
    else:
        print("\n  [WARN] Process not found! Checking log...")
        out, _ = ssh(f"tail -20 {REMOTE_DIR}/logs/exp_choppy_oc.log")
        print(f"  Log:\n{out}")

    sftp.close()
    client.close()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
