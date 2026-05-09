#!/usr/bin/env python3
"""Deploy R150, R151, R153 to remote server and launch."""
import paramiko, time, os, sys
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 41109
USER = "root"
PASS = "3sCdENtzYfse"

SCRIPTS = [
    "experiments/run_r150_6strat_lot_optimizer.py",
    "experiments/run_r151_ml_entry_test.py",
    "experiments/run_r153_correlation_monitor.py",
]

RESULT_DIRS = [
    "results/r150_6strat_lot_optimizer",
    "results/r151_ml_entry_test",
    "results/r153_correlation_monitor",
]

def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=30, banner_timeout=30)
            return c
        except Exception as e:
            print(f"  Connect attempt {attempt+1} failed: {e}")
            time.sleep(3)
    raise RuntimeError("Cannot connect after 5 attempts")

def run(ssh, cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(), e.read().decode()

def main():
    base = Path(__file__).parent.parent

    # Step 1: Upload scripts
    print("=" * 60)
    print("  Deploying R150, R151, R153 to remote server")
    print("=" * 60)

    for script in SCRIPTS:
        local_path = base / script
        if not local_path.exists():
            print(f"  ERROR: {local_path} not found!")
            return
        print(f"  Found: {script}")

    for script in SCRIPTS:
        local_path = base / script
        remote_path = f"/root/gold-quant-research/{script}"

        print(f"\n  Uploading {script}...")
        ssh = connect()
        sftp = ssh.open_sftp()
        sftp.put(str(local_path), remote_path)
        sftp.close()
        ssh.close()
        print(f"    Done.")

    # Step 2: Create result dirs
    print("\n  Creating result directories...")
    ssh = connect()
    for d in RESULT_DIRS:
        run(ssh, f"mkdir -p /root/gold-quant-research/{d}")
    ssh.close()

    # Step 3: Launch experiments
    experiments = [
        ("R150", "run_r150_6strat_lot_optimizer.py", "r150_6strat_lot_optimizer/r150_stdout.txt"),
        ("R151", "run_r151_ml_entry_test.py", "r151_ml_entry_test/r151_stdout.txt"),
        ("R153", "run_r153_correlation_monitor.py", "r153_correlation_monitor/r153_stdout.txt"),
    ]

    for name, script, stdout_file in experiments:
        print(f"\n  Launching {name}...")
        ssh = connect()
        cmd = (
            f"cd /root/gold-quant-research && "
            f"nohup python3 experiments/{script} "
            f"> results/{stdout_file} 2>&1 &"
        )
        ssh.exec_command(cmd)
        time.sleep(2)

        out, _ = run(ssh, f"ps aux | grep {script} | grep -v grep")
        if out.strip():
            pid = out.strip().split()[1]
            print(f"    {name} running (PID: {pid})")
        else:
            print(f"    WARNING: {name} process not found, checking output...")
            out, _ = run(ssh, f"tail -5 /root/gold-quant-research/results/{stdout_file} 2>/dev/null")
            print(f"    {out.strip()[:200]}")
        ssh.close()

    # Step 4: Quick status check
    print(f"\n{'=' * 60}")
    print("  All experiments launched. Status check:")
    print("=" * 60)
    time.sleep(3)

    ssh = connect()
    out, _ = run(ssh, "ps aux | grep 'run_r15[0-9]' | grep -v grep")
    for line in out.strip().split('\n'):
        if line.strip():
            parts = line.split()
            pid = parts[1]
            cmd_part = ' '.join(parts[10:])
            print(f"  PID {pid}: {cmd_part[:80]}")

    print("\n  Monitor with:")
    print("    tail -f results/r150_6strat_lot_optimizer/r150_stdout.txt")
    print("    tail -f results/r151_ml_entry_test/r151_stdout.txt")
    print("    tail -f results/r153_correlation_monitor/r153_stdout.txt")
    ssh.close()


if __name__ == '__main__':
    main()
