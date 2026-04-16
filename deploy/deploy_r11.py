"""Deploy Round 11 experiment to server and start running."""
import paramiko
import os
import sys
import time
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"

PROJECT_ROOT = Path(__file__).parent.parent
REMOTE_DIR = "/root/gold-quant-trading"

FILES_TO_UPLOAD = [
    "strategies/signals.py",
    "backtest/engine.py",
    "backtest/runner.py",
    "backtest/__init__.py",
    "backtest/stats.py",
    "scripts/experiments/run_round11.py",
    "config.py",
    "paper_trader.py",
]

def ssh_exec(client, cmd, timeout=30):
    print(f"  $ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        print(f"    {out.strip()[:500]}")
    if err.strip():
        print(f"    [stderr] {err.strip()[:300]}")
    return out, err

def main():
    print(f"Connecting to {HOST}:{PORT}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
    print("Connected!\n")

    # Check server state
    print("=== Server Info ===")
    ssh_exec(client, "nproc; python3 --version; df -h / | tail -1")

    # Check if project exists
    print("\n=== Checking project ===")
    out, _ = ssh_exec(client, f"ls {REMOTE_DIR}/backtest/engine.py 2>/dev/null && echo EXISTS || echo MISSING")

    if "MISSING" in out:
        print("\nProject not found on server. Need full upload.")
        print("Cloning from git or uploading full project...")
        # Try git clone first
        ssh_exec(client, f"mkdir -p {REMOTE_DIR}")
        # Upload all essential files
        sftp = client.open_sftp()
        # Create directory structure
        for f in FILES_TO_UPLOAD:
            remote_path = f"{REMOTE_DIR}/{f}"
            remote_dir = os.path.dirname(remote_path)
            try:
                sftp.stat(remote_dir)
            except FileNotFoundError:
                # Create dirs recursively
                parts = remote_dir.split('/')
                for i in range(2, len(parts) + 1):
                    d = '/'.join(parts[:i])
                    try:
                        sftp.stat(d)
                    except FileNotFoundError:
                        sftp.mkdir(d)
        sftp.close()

    # Upload modified files
    print("\n=== Uploading files ===")
    sftp = client.open_sftp()
    for f in FILES_TO_UPLOAD:
        local_path = str(PROJECT_ROOT / f)
        remote_path = f"{REMOTE_DIR}/{f}"
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {f}")
            continue
        # Ensure remote dir exists
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            parts = remote_dir.split('/')
            for i in range(2, len(parts) + 1):
                d = '/'.join(parts[:i])
                try:
                    sftp.stat(d)
                except FileNotFoundError:
                    sftp.mkdir(d)
        sftp.put(local_path, remote_path)
        size = os.path.getsize(local_path)
        print(f"  OK: {f} ({size:,} bytes)")
    sftp.close()

    # Upload data files if not present
    print("\n=== Checking data ===")
    out, _ = ssh_exec(client, f"ls {REMOTE_DIR}/data/download/xauusd-m15-bid-*.csv 2>/dev/null | head -1")
    if not out.strip():
        print("  Data files not found on server!")
        print("  Checking for data in common locations...")
        out2, _ = ssh_exec(client, "find / -name 'xauusd-m15-bid-*.csv' -maxdepth 5 2>/dev/null | head -3")
        if out2.strip():
            data_path = out2.strip().split('\n')[0]
            data_dir = os.path.dirname(data_path)
            print(f"  Found data at: {data_dir}")
            ssh_exec(client, f"mkdir -p {REMOTE_DIR}/data/download; ln -sf {data_dir}/*.csv {REMOTE_DIR}/data/download/ 2>/dev/null")
        else:
            print("  WARNING: No data files on server. Need to upload ~100MB of CSV data.")
            print("  Uploading data files...")
            sftp = client.open_sftp()
            data_dir = PROJECT_ROOT / "data" / "download"
            try:
                sftp.stat(f"{REMOTE_DIR}/data/download")
            except FileNotFoundError:
                try:
                    sftp.stat(f"{REMOTE_DIR}/data")
                except FileNotFoundError:
                    sftp.mkdir(f"{REMOTE_DIR}/data")
                sftp.mkdir(f"{REMOTE_DIR}/data/download")

            for csv_file in data_dir.glob("xauusd-*.csv"):
                remote_csv = f"{REMOTE_DIR}/data/download/{csv_file.name}"
                size_mb = csv_file.stat().st_size / 1024 / 1024
                print(f"  Uploading {csv_file.name} ({size_mb:.1f}MB)...", end='', flush=True)
                sftp.put(str(csv_file), remote_csv)
                print(" OK")
            sftp.close()
    else:
        print(f"  Data found: {out.strip()}")

    # Install dependencies
    print("\n=== Installing dependencies ===")
    ssh_exec(client, "pip install pandas numpy scipy -q 2>/dev/null; python3 -c 'import pandas,numpy,scipy; print(\"deps OK\")'", timeout=60)

    # Verify setup
    print("\n=== Verifying setup ===")
    ssh_exec(client, f"cd {REMOTE_DIR} && python3 -c \"from backtest.engine import BacktestEngine; print('engine OK')\"")
    ssh_exec(client, f"cd {REMOTE_DIR} && python3 -c \"from strategies.signals import prepare_indicators; print('signals OK')\"")

    # Create output dir and start experiment
    print("\n=== Starting Round 11 ===")
    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/round11_results {REMOTE_DIR}/logs")

    # Start Phase 1 first (quick IC analysis)
    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u scripts/experiments/run_round11.py "
           f"> logs/round11.log 2>&1 &")
    print(f"  Launching: {cmd}")
    ssh_exec(client, cmd)
    time.sleep(2)

    # Check if it's running
    out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
    if "run_round11" in out:
        print("\n✅ Round 11 is running!")
        print(f"   Monitor: ssh -p {PORT} {USER}@{HOST} 'tail -f {REMOTE_DIR}/logs/round11.log'")
    else:
        print("\n❌ Process not found. Checking log...")
        ssh_exec(client, f"tail -20 {REMOTE_DIR}/logs/round11.log")

    # Show initial log
    time.sleep(3)
    print("\n=== Initial log output ===")
    ssh_exec(client, f"tail -30 {REMOTE_DIR}/logs/round11.log")

    client.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
