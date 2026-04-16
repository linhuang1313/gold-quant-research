"""Deploy Round 11 Phase 6-8 extension to server."""
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
    "scripts/experiments/run_round11.py",
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

    # Check current R11 status
    print("=== R11 Status ===")
    out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
    r11_running = "run_round11" in out

    if r11_running:
        print("  Phase 1-5 still running. Checking progress...")
        ssh_exec(client, f"tail -5 {REMOTE_DIR}/logs/round11.log")
        print("\n  Will upload new code and start Phase 6-8 after current run finishes,")
        print("  or start Phase 6-8 in parallel (they use separate output files).")

    # Upload modified files
    print("\n=== Uploading updated files ===")
    sftp = client.open_sftp()
    for f in FILES_TO_UPLOAD:
        local_path = str(PROJECT_ROOT / f)
        remote_path = f"{REMOTE_DIR}/{f}"
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {f}")
            continue
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

    # Verify import
    print("\n=== Verifying ===")
    ssh_exec(client, f"cd {REMOTE_DIR} && python3 -c \"from backtest.engine import BacktestEngine; print('engine OK')\"")
    ssh_exec(client, f"cd {REMOTE_DIR} && python3 -c \"from strategies.signals import prepare_indicators; print('signals OK')\"")

    if r11_running:
        # Kill the old process and restart with all phases
        print("\n=== Stopping old R11 process ===")
        ssh_exec(client, "pkill -f run_round11 || true")
        time.sleep(2)

    # Start Phase 6-8 (new experiments only)
    print("\n=== Starting Phase 6-8 ===")
    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/round11_results {REMOTE_DIR}/logs")

    # Run phases 6, 7, 8 sequentially
    cmd = (f"cd {REMOTE_DIR} && nohup bash -c '"
           f"python3 -u scripts/experiments/run_round11.py 6 >> logs/round11_ext.log 2>&1 && "
           f"python3 -u scripts/experiments/run_round11.py 7 >> logs/round11_ext.log 2>&1 && "
           f"python3 -u scripts/experiments/run_round11.py 8 >> logs/round11_ext.log 2>&1"
           f"' > /dev/null 2>&1 &")
    print(f"  Launching phases 6-8...")
    ssh_exec(client, cmd)
    time.sleep(3)

    # Check if it's running
    out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
    if "run_round11" in out:
        print("\n  Phase 6-8 is running!")
        print(f"  Monitor: ssh -p {PORT} {USER}@{HOST} 'tail -f {REMOTE_DIR}/logs/round11_ext.log'")
    else:
        print("\n  Process may not have started. Checking log...")
        ssh_exec(client, f"tail -20 {REMOTE_DIR}/logs/round11_ext.log")

    time.sleep(2)
    print("\n=== Initial log output ===")
    ssh_exec(client, f"tail -30 {REMOTE_DIR}/logs/round11_ext.log")

    client.close()
    print("\nDone!")

if __name__ == '__main__':
    main()
