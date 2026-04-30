"""Deploy Round 50 (L8 Full-Blast Grid Search) to Server and start."""
import paramiko
import sys
import os
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 41109
USER = "root"
PASS = "3sCdENtzYfse"
REMOTE_DIR = "/root/gold-quant-research"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

FILES_TO_UPLOAD = [
    ("backtest/__init__.py", "backtest/__init__.py"),
    ("backtest/engine.py", "backtest/engine.py"),
    ("backtest/runner.py", "backtest/runner.py"),
    ("backtest/stats.py", "backtest/stats.py"),
    ("backtest/fast_screen.py", "backtest/fast_screen.py"),
    ("indicators.py", "indicators.py"),
    ("research_config.py", "research_config.py"),
    ("experiments/run_round50_brute_force.py", "experiments/run_round50_brute_force.py"),
]


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            print(f"  Connecting to {HOST}:{PORT} (attempt {attempt+1}/5)...")
            client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
            print("  Connected!")
            return client
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 4:
                time.sleep(10)
    raise RuntimeError(f"Cannot connect to {HOST}:{PORT}")


def ssh_exec(client, cmd, timeout=60):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    return out.strip(), err.strip()


def upload_files(client):
    sftp = client.open_sftp()
    for local_rel, remote_rel in FILES_TO_UPLOAD:
        local_path = os.path.join(LOCAL_DIR, local_rel)
        remote_path = f"{REMOTE_DIR}/{remote_rel}"
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        remote_dir = os.path.dirname(remote_path).replace("\\", "/")
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            ssh_exec(client, f"mkdir -p {remote_dir}")

        print(f"  Uploading {local_rel} -> {remote_rel}...")
        for attempt in range(3):
            try:
                sftp.put(local_path, remote_path)
                print(f"    OK ({os.path.getsize(local_path):,} bytes)")
                break
            except Exception as e:
                print(f"    RETRY {attempt+1}: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)
    sftp.close()


def main():
    print("=" * 60)
    print("Deploy R50 — L8 Full-Blast Grid Search")
    print(f"  Server: {HOST}:{PORT}")
    print(f"  Remote: {REMOTE_DIR}")
    print("=" * 60)

    client = connect()

    # Check for running experiments
    out, _ = ssh_exec(client, "ps aux | grep -E 'run_round' | grep python | grep -v grep")
    if out:
        print(f"\n  WARNING: Running experiments found:")
        for line in out.split("\n"):
            parts = line.split()
            if len(parts) > 10:
                print(f"    PID={parts[1]} CMD={' '.join(parts[10:])}")
        resp = input("\n  Continue anyway? (y/N): ").strip().lower()
        if resp != 'y':
            print("  Aborted.")
            client.close()
            return

    # System info
    print(f"\n--- Server Info ---")
    cores, _ = ssh_exec(client, 'nproc')
    print(f"  Cores: {cores}")
    mem, _ = ssh_exec(client, 'free -h | grep Mem')
    print(f"  Mem:   {mem}")
    disk, _ = ssh_exec(client, 'df -h / | tail -1')
    print(f"  Disk:  {disk}")

    # Ensure repo structure
    print(f"\n--- Setting up directories ---")
    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/backtest {REMOTE_DIR}/experiments "
             f"{REMOTE_DIR}/results/round50_results {REMOTE_DIR}/data/download")

    # Upload files
    print(f"\n--- Uploading files ---")
    upload_files(client)

    # Check data files
    print(f"\n--- Checking data files ---")
    out, _ = ssh_exec(client, f"ls -la {REMOTE_DIR}/data/download/xauusd-*.csv 2>/dev/null | head -10")
    if out:
        print(f"  Data files found:")
        for line in out.split("\n"):
            print(f"    {line}")
    else:
        print("  WARNING: No data files found! You need to upload data first.")
        print(f"  Expected: {REMOTE_DIR}/data/download/xauusd-m15-bid-*.csv")
        print(f"            {REMOTE_DIR}/data/download/xauusd-h1-bid-*.csv")
        client.close()
        return

    # Verify imports
    print(f"\n--- Verifying imports ---")
    out, err = ssh_exec(client,
        f"cd {REMOTE_DIR} && python3 -c '"
        "from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS; "
        "from backtest.stats import deflated_sharpe, compute_pbo, probabilistic_sharpe; "
        "print(\"IMPORT_OK\")'",
        timeout=30)
    print(f"  Import check: {out}")
    if 'IMPORT_OK' not in out:
        print(f"  ERROR: {err}")
        client.close()
        return

    # Start R50
    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u experiments/run_round50_brute_force.py "
           f"> results/round50_results/stdout.txt 2>&1 &")
    print(f"\n--- Starting R50 ---")
    print(f"  Command: {cmd}")
    client.exec_command(cmd)
    time.sleep(5)

    out, _ = ssh_exec(client, "ps aux | grep run_round50 | grep -v grep")
    if out:
        print("  R50 started successfully!")
        for line in out.split("\n"):
            parts = line.split()
            if len(parts) > 10:
                print(f"    PID={parts[1]} CPU={parts[2]}% CMD={' '.join(parts[10:])}")
    else:
        print("  WARNING: R50 may not have started. Checking logs...")
        out, _ = ssh_exec(client,
            f"tail -30 {REMOTE_DIR}/results/round50_results/stdout.txt 2>/dev/null",
            timeout=30)
        print(out)

    # Check initial log
    time.sleep(3)
    out, _ = ssh_exec(client,
        f"tail -20 {REMOTE_DIR}/results/round50_results/stdout.txt 2>/dev/null")
    if out:
        print(f"\n--- Initial log ---")
        print(out)

    client.close()
    print(f"\n{'='*60}")
    print("Deploy complete!")
    print(f"Monitor:  python deploy/_check_r50.py")
    print(f"SSH tail: ssh -p {PORT} {USER}@{HOST} "
          f"'tail -f {REMOTE_DIR}/results/round50_results/stdout.txt'")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
