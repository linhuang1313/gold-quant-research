"""Deploy Round 17 (Capital Curve Engineering) to Server D and start."""
import paramiko
import sys
import os
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

FILES_TO_UPLOAD = [
    ("backtest/engine.py", "backtest/engine.py"),
    ("backtest/runner.py", "backtest/runner.py"),
    ("backtest/stats.py", "backtest/stats.py"),
    ("indicators.py", "indicators.py"),
    ("research_config.py", "research_config.py"),
    ("experiments/run_round17.py", "scripts/experiments/run_round17.py"),
]


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            print(f"  Connecting to Server D (attempt {attempt+1}/5)...")
            client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
            print("  Connected!")
            return client
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 4:
                time.sleep(10)
    raise RuntimeError("Cannot connect to Server D")


def ssh_exec(client, cmd, timeout=30):
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
    print("Deploy Round 17 — Capital Curve Engineering -> Server D")
    print("=" * 60)

    client = connect()

    # Check for running experiments
    out, _ = ssh_exec(client, "ps aux | grep -E 'run_round' | grep python | grep -v grep")
    if out:
        print(f"\n  WARNING: Running processes found:")
        for line in out.split("\n"):
            parts = line.split()
            if len(parts) > 10:
                print(f"    PID={parts[1]} CMD={' '.join(parts[10:])}")
        print("\n  Kill existing experiments? Proceeding in 5s...")
        time.sleep(5)
        ssh_exec(client, "pkill -f 'run_round1[3-7]' 2>/dev/null")
        time.sleep(2)

    # System info
    print(f"\n--- Server Info ---")
    print(f"  Cores: {ssh_exec(client, 'nproc')[0]}")
    print(f"  Disk:  {ssh_exec(client, 'df -h / | tail -1')[0]}")
    print(f"  Mem:   {ssh_exec(client, 'free -h | grep Mem')[0]}")

    # Upload
    print(f"\n--- Uploading files ---")
    upload_files(client)

    # Create dirs
    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/logs {REMOTE_DIR}/results/round17_results")

    # Verify imports
    print(f"\n--- Verifying imports ---")
    out, err = ssh_exec(client,
        f"cd {REMOTE_DIR} && python3 -c '"
        "from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS; "
        "print(\"OK\")'",
        timeout=30)
    print(f"  Import check: {out}")
    if 'OK' not in out:
        print(f"  ERROR: {err}")
        client.close()
        return

    # Start R17
    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u scripts/experiments/run_round17.py "
           f"> logs/round17.log 2>&1 &")
    print(f"\n--- Starting R17 ---")
    client.exec_command(cmd)
    time.sleep(5)

    out, _ = ssh_exec(client, "ps aux | grep run_round17 | grep -v grep")
    if out:
        print("  R17 started successfully!")
        for line in out.split("\n"):
            parts = line.split()
            if len(parts) > 10:
                print(f"    PID={parts[1]} CPU={parts[2]}% CMD={' '.join(parts[10:])}")
    else:
        print("  WARNING: R17 may not have started. Checking logs...")
        out, _ = ssh_exec(client, f"tail -30 {REMOTE_DIR}/logs/round17.log 2>/dev/null",
                         timeout=30)
        print(out)

    # Check initial log
    time.sleep(3)
    out, _ = ssh_exec(client, f"tail -10 {REMOTE_DIR}/logs/round17.log 2>/dev/null")
    if out:
        print(f"\n--- Initial log ---")
        print(out)

    client.close()
    print(f"\n{'='*60}")
    print("Deploy complete! Monitor with: python deploy/_check_r17.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
