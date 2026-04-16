"""Deploy Round 13 files to server and start experiments."""
import paramiko
import sys
import os
import time

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
    ("strategies/signals.py", "strategies/signals.py"),
    ("config.py", "config.py"),
    ("scripts/experiments/run_round13.py", "scripts/experiments/run_round13.py"),
    ("scripts/check_r13.py", "scripts/check_r13.py"),
]

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            print(f"  Connecting (attempt {attempt+1}/5)...")
            client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
            print("  Connected!")
            return client
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < 4:
                time.sleep(10)
    raise RuntimeError("Cannot connect")

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
        print(f"  Uploading {local_rel}...")
        for attempt in range(3):
            try:
                sftp.put(local_path, remote_path)
                print(f"    OK ({os.path.getsize(local_path)} bytes)")
                break
            except Exception as e:
                print(f"    RETRY {attempt+1}: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)
    sftp.close()

def main():
    print("=== Deploy Round 13 ===")
    client = connect()

    out, _ = ssh_exec(client, "ps aux | grep run_round1 | grep -v grep")
    if out:
        print(f"\nRunning processes:\n{out}")
        print("WARNING: Other experiments may be running.")

    ssh_exec(client, "pkill -f run_round13 2>/dev/null")
    time.sleep(2)

    print("\n--- Uploading files ---")
    upload_files(client)

    print("\n--- Verifying imports ---")
    out, err = ssh_exec(client,
        f"cd {REMOTE_DIR} && python3 -c '"
        "from backtest.runner import DataBundle, run_variant, _hma, _kama, add_dual_kc; "
        "print(\"OK\")'",
        timeout=30)
    print(f"  Import check: {out}")
    if 'OK' not in out:
        print(f"  ERROR: {err}")
        client.close()
        return

    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/logs {REMOTE_DIR}/round13_results")

    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u scripts/experiments/run_round13.py "
           f"> logs/round13.log 2>&1 &")
    print(f"\n--- Starting R13 ---")
    client.exec_command(cmd)
    time.sleep(5)

    out, _ = ssh_exec(client, "ps aux | grep run_round13 | grep -v grep")
    if out:
        print("R13 started successfully!")
        print(out)
    else:
        print("WARNING: R13 may not have started. Check logs.")
        out, _ = ssh_exec(client, f"tail -30 {REMOTE_DIR}/logs/round13.log 2>/dev/null",
                         timeout=30)
        print(out)

    client.close()

if __name__ == '__main__':
    main()
