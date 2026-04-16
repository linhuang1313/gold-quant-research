"""Deploy Round 12 files to server and start experiments."""
import paramiko
import sys
import os
import time

HOST = "connect.westd.seetacloud.com"
PORT = 30367
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

FILES_TO_UPLOAD = [
    ("strategies/signals.py", "strategies/signals.py"),
    ("backtest/engine.py", "backtest/engine.py"),
    ("backtest/runner.py", "backtest/runner.py"),
    ("scripts/experiments/run_round12.py", "scripts/experiments/run_round12.py"),
    ("scripts/check_r12.py", "scripts/check_r12.py"),
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

def ssh_exec(client, cmd, timeout=15):
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
    wait_for_r11 = "--wait" in sys.argv
    phase = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ('--wait',) else None

    print("=== Deploy Round 12 ===")
    client = connect()

    # Check R11
    out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
    if out:
        print(f"\nR11 is still running!")
        if wait_for_r11:
            print("Waiting for R11 to finish...")
            while True:
                time.sleep(60)
                out, _ = ssh_exec(client, "ps aux | grep run_round11 | grep -v grep")
                if not out:
                    print("R11 finished!")
                    break
                print("  Still running...")
        else:
            print("WARNING: R11 still running. Will proceed with R12 deployment.")

    # Kill any existing R12
    ssh_exec(client, "pkill -f run_round12 2>/dev/null")
    time.sleep(2)

    # Upload
    print("\n--- Uploading files ---")
    upload_files(client)

    # Verify imports
    print("\n--- Verifying imports ---")
    out, err = ssh_exec(client,
        f"cd {REMOTE_DIR} && python3 -c 'from backtest.runner import DataBundle, run_variant; print(\"OK\")'",
        timeout=30)
    print(f"  Import check: {out}")
    if 'OK' not in out:
        print(f"  ERROR: {err}")
        client.close()
        return

    # Create directories
    ssh_exec(client, f"mkdir -p {REMOTE_DIR}/logs {REMOTE_DIR}/round12_results")

    # Start R12
    phase_arg = phase if phase else ""
    cmd = (f"cd {REMOTE_DIR} && nohup python3 -u scripts/experiments/run_round12.py {phase_arg} "
           f"> logs/round12.log 2>&1 &")
    print(f"\n--- Starting R12 {phase_arg or '(all phases)'} ---")
    client.exec_command(cmd)
    time.sleep(5)

    out, _ = ssh_exec(client, "ps aux | grep run_round12 | grep -v grep")
    if out:
        print("R12 started successfully!")
        print(out)
    else:
        print("WARNING: R12 may not have started. Check logs.")
        out, _ = ssh_exec(client, f"tail -20 {REMOTE_DIR}/logs/round12.log 2>/dev/null")
        print(out)

    client.close()

if __name__ == '__main__':
    main()
