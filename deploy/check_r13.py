"""Check Round 13 experiment status, logs, and results."""
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

RESULT_FILES = [
    "R13-A1_ema_scan.txt",
    "R13-A2_mult_scan.txt",
    "R13-A3_heatmap.txt",
    "R13-A4_kfold.txt",
    "R13-B1_breakeven.txt",
    "R13-B2_be_kfold.txt",
    "R13-B3_exit_profile.txt",
    "R13-C1_dual_kc.txt",
    "R13-C2_dual_params.txt",
    "R13-C3_dual_kfold.txt",
    "R13-D1_hma.txt",
    "R13-D2_kama.txt",
    "R13-D3_ma_kfold.txt",
    "R13-E1_rolling_trail.txt",
    "R13-F_purged_wf.txt",
    "R13-G1_combined.txt",
    "R13-G2_monte_carlo.txt",
    "R13-G3_comparison.txt",
    "R13_summary.txt",
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

def check_status(client):
    print("\n--- Process Status ---")
    out, _ = ssh_exec(client, "ps aux | grep run_round13 | grep -v grep")
    if out:
        print(f"  R13 RUNNING: {out}")
    else:
        print("  R13 not running")

def check_log(client, lines=50):
    print(f"\n--- Last {lines} lines of log ---")
    out, _ = ssh_exec(client, f"tail -{lines} {REMOTE_DIR}/logs/round13.log 2>/dev/null",
                       timeout=30)
    if out:
        print(out)
    else:
        print("  No log found")

def check_results(client):
    print("\n--- Result Files ---")
    for fname in RESULT_FILES:
        out, _ = ssh_exec(client, f"wc -l {REMOTE_DIR}/round13_results/{fname} 2>/dev/null")
        if out and "No such file" not in out:
            print(f"  OK: {fname} ({out.split()[0]} lines)")
        else:
            print(f"  --: {fname}")

def download_results(client):
    print("\n--- Downloading Results ---")
    local_dir = os.path.join(LOCAL_DIR, "round13_results")
    os.makedirs(local_dir, exist_ok=True)
    sftp = client.open_sftp()
    downloaded = 0
    for fname in RESULT_FILES:
        remote_path = f"{REMOTE_DIR}/round13_results/{fname}"
        local_path = os.path.join(local_dir, fname)
        try:
            sftp.stat(remote_path)
            sftp.get(remote_path, local_path)
            print(f"  Downloaded: {fname}")
            downloaded += 1
        except FileNotFoundError:
            pass
    sftp.close()
    print(f"\n  Total downloaded: {downloaded}/{len(RESULT_FILES)}")

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"

    client = connect()

    if action == "status":
        check_status(client)
        check_results(client)
    elif action == "log":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        check_log(client, lines)
    elif action == "results":
        check_results(client)
    elif action == "download":
        download_results(client)
    elif action == "all":
        check_status(client)
        check_log(client, 30)
        check_results(client)
    elif action == "kill":
        ssh_exec(client, "pkill -f run_round13 2>/dev/null")
        print("Killed R13 processes")
    else:
        print(f"Usage: python check_r13.py [status|log|results|download|all|kill]")

    client.close()

if __name__ == '__main__':
    main()
