#!/usr/bin/env python3
"""Deploy R182 Robustness Validation to remote server and launch."""
import paramiko, time, os
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 41109
USER = "root"
PASS = "3sCdENtzYfse"

REMOTE_BASE = "/root/gold-quant-research"

SCRIPTS = [
    "experiments/run_r181_full_audit.py",
    "experiments/run_r182_robustness.py",
]
RESULT_DIRS = ["results/r182_robustness"]


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


def upload_file(local_path, remote_path, retries=5):
    for attempt in range(retries):
        try:
            ssh = connect()
            run(ssh, f"mkdir -p {os.path.dirname(remote_path)}")
            sftp = ssh.open_sftp()
            sftp.put(str(local_path), remote_path)
            sftp.close()
            ssh.close()
            return
        except Exception as e:
            print(f"\n    Upload attempt {attempt+1} failed: {e}", end="", flush=True)
            try: ssh.close()
            except: pass
            time.sleep(5)
    raise RuntimeError(f"Upload failed after {retries} attempts: {local_path}")


def main():
    base = Path(__file__).parent.parent

    print("=" * 70)
    print("  R182 Robustness Validation -- Deployment")
    print(f"  Server: {HOST}:{PORT}")
    print("=" * 70)

    print("\n  Step 1: Uploading scripts...")
    for script in SCRIPTS:
        local_path = base / script
        remote_path = f"{REMOTE_BASE}/{script}"
        print(f"    Uploading {script}...", end=" ", flush=True)
        upload_file(local_path, remote_path)
        print("OK")

    print("\n  Step 2: Creating result directories...")
    ssh = connect()
    for d in RESULT_DIRS:
        run(ssh, f"mkdir -p {REMOTE_BASE}/{d}")
    ssh.close()
    print("    Done.")

    print("\n  Step 3: Launching R182...")
    ssh = connect()
    cmd = (
        f"cd {REMOTE_BASE} && "
        f"nohup python3 experiments/run_r182_robustness.py "
        f"> results/r182_robustness/r182_stdout.txt 2>&1 &"
    )
    ssh.exec_command(cmd)
    time.sleep(3)

    out, _ = run(ssh, "ps aux | grep run_r182 | grep -v grep")
    if out.strip():
        pid = out.strip().split()[1]
        print(f"    R182 running (PID: {pid})")
    else:
        print("    WARNING: R182 process not found, checking output...")
        out, _ = run(ssh, f"tail -10 {REMOTE_BASE}/results/r182_robustness/r182_stdout.txt 2>/dev/null")
        print(f"    {out.strip()[:300]}")
    ssh.close()

    print(f"\n{'=' * 70}")
    print("  R182 launched successfully.")
    print(f"  Monitor: tail -f {REMOTE_BASE}/results/r182_robustness/r182_stdout.txt")
    print(f"  Estimated runtime: 5-15 minutes")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
