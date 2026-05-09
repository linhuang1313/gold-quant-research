#!/usr/bin/env python3
"""Deploy R184 Keltner Entry Filter Impact to remote server."""
import paramiko, time, os
from pathlib import Path

HOST = "connect.westd.seetacloud.com"
PORT = 41109
USER = "root"
PASS = "3sCdENtzYfse"
REMOTE_BASE = "/root/gold-quant-research"


def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, banner_timeout=30)
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
            sftp.close(); ssh.close()
            return
        except Exception as e:
            print(f"\n    Upload attempt {attempt+1} failed: {e}", end="", flush=True)
            try: ssh.close()
            except: pass
            time.sleep(5)
    raise RuntimeError(f"Upload failed after {retries} attempts")


def main():
    base = Path(__file__).parent.parent
    print("=" * 70)
    print("  R184 Keltner Entry Filter Impact -- Deployment")
    print("=" * 70)

    print("\n  Uploading...", end=" ", flush=True)
    upload_file(base / "experiments" / "run_r184_keltner_filters.py",
                f"{REMOTE_BASE}/experiments/run_r184_keltner_filters.py")
    print("OK")

    ssh = connect()
    run(ssh, f"mkdir -p {REMOTE_BASE}/results/r184_keltner_filters")
    cmd = (f"cd {REMOTE_BASE} && nohup python3 experiments/run_r184_keltner_filters.py "
           f"> results/r184_keltner_filters/r184_stdout.txt 2>&1 &")
    ssh.exec_command(cmd)
    time.sleep(3)
    out, _ = run(ssh, "ps aux | grep run_r184 | grep -v grep")
    if out.strip():
        print(f"  R184 running (PID: {out.strip().split()[1]})")
    else:
        print("  WARNING: process not found")
        out, _ = run(ssh, f"tail -5 {REMOTE_BASE}/results/r184_keltner_filters/r184_stdout.txt 2>/dev/null")
        print(f"  {out.strip()[:200]}")
    ssh.close()
    print(f"\n  Estimated runtime: 5-15 min")
    print(f"  Monitor: tail -f {REMOTE_BASE}/results/r184_keltner_filters/r184_stdout.txt")


if __name__ == '__main__':
    main()
