#!/usr/bin/env python3
"""Deploy R183 Keltner R:R Optimization to remote server and launch."""
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
    raise RuntimeError(f"Upload failed after {retries} attempts")


def main():
    base = Path(__file__).parent.parent
    print("=" * 70)
    print("  R183 Keltner R:R Optimization -- Deployment")
    print(f"  Server: {HOST}:{PORT}")
    print("=" * 70)

    print("\n  Step 1: Uploading R183...")
    local = base / "experiments" / "run_r183_keltner_rr.py"
    upload_file(local, f"{REMOTE_BASE}/experiments/run_r183_keltner_rr.py")
    print(" OK")

    print("\n  Step 2: Creating result dir...")
    ssh = connect()
    run(ssh, f"mkdir -p {REMOTE_BASE}/results/r183_keltner_rr")
    ssh.close()
    print("    Done.")

    print("\n  Step 3: Launching R183...")
    ssh = connect()
    cmd = (f"cd {REMOTE_BASE} && nohup python3 experiments/run_r183_keltner_rr.py "
           f"> results/r183_keltner_rr/r183_stdout.txt 2>&1 &")
    ssh.exec_command(cmd)
    time.sleep(3)
    out, _ = run(ssh, "ps aux | grep run_r183 | grep -v grep")
    if out.strip():
        pid = out.strip().split()[1]
        print(f"    R183 running (PID: {pid})")
    else:
        print("    WARNING: process not found, checking output...")
        out, _ = run(ssh, f"tail -10 {REMOTE_BASE}/results/r183_keltner_rr/r183_stdout.txt 2>/dev/null")
        print(f"    {out.strip()[:300]}")
    ssh.close()

    print(f"\n{'=' * 70}")
    print(f"  R183 launched. Estimated runtime: 5-15 min")
    print(f"  Monitor: tail -f {REMOTE_BASE}/results/r183_keltner_rr/r183_stdout.txt")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
