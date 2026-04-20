"""Pull complete Choppy OC output from D2 (45630)."""
import paramiko
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'choppy_oc_results')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    remote_files = [
        f"{REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt",
        f"{REMOTE_DIR}/logs/exp_choppy_oc.log",
    ]

    for rf in remote_files:
        fname = os.path.basename(rf)
        local = os.path.join(LOCAL_DIR, fname)
        try:
            sftp.get(rf, local)
            size = os.path.getsize(local)
            print(f"[OK] {fname} ({size:,} bytes)")
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

    sftp.close()
    client.close()
    print(f"\nSaved to {LOCAL_DIR}")


if __name__ == "__main__":
    main()
