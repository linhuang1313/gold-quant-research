"""Pull all R13 result files from Server D."""
import paramiko
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/round13_results"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'round13')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    files = sftp.listdir(REMOTE_DIR)
    print(f"Found {len(files)} files in {REMOTE_DIR}\n")

    for f in sorted(files):
        remote_path = f"{REMOTE_DIR}/{f}"
        local_path = os.path.join(LOCAL_DIR, f)
        try:
            sftp.get(remote_path, local_path)
            size = os.path.getsize(local_path)
            print(f"  [OK] {f} ({size:,} bytes)")
        except Exception as e:
            print(f"  [ERR] {f}: {e}")

    sftp.close()
    client.close()
    print(f"\nAll files saved to: {LOCAL_DIR}")


if __name__ == "__main__":
    main()
