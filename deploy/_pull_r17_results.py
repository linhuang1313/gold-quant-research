"""Pull all R17 result files from D1 (35258)."""
import paramiko
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/results/round17_results"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'round17_results')


def ssh(client, cmd):
    _, o, _ = client.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    files = sftp.listdir(REMOTE_DIR)
    for f in sorted(files):
        remote = f"{REMOTE_DIR}/{f}"
        local = os.path.join(LOCAL_DIR, f)
        sftp.get(remote, local)
        size = os.path.getsize(local)
        print(f"  [OK] {f} ({size:,} bytes)")

    sftp.close()
    client.close()
    print(f"\nTotal: {len(files)} files -> {LOCAL_DIR}")


if __name__ == "__main__":
    main()
