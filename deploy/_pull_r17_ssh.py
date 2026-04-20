"""Pull R17 results via SSH cat (faster than SFTP for small files)."""
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


def ssh(client, cmd, timeout=30):
    _, o, _ = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='replace')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    file_list = ssh(client, f"ls {REMOTE_DIR}/").strip().split('\n')
    for f in sorted(file_list):
        f = f.strip()
        if not f:
            continue
        content = ssh(client, f"cat {REMOTE_DIR}/{f}", timeout=15)
        local = os.path.join(LOCAL_DIR, f)
        with open(local, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print(f"  [OK] {f} ({len(content):,} chars)")

    client.close()
    print(f"\nDone: {len(file_list)} files")


if __name__ == "__main__":
    main()
