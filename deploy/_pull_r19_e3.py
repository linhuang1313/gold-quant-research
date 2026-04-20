"""Pull R19 E3 + summary from D2 via SSH cat."""
import paramiko
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 45630
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/results/round19_results"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'round19_results')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

    for fname in ["R19-E3_walk_forward.txt", "R19_summary.txt"]:
        _, o, _ = client.exec_command(f"cat {REMOTE_DIR}/{fname}", timeout=10)
        content = o.read().decode('utf-8', errors='replace')
        local = os.path.join(LOCAL_DIR, fname)
        with open(local, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] {fname} ({len(content)} chars)")

    client.close()


if __name__ == "__main__":
    main()
