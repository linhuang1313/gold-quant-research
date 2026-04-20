"""Pull R17 results with retry and keepalive."""
import paramiko
import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading/results/round17_results"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'round17_results')


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    transport = client.get_transport()
    transport.set_keepalive(10)
    return client


def ssh(client, cmd, timeout=15):
    _, o, _ = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='replace')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    for attempt in range(3):
        try:
            print(f"Attempt {attempt+1}...")
            client = connect()
            file_list = ssh(client, f"ls {REMOTE_DIR}/").strip().split('\n')
            print(f"  Found {len(file_list)} files")

            for f in sorted(file_list):
                f = f.strip()
                if not f:
                    continue
                try:
                    content = ssh(client, f"cat {REMOTE_DIR}/{f}", timeout=10)
                    local = os.path.join(LOCAL_DIR, f)
                    with open(local, 'w', encoding='utf-8') as fh:
                        fh.write(content)
                    print(f"  [OK] {f} ({len(content):,} chars)")
                except Exception as e:
                    print(f"  [FAIL] {f}: {e}")

            client.close()
            print("Done!")
            return
        except Exception as e:
            print(f"  Failed: {e}")
            time.sleep(5)

    print("All attempts failed!")


if __name__ == "__main__":
    main()
