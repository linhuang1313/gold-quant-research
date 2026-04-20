"""Pull all R18 results from Server C."""
import paramiko
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
BASE = "/root/gold-quant-trading/results/round18_results/"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    def ssh(cmd):
        _, o, _ = client.exec_command(cmd, timeout=30)
        return o.read().decode('utf-8', errors='replace').strip()

    files = ssh(f"ls -1 {BASE}").split('\n')
    for f in files:
        f = f.strip()
        if not f:
            continue
        print(f"=== {f} ===")
        content = ssh(f"cat {BASE}{f}")
        print(content)
        print()

    client.close()

if __name__ == "__main__":
    main()
