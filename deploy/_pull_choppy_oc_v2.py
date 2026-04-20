"""Pull Choppy OC output from D2 via SSH read (bypass SFTP issues)."""
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


def ssh(client, cmd, timeout=60):
    _, o, _ = client.exec_command(cmd, timeout=timeout)
    return o.read().decode('utf-8', errors='replace')


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to D2...")
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30,
                   banner_timeout=30, auth_timeout=30)

    print("Reading output file...")
    content = ssh(client, f"cat {REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt", timeout=120)

    local_path = os.path.join(LOCAL_DIR, "exp_choppy_opportunity_cost_output.txt")
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved: {local_path} ({len(content):,} chars)")

    client.close()


if __name__ == "__main__":
    main()
