"""Pull choppy OC output from Server D2 (port 45630)."""
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
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def ssh(client, cmd):
    _, o, _ = client.exec_command(cmd, timeout=30)
    return o.read().decode('utf-8', errors='replace').strip()


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    # Check if choppy experiment is still running
    procs = ssh(client, "ps aux | grep run_exp_choppy | grep python | grep -v grep")
    if procs:
        print("Choppy experiment STILL RUNNING:")
        print(procs)
    else:
        print("Choppy experiment: NOT running (finished or never started)")

    # Pull the output file
    remote_file = f"{REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt"
    local_file = os.path.join(LOCAL_DIR, "exp_choppy_oc_d2_output.txt")
    try:
        sftp.get(remote_file, local_file)
        size = os.path.getsize(local_file)
        print(f"\nDownloaded: {size:,} bytes -> {local_file}")
    except Exception as e:
        print(f"Error: {e}")

    # Also get the log
    remote_log = f"{REMOTE_DIR}/logs/exp_choppy_oc.log"
    local_log = os.path.join(LOCAL_DIR, "exp_choppy_oc_d2_log.txt")
    try:
        sftp.get(remote_log, local_log)
        size = os.path.getsize(local_log)
        print(f"Downloaded log: {size:,} bytes -> {local_log}")
    except Exception as e:
        print(f"Log: {e}")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
