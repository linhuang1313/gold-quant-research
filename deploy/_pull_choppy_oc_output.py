"""Pull the full output file from Server C for EXP-CHOPPY-OC."""
import paramiko
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

LOCAL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'exp_choppy_oc_output.txt')


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()

    remote_path = f"{REMOTE_DIR}/exp_choppy_opportunity_cost_output.txt"

    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)

    try:
        sftp.get(remote_path, LOCAL_OUT)
        size = os.path.getsize(LOCAL_OUT)
        print(f"Downloaded output file: {size:,} bytes -> {LOCAL_OUT}")
    except Exception as e:
        print(f"Error: {e}")

    # Also get the log
    local_log = LOCAL_OUT.replace('_output.txt', '_log.txt')
    try:
        sftp.get(f"{REMOTE_DIR}/logs/exp_choppy_oc.log", local_log)
        size = os.path.getsize(local_log)
        print(f"Downloaded log file: {size:,} bytes -> {local_log}")
    except Exception as e:
        print(f"Log error: {e}")

    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
