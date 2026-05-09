"""Download R96-R100 results from remote server."""
import paramiko
import os
import json
import time

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research/results'
LOCAL_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def connect():
    for attempt in range(5):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
                      timeout=120, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:
            print(f"  connect attempt {attempt+1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("Cannot connect")


def download_file(remote_path, local_path):
    c = connect()
    _, out, _ = c.exec_command(f"cat {remote_path}", timeout=60)
    data = out.read()
    c.close()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(data)
    print(f"  Downloaded: {os.path.basename(local_path)} ({len(data)} bytes)")
    return data


def main():
    experiments = [
        ('r96_slippage', 'r96_results.json'),
        ('r97_correlation', 'r97_results.json'),
        ('r98_ml_threshold', 'r98_results.json'),
        ('r99_time_effects', 'r99_results.json'),
        ('r100_circuit_breaker', 'r100_results.json'),
    ]

    print("Downloading R96-R100 results...")
    for result_dir, json_file in experiments:
        remote = f"{REMOTE_BASE}/{result_dir}/{json_file}"
        local = os.path.join(LOCAL_BASE, result_dir, json_file)
        try:
            download_file(remote, local)
        except Exception as e:
            print(f"  ERROR downloading {result_dir}: {e}")

    # Also download stdout files
    stdout_files = [
        ('r96_slippage', 'r96_stdout.txt'),
        ('r97_correlation', 'r97_stdout.txt'),
        ('r98_ml_threshold', 'r98_stdout.txt'),
        ('r99_time_effects', 'r99_stdout.txt'),
        ('r100_circuit_breaker', 'r100_stdout.txt'),
    ]
    print("\nDownloading stdout logs...")
    for result_dir, stdout_file in stdout_files:
        remote = f"{REMOTE_BASE}/{result_dir}/{stdout_file}"
        local = os.path.join(LOCAL_BASE, result_dir, stdout_file)
        try:
            download_file(remote, local)
        except Exception as e:
            print(f"  ERROR downloading {result_dir} stdout: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
