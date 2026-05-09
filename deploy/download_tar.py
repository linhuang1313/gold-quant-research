"""Pack all results into tar on server, download, extract locally.
Each step uses a fresh connection to handle unstable server."""
import paramiko
import os
import time
import tarfile

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_RESULTS = '/root/gold-quant-research/results'
LOCAL_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
REMOTE_TAR = '/tmp/all_results.tar.gz'
LOCAL_TAR = os.path.join(LOCAL_RESULTS, '_all_results.tar.gz')


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
    raise RuntimeError("Cannot connect after 5 attempts")


def step1_create_tar():
    print("[1/3] Creating tar.gz on server...")
    c = connect()
    cmd = (
        "cd /root/gold-quant-research/results && "
        "tar czf /tmp/all_results.tar.gz "
        "--exclude='*.pkl' --exclude='*.csv' ."
    )
    _, stdout, stderr = c.exec_command(cmd, timeout=180)
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        err = stderr.read().decode()
        c.close()
        raise RuntimeError(f"tar failed (rc={rc}): {err}")
    print("  tar created OK")
    c.close()


def step2_check_size():
    print("[1.5] Checking tar size...")
    c = connect()
    _, stdout, _ = c.exec_command("stat -c%s /tmp/all_results.tar.gz", timeout=30)
    size = int(stdout.read().decode().strip())
    print(f"  Remote tar size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    c.close()
    return size


def step3_download(expected_size):
    print("[2/3] Downloading tar.gz...")
    os.makedirs(LOCAL_RESULTS, exist_ok=True)
    c = connect()
    sftp = c.open_sftp()
    start = time.time()
    sftp.get(REMOTE_TAR, LOCAL_TAR)
    elapsed = time.time() - start
    sftp.close()
    c.close()
    local_size = os.path.getsize(LOCAL_TAR)
    print(f"  Downloaded: {local_size:,} bytes in {elapsed:.1f}s ({local_size/1024/1024/elapsed:.1f} MB/s)")
    if local_size != expected_size:
        raise RuntimeError(f"Size mismatch: local={local_size} remote={expected_size}")


def step4_extract():
    print("[3/3] Extracting...")
    with tarfile.open(LOCAL_TAR, 'r:gz') as tf:
        tf.extractall(LOCAL_RESULTS)
    os.remove(LOCAL_TAR)
    count = sum(len(files) for _, _, files in os.walk(LOCAL_RESULTS))
    print(f"  Done! {count} result files in local results/")


def main():
    step1_create_tar()
    time.sleep(2)
    size = step2_check_size()
    time.sleep(2)
    step3_download(size)
    step4_extract()
    print("\nAll results downloaded successfully!")


if __name__ == '__main__':
    main()
