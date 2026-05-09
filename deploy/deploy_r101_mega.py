"""Deploy R101 mega experiment to remote server (long-running, ~100 hours).
Uses exec_command + base64, launched with nohup for persistence.
Has checkpoint support - will resume if restarted."""
import paramiko
import os
import time
import base64

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = 'run_r101_mega_experiment.py'
RESULT_DIR = 'r101_mega_experiment'
STDOUT_FILE = 'r101_stdout.txt'


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


def upload_via_base64(local_path, remote_path):
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('ascii')
    chunk_size = 50000
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

    c = connect()
    _, out, _ = c.exec_command(f"rm -f {remote_path}.b64", timeout=15)
    out.channel.recv_exit_status()
    c.close()

    for i, chunk in enumerate(chunks):
        c = connect()
        op = ">>" if i > 0 else ">"
        cmd = f"echo '{chunk}' {op} {remote_path}.b64"
        _, out, err = c.exec_command(cmd, timeout=30)
        rc = out.channel.recv_exit_status()
        c.close()
        if rc != 0:
            raise RuntimeError(f"Chunk {i} upload failed")
        time.sleep(0.3)

    c = connect()
    _, out, err = c.exec_command(
        f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64", timeout=30)
    rc = out.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"base64 decode failed: {err.read().decode()}")
    _, out, _ = c.exec_command(f"wc -c {remote_path}", timeout=15)
    remote_size = out.read().decode().strip().split()[0]
    c.close()
    local_size = len(content)
    print(f"  Uploaded: {os.path.basename(local_path)} ({local_size} -> {remote_size} bytes)")


def main():
    print("=" * 60)
    print("Deploying R101 Mega Experiment (~100 hours)")
    print("=" * 60)

    # Step 1: Check for existing R101 process
    print("\n[1/5] Checking for existing R101 process...")
    c = connect()
    _, out, _ = c.exec_command(f"pgrep -f {SCRIPT}", timeout=15)
    existing_pid = out.read().decode().strip()
    c.close()

    if existing_pid:
        print(f"  R101 already running (PID: {existing_pid})")
        print("  Checking checkpoint status...")
        c = connect()
        _, out, _ = c.exec_command(
            f"cat {REMOTE_BASE}/results/{RESULT_DIR}/checkpoint.json 2>/dev/null | python3 -c "
            f"\"import sys,json; d=json.load(sys.stdin); "
            f"print(f'Part A: {{\\\"complete\\\" if d.get(\\\"part_a_complete\\\") else \\\"running\\\"}}'); "
            f"print(f'Part B: {{\\\"complete\\\" if d.get(\\\"part_b_complete\\\") else \\\"running\\\"}}'); "
            f"print(f'Part C: {{\\\"complete\\\" if d.get(\\\"part_c_complete\\\") else \\\"running\\\"}}')\"",
            timeout=15)
        status = out.read().decode().strip()
        c.close()
        if status:
            print(f"  Checkpoint status:\n{status}")
        print("\n  To restart: kill the process first, then rerun this script.")
        print(f"  Kill command: ssh root@{SERVER} -p {PORT} 'kill {existing_pid}'")
        return

    # Step 2: Upload script
    print("\n[2/5] Uploading R101 script...")
    local_file = os.path.join(LOCAL_BASE, 'experiments', SCRIPT)
    remote_file = f"{REMOTE_BASE}/experiments/{SCRIPT}"
    upload_via_base64(local_file, remote_file)

    # Step 3: Create output dirs
    print("\n[3/5] Creating output directories...")
    c = connect()
    _, out, _ = c.exec_command(f"mkdir -p {REMOTE_BASE}/results/{RESULT_DIR}", timeout=15)
    out.channel.recv_exit_status()
    c.close()

    # Step 4: Launch with nohup
    print("\n[4/5] Launching R101 mega experiment...")
    c = connect()
    cmd = (f"cd {REMOTE_BASE} && nohup python3 -u experiments/{SCRIPT} "
           f"> results/{RESULT_DIR}/{STDOUT_FILE} 2>&1 &")
    _, out, _ = c.exec_command(cmd, timeout=30)
    out.channel.recv_exit_status()
    c.close()

    # Step 5: Verify
    time.sleep(5)
    print("\n[5/5] Verifying R101 launch...")
    c = connect()
    _, out, _ = c.exec_command(f"pgrep -f {SCRIPT}", timeout=15)
    pid = out.read().decode().strip()
    c.close()

    if pid:
        print(f"  R101 running with PID: {pid}")
    else:
        print("  WARNING: R101 process not found!")
        print("  Checking error output...")
        c = connect()
        _, out, _ = c.exec_command(
            f"tail -30 {REMOTE_BASE}/results/{RESULT_DIR}/{STDOUT_FILE}", timeout=15)
        err_output = out.read().decode()
        c.close()
        print(err_output)
        return

    # Show initial output
    time.sleep(10)
    c = connect()
    _, out, _ = c.exec_command(
        f"head -30 {REMOTE_BASE}/results/{RESULT_DIR}/{STDOUT_FILE}", timeout=15)
    output = out.read().decode().strip()
    c.close()
    if output:
        print(f"\n--- Initial R101 output ---")
        print(output[:1000])

    print("\n" + "=" * 60)
    print("R101 Mega Experiment deployed successfully!")
    print(f"  PID: {pid}")
    print(f"  Estimated runtime: ~100 hours (4+ days)")
    print(f"  Checkpoint file: results/{RESULT_DIR}/checkpoint.json")
    print(f"  Monitor: ssh root@{SERVER} -p {PORT} 'tail -20 {REMOTE_BASE}/results/{RESULT_DIR}/{STDOUT_FILE}'")
    print("=" * 60)


if __name__ == '__main__':
    main()
