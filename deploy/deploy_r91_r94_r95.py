"""Deploy R91, R94, R95 experiments to remote server and run in parallel.
Uses exec_command + base64 to bypass SFTP instability."""
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
    print("Deploying R91 + R94 + R95 (parallel on remote)")
    print("=" * 60)

    experiments = [
        ('run_r91_warsh_regime.py', 'r91_warsh_regime', 'r91_stdout.txt'),
        ('run_r94_exit_optimization.py', 'r94_exit_optimization', 'r94_stdout.txt'),
        ('run_r95_keltner_multipos.py', 'r95_keltner_multipos', 'r95_stdout.txt'),
    ]

    # Step 1: Kill any old processes
    print("\n[1/4] Killing old experiment processes...")
    c = connect()
    for script, _, _ in experiments:
        _, out, _ = c.exec_command(f"pkill -f {script} || true", timeout=15)
        out.channel.recv_exit_status()
    c.close()
    time.sleep(2)

    # Step 2: Upload all scripts
    print("\n[2/4] Uploading experiment scripts...")
    for script, _, _ in experiments:
        local_file = os.path.join(LOCAL_BASE, 'experiments', script)
        remote_file = f"{REMOTE_BASE}/experiments/{script}"
        if os.path.exists(local_file):
            upload_via_base64(local_file, remote_file)
        else:
            print(f"  WARNING: {local_file} not found, skipping")

    # Step 3: Launch all experiments
    print("\n[3/4] Launching experiments...")
    c = connect()
    for script, result_dir, stdout_file in experiments:
        cmds = [
            f"mkdir -p {REMOTE_BASE}/results/{result_dir}",
            (f"cd {REMOTE_BASE} && nohup python3 -u experiments/{script} "
             f"> results/{result_dir}/{stdout_file} 2>&1 &"),
        ]
        for cmd in cmds:
            _, out, err = c.exec_command(cmd, timeout=30)
            out.channel.recv_exit_status()
        print(f"  Launched: {script}")
        time.sleep(1)
    c.close()

    # Step 4: Verify
    time.sleep(10)
    print("\n[4/4] Verifying processes...")
    c = connect()
    _, out, _ = c.exec_command("ps aux | grep -E 'r91|r94|r95' | grep -v grep", timeout=15)
    ps = out.read().decode()
    c.close()

    running = []
    for script, _, _ in experiments:
        name = script.replace('run_', '').replace('.py', '')
        if name in ps:
            running.append(name)
            print(f"  RUNNING: {script}")
        else:
            print(f"  WARNING: {script} not found in process list")

    # Show initial output
    time.sleep(5)
    print("\n--- Initial output ---")
    for _, result_dir, stdout_file in experiments:
        c = connect()
        _, out, _ = c.exec_command(
            f"head -20 {REMOTE_BASE}/results/{result_dir}/{stdout_file} 2>/dev/null", timeout=15)
        output = out.read().decode().strip()
        c.close()
        if output:
            print(f"\n[{result_dir}]:")
            print(output[:500])

    print("\n" + "=" * 60)
    print(f"Deploy complete! {len(running)}/3 experiments running.")
    print("Use download script to fetch results when complete.")


if __name__ == '__main__':
    main()
