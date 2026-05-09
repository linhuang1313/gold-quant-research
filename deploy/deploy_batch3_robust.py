"""Robust deployer for R107 and R109 (large files, unstable connection)."""
import paramiko, os, time, base64

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def connect():
    for attempt in range(8):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
                      timeout=120, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"    connect attempt {attempt+1} failed: {e}, wait {wait}s")
            time.sleep(wait)
    raise RuntimeError("Cannot connect after 8 attempts")


def run_cmd(cmd, timeout=30):
    for attempt in range(3):
        try:
            c = connect()
            _, out, err = c.exec_command(cmd, timeout=timeout)
            rc = out.channel.recv_exit_status()
            stdout = out.read().decode('utf-8', errors='replace')
            stderr = err.read().decode('utf-8', errors='replace')
            c.close()
            return rc, stdout, stderr
        except Exception as e:
            print(f"    cmd attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return -1, "", "all attempts failed"


def upload_file(local_path, remote_path):
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('ascii')

    chunk_size = 20000
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
    print(f"    File: {os.path.basename(local_path)} ({len(content)} bytes, {len(chunks)} chunks)")

    run_cmd(f"rm -f {remote_path}.b64")

    for i, chunk in enumerate(chunks):
        op = ">>" if i > 0 else ">"
        for attempt in range(5):
            try:
                c = connect()
                cmd = f"echo '{chunk}' {op} {remote_path}.b64"
                _, out, _ = c.exec_command(cmd, timeout=30)
                out.channel.recv_exit_status()
                c.close()
                break
            except Exception as e:
                print(f"      chunk {i+1}/{len(chunks)} attempt {attempt+1} failed: {e}")
                time.sleep(3)
        else:
            raise RuntimeError(f"Failed to upload chunk {i}")
        time.sleep(0.5)
        if (i + 1) % 10 == 0:
            print(f"      {i+1}/{len(chunks)} chunks uploaded...")

    rc, stdout, stderr = run_cmd(
        f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64")
    if rc != 0:
        raise RuntimeError(f"base64 decode failed: {stderr}")

    rc, stdout, _ = run_cmd(f"wc -c {remote_path}")
    remote_size = stdout.strip().split()[0] if stdout.strip() else "?"
    print(f"    Uploaded: {os.path.basename(local_path)} -> {remote_size} bytes on server")


def launch(tag, script):
    result_dir = f"{REMOTE_BASE}/results/{tag}"
    run_cmd(f"mkdir -p {result_dir}")
    time.sleep(1)
    run_cmd(
        f"cd {REMOTE_BASE} && nohup python3 -u experiments/{script} "
        f"> {result_dir}/{tag}_stdout.txt 2>&1 &",
        timeout=30)
    time.sleep(5)
    rc, stdout, _ = run_cmd(f"pgrep -f {script}")
    pid = stdout.strip()
    if pid:
        print(f"    {tag} RUNNING (PID: {pid.split()[0]})")
    else:
        print(f"    {tag} WARNING: not running!")
        rc, stdout, _ = run_cmd(f"tail -10 {result_dir}/{tag}_stdout.txt")
        print(f"    Output: {stdout[:300]}")


def main():
    print("=" * 60)
    print("  Deploying R107 + R109 (robust mode)")
    print("=" * 60)

    experiments = [
        ('r107', 'run_r107_ml_entry.py'),
        ('r109', 'run_r109_meta_ensemble.py'),
    ]

    for tag, script in experiments:
        print(f"\n--- {tag} ---")
        local = os.path.join(LOCAL_BASE, 'experiments', script)
        remote = f"{REMOTE_BASE}/experiments/{script}"
        upload_file(local, remote)
        launch(tag, script)
        print(f"    {tag} deployed successfully!")
        time.sleep(5)

    print("\n" + "=" * 60)
    print("  All batch 3 deployed. Checking status...")
    print("=" * 60)
    time.sleep(3)

    for tag, script in [('r103', 'run_r103'), ('r104', 'run_r104'),
                        ('r105', 'run_r105'), ('r106', 'run_r106'),
                        ('r107', 'run_r107'), ('r108', 'run_r108'),
                        ('r109', 'run_r109'), ('r110', 'run_r110')]:
        rc, stdout, _ = run_cmd(f"pgrep -f {script}")
        pid = stdout.strip()
        status = f"RUNNING (PID {pid.split()[0]})" if pid else "STOPPED/DONE"
        print(f"    {tag}: {status}")


if __name__ == '__main__':
    main()
