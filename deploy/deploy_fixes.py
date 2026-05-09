"""Redeploy fixed R107, R109, R110 scripts."""
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
            print(f"    attempt {attempt+1} failed: {e}, wait {wait}s")
            time.sleep(wait)
    raise RuntimeError("Cannot connect")


def run_cmd(cmd, timeout=30):
    for attempt in range(3):
        try:
            c = connect()
            _, out, err = c.exec_command(cmd, timeout=timeout)
            rc = out.channel.recv_exit_status()
            stdout = out.read().decode('utf-8', errors='replace')
            c.close()
            return rc, stdout
        except Exception as e:
            print(f"    cmd attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return -1, ""


def upload_file(local_path, remote_path):
    with open(local_path, 'rb') as f:
        content = f.read()
    b64 = base64.b64encode(content).decode('ascii')
    chunk_size = 20000
    chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

    run_cmd(f"rm -f {remote_path}.b64")

    for i, chunk in enumerate(chunks):
        op = ">>" if i > 0 else ">"
        for attempt in range(5):
            try:
                c = connect()
                c.exec_command(f"echo '{chunk}' {op} {remote_path}.b64", timeout=30)[1].channel.recv_exit_status()
                c.close()
                break
            except Exception as e:
                print(f"      chunk {i+1}/{len(chunks)} attempt {attempt+1} failed")
                time.sleep(3)
        time.sleep(0.5)

    run_cmd(f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64")
    _, out = run_cmd(f"wc -c {remote_path}")
    print(f"    Uploaded: {os.path.basename(local_path)} -> {out.strip()}")


def main():
    scripts = [
        ('r107', 'run_r107_ml_entry.py', 'r107_ml_entry'),
        ('r109', 'run_r109_meta_ensemble.py', 'r109_meta_ensemble'),
        ('r110', 'run_r110_stress_scenarios.py', 'r110_stress_scenarios'),
    ]

    print("=" * 60)
    print("  Redeploying fixed R107, R109, R110")
    print("=" * 60)

    for tag, script, result_dir in scripts:
        print(f"\n--- {tag} ---")

        # Kill old process if running
        run_cmd(f"pkill -f {script}")
        time.sleep(2)

        # Upload
        local = os.path.join(LOCAL_BASE, 'experiments', script)
        remote = f"{REMOTE_BASE}/experiments/{script}"
        upload_file(local, remote)

        # Launch
        run_cmd(f"mkdir -p {REMOTE_BASE}/results/{result_dir}")
        run_cmd(
            f"cd {REMOTE_BASE} && nohup python3 -u experiments/{script} "
            f"> results/{result_dir}/{tag}_stdout.txt 2>&1 &",
            timeout=30)
        time.sleep(5)

        _, pid_out = run_cmd(f"pgrep -f {script}")
        pid = pid_out.strip()
        if pid:
            print(f"    {tag} RUNNING (PID: {pid.split()[0]})")
        else:
            print(f"    {tag} WARNING: not running!")
            _, tail = run_cmd(f"tail -10 {REMOTE_BASE}/results/{result_dir}/{tag}_stdout.txt")
            print(f"    Output: {tail[:300]}")

    print("\n  All fixed scripts redeployed!")


if __name__ == '__main__':
    main()
