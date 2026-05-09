#!/usr/bin/env python3
"""Deploy R144-R146 experiments to remote server and launch."""
import paramiko, os, sys, time

HOST = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASS = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENTS = [
    ('r144', 'run_r144_extreme_deep.py'),
    ('r145', 'run_r145_xgb_refinement.py'),
    ('r146', 'run_r146_s3s4_validation.py'),
]


def connect(retries=5):
    for attempt in range(retries):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30, banner_timeout=30)
            return ssh
        except Exception as e:
            print(f"    Connect attempt {attempt+1}/{retries} failed: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError("Failed to connect after retries")


def upload_files():
    print("\n=== Uploading experiment scripts ===")

    # Create result dirs first
    ssh = connect()
    for tag, _ in EXPERIMENTS:
        result_dir = f"{REMOTE_BASE}/results/{tag}"
        ssh.exec_command(f"mkdir -p {result_dir}", timeout=10)
    ssh.close()
    time.sleep(1)

    for tag, script in EXPERIMENTS:
        local = os.path.join(LOCAL_BASE, 'experiments', script)
        remote = f"{REMOTE_BASE}/experiments/{script}"
        print(f"  Uploading {script}...", end=" ", flush=True)
        for attempt in range(3):
            try:
                ssh = connect()
                sftp = ssh.open_sftp()
                sftp.put(local, remote)
                sftp.close()
                ssh.close()
                print("OK")
                break
            except Exception as e:
                print(f"\n    Attempt {attempt+1} failed: {e}")
                time.sleep(5)
                if attempt == 2:
                    print(f"    FAILED to upload {script}")
        time.sleep(2)

    print("  Upload complete!")


def launch_all():
    print("\n=== Launching experiments ===")
    for tag, script in EXPERIMENTS:
        print(f"\n  Launching {tag} ({script})...", flush=True)
        ssh = connect()
        stdout_path = f"{REMOTE_BASE}/results/{tag}/{tag}_stdout.txt"
        cmd = (f"bash -c 'cd {REMOTE_BASE} && "
               f"nohup python3 -u experiments/{script} "
               f"> {stdout_path} 2>&1 &'")
        ssh.exec_command(cmd, timeout=5)
        time.sleep(3)
        ssh.close()
        print(f"    Sent launch command for {tag}")
        time.sleep(3)

    time.sleep(5)
    print("\n  Verifying launches...", flush=True)
    ssh = connect()
    for tag, script in EXPERIMENTS:
        _, out, _ = ssh.exec_command(f"ps aux | grep {script} | grep -v grep", timeout=10)
        result = out.read().decode().strip()
        if result:
            print(f"    {tag}: RUNNING")
        else:
            print(f"    {tag}: NOT FOUND - may need manual start")
    ssh.close()
    print("\n  Launch complete!")


def check_status():
    print("\n=== Checking status ===")
    ssh = connect()
    for tag, script in EXPERIMENTS:
        print(f"\n  {tag}:")
        _, out, _ = ssh.exec_command(f"ps aux | grep {script} | grep -v grep | wc -l", timeout=10)
        running = out.read().decode().strip()
        print(f"    Running: {'YES' if running != '0' else 'NO'}")

        stdout_path = f"{REMOTE_BASE}/results/{tag}/{tag}_stdout.txt"
        _, out, _ = ssh.exec_command(f"wc -l {stdout_path} 2>/dev/null || echo '0 lines'", timeout=10)
        print(f"    Stdout: {out.read().decode().strip()}")

        result_path = f"{REMOTE_BASE}/results/{tag}/{tag}_results.json"
        _, out, _ = ssh.exec_command(f"ls -la {result_path} 2>/dev/null || echo 'NOT YET'", timeout=10)
        print(f"    Results: {out.read().decode().strip()}")
    ssh.close()


def download_results():
    print("\n=== Downloading results ===")
    ssh = connect()
    sftp = ssh.open_sftp()
    for tag, _ in EXPERIMENTS:
        local_dir = os.path.join(LOCAL_BASE, 'results', tag)
        os.makedirs(local_dir, exist_ok=True)
        remote_dir = f"{REMOTE_BASE}/results/{tag}"
        try:
            files = sftp.listdir(remote_dir)
            for f in files:
                remote_path = f"{remote_dir}/{f}"
                local_path = os.path.join(local_dir, f)
                print(f"  Downloading {tag}/{f}...", end=" ")
                sftp.get(remote_path, local_path)
                print("OK")
        except Exception as e:
            print(f"  {tag}: {e}")
    sftp.close()
    ssh.close()


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if action == 'all':
        upload_files()
        launch_all()
    elif action == 'upload':
        upload_files()
    elif action == 'launch':
        launch_all()
    elif action == 'status':
        check_status()
    elif action == 'download':
        download_results()
    else:
        print(f"Usage: {sys.argv[0]} [all|upload|launch|status|download]")
