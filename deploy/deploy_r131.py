"""Deploy R131 deep validation to remote server."""
import paramiko, os, time, base64, sys

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER, port=PORT, username=USER, password=PASSWD,
              timeout=30, banner_timeout=30)
    return c

def upload_file(ssh, local_path, remote_path):
    with open(local_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    chunk_size = 40000
    ssh.exec_command(f"rm -f /tmp/_upload.b64")
    time.sleep(0.3)
    for i in range(0, len(b64), chunk_size):
        chunk = b64[i:i+chunk_size]
        for attempt in range(3):
            try:
                ssh.exec_command(f"echo '{chunk}' >> /tmp/_upload.b64")
                time.sleep(0.2)
                break
            except Exception as e:
                if attempt == 2: raise
                time.sleep(1)
                ssh = get_ssh()
    cmd = f"base64 -d /tmp/_upload.b64 > {remote_path} && wc -c {remote_path}"
    _, stdout, _ = ssh.exec_command(cmd)
    print(f"  Uploaded {os.path.basename(local_path)}: {stdout.read().decode().strip()}")
    return ssh

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    ssh = get_ssh()

    if action in ('all', 'upload'):
        print("Uploading R131 script...")
        local = os.path.join(LOCAL_BASE, 'experiments', 'run_r131_deep_validation.py')
        remote = f"{REMOTE_BASE}/experiments/run_r131_deep_validation.py"
        ssh = upload_file(ssh, local, remote)

    if action in ('all', 'launch'):
        print("\nLaunching R131...")
        cmd = (f"cd {REMOTE_BASE} && "
               f"mkdir -p results/r131_deep_validation && "
               f"nohup python -u experiments/run_r131_deep_validation.py "
               f"> results/r131_deep_validation/r131_stdout.txt 2>&1 &")
        ssh.exec_command(cmd)
        time.sleep(2)
        _, stdout, _ = ssh.exec_command("ps aux | grep r131 | grep -v grep")
        ps = stdout.read().decode().strip()
        if ps:
            print(f"  Running: {ps[:120]}")
        else:
            print("  WARNING: Process not found!")

    if action == 'status':
        print("Checking R131 status...")
        _, stdout, _ = ssh.exec_command("ps aux | grep r131 | grep -v grep")
        ps = stdout.read().decode().strip()
        if ps:
            print(f"  Running: {ps[:120]}")
        else:
            print("  Not running")
        _, stdout, _ = ssh.exec_command(f"tail -30 {REMOTE_BASE}/results/r131_deep_validation/r131_stdout.txt 2>/dev/null")
        print(stdout.read().decode())

    if action == 'download':
        print("Downloading R131 results...")
        remote_file = f"{REMOTE_BASE}/results/r131_deep_validation/r131_results.json"
        local_dir = os.path.join(LOCAL_BASE, 'results', 'r131_deep_validation')
        os.makedirs(local_dir, exist_ok=True)
        _, stdout, _ = ssh.exec_command(f"cat {remote_file}")
        data = stdout.read().decode()
        if data.strip():
            with open(os.path.join(local_dir, 'r131_results.json'), 'w') as f:
                f.write(data)
            print(f"  Saved r131_results.json ({len(data)} bytes)")
        else:
            print("  No results file yet")

        _, stdout, _ = ssh.exec_command(f"cat {REMOTE_BASE}/results/r131_deep_validation/r131_stdout.txt")
        data = stdout.read().decode()
        if data.strip():
            with open(os.path.join(local_dir, 'r131_stdout.txt'), 'w') as f:
                f.write(data)
            print(f"  Saved r131_stdout.txt ({len(data)} bytes)")

    ssh.close()

if __name__ == '__main__':
    main()
