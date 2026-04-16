"""Upload and launch 24h marathon on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko, time

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    return out, err

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# 1. git pull (use AutoDL academic proxy, force overwrite untracked files)
print("=== Git pull (with AutoDL proxy) ===")
out, err = run_cmd(ssh, f"cd {REMOTE_DIR} && source /etc/network_turbo && git checkout -- . && git clean -f run_exp_choppy_fine.py run_weekend_batch.py run_24h_marathon.py 2>/dev/null; git pull")
print(out)
if err:
    print(f"stderr: {err}")

# 2. Check and kill existing experiments
print("\n=== Checking existing processes ===")
out, _ = run_cmd(ssh, "ps aux | grep python | grep -E '(run_|marathon)' | grep -v grep")
if out:
    print(f"Found running experiments:\n{out}")
    print("\nKilling old experiments...")
    scripts = ['run_weekend_batch', 'run_exp_choppy_fine', 'run_exp_ema100',
               'run_24h_marathon']
    for script in scripts:
        run_cmd(ssh, f"pkill -f '{script}' 2>/dev/null")
    time.sleep(3)
    out2, _ = run_cmd(ssh, "ps aux | grep python | grep -E '(run_|marathon)' | grep -v grep")
    if out2:
        print(f"Still running:\n{out2}")
        print("Force killing...")
        for script in scripts:
            run_cmd(ssh, f"pkill -9 -f '{script}' 2>/dev/null")
        time.sleep(2)
else:
    print("No existing experiments running")

# 3. Create output directory
print("\n=== Setting up ===")
run_cmd(ssh, f"mkdir -p {REMOTE_DIR}/marathon_results")

# 4. Upload marathon script
print("Uploading run_24h_marathon.py...")
sftp = ssh.open_sftp()
sftp.put("run_24h_marathon.py", f"{REMOTE_DIR}/run_24h_marathon.py")
sftp.close()
print("Upload complete")

# 5. Launch (use exec_command without waiting for output — nohup background)
print("\n=== Launching marathon ===")
cmd = (f"cd {REMOTE_DIR} && nohup {PYTHON} -u run_24h_marathon.py "
       f"> marathon_stdout.txt 2>&1 &")
stdin, stdout, stderr = ssh.exec_command(cmd)
time.sleep(5)

# 6. Verify
out, _ = run_cmd(ssh, f"ps aux | grep run_24h_marathon | grep -v grep")
if out:
    print(f"Marathon launched successfully!")
    print(out)
else:
    print("WARNING: Marathon process not found!")
    out, _ = run_cmd(ssh, f"tail -20 {REMOTE_DIR}/marathon_stdout.txt")
    print(f"Last output:\n{out}")

# 7. Show status
print("\n=== Initial output ===")
time.sleep(5)
out, _ = run_cmd(ssh, f"tail -20 {REMOTE_DIR}/marathon_stdout.txt")
print(out)

ssh.close()
print("\nDone. Monitor with: python _check_marathon.py")
