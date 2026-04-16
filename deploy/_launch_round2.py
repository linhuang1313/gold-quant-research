"""Upload and launch Round 2 experiments on remote server."""
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

# 1. Kill old python processes
print("Killing old python processes...")
run_cmd(ssh, "pkill -f 'python.*run_' 2>/dev/null; sleep 2")
out, _ = run_cmd(ssh, "ps aux | grep python | grep -v grep | wc -l")
print(f"  Remaining python processes: {out}")

# 2. Git pull with proxy
print("\nGit pull...")
out, err = run_cmd(ssh, f"cd {REMOTE_DIR} && source /etc/network_turbo 2>/dev/null; "
                        f"git checkout -- . && git clean -f run_round2.py 2>/dev/null; "
                        f"git pull", timeout=90)
print(f"  {out}")
if err and 'error' in err.lower():
    print(f"  ERR: {err}")

# 3. Upload script
print("\nUploading run_round2.py...")
sftp = ssh.open_sftp()
sftp.put("run_round2.py", f"{REMOTE_DIR}/run_round2.py")
sftp.close()
print("  Uploaded.")

# 4. Create output dir
run_cmd(ssh, f"mkdir -p {REMOTE_DIR}/round2_results")

# 5. Launch
print("\nLaunching Round 2...")
cmd = (f"cd {REMOTE_DIR} && nohup {PYTHON} -u run_round2.py "
       f"> round2_stdout.txt 2>&1 &")
stdin, stdout, stderr = ssh.exec_command(cmd)
time.sleep(5)

# 6. Verify
out, _ = run_cmd(ssh, "ps aux | grep run_round2 | grep -v grep")
if out:
    print(f"Launched OK:\n  {out}")
else:
    print("WARNING: process not found!")
    out, _ = run_cmd(ssh, f"tail -20 {REMOTE_DIR}/round2_stdout.txt")
    print(f"  stdout: {out}")

ssh.close()
print("\nDone. Monitor with: python _check_round2.py")
