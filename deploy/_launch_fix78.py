"""Upload and launch Phase 7/8/10 fix on remote server."""
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

# Upload
print("Uploading run_marathon_fix78.py...")
sftp = ssh.open_sftp()
sftp.put("run_marathon_fix78.py", f"{REMOTE_DIR}/run_marathon_fix78.py")
sftp.close()

# Launch (won't interfere with Phase 11 since they use different output files)
print("Launching fix78...")
cmd = (f"cd {REMOTE_DIR} && nohup {PYTHON} -u run_marathon_fix78.py "
       f"> marathon_fix78_stdout.txt 2>&1 &")
stdin, stdout, stderr = ssh.exec_command(cmd)
time.sleep(5)

# Verify
out, _ = run_cmd(ssh, f"ps aux | grep fix78 | grep -v grep")
if out:
    print(f"Launched OK:\n{out}")
else:
    print("WARNING: process not found!")
    out, _ = run_cmd(ssh, f"tail -10 {REMOTE_DIR}/marathon_fix78_stdout.txt")
    print(out)

# Check marathon Phase 11 still running
out, _ = run_cmd(ssh, f"ps aux | grep run_24h_marathon | grep -v grep")
if out:
    print(f"\nPhase 11 still running (good, no conflict):\n{out}")
else:
    print("\nMarathon already finished")

ssh.close()
print("\nDone. Results will be in marathon_results/phase07_recent_FIXED.txt etc.")
