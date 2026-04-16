"""Re-upload updated Round 4 script."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

sftp = ssh.open_sftp()
sftp.put("run_round4.py", f"{REMOTE_DIR}/run_round4.py")
sftp.close()

# Verify chainer is still running
stdin, stdout, stderr = ssh.exec_command("ps aux | grep chain_full | grep -v grep")
out = stdout.read().decode('utf-8', errors='replace').strip()
print(f"Chainer: {out or 'NOT RUNNING'}")

stdin, stdout, stderr = ssh.exec_command("ps aux | grep run_round | grep -v grep")
out = stdout.read().decode('utf-8', errors='replace').strip()
print(f"Running: {out or 'none'}")

ssh.close()
print("Round 4 script updated on server.")
