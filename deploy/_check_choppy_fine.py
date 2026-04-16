"""Check choppy fine sweep progress on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"
OUTPUT = "exp_choppy_fine_output.txt"

def run_cmd(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"ERROR: {e}"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Process alive?
ps = run_cmd(ssh, "ps aux | grep run_exp_choppy_fine | grep -v grep")
print("=== Process ===")
print(ps if ps else "NOT RUNNING")

# File size
sz = run_cmd(ssh, f"wc -c {PROJECT}/{OUTPUT} 2>/dev/null")
print(f"\n=== File size: {sz} ===")

# Last 40 lines
tail = run_cmd(ssh, f"tail -40 {PROJECT}/{OUTPUT} 2>/dev/null")
print(f"\n=== Last 40 lines ===")
print(tail)

ssh.close()
