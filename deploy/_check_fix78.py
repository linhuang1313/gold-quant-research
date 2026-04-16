"""Check fix78 progress."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Process
out = run_cmd(ssh, "ps aux | grep fix78 | grep -v grep")
print(f"Process: {out or 'NOT RUNNING'}")

# Output files
for f in ["phase08_montecarlo_FIXED.txt", "phase10_stats_FIXED.txt"]:
    size = run_cmd(ssh, f"wc -c {REMOTE_DIR}/marathon_results/{f} 2>/dev/null | awk '{{print $1}}'")
    print(f"  {f}: {size or '0'} bytes")

# Stdout
out = run_cmd(ssh, f"tail -20 {REMOTE_DIR}/marathon_fix78_stdout.txt")
print(f"\nLast output:\n{out}")

ssh.close()
