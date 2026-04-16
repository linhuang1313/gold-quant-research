"""Monitor Round 2 experiments progress."""
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

# Process status
out = run_cmd(ssh, "ps aux | grep run_round2 | grep -v grep")
print(f"Process: {out or 'NOT RUNNING'}")

# Master log
out = run_cmd(ssh, f"cat {REMOTE_DIR}/round2_results/00_master_log.txt 2>/dev/null")
if out:
    print(f"\n--- Master Log ---")
    print(out)

# File sizes
print(f"\n--- Result Files ---")
out = run_cmd(ssh, f"ls -la {REMOTE_DIR}/round2_results/ 2>/dev/null")
if out:
    for line in out.split('\n'):
        if '.txt' in line:
            print(f"  {line.split()[-1]}: {line.split()[4]} bytes")

# Current stdout tail
out = run_cmd(ssh, f"tail -30 {REMOTE_DIR}/round2_stdout.txt 2>/dev/null")
if out:
    print(f"\n--- Last Output ---")
    print(out)

# CPU usage
out = run_cmd(ssh, "top -bn1 | head -5")
print(f"\n--- CPU ---")
print(out)

ssh.close()
