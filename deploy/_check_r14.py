"""Quick check R14 progress."""
import paramiko
import time

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def ssh_exec(ssh, cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    return out.strip()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

print("=== R14 Progress Check ===\n")

ps = ssh_exec(ssh, "ps aux | grep run_round14 | grep -v grep | wc -l")
print(f"Active processes: {ps}")

files = ssh_exec(ssh, f"ls -la {REMOTE_DIR}/round14_results/*.txt 2>/dev/null")
if files:
    print(f"\nResult files:")
    for line in files.split('\n'):
        print(f"  {line}")

log_lines = ssh_exec(ssh, f"wc -l {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null")
print(f"\nLog size: {log_lines}")

print(f"\nLast 50 lines of log:")
tail = ssh_exec(ssh, f"tail -50 {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null")
print(tail)

phases = ssh_exec(ssh, f"grep -E '(>>>|<<<|Phase|FAILED)' {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null")
if phases:
    print(f"\n=== Phase Progress ===")
    for line in phases.split('\n'):
        print(f"  {line}")

ssh.close()
