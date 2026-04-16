"""Check Round 5 experiment progress on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 25821
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# 1. Process status
print("=== Process Status ===")
out = run_cmd(ssh, "ps aux | grep run_round5 | grep -v grep")
print(out if out else "  NOT RUNNING")

# 2. Master log
print("\n=== Master Log ===")
out = run_cmd(ssh, f"cat {REMOTE_DIR}/round5_results/00_master_log.txt 2>/dev/null || echo 'No master log yet'")
print(out)

# 3. Result files
print("\n=== Result Files ===")
out = run_cmd(ssh, f"ls -lhS {REMOTE_DIR}/round5_results/ 2>/dev/null")
print(out if out else "  No results yet")

# 4. Latest stdout
print("\n=== Latest stdout (last 30 lines) ===")
out = run_cmd(ssh, f"tail -30 {REMOTE_DIR}/round5_stdout.txt 2>/dev/null || echo 'No stdout yet'")
print(out)

# 5. Errors
print("\n=== Error Check ===")
out = run_cmd(ssh, f"grep -c 'FAILED\\|Traceback\\|Error' {REMOTE_DIR}/round5_stdout.txt 2>/dev/null || echo '0'")
errors = int(out) if out.isdigit() else 0
print(f"  Error count: {errors}")
if errors > 0:
    out = run_cmd(ssh, f"grep -A2 'FAILED\\|Traceback' {REMOTE_DIR}/round5_stdout.txt 2>/dev/null | tail -20")
    print(out)

# 6. CPU/Memory
print("\n=== Server Load ===")
out = run_cmd(ssh, "uptime && echo '---' && free -h | head -2")
print(out)

ssh.close()
