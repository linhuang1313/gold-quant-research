"""Monitor Round 3 experiments progress."""
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
out = run_cmd(ssh, "ps aux | grep -E 'run_round[23]|chain_r2' | grep -v grep")
print(f"Processes:\n{out or '  (none running)'}")

# Chain log
out = run_cmd(ssh, f"cat {REMOTE_DIR}/chain_log.txt 2>/dev/null")
if out:
    print(f"\n--- Chain Log ---\n{out}")

# Round 2 master log (tail)
out = run_cmd(ssh, f"tail -5 {REMOTE_DIR}/round2_results/00_master_log.txt 2>/dev/null")
if out:
    print(f"\n--- Round 2 Status ---\n{out}")

# Round 3 master log
out = run_cmd(ssh, f"cat {REMOTE_DIR}/round3_results/00_master_log.txt 2>/dev/null")
if out:
    print(f"\n--- Round 3 Master Log ---\n{out}")

# Round 3 result files
print(f"\n--- Round 3 Files ---")
out = run_cmd(ssh, f"ls -la {REMOTE_DIR}/round3_results/ 2>/dev/null")
if out:
    for line in out.split('\n'):
        if '.txt' in line:
            parts = line.split()
            print(f"  {parts[-1]}: {parts[4]} bytes")
else:
    print("  (no files yet)")

# Last output
out = run_cmd(ssh, f"tail -15 {REMOTE_DIR}/round3_stdout.txt 2>/dev/null")
if out:
    print(f"\n--- Round 3 Last Output ---\n{out}")

ssh.close()
