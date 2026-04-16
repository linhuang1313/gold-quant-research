"""Check weekend batch experiment progress on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"
OUTPUT = "exp_weekend_batch_output.txt"

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
ps = run_cmd(ssh, "ps aux | grep run_weekend_batch | grep -v grep")
print("=== Process ===")
print(ps if ps else "NOT RUNNING (may have finished)")

# File size
sz = run_cmd(ssh, f"wc -c {PROJECT}/{OUTPUT} 2>/dev/null")
print(f"\n=== File size: {sz} ===")

# Check which EXP sections completed
print("\n=== Completed sections ===")
sections = run_cmd(ssh, f"grep -E '^EXP-|elapsed|ALL EXPERIMENTS' {PROJECT}/{OUTPUT} 2>/dev/null")
print(sections)

# Last 30 lines
tail = run_cmd(ssh, f"tail -30 {PROJECT}/{OUTPUT} 2>/dev/null")
print(f"\n=== Last 30 lines ===")
print(tail)

# Also check choppy/ema100
for name in ["choppy_fine", "ema100_ablation"]:
    ps2 = run_cmd(ssh, f"ps aux | grep run_exp_{name} | grep -v grep | wc -l")
    sz2 = run_cmd(ssh, f"wc -c {PROJECT}/exp_{name}_output.txt 2>/dev/null")
    done = run_cmd(ssh, f"grep -c 'Completed:' {PROJECT}/exp_{name}_output.txt 2>/dev/null")
    status = "DONE" if done and done.strip() != "0" else ("RUNNING" if ps2.strip() != "0" else "UNKNOWN")
    print(f"\n  {name}: {status} ({sz2})")

ssh.close()
