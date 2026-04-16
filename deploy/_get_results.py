"""Get specific sections from remote output files."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PROJECT = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        return stdout.read().decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f"ERROR: {e}"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# Choppy fine: get Part 1 results (lines with threshold values)
print("=== CHOPPY FINE: Part 1 Results ===")
out = run_cmd(ssh, f"grep -E '^ +0\\.[0-9]' {PROJECT}/exp_choppy_fine_output.txt 2>/dev/null")
print(out)

# Choppy fine: Part 2 K-Fold results
print("\n=== CHOPPY FINE: Part 2 K-Fold ===")
out = run_cmd(ssh, f"grep -E 'Choppy=|Fold[0-9]:|Result:' {PROJECT}/exp_choppy_fine_output.txt 2>/dev/null")
print(out)

# EMA100: all results
print("\n=== EMA100 ABLATION ===")
out = run_cmd(ssh, f"grep -E 'Baseline|NoEMA|Part|EMA100|trades|delta' {PROJECT}/exp_ema100_ablation_output.txt 2>/dev/null")
print(out)

# Weekend batch: all result lines
print("\n=== WEEKEND BATCH ===")
out = run_cmd(ssh, f"grep -E 'TDTP|Variant|Delta|Fold|Result:|elapsed|EXP-|SL_|ADX_|Fixed|Historical' {PROJECT}/exp_weekend_batch_output.txt 2>/dev/null")
print(out)

ssh.close()
