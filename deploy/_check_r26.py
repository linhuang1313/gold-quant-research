"""Check R26 results."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=60, banner_timeout=60)

BASE = '/root/gold-quant-research'

# Check process
_, o, _ = ssh.exec_command('ps aux | grep run_round26 | grep -v grep', timeout=10)
proc = o.read().decode(errors='replace').strip()

# Get output
_, o, _ = ssh.exec_command(f'cat {BASE}/results/round26_results/R26_output.txt 2>&1', timeout=30)
output = o.read().decode(errors='replace')

if output.strip():
    print(output)
else:
    print("[No output file yet, checking stdout...]")
    _, o, _ = ssh.exec_command(f'tail -50 {BASE}/results/round26_results/R26_stdout.txt 2>&1', timeout=30)
    print(o.read().decode(errors='replace'))

if proc:
    print(f"\n[STILL RUNNING]\n{proc}")
else:
    print("\n[FINISHED]")

ssh.close()
