"""Check R25 progress on remote server."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

BASE = '/root/gold-quant-research'

_, o, _ = ssh.exec_command(f'cat {BASE}/results/round25_results/R25_stdout.txt 2>&1', timeout=30)
print(o.read().decode(errors='replace'))

_, o, _ = ssh.exec_command('ps aux | grep run_round25 | grep -v grep', timeout=10)
out = o.read().decode(errors='replace').strip()
if out:
    print(f"\n[PROCESS RUNNING]\n{out}")
else:
    print("\n[PROCESS FINISHED]")
    _, o, _ = ssh.exec_command(f'cat {BASE}/results/round25_results/R25_output.txt 2>&1 | wc -l', timeout=10)
    print(f"Output file lines: {o.read().decode().strip()}")

ssh.close()
