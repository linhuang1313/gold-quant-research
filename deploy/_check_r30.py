"""Check R30 status and tail output."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=120, banner_timeout=120)

BASE = "/root/gold-quant-research"

def run(cmd, timeout=15):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

print("=== Process ===")
print(run('ps aux | grep run_round30 | grep -v grep'))

print("\n=== R30 Output Lines ===")
print(run(f'wc -l {BASE}/results/round30_results/R30_output.txt 2>/dev/null'))

print("\n=== R30 Stdout Lines ===")
print(run(f'wc -l {BASE}/results/round30_results/R30_stdout.txt 2>/dev/null'))

print("\n=== Last 60 lines of output ===")
print(run(f'tail -60 {BASE}/results/round30_results/R30_output.txt 2>/dev/null'))

print("\n=== Last 10 lines of stdout ===")
print(run(f'tail -10 {BASE}/results/round30_results/R30_stdout.txt 2>/dev/null'))

ssh.close()
