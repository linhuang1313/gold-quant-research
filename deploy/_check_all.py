"""Check R30b + R31 status with more detail."""
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

print("=== Processes ===")
print(run('ps aux | grep run_round | grep -v grep'))

print("\n=== R30b Lines ===")
print(run(f'wc -l {BASE}/results/round30_results/R30b_phaseAD_output.txt 2>/dev/null'))

print("\n=== R30b Last 80 lines ===")
print(run(f'tail -80 {BASE}/results/round30_results/R30b_phaseAD_output.txt 2>/dev/null'))

print("\n=== R31 Lines ===")
print(run(f'wc -l {BASE}/results/round31_results/R31_output.txt 2>/dev/null'))

print("\n=== R31 Last 30 lines ===")
print(run(f'tail -30 {BASE}/results/round31_results/R31_output.txt 2>/dev/null'))

ssh.close()
