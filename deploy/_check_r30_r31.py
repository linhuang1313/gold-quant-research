"""Check R30 + R31 status and tail output."""
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

for tag, fname in [("R30", "round30_results/R30_output.txt"),
                    ("R31", "round31_results/R31_output.txt")]:
    full = f'{BASE}/results/{fname}'
    print(f"\n=== {tag} Lines ===")
    print(run(f'wc -l {full} 2>/dev/null'))
    print(f"\n=== {tag} Last 50 lines ===")
    print(run(f'tail -50 {full} 2>/dev/null'))

ssh.close()
