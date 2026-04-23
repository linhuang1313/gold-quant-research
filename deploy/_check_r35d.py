"""Check R35-D progress."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=120, banner_timeout=120)

BASE = "/root/gold-quant-research"

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

ps = run('ps aux | grep run_round35d | grep -v grep')
print("=== Process ===")
print(ps if ps else "(not running)")

output = run(f'cat {BASE}/results/round35_results/R35D_output.txt 2>/dev/null')
if output:
    lines = output.split('\n')
    print(f"\n=== R35-D output ({len(lines)} lines) ===")
    for line in lines[-80:]:
        print(line)
else:
    stdout = run(f'cat {BASE}/results/round35_results/R35D_stdout.txt 2>/dev/null')
    if stdout:
        lines = stdout.split('\n')
        print(f"\n=== R35-D stdout ({len(lines)} lines) ===")
        for line in lines[-60:]:
            print(line)

ssh.close()
