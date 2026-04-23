"""Check R35 progress on remote server."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=120, banner_timeout=120)

PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

ps = run('ps aux | grep run_round35 | grep -v grep')
print("=== Process ===")
print(ps if ps else "(not running)")

stdout = run(f'cat {BASE}/results/round35_results/R35_stdout.txt 2>/dev/null')
if stdout:
    lines = stdout.split('\n')
    print(f"\n=== R35 stdout ({len(lines)} lines) ===")
    for line in lines[-80:]:
        print(line)
else:
    print("\n(no output yet)")

output = run(f'cat {BASE}/results/round35_results/R35_output.txt 2>/dev/null')
if output:
    lines = output.split('\n')
    print(f"\n=== R35 output ({len(lines)} lines) ===")
    for line in lines[-80:]:
        print(line)

ssh.close()
