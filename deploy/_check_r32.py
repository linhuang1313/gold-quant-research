"""Check R32 status and install missing deps if needed."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

for attempt in range(3):
    try:
        ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
                    password='5zQ8khQzttDN', timeout=120, banner_timeout=120)
        break
    except Exception as e:
        print(f"  SSH attempt {attempt+1} failed: {e}")
        time.sleep(5)

PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

# Check process
ps = run('ps aux | grep run_round32 | grep -v grep')
if ps:
    print("[R32] RUNNING")
    print(ps[:200])
else:
    print("[R32] NOT running")

# Install missing deps
print("\n[DEPS]")
for pkg in ['xgboost', 'sklearn']:
    chk = run(f'{PY} -c "import {pkg}" 2>&1')
    if 'Error' in chk or 'error' in chk:
        print(f"  {pkg}: MISSING, installing...")
        run(f'{PY} -m pip install {"scikit-learn" if pkg == "sklearn" else pkg} -q', timeout=120)
    else:
        print(f"  {pkg}: OK")

# Tail output
print("\n[TAIL stdout]")
print(run(f'tail -60 {BASE}/results/round32_results/R32_stdout.txt 2>/dev/null'))

print("\n[TAIL output]")
print(run(f'tail -40 {BASE}/results/round32_results/R32_output.txt 2>/dev/null'))

ssh.close()
