"""Deploy and launch R27 on remote server."""
import paramiko, sys, io, time
from scp import SCPClient
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=60, banner_timeout=60)

PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

print("[1] Uploading R27 script...")
scp = SCPClient(ssh.get_transport())
scp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round27.py',
        f'{BASE}/experiments/run_round27.py')
scp.close()
print("  Done!")

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

run(f'mkdir -p {BASE}/results/round27_results')
run('pkill -f run_round2 2>/dev/null')
time.sleep(2)

print("[2] Launching R27...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round27.py '
    f'> results/round27_results/R27_stdout.txt 2>&1 &'
)
time.sleep(5)

out = run('ps aux | grep run_round27 | grep -v grep')
if 'run_round27' in out:
    print("R27 is RUNNING!")
    print(out)
else:
    print("WARNING: may not have started")
    print(run(f'head -20 {BASE}/results/round27_results/R27_stdout.txt'))

ssh.close()
print("\nDone!")
