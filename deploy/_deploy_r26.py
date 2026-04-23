"""Deploy and launch R26 on remote server."""
import paramiko, sys, io, time
from scp import SCPClient
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.bjb1.seetacloud.com"
PORT = 45411
USER = "root"
PASSWD = "5zQ8khQzttDN"
PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASSWD, timeout=60, banner_timeout=60)

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err and 'grep' not in err: print(f'[ERR] {err}')
    return out

# Upload R26 script
print("[1] Uploading R26 script...")
scp = SCPClient(ssh.get_transport())
scp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round26.py',
        f'{BASE}/experiments/run_round26.py')
scp.close()
print("  Done!")

# Create output dir
run(f'mkdir -p {BASE}/results/round26_results')

# Kill old processes
print("\n[2] Killing old processes...")
run('pkill -f run_round2 2>/dev/null')
time.sleep(2)

# Launch
print("\n[3] Launching R26...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round26.py '
    f'> results/round26_results/R26_stdout.txt 2>&1 &'
)
time.sleep(5)

out = run('ps aux | grep run_round26 | grep -v grep')
if 'run_round26' in out:
    print("\nR26 is RUNNING!")
else:
    print("\nWARNING: may not have started. Checking log...")
    run(f'head -20 {BASE}/results/round26_results/R26_stdout.txt 2>&1')

ssh.close()
print("\nDone!")
