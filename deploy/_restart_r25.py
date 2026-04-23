"""Restart R25 with current data."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=60, banner_timeout=60)

PY = '/root/miniconda3/bin/python'
BASE = '/root/gold-quant-research'

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    return out

# Kill old
run('pkill -f run_round25 2>/dev/null')
time.sleep(2)

# Clear old output
run(f'rm -f {BASE}/results/round25_results/R25_stdout.txt {BASE}/results/round25_results/R25_output.txt')

# Launch
print("Launching R25...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round25.py '
    f'> results/round25_results/R25_stdout.txt 2>&1 &'
)
time.sleep(5)

out = run('ps aux | grep run_round25 | grep -v grep')
if 'run_round25' in out:
    print("R25 is RUNNING!")
    print(out)
else:
    print("WARNING: not started")
    print(run(f'cat {BASE}/results/round25_results/R25_stdout.txt 2>&1'))

ssh.close()
print("\nDone!")
