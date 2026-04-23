"""Deploy R35-D fix to remote server."""
import paramiko, sys, io, time

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

print("[R35-D] Uploading fix...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round35d_fix.py',
         f'{BASE}/experiments/run_round35d_fix.py')
sftp.close()
print("  Done!")

print("[R35-D] Launching...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round35d_fix.py '
    f'> results/round35_results/R35D_stdout.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep run_round35d | grep -v grep')
if 'run_round35d' in ps:
    print("R35-D is RUNNING!")
    print(run(f'head -20 {BASE}/results/round35_results/R35D_stdout.txt'))
else:
    print("WARNING: R35-D may not have started")
    err = run(f'cat {BASE}/results/round35_results/R35D_stdout.txt 2>/dev/null')
    if err: print(err[:3000])

ssh.close()
print("\nDone!")
