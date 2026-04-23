"""Deploy and launch R28 on remote server."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=60, banner_timeout=60)

PY = "/root/miniconda3/bin/python"
BASE = "/root/gold-quant-research"

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

print("[1] Uploading R28 script...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round28_l7mh8_kfold.py',
         f'{BASE}/experiments/run_round28_l7mh8_kfold.py')
sftp.close()
print("  Done!")

run(f'mkdir -p {BASE}/results/round28_results')
run('pkill -f run_round28 2>/dev/null')
time.sleep(2)

print("[2] Launching R28...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round28_l7mh8_kfold.py '
    f'> results/round28_results/R28_stdout.txt 2>&1 &'
)
time.sleep(5)

out = run('ps aux | grep run_round28 | grep -v grep')
if 'run_round28' in out:
    print("R28 is RUNNING!")
    print(out)
else:
    print("WARNING: checking log...")
    print(run(f'head -30 {BASE}/results/round28_results/R28_stdout.txt'))

ssh.close()
print("\nDone!")
