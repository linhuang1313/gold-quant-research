"""Deploy R30 to remote server."""
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

# Check for running experiments
ps = run('ps aux | grep "run_round" | grep -v grep')
if ps:
    print("=== Running processes ===")
    print(ps)
    print()

# Upload R30
print("[R30] Uploading experiment script...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round30_tsmom_deep.py',
         f'{BASE}/experiments/run_round30_tsmom_deep.py')
sftp.close()
print("  Done!")

run(f'mkdir -p {BASE}/results/round30_results')
run('pkill -f run_round30 2>/dev/null')
time.sleep(2)

print("[R30] Launching...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round30_tsmom_deep.py '
    f'> results/round30_results/R30_stdout.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep run_round30 | grep -v grep')
if 'run_round30' in ps:
    print("R30 is RUNNING!")
    print(run(f'head -10 {BASE}/results/round30_results/R30_stdout.txt'))
else:
    print("WARNING: R30 may not have started")
    err = run(f'cat {BASE}/results/round30_results/R30_stdout.txt 2>/dev/null')
    if err:
        print(err[:2000])

ssh.close()
print("\nDone!")
