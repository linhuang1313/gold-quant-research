"""Deploy R33 external data experiments to remote server."""
import paramiko, sys, io, time, os

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

# Create dirs
run(f'mkdir -p {BASE}/data/external')
run(f'mkdir -p {BASE}/results/round33_results')

# Upload external data + experiment script
sftp = ssh.open_sftp()
print("[R33] Uploading external data...")
local_ext = r"c:\Users\hlin2\gold-quant-research\data\external"
for f in ['DXY_daily.csv', 'US10Y_daily.csv', 'SPX_daily.csv', 'GVZ_daily.csv']:
    local_path = os.path.join(local_ext, f)
    if os.path.exists(local_path):
        sftp.put(local_path, f'{BASE}/data/external/{f}')
        print(f"  Uploaded: {f}")

print("[R33] Uploading experiment script...")
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round33_external.py',
         f'{BASE}/experiments/run_round33_external.py')
sftp.close()
print("  Done!")

run('pkill -f run_round33 2>/dev/null')
time.sleep(2)

print("[R33] Launching...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round33_external.py '
    f'> results/round33_results/R33_stdout.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep run_round33 | grep -v grep')
if 'run_round33' in ps:
    print("R33 is RUNNING!")
    print(run(f'head -20 {BASE}/results/round33_results/R33_stdout.txt'))
else:
    print("WARNING: R33 may not have started")
    err = run(f'cat {BASE}/results/round33_results/R33_stdout.txt 2>/dev/null')
    if err: print(err[:2000])

ssh.close()
print("\nDone!")
