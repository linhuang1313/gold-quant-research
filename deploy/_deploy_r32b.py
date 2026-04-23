"""Deploy R32b fix for Phase B3/C + A-KFold to remote server."""
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

print("[R32b] Uploading...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round32b_fix_phaseBC.py',
         f'{BASE}/experiments/run_round32b_fix_phaseBC.py')
sftp.close()
print("  Done!")

run('pkill -f run_round32b 2>/dev/null')
time.sleep(2)

print("[R32b] Launching...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round32b_fix_phaseBC.py '
    f'> results/round32_results/R32b_stdout.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep run_round32b | grep -v grep')
if 'run_round32b' in ps:
    print("R32b is RUNNING!")
    print(run(f'head -20 {BASE}/results/round32_results/R32b_stdout.txt'))
else:
    print("WARNING: R32b may not have started")
    err = run(f'cat {BASE}/results/round32_results/R32b_stdout.txt 2>/dev/null')
    if err: print(err[:2000])

ssh.close()
print("\nDone!")
