"""Deploy R32 new directions to remote server."""
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

ps = run('ps aux | grep "run_round" | grep -v grep')
if ps:
    print("=== Running processes ===")
    print(ps)
    print()

# Check if xgboost is available
print("[R32] Checking dependencies...")
xgb_check = run(f'{PY} -c "import xgboost; print(xgboost.__version__)" 2>&1')
print(f"  xgboost: {xgb_check}")
sklearn_check = run(f'{PY} -c "import sklearn; print(sklearn.__version__)" 2>&1')
print(f"  sklearn: {sklearn_check}")

# Install xgboost if needed
if 'No module' in xgb_check or 'Error' in xgb_check:
    print("  Installing xgboost...")
    run(f'{PY} -m pip install xgboost -q', timeout=120)

print("\n[R32] Uploading...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round32_new_directions.py',
         f'{BASE}/experiments/run_round32_new_directions.py')
sftp.close()
print("  Done!")

run(f'mkdir -p {BASE}/results/round32_results')
run('pkill -f run_round32 2>/dev/null')
time.sleep(2)

print("[R32] Launching...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round32_new_directions.py '
    f'> results/round32_results/R32_stdout.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep run_round32 | grep -v grep')
if 'run_round32' in ps:
    print("R32 is RUNNING!")
    print(run(f'head -15 {BASE}/results/round32_results/R32_stdout.txt'))
else:
    print("WARNING: R32 may not have started")
    err = run(f'cat {BASE}/results/round32_results/R32_stdout.txt 2>/dev/null')
    if err: print(err[:2000])

ssh.close()
print("\nDone!")
