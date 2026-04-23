"""Check R28 status, fetch results if done, then deploy R29."""
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

# Check R28
ps = run('ps aux | grep run_round28 | grep -v grep')
if 'run_round28' in ps:
    print("[R28] Still RUNNING. Waiting...")
    for attempt in range(30):
        time.sleep(60)
        ps = run('ps aux | grep run_round28 | grep -v grep')
        out_lines = run(f'wc -l {BASE}/results/round28_results/R28_L7MH8_kfold.txt 2>/dev/null')
        print(f"  Check {attempt+1}: {'RUNNING' if 'run_round28' in ps else 'DONE'}, lines: {out_lines}")
        if 'run_round28' not in ps:
            break
    else:
        print("  R28 still running after 30 min wait. Deploying R29 anyway.")

# Fetch R28 results
print("\n[R28] Fetching results...")
r28_out = run(f'cat {BASE}/results/round28_results/R28_L7MH8_kfold.txt 2>/dev/null', timeout=15)
if r28_out:
    print("--- R28 Results ---")
    print(r28_out[-3000:])  # last 3000 chars
    print("--- End R28 ---")
else:
    print("  No R28 output yet")

# Deploy R29
print("\n[R29] Uploading...")
sftp = ssh.open_sftp()
sftp.put(r'c:\Users\hlin2\gold-quant-research\experiments\run_round29_new_factors.py',
         f'{BASE}/experiments/run_round29_new_factors.py')
sftp.close()
print("  Done!")

run(f'mkdir -p {BASE}/results/round29_results')
run('pkill -f run_round29 2>/dev/null')
time.sleep(2)

print("[R29] Launching...")
chan = ssh.get_transport().open_session()
chan.exec_command(
    f'cd {BASE} && nohup {PY} -u experiments/run_round29_new_factors.py '
    f'> results/round29_results/R29_stdout.txt 2>&1 &'
)
time.sleep(5)

ps = run('ps aux | grep run_round29 | grep -v grep')
if 'run_round29' in ps:
    print("R29 is RUNNING!")
else:
    print("WARNING: R29 may not have started")
    print(run(f'head -20 {BASE}/results/round29_results/R29_stdout.txt'))

ssh.close()
print("\nDone!")
