"""Launch R25 on remote server. Minimal SSH interaction to avoid timeout."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

PY = '/root/miniconda3/bin/python'
BASE = '/root/gold-quant-research'

def run(cmd, timeout=30):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    return out, err

# Kill any existing R25
print("Killing old processes...")
run('pkill -f run_round25 2>/dev/null')
time.sleep(2)

# Quick sanity check
print("Checking files...")
out, _ = run(f'ls {BASE}/backtest/stats.py {BASE}/indicators.py {BASE}/research_config.py 2>&1')
print(out)

out, _ = run(f'wc -l {BASE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv')
print(f"M15 lines: {out}")

# Launch
print("\nLaunching R25...")
run(f'cd {BASE} && nohup {PY} -u experiments/run_round25.py > results/round25_results/R25_stdout.txt 2>&1 &')
time.sleep(5)

out, _ = run('ps aux | grep run_round25 | grep -v grep')
if 'run_round25' in out:
    print("R25 is RUNNING!")
else:
    print("WARNING: may not have started.")
    out, _ = run(f'head -20 {BASE}/results/round25_results/R25_stdout.txt 2>&1')
    print(out)

# Show first output
out, _ = run(f'cat {BASE}/results/round25_results/R25_stdout.txt 2>&1')
print(f"\n--- stdout so far ---\n{out}")

ssh.close()
print("\nDone! Monitor with:")
print(f"  ssh -p 45411 root@connect.bjb1.seetacloud.com")
print(f"  tail -f {BASE}/results/round25_results/R25_stdout.txt")
