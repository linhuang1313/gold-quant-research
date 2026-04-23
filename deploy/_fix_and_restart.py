"""Kill bad run, check data, diagnose nan issue."""
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
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err and 'grep' not in err: print(f'[ERR] {err}')
    return out

print("=== [1] Kill current process ===")
run('pkill -f run_round25 2>/dev/null')
time.sleep(2)

print("\n=== [2] Check M15 data ===")
run(f'wc -l {BASE}/data/download/xauusd-m15-bid*.csv')
run(f'ls -la {BASE}/data/download/xauusd-m15-bid*.csv')
run(f'tail -2 {BASE}/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv')

print("\n=== [3] Check all files in data/download ===")
run(f'ls -la {BASE}/data/download/')

print("\n=== [4] Diagnose nan issue ===")
diag = f'''
cd {BASE} && {PY} -c "
import sys; sys.path.insert(0, '.')
from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
import copy

data = DataBundle.load_default()
print(f'M15: {{len(data.m15_df)}} bars, range: {{data.m15_df.index[0]}} -> {{data.m15_df.index[-1]}}')
print(f'H1: {{len(data.h1_df)}} bars, range: {{data.h1_df.index[0]}} -> {{data.h1_df.index[-1]}}')

kw = dict(**LIVE_PARITY_KWARGS)
kw['time_adaptive_trail'] = {{'start': 2, 'decay': 0.75, 'floor': 0.003}}
kw['min_entry_gap_hours'] = 1.0

s = run_variant(data, 'diag_test', verbose=True, **kw)
print(f'Result: N={{s[\"n\"]}}, sharpe={{s[\"sharpe\"]}}, pnl={{s[\"total_pnl\"]}}, wr={{s[\"win_rate\"]}}')
print(f'Type of sharpe: {{type(s[\"sharpe\"])}}')

trades = s.get('_trades', [])
if trades:
    pnls = [t.pnl for t in trades[:10]]
    print(f'First 10 trade PnLs: {{pnls}}')
" 2>&1
'''
run(diag.strip(), timeout=180)

ssh.close()
print("\nDone!")
