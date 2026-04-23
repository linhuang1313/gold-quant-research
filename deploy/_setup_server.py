import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

PY = '/root/miniconda3/bin/python'
PIP = '/root/miniconda3/bin/pip'

def run(cmd, timeout=120):
    print(f'>>> {cmd}')
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err: print(f'[ERR] {err}')
    print()
    return out

# 1. Install pandas
print("=" * 60)
print("[1] Installing pandas...")
run(f'{PIP} install pandas --quiet 2>&1 | tail -5')
run(f'{PIP} list 2>/dev/null | grep -iE "numpy|pandas"')

# 2. Setup directory structure
print("[2] Setting up directory structure...")
run('mkdir -p /root/gold-quant-research/data/download /root/gold-quant-research/results/round25_results')

# 3. Move/link files from research/ to gold-quant-research/
print("[3] Linking files...")
run('cp /root/research/engine.py /root/gold-quant-research/backtest/engine.py')
run('cp /root/research/runner.py /root/gold-quant-research/backtest/runner.py')
run('cp /root/research/__init__.py /root/gold-quant-research/backtest/__init__.py')
run('cp /root/research/run_round25.py /root/gold-quant-research/experiments/run_round25.py')
run('cp /root/research/*.csv /root/gold-quant-research/data/download/')

# 4. Check data completeness
print("[4] Checking data...")
run('wc -l /root/gold-quant-research/data/download/*.csv')
run('head -2 /root/gold-quant-research/data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv')
run('tail -2 /root/gold-quant-research/data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv')

# 5. Check runner.py for data file paths
print("[5] Checking data file references in runner.py...")
run('grep -n "csv" /root/gold-quant-research/backtest/runner.py | head -10')

# 6. Quick test import
print("[6] Quick import test...")
run(f'cd /root/gold-quant-research && {PY} -c "from backtest.runner import DataBundle; print(DataBundle)" 2>&1')

print("\n" + "=" * 60)
print("Server setup complete!")
print("=" * 60)

ssh.close()
