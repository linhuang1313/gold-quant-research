import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

PY = '/root/miniconda3/bin/python'

def run(cmd, timeout=120):
    print(f'>>> {cmd}')
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err: print(f'[ERR] {err}')
    print()
    return out

print("=" * 60)
print("Server verification")
print("=" * 60)

run('ls -la /root/gold-quant-research/')
run('ls -la /root/gold-quant-research/backtest/')
run('ls -la /root/gold-quant-research/experiments/')
run('ls -la /root/gold-quant-research/data/download/')
run('wc -l /root/gold-quant-research/data/download/*.csv')
run('ls -la /root/gold-quant-research/research_config.py /root/gold-quant-research/indicators.py 2>&1')

print("[Import test]")
run(f'cd /root/gold-quant-research && {PY} -c "from backtest.runner import DataBundle; d = DataBundle.load_default(); print(f\'M15: {{len(d.m15_df)}} bars, H1: {{len(d.h1_df)}} bars\')" 2>&1', timeout=120)

# Check if R25 already running
run('ps aux | grep run_round25 | grep -v grep')

ssh.close()
