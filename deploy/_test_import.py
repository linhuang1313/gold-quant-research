import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

PY = '/root/miniconda3/bin/python'

cmds = [
    f'cd /root/gold-quant-research && {PY} -c "from backtest.runner import DataBundle; d = DataBundle.load_default(); print(f\'OK M15={{len(d.m15_df)}} H1={{len(d.h1_df)}}\')" 2>&1',
    f'wc -l /root/gold-quant-research/data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv',
]

for c in cmds:
    print(f'>>> {c}')
    _, o, e = ssh.exec_command(c, timeout=120)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err: print(f'[ERR] {err}')
    print()

ssh.close()
