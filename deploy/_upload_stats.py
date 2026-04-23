import paramiko
from scp import SCPClient

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)
scp = SCPClient(ssh.get_transport())

print("Uploading backtest/stats.py ...")
scp.put(r'c:\Users\hlin2\gold-quant-research\backtest\stats.py',
        '/root/gold-quant-research/backtest/stats.py')
print("Done!")

# Clear pycache
_, o, _ = ssh.exec_command('rm -rf /root/gold-quant-research/backtest/__pycache__')
o.read()

# Quick import test
PY = '/root/miniconda3/bin/python'
print("\nImport test...")
_, o, e = ssh.exec_command(
    f'cd /root/gold-quant-research && {PY} -c "'
    'from backtest.runner import DataBundle; '
    'd = DataBundle.load_default(); '
    "print(f'M15: {len(d.m15_df)} bars, H1: {len(d.h1_df)} bars')\" 2>&1",
    timeout=120)
print(o.read().decode(errors='replace').strip())
err = e.read().decode(errors='replace').strip()
if err: print(f'[ERR] {err}')

scp.close()
ssh.close()
