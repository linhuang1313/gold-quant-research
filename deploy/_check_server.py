"""Check server state: data files, packages, disk space."""
import paramiko

SERVER = 'connect.westd.seetacloud.com'
PORT = 41109
USER = 'root'
PASSWD = '3sCdENtzYfse'
REMOTE_BASE = '/root/gold-quant-research'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVER, port=PORT, username=USER, password=PASSWD, timeout=30, banner_timeout=30)

cmds = [
    ('Data files', f'ls -la {REMOTE_BASE}/data/download/ 2>/dev/null | head -10'),
    ('Macro data', f'ls -la {REMOTE_BASE}/data/external/ 2>/dev/null'),
    ('Packages', 'pip list 2>/dev/null | grep -iE "xgboost|torch|scikit|scipy|hmmlearn"'),
    ('Disk', 'df -h /'),
    ('Experiments', f'ls {REMOTE_BASE}/experiments/run_r13*.py {REMOTE_BASE}/experiments/run_r14*.py 2>/dev/null'),
]

for label, cmd in cmds:
    print(f"\n=== {label} ===")
    _, out, err = c.exec_command(cmd, timeout=15)
    print(out.read().decode().strip())

c.close()
