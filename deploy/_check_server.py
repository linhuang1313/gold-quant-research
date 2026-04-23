import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=30)

cmds = [
    'which conda 2>&1',
    'ls /root/miniconda3/bin/python* 2>&1',
    'ls /usr/bin/python* /usr/local/bin/python* 2>&1',
    '/root/miniconda3/bin/python --version 2>&1',
    '/root/miniconda3/bin/pip list 2>/dev/null | grep -iE "numpy|pandas" 2>&1',
    'cat /etc/os-release | head -3 2>&1',
    'ls -la /root/research/*.csv 2>&1',
    'wc -l /root/research/*.csv 2>&1',
]

for c in cmds:
    print(f'>>> {c}')
    _, o, e = ssh.exec_command(c, timeout=30)
    out = o.read().decode(errors='replace').strip()
    err = e.read().decode(errors='replace').strip()
    if out: print(out)
    if err: print(f'[ERR] {err}')
    print()

ssh.close()
