"""Pull and launch 3 parallel experiments on remote server."""
import paramiko, time, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
WORKDIR = '/root/gold-quant-trading'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

# 1. Pull
print("=== Git Pull ===")
stdin, stdout, stderr = ssh.exec_command(f'cd {WORKDIR} && git pull 2>&1')
print(stdout.read().decode('utf-8', errors='replace'))

# 2. Launch all 3 experiments
scripts = [
    ('run_exp_choppy_ablation.py', 'exp_choppy_ablation_output.txt'),
    ('run_exp_sl_sensitivity.py', 'exp_sl_sensitivity_output.txt'),
    ('run_exp_spread_model.py', 'exp_spread_model_output.txt'),
]

for script, output in scripts:
    launcher = f"""#!/bin/bash
cd {WORKDIR}
nohup {PYTHON} -u {script} > {output} 2>&1 &
echo $!
"""
    ssh.exec_command(f"cat > /tmp/launch_{script}.sh << 'ENDSCRIPT'\n{launcher}\nENDSCRIPT")
    time.sleep(0.5)
    stdin, stdout, stderr = ssh.exec_command(f'chmod +x /tmp/launch_{script}.sh && bash /tmp/launch_{script}.sh')
    pid = stdout.read().decode('utf-8', errors='replace').strip()
    print(f"Launched {script}: PID={pid}")
    time.sleep(1)

time.sleep(5)

# 3. Verify all
print("\n=== All Python processes ===")
stdin, stdout, stderr = ssh.exec_command('ps aux | grep python | grep -v grep | grep -v jupyter | grep -v tensorboard | grep -v data_server')
print(stdout.read().decode('utf-8', errors='replace'))

# 4. Check load
stdin2, stdout2, stderr2 = ssh.exec_command('uptime')
print("Load:", stdout2.read().decode('utf-8', errors='replace'))

ssh.close()
print("Done.")
