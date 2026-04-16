"""Launch 2 experiments + check all processes."""
import paramiko, time, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
W = '/root/gold-quant-trading'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

scripts = [
    ('run_exp_ef_parallel.py', 'exp_ef_parallel_output.txt'),
    ('run_exp_session_analysis.py', 'exp_session_analysis_output.txt'),
]

for script, out in scripts:
    cmd = f'cd {W} && nohup {PYTHON} -u {script} > {out} 2>&1 & echo PID=$!'
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    try:
        result = stdout.read(1024).decode('utf-8', errors='replace').strip()
        print(f"Launched {script}: {result}")
    except Exception as e:
        print(f"Launched {script}: (timeout reading PID, likely started)")
    time.sleep(3)

time.sleep(5)

print("\n=== All running experiments ===")
stdin, stdout, stderr = ssh.exec_command(
    "ps aux | grep 'python.*run_' | grep -v grep",
    timeout=10
)
output = stdout.read().decode('utf-8', errors='replace').strip()
for line in output.split('\n'):
    parts = line.split()
    if len(parts) > 10:
        pid = parts[1]
        cpu = parts[2]
        script_name = ''
        for p in parts[10:]:
            if 'run_' in p:
                script_name = p
                break
        if script_name:
            print(f"  PID={pid}  CPU={cpu}%  {script_name}")

stdin2, stdout2, _ = ssh.exec_command('uptime', timeout=10)
print(f"\nLoad: {stdout2.read().decode('utf-8', errors='replace').strip()}")

ssh.close()
print("Done.")
