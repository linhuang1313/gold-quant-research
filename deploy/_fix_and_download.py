"""Pull fix, relaunch EXP-K, download completed results."""
import paramiko, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
W = '/root/gold-quant-trading'
LOCAL = os.path.dirname(os.path.abspath(__file__))

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

# 1. Pull fix
print("=== Git Pull ===")
stdin, stdout, stderr = ssh.exec_command('cd ' + W + ' && git pull 2>&1', timeout=30)
print(stdout.read().decode('utf-8', errors='replace').strip())

# 2. Relaunch EXP-K
print("\n=== Relaunch EXP-K ===")
cmd = 'cd ' + W + ' && nohup ' + PYTHON + ' -u run_exp_k_regime_bounds.py > exp_k_regime_bounds_output.txt 2>&1 & echo $!'
stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
import time
try:
    pid = stdout.read(256).decode('utf-8', errors='replace').strip()
    print('  Launched: PID=' + pid)
except:
    print('  Launched (PID read timeout)')

# 3. Download completed results
sftp = ssh.open_sftp()
completed_files = [
    'exp_ef_parallel_output.txt',
    'exp_n_statemachine_output.txt',
    'exp_o_partial_tp_output.txt',
    'exp_p_h4_regime_output.txt',
    'exp_q_rsi_trailing_output.txt',
    'exp_session_analysis_output.txt',
]

print("\n=== Downloading completed results ===")
for f in completed_files:
    remote_path = W + '/' + f
    local_path = os.path.join(LOCAL, f)
    try:
        sftp.get(remote_path, local_path)
        size = os.path.getsize(local_path)
        print('  ' + f + ': ' + str(size) + ' bytes')
    except Exception as e:
        print('  ' + f + ': ERROR - ' + str(e))

sftp.close()
ssh.close()
print("\nDone.")
