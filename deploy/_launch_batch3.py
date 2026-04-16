"""Pull and launch 7 parallel experiments (EXP-K through EXP-Q)."""
import paramiko, time, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
W = '/root/gold-quant-trading'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

print("=== Git Pull ===")
stdin, stdout, stderr = ssh.exec_command(
    'cd ' + W + ' && git pull 2>&1', timeout=30)
print(stdout.read().decode('utf-8', errors='replace').strip())

scripts = [
    ('run_exp_k_regime_bounds.py', 'exp_k_regime_bounds_output.txt'),
    ('run_exp_l_trend_weights.py', 'exp_l_trend_weights_output.txt'),
    ('run_exp_m_slippage.py', 'exp_m_slippage_output.txt'),
    ('run_exp_n_statemachine.py', 'exp_n_statemachine_output.txt'),
    ('run_exp_o_partial_tp.py', 'exp_o_partial_tp_output.txt'),
    ('run_exp_p_h4_regime.py', 'exp_p_h4_regime_output.txt'),
    ('run_exp_q_rsi_trailing.py', 'exp_q_rsi_trailing_output.txt'),
]

print("\n=== Launching ===")
for script, output in scripts:
    cmd = 'cd ' + W + ' && nohup ' + PYTHON + ' -u ' + script + ' > ' + output + ' 2>&1 & echo $!'
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=10)
    try:
        pid = stdout.read(256).decode('utf-8', errors='replace').strip()
        print('  ' + script + ': PID=' + pid)
    except Exception:
        print('  ' + script + ': launched (PID read timeout)')
    time.sleep(2)

time.sleep(8)

print("\n=== All experiment processes ===")
stdin, stdout, stderr = ssh.exec_command(
    "ps aux | grep 'python.*run_exp' | grep -v grep", timeout=10)
output = stdout.read().decode('utf-8', errors='replace').strip()
count = 0
for line in output.split('\n'):
    parts = line.split()
    if len(parts) > 10:
        pid = parts[1]
        cpu = parts[2]
        for p in parts[10:]:
            if 'run_' in p:
                print('  PID=' + pid + '  CPU=' + cpu + '%  ' + p)
                count += 1
                break

# Also show the 2 original experiments
stdin2, stdout2, stderr2 = ssh.exec_command(
    "ps aux | grep 'python.*run_trail\\|python.*run_exp_batch_postfix' | grep -v grep", timeout=10)
output2 = stdout2.read().decode('utf-8', errors='replace').strip()
for line in output2.split('\n'):
    parts = line.split()
    if len(parts) > 10:
        pid = parts[1]
        cpu = parts[2]
        for p in parts[10:]:
            if 'run_' in p:
                print('  PID=' + pid + '  CPU=' + cpu + '%  ' + p)
                count += 1
                break

print('\nTotal: ' + str(count) + ' processes')

stdin3, stdout3, _ = ssh.exec_command('uptime', timeout=10)
print('Load: ' + stdout3.read().decode('utf-8', errors='replace').strip())

ssh.close()
print('Done.')
