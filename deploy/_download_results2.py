"""Download latest experiment results from remote server."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

files = [
    'exp_u_kc_reentry_output.txt',
    'exp_l_trend_weights_output.txt',
    'exp_w_loss_profile_output.txt',
    'exp_r_baseline_output.txt',
    'exp_s_spread_output.txt',
    'exp_m_slippage_output.txt',
    'exp_batch_postfix_output.txt',
]

for f in files:
    cmd = f'tail -100 /root/gold-quant-trading/{f} 2>/dev/null'
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    if out:
        sep = '=' * 80
        print(f'\n{sep}')
        print(f'>>> {f} (last 100 lines)')
        print(sep)
        print(out)
    else:
        print(f'\n>>> {f}: EMPTY or NOT FOUND')

ssh.close()
