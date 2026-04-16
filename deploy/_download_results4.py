"""Download EXP Batch B/C/D details."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace').strip()

# EXP Batch lines 98-250 (EXP-B, C, D sections), filter out M15 RSI noise
out = run("sed -n '96,350p' /root/gold-quant-trading/exp_batch_postfix_output.txt | grep -v 'M15 RSI'")
print(out)

print("\n\n=== EXP Batch final summary (last 60 lines) ===")
out2 = run("tail -60 /root/gold-quant-trading/exp_batch_postfix_output.txt | grep -v 'M15 RSI'")
print(out2)

ssh.close()
