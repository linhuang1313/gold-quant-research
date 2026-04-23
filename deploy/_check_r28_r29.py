"""Check R28 and R29 status."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=120, banner_timeout=120)

def run(cmd, timeout=15):
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors='replace').strip()

print("=== Processes ===")
print(run('ps aux | grep run_round | grep -v grep'))

print("\n=== R28 Output ===")
print(run('wc -l /root/gold-quant-research/results/round28_results/R28_L7MH8_kfold.txt 2>/dev/null'))
print(run('tail -40 /root/gold-quant-research/results/round28_results/R28_L7MH8_kfold.txt 2>/dev/null'))

print("\n=== R29 Output ===")
print(run('wc -l /root/gold-quant-research/results/round29_results/R29_output.txt 2>/dev/null'))
print(run('tail -20 /root/gold-quant-research/results/round29_results/R29_output.txt 2>/dev/null'))

ssh.close()
