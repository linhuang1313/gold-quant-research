"""Download R25 results from server."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root',
            password='5zQ8khQzttDN', timeout=60, banner_timeout=60)

BASE = '/root/gold-quant-research'

_, o, _ = ssh.exec_command(f'cat {BASE}/results/round25_results/R25_output.txt 2>&1', timeout=60)
result = o.read().decode(errors='replace')
print(result)

ssh.close()
