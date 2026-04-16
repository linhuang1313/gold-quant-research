"""Download full L4 audit output from remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

sftp = ssh.open_sftp()
sftp.get('/root/gold-quant-trading/exp_l4_audit_output.txt', 'exp_l4_audit_output.txt')
sftp.close()

# Also check if process is still running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep run_exp_l4_audit | grep -v grep', timeout=15)
ps = stdout.read().decode().strip()
if ps:
    print(f"Still running: {ps[:120]}")
else:
    print("Process FINISHED")

stdin, stdout, stderr = ssh.exec_command('wc -c < /root/gold-quant-trading/exp_l4_audit_output.txt', timeout=15)
print(f"File size: {stdout.read().decode().strip()} bytes")

ssh.close()
print("Downloaded to exp_l4_audit_output.txt")
