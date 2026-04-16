"""Check L4 audit progress on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    return stdout.read().decode('utf-8', errors='replace').strip()

size = run('wc -c < /root/gold-quant-trading/exp_l4_audit_output.txt 2>/dev/null')
print(f"Output size: {size} bytes")

print("\n--- Last 30 lines ---")
print(run('tail -30 /root/gold-quant-trading/exp_l4_audit_output.txt 2>/dev/null'))

ps = run('ps aux | grep run_exp_l4_audit | grep -v grep')
status = ps[:150] if ps else "NOT RUNNING"
print(f"\nProcess: {status}")

ssh.close()
