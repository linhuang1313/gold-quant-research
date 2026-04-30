"""Detailed R50 status check."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westd.seetacloud.com', port=41109, username='root', password='3sCdENtzYfse', timeout=30)

def ssh(cmd):
    _, out, _ = c.exec_command(cmd, timeout=30)
    return out.read().decode('utf-8', errors='replace').strip()

print("=== Process ===")
print(ssh("ps aux | grep run_round50 | grep python | grep -v grep | wc -l"))

print("\n=== stdout lines ===")
print(ssh("wc -l /root/gold-quant-research/results/round50_results/stdout.txt"))

print("\n=== Key lines (Base/Done/Error/Layer) ===")
print(ssh("grep -E 'Base [0-9]+/|Done in|FATAL|Traceback|Layer [0-9]|checkpoint' /root/gold-quant-research/results/round50_results/stdout.txt | tail -30"))

print("\n=== Last 5 lines ===")
print(ssh("tail -5 /root/gold-quant-research/results/round50_results/stdout.txt"))

print("\n=== Result files ===")
print(ssh("ls -la /root/gold-quant-research/results/round50_results/"))

c.close()
