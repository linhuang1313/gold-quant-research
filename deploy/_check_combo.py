"""Check if combo test is running."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=15)
    stdout.channel.settimeout(15)
    return stdout.read().decode('utf-8', errors='replace').strip()

out = run("ps aux | grep run_exp_combo | grep python | grep -v grep")
print(f"Running: {'YES' if out else 'NO'}")
if out:
    print(out)

out = run("wc -l /root/gold-quant-trading/exp_combo_output.txt 2>/dev/null")
print(f"Output lines: {out}")

out = run("tail -10 /root/gold-quant-trading/exp_combo_output.txt 2>/dev/null")
print(f"Last 10 lines:\n{out}")

ssh.close()
