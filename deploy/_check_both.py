import paramiko, sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=45411, username='root', password='5zQ8khQzttDN', timeout=15)

# Check running processes
_, o, _ = ssh.exec_command('ps aux | grep -E "round37b|sl_trail" | grep -v grep')
ps = o.read().decode().strip()
print("=== RUNNING PROCESSES ===")
print(ps if ps else "None running.")

# R37B status
_, o, _ = ssh.exec_command('wc -l /root/gold-quant-research/R37B_output.txt 2>/dev/null')
print(f"\nR37B output: {o.read().decode().strip()}")
_, o, _ = ssh.exec_command('tail -30 /root/gold-quant-research/R37B_output.txt 2>/dev/null')
data = o.read().decode('utf-8', errors='replace').strip()
# Remove emoji chars for Windows console
data = data.replace('\u26a0', '[!]').replace('\u2705', '[OK]')
print(f"R37B last 30 lines:\n{data}")

# R30 status
_, o, _ = ssh.exec_command('wc -l /root/gold-quant-research/results/round30_results/R30_sl_trail_balance.txt 2>/dev/null')
print(f"\nR30 output: {o.read().decode().strip()}")
_, o, _ = ssh.exec_command('tail -30 /root/gold-quant-research/results/round30_results/R30_sl_trail_balance.txt 2>/dev/null')
data = o.read().decode('utf-8', errors='replace').strip()
for ch in ['\u26a0', '\u2705', '\ufe0f', '\u2714', '\u274c', '\u2757']:
    data = data.replace(ch, '')
print(f"R30 last 30 lines:\n{data}")

# Check stderr if no output
_, o, _ = ssh.exec_command('tail -10 /root/gold-quant-research/results/round30_results/R30_sl_stdout.txt 2>/dev/null')
stderr_data = o.read().decode('utf-8', errors='replace').strip()
if stderr_data:
    for ch in ['\u26a0', '\u2705', '\ufe0f', '\u2714', '\u274c', '\u2757']:
        stderr_data = stderr_data.replace(ch, '')
    print(f"\nR30 stdout/stderr:\n{stderr_data}")

ssh.close()
