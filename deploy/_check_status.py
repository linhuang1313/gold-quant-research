"""Quick status check for R119-R130."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('connect.westd.seetacloud.com', port=41109, username='root',
          password='3sCdENtzYfse', timeout=120, banner_timeout=60, auth_timeout=60)

# Find all result files
_, out, _ = c.exec_command(
    'find /root/gold-quant-research/results -name "*_results.json" -newer /root/gold-quant-research/experiments/run_r119_ml_entry_v2.py '
    '| sort',
    timeout=30)
print("=== Result JSON files ===")
print(out.read().decode())

# Check running processes
_, out, _ = c.exec_command('ps aux | grep "run_r1[12]" | grep -v grep', timeout=15)
print("=== Running experiments ===")
print(out.read().decode())

c.close()
