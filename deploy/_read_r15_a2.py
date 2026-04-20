"""Read R15 A2 Floor Scan results."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westc.seetacloud.com", port=16005, username="root",
          password="r1zlTZQUb+E4", timeout=30)
_, stdout, _ = c.exec_command("cat /root/gold-quant-trading/results/round15_results/R15_A2_floor_scan.txt", timeout=30)
print(stdout.read().decode('utf-8', errors='replace'))
c.close()
