#!/usr/bin/env python3
"""Read R7-5 and R7-6 results from Server A."""
import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=42894, username="root", password="r1zlTZQUb+E4", timeout=30)

for f in ["r7_5_tp_interact.txt", "r7_6_recent_zoom.txt"]:
    _, stdout, _ = c.exec_command(f"cat /root/gold-quant-trading/round7_results/{f} 2>/dev/null", timeout=30)
    content = stdout.read().decode('utf-8', errors='replace')
    print(f"\n{'='*60}")
    print(f"  {f}")
    print('='*60)
    print(content)

# Also read R6B-B1 from Server B
c.close()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=25821, username="root", password="r1zlTZQUb+E4", timeout=30)

_, stdout, _ = c.exec_command("cat /root/gold-quant-trading/round6_results/r6_b1_l6_eval.txt 2>/dev/null", timeout=30)
content = stdout.read().decode('utf-8', errors='replace')
print(f"\n{'='*60}")
print(f"  R6B-B1 (Server B)")
print('='*60)
print(content)

c.close()
