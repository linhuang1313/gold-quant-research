#!/usr/bin/env python3
"""Deploy R6B on Server B: try multiple proxy/direct methods to git pull."""
import paramiko
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SERVER_B = {"host": "connect.westb.seetacloud.com", "port": 25821, "user": "root", "password": "r1zlTZQUb+E4"}

def run_cmd(client, cmd, timeout=300):
    print(f"\n>>> {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        for line in out.strip().split('\n')[-20:]:
            print(f"  {line}")
    if err.strip():
        for line in err.strip().split('\n')[-10:]:
            print(f"  ERR: {line}")
    return out, err

print("=" * 60)
print("Deploy R6B on Server B (48 cores) - v2")
print("=" * 60)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(SERVER_B["host"], port=SERVER_B["port"], username=SERVER_B["user"], password=SERVER_B["password"], timeout=30)
print("Connected!")

# Step 0: Check which proxy ports are available
print("\n--- Checking proxy availability ---")
run_cmd(client, "curl -s --connect-timeout 3 -x http://127.0.0.1:7890 https://github.com 2>&1 | head -1 || echo 'PROXY 7890 FAILED'")
run_cmd(client, "curl -s --connect-timeout 3 -x http://127.0.0.1:1080 https://github.com 2>&1 | head -1 || echo 'PROXY 1080 FAILED'")
run_cmd(client, "curl -s --connect-timeout 3 -x http://127.0.0.1:8080 https://github.com 2>&1 | head -1 || echo 'PROXY 8080 FAILED'")

# Try direct access
print("\n--- Trying direct git pull ---")
out, err = run_cmd(client, "cd /root/gold-quant-trading && git pull origin main 2>&1")

if "fatal" in out.lower() or "fatal" in err.lower():
    print("\n--- Direct pull failed, trying with GH_PROXY mirror ---")
    # Try setting alternative remote using ghproxy or fastgit
    run_cmd(client, "cd /root/gold-quant-trading && git remote set-url origin https://ghfast.top/https://github.com/linhuang1313/gold-quant-trading.git && git pull origin main 2>&1")
    # Reset remote back
    run_cmd(client, "cd /root/gold-quant-trading && git remote set-url origin https://github.com/linhuang1313/gold-quant-trading.git")

# Check if pull succeeded
out, _ = run_cmd(client, "cd /root/gold-quant-trading && git log --oneline -3")

if "6ddcda9" in out or "abda7ec" in out:
    print("\n*** Git pull successful! ***")
else:
    print("\n--- Still old version, trying gitclone mirror ---")
    run_cmd(client, "cd /root/gold-quant-trading && git remote set-url origin https://gitclone.com/github.com/linhuang1313/gold-quant-trading.git && git pull origin main 2>&1")
    run_cmd(client, "cd /root/gold-quant-trading && git remote set-url origin https://github.com/linhuang1313/gold-quant-trading.git")
    run_cmd(client, "cd /root/gold-quant-trading && git log --oneline -3")

# Check if scripts dir exists now
out, _ = run_cmd(client, "ls /root/gold-quant-trading/scripts/experiments/run_round6b.py 2>/dev/null && echo 'FILE_EXISTS' || echo 'FILE_MISSING'")

if "FILE_EXISTS" in out:
    print("\n--- R6B script found, checking fix ---")
    run_cmd(client, "grep -n 'col_header' /root/gold-quant-trading/scripts/experiments/run_round6b.py")
    
    # Find python path
    out, _ = run_cmd(client, "which python3 || which python || echo '/root/miniconda3/bin/python'")
    python_path = out.strip().split('\n')[-1].strip()
    print(f"\nUsing python: {python_path}")
    
    # Start R6B
    print("\n--- Starting R6B ---")
    run_cmd(client, f"cd /root/gold-quant-trading && nohup {python_path} -u scripts/experiments/run_round6b.py > round6_results/round6b_stdout.txt 2>&1 &")
    
    time.sleep(5)
    run_cmd(client, "ps aux | grep 'round6b' | grep -v grep")
    run_cmd(client, "head -20 /root/gold-quant-trading/round6_results/round6b_stdout.txt 2>/dev/null || echo 'NO OUTPUT YET'")
else:
    print("\n*** FAILED: R6B script not found after git pull attempts ***")
    print("Manual intervention needed: upload the file via scp or fix git access")

client.close()
print("\nDone.")
