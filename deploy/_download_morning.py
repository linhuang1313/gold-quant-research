"""Download all overnight results."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    stdout.channel.settimeout(timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

# 1. Check what's running
print("=== Running processes ===")
out = run("ps aux | grep python | grep -E 'run_exp|overnight' | grep -v grep")
print(out if out else "(none)")

# 2. Combo results (full)
print("\n" + "=" * 80)
print(">>> COMBO TEST RESULTS")
print("=" * 80)
out = run("cat /root/gold-quant-trading/exp_combo_output.txt | grep -v 'M15 RSI'", timeout=30)
print(out)

# 3. Overnight batch
print("\n" + "=" * 80)
print(">>> OVERNIGHT BATCH STATUS")
print("=" * 80)
out = run("wc -l /root/gold-quant-trading/overnight_batch_output.txt 2>/dev/null")
print(f"Lines: {out}")
out = run("tail -100 /root/gold-quant-trading/overnight_batch_output.txt 2>/dev/null | grep -v 'M15 RSI'", timeout=20)
print(out if out else "(empty or not started)")

# 4. EXP-K
print("\n" + "=" * 80)
print(">>> EXP-K REGIME BOUNDS STATUS")
print("=" * 80)
out = run("wc -l /root/gold-quant-trading/exp_k_regime_output.txt 2>/dev/null")
print(f"Lines: {out}")
out = run("tail -60 /root/gold-quant-trading/exp_k_regime_output.txt 2>/dev/null", timeout=15)
print(out if out else "(empty)")

ssh.close()
