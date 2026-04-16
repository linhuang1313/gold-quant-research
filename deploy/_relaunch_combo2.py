"""Pull fix and relaunch combo test — step by step with better error handling."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
WORKDIR = '/root/gold-quant-trading'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd, timeout=120):
    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        stdout.channel.settimeout(timeout)
        out = stdout.read().decode('utf-8', errors='replace').strip()
        err = stderr.read().decode('utf-8', errors='replace').strip()
        return out, err
    except Exception as e:
        return "", str(e)

# 1. Kill old
print("1. Kill old combo processes...")
run("pkill -f run_exp_combo || true")
time.sleep(2)

# 2. Git pull with timeout
print("2. Git pull (may take a while)...")
out, err = run(f"cd {WORKDIR} && git pull --rebase=false origin main 2>&1", timeout=120)
print(f"   stdout: {out[:200]}")
if err:
    print(f"   stderr: {err[:200]}")

# 3. If git pull fails, try direct file fix via sed
print("\n3. Verify/fix the file directly...")
out, _ = run(f"grep -n 'close_position' {WORKDIR}/run_exp_combo.py")
print(f"   Current: {out}")

# Check if fix is already there
if "bar_time, reason" in out:
    print("   Fix already applied!")
else:
    print("   Applying fix via sed...")
    run(f"sed -i 's/self._close_position(pos, exit_price, reason, bar_time)/self._close_position(pos, exit_price, bar_time, reason)/' {WORKDIR}/run_exp_combo.py")
    out, _ = run(f"grep -n 'close_position' {WORKDIR}/run_exp_combo.py")
    print(f"   After fix: {out}")

# 4. Launch
print("\n4. Launching combo test...")
run(f"cd {WORKDIR} && nohup {PYTHON} -u run_exp_combo.py > exp_combo_output.txt 2>&1 &")
time.sleep(8)

# 5. Verify
out, _ = run("ps aux | grep combo | grep python | grep -v grep")
print(f"   Running: {out if out else '(NOT running!)'}")

out, _ = run(f"wc -c {WORKDIR}/exp_combo_output.txt 2>/dev/null")
print(f"   Output size: {out}")

out, _ = run(f"tail -5 {WORKDIR}/exp_combo_output.txt 2>/dev/null")
print(f"   Last lines:\n{out}")

ssh.close()
print("\nDone.")
