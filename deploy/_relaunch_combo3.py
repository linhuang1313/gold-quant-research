"""Direct fix + relaunch — skip git pull, use sed directly."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
WORKDIR = '/root/gold-quant-trading'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=30)

def run(cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    stdout.channel.settimeout(timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    return out

# 1. Kill old
print("1. Kill old processes...")
run("pkill -f run_exp_combo 2>/dev/null || true")
time.sleep(2)

# 2. Direct fix via sed (swap reason, bar_time to bar_time, reason)
print("2. Applying fix via sed...")
run(f"sed -i 's/self._close_position(pos, exit_price, reason, bar_time)/self._close_position(pos, exit_price, bar_time, reason)/g' {WORKDIR}/run_exp_combo.py")

# 3. Verify
print("3. Verifying fix...")
out = run(f"grep 'close_position' {WORKDIR}/run_exp_combo.py")
print(f"   {out}")

if "bar_time, reason" not in out:
    print("   ERROR: Fix not applied!")
    ssh.close()
    sys.exit(1)

print("   Fix confirmed!")

# 4. Launch
print("4. Launching...")
stdin, stdout, stderr = ssh.exec_command(
    f"cd {WORKDIR} && nohup {PYTHON} -u run_exp_combo.py > exp_combo_output.txt 2>&1 &\necho PID=$!"
)
stdout.channel.settimeout(10)
pid_out = stdout.read().decode('utf-8', errors='replace').strip()
print(f"   {pid_out}")
time.sleep(8)

# 5. Verify running
out = run("ps aux | grep run_exp_combo | grep python | grep -v grep | head -1")
print(f"5. Running: {'YES' if out else 'NO'}")
if out:
    print(f"   {out}")

# 6. Check output
out = run(f"wc -l {WORKDIR}/exp_combo_output.txt 2>/dev/null")
print(f"6. Output lines: {out}")

out = run(f"tail -3 {WORKDIR}/exp_combo_output.txt 2>/dev/null")
print(f"   Last lines: {out}")

ssh.close()
print("\nDone.")
