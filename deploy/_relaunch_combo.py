"""Pull fix and relaunch combo test on remote server."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PYTHON = '/root/miniconda3/bin/python'
WORKDIR = '/root/gold-quant-trading'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.westc.seetacloud.com', port=30886, username='root',
            password='r1zlTZQUb+E4', timeout=15)

def run(cmd, timeout=60):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

# 1. Kill any remaining combo processes
print("Killing old combo processes...")
out = run(f"pkill -f run_exp_combo || true")
time.sleep(1)

# 2. Git pull
print("Git pull...")
out = run(f"cd {WORKDIR} && git pull 2>&1")
print(out)

# 3. Verify the fix is there
print("\nVerifying fix (line 206)...")
out = run(f"grep -n 'close_position' {WORKDIR}/run_exp_combo.py")
print(out)

# 4. Launch combo test
print("\nLaunching combo test...")
launcher = f"cd {WORKDIR} && nohup {PYTHON} -u run_exp_combo.py > exp_combo_output.txt 2>&1 &\necho $!"
stdin, stdout, stderr = ssh.exec_command(launcher)
pid = stdout.read().decode('utf-8', errors='replace').strip()
print(f"Launched: PID={pid}")

time.sleep(5)

# 5. Verify it's running
print("\nVerifying...")
out = run("ps aux | grep combo | grep -v grep")
print(out if out else "(not running!)")

# 6. Check initial output
out = run(f"head -5 {WORKDIR}/exp_combo_output.txt 2>/dev/null")
print(f"\nInitial output:\n{out}")

ssh.close()
print("\nDone.")
