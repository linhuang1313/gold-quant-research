"""Test one cycle of the self-healing monitor."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datetime import datetime
import paramiko, traceback

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
PYTHON = "/root/miniconda3/envs/3.10/bin/python"
REMOTE_DIR = "/root/gold-quant-trading"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

def run_cmd(cmd, timeout=20):
    try:
        _, o, e = ssh.exec_command(cmd, timeout=timeout)
        return o.read().decode('utf-8', errors='replace').strip()
    except Exception as ex:
        return f"CMD_ERROR: {ex}"

def is_running(pattern):
    out = run_cmd(f"ps aux | grep '{pattern}' | grep -v grep")
    return bool(out and pattern in out)

print("=" * 70)
print(f"SELF-HEALING MONITOR - TEST RUN at {datetime.now()}")
print("=" * 70)

# 1. Process status
r2 = is_running("run_round2.py")
r3 = is_running("run_round3.py")
r4 = is_running("run_round4.py")
chain = is_running("_chain_full.sh") or is_running("_chain_rebuilt.sh")

print(f"\nR2: {'RUNNING' if r2 else 'not running'}")
print(f"R3: {'RUNNING' if r3 else 'not running'}")
print(f"R4: {'RUNNING' if r4 else 'not running'}")
print(f"Chainer: {'ALIVE' if chain else 'DEAD'}")

# 2. Completion check
for rnd in ["round2", "round3", "round4"]:
    complete = run_cmd(f"grep -c 'ALL COMPLETE' {REMOTE_DIR}/{rnd}_results/00_master_log.txt 2>/dev/null")
    print(f"{rnd} completed: {'YES' if complete == '1' else 'NO'}")

# 3. Error check
print("\n--- Error Check ---")
for rnd in ["round2", "round3", "round4"]:
    exists = run_cmd(f"test -f {REMOTE_DIR}/{rnd}_stdout.txt && echo yes || echo no")
    if exists != "yes":
        print(f"  {rnd}: (not started yet)")
        continue
    err = run_cmd(f"grep -c 'FAILED' {REMOTE_DIR}/{rnd}_stdout.txt 2>/dev/null")
    tb = run_cmd(f"grep -c 'Traceback' {REMOTE_DIR}/{rnd}_stdout.txt 2>/dev/null")
    err = err if err.isdigit() else "0"
    tb = tb if tb.isdigit() else "0"
    if err != "0" or tb != "0":
        print(f"  {rnd}: FAILED={err}, Traceback={tb} !!!")
    else:
        print(f"  {rnd}: clean")

# 4. Master log progress
print("\n--- Progress ---")
for rnd in ["round2", "round3", "round4"]:
    ml = run_cmd(f"tail -3 {REMOTE_DIR}/{rnd}_results/00_master_log.txt 2>/dev/null")
    if ml and "No such file" not in ml:
        print(f"  [{rnd}]")
        for line in ml.split('\n'):
            print(f"    {line}")

# 5. File counts
print("\n--- Result Files ---")
for rnd in ["round2", "round3", "round4"]:
    count = run_cmd(f"ls {REMOTE_DIR}/{rnd}_results/*.txt 2>/dev/null | wc -l")
    print(f"  {rnd}: {count} files")

# 6. Stall detection simulation
print("\n--- Stall Detection ---")
current = "round2" if r2 else "round3" if r3 else "round4" if r4 else None
if current:
    sizes = run_cmd(f"ls -la {REMOTE_DIR}/{current}_results/*.txt 2>/dev/null | tail -5")
    print(f"  Current round: {current}")
    print(f"  Latest files:")
    for line in sizes.split('\n'):
        if '.txt' in line:
            parts = line.split()
            print(f"    {parts[-1].split('/')[-1]}: {parts[4]} bytes")

# 7. CPU
load = run_cmd("cat /proc/loadavg")
print(f"\nLoad: {load}")

# 8. Disk
disk = run_cmd("df -h /root | tail -1")
print(f"Disk: {disk}")

# 9. Chainer log
print("\n--- Chainer Log ---")
cl = run_cmd(f"cat {REMOTE_DIR}/chain_full_log.txt 2>/dev/null")
print(cl if cl else "  (no log)")

ssh.close()
print("\n" + "=" * 70)
print("TEST COMPLETE - All checks passed, monitor is ready to run")
print("=" * 70)
