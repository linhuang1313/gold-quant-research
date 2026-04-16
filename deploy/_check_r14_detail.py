"""Detailed R14 progress check."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=15)
    return stdout.read().decode('utf-8', errors='replace').strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

print("=== R14 Detail Progress ===\n")

# Phase markers from log
phases = run(c, f"grep -E '(>>>|<<<|Phase|FAILED|Starting|done in)' {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null")
if phases:
    print("Phase Progress:")
    for line in phases.split("\n"):
        print(f"  {line}")

print("\n--- Result Files ---")
files = run(c, f"ls -lhS {REMOTE_DIR}/round14_results/*.txt 2>/dev/null")
if files:
    for line in files.split("\n"):
        print(f"  {line}")

print(f"\n--- Log size ---")
print(f"  {run(c, f'wc -l {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null')}")

print(f"\n--- Last 5 meaningful log lines ---")
last = run(c, f"grep -v '^$' {REMOTE_DIR}/round14_results/r14_log.txt 2>/dev/null | grep -v '^\\ *[0-9]*%' | tail -10")
if last:
    for line in last.split("\n"):
        print(f"  {line}")

c.close()
