"""Check R16 progress on Server C."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westc.seetacloud.com"
PORT = 16005

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=20)
    return stdout.read().decode('utf-8', errors='replace').strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4", timeout=15)

print("=== Server C: R16 Progress ===\n")

procs = run(c, "ps aux | grep run_round16 | grep python | grep -v grep")
if procs:
    print("Running:")
    for line in procs.split("\n"):
        parts = line.split()
        if len(parts) > 10:
            print(f"  PID={parts[1]} CPU={parts[2]}% MEM={parts[3]}% CMD={' '.join(parts[10:])}")
else:
    print("  No R16 process running!")

print("\n--- Result files ---")
files = run(c, "ls -lhS /root/gold-quant-trading/results/round16_results/*.txt 2>/dev/null")
if files:
    for line in files.split("\n"):
        print(f"  {line}")
else:
    print("  No result files yet")

print(f"\n--- Log size ---")
log_path = "/root/gold-quant-trading/logs/round16.log"
print(f"  {run(c, f'wc -l {log_path} 2>/dev/null')}")

print(f"\n--- Phase progress ---")
phases = run(c, f"grep -E '(R16-|Phase|COMPLETE|FAIL|done)' {log_path} 2>/dev/null | tail -30")
if phases:
    for line in phases.split("\n"):
        print(f"  {line}")

print(f"\n--- Last 10 log lines ---")
tail = run(c, f"grep -v '^$' {log_path} 2>/dev/null | tail -10")
if tail:
    for line in tail.split("\n"):
        print(f"  {line}")

summary = run(c, "cat /root/gold-quant-trading/results/round16_results/R16_summary.txt 2>/dev/null")
if summary:
    print(f"\n--- R16 Summary ---")
    print(summary)

c.close()
