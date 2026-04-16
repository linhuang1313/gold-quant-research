"""Check R13 progress on Server D."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=20)
    return stdout.read().decode('utf-8', errors='replace').strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4", timeout=15)

print("=== Server D: R13 Progress ===\n")

print("--- R13 result files ---")
files = run(c, "ls -lhS /root/gold-quant-trading/round13_results/*.txt 2>/dev/null")
if files:
    for line in files.split("\n"):
        print(f"  {line}")

print("\n--- R13 log ---")
log_path = "/root/gold-quant-trading/logs/round13.log"
log_exists = run(c, f"test -f {log_path} && echo YES || echo NO")
if log_exists == "YES":
    print(f"  Log size: {run(c, f'wc -l {log_path}')}")

    phases = run(c, f"grep -E '(>>>|<<<|Phase|FAILED|Starting|done in|DONE|Complete)' {log_path} 2>/dev/null")
    if phases:
        print("\n  Phase progress:")
        for line in phases.split("\n"):
            print(f"    {line}")

    print(f"\n  Last 15 meaningful lines:")
    tail = run(c, f"grep -v '^$' {log_path} | grep -v '^\\ *[0-9]*%' | tail -15")
    if tail:
        for l in tail.split("\n"):
            print(f"    {l}")
else:
    print(f"  Log file not found at {log_path}")
    alt = run(c, "find /root/gold-quant-trading -name '*round13*' -o -name '*r13*' 2>/dev/null | head -20")
    if alt:
        print(f"  Found related files:")
        for l in alt.split("\n"):
            print(f"    {l}")

print("\n--- R13 master log ---")
master = run(c, "cat /root/gold-quant-trading/round13_results/00_master_log.txt 2>/dev/null")
if master:
    print(master)
else:
    print("  No master log yet")

c.close()
