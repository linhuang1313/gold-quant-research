"""Quick check R13 on Server D."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = "connect.westd.seetacloud.com"
PORT = 35258

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace').strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password="r1zlTZQUb+E4",
          timeout=30, banner_timeout=60, auth_timeout=60)

print("=== Server D: R13 Alpha Refinement ===\n")

print("--- Result files ---")
files = run(c, "ls -lhS /root/gold-quant-trading/round13_results/*.txt 2>/dev/null")
print(files if files else "  No result files")

print("\n--- Master log ---")
master = run(c, "cat /root/gold-quant-trading/round13_results/00_master_log.txt 2>/dev/null")
print(master if master else "  No master log")

print("\n--- Phase progress ---")
phases = run(c, r"grep -E '(>>>|Phase|Starting|done|Complete|FAIL|R13-)' /root/gold-quant-trading/logs/round13.log 2>/dev/null | tail -40")
print(phases if phases else "  No phase markers found")

print("\n--- Last 20 log lines ---")
tail = run(c, "tail -20 /root/gold-quant-trading/logs/round13.log 2>/dev/null")
print(tail if tail else "  No log")

print("\n--- Log size ---")
print(run(c, "wc -l /root/gold-quant-trading/logs/round13.log 2>/dev/null"))

c.close()
