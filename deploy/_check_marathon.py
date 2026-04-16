"""Monitor 24h marathon progress on remote server."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import paramiko

HOST = "connect.westc.seetacloud.com"
PORT = 30886
USER = "root"
PASS = "r1zlTZQUb+E4"
REMOTE_DIR = "/root/gold-quant-trading"

def run_cmd(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode('utf-8', errors='replace').strip()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)

# 1. Process check
print("=== Process Status ===")
out = run_cmd(ssh, "ps aux | grep python | grep -E '(run_|marathon)' | grep -v grep")
if out:
    for line in out.split('\n'):
        print(f"  {line.strip()}")
else:
    print("  No marathon processes running!")

# 2. Master log
print("\n=== Master Log (last 30 lines) ===")
out = run_cmd(ssh, f"tail -30 {REMOTE_DIR}/marathon_results/00_master_log.txt 2>/dev/null || echo 'No master log yet'")
print(out)

# 3. Phase files status
print("\n=== Phase Files ===")
out = run_cmd(ssh, f"ls -la {REMOTE_DIR}/marathon_results/ 2>/dev/null")
if out:
    for line in out.split('\n'):
        print(f"  {line}")
else:
    print("  No results directory yet")

# 4. Current phase output (last 15 lines)
print("\n=== Current Phase Output (last 15 lines) ===")
out = run_cmd(ssh, f"tail -15 {REMOTE_DIR}/marathon_stdout.txt 2>/dev/null || echo 'No stdout yet'")
print(out)

# 5. CPU usage
print("\n=== CPU Usage ===")
out = run_cmd(ssh, "top -bn1 | head -5")
print(out)

# 6. Completed phases summary
print("\n=== Completed Phase Summaries ===")
phases = [
    "phase02_tdtp_combo.txt",
    "phase03_stress.txt",
    "phase04_regime_trail.txt",
    "phase05_maxhold.txt",
    "phase06_optimal_12fold.txt",
    "phase07_recent.txt",
    "phase08_montecarlo.txt",
    "phase09_annual.txt",
    "phase10_final.txt",
    "phase11_optional.txt",
]
for phase_file in phases:
    path = f"{REMOTE_DIR}/marathon_results/{phase_file}"
    out = run_cmd(ssh, f"tail -3 {path} 2>/dev/null")
    if out and "Completed" in out or "Elapsed" in out:
        elapsed_line = [l for l in out.split('\n') if 'Elapsed' in l]
        print(f"  ✅ {phase_file}: {elapsed_line[0].strip() if elapsed_line else 'DONE'}")
    elif out and "FAILED" in out:
        print(f"  ❌ {phase_file}: FAILED")
    else:
        size = run_cmd(ssh, f"wc -c {path} 2>/dev/null | awk '{{print $1}}'")
        if size and size != '0':
            print(f"  ⏳ {phase_file}: running ({size} bytes)")
        else:
            print(f"  ⬜ {phase_file}: not started")

ssh.close()
