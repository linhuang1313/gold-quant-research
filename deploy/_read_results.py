"""Read key result files from both servers."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def run(c, cmd):
    _, stdout, _ = c.exec_command(cmd, timeout=30)
    return stdout.read().decode('utf-8', errors='replace').strip()

# Server D - R13 results
print("=" * 80)
print("SERVER D: R13 Alpha Refinement - Key Results")
print("=" * 80)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=35258, username="root",
          password="r1zlTZQUb+E4", timeout=30, banner_timeout=60, auth_timeout=60)

for fname in ["R13-A1_ema_scan.txt", "R13-A2_mult_scan.txt", "R13-A3_heatmap.txt",
              "R13-A4_kfold.txt", "R13-B1_breakeven.txt", "R13-B2_be_kfold.txt",
              "R13-C1_dual_kc.txt", "R13-C2_dual_params.txt", "R13-C3_dual_kfold.txt",
              "R13-D1_hma.txt", "R13-D2_kama.txt", "R13-D3_ma_kfold.txt"]:
    content = run(c, f"cat /root/gold-quant-trading/round13_results/{fname} 2>/dev/null")
    if content:
        print(f"\n--- {fname} ---")
        print(content)

# current phase
print("\n--- Current running phase ---")
phase_markers = run(c, r"grep -n 'R13-\|Phase\|=====\|Experiment' /root/gold-quant-trading/logs/round13.log 2>/dev/null | tail -10")
print(phase_markers if phase_markers else "No phase markers")

c.close()
