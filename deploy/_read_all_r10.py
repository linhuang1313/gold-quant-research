#!/usr/bin/env python3
"""Read all R10 result files from server."""
import paramiko, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

files = [
    "00_master_log.txt",
    "r10_1_l8_construction.txt",
    "r10_2_l8_kfold.txt",
    "r10_3_l8_monte_carlo.txt",
    "r10_4_l8_yearly.txt",
    "r10_5_l8_extreme.txt",
    "r10_6_trail_grid.txt",
    "r10_7_time_trail.txt",
    "r10_8_breakeven.txt",
    "r10_9_statemachine.txt",
    "r10_10_h4_filter.txt",
    "r10_11_kc_params.txt",
    "r10_12_hist_spread.txt",
    "r10_13_loss_profile.txt",
    "r10_14_momentum.txt",
    "r10_15_bankruptcy.txt",
    "r10_16_param_cliff.txt",
    "r10_17_wf_windows.txt",
    "r10_18_purged_kfold.txt",
    "r10_19_regime_trans.txt",
    "r10_20_overlap.txt",
]

os.makedirs("round10_results", exist_ok=True)

for f in files:
    remote = f"/root/gold-quant-trading/round10_results/{f}"
    i, o, e = c.exec_command(f"cat {remote}", timeout=30)
    content = o.read().decode("utf-8", errors="replace")
    if not content.strip():
        print(f"  {f}: (empty/not found)")
        continue
    print(f"\n{'='*80}")
    print(f"=== {f} ===")
    print(f"{'='*80}")
    print(content)
    local = os.path.join("round10_results", f)
    with open(local, "w", encoding="utf-8") as fh:
        fh.write(content)

c.close()
print("\n\nAll files saved to round10_results/")
