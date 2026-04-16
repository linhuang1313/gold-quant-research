#!/usr/bin/env python3
"""Read all R9 result files from server."""
import paramiko, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=30367, username="root", password="r1zlTZQUb+E4", timeout=15)

files = [
    "r9_1_l7_construction.txt",
    "r9_2_l7_kfold.txt",
    "r9_3_l7_monte_carlo.txt",
    "r9_4_l7_yearly_heatmap.txt",
    "r9_5_sl_maxhold_2d.txt",
    "r9_6_choppy_adx_2d.txt",
    "r9_7_trailing_variants.txt",
    "r9_8_time_decay.txt",
    "r9_9_rolling_wf.txt",
    "r9_10_spread_models.txt",
    "r9_11_extreme_periods.txt",
    "r9_12_bankruptcy.txt",
    "r9_13_rsi_opt.txt",
    "r9_14_orb_opt.txt",
    "r9_15_kc_params.txt",
    "r9_16_session.txt",
    "r9_17_cooldown.txt",
    "r9_18_atr_spike.txt",
]

os.makedirs("round9_results", exist_ok=True)

for f in files:
    remote = f"/root/gold-quant-trading/round9_results/{f}"
    i, o, e = c.exec_command(f"cat {remote}", timeout=30)
    content = o.read().decode("utf-8", errors="replace")
    print(f"\n{'='*80}")
    print(f"=== {f} ===")
    print(f"{'='*80}")
    print(content)
    local = os.path.join("round9_results", f)
    with open(local, "w", encoding="utf-8") as fh:
        fh.write(content)

c.close()
print("\n\nAll files saved to round9_results/")
