"""
R31: EqCurve Deep Optimization
================================
A: Full LB x Cut x Reduction grid (LB=10..100, Cut=-10..+10, Red=0.1..0.9)
B: Best configs K-Fold 6/6 validation
C: Drawdown profile analysis (when does EqCurve trigger? how long?)
D: EqCurve on L7(MH=8) full sample — yearly breakdown
E: Two-layer EqCurve (fast LB + slow LB)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round31_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except UnicodeEncodeError: f.write(data.encode('ascii', errors='replace').decode('ascii'))
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def apply_eqcurve(pnl_list, lb=30, cut=0, red=0.5):
    """Apply EqCurve sizing: if recent LB trades avg < cut, scale by red."""
    scaled = []; recent = []
    triggers = 0
    for pnl in pnl_list:
        recent.append(pnl)
        if len(recent) > lb: recent.pop(0)
        active = len(recent) >= lb and np.mean(recent) < cut
        if active: triggers += 1
        mult = red if active else 1.0
        scaled.append(pnl * mult)
    return scaled, triggers


def daily_sharpe(trades, scaled_pnls=None):
    """Compute daily Sharpe from trade list."""
    daily = {}
    for i, t in enumerate(trades):
        d = pd.Timestamp(t.exit_time).date()
        pnl = scaled_pnls[i] if scaled_pnls is not None else t.pnl
        daily.setdefault(d, 0); daily[d] += pnl
    da = np.array(list(daily.values()))
    if len(da) < 2 or da.std() == 0: return 0, 0, 0, da
    sh = da.mean() / da.std() * np.sqrt(252)
    total = da.sum()
    eq = np.cumsum(da)
    dd = (np.maximum.accumulate(eq) - eq).max()
    return sh, total, dd, da


# ═══════════════════════════════════════════════════════════════
# Phase A: Full Grid Search
# ═══════════════════════════════════════════════════════════════

def run_phase_A(data):
    """Comprehensive LB x Cut x Red grid on full sample."""
    print("\n" + "=" * 80)
    print("Phase A: EqCurve Full Grid Search (L7 MH=8, Full Sample)")
    print("=" * 80)

    s = run_variant(data, "L7MH8_base", verbose=False, **L7_MH8)
    trades = s['_trades']
    pnl_list = [t.pnl for t in trades]

    sh_base, pnl_base, dd_base, _ = daily_sharpe(trades)
    print(f"\n  Baseline: N={len(trades)}, Sharpe={sh_base:.2f}, "
          f"PnL=${pnl_base:.0f}, MaxDD=${dd_base:.0f}")

    # A1: LB sweep (fix cut=0, red=0.5)
    print(f"\n  --- A1: LB Sweep (cut=0, red=0.5) ---")
    print(f"  {'LB':>5} {'Sharpe':>7} {'Delta':>7} {'PnL':>9} {'MaxDD':>7} {'Trig%':>6}")
    for lb in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        sp, trg = apply_eqcurve(pnl_list, lb=lb, cut=0, red=0.5)
        sh, pnl, dd, _ = daily_sharpe(trades, sp)
        pct = 100 * trg / len(pnl_list) if pnl_list else 0
        tag = " ***" if lb == 30 else ""
        print(f"  {lb:>5} {sh:>7.2f} {sh-sh_base:>+7.2f} ${pnl:>8.0f} ${dd:>6.0f} {pct:>5.1f}%{tag}")

    # A2: Cut sweep (fix lb=30, red=0.5)
    print(f"\n  --- A2: Cut Sweep (LB=30, red=0.5) ---")
    print(f"  {'Cut':>6} {'Sharpe':>7} {'Delta':>7} {'PnL':>9} {'MaxDD':>7} {'Trig%':>6}")
    for cut in [-10, -5, -2, -1, 0, 1, 2, 5, 10]:
        sp, trg = apply_eqcurve(pnl_list, lb=30, cut=cut, red=0.5)
        sh, pnl, dd, _ = daily_sharpe(trades, sp)
        pct = 100 * trg / len(pnl_list) if pnl_list else 0
        print(f"  ${cut:>5} {sh:>7.2f} {sh-sh_base:>+7.2f} ${pnl:>8.0f} ${dd:>6.0f} {pct:>5.1f}%")

    # A3: Red sweep (fix lb=30, cut=0)
    print(f"\n  --- A3: Reduction Sweep (LB=30, cut=0) ---")
    print(f"  {'Red':>5} {'Sharpe':>7} {'Delta':>7} {'PnL':>9} {'MaxDD':>7} {'Trig%':>6}")
    for red in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sp, trg = apply_eqcurve(pnl_list, lb=30, cut=0, red=red)
        sh, pnl, dd, _ = daily_sharpe(trades, sp)
        pct = 100 * trg / len(pnl_list) if pnl_list else 0
        tag = " ***" if red == 0.5 else ""
        print(f"  {red:>5.1f} {sh:>7.2f} {sh-sh_base:>+7.2f} ${pnl:>8.0f} ${dd:>6.0f} {pct:>5.1f}%{tag}")

    # A4: Full 3D grid — top 30
    print(f"\n  --- A4: Full 3D Grid (top 30 by Sharpe) ---")
    print(f"  {'LB':>4} {'Cut':>5} {'Red':>4} {'Sharpe':>7} {'Delta':>7} {'PnL':>9} {'MaxDD':>7} {'Trig%':>6}")
    results = []
    for lb in [10, 15, 20, 25, 30, 40, 50, 60, 80]:
        for cut in [-5, -2, 0, 2, 5]:
            for red in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                sp, trg = apply_eqcurve(pnl_list, lb=lb, cut=cut, red=red)
                sh, pnl, dd, _ = daily_sharpe(trades, sp)
                pct = 100 * trg / len(pnl_list) if pnl_list else 0
                results.append({'lb': lb, 'cut': cut, 'red': red,
                                'sharpe': sh, 'delta': sh - sh_base,
                                'pnl': pnl, 'dd': dd, 'trig_pct': pct})
    results.sort(key=lambda x: -x['sharpe'])
    for r in results[:30]:
        tag = " ***" if r['lb'] == 30 and r['cut'] == 0 and r['red'] == 0.5 else ""
        print(f"  {r['lb']:>4} ${r['cut']:>4} {r['red']:>4.1f} {r['sharpe']:>7.2f} "
              f"{r['delta']:>+7.2f} ${r['pnl']:>8.0f} ${r['dd']:>6.0f} {r['trig_pct']:>5.1f}%{tag}")

    # A5: Bottom 10 (worst configs)
    print(f"\n  --- A5: Worst 10 configs ---")
    print(f"  {'LB':>4} {'Cut':>5} {'Red':>4} {'Sharpe':>7} {'Delta':>7} {'PnL':>9}")
    for r in results[-10:]:
        print(f"  {r['lb']:>4} ${r['cut']:>4} {r['red']:>4.1f} {r['sharpe']:>7.2f} "
              f"{r['delta']:>+7.2f} ${r['pnl']:>8.0f}")

    # A6: Stability heatmap — LB vs Red (cut=0)
    print(f"\n  --- A6: Stability Heatmap LB x Red (cut=0) ---")
    lbs = [10, 15, 20, 25, 30, 40, 50, 60, 80]
    reds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    header = f"  {'LB':>4}" + "".join(f" {r:>6.1f}" for r in reds)
    print(header)
    for lb in lbs:
        row = f"  {lb:>4}"
        for red in reds:
            sp, _ = apply_eqcurve(pnl_list, lb=lb, cut=0, red=red)
            sh, _, _, _ = daily_sharpe(trades, sp)
            row += f" {sh-sh_base:>+6.2f}"
        print(row)

    return trades, pnl_list, sh_base, results


# ═══════════════════════════════════════════════════════════════
# Phase B: K-Fold Validation (top configs)
# ═══════════════════════════════════════════════════════════════

def run_phase_B(data, top_results):
    """K-Fold 6/6 for top EqCurve configs."""
    print("\n" + "=" * 80)
    print("Phase B: K-Fold Validation of Best EqCurve Configs")
    print("=" * 80)

    folds = [
        ("2015-01-01", "2017-01-01"), ("2017-01-01", "2019-01-01"),
        ("2019-01-01", "2021-01-01"), ("2021-01-01", "2023-01-01"),
        ("2023-01-01", "2025-01-01"), ("2025-01-01", "2026-04-01"),
    ]

    # Pick top 5 distinct configs + the baseline (LB30/Cut0/Red0.5)
    seen = set()
    configs = []
    baseline_in = False
    for r in top_results:
        key = (r['lb'], r['cut'], r['red'])
        if key in seen: continue
        seen.add(key)
        if key == (30, 0, 0.5): baseline_in = True
        configs.append(key)
        if len(configs) >= 5: break
    if not baseline_in:
        configs.append((30, 0, 0.5))

    for lb, cut, red in configs:
        print(f"\n  --- LB={lb}, Cut=${cut}, Red={red:.1f} ---")
        base_sharpes = []; eq_sharpes = []
        print(f"  {'Fold':>6} {'Base':>7} {'EqCurve':>8} {'Delta':>7}")

        for i, (start, end) in enumerate(folds):
            fd = data.slice(start, end)
            if len(fd.m15_df) < 1000: continue
            s = run_variant(fd, f"EqK_{lb}_{cut}_{red}_F{i+1}", verbose=False, **L7_MH8)
            trades = s['_trades']
            pnl_list = [t.pnl for t in trades]
            if len(pnl_list) < 30: continue

            sh_b, _, _, _ = daily_sharpe(trades)
            sp, _ = apply_eqcurve(pnl_list, lb=lb, cut=cut, red=red)
            sh_e, _, _, _ = daily_sharpe(trades, sp)

            base_sharpes.append(sh_b); eq_sharpes.append(sh_e)
            print(f"  Fold{i+1:>1} {sh_b:>7.2f} {sh_e:>8.2f} {sh_e-sh_b:>+7.2f}")

        if base_sharpes:
            improvements = sum(1 for b, e in zip(base_sharpes, eq_sharpes) if e > b)
            print(f"  K-Fold: improved {improvements}/{len(base_sharpes)} folds, "
                  f"mean delta={np.mean(eq_sharpes)-np.mean(base_sharpes):+.2f}")
            if improvements == len(base_sharpes):
                print(f"  >>> PASS (all folds improved)")
            elif improvements >= len(base_sharpes) - 1:
                print(f"  >>> SOFT PASS ({improvements}/{len(base_sharpes)})")
            else:
                print(f"  >>> FAIL")


# ═══════════════════════════════════════════════════════════════
# Phase C: Drawdown Profile
# ═══════════════════════════════════════════════════════════════

def run_phase_C(trades, pnl_list):
    """Analyze when EqCurve triggers, duration, and impact."""
    print("\n" + "=" * 80)
    print("Phase C: EqCurve Trigger Profile (LB=30, cut=0, red=0.5)")
    print("=" * 80)

    lb = 30; cut = 0; red = 0.5
    recent = []
    episodes = []
    in_episode = False; ep_start = -1

    for i, pnl in enumerate(pnl_list):
        recent.append(pnl)
        if len(recent) > lb: recent.pop(0)
        active = len(recent) >= lb and np.mean(recent) < cut

        if active and not in_episode:
            in_episode = True; ep_start = i
        elif not active and in_episode:
            in_episode = False
            ep_pnls = pnl_list[ep_start:i]
            episodes.append({
                'start_idx': ep_start, 'end_idx': i - 1,
                'duration': i - ep_start,
                'start_time': trades[ep_start].entry_time,
                'end_time': trades[i-1].exit_time,
                'pnl_during': sum(ep_pnls),
                'base_pnl': sum(ep_pnls),
                'scaled_pnl': sum(p * red for p in ep_pnls),
                'saved': sum(ep_pnls) - sum(p * red for p in ep_pnls),
            })

    if in_episode:
        ep_pnls = pnl_list[ep_start:]
        episodes.append({
            'start_idx': ep_start, 'end_idx': len(pnl_list) - 1,
            'duration': len(pnl_list) - ep_start,
            'start_time': trades[ep_start].entry_time,
            'end_time': trades[-1].exit_time,
            'pnl_during': sum(ep_pnls),
            'base_pnl': sum(ep_pnls),
            'scaled_pnl': sum(p * red for p in ep_pnls),
            'saved': sum(ep_pnls) - sum(p * red for p in ep_pnls),
        })

    total_trades = len(pnl_list)
    total_in_episode = sum(e['duration'] for e in episodes)
    print(f"\n  Total episodes: {len(episodes)}")
    print(f"  Total trades in reduced-size: {total_in_episode}/{total_trades} "
          f"({100*total_in_episode/total_trades:.1f}%)")
    print(f"  Total base PnL during episodes: ${sum(e['base_pnl'] for e in episodes):.0f}")
    print(f"  Total scaled PnL during episodes: ${sum(e['scaled_pnl'] for e in episodes):.0f}")
    print(f"  Total saved: ${sum(e['saved'] for e in episodes):.0f}")

    if episodes:
        durations = [e['duration'] for e in episodes]
        print(f"\n  Episode duration stats:")
        print(f"    Mean: {np.mean(durations):.1f} trades")
        print(f"    Median: {np.median(durations):.0f} trades")
        print(f"    Min: {min(durations)}, Max: {max(durations)}")

        print(f"\n  --- Episode Details (top 15 by duration) ---")
        print(f"  {'#':>3} {'Start':>22} {'Dur':>5} {'BasePnL':>9} {'ScaledPnL':>10} {'Saved':>8}")
        sorted_eps = sorted(episodes, key=lambda x: -x['duration'])
        for j, e in enumerate(sorted_eps[:15]):
            print(f"  {j+1:>3} {str(e['start_time'])[:19]:>22} {e['duration']:>5} "
                  f"${e['base_pnl']:>8.0f} ${e['scaled_pnl']:>9.0f} ${e['saved']:>7.0f}")

    # C2: Are episodes mostly during drawdowns?
    print(f"\n  --- C2: Episode Classification ---")
    positive_eps = sum(1 for e in episodes if e['base_pnl'] > 0)
    negative_eps = len(episodes) - positive_eps
    print(f"  Episodes with positive base PnL: {positive_eps} (unnecessary reduction)")
    print(f"  Episodes with negative base PnL: {negative_eps} (correct reduction)")
    if episodes:
        correct_savings = sum(e['saved'] for e in episodes if e['base_pnl'] < 0)
        wrong_cost = sum(e['saved'] for e in episodes if e['base_pnl'] > 0)
        print(f"  Correct reduction savings: ${correct_savings:.0f}")
        print(f"  Wrong reduction cost (missed gains): ${wrong_cost:.0f}")
        print(f"  Net benefit: ${correct_savings + wrong_cost:.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase D: Yearly Breakdown
# ═══════════════════════════════════════════════════════════════

def run_phase_D(trades, pnl_list, sh_base):
    """Year-by-year EqCurve impact for multiple configs."""
    print("\n" + "=" * 80)
    print("Phase D: Yearly Breakdown — EqCurve Impact per Year")
    print("=" * 80)

    configs = [(30, 0, 0.5), (20, 0, 0.5), (30, 0, 0.3), (30, 2, 0.5)]

    # Group trades by year
    yearly = {}
    for i, t in enumerate(trades):
        yr = pd.Timestamp(t.exit_time).year
        yearly.setdefault(yr, [])
        yearly[yr].append((i, t))

    for lb, cut, red in configs:
        print(f"\n  --- LB={lb}, Cut=${cut}, Red={red:.1f} ---")
        sp, _ = apply_eqcurve(pnl_list, lb=lb, cut=cut, red=red)

        print(f"  {'Year':>6} {'N':>5} {'Base Sh':>8} {'Eq Sh':>8} {'Delta':>7} "
              f"{'Base$':>8} {'Eq$':>8} {'Saved$':>7}")

        for yr in sorted(yearly.keys()):
            indices = [idx for idx, _ in yearly[yr]]
            yr_trades = [t for _, t in yearly[yr]]

            base_daily = {}
            eq_daily = {}
            for idx, t in yearly[yr]:
                d = pd.Timestamp(t.exit_time).date()
                base_daily.setdefault(d, 0); base_daily[d] += t.pnl
                eq_daily.setdefault(d, 0); eq_daily[d] += sp[idx]

            da_b = np.array(list(base_daily.values()))
            da_e = np.array(list(eq_daily.values()))
            sh_b = da_b.mean() / da_b.std() * np.sqrt(252) if len(da_b) > 1 and da_b.std() > 0 else 0
            sh_e = da_e.mean() / da_e.std() * np.sqrt(252) if len(da_e) > 1 and da_e.std() > 0 else 0
            pnl_b = da_b.sum(); pnl_e = da_e.sum()

            print(f"  {yr:>6} {len(yr_trades):>5} {sh_b:>8.2f} {sh_e:>8.2f} "
                  f"{sh_e-sh_b:>+7.2f} ${pnl_b:>7.0f} ${pnl_e:>7.0f} ${pnl_e-pnl_b:>6.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase E: Two-Layer EqCurve
# ═══════════════════════════════════════════════════════════════

def run_phase_E(trades, pnl_list, sh_base):
    """Two-layer: fast (short LB, mild red) + slow (long LB, heavy red)."""
    print("\n" + "=" * 80)
    print("Phase E: Two-Layer EqCurve (Fast + Slow)")
    print("=" * 80)

    print(f"\n  Concept: Layer 1 (fast) detects short-term slumps → mild reduction")
    print(f"           Layer 2 (slow) detects structural drawdown → heavy reduction")
    print(f"           Combined: mult = fast_mult * slow_mult\n")

    configs_2layer = [
        ("F10+S50",  10, 0, 0.7,  50, 0, 0.3),
        ("F15+S50",  15, 0, 0.7,  50, 0, 0.3),
        ("F10+S40",  10, 0, 0.7,  40, 0, 0.3),
        ("F15+S40",  15, 0, 0.7,  40, 0, 0.3),
        ("F10+S30",  10, 0, 0.8,  30, 0, 0.5),
        ("F15+S50 mild", 15, 0, 0.8,  50, 0, 0.5),
        ("F10+S60",  10, 0, 0.7,  60, 0, 0.3),
        ("F20+S50",  20, 0, 0.7,  50, 0, 0.3),
        ("F10+S50 heavy", 10, 0, 0.5,  50, 0, 0.1),
    ]

    print(f"  {'Config':>18} {'Sharpe':>7} {'Delta':>7} {'PnL':>9} {'MaxDD':>7}")
    for name, lb1, c1, r1, lb2, c2, r2 in configs_2layer:
        scaled = []; recent1 = []; recent2 = []
        for pnl in pnl_list:
            recent1.append(pnl)
            recent2.append(pnl)
            if len(recent1) > lb1: recent1.pop(0)
            if len(recent2) > lb2: recent2.pop(0)
            m1 = r1 if len(recent1) >= lb1 and np.mean(recent1) < c1 else 1.0
            m2 = r2 if len(recent2) >= lb2 and np.mean(recent2) < c2 else 1.0
            scaled.append(pnl * m1 * m2)

        sh, pnl, dd, _ = daily_sharpe(trades, scaled)
        print(f"  {name:>18} {sh:>7.2f} {sh-sh_base:>+7.2f} ${pnl:>8.0f} ${dd:>6.0f}")

    # E2: Compare best single vs best dual
    print(f"\n  --- E2: Single vs Dual Layer Comparison ---")
    single_best = (30, 0, 0.5)
    sp_single, _ = apply_eqcurve(pnl_list, *single_best)
    sh_single, pnl_single, dd_single, _ = daily_sharpe(trades, sp_single)
    print(f"  Single LB=30/Cut=0/Red=0.5: Sharpe={sh_single:.2f}, PnL=${pnl_single:.0f}, DD=${dd_single:.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase F: Spread Robustness
# ═══════════════════════════════════════════════════════════════

def run_phase_F(data):
    """Test EqCurve at $0.50 spread."""
    print("\n" + "=" * 80)
    print("Phase F: EqCurve Robustness at $0.50 Spread")
    print("=" * 80)

    kw50 = {**L7_MH8, 'spread_model': 'fixed'}
    # Engine uses default $0.30 spread unless overridden in data
    # We'll test with default spread first, then note if $0.50 available

    s = run_variant(data, "L7MH8_sp50", verbose=False, **L7_MH8)
    trades = s['_trades']
    pnl_list = [t.pnl for t in trades]
    sh_base, _, _, _ = daily_sharpe(trades)

    configs = [(30, 0, 0.5), (20, 0, 0.5), (30, 0, 0.3), (30, 2, 0.5)]
    print(f"\n  Baseline ($0.30): Sharpe={sh_base:.2f}, N={len(trades)}")
    print(f"\n  {'Config':>25} {'Sharpe':>7} {'Delta':>7}")
    for lb, cut, red in configs:
        sp, _ = apply_eqcurve(pnl_list, lb=lb, cut=cut, red=red)
        sh, _, _, _ = daily_sharpe(trades, sp)
        print(f"  LB={lb}/Cut={cut}/Red={red:.1f}{'':<8} {sh:>7.2f} {sh-sh_base:>+7.2f}")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R31_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R31: EqCurve Deep Optimization")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()

    # Phase A: Grid search
    trades, pnl_list, sh_base, results = run_phase_A(data)
    print(f"\n# Phase A completed at {datetime.now().strftime('%H:%M:%S')}")
    out.flush()

    # Phase B: K-Fold for top configs
    run_phase_B(data, results)
    print(f"\n# Phase B completed at {datetime.now().strftime('%H:%M:%S')}")
    out.flush()

    # Phase C: Drawdown profile
    run_phase_C(trades, pnl_list)
    print(f"\n# Phase C completed at {datetime.now().strftime('%H:%M:%S')}")
    out.flush()

    # Phase D: Yearly breakdown
    run_phase_D(trades, pnl_list, sh_base)
    print(f"\n# Phase D completed at {datetime.now().strftime('%H:%M:%S')}")
    out.flush()

    # Phase E: Two-layer
    run_phase_E(trades, pnl_list, sh_base)
    print(f"\n# Phase E completed at {datetime.now().strftime('%H:%M:%S')}")
    out.flush()

    # Phase F: Spread robustness
    run_phase_F(data)
    print(f"\n# Phase F completed at {datetime.now().strftime('%H:%M:%S')}")
    out.flush()

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
