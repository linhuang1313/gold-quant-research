"""
R38: SL/TP 收紧 × Trail Activation 放宽 — 联动回测
====================================================
用户提出两个方向:
  方向 A: SL 3.5→3.0×ATR, TP 8.0→6.0×ATR (收紧对称化)
  方向 B: Trail activation 从 regime 默认 → 更高起步 (让赢家跑更远)

测试矩阵 (所有配置在 L7 基线上变动):
  BASELINE: 当前实盘 L7 (SL=3.5, TP=8.0, trail as-is)
  A1: SL=3.0, TP=6.0           (方向A 核心)
  A2: SL=3.0, TP=8.0           (只改SL, TP不动)
  A3: SL=2.5, TP=5.0           (更激进收紧, 探索边界)
  B1: trail_act +50%           (方向B 核心: 0.30→0.45/0.20→0.30/0.08→0.12)
  B2: trail_act ×2             (更激进: 0.30→0.60/0.20→0.40/0.08→0.16)
  AB: SL=3.0, TP=6.0 + B1     (A+B 组合)

判定标准 (同 R30):
  1. 全样本 Sharpe ≥ BASELINE 且 ≥ 1.0
  2. 6-Fold K-Fold 全正 (6/6)
  3. K-Fold Sharpe std ≤ 1.5
  4. MaxDD 不显著恶化 (≤ BASELINE × 1.3)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import indicators
from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS, sanitize_for_json


OUT_DIR = Path("results/round38_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except Exception:
                pass
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def get_l7_base():
    """L7 实盘所有锁定参数 (与 R30 一致)"""
    kw = {**LIVE_PARITY_KWARGS}
    kw['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 8
    return kw


def build_configs():
    """Build all test configurations."""
    configs = []

    # BASELINE: current live L7
    kw = get_l7_base()
    configs.append(("BASELINE", 150, kw))

    # A1: SL=3.0, TP=6.0
    kw = get_l7_base()
    kw['sl_atr_mult'] = 3.0
    kw['tp_atr_mult'] = 6.0
    configs.append(("A1_SL3.0_TP6.0", 150, kw))

    # A2: SL=3.0, TP=8.0 (only SL changed)
    kw = get_l7_base()
    kw['sl_atr_mult'] = 3.0
    configs.append(("A2_SL3.0_TP8.0", 150, kw))

    # A3: SL=2.5, TP=5.0 (aggressive)
    kw = get_l7_base()
    kw['sl_atr_mult'] = 2.5
    kw['tp_atr_mult'] = 5.0
    configs.append(("A3_SL2.5_TP5.0", 150, kw))

    # B1: trail activation +50%
    kw = get_l7_base()
    kw['regime_config'] = {
        'low':    {'trail_act': 0.45, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.30, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.12, 'trail_dist': 0.01},
    }
    configs.append(("B1_trail_act_1.5x", 150, kw))

    # B2: trail activation ×2
    kw = get_l7_base()
    kw['regime_config'] = {
        'low':    {'trail_act': 0.60, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.40, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.16, 'trail_dist': 0.01},
    }
    configs.append(("B2_trail_act_2x", 150, kw))

    # AB: A1 + B1 combined
    kw = get_l7_base()
    kw['sl_atr_mult'] = 3.0
    kw['tp_atr_mult'] = 6.0
    kw['regime_config'] = {
        'low':    {'trail_act': 0.45, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.30, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.12, 'trail_dist': 0.01},
    }
    configs.append(("AB_SL3_TP6_trail1.5x", 150, kw))

    return configs


def run_with_sl_max(data, label, sl_max, kw, run_fn=run_variant, **extra):
    """Temporarily patch ATR_SL_MAX, run, restore."""
    original = indicators.ATR_SL_MAX
    try:
        indicators.ATR_SL_MAX = sl_max
        if run_fn is run_kfold:
            return run_fn(data, kw, **extra)
        return run_fn(data, label, **kw, **extra)
    finally:
        indicators.ATR_SL_MAX = original


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R38_sl_trail_rebalance.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print("# R38: SL/TP Tightening x Trail Activation Widening")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Purpose: Test Direction A (SL/TP tightening) and Direction B (trail activation widening)")
    print()

    configs = build_configs()
    data = DataBundle.load_default()

    # ── 1. Full Sample ──
    print("=" * 100)
    print("1. Full-Sample Comparison (2015-01-01 ~ latest)")
    print("=" * 100)
    header = (f"  {'Label':<24}{'SL':>5}{'TP':>5}"
              f"{'TrailAct_N':>11}{'N':>6}{'Sharpe':>8}{'PnL':>10}"
              f"{'WR':>6}{'MaxDD':>9}{'AvgWin':>8}{'AvgLoss':>9}{'AvgTrade':>10}")
    print(header)
    print("  " + "-" * 110)

    full_results = {}
    for label, sl_max, kw in configs:
        s = run_with_sl_max(data, label, sl_max, kw, verbose=False)
        full_results[label] = s
        avg_win = s.get('avg_win', 0) or 0
        avg_loss = s.get('avg_loss', 0) or 0
        avg_trade = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
        trail_act_n = kw['regime_config']['normal']['trail_act']
        print(f"  {label:<24}{kw['sl_atr_mult']:>5.1f}{kw['tp_atr_mult']:>5.1f}"
              f"{trail_act_n:>11.2f}{s['n']:>6}{s['sharpe']:>8.2f}${s['total_pnl']:>9.0f}"
              f"{s['win_rate']:>5.1f}%${s['max_dd']:>8.0f}"
              f"${avg_win:>7.1f}${avg_loss:>8.1f}${avg_trade:>9.2f}")
    out.flush()

    # ── 2. K-Fold ──
    print(f"\n{'='*100}")
    print("2. 6-Fold K-Fold Cross Validation")
    print("=" * 100)

    kfold_summary = {}
    all_kf_results = {}
    for label, sl_max, kw in configs:
        trail_act_n = kw['regime_config']['normal']['trail_act']
        print(f"\n--- {label} (SL={kw['sl_atr_mult']}, TP={kw['tp_atr_mult']}, "
              f"trail_act_normal={trail_act_n}) ---")
        kf = run_with_sl_max(
            data, label, sl_max, kw,
            run_fn=run_kfold, n_folds=6, label_prefix=f"{label}_"
        )
        sharpes, pnls = [], []
        print(f"  {'Fold':<8}{'Period':<27}{'N':>5}{'Sharpe':>8}{'PnL':>10}{'WR':>6}{'MaxDD':>9}")
        for r in kf:
            fold = r.get('fold', '?')
            period = f"{r.get('test_start','?')}~{r.get('test_end','?')}"
            print(f"  {fold:<8}{period:<27}{r['n']:>5}{r['sharpe']:>8.2f}"
                  f"${r['total_pnl']:>9.0f}{r['win_rate']:>5.1f}%${r['max_dd']:>8.0f}")
            sharpes.append(r['sharpe'])
            pnls.append(r['total_pnl'])
        out.flush()

        pos = sum(1 for s in sharpes if s > 0)
        kfold_summary[label] = {
            'pos': pos, 'total': len(sharpes),
            'mean': float(np.mean(sharpes)),
            'std': float(np.std(sharpes)),
            'min': float(np.min(sharpes)),
            'max': float(np.max(sharpes)),
            'pnl_sum': float(sum(pnls)),
        }
        all_kf_results[label] = kf
        print(f"  K-Fold: {pos}/{len(sharpes)} pos, mean={np.mean(sharpes):.2f}, "
              f"std={np.std(sharpes):.2f}, min={np.min(sharpes):.2f}")

    # ── 3. Decision Matrix ──
    print(f"\n{'='*100}")
    print("3. Decision Matrix")
    print("=" * 100)
    print(f"\n  {'Label':<24}{'Full Sharpe':>12}{'KF pos':>8}{'KF mean':>10}"
          f"{'KF std':>9}{'KF min':>9}{'Full PnL':>10}{'Verdict':>18}")
    print("  " + "-" * 105)

    baseline_sharpe = full_results.get('BASELINE', {}).get('sharpe', 0)
    baseline_maxdd = full_results.get('BASELINE', {}).get('max_dd', 0)

    verdicts = {}
    for label, sl_max, kw in configs:
        full_s = full_results[label]['sharpe']
        full_dd = full_results[label]['max_dd']
        ks = kfold_summary[label]
        issues = []
        if label != 'BASELINE' and full_s < baseline_sharpe:
            issues.append("Sharpe<BASE")
        if full_s < 1.0:
            issues.append("Sharpe<1")
        if ks['pos'] < ks['total']:
            issues.append(f"KF{ks['pos']}/{ks['total']}")
        if ks['std'] > 1.5:
            issues.append("KFstd>1.5")
        if ks['min'] < 0:
            issues.append("KFmin<0")
        if label != 'BASELINE' and abs(full_dd) > abs(baseline_maxdd) * 1.3:
            issues.append("MaxDD>130%")
        verdict = "PASS" if not issues else ",".join(issues)
        verdicts[label] = verdict
        print(f"  {label:<24}{full_s:>12.2f}{ks['pos']:>4}/{ks['total']:<3}"
              f"{ks['mean']:>10.2f}{ks['std']:>9.2f}{ks['min']:>9.2f}"
              f"${full_results[label]['total_pnl']:>9.0f}{verdict:>18}")

    # ── 4. Direction A vs B head-to-head ──
    print(f"\n{'='*100}")
    print("4. Head-to-Head: Direction A vs Direction B vs Baseline")
    print("=" * 100)

    for group_name, labels in [
        ("Direction A (SL/TP tightening)", ["BASELINE", "A1_SL3.0_TP6.0", "A2_SL3.0_TP8.0", "A3_SL2.5_TP5.0"]),
        ("Direction B (Trail activation)", ["BASELINE", "B1_trail_act_1.5x", "B2_trail_act_2x"]),
        ("Combined A+B", ["BASELINE", "A1_SL3.0_TP6.0", "B1_trail_act_1.5x", "AB_SL3_TP6_trail1.5x"]),
    ]:
        print(f"\n  {group_name}:")
        print(f"    {'Label':<24}{'Sharpe':>8}{'KF mean':>10}{'N':>6}{'PnL':>10}{'WR':>6}{'AvgTrade':>10}{'Verdict':>10}")
        for lbl in labels:
            if lbl not in full_results:
                continue
            s = full_results[lbl]
            ks = kfold_summary[lbl]
            avg_t = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
            print(f"    {lbl:<24}{s['sharpe']:>8.2f}{ks['mean']:>10.2f}"
                  f"{s['n']:>6}${s['total_pnl']:>9.0f}{s['win_rate']:>5.1f}%${avg_t:>9.2f}"
                  f"{'  '+verdicts[lbl]:>10}")

    # ── 5. Fold-by-fold comparison (does A or B win in MORE folds?) ──
    print(f"\n{'='*100}")
    print("5. Fold-by-Fold Comparison (A1 vs B1 vs BASELINE)")
    print("=" * 100)
    key_labels = ["BASELINE", "A1_SL3.0_TP6.0", "B1_trail_act_1.5x", "AB_SL3_TP6_trail1.5x"]
    fold_names = [f"Fold{i}" for i in range(1, 7)]
    print(f"  {'Fold':<10}", end="")
    for lbl in key_labels:
        print(f"{lbl:>24}", end="")
    print(f"{'Best':>12}")
    for fn in fold_names:
        print(f"  {fn:<10}", end="")
        fold_sharpes = {}
        for lbl in key_labels:
            if lbl not in all_kf_results:
                continue
            for r in all_kf_results[lbl]:
                if r.get('fold') == fn:
                    fold_sharpes[lbl] = r['sharpe']
                    print(f"{r['sharpe']:>24.2f}", end="")
                    break
        best = max(fold_sharpes, key=fold_sharpes.get) if fold_sharpes else "?"
        short = best.split("_")[0] if "_" in best else best
        print(f"{short:>12}")

    # ── 6. Recommendation ──
    print(f"\n{'='*100}")
    print("6. Recommendation")
    print("=" * 100)

    passing = [(l, full_results[l]['sharpe'], kfold_summary[l]['mean'])
               for l, _, _ in configs
               if verdicts[l] == "PASS" and l != "BASELINE"]

    if passing:
        passing.sort(key=lambda x: (-x[2], -x[1]))
        print("\n  Configs that PASS all criteria (sorted by KF mean Sharpe):")
        for p in passing:
            print(f"    {p[0]:<24}  full={p[1]:.2f}  KF_mean={p[2]:.2f}")
        print(f"\n  >>> RECOMMENDED: {passing[0][0]}")
        print(f"      Next step: paper trade >= 30 trades before live deployment")
    else:
        print("\n  >>> No config passes all criteria.")
        print("      BASELINE remains optimal. No change recommended.")

    # ── Save JSON ──
    json_out = {
        "experiment": "R38",
        "timestamp": datetime.now().isoformat(),
        "baseline_sharpe": baseline_sharpe,
        "full_results": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer, float, int)) else vv
                              for kk, vv in v.items() if not kk.startswith('_')}
                         for k, v in full_results.items()},
        "kfold_summary": kfold_summary,
        "verdicts": verdicts,
    }
    json_path = OUT_DIR / "R38_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")
    print(f"# Results: {out_path}")
    print(f"# JSON: {json_path}")

    sys.stdout = old_stdout
    out.close()
    print(f"Done! Results saved to {out_path}")


if __name__ == "__main__":
    main()
