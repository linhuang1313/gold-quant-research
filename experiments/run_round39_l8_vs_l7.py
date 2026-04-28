"""
R39: L8 (UT3g_micro) vs L7 — 全面对比 + L8 版 SL/TP/Trail 变体 (并行版)
========================================================================
L8 "超紧追踪" 模拟盘 40 笔 WR=90% PnL=+$120.94, 用户认为优于 L7.
本实验:
  1. L7 BASELINE vs L8 BASELINE 全样本 + K-Fold head-to-head
  2. 在 L8 基线上跑 R38 同款 A/B 方向, 看 L8 是否也稳健
  3. L8 + TATrail / MH=8 / 全混合 — 看哪个组合最强

测试矩阵 (9 变体):
  L7_BASELINE:      当前实盘 L7
  L8_BASELINE:      超紧追踪 (UT3g_micro + ADX14 + MH20 + no TATrail)
  L8_A1_SL3_TP6:    L8 + SL=3.0, TP=6.0
  L8_A3_SL2.5_TP5:  L8 + SL=2.5, TP=5.0
  L8_B1_trail_1.5x: L8 + trail_act +50%
  L8_AB:            L8 + SL=3.0/TP=6.0 + trail_act +50%
  L8_TATrail:       L8 + TATrail ON
  L8_MH8:           L8 + MH=8
  L8_hybrid_full:   L8 + TATrail + MH=8
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import indicators
from backtest.runner import (
    DataBundle, run_variant, run_kfold, run_variants_parallel,
    LIVE_PARITY_KWARGS, sanitize_for_json
)

OUT_DIR = Path("results/round39_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SL_MAX = 150


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


def get_l8_base():
    kw = {**LIVE_PARITY_KWARGS}
    kw['keltner_adx_threshold'] = 14
    kw['regime_config'] = {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    }
    kw['time_adaptive_trail'] = False
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 20
    return kw


def build_configs():
    configs = []
    configs.append(("L7_BASELINE", get_l7_base()))
    configs.append(("L8_BASELINE", get_l8_base()))

    kw = get_l8_base(); kw['sl_atr_mult'] = 3.0; kw['tp_atr_mult'] = 6.0
    configs.append(("L8_A1_SL3_TP6", kw))

    kw = get_l8_base(); kw['sl_atr_mult'] = 2.5; kw['tp_atr_mult'] = 5.0
    configs.append(("L8_A3_SL2.5_TP5", kw))

    kw = get_l8_base()
    kw['regime_config'] = {
        'low':    {'trail_act': 0.33, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.21, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.09, 'trail_dist': 0.008},
    }
    configs.append(("L8_B1_trail_1.5x", kw))

    kw = get_l8_base()
    kw['sl_atr_mult'] = 3.0; kw['tp_atr_mult'] = 6.0
    kw['regime_config'] = {
        'low':    {'trail_act': 0.33, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.21, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.09, 'trail_dist': 0.008},
    }
    configs.append(("L8_AB_SL3_trail1.5x", kw))

    kw = get_l8_base()
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    configs.append(("L8_TATrail", kw))

    kw = get_l8_base(); kw['keltner_max_hold_m15'] = 8
    configs.append(("L8_MH8", kw))

    kw = get_l8_base()
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['keltner_max_hold_m15'] = 8
    configs.append(("L8_hybrid_full", kw))

    return configs


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R39_l8_vs_l7.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print("# R39: L8 (UT3g_micro) vs L7 -- Full Comparison (PARALLEL)")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    configs = build_configs()

    # Patch ATR_SL_MAX globally (all variants use 150)
    indicators.ATR_SL_MAX = SL_MAX

    data = DataBundle.load_default()

    # ══════════════════════════════════════════════════════════════
    # 1. Full Sample — PARALLEL
    # ══════════════════════════════════════════════════════════════
    print("=" * 110)
    print("1. Full-Sample Comparison -- PARALLEL (2015-01-01 ~ latest)")
    print("=" * 110)

    variants = [{'label': label, **kw} for label, kw in configs]
    par_results = run_variants_parallel(data, variants)

    full_results = {}
    header = (f"  {'Label':<24}{'ADX':>5}{'MH':>4}{'TAT':>5}{'SL':>5}{'TP':>5}"
              f"{'TrailN':>7}{'N':>6}{'Sharpe':>8}{'PnL':>10}"
              f"{'WR':>6}{'MaxDD':>9}{'AvgTrade':>10}")
    print(header)
    print("  " + "-" * 100)

    for (label, kw), s in zip(configs, par_results):
        full_results[label] = s
        avg_trade = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
        trail_n = kw['regime_config']['normal']['trail_act']
        adx = kw.get('keltner_adx_threshold', 18)
        mh = kw.get('keltner_max_hold_m15', 20)
        tat = "ON" if kw.get('time_adaptive_trail', False) else "OFF"
        print(f"  {label:<24}{adx:>5}{mh:>4}{tat:>5}"
              f"{kw['sl_atr_mult']:>5.1f}{kw['tp_atr_mult']:>5.1f}"
              f"{trail_n:>7.3f}{s['n']:>6}{s['sharpe']:>8.2f}${s['total_pnl']:>9.0f}"
              f"{s['win_rate']:>5.1f}%${s['max_dd']:>8.0f}${avg_trade:>9.2f}")
    out.flush()
    t_full = time.time() - t0
    print(f"\n  Full-sample elapsed: {t_full/60:.1f} min")

    # ══════════════════════════════════════════════════════════════
    # 2. K-Fold — each config's 6 folds run in parallel
    # ══════════════════════════════════════════════════════════════
    t_kf0 = time.time()
    print(f"\n{'='*110}")
    print("2. 6-Fold K-Fold Cross Validation (folds parallel per config)")
    print("=" * 110)

    kfold_summary = {}
    all_kf_results = {}
    for label, kw in configs:
        trail_n = kw['regime_config']['normal']['trail_act']
        adx = kw.get('keltner_adx_threshold', 18)
        tat = "+TAT" if kw.get('time_adaptive_trail', False) else ""
        print(f"\n--- {label} (ADX={adx}, trail_norm={trail_n:.3f}{tat}) ---")
        kf = run_kfold(data, kw, n_folds=6, label_prefix=f"{label}_", parallel=True)

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

    t_kf = time.time() - t_kf0
    print(f"\n  K-Fold total elapsed: {t_kf/60:.1f} min")

    # ══════════════════════════════════════════════════════════════
    # 3. Decision Matrix
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("3. Decision Matrix")
    print("=" * 110)
    print(f"\n  {'Label':<24}{'Full Sharpe':>12}{'KF pos':>8}{'KF mean':>10}"
          f"{'KF std':>9}{'KF min':>9}{'Full PnL':>10}{'Verdict':>18}")
    print("  " + "-" * 105)

    l7_sharpe = full_results['L7_BASELINE']['sharpe']
    l7_maxdd = full_results['L7_BASELINE']['max_dd']

    verdicts = {}
    for label, kw in configs:
        full_s = full_results[label]['sharpe']
        full_dd = full_results[label]['max_dd']
        ks = kfold_summary[label]
        issues = []
        if label != 'L7_BASELINE' and full_s < l7_sharpe:
            issues.append("Sharpe<L7")
        if full_s < 1.0:
            issues.append("Sharpe<1")
        if ks['pos'] < ks['total']:
            issues.append(f"KF{ks['pos']}/{ks['total']}")
        if ks['std'] > 1.5:
            issues.append("KFstd>1.5")
        if ks['min'] < 0:
            issues.append("KFmin<0")
        if label != 'L7_BASELINE' and abs(full_dd) > abs(l7_maxdd) * 1.3:
            issues.append("MaxDD>130%")
        verdict = "PASS" if not issues else ",".join(issues)
        verdicts[label] = verdict
        print(f"  {label:<24}{full_s:>12.2f}{ks['pos']:>4}/{ks['total']:<3}"
              f"{ks['mean']:>10.2f}{ks['std']:>9.2f}{ks['min']:>9.2f}"
              f"${full_results[label]['total_pnl']:>9.0f}{verdict:>18}")

    # ══════════════════════════════════════════════════════════════
    # 4. L7 vs L8 Head-to-Head
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("4. L7 vs L8 Head-to-Head (core comparison)")
    print("=" * 110)
    core_labels = ["L7_BASELINE", "L8_BASELINE", "L8_TATrail", "L8_MH8", "L8_hybrid_full"]
    print(f"\n  {'Label':<24}{'Sharpe':>8}{'KF mean':>10}{'N':>6}{'PnL':>10}{'WR':>6}"
          f"{'AvgTrade':>10}{'MaxDD':>9}{'Verdict':>12}")
    for lbl in core_labels:
        s = full_results[lbl]
        ks = kfold_summary[lbl]
        avg_t = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
        print(f"  {lbl:<24}{s['sharpe']:>8.2f}{ks['mean']:>10.2f}"
              f"{s['n']:>6}${s['total_pnl']:>9.0f}{s['win_rate']:>5.1f}%${avg_t:>9.2f}"
              f"${s['max_dd']:>8.0f}{'  '+verdicts[lbl]:>12}")

    # ══════════════════════════════════════════════════════════════
    # 5. L8 Direction A/B Variants
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("5. L8 Direction A/B Variants")
    print("=" * 110)
    variant_labels = ["L8_BASELINE", "L8_A1_SL3_TP6", "L8_A3_SL2.5_TP5",
                      "L8_B1_trail_1.5x", "L8_AB_SL3_trail1.5x"]
    print(f"\n  {'Label':<24}{'Sharpe':>8}{'KF mean':>10}{'N':>6}{'PnL':>10}{'WR':>6}"
          f"{'AvgTrade':>10}{'Verdict':>12}")
    for lbl in variant_labels:
        s = full_results[lbl]
        ks = kfold_summary[lbl]
        avg_t = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
        print(f"  {lbl:<24}{s['sharpe']:>8.2f}{ks['mean']:>10.2f}"
              f"{s['n']:>6}${s['total_pnl']:>9.0f}{s['win_rate']:>5.1f}%${avg_t:>9.2f}"
              f"{'  '+verdicts[lbl]:>12}")

    # ══════════════════════════════════════════════════════════════
    # 6. Fold-by-Fold L7 vs L8
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("6. Fold-by-Fold: L7 vs L8 vs L8 Hybrids")
    print("=" * 110)
    key_labels = ["L7_BASELINE", "L8_BASELINE", "L8_TATrail", "L8_MH8", "L8_hybrid_full"]
    fold_names = [f"Fold{i}" for i in range(1, 7)]
    print(f"  {'Fold':<8}", end="")
    for lbl in key_labels:
        short = lbl.replace("_BASELINE", "").replace("_hybrid_full", "_hyb")
        print(f"{short:>16}", end="")
    print(f"{'Best':>12}")
    for fn in fold_names:
        print(f"  {fn:<8}", end="")
        fold_sharpes = {}
        for lbl in key_labels:
            for r in all_kf_results[lbl]:
                if r.get('fold') == fn:
                    fold_sharpes[lbl] = r['sharpe']
                    print(f"{r['sharpe']:>16.2f}", end="")
                    break
        best = max(fold_sharpes, key=fold_sharpes.get) if fold_sharpes else "?"
        short = best.replace("_BASELINE", "").replace("_hybrid_full", "_hyb")
        print(f"{short:>12}")

    # ══════════════════════════════════════════════════════════════
    # 7. Recommendation
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print("7. Recommendation")
    print("=" * 110)

    passing = [(l, full_results[l]['sharpe'], kfold_summary[l]['mean'])
               for l, _ in configs
               if verdicts[l] == "PASS" and l != "L7_BASELINE"]

    l7_kf_mean = kfold_summary['L7_BASELINE']['mean']
    if passing:
        passing.sort(key=lambda x: (-x[2], -x[1]))
        print("\n  Configs that PASS all criteria vs L7 (sorted by KF mean Sharpe):")
        for p in passing:
            delta_sharpe = p[1] - l7_sharpe
            delta_kf = p[2] - l7_kf_mean
            print(f"    {p[0]:<24}  full={p[1]:.2f} (delta={delta_sharpe:+.2f})  "
                  f"KF_mean={p[2]:.2f} (delta={delta_kf:+.2f})")
        winner = passing[0][0]
        print(f"\n  >>> BEST L8 VARIANT: {winner}")
        if passing[0][2] > l7_kf_mean:
            print(f"      {winner} KF mean ({passing[0][2]:.2f}) > L7 ({l7_kf_mean:.2f})")
            print(f"      Next step: paper trade >= 30 trades with these exact params")
        else:
            print(f"      However, L7 KF mean ({l7_kf_mean:.2f}) "
                  f"still >= all L8 variants. L7 remains optimal.")
    else:
        print("\n  >>> No L8 config passes all criteria vs L7.")
        print("      L7 BASELINE remains optimal. No change recommended.")

    # ── Save JSON ──
    json_out = {
        "experiment": "R39",
        "timestamp": datetime.now().isoformat(),
        "l7_sharpe": l7_sharpe,
        "full_results": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer, float, int)) else vv
                              for kk, vv in v.items() if not kk.startswith('_')}
                         for k, v in full_results.items()},
        "kfold_summary": kfold_summary,
        "verdicts": verdicts,
    }
    json_path = OUT_DIR / "R39_results.json"
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
