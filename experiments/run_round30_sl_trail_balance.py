"""
R30: ATR_SL_MAX × Trailing × TATrail Floor 联动验证
=====================================================
背景:
  实盘 L7 部署后 16 笔 (4-22 至 4-24):
    胜率 75% (12 赢 4 亏)
    平均赢 $16.80, 平均亏 $60.02 (最大 $153)
    实际盈亏比 0.28 (设计 2.29)
    EV = -$2.40 / 笔 (负期望)
    净盈亏 -$38.43

  根因诊断:
    1. live 仓 ATR_SL_MAX 从 50→150 (4-23 修复 bug)
    2. 但 trail_dist 0.04 + tatrail_floor 0.003 没同步调整
    3. 形成 "赢 $18 / 亏 $150" 的失衡结构

  另一个隐藏问题:
    R28 L7 K-Fold 6/6 PASS 是在 ATR_SL_MAX=50 (bug) 下跑的
    实盘修复后回测口径已不一致, 必须重新验证

测试矩阵:
  A. Baseline (R28 旧)        : SL_MAX=50,  trail=0.04, floor=0.003
  B. 当前实盘 (SL fixed only) : SL_MAX=150, trail=0.04, floor=0.003
  C. trail wider               : SL_MAX=150, trail=0.08, floor=0.003
  D. trail wider + floor 提高  : SL_MAX=150, trail=0.08, floor=0.010 (= paper L7-TrailB)

  其他参数全部锁定 = 实盘 L7:
    sl_atr_mult=3.5, tp_atr_mult=8.0, ADX threshold=18,
    intraday_adaptive=True, choppy_threshold=0.50,
    rsi_adx_filter=40, max_positions=1, max_hold_m15=8,
    tatrail_start=2, tatrail_decay=0.75, min_entry_gap_hours=1.0,
    low/high regime trail 不变 (0.40/0.10, 0.12/0.02)

判定标准 (任何配置要替换实盘必须满足):
  1. 全样本 Sharpe ≥ 当前实盘配置 (B) Sharpe 且 ≥ 1.0
  2. 6-Fold K-Fold 全正 (6/6)
  3. K-Fold Sharpe std ≤ 1.5 (稳定性)
  4. 平均亏损单 ≤ 全样本平均赢单的 2 倍 (盈亏比基本合理)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from datetime import datetime

import indicators
from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS


OUT_DIR = Path("results/round30_results")
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
    """L7 实盘所有锁定参数 (regime/trail/floor 由调用方按场景覆盖)"""
    kw = {**LIVE_PARITY_KWARGS}
    kw['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},  # 占位, 下面覆盖
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003  # 占位, 下面覆盖
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 8
    return kw


def build_config(label, sl_max, trail_normal_dist, tatrail_floor):
    """构造一个 R30 配置, 同时返回需要 monkey-patch 的 SL_MAX"""
    kw = get_l7_base()
    kw['regime_config']['normal'] = {
        'trail_act': 0.20,
        'trail_dist': trail_normal_dist,
    }
    kw['time_adaptive_trail_floor'] = tatrail_floor
    return label, sl_max, kw


def run_with_sl_max(data, label, sl_max, kw, run_fn=run_variant, **extra):
    """临时 patch indicators.ATR_SL_MAX 后跑一次, 跑完恢复"""
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
    out_path = OUT_DIR / "R30_sl_trail_balance.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print("# R30: ATR_SL_MAX x Trailing x TATrail Floor 联动验证")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 目的: 验证实盘 L7 16 笔负期望是否由 SL_MAX 修复未同步 trail 引起")
    print()

    configs = [
        build_config("A_R28_baseline",   sl_max=50,  trail_normal_dist=0.04, tatrail_floor=0.003),
        build_config("B_current_live",   sl_max=150, trail_normal_dist=0.04, tatrail_floor=0.003),
        build_config("C_trail_wider",    sl_max=150, trail_normal_dist=0.08, tatrail_floor=0.003),
        build_config("D_paper_trailB",   sl_max=150, trail_normal_dist=0.08, tatrail_floor=0.010),
    ]

    data = DataBundle.load_default()

    # ── 1. 全样本对比 ──
    print("=" * 80)
    print("1. Full-Sample Comparison (2015-01-01 ~ 2026-04-10)")
    print("=" * 80)
    print(f"\n  {'Label':<18}{'SL_MAX':>8}{'TrailD':>8}{'Floor':>7}"
          f"{'N':>6}{'Sharpe':>8}{'PnL':>10}{'WR':>6}{'MaxDD':>9}{'AvgWin':>8}{'AvgLoss':>9}")
    print("  " + "-" * 100)

    full_results = {}
    for label, sl_max, kw in configs:
        s = run_with_sl_max(data, label, sl_max, kw, verbose=False)
        full_results[label] = s
        avg_win = s.get('avg_win', 0) or 0
        avg_loss = s.get('avg_loss', 0) or 0
        trail_d = kw['regime_config']['normal']['trail_dist']
        floor = kw['time_adaptive_trail_floor']
        print(f"  {label:<18}{sl_max:>8}{trail_d:>8.2f}{floor:>7.3f}"
              f"{s['n']:>6}{s['sharpe']:>8.2f}${s['total_pnl']:>9.0f}"
              f"{s['win_rate']:>5.1f}%${s['max_dd']:>8.0f}"
              f"${avg_win:>7.1f}${avg_loss:>8.1f}")
    out.flush()

    # ── 2. 每个配置跑 6-Fold K-Fold ──
    print(f"\n{'='*80}")
    print("2. 6-Fold K-Fold for each config")
    print("=" * 80)

    kfold_summary = {}
    for label, sl_max, kw in configs:
        print(f"\n--- {label} (SL_MAX={sl_max}, "
              f"trail_normal_dist={kw['regime_config']['normal']['trail_dist']}, "
              f"floor={kw['time_adaptive_trail_floor']}) ---")
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
        print(f"  K-Fold: {pos}/{len(sharpes)} pos, mean={np.mean(sharpes):.2f}, "
              f"std={np.std(sharpes):.2f}, min={np.min(sharpes):.2f}")

    # ── 3. 终判 ──
    print(f"\n{'='*80}")
    print("3. Decision Matrix")
    print("=" * 80)
    print(f"\n  {'Label':<18}{'Full Sharpe':>12}{'KF pos':>8}{'KF mean':>10}"
          f"{'KF std':>9}{'KF min':>9}{'Verdict':>16}")
    print("  " + "-" * 90)

    baseline_b = full_results.get('B_current_live', {}).get('sharpe', 0)
    for label, sl_max, kw in configs:
        full_s = full_results[label]['sharpe']
        ks = kfold_summary[label]
        verdict = []
        if full_s < baseline_b:   verdict.append("Sharpe<B")
        if full_s < 1.0:           verdict.append("Sharpe<1")
        if ks['pos'] < ks['total']: verdict.append(f"KF{ks['pos']}/{ks['total']}")
        if ks['std'] > 1.5:        verdict.append("KFstd>1.5")
        if ks['min'] < 0:          verdict.append("KFmin<0")
        verdict_str = "PASS" if not verdict else ",".join(verdict)
        print(f"  {label:<18}{full_s:>12.2f}{ks['pos']:>4}/{ks['total']:<3}"
              f"{ks['mean']:>10.2f}{ks['std']:>9.2f}{ks['min']:>9.2f}{verdict_str:>16}")

    # ── 4. 推荐 ──
    print(f"\n{'='*80}")
    print("4. 推荐部署 (按判定标准)")
    print("=" * 80)
    candidates = []
    for label, sl_max, kw in configs:
        if label.startswith('A'):  # baseline 已下线
            continue
        full_s = full_results[label]['sharpe']
        ks = kfold_summary[label]
        if (full_s >= baseline_b and full_s >= 1.0 and
                ks['pos'] == ks['total'] and ks['std'] <= 1.5 and ks['min'] >= 0):
            candidates.append((label, full_s, ks['mean'], ks['std']))

    if candidates:
        candidates.sort(key=lambda x: (-x[1], x[3]))
        print("\n  通过判定标准的配置 (按 full Sharpe 降序):")
        for c in candidates:
            print(f"    {c[0]:<18}  full Sharpe={c[1]:.2f}  KF mean={c[2]:.2f}  std={c[3]:.2f}")
        print(f"\n  >>> 推荐: {candidates[0][0]}")
        print(f"      下一步: paper trade 满 30 笔后才能进入 live 部署")
    else:
        print("\n  >>> 无配置完全通过判定标准, 全部需要进一步研究")
        print("      建议: 检查 16 笔实盘是否样本异常 (单一 regime?)")

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")
    print(f"# Result file: {out_path}")

    sys.stdout = old_stdout
    out.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
