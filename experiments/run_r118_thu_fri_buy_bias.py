#!/usr/bin/env python3
"""
R118 — Thu/Fri UTC 16-22 BUY Bias Test
========================================
Tests the hypothesis: "Thursday/Friday + UTC 16-22 has a strong BUY bias (85% WR)"

Phase 1: Statistical validation — actual WR and avg return by DOW × UTC hour
Phase 2: Simple BUY strategy — enter at each hour in [16..22], exit N hours later
Phase 3: Entry/exit grid — find optimal entry→exit within and around the window
Phase 4: Comparison with UTC 12-14 (the known best window from R99)
Phase 5: K-Fold validation (5 folds, 2015-2026)

Uses 11-year Dukascopy H1 data.
"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r118_thu_fri_buy_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
t0 = time.time()

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]


def sharpe(arr, ann=252):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(ann)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def metrics(pnl_arr):
    if len(pnl_arr) < 5:
        return {'n': len(pnl_arr), 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg': 0}
    wins = (pnl_arr > 0).sum()
    return {
        'n': len(pnl_arr), 'sharpe': round(sharpe(pnl_arr), 3),
        'pnl': round(float(pnl_arr.sum()), 2), 'max_dd': round(max_dd(pnl_arr), 2),
        'wr': round(wins / len(pnl_arr) * 100, 1),
        'avg': round(float(pnl_arr.mean()), 4),
    }


def load_h1():
    df = pd.read_csv(DATA_DIR / "download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]

    tr = pd.concat([df['High'] - df['Low'],
                     (df['High'] - df['Close'].shift()).abs(),
                     (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()

    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['date'] = df.index.date

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    return df.dropna(subset=['ATR14'])


def bt_buy_hold(data, entry_hour, hold_hours, dow_filter, sl_atr=2.0, tp_atr=0):
    """BUY at entry_hour close, hold for hold_hours, with ATR-based SL.
    Vectorized entry detection + numpy array exit scanning for speed."""
    close_arr = data['Close'].values.astype(np.float64)
    low_arr = data['Low'].values.astype(np.float64)
    high_arr = data['High'].values.astype(np.float64)
    atr_arr = data['ATR14'].values.astype(np.float64)
    hour_arr = data['hour'].values
    dow_arr = data['dow'].values
    n = len(data)

    dow_set = set(dow_filter)
    entry_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if hour_arr[i] == entry_hour and dow_arr[i] in dow_set and not np.isnan(atr_arr[i]) and atr_arr[i] >= 0.1:
            entry_mask[i] = True

    entry_indices = np.where(entry_mask)[0]
    trades = []

    for idx in entry_indices:
        atr = atr_arr[idx]
        entry_price = close_arr[idx] + SPREAD / 2
        sl_price = entry_price - atr * sl_atr
        tp_price = entry_price + atr * tp_atr if tp_atr > 0 else 1e9

        exit_found = False
        end_idx = min(idx + hold_hours, n - 1)

        for j in range(idx + 1, end_idx + 1):
            if low_arr[j] <= sl_price:
                pnl = -(atr * sl_atr) * UNIT_LOT * PV
                trades.append({'pnl': pnl, 'reason': 'SL'})
                exit_found = True
                break
            if tp_atr > 0 and high_arr[j] >= tp_price:
                pnl = (atr * tp_atr) * UNIT_LOT * PV
                trades.append({'pnl': pnl, 'reason': 'TP'})
                exit_found = True
                break

        if not exit_found:
            exit_price = close_arr[end_idx]
            pnl = (exit_price - entry_price - SPREAD) * UNIT_LOT * PV
            trades.append({'pnl': pnl, 'reason': 'Time'})

    return trades


def main():
    print("=" * 80)
    print("  R118 — Thu/Fri UTC 16-22 BUY Bias Test")
    print("  Hypothesis: 'Thu/Fri + UTC 16-22 = STRONG BUY BIAS, 85% WR'")
    print("=" * 80)

    h1 = load_h1()
    print(f"  H1 data: {len(h1)} bars ({h1.index[0]} ~ {h1.index[-1]})")
    all_results = {}

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Statistical Validation — DOW × Hour Return Matrix
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 1: Raw 1-Hour Forward Return by DOW × UTC Hour")
    print("  (BUY at Close of hour H, measure return at Close of hour H+1)")
    print("=" * 70)

    h1['fwd_ret'] = h1['Close'].shift(-1) / h1['Close'] - 1
    h1['fwd_pnl'] = (h1['Close'].shift(-1) - h1['Close'] - SPREAD) * UNIT_LOT * PV

    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    target_hours = list(range(16, 23))

    print(f"\n  {'DOW':>5s}", end="")
    for hr in target_hours:
        print(f"  UTC{hr:02d}", end="")
    print(f"  | UTC12  UTC13  UTC14")
    print("  " + "-" * 85)

    phase1 = {}
    for dow in range(5):
        row_str = f"  {dow_names[dow]:>5s}"
        for hr in target_hours + [12, 13, 14]:
            mask = (h1['dow'] == dow) & (h1['hour'] == hr) & h1['fwd_ret'].notna()
            subset = h1[mask]
            if len(subset) < 20:
                row_str += "    n/a"
                continue
            wr = (subset['fwd_pnl'] > 0).sum() / len(subset) * 100
            avg_ret = subset['fwd_ret'].mean() * 100
            key = f"{dow_names[dow]}_UTC{hr}"
            phase1[key] = {
                'n': len(subset), 'wr': round(wr, 1),
                'avg_ret_pct': round(avg_ret, 4),
                'avg_pnl': round(float(subset['fwd_pnl'].mean()), 4),
                'total_pnl': round(float(subset['fwd_pnl'].sum()), 2),
            }
            row_str += f"  {wr:5.1f}%"
            if hr == 14:
                row_str = row_str  # already printed
        print(row_str)

    # Thu+Fri combined for UTC 16-22
    print(f"\n  Thu+Fri combined:")
    print(f"  {'Hour':>6s}  {'N':>6s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'AvgRet%':>8s}")
    for hr in target_hours:
        mask = (h1['dow'].isin([3, 4])) & (h1['hour'] == hr) & h1['fwd_ret'].notna()
        subset = h1[mask]
        if len(subset) < 20:
            continue
        wr = (subset['fwd_pnl'] > 0).sum() / len(subset) * 100
        avg_ret = subset['fwd_ret'].mean() * 100
        avg_pnl = subset['fwd_pnl'].mean()
        total = subset['fwd_pnl'].sum()
        print(f"  UTC{hr:02d}  {len(subset):6d}  {wr:5.1f}%  {avg_pnl:8.4f}  ${total:>9.2f}  {avg_ret:7.4f}%")
        phase1[f"ThuFri_UTC{hr}"] = {
            'n': len(subset), 'wr': round(wr, 1),
            'avg_pnl': round(avg_pnl, 4), 'total_pnl': round(total, 2),
        }

    # Comparison: UTC 12-14 (known best window)
    print(f"\n  Comparison — UTC 12-14 (known best from R99):")
    for hr in [12, 13, 14]:
        mask = (h1['dow'].isin([3, 4])) & (h1['hour'] == hr) & h1['fwd_ret'].notna()
        subset = h1[mask]
        if len(subset) < 20:
            continue
        wr = (subset['fwd_pnl'] > 0).sum() / len(subset) * 100
        avg_pnl = subset['fwd_pnl'].mean()
        total = subset['fwd_pnl'].sum()
        print(f"  UTC{hr:02d}  {len(subset):6d}  {wr:5.1f}%  {avg_pnl:8.4f}  ${total:>9.2f}")

    # All days combined for UTC 16-22
    print(f"\n  All days combined (Mon-Fri) for UTC 16-22:")
    for hr in target_hours:
        mask = (h1['dow'].isin([0, 1, 2, 3, 4])) & (h1['hour'] == hr) & h1['fwd_ret'].notna()
        subset = h1[mask]
        if len(subset) < 20:
            continue
        wr = (subset['fwd_pnl'] > 0).sum() / len(subset) * 100
        avg_pnl = subset['fwd_pnl'].mean()
        total = subset['fwd_pnl'].sum()
        print(f"  UTC{hr:02d}  {len(subset):6d}  {wr:5.1f}%  {avg_pnl:8.4f}  ${total:>9.2f}")

    all_results['phase1_raw_returns'] = phase1

    # ════════════════════════════════════════════════════════════════
    # Phase 2: BUY Strategy Backtest — Entry at UTC 16-22, Various Hold Periods
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 2: BUY Strategy — Entry at UTC 16-22, Hold 1-12 Hours")
    print("  (Thu+Fri only, BUY at entry_hour close, SL=2×ATR)")
    print("=" * 70)

    configs = []
    for entry_h in [16, 17, 18, 19, 20, 21, 22]:
        for hold in [1, 2, 4, 6, 8, 12]:
            configs.append((f"E{entry_h}_H{hold}", entry_h, hold, [3, 4], 2.0, 0))
    # Add TP variants for promising entries
    for entry_h in [16, 17, 18]:
        configs.append((f"E{entry_h}_H8_TP2", entry_h, 8, [3, 4], 2.0, 2.0))
        configs.append((f"E{entry_h}_H12_TP3", entry_h, 12, [3, 4], 2.0, 3.0))

    phase2 = {}
    print(f"\n  {'Config':<20s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}  {'Avg':>8s}")
    print("  " + "-" * 75)

    best_sharpe = -999
    best_config = None

    for label, eh, hold, dows, sl, tp in configs:
        trades = bt_buy_hold(h1, entry_hour=eh, hold_hours=hold, dow_filter=dows,
                             sl_atr=sl, tp_atr=tp)
        if not trades:
            continue
        pnls = np.array([t['pnl'] for t in trades])
        m = metrics(pnls)
        if m['sharpe'] > 0.1:  # only show interesting ones
            print(f"  {label:<20s}  {m['n']:5d}  {m['sharpe']:7.3f}  ${m['pnl']:>9.0f}  {m['wr']:5.1f}%  ${m['max_dd']:>7.0f}  {m['avg']:8.4f}")
        phase2[label] = m
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_config = label

    # Top 10 by Sharpe
    top10 = sorted(phase2.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:10]
    print(f"\n  Top 10 Configs by Sharpe:")
    print(f"  {'Config':<20s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}")
    for label, m in top10:
        print(f"  {label:<20s}  {m['n']:5d}  {m['sharpe']:7.3f}  ${m['pnl']:>9.0f}  {m['wr']:5.1f}%  ${m['max_dd']:>7.0f}")

    all_results['phase2_backtest'] = phase2
    all_results['phase2_best'] = best_config

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Comparison — UTC 12-14 BUY vs UTC 16-22 BUY
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 3: Head-to-Head — UTC 12-14 vs UTC 16-22 (Thu+Fri BUY)")
    print("=" * 70)

    phase3 = {}
    compare_configs = [
        ("UTC12_H4_ThuFri", 12, 4, [3, 4], 2.0, 0),
        ("UTC12_H8_ThuFri", 12, 8, [3, 4], 2.0, 0),
        ("UTC13_H4_ThuFri", 13, 4, [3, 4], 2.0, 0),
        ("UTC13_H8_ThuFri", 13, 8, [3, 4], 2.0, 0),
        ("UTC14_H4_ThuFri", 14, 4, [3, 4], 2.0, 0),
        ("UTC16_H4_ThuFri", 16, 4, [3, 4], 2.0, 0),
        ("UTC16_H8_ThuFri", 16, 8, [3, 4], 2.0, 0),
        ("UTC17_H4_ThuFri", 17, 4, [3, 4], 2.0, 0),
        ("UTC18_H8_ThuFri", 18, 8, [3, 4], 2.0, 0),
        ("UTC20_H8_ThuFri", 20, 8, [3, 4], 2.0, 0),
        # All days
        ("UTC12_H4_AllDays", 12, 4, [0, 1, 2, 3, 4], 2.0, 0),
        ("UTC13_H4_AllDays", 13, 4, [0, 1, 2, 3, 4], 2.0, 0),
        ("UTC16_H4_AllDays", 16, 4, [0, 1, 2, 3, 4], 2.0, 0),
        ("UTC17_H4_AllDays", 17, 4, [0, 1, 2, 3, 4], 2.0, 0),
        ("UTC18_H8_AllDays", 18, 8, [0, 1, 2, 3, 4], 2.0, 0),
    ]

    print(f"\n  {'Config':<25s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}")
    print("  " + "-" * 70)
    for label, eh, hold, dows, sl, tp in compare_configs:
        trades = bt_buy_hold(h1, entry_hour=eh, hold_hours=hold, dow_filter=dows,
                             sl_atr=sl, tp_atr=tp)
        if not trades:
            continue
        pnls = np.array([t['pnl'] for t in trades])
        m = metrics(pnls)
        print(f"  {label:<25s}  {m['n']:5d}  {m['sharpe']:7.3f}  ${m['pnl']:>9.0f}  {m['wr']:5.1f}%  ${m['max_dd']:>7.0f}")
        phase3[label] = m

    all_results['phase3_comparison'] = phase3

    # ════════════════════════════════════════════════════════════════
    # Phase 4: Multi-Hour Window — Enter at 16, Exit at 22 (or next day)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 4: Window Strategy — Enter UTC 16, Exit at UTC 22 / Next-Day")
    print("  (Hold through the entire 16-22 window)")
    print("=" * 70)

    window_configs = [
        ("W16_22_ThuFri", 16, 22, [3, 4], 2.0, 0),
        ("W16_22_ThuFri_TP2", 16, 22, [3, 4], 2.0, 2.0),
        ("W16_00_ThuFri", 16, 8, [3, 4], 2.0, 0),       # overnight to midnight+8
        ("W17_22_ThuFri", 17, 22, [3, 4], 2.0, 0),
        ("W18_22_ThuFri", 18, 22, [3, 4], 2.0, 0),
        ("W16_22_AllDays", 16, 22, [0, 1, 2, 3, 4], 2.0, 0),
        # SL variants
        ("W16_22_ThuFri_SL3", 16, 22, [3, 4], 3.0, 0),
        ("W16_22_ThuFri_SL1", 16, 22, [3, 4], 1.5, 0),
    ]

    phase4 = {}
    print(f"\n  {'Config':<25s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}")
    print("  " + "-" * 70)

    for label, eh, hold_or_exit, dows, sl, tp in window_configs:
        hold = hold_or_exit if hold_or_exit > 12 else hold_or_exit
        if hold_or_exit == 22 and eh < 22:
            hold = 22 - eh
        trades = bt_buy_hold(h1, entry_hour=eh, hold_hours=hold, dow_filter=dows,
                             sl_atr=sl, tp_atr=tp)
        if not trades:
            continue
        pnls = np.array([t['pnl'] for t in trades])
        m = metrics(pnls)
        print(f"  {label:<25s}  {m['n']:5d}  {m['sharpe']:7.3f}  ${m['pnl']:>9.0f}  {m['wr']:5.1f}%  ${m['max_dd']:>7.0f}")
        phase4[label] = m

    all_results['phase4_window'] = phase4

    # ════════════════════════════════════════════════════════════════
    # Phase 5: K-Fold Validation on Top Configs
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 5: K-Fold Validation (5 folds)")
    print("=" * 70)

    # Pick top configs from Phase 2 + Phase 3 + Phase 4
    kf_configs = [
        ("E16_H4_ThuFri", 16, 4, [3, 4], 2.0, 0),
        ("E16_H8_ThuFri", 16, 8, [3, 4], 2.0, 0),
        ("E17_H4_ThuFri", 17, 4, [3, 4], 2.0, 0),
        ("E18_H8_ThuFri", 18, 8, [3, 4], 2.0, 0),
        ("E20_H8_ThuFri", 20, 8, [3, 4], 2.0, 0),
        ("E12_H4_ThuFri", 12, 4, [3, 4], 2.0, 0),
        ("E13_H4_ThuFri", 13, 4, [3, 4], 2.0, 0),
        ("E16_H4_AllDays", 16, 4, [0, 1, 2, 3, 4], 2.0, 0),
        ("E13_H4_AllDays", 13, 4, [0, 1, 2, 3, 4], 2.0, 0),
    ]

    kf_results = {}
    for label, eh, hold, dows, sl, tp in kf_configs:
        fold_sharpes = []
        fold_pnls = []
        fold_trades = []
        fold_wrs = []
        for fname, start, end in FOLDS:
            sub = h1[(h1.index >= start) & (h1.index < end)]
            trades = bt_buy_hold(sub, entry_hour=eh, hold_hours=hold, dow_filter=dows,
                                 sl_atr=sl, tp_atr=tp)
            if not trades:
                fold_sharpes.append(0.0); fold_pnls.append(0.0)
                fold_trades.append(0); fold_wrs.append(0.0)
                continue
            pnls = np.array([t['pnl'] for t in trades])
            fold_sharpes.append(round(sharpe(pnls), 3))
            fold_pnls.append(round(pnls.sum(), 2))
            fold_trades.append(len(trades))
            fold_wrs.append(round((pnls > 0).sum() / len(pnls) * 100, 1))

        pos = sum(1 for s in fold_sharpes if s > 0)
        status = "PASS" if pos >= 3 else "FAIL"
        mean_s = round(np.mean(fold_sharpes), 3)
        mean_wr = round(np.mean(fold_wrs), 1)
        print(f"  {label:<25s}: sharpes={fold_sharpes}")
        print(f"  {'':25s}  WRs={fold_wrs}")
        print(f"  {'':25s}  -> {pos}/5 [{status}] mean_sharpe={mean_s} mean_wr={mean_wr}%")
        print()

        kf_results[label] = {
            'fold_sharpes': fold_sharpes, 'fold_pnls': fold_pnls,
            'fold_trades': fold_trades, 'fold_wrs': fold_wrs,
            'positive': pos, 'mean_sharpe': mean_s, 'mean_wr': mean_wr,
            'pass': pos >= 3,
        }

    all_results['phase5_kfold'] = kf_results

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("  R118 FINAL SUMMARY")
    print("=" * 80)

    print(f"\n  Hypothesis: 'Thu/Fri + UTC 16-22 = STRONG BUY BIAS, 85% WR'")
    print(f"\n  Phase 1 — Raw 1-hour forward return WR (Thu+Fri, BUY):")
    for hr in target_hours:
        k = f"ThuFri_UTC{hr}"
        if k in phase1:
            v = phase1[k]
            verdict = "CONFIRMED" if v['wr'] > 55 else "NOT CONFIRMED"
            print(f"    UTC {hr}: WR={v['wr']}%, avg_pnl={v['avg_pnl']:.4f} [{verdict}]")

    print(f"\n  Phase 2 — Best backtest config: {best_config}")
    if best_config and best_config in phase2:
        b = phase2[best_config]
        print(f"    Sharpe={b['sharpe']}, WR={b['wr']}%, PnL=${b['pnl']:.0f}, MaxDD=${b['max_dd']:.0f}")

    print(f"\n  Phase 5 — K-Fold results:")
    for k, v in kf_results.items():
        status = "PASS" if v['pass'] else "FAIL"
        print(f"    {k:<25s}: {v['positive']}/5 [{status}] mean_sharpe={v['mean_sharpe']} mean_wr={v['mean_wr']}%")

    # Final verdict
    print(f"\n  {'='*60}")
    thu_fri_16_22_wrs = [phase1.get(f"ThuFri_UTC{h}", {}).get('wr', 0) for h in range(16, 23)]
    avg_wr = np.mean(thu_fri_16_22_wrs) if thu_fri_16_22_wrs else 0
    print(f"  VERDICT: Thu/Fri UTC 16-22 average 1h-forward BUY WR = {avg_wr:.1f}%")
    if avg_wr > 55:
        print(f"  -> Mild BUY bias exists, but far from 85%")
    elif avg_wr > 50:
        print(f"  -> Marginal bias, not tradeable on its own")
    else:
        print(f"  -> No meaningful BUY bias detected")

    kf_passes = sum(1 for v in kf_results.values()
                    if 'ThuFri' in list(kf_results.keys())[list(kf_results.values()).index(v)]
                    and '16' in list(kf_results.keys())[list(kf_results.values()).index(v)]
                    and v['pass'])
    print(f"  K-Fold validated UTC 16-18 configs: {kf_passes}")
    print(f"  {'='*60}")

    all_results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r118_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Saved: {out_file}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
