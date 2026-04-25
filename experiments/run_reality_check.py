"""
Reality Check: Backtest Credibility Diagnosis
==============================================
5-step diagnosis to quantify backtest-to-live gap:
  Step 1: Sharpe calculation fix (full calendar days vs trade-only days)
  Step 2: Spread sensitivity ($0.30 → $3.00)
  Step 3: Live trade replay (36 trades vs engine output)
  Step 4: PnL distribution comparison
  Step 5: True OOS (train 2015-2020, test 2020-2026)
"""
import sys, os, time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats, aggregate_daily_pnl
from backtest.engine import TradeRecord
import research_config as config

OUT_DIR = Path("results/reality_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def corrected_sharpe(trades, start_date=None, end_date=None):
    """Sharpe with ALL calendar trading days (non-trade days = PnL 0)."""
    if not trades:
        return 0.0, 0.0, 0, 0

    trade_daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        trade_daily[d] = trade_daily.get(d, 0) + t.pnl

    if start_date is None:
        start_date = min(trade_daily.keys())
    if end_date is None:
        end_date = max(trade_daily.keys())

    all_dates = pd.bdate_range(start_date, end_date)
    full_daily = []
    for d in all_dates:
        full_daily.append(trade_daily.get(d.date(), 0.0))

    arr = np.array(full_daily)
    n_total = len(arr)
    n_trade = len(trade_daily)

    if n_total < 2 or np.std(arr, ddof=1) <= 0:
        return 0.0, 0.0, n_trade, n_total

    original_arr = np.array(list(trade_daily.values()))
    original_sharpe = float(np.mean(original_arr) / np.std(original_arr, ddof=1) * np.sqrt(252))
    corrected = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))

    return original_sharpe, corrected, n_trade, n_total


# ═══════════════════════════════════════════════════════════════
# Step 1: Sharpe Calculation Diagnosis
# ═══════════════════════════════════════════════════════════════

def step1_sharpe_fix(data):
    print("\n" + "=" * 90)
    print("  STEP 1: Sharpe Calculation Diagnosis")
    print("  (Original: only trade days) vs (Corrected: all business days with PnL=0 fill)")
    print("=" * 90)

    result = run_variant(data, "L7_MH8_full", verbose=False, **L7_MH8, spread_cost=0.30)
    trades = result['_trades']
    print(f"\n  Total trades: {result['n']}")
    print(f"  Total PnL: ${result['total_pnl']:.2f}")

    orig_sh, corr_sh, n_trade_days, n_total_days = corrected_sharpe(trades)

    print(f"\n  --- Sharpe Comparison ---")
    print(f"  Original Sharpe (trade-days only):   {orig_sh:.2f}  ({n_trade_days} days with trades)")
    print(f"  Corrected Sharpe (all biz days):     {corr_sh:.2f}  ({n_total_days} total biz days)")
    print(f"  Ratio (corrected/original):          {corr_sh/orig_sh:.3f}" if orig_sh > 0 else "")
    print(f"  Trade-day coverage:                  {n_trade_days/n_total_days*100:.1f}%")

    if orig_sh > 0:
        theoretical_ratio = np.sqrt(n_trade_days / n_total_days) * (n_trade_days / n_total_days)
        print(f"  Theoretical deflation factor:        ~{theoretical_ratio:.3f}")

    # K-Fold with corrected Sharpe
    print(f"\n  --- K-Fold with Corrected Sharpe ---")
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]
    fold_orig = []
    fold_corr = []
    print(f"  {'Fold':<8} {'Period':<25} {'Orig Sharpe':>12} {'Corr Sharpe':>12} {'TradeDays':>10} {'TotalDays':>10}")
    print(f"  {'-'*8} {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for fname, start, end in folds:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000:
            continue
        r = run_variant(fold_data, f"S1_{fname}", verbose=False, **L7_MH8, spread_cost=0.30)
        orig, corr, ntd, ntot = corrected_sharpe(
            r['_trades'],
            pd.Timestamp(start).date(),
            pd.Timestamp(end).date()
        )
        fold_orig.append(orig)
        fold_corr.append(corr)
        print(f"  {fname:<8} {start} ~ {end:<10} {orig:>12.2f} {corr:>12.2f} {ntd:>10} {ntot:>10}")

    print(f"\n  K-Fold Summary (Original): mean={np.mean(fold_orig):.2f}, std={np.std(fold_orig):.2f}, "
          f"all_positive={all(s > 0 for s in fold_orig)}")
    print(f"  K-Fold Summary (Corrected): mean={np.mean(fold_corr):.2f}, std={np.std(fold_corr):.2f}, "
          f"all_positive={all(s > 0 for s in fold_corr)}")

    return result


# ═══════════════════════════════════════════════════════════════
# Step 2: Spread Sensitivity
# ═══════════════════════════════════════════════════════════════

def step2_spread_sensitivity(data):
    print("\n" + "=" * 90)
    print("  STEP 2: Spread + Slippage Sensitivity")
    print("  Testing spread_cost from $0.00 to $3.00")
    print("  (Engine deducts spread once at exit; entry uses next-bar Open without bid/ask offset)")
    print("=" * 90)

    spreads = [0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00]

    print(f"\n  {'Spread':>8} {'N':>6} {'OrigSharpe':>11} {'CorrSharpe':>11} {'PnL':>10} {'WR%':>6} {'AvgWin':>8} {'AvgLoss':>8} {'RR':>5} {'MaxDD':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*11} {'-'*11} {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")

    results_table = []
    for sp in spreads:
        r = run_variant(data, f"SP_{sp:.2f}", verbose=False, **L7_MH8, spread_cost=sp)
        orig, corr, ntd, ntot = corrected_sharpe(r['_trades'])
        results_table.append({
            'spread': sp, 'n': r['n'], 'orig_sharpe': orig, 'corr_sharpe': corr,
            'pnl': r['total_pnl'], 'wr': r['win_rate'], 'avg_win': r['avg_win'],
            'avg_loss': r['avg_loss'], 'rr': r['rr'], 'max_dd': r['max_dd']
        })
        print(f"  ${sp:>7.2f} {r['n']:>6} {orig:>11.2f} {corr:>11.2f} ${r['total_pnl']:>9.0f} "
              f"{r['win_rate']:>5.1f}% ${r['avg_win']:>7.2f} ${r['avg_loss']:>7.2f} {r['rr']:>4.2f} ${r['max_dd']:>7.0f}")

    # Find break-even spread
    print(f"\n  --- Break-Even Analysis ---")
    for rt in results_table:
        if rt['pnl'] <= 0:
            print(f"  Strategy becomes UNPROFITABLE at spread >= ${rt['spread']:.2f}")
            break
    else:
        print(f"  Strategy still profitable at spread = ${spreads[-1]:.2f}")

    for rt in results_table:
        if rt['corr_sharpe'] < 1.0:
            print(f"  Corrected Sharpe drops below 1.0 at spread >= ${rt['spread']:.2f}")
            break

    # Real-world spread estimate
    print(f"\n  --- Real-World Spread Estimate ---")
    print(f"  Typical XAUUSD spread: $0.20-$0.50 (normal), $1-3 (high vol)")
    print(f"  Estimated effective spread (incl slippage): $0.50-$1.50")
    if len(results_table) >= 6:
        r050 = [rt for rt in results_table if rt['spread'] == 0.50][0]
        r150 = [rt for rt in results_table if rt['spread'] == 1.50][0]
        print(f"  At $0.50: Corrected Sharpe = {r050['corr_sharpe']:.2f}")
        print(f"  At $1.50: Corrected Sharpe = {r150['corr_sharpe']:.2f}")

    return results_table


# ═══════════════════════════════════════════════════════════════
# Step 3: Live Trade Replay
# ═══════════════════════════════════════════════════════════════

def step3_live_replay(data):
    print("\n" + "=" * 90)
    print("  STEP 3: Live Trade Replay (2026-04-14 ~ 2026-04-24)")
    print("  Running engine on the exact period of 36 live trades")
    print("=" * 90)

    live_window = data.slice("2026-04-14", "2026-04-25")
    if len(live_window.m15_df) < 100:
        print("  WARNING: Not enough data for live period. Skipping Step 3.")
        return None

    r = run_variant(live_window, "LivePeriod_L7", verbose=False, **L7_MH8, spread_cost=0.30)
    trades = r['_trades']

    print(f"\n  Backtest on live period: {r['n']} trades, PnL=${r['total_pnl']:.2f}, WR={r['win_rate']:.1f}%")

    # Live trades for comparison
    live_trades = [
        {'id': 1,  'time': '2026-04-14 11:28', 'dir': 'BUY',  'lots': 0.01, 'pnl':    6.48, 'reason': 'Trailing'},
        {'id': 2,  'time': '2026-04-14 14:09', 'dir': 'BUY',  'lots': 0.01, 'pnl':   16.54, 'reason': 'Trailing'},
        {'id': 3,  'time': '2026-04-14 14:50', 'dir': 'BUY',  'lots': 0.01, 'pnl':  -15.38, 'reason': 'SL'},
        {'id': 4,  'time': '2026-04-14 22:37', 'dir': 'BUY',  'lots': 0.01, 'pnl':    7.89, 'reason': 'Trailing'},
        {'id': 5,  'time': '2026-04-14 23:19', 'dir': 'BUY',  'lots': 0.01, 'pnl':    3.35, 'reason': 'Trailing'},
        {'id': 6,  'time': '2026-04-15 00:06', 'dir': 'BUY',  'lots': 0.01, 'pnl':    4.34, 'reason': 'Trailing'},
        {'id': 7,  'time': '2026-04-15 00:40', 'dir': 'BUY',  'lots': 0.01, 'pnl':    5.33, 'reason': 'Trailing'},
        {'id': 8,  'time': '2026-04-15 02:04', 'dir': 'BUY',  'lots': 0.01, 'pnl':    2.58, 'reason': 'Trailing'},
        {'id': 9,  'time': '2026-04-15 02:58', 'dir': 'BUY',  'lots': 0.01, 'pnl':    3.32, 'reason': 'Trailing'},
        {'id': 10, 'time': '2026-04-15 08:15', 'dir': 'BUY',  'lots': 0.01, 'pnl':    4.33, 'reason': 'Trailing'},
        {'id': 11, 'time': '2026-04-15 08:19', 'dir': 'BUY',  'lots': 0.01, 'pnl':   16.03, 'reason': 'Trailing'},
        {'id': 12, 'time': '2026-04-15 12:37', 'dir': 'BUY',  'lots': 0.01, 'pnl':  -45.94, 'reason': 'SL'},
        {'id': 13, 'time': '2026-04-21 21:48', 'dir': 'SELL', 'lots': 0.03, 'pnl':   81.09, 'reason': 'Trailing'},
        {'id': 14, 'time': '2026-04-21 23:28', 'dir': 'SELL', 'lots': 0.03, 'pnl':    6.75, 'reason': 'Trailing'},
        {'id': 15, 'time': '2026-04-21 23:52', 'dir': 'SELL', 'lots': 0.03, 'pnl':   19.59, 'reason': 'Trailing'},
        {'id': 16, 'time': '2026-04-21 23:55', 'dir': 'SELL', 'lots': 0.03, 'pnl':   12.30, 'reason': 'Trailing'},
        {'id': 17, 'time': '2026-04-22 00:12', 'dir': 'SELL', 'lots': 0.03, 'pnl':   11.97, 'reason': 'Trailing'},
        {'id': 18, 'time': '2026-04-22 00:28', 'dir': 'SELL', 'lots': 0.03, 'pnl':    8.07, 'reason': 'Trailing'},
        {'id': 19, 'time': '2026-04-22 01:17', 'dir': 'SELL', 'lots': 0.03, 'pnl':   22.32, 'reason': 'Trailing'},
        {'id': 20, 'time': '2026-04-22 02:42', 'dir': 'SELL', 'lots': 0.03, 'pnl':   27.84, 'reason': 'Trailing'},
        {'id': 21, 'time': '2026-04-22 05:01', 'dir': 'SELL', 'lots': 0.03, 'pnl': -153.00, 'reason': 'SL'},
        {'id': 24, 'time': '2026-04-23 02:00', 'dir': 'SELL', 'lots': 0.03, 'pnl':  -34.62, 'reason': 'Timeout'},
        {'id': 25, 'time': '2026-04-23 07:08', 'dir': 'SELL', 'lots': 0.03, 'pnl':   14.22, 'reason': 'Trailing'},
        {'id': 26, 'time': '2026-04-23 07:15', 'dir': 'SELL', 'lots': 0.03, 'pnl':   33.30, 'reason': 'Trailing'},
        {'id': 27, 'time': '2026-04-23 09:15', 'dir': 'SELL', 'lots': 0.03, 'pnl':   19.47, 'reason': 'Trailing'},
        {'id': 28, 'time': '2026-04-23 13:17', 'dir': 'SELL', 'lots': 0.03, 'pnl':  -29.37, 'reason': 'Timeout'},
        {'id': 30, 'time': '2026-04-23 16:05', 'dir': 'SELL', 'lots': 0.03, 'pnl':   13.50, 'reason': 'Trailing'},
        {'id': 31, 'time': '2026-04-23 16:54', 'dir': 'SELL', 'lots': 0.03, 'pnl':   14.67, 'reason': 'Trailing'},
        {'id': 32, 'time': '2026-04-23 22:05', 'dir': 'SELL', 'lots': 0.03, 'pnl':  -23.10, 'reason': 'SL'},
        {'id': 33, 'time': '2026-04-24 00:44', 'dir': 'SELL', 'lots': 0.02, 'pnl':   22.56, 'reason': 'Trailing'},
        {'id': 34, 'time': '2026-04-24 06:18', 'dir': 'SELL', 'lots': 0.02, 'pnl':    0.86, 'reason': 'Timeout'},
        {'id': 35, 'time': '2026-04-24 11:25', 'dir': 'SELL', 'lots': 0.02, 'pnl':   12.88, 'reason': 'Trailing'},
        {'id': 36, 'time': '2026-04-24 14:26', 'dir': 'SELL', 'lots': 0.02, 'pnl':  -46.86, 'reason': 'Timeout'},
    ]

    # Keltner-only live trades
    keltner_live = [t for t in live_trades if t['id'] not in [22, 23, 29]]
    live_pnls = [t['pnl'] for t in keltner_live]
    live_wins = [p for p in live_pnls if p > 0]
    live_losses = [p for p in live_pnls if p <= 0]

    print(f"\n  --- Live vs Backtest Summary (Keltner only) ---")
    print(f"  {'Metric':<25} {'Live':>12} {'Backtest':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    live_n = len(keltner_live)
    bt_n = r.get('keltner_n', r['n'])
    print(f"  {'Trade count':<25} {live_n:>12} {bt_n:>12} {bt_n - live_n:>+12}")

    live_wr = len(live_wins) / live_n * 100
    bt_wr = r['win_rate']
    print(f"  {'Win rate %':<25} {live_wr:>11.1f}% {bt_wr:>11.1f}% {bt_wr - live_wr:>+11.1f}%")

    live_total = sum(live_pnls)
    print(f"  {'Total PnL':<25} ${live_total:>11.2f} ${r['total_pnl']:>11.2f}")

    live_avg_win = np.mean(live_wins) if live_wins else 0
    live_avg_loss = abs(np.mean(live_losses)) if live_losses else 0
    print(f"  {'Avg win':<25} ${live_avg_win:>11.2f} ${r['avg_win']:>11.2f}")
    print(f"  {'Avg loss':<25} ${live_avg_loss:>11.2f} ${r['avg_loss']:>11.2f}")

    live_rr = live_avg_win / live_avg_loss if live_avg_loss > 0 else 0
    print(f"  {'RR (win/loss)':<25} {live_rr:>12.2f} {r['rr']:>12.2f}")

    # Backtest trade detail for matching
    print(f"\n  --- Backtest Trade Detail (for matching) ---")
    print(f"  {'#':>3} {'ExitTime':<20} {'Dir':<5} {'Entry':>9} {'Exit':>9} {'PnL':>9} {'Bars':>5} {'Reason':<15}")
    for i, t in enumerate(trades[:40], 1):
        exit_ts = pd.Timestamp(t.exit_time).strftime('%m-%d %H:%M')
        print(f"  {i:>3} {exit_ts:<20} {t.direction:<5} {t.entry_price:>9.2f} {t.exit_price:>9.2f} "
              f"${t.pnl:>8.2f} {t.bars_held:>5} {t.exit_reason:<15}")

    # Exit reason distribution
    bt_reasons = {}
    for t in trades:
        bt_reasons[t.exit_reason] = bt_reasons.get(t.exit_reason, 0) + 1
    live_reasons = {}
    for t in keltner_live:
        live_reasons[t['reason']] = live_reasons.get(t['reason'], 0) + 1

    print(f"\n  --- Exit Reason Distribution ---")
    print(f"  {'Reason':<20} {'Live':>8} {'Backtest':>8}")
    all_reasons = set(list(bt_reasons.keys()) + list(live_reasons.keys()))
    for reason in sorted(all_reasons):
        lc = live_reasons.get(reason, 0)
        bc = bt_reasons.get(reason, 0)
        print(f"  {reason:<20} {lc:>8} {bc:>8}")

    return trades


# ═══════════════════════════════════════════════════════════════
# Step 4: PnL Distribution Analysis
# ═══════════════════════════════════════════════════════════════

def step4_pnl_distribution(data, full_result):
    print("\n" + "=" * 90)
    print("  STEP 4: PnL Distribution Analysis")
    print("=" * 90)

    trades = full_result['_trades']
    bt_pnls = np.array([t.pnl for t in trades])

    live_pnls = np.array([
        6.48, 16.54, -15.38, 7.89, 3.35, 4.34, 5.33, 2.58, 3.32, 4.33, 16.03, -45.94,
        81.09, 6.75, 19.59, 12.30, 11.97, 8.07, 22.32, 27.84, -153.00,
        -34.62, 14.22, 33.30, 19.47, -29.37, 13.50, 14.67, -23.10,
        22.56, 0.86, 12.88, -46.86
    ])

    print(f"\n  --- PnL Statistics ---")
    print(f"  {'Metric':<25} {'Backtest':>12} {'Live (33 trades)':>16}")
    print(f"  {'-'*25} {'-'*12} {'-'*16}")
    print(f"  {'Mean PnL':<25} ${np.mean(bt_pnls):>11.2f} ${np.mean(live_pnls):>15.2f}")
    print(f"  {'Median PnL':<25} ${np.median(bt_pnls):>11.2f} ${np.median(live_pnls):>15.2f}")
    print(f"  {'Std PnL':<25} ${np.std(bt_pnls):>11.2f} ${np.std(live_pnls):>15.2f}")
    print(f"  {'Skewness':<25} {float(pd.Series(bt_pnls).skew()):>12.3f} {float(pd.Series(live_pnls).skew()):>16.3f}")
    print(f"  {'Kurtosis':<25} {float(pd.Series(bt_pnls).kurtosis()):>12.3f} {float(pd.Series(live_pnls).kurtosis()):>16.3f}")

    # Percentile comparison
    print(f"\n  --- PnL Percentiles ---")
    print(f"  {'Percentile':<15} {'Backtest':>12} {'Live':>12}")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        bt_p = np.percentile(bt_pnls, pct)
        live_p = np.percentile(live_pnls, pct)
        print(f"  {pct:>5}th %ile     ${bt_p:>11.2f} ${live_p:>11.2f}")

    # Win PnL distribution
    bt_wins = bt_pnls[bt_pnls > 0]
    bt_losses = bt_pnls[bt_pnls <= 0]
    live_wins = live_pnls[live_pnls > 0]
    live_losses = live_pnls[live_pnls <= 0]

    print(f"\n  --- Win/Loss Breakdown ---")
    print(f"  {'Metric':<25} {'Backtest':>12} {'Live':>12}")
    print(f"  {'Win count':<25} {len(bt_wins):>12} {len(live_wins):>12}")
    print(f"  {'Loss count':<25} {len(bt_losses):>12} {len(live_losses):>12}")
    print(f"  {'Win mean':<25} ${np.mean(bt_wins):>11.2f} ${np.mean(live_wins):>11.2f}")
    print(f"  {'Win median':<25} ${np.median(bt_wins):>11.2f} ${np.median(live_wins):>11.2f}")
    print(f"  {'Loss mean':<25} ${np.mean(bt_losses):>11.2f} ${np.mean(live_losses):>11.2f}")
    print(f"  {'Loss median':<25} ${np.median(bt_losses):>11.2f} ${np.median(live_losses):>11.2f}")
    print(f"  {'Max win':<25} ${np.max(bt_wins):>11.2f} ${np.max(live_wins):>11.2f}")
    print(f"  {'Max loss':<25} ${np.min(bt_losses):>11.2f} ${np.min(live_losses):>11.2f}")

    # Histogram buckets (text-based)
    print(f"\n  --- PnL Histogram (Backtest, $10 buckets) ---")
    bt_bins = np.arange(-200, 210, 10)
    bt_hist, _ = np.histogram(bt_pnls, bins=bt_bins)
    max_count = max(bt_hist) if max(bt_hist) > 0 else 1
    for i in range(len(bt_hist)):
        if bt_hist[i] > 0:
            bar = '#' * int(bt_hist[i] / max_count * 50)
            print(f"  [{bt_bins[i]:>6.0f},{bt_bins[i+1]:>6.0f}) {bt_hist[i]:>5} {bar}")

    print(f"\n  --- PnL Histogram (Live, $10 buckets) ---")
    live_hist, _ = np.histogram(live_pnls, bins=bt_bins)
    max_count_l = max(live_hist) if max(live_hist) > 0 else 1
    for i in range(len(live_hist)):
        if live_hist[i] > 0:
            bar = '#' * int(live_hist[i] / max_count_l * 50)
            print(f"  [{bt_bins[i]:>6.0f},{bt_bins[i+1]:>6.0f}) {live_hist[i]:>5} {bar}")


# ═══════════════════════════════════════════════════════════════
# Step 5: True Out-of-Sample (Train 2015-2020, Test 2020-2026)
# ═══════════════════════════════════════════════════════════════

def step5_true_oos(data):
    print("\n" + "=" * 90)
    print("  STEP 5: True Out-of-Sample Validation")
    print("  Parameters fixed from L7_MH8 (no re-optimization)")
    print("  Train period: 2015-01-01 to 2020-01-01 (confirmation only)")
    print("  Test period:  2020-01-01 to 2026-04-10 (true OOS)")
    print("=" * 90)

    spread_levels = [0.30, 0.50, 1.00, 1.50]

    for sp in spread_levels:
        print(f"\n  === Spread = ${sp:.2f} ===")

        # Train (IS) period
        train_data = data.slice("2015-01-01", "2020-01-01")
        r_train = run_variant(train_data, f"OOS_Train_sp{sp}", verbose=False, **L7_MH8, spread_cost=sp)
        orig_tr, corr_tr, ntd_tr, ntot_tr = corrected_sharpe(
            r_train['_trades'], pd.Timestamp("2015-01-01").date(), pd.Timestamp("2020-01-01").date()
        )

        # Test (OOS) period
        test_data = data.slice("2020-01-01", "2026-04-11")
        r_test = run_variant(test_data, f"OOS_Test_sp{sp}", verbose=False, **L7_MH8, spread_cost=sp)
        orig_te, corr_te, ntd_te, ntot_te = corrected_sharpe(
            r_test['_trades'], pd.Timestamp("2020-01-01").date(), pd.Timestamp("2026-04-10").date()
        )

        print(f"  {'Period':<12} {'N':>6} {'OrigSharpe':>11} {'CorrSharpe':>11} {'PnL':>10} {'WR%':>6} {'RR':>5}")
        print(f"  {'-'*12} {'-'*6} {'-'*11} {'-'*11} {'-'*10} {'-'*6} {'-'*5}")
        print(f"  {'Train(IS)':<12} {r_train['n']:>6} {orig_tr:>11.2f} {corr_tr:>11.2f} ${r_train['total_pnl']:>9.0f} "
              f"{r_train['win_rate']:>5.1f}% {r_train['rr']:>4.2f}")
        print(f"  {'Test(OOS)':<12} {r_test['n']:>6} {orig_te:>11.2f} {corr_te:>11.2f} ${r_test['total_pnl']:>9.0f} "
              f"{r_test['win_rate']:>5.1f}% {r_test['rr']:>4.2f}")

        if corr_tr > 0:
            decay = (corr_tr - corr_te) / corr_tr * 100
            print(f"  OOS Sharpe decay: {decay:.1f}%")

    # Yearly breakdown at spread=$0.50
    print(f"\n  --- Yearly Breakdown (Corrected Sharpe, spread=$0.50) ---")
    print(f"  {'Year':<6} {'N':>6} {'OrigSharpe':>11} {'CorrSharpe':>11} {'PnL':>10} {'WR%':>6}")
    years = [(str(y), str(y+1)) for y in range(2015, 2026)]
    years[-1] = ("2025", "2026-04-11")

    for start, end in years:
        yr_start = f"{start}-01-01"
        yr_end = f"{end}-01-01" if len(end) == 4 else end
        yr_data = data.slice(yr_start, yr_end)
        if len(yr_data.m15_df) < 500:
            continue
        r = run_variant(yr_data, f"Y{start}", verbose=False, **L7_MH8, spread_cost=0.50)
        if r['n'] == 0:
            print(f"  {start:<6} {0:>6} {'---':>11} {'---':>11} ${'0':>9} {'---':>6}")
            continue
        orig, corr, _, _ = corrected_sharpe(
            r['_trades'],
            pd.Timestamp(yr_start).date(),
            pd.Timestamp(yr_end).date()
        )
        print(f"  {start:<6} {r['n']:>6} {orig:>11.2f} {corr:>11.2f} ${r['total_pnl']:>9.0f} {r['win_rate']:>5.1f}%")

    # Crisis-period zoom
    print(f"\n  --- Crisis Period Performance (spread=$1.00) ---")
    crisis_periods = [
        ("COVID crash",     "2020-02-15", "2020-04-15"),
        ("2022 rate hikes", "2022-03-01", "2022-10-01"),
        ("2023 banking",    "2023-03-01", "2023-05-01"),
        ("2025 Dec pivot",  "2025-11-01", "2026-01-15"),
        ("2026 Liberation", "2026-03-15", "2026-04-11"),
    ]
    for name, cs, ce in crisis_periods:
        cd = data.slice(cs, ce)
        if len(cd.m15_df) < 200:
            print(f"  {name:<22}: insufficient data")
            continue
        r = run_variant(cd, f"Crisis_{name[:8]}", verbose=False, **L7_MH8, spread_cost=1.00)
        orig, corr, _, _ = corrected_sharpe(r['_trades'], pd.Timestamp(cs).date(), pd.Timestamp(ce).date())
        print(f"  {name:<22}: N={r['n']:>4}, CorrSharpe={corr:>6.2f}, PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    out_path = OUT_DIR / "reality_check_output.txt"
    f_out = open(out_path, 'w', encoding='utf-8')
    tee = Tee(sys.stdout, f_out)
    sys.stdout = tee

    print("=" * 90)
    print("  REALITY CHECK: Backtest Credibility Diagnosis")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    t0 = time.time()
    data = DataBundle.load_default()

    print(f"\n  Data loaded: M15={len(data.m15_df)} bars, H1={len(data.h1_df)} bars")
    print(f"  Date range: {data.m15_df.index[0]} to {data.m15_df.index[-1]}")

    full_result = step1_sharpe_fix(data)
    step2_spread_sensitivity(data)
    step3_live_replay(data)
    step4_pnl_distribution(data, full_result)
    step5_true_oos(data)

    elapsed = time.time() - t0
    print(f"\n\n{'=' * 90}")
    print(f"  REALITY CHECK COMPLETE")
    print(f"  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  Results saved to: {out_path}")
    print(f"{'=' * 90}")

    sys.stdout = sys.__stdout__
    f_out.close()
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()
