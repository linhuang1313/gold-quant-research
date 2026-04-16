#!/usr/bin/env python3
"""
EXP48-53 批量串行执行 — 基于黄金定价核心特性的新实验
======================================================
  EXP48: Keltner 均值回归策略（全新互补策略）
  EXP49: 波动率聚集自适应 Trailing
  EXP50: 大波动后方向偏倚（post-hoc sizing）
  EXP51: Keltner 通道宽度变化率（入场质量因子）
  EXP52: 持仓内动态止损调整
  EXP53: 多策略组合权重优化

共享一次数据加载 + 两次基线回测。无点差。
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "exp48_53_output.txt"


class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

print("=" * 70)
print("EXP48-53 BATCH — GOLD PRICING CORE NEW STRATEGIES")
print(f"Started: {datetime.now()}")
print("=" * 70)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

print("\n--- Running Current baseline ---")
baseline_cur = run_variant(data, "Current", **CURRENT)
trades_cur = baseline_cur['_trades']

print("\n--- Running Mega baseline ---")
baseline_mega = run_variant(data, "Mega", **MEGA)
trades_mega = baseline_mega['_trades']

h1_df = data.h1_df.copy()

print(f"\n  Shared baselines ready:")
print(f"  Current: N={baseline_cur['n']:,} Sharpe={baseline_cur['sharpe']:.2f} PnL=${baseline_cur['total_pnl']:,.0f}")
print(f"  Mega:    N={baseline_mega['n']:,} Sharpe={baseline_mega['sharpe']:.2f} PnL=${baseline_mega['total_pnl']:,.0f}")


def compute_sharpe(trades, pnls):
    daily = defaultdict(float)
    for t, pnl in zip(trades, pnls):
        day = t.entry_time.strftime('%Y-%m-%d')
        daily[day] += pnl
    vals = list(daily.values())
    if len(vals) > 1 and np.std(vals) > 0:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0


def to_utc_ts(dt):
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts


# Pre-build daily OHLC from H1
d1_df = h1_df.resample('1D').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()
d1_df['ret'] = d1_df['Close'].pct_change() * 100
d1_df['d_range'] = d1_df['High'] - d1_df['Low']
d1_df['d_atr14'] = d1_df['d_range'].rolling(14).mean()
d1_df['vol_ratio'] = d1_df['d_range'] / d1_df['d_atr14']
print(f"  D1 data: {len(d1_df)} days")


def get_prev_day_row(entry_time):
    """Get previous completed day's D1 row for a trade entry."""
    ts = to_utc_ts(entry_time)
    # Match d1_df index tz-awareness
    if d1_df.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    elif d1_df.index.tz is None and ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    prev = d1_df.loc[:ts]
    if len(prev) < 2:
        return None
    return prev.iloc[-2]


# ═══════════════════════════════════════════════════════════════
# EXP48: KELTNER MEAN REVERSION STRATEGY (POST-HOC SIMULATION)
# ═══════════════════════════════════════════════════════════════

def run_exp48():
    print("\n\n" + "=" * 70)
    print("EXP48: KELTNER MEAN REVERSION STRATEGY (NEW)")
    print("=" * 70)
    t0 = time.time()

    # Simulate a mean-reversion strategy on H1 data:
    #   Entry: price touches KC band but closes back inside (wick rejection)
    #         + ADX < threshold (low trend = mean-reversion environment)
    #   TP: KC mid (EMA25), SL: N x ATR
    #   Max hold: 20 H1 bars

    results_by_config = []

    for adx_max in [15, 18, 20, 22, 25]:
        for sl_mult in [1.5, 2.0, 2.5]:
            trades_mr = []
            in_trade = False
            trade_dir = None
            entry_price = 0.0
            entry_time = None
            sl_price = 0.0
            tp_price = 0.0
            bars_held = 0

            for i in range(105, len(h1_df)):
                bar = h1_df.iloc[i]
                bar_time = h1_df.index[i]
                close = float(bar['Close'])
                high = float(bar['High'])
                low = float(bar['Low'])
                kc_upper = float(bar.get('KC_upper', 0))
                kc_lower = float(bar.get('KC_lower', 0))
                kc_mid = float(bar.get('KC_mid', 0))
                adx = float(bar.get('ADX', 0))
                atr = float(bar.get('ATR', 0))

                if any(pd.isna(v) for v in [kc_upper, kc_lower, kc_mid, adx, atr]):
                    continue
                if atr <= 0 or kc_mid <= 0:
                    continue

                # Check exits first
                if in_trade:
                    bars_held += 1
                    pnl = 0.0
                    reason = None

                    if trade_dir == 'BUY':
                        if low <= sl_price:
                            pnl = sl_price - entry_price
                            reason = 'SL'
                        elif high >= tp_price:
                            pnl = tp_price - entry_price
                            reason = 'TP'
                        elif bars_held >= 20:
                            pnl = close - entry_price
                            reason = 'Timeout'
                    else:  # SELL
                        if high >= sl_price:
                            pnl = entry_price - sl_price
                            reason = 'SL'
                        elif low <= tp_price:
                            pnl = entry_price - tp_price
                            reason = 'TP'
                        elif bars_held >= 20:
                            pnl = entry_price - close
                            reason = 'Timeout'

                    if reason:
                        # 0.01 lot * 100 (POINT_VALUE_PER_LOT)
                        trades_mr.append({
                            'dir': trade_dir, 'entry': entry_price,
                            'pnl': round(pnl * 0.01 * 100, 2),
                            'reason': reason, 'bars': bars_held,
                            'entry_time': entry_time,
                        })
                        in_trade = False
                        continue

                # Check entries (not in trade, ADX low)
                if not in_trade and adx < adx_max:
                    # Short MR: wick above upper band, close inside
                    if high > kc_upper and close < kc_upper and close > kc_mid:
                        in_trade = True
                        trade_dir = 'SELL'
                        entry_price = close
                        entry_time = bar_time
                        sl_price = close + atr * sl_mult
                        tp_price = kc_mid
                        bars_held = 0

                    # Long MR: wick below lower band, close inside
                    elif low < kc_lower and close > kc_lower and close < kc_mid:
                        in_trade = True
                        trade_dir = 'BUY'
                        entry_price = close
                        entry_time = bar_time
                        sl_price = close - atr * sl_mult
                        tp_price = kc_mid
                        bars_held = 0

            n = len(trades_mr)
            if n < 20:
                continue
            total_pnl = sum(t['pnl'] for t in trades_mr)
            wins = sum(1 for t in trades_mr if t['pnl'] > 0)
            sl_count = sum(1 for t in trades_mr if t['reason'] == 'SL')
            tp_count = sum(1 for t in trades_mr if t['reason'] == 'TP')
            to_count = sum(1 for t in trades_mr if t['reason'] == 'Timeout')

            daily_pnl = defaultdict(float)
            for t in trades_mr:
                day = t['entry_time'].strftime('%Y-%m-%d')
                daily_pnl[day] += t['pnl']
            vals = list(daily_pnl.values())
            sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0

            results_by_config.append({
                'adx_max': adx_max, 'sl_mult': sl_mult, 'n': n,
                'pnl': total_pnl, 'wr': 100.0 * wins / n, 'sharpe': sharpe,
                'sl': sl_count, 'tp': tp_count, 'to': to_count,
                'ppt': total_pnl / n,
            })

    print(f"\n  Mean Reversion Strategy Scan ({len(results_by_config)} configs):")
    print(f"  {'ADX<':>5} {'SL_x':>5} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'Sharpe':>8} {'SL':>5} {'TP':>5} {'TO':>5}")
    print(f"  {'-'*70}")

    results_by_config.sort(key=lambda r: r['sharpe'], reverse=True)
    for r in results_by_config:
        print(f"  {r['adx_max']:>5} {r['sl_mult']:>5.1f} {r['n']:>6} ${r['pnl']:>9,.0f} ${r['ppt']:>6.2f} "
              f"{r['wr']:>5.1f}% {r['sharpe']:>8.2f} {r['sl']:>5} {r['tp']:>5} {r['to']:>5}")

    if results_by_config:
        best = results_by_config[0]
        print(f"\n  BEST: ADX<{best['adx_max']} SL={best['sl_mult']}x => "
              f"Sharpe={best['sharpe']:.2f} PnL=${best['pnl']:,.0f} N={best['n']} WR={best['wr']:.1f}%")

        # Complementarity check: Keltner trades in high ADX, MR in low ADX
        keltner_days = set(t.entry_time.strftime('%Y-%m-%d') for t in trades_cur if t.strategy == 'keltner')

        # Re-run best config to collect MR entry days
        mr_days = set()
        in_trade = False
        for i in range(105, len(h1_df)):
            bar = h1_df.iloc[i]
            adx_val = float(bar.get('ADX', 0))
            if pd.isna(adx_val):
                continue
            high_v, low_v, close_v = float(bar['High']), float(bar['Low']), float(bar['Close'])
            kc_u = float(bar.get('KC_upper', 0))
            kc_l = float(bar.get('KC_lower', 0))
            kc_m = float(bar.get('KC_mid', 0))
            if any(pd.isna(v) for v in [kc_u, kc_l, kc_m]):
                continue
            if adx_val < best['adx_max'] and not in_trade:
                if (high_v > kc_u and close_v < kc_u) or (low_v < kc_l and close_v > kc_l):
                    mr_days.add(h1_df.index[i].strftime('%Y-%m-%d'))

        overlap = keltner_days & mr_days
        print(f"\n  Complementarity with Keltner:")
        print(f"    Keltner active days: {len(keltner_days)}")
        print(f"    MR active days: {len(mr_days)}")
        print(f"    Overlap: {len(overlap)} ({100*len(overlap)/max(len(mr_days),1):.1f}% of MR days)")

        # Year-by-year for best config: re-run to get yearly breakdown
        print(f"\n  Best Config Year-by-Year:")
        print(f"  {'Year':<6} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*40}")

        yearly = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0})
        in_trade = False
        trade_dir = None
        entry_price = 0.0
        entry_time = None
        sl_price = 0.0
        tp_price = 0.0
        bars_held = 0

        for i in range(105, len(h1_df)):
            bar = h1_df.iloc[i]
            bt = h1_df.index[i]
            close_v = float(bar['Close'])
            high_v = float(bar['High'])
            low_v = float(bar['Low'])
            kc_upper = float(bar.get('KC_upper', 0))
            kc_lower = float(bar.get('KC_lower', 0))
            kc_mid_v = float(bar.get('KC_mid', 0))
            adx_v = float(bar.get('ADX', 0))
            atr_v = float(bar.get('ATR', 0))
            if any(pd.isna(v) for v in [kc_upper, kc_lower, kc_mid_v, adx_v, atr_v]):
                continue
            if atr_v <= 0 or kc_mid_v <= 0:
                continue

            if in_trade:
                bars_held += 1
                pnl_v = 0.0
                reason = None
                if trade_dir == 'BUY':
                    if low_v <= sl_price: pnl_v, reason = sl_price - entry_price, 'SL'
                    elif high_v >= tp_price: pnl_v, reason = tp_price - entry_price, 'TP'
                    elif bars_held >= 20: pnl_v, reason = close_v - entry_price, 'Timeout'
                else:
                    if high_v >= sl_price: pnl_v, reason = entry_price - sl_price, 'SL'
                    elif low_v <= tp_price: pnl_v, reason = entry_price - tp_price, 'TP'
                    elif bars_held >= 20: pnl_v, reason = entry_price - close_v, 'Timeout'
                if reason:
                    yr = entry_time.year
                    pnl_dollar = round(pnl_v * 0.01 * 100, 2)
                    yearly[yr]['n'] += 1
                    yearly[yr]['pnl'] += pnl_dollar
                    if pnl_dollar > 0:
                        yearly[yr]['wins'] += 1
                    in_trade = False
                    continue

            if not in_trade and adx_v < best['adx_max']:
                if high_v > kc_upper and close_v < kc_upper and close_v > kc_mid_v:
                    in_trade, trade_dir, entry_price = True, 'SELL', close_v
                    entry_time = bt
                    sl_price = close_v + atr_v * best['sl_mult']
                    tp_price = kc_mid_v
                    bars_held = 0
                elif low_v < kc_lower and close_v > kc_lower and close_v < kc_mid_v:
                    in_trade, trade_dir, entry_price = True, 'BUY', close_v
                    entry_time = bt
                    sl_price = close_v - atr_v * best['sl_mult']
                    tp_price = kc_mid_v
                    bars_held = 0

        for yr in sorted(yearly.keys()):
            d = yearly[yr]
            wr = 100.0 * d['wins'] / d['n'] if d['n'] > 0 else 0
            ppt = d['pnl'] / d['n'] if d['n'] > 0 else 0
            print(f"  {yr:<6} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")

    print(f"\n  EXP48 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP49: VOLATILITY CLUSTERING ADAPTIVE TRAILING
# ═══════════════════════════════════════════════════════════════

def run_exp49():
    print("\n\n" + "=" * 70)
    print("EXP49: VOLATILITY CLUSTERING ADAPTIVE TRAILING")
    print("=" * 70)
    t0 = time.time()

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)

        by_vol = {'low': [], 'normal': [], 'high': [], 'extreme': []}
        for t in trades:
            row = get_prev_day_row(t.entry_time)
            if row is None:
                by_vol['normal'].append(t)
                continue
            vr = row.get('vol_ratio', 1.0)
            if pd.isna(vr):
                vr = 1.0
            if vr < 0.7:
                by_vol['low'].append(t)
            elif vr < 1.3:
                by_vol['normal'].append(t)
            elif vr < 2.0:
                by_vol['high'].append(t)
            else:
                by_vol['extreme'].append(t)

        print(f"\n  {label} Trades by Prev-Day Vol Regime:")
        print(f"  {'Regime':<12} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*45}")
        for cat in ['low', 'normal', 'high', 'extreme']:
            bt = by_vol[cat]
            if not bt:
                continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  {cat:<12} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

        # Simulate vol-cluster adaptive sizing/trailing
        schemes = [
            ("Flat 1.0x (baseline)",
             lambda vr, reason: 1.0),
            ("HighVol trail 1.2x, LowVol trail 0.8x",
             lambda vr, reason: (1.2 if vr > 1.3 else (0.8 if vr < 0.7 else 1.0))
                                if 'railing' in str(reason) else 1.0),
            ("HighVol ALL 1.2x, LowVol ALL 0.8x",
             lambda vr, reason: 1.2 if vr > 1.3 else (0.8 if vr < 0.7 else 1.0)),
            ("Extreme(>2x) reduce 0.6x",
             lambda vr, reason: 0.6 if vr > 2.0 else 1.0),
            ("HighVol 1.3x (ride the cluster)",
             lambda vr, reason: 1.3 if vr > 1.3 else 1.0),
            ("2-day low vol: tighten 0.7x",
             lambda vr, reason: 0.7 if vr < 0.5 else 1.0),
        ]

        print(f"\n  {label} Vol-Cluster Adaptive Sizing:")
        print(f"  {'Scheme':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")

        for sname, sfunc in schemes:
            pnls = []
            for t in trades:
                row = get_prev_day_row(t.entry_time)
                vr = float(row['vol_ratio']) if row is not None and not pd.isna(row.get('vol_ratio')) else 1.0
                reason = t.exit_reason
                scale = sfunc(vr, reason)
                pnls.append(t.pnl * scale)
            sh = compute_sharpe(trades, pnls)
            print(f"  {sname:<45} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    # Yearly stability
    print(f"\n  Yearly: HighVol 1.3x vs Baseline (Current):")
    print(f"  {'Year':<6} {'Base_Sh':>8} {'VolClust_Sh':>12} {'Delta':>7}")
    print(f"  {'-'*36}")
    for year in range(2015, 2027):
        start_str = f"{year}-01-01"
        end_str = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
        yr = [t for t in trades_cur if start_str <= t.entry_time.strftime('%Y-%m-%d') < end_str]
        if len(yr) < 20:
            continue
        bp = [t.pnl for t in yr]
        bsh = compute_sharpe(yr, bp)
        vp = []
        for t in yr:
            row = get_prev_day_row(t.entry_time)
            vr = float(row['vol_ratio']) if row is not None and not pd.isna(row.get('vol_ratio')) else 1.0
            vp.append(t.pnl * (1.3 if vr > 1.3 else 1.0))
        vsh = compute_sharpe(yr, vp)
        print(f"  {year:<6} {bsh:>8.2f} {vsh:>12.2f} {vsh-bsh:>+7.2f}")

    print(f"\n  EXP49 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP50: LARGE MOVE REVERSAL BIAS (FIXED TZ)
# ═══════════════════════════════════════════════════════════════

def run_exp50():
    print("\n\n" + "=" * 70)
    print("EXP50: LARGE DAILY MOVE DIRECTION BIAS")
    print("=" * 70)
    t0 = time.time()

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)

        cats = {'Big up (>1%)': [], 'Big down (<-1%)': [], 'Normal': []}
        for t in trades:
            row = get_prev_day_row(t.entry_time)
            if row is None:
                cats['Normal'].append(t)
                continue
            ret = row.get('ret', 0)
            if pd.isna(ret):
                ret = 0
            if ret > 1.0:
                cats['Big up (>1%)'].append(t)
            elif ret < -1.0:
                cats['Big down (<-1%)'].append(t)
            else:
                cats['Normal'].append(t)

        print(f"\n  {label} Trades by Prev-Day Return:")
        print(f"  {'Category':<20} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'BUY_$/t':>8} {'SELL_$/t':>9}")
        print(f"  {'-'*72}")
        for cat, bt in cats.items():
            if not bt:
                continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            buy_t = [t for t in bt if t.direction == 'BUY']
            sell_t = [t for t in bt if t.direction == 'SELL']
            buy_ppt = sum(t.pnl for t in buy_t) / len(buy_t) if buy_t else 0
            sell_ppt = sum(t.pnl for t in sell_t) / len(sell_t) if sell_t else 0
            print(f"  {cat:<20} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} "
                  f"{100*wins/len(bt):>5.1f}% ${buy_ppt:>7.2f} ${sell_ppt:>8.2f}")

        # Direction bias sizing simulations
        rules = [
            ("Flat 1.0x",
             lambda t, ret: 1.0),
            ("After up>1%: SELL 1.3x BUY 0.8x",
             lambda t, ret: (1.3 if t.direction == 'SELL' else 0.8) if ret > 1.0 else 1.0),
            ("After dn<-1%: BUY 1.3x SELL 0.8x",
             lambda t, ret: (1.3 if t.direction == 'BUY' else 0.8) if ret < -1.0 else 1.0),
            ("Both: fade reversal 1.3x/0.8x",
             lambda t, ret: (1.3 if t.direction == 'SELL' else 0.8) if ret > 1.0
                            else ((1.3 if t.direction == 'BUY' else 0.8) if ret < -1.0 else 1.0)),
            ("Stronger: fade 1.5x/0.6x",
             lambda t, ret: (1.5 if t.direction == 'SELL' else 0.6) if ret > 1.0
                            else ((1.5 if t.direction == 'BUY' else 0.6) if ret < -1.0 else 1.0)),
            ("Threshold 0.5%: fade 1.2x/0.9x",
             lambda t, ret: (1.2 if t.direction == 'SELL' else 0.9) if ret > 0.5
                            else ((1.2 if t.direction == 'BUY' else 0.9) if ret < -0.5 else 1.0)),
            ("Threshold 1.5%: fade 1.3x/0.7x",
             lambda t, ret: (1.3 if t.direction == 'SELL' else 0.7) if ret > 1.5
                            else ((1.3 if t.direction == 'BUY' else 0.7) if ret < -1.5 else 1.0)),
        ]

        print(f"\n  {label} Direction Bias Sizing:")
        print(f"  {'Rule':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")
        for rname, rfunc in rules:
            pnls = []
            for t in trades:
                row = get_prev_day_row(t.entry_time)
                ret = float(row['ret']) if row is not None and not pd.isna(row.get('ret')) else 0
                pnls.append(t.pnl * rfunc(t, ret))
            sh = compute_sharpe(trades, pnls)
            print(f"  {rname:<45} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    print(f"\n  EXP50 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP51: KC BANDWIDTH CHANGE RATE AS ENTRY QUALITY FACTOR
# ═══════════════════════════════════════════════════════════════

def run_exp51():
    print("\n\n" + "=" * 70)
    print("EXP51: KELTNER CHANNEL BANDWIDTH CHANGE RATE")
    print("=" * 70)
    t0 = time.time()

    # Compute bandwidth and its rate of change on H1
    h1_df['KC_bandwidth'] = (h1_df['KC_upper'] - h1_df['KC_lower']) / h1_df['KC_mid']
    for period in [3, 5, 8]:
        h1_df[f'KC_bw_chg_{period}'] = h1_df['KC_bandwidth'] - h1_df['KC_bandwidth'].shift(period)

    def get_bw_metrics(entry_time):
        ts = to_utc_ts(entry_time)
        idx = h1_df.index.get_indexer([ts], method='ffill')[0]
        if idx < 0 or idx >= len(h1_df):
            return None, None, None
        bar = h1_df.iloc[idx]
        bw = bar.get('KC_bandwidth', None)
        chg3 = bar.get('KC_bw_chg_3', None)
        chg5 = bar.get('KC_bw_chg_5', None)
        return (float(bw) if not pd.isna(bw) else None,
                float(chg3) if not pd.isna(chg3) else None,
                float(chg5) if not pd.isna(chg5) else None)

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        keltner = [t for t in trades if t.strategy == 'keltner']
        print(f"\n  {label} ({len(keltner)} keltner trades):")

        # Part 1: bandwidth change vs trade outcome
        expanding, contracting, stable = [], [], []
        for t in keltner:
            bw, chg3, chg5 = get_bw_metrics(t.entry_time)
            if chg5 is None:
                continue
            if chg5 > 0.001:
                expanding.append(t)
            elif chg5 < -0.001:
                contracting.append(t)
            else:
                stable.append(t)

        print(f"\n  KC Bandwidth 5-bar change at entry:")
        print(f"  {'State':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
        print(f"  {'-'*48}")
        for name, bt in [("Expanding", expanding), ("Stable", stable), ("Contracting", contracting)]:
            if not bt:
                continue
            pnl = sum(t.pnl for t in bt)
            wins = sum(1 for t in bt if t.pnl > 0)
            print(f"  {name:<15} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

        # Part 2: bandwidth quantile buckets
        bw_pnl_data = []
        for t in keltner:
            bw, chg3, chg5 = get_bw_metrics(t.entry_time)
            if chg5 is not None:
                bw_pnl_data.append((chg5, t.pnl, t))

        p25 = p50 = p75 = 0.0
        if bw_pnl_data:
            chg_vals = [x[0] for x in bw_pnl_data]
            p25, p50, p75 = np.percentile(chg_vals, [25, 50, 75])
            buckets = [
                (f"Q1 (<{p25:.4f})", lambda c, _p25=p25: c < _p25),
                (f"Q2 ({p25:.4f}-{p50:.4f})", lambda c, _p25=p25, _p50=p50: _p25 <= c < _p50),
                (f"Q3 ({p50:.4f}-{p75:.4f})", lambda c, _p50=p50, _p75=p75: _p50 <= c < _p75),
                (f"Q4 (>{p75:.4f})", lambda c, _p75=p75: c >= _p75),
            ]
            print(f"\n  Bandwidth Change Quantiles:")
            print(f"  {'Bucket':<25} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
            print(f"  {'-'*57}")
            for bname, bfunc in buckets:
                bt = [(c, p, t) for c, p, t in bw_pnl_data if bfunc(c)]
                if not bt:
                    continue
                pnl = sum(p for _, p, _ in bt)
                wins = sum(1 for _, p, _ in bt if p > 0)
                print(f"  {bname:<25} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")

        # Part 3: Squeeze-to-expansion detection
        squeeze_expand = []
        normal_expand = []
        for t in keltner:
            bw, chg3, chg5 = get_bw_metrics(t.entry_time)
            if chg3 is None or chg5 is None:
                continue
            if chg5 > 0.001 and chg3 > chg5:  # accelerating expansion
                squeeze_expand.append(t)
            elif chg5 > 0.001:
                normal_expand.append(t)

        if squeeze_expand and normal_expand:
            sq_ppt = sum(t.pnl for t in squeeze_expand) / len(squeeze_expand)
            ne_ppt = sum(t.pnl for t in normal_expand) / len(normal_expand)
            print(f"\n  Squeeze-to-Expansion: N={len(squeeze_expand)} $/t=${sq_ppt:.2f}")
            print(f"  Normal Expansion:     N={len(normal_expand)} $/t=${ne_ppt:.2f}")
            print(f"  Diff: ${sq_ppt - ne_ppt:+.2f}")

        # Part 4: Filter simulation on full trade set
        print(f"\n  {label} Bandwidth Filter Sharpe:")

        # Need the p25 value for filter lambdas
        local_p25 = p25

        filters = [
            ("No filter",
             lambda t: True),
            ("Only expanding (chg5>0)",
             lambda t: (get_bw_metrics(t.entry_time)[2] or 0) > 0.001
                       if t.strategy == 'keltner' else True),
            ("Only contracting (chg5<0)",
             lambda t: (get_bw_metrics(t.entry_time)[2] or 0) < -0.001
                       if t.strategy == 'keltner' else True),
            ("Skip extreme contraction (Q1)",
             lambda t, _p25=local_p25: (get_bw_metrics(t.entry_time)[2] or 0) > _p25
                       if t.strategy == 'keltner' else True),
        ]
        print(f"  {'Filter':<35} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10}")
        print(f"  {'-'*68}")
        for fname, ffunc in filters:
            kept = [t for t in trades if ffunc(t)]
            if len(kept) < 50:
                continue
            eq = [0.0]
            for t in kept:
                eq.append(eq[-1] + t.pnl)
            s = calc_stats(kept, eq)
            print(f"  {fname:<35} {s['n']:>6} {s['sharpe']:>8.2f} "
                  f"{s['sharpe']-bstats['sharpe']:>+7.2f} ${s['total_pnl']:>9,.0f}")

    print(f"\n  EXP51 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP52: INTRA-TRADE DYNAMIC STOP LOSS (POST-HOC SIMULATION)
# ═══════════════════════════════════════════════════════════════

def run_exp52():
    print("\n\n" + "=" * 70)
    print("EXP52: INTRA-TRADE DYNAMIC STOP LOSS ADJUSTMENT")
    print("=" * 70)
    t0 = time.time()

    for label, trades in [("Current", trades_cur), ("Mega", trades_mega)]:
        keltner = [t for t in trades if t.strategy == 'keltner']

        # Analyze ATR change during holding period
        atr_changes = {
            'increased': {'n': 0, 'pnl': 0.0},
            'decreased': {'n': 0, 'pnl': 0.0},
            'stable': {'n': 0, 'pnl': 0.0},
        }
        spike_trades = []

        for t in keltner:
            ts_entry = to_utc_ts(t.entry_time)
            ts_exit = to_utc_ts(t.exit_time)
            mask = (h1_df.index >= ts_entry) & (h1_df.index <= ts_exit)
            bars = h1_df.loc[mask]
            if len(bars) < 2:
                continue

            entry_atr = float(bars.iloc[0]['ATR'])
            if pd.isna(entry_atr) or entry_atr <= 0:
                continue
            max_atr = float(bars['ATR'].max())
            exit_atr = float(bars.iloc[-1]['ATR'])
            if pd.isna(max_atr) or pd.isna(exit_atr):
                continue

            atr_change_pct = (exit_atr - entry_atr) / entry_atr * 100

            if atr_change_pct > 10:
                atr_changes['increased']['n'] += 1
                atr_changes['increased']['pnl'] += t.pnl
            elif atr_change_pct < -10:
                atr_changes['decreased']['n'] += 1
                atr_changes['decreased']['pnl'] += t.pnl
            else:
                atr_changes['stable']['n'] += 1
                atr_changes['stable']['pnl'] += t.pnl

            atr_spike_pct = (max_atr - entry_atr) / entry_atr * 100
            if atr_spike_pct > 50:
                spike_trades.append(t)

        print(f"\n  {label} ({len(keltner)} keltner trades):")
        print(f"\n  ATR change during holding period:")
        print(f"  {'Change':<15} {'N':>6} {'PnL':>10} {'$/t':>7}")
        print(f"  {'-'*40}")
        for cat in ['increased', 'stable', 'decreased']:
            d = atr_changes[cat]
            if d['n'] == 0:
                continue
            print(f"  ATR {cat:<10} {d['n']:>6} ${d['pnl']:>9,.0f} ${d['pnl']/d['n']:>6.2f}")

        if spike_trades:
            spike_pnl = sum(t.pnl for t in spike_trades)
            spike_wins = sum(1 for t in spike_trades if t.pnl > 0)
            non_spike_pnl = sum(t.pnl for t in keltner) - spike_pnl
            print(f"\n  ATR spike >50% during trade: N={len(spike_trades)} "
                  f"PnL=${spike_pnl:,.0f} $/t=${spike_pnl/len(spike_trades):.2f} "
                  f"WR={100*spike_wins/len(spike_trades):.1f}%")
            print(f"  Non-spike: N={len(keltner)-len(spike_trades)} PnL=${non_spike_pnl:,.0f}")

        # Simulate dynamic SL strategies
        base_pnls = [t.pnl for t in keltner]
        base_sh = compute_sharpe(keltner, base_pnls)

        schemes = [
            ("Flat (baseline)",
             lambda t, ea, ma, xa: 1.0),
            ("ATR spike >50%: if profitable take 0.8x",
             lambda t, ea, ma, xa: 0.8 if ea > 0 and (ma - ea) / ea > 0.5 and t.pnl > 0 else 1.0),
            ("ATR spike >50%: always reduce 0.7x",
             lambda t, ea, ma, xa: 0.7 if ea > 0 and (ma - ea) / ea > 0.5 else 1.0),
            ("ATR increase >30%: tighten 0.85x",
             lambda t, ea, ma, xa: 0.85 if ea > 0 and (xa - ea) / ea > 0.3 else 1.0),
            ("ATR decrease >20%: extend 1.15x",
             lambda t, ea, ma, xa: 1.15 if ea > 0 and (ea - xa) / ea > 0.2 else 1.0),
            ("Combined: spike protect + compress ext",
             lambda t, ea, ma, xa: (0.8 if ea > 0 and (ma - ea) / ea > 0.5
                                    else (1.15 if ea > 0 and (ea - xa) / ea > 0.2 else 1.0))),
        ]

        print(f"\n  Simulated Dynamic SL Strategies:")
        print(f"  {'Scheme':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")

        for sname, sfunc in schemes:
            pnls = []
            for t in keltner:
                ts_entry = to_utc_ts(t.entry_time)
                ts_exit = to_utc_ts(t.exit_time)
                mask = (h1_df.index >= ts_entry) & (h1_df.index <= ts_exit)
                bars = h1_df.loc[mask]
                if len(bars) < 2:
                    pnls.append(t.pnl)
                    continue
                ea = float(bars.iloc[0]['ATR'])
                if pd.isna(ea) or ea <= 0:
                    pnls.append(t.pnl)
                    continue
                ma = float(bars['ATR'].max())
                xa = float(bars.iloc[-1]['ATR'])
                if pd.isna(ma) or pd.isna(xa):
                    pnls.append(t.pnl)
                    continue
                pnls.append(t.pnl * sfunc(t, ea, ma, xa))
            sh = compute_sharpe(keltner, pnls)
            print(f"  {sname:<45} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

    print(f"\n  EXP52 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXP53: MULTI-STRATEGY PORTFOLIO WEIGHT OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def run_exp53():
    print("\n\n" + "=" * 70)
    print("EXP53: MULTI-STRATEGY PORTFOLIO ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    for label, trades, bstats in [("Current", trades_cur, baseline_cur),
                                   ("Mega", trades_mega, baseline_mega)]:
        # Part 1: Per-strategy statistics
        strat_trades = defaultdict(list)
        for t in trades:
            strat_trades[t.strategy].append(t)

        print(f"\n  {label} Per-Strategy Stats:")
        print(f"  {'Strategy':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'Sharpe':>8} {'MaxDD':>8}")
        print(f"  {'-'*68}")

        strat_sharpes = {}
        for strat in sorted(strat_trades.keys()):
            st = strat_trades[strat]
            if len(st) < 10:
                continue
            pnl = sum(t.pnl for t in st)
            wins = sum(1 for t in st if t.pnl > 0)
            eq = [0.0]
            for t in st:
                eq.append(eq[-1] + t.pnl)
            stats = calc_stats(st, eq)
            strat_sharpes[strat] = stats['sharpe']
            print(f"  {strat:<15} {len(st):>6} ${pnl:>9,.0f} ${pnl/len(st):>6.2f} "
                  f"{100*wins/len(st):>5.1f}% {stats['sharpe']:>8.2f} ${stats['max_dd']:>7,.0f}")

        # Part 2: Strategy daily PnL correlation
        strat_daily = {}
        all_days = set()
        for strat, st in strat_trades.items():
            if len(st) < 10:
                continue
            daily = defaultdict(float)
            for t in st:
                day = t.entry_time.strftime('%Y-%m-%d')
                daily[day] += t.pnl
                all_days.add(day)
            strat_daily[strat] = daily

        strat_names = sorted(strat_daily.keys())
        if len(strat_names) >= 2:
            all_days_sorted = sorted(all_days)
            matrix = np.zeros((len(strat_names), len(all_days_sorted)))
            for i, s in enumerate(strat_names):
                for j, d in enumerate(all_days_sorted):
                    matrix[i, j] = strat_daily[s].get(d, 0)

            corr = np.corrcoef(matrix)
            print(f"\n  Daily PnL Correlation Matrix:")
            print(f"  {'':>15}", end="")
            for s in strat_names:
                print(f" {s:>12}", end="")
            print()
            for i, s in enumerate(strat_names):
                print(f"  {s:>15}", end="")
                for j in range(len(strat_names)):
                    print(f" {corr[i,j]:>12.3f}", end="")
                print()

        # Part 3: Overlap analysis
        print(f"\n  Trading Day Overlap:")
        for i, s1 in enumerate(strat_names):
            for j, s2 in enumerate(strat_names):
                if j <= i:
                    continue
                days1 = set(strat_daily[s1].keys())
                days2 = set(strat_daily[s2].keys())
                overlap = days1 & days2
                print(f"  {s1} x {s2}: {len(overlap)} overlap days "
                      f"({100*len(overlap)/max(len(days1),1):.1f}% of {s1})")

        # Part 4: Weight optimization
        print(f"\n  Weight Optimization ({label}):")
        print(f"  {'Scheme':<45} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
        print(f"  {'-'*72}")

        base_pnls = [t.pnl for t in trades]
        base_sh = compute_sharpe(trades, base_pnls)

        # Sharpe-weighted normalization
        total_pos_sharpe = sum(max(0, v) for v in strat_sharpes.values())
        sharpe_weights = {}
        for s in strat_names:
            if total_pos_sharpe > 0:
                sharpe_weights[s] = max(0, strat_sharpes.get(s, 0)) / total_pos_sharpe * len(strat_names)
            else:
                sharpe_weights[s] = 1.0

        # Min Sharpe for "drop worst" scheme
        min_sharpe_val = min(strat_sharpes.values()) if strat_sharpes else 0

        schemes = [
            ("Equal weight (baseline)", {s: 1.0 for s in strat_names}),
            ("Keltner 1.5x, others 0.5x",
             {s: (1.5 if s == 'keltner' else 0.5) for s in strat_names}),
            ("Keltner 2.0x, others 0.3x",
             {s: (2.0 if s == 'keltner' else 0.3) for s in strat_names}),
            ("Keltner only",
             {s: (1.0 if s == 'keltner' else 0.0) for s in strat_names}),
            ("Drop worst Sharpe",
             {s: (0.0 if strat_sharpes.get(s, 0) == min_sharpe_val else 1.0) for s in strat_names}),
            ("Sharpe-weighted", sharpe_weights),
        ]

        for sname, weights in schemes:
            pnls = []
            for t in trades:
                w = weights.get(t.strategy, 1.0)
                pnls.append(t.pnl * w)
            active = [(t, p) for t, p in zip(trades, pnls) if abs(p) > 0.001]
            if not active:
                continue
            active_trades = [x[0] for x in active]
            active_pnls = [x[1] for x in active]
            sh = compute_sharpe(active_trades, active_pnls)
            print(f"  {sname:<45} ${sum(active_pnls):>9,.0f} {sh:>8.2f} {sh-base_sh:>+7.2f}")

        # Part 5: Rolling 30-day dynamic weights
        print(f"\n  Dynamic Weighting (recent 30-day Sharpe):")
        # Group trades by month
        monthly = defaultdict(lambda: defaultdict(list))
        for t in trades:
            m = t.entry_time.strftime('%Y-%m')
            monthly[m][t.strategy].append(t.pnl)

        months_sorted = sorted(monthly.keys())
        if len(months_sorted) > 3:
            dynamic_pnl_total = 0.0
            n_weighted = 0
            for mi in range(1, len(months_sorted)):
                prev_month = months_sorted[mi - 1]
                cur_month = months_sorted[mi]
                # Compute Sharpe for each strat from prev month
                prev_sharpes = {}
                for s in strat_names:
                    pp = monthly[prev_month].get(s, [])
                    if len(pp) >= 3 and np.std(pp) > 0:
                        prev_sharpes[s] = np.mean(pp) / np.std(pp) * np.sqrt(12)
                    else:
                        prev_sharpes[s] = 0
                total_ps = sum(max(0, v) for v in prev_sharpes.values())
                dyn_w = {}
                for s in strat_names:
                    if total_ps > 0:
                        dyn_w[s] = max(0, prev_sharpes[s]) / total_ps * len(strat_names)
                    else:
                        dyn_w[s] = 1.0

                for s in strat_names:
                    for pnl_val in monthly[cur_month].get(s, []):
                        dynamic_pnl_total += pnl_val * dyn_w.get(s, 1.0)
                        n_weighted += 1

            print(f"  Dynamic-weighted PnL: ${dynamic_pnl_total:,.0f} ({n_weighted} trades)")
            print(f"  Equal-weighted PnL:   ${bstats['total_pnl']:,.0f} ({bstats['n']} trades)")

    print(f"\n  EXP53 done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════

experiments = [
    ("EXP48", run_exp48),
    ("EXP49", run_exp49),
    ("EXP50", run_exp50),
    ("EXP51", run_exp51),
    ("EXP52", run_exp52),
    ("EXP53", run_exp53),
]

for name, func in experiments:
    try:
        func()
    except Exception as e:
        print(f"\n  !!! {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
    gc.collect()

total_elapsed = time.time() - t_total
print("\n\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print(f"  Total runtime: {total_elapsed/60:.1f} minutes")
print(f"  Current: Sharpe={baseline_cur['sharpe']:.2f} PnL=${baseline_cur['total_pnl']:,.0f}")
print(f"  Mega:    Sharpe={baseline_mega['sharpe']:.2f} PnL=${baseline_mega['total_pnl']:,.0f}")
print(f"  Output saved to: {OUTPUT_FILE}")
print(f"  Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
