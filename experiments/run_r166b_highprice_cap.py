#!/usr/bin/env python3
"""
R166b — Adaptive Cap 高金价专项验证
====================================
R166 用 2015-2026 全样本验证了 PriceCap base=$20 (比例 0.01) 效果最佳，
但数据中 $4,000+ 的区间占比不大。

本实验聚焦高金价区间，验证：
1) $3,000+ 区间内各 cap 方案的表现
2) $4,000+ 区间（2026年数据）的专项测试
3) 不同比例系数在高金价下的 sensitivity
4) 与当前固定 $35 cap 的直接对比
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import (
    DataBundle, run_variant, LIVE_PARITY_KWARGS,
    load_m15, load_h1_aligned, H1_CSV_PATH,
)

OUTPUT_DIR = Path("results/r166b_highprice_cap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

t0 = time.time()


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _normalize_ts(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR'] = psar; df['PSAR_dir'] = direction
    return df


def _mk(d, entry, exit_p, entry_t, exit_t, pnl, reason):
    return {'dir': d, 'entry': entry, 'exit': exit_p,
            'entry_time': entry_t, 'exit_time': exit_t,
            'pnl': round(pnl, 4), 'reason': reason}


def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']; d = pos['dir']; entry = pos['entry']; bar0 = pos['bar']
    sl_dist = sl_atr * atr; tp_dist = tp_atr * atr
    held = i - bar0
    if d == 'BUY':
        sl_price = entry - sl_dist; tp_price = entry + tp_dist
        if lo <= sl_price:
            raw = (sl_price - entry - spread) * lot * pv
            pnl = max(raw, -cap) if cap > 0 else raw
            return _mk(d, entry, sl_price, times[bar0], times[i], pnl, 'SL')
        if hi >= tp_price:
            raw = (tp_price - entry - spread) * lot * pv
            return _mk(d, entry, tp_price, times[bar0], times[i], raw, 'TP')
        trail_act_dist = trail_act * atr; t_dist = trail_dist * atr
        profit = hi - entry - spread
        if profit >= trail_act_dist:
            trail_stop = hi - t_dist
            if lo <= trail_stop:
                raw = (trail_stop - entry - spread) * lot * pv
                pnl = max(raw, -cap) if cap > 0 else raw
                return _mk(d, entry, trail_stop, times[bar0], times[i], pnl, 'Trail')
        if held >= max_hold:
            raw = (cl - entry - spread) * lot * pv
            pnl = max(raw, -cap) if cap > 0 else raw
            return _mk(d, entry, cl, times[bar0], times[i], pnl, 'Timeout')
    else:
        sl_price = entry + sl_dist; tp_price = entry - tp_dist
        if hi >= sl_price:
            raw = (entry - sl_price - spread) * lot * pv
            pnl = max(raw, -cap) if cap > 0 else raw
            return _mk(d, entry, sl_price, times[bar0], times[i], pnl, 'SL')
        if lo <= tp_price:
            raw = (entry - tp_price - spread) * lot * pv
            return _mk(d, entry, tp_price, times[bar0], times[i], raw, 'TP')
        trail_act_dist = trail_act * atr; t_dist = trail_dist * atr
        profit = entry - lo - spread
        if profit >= trail_act_dist:
            trail_stop = lo + t_dist
            if hi >= trail_stop:
                raw = (entry - trail_stop - spread) * lot * pv
                pnl = max(raw, -cap) if cap > 0 else raw
                return _mk(d, entry, trail_stop, times[bar0], times[i], pnl, 'Trail')
        if held >= max_hold:
            raw = (entry - cl - spread) * lot * pv
            pnl = max(raw, -cap) if cap > 0 else raw
            return _mk(d, entry, cl, times[bar0], times[i], pnl, 'Timeout')
    return None


def bt_psar(h1_df, spread, lot, maxloss_cap=0):
    df = add_psar(h1_df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; pdir = df['PSAR_dir'].values; times = df.index; n = len(df)
    trades = []; pos = None; prev_dir = pdir[0]; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        3.5, 3.0, 0.14, 0.025, 12, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i] != prev_dir:
            if pdir[i] == 1:
                pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            else:
                pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        prev_dir = pdir[i]
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0, slow=60):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['SMA'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR', 'SMA'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; sma = df['SMA'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        3.5, 3.0, 0.14, 0.025, 12, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if c[i] > sma[i] and c[i-1] <= sma[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sma[i] and c[i-1] >= sma[i-1]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(5, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        3.5, 3.0, 0.14, 0.025, 12, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        utc_hour = times[i].hour if hasattr(times[i], 'hour') else pd.Timestamp(times[i]).hour
        if utc_hour != 8: continue
        sess_high = max(h[i-j] for j in range(1, 5))
        sess_low = min(lo[i-j] for j in range(1, 5))
        if c[i] > sess_high:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < sess_low:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS,
          'keltner_max_hold_m15': 8,
          'maxloss_cap': maxloss_cap}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    trades = []
    for t in result.get('_trades', []):
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason,
        })
    return trades


def merge_portfolio_trades(strat_trades_dict, lot_scale=None):
    if lot_scale is None:
        lot_scale = R89_LOTS
    all_trades = []
    for strat_name, trades in strat_trades_dict.items():
        mult = lot_scale.get(strat_name, UNIT_LOT) / UNIT_LOT
        for t in trades:
            all_trades.append({
                'strategy': strat_name,
                'dir': t['dir'], 'entry': t['entry'], 'exit': t['exit'],
                'entry_time': _normalize_ts(t['entry_time']),
                'exit_time': _normalize_ts(t['exit_time']),
                'pnl': t['pnl'] * mult,
                'pnl_unit': t['pnl'],
                'reason': t['reason'],
            })
    all_trades.sort(key=lambda x: x['exit_time'])
    return all_trades


def trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['exit_time']).dt.date
    return df.groupby('date')['pnl'].sum()


def sharpe(daily):
    if len(daily) < 2: return 0
    m = daily.mean(); s = daily.std()
    return round(m / s * np.sqrt(252), 3) if s > 0 else 0


def max_dd(daily):
    cum = daily.cumsum(); peak = cum.cummax()
    dd = cum - peak
    return round(abs(dd.min()), 2) if len(dd) > 0 else 0


def compute_stats(trades, label=""):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0,
                'max_dd': 0, 'avg_pnl': 0, 'worst_trade': 0}
    pnls = np.array([t['pnl'] for t in trades])
    daily = trades_to_daily(trades)
    return {
        'label': label,
        'n': len(trades),
        'sharpe': sharpe(daily),
        'pnl': round(float(pnls.sum()), 2),
        'wr': round(100 * np.mean(pnls > 0), 1),
        'max_dd': max_dd(daily),
        'avg_pnl': round(float(pnls.mean()), 4),
        'worst_trade': round(float(pnls.min()), 2),
    }


def apply_price_cap(trades, base_cap, ref_price=2000):
    """Post-process: clip losing trades by price-proportional cap."""
    adjusted = []
    for t in trades:
        entry_price = t['entry']
        dynamic_cap = base_cap * (entry_price / ref_price)
        scaled_cap = dynamic_cap * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
        adj = dict(t)
        if adj['pnl'] < -scaled_cap:
            adj['pnl'] = -scaled_cap
        adjusted.append(adj)
    return adjusted


def apply_fixed_cap(trades, fixed_cap=35):
    """Post-process: clip losing trades by fixed dollar cap (scaled by lot)."""
    adjusted = []
    for t in trades:
        scaled_cap = fixed_cap * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
        adj = dict(t)
        if adj['pnl'] < -scaled_cap:
            adj['pnl'] = -scaled_cap
        adjusted.append(adj)
    return adjusted


def main():
    print("=" * 80)
    print("  R166b: Adaptive Cap High-Price Validation")
    print("  Focus: Gold $3,000+ and $4,000+ regimes")
    print("=" * 80, flush=True)

    # Load data
    print("\n  Loading data...", flush=True)
    bundle = DataBundle.load_custom()
    h1_df = load_h1_aligned(H1_CSV_PATH, bundle.m15_df.index[0])
    print(f"  H1 range: {h1_df.index[0]} -> {h1_df.index[-1]}", flush=True)
    print(f"  H1 last close: ${h1_df['Close'].iloc[-1]:.2f}", flush=True)

    # Run all 4 strategies with NO cap (cap=0 means unlimited)
    print("\n  Running all 4 strategies with NO cap (raw PnL)...", flush=True)
    strat_raw = {}
    strat_raw['L8_MAX'] = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=0)
    strat_raw['PSAR'] = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0)
    strat_raw['TSMOM'] = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0)
    strat_raw['SESS_BO'] = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0)
    portfolio_raw = merge_portfolio_trades(strat_raw)
    print(f"  Total raw trades: {len(portfolio_raw)}", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Price regime breakdown with different caps
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 1: Performance by Price Regime")
    print("=" * 80, flush=True)

    regimes = [
        ("<$2000", 0, 2000),
        ("$2000-3000", 2000, 3000),
        ("$3000-3500", 3000, 3500),
        ("$3500-4000", 3500, 4000),
        ("$4000-4500", 4000, 4500),
        ("$4500-5000", 4500, 5000),
    ]

    cap_methods = {
        'Fixed_35': lambda t: 35 * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT),
        'Price_0.01': lambda t: 0.01 * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT),
        'Price_0.0075': lambda t: 0.0075 * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT),
        'Price_0.0125': lambda t: 0.0125 * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT),
        'Price_0.015': lambda t: 0.015 * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT),
        'No_Cap': lambda t: 999999,
    }

    phase1 = {}
    for rname, lo_price, hi_price in regimes:
        regime_trades_raw = [t for t in portfolio_raw if lo_price <= t['entry'] < hi_price]
        if not regime_trades_raw:
            phase1[rname] = {'n': 0}
            continue

        regime_result = {'n': len(regime_trades_raw), 'price_range': f"${lo_price}-${hi_price}"}
        for cap_name, cap_fn in cap_methods.items():
            capped = []
            for t in regime_trades_raw:
                adj = dict(t)
                cap_val = cap_fn(t)
                if adj['pnl'] < -cap_val:
                    adj['pnl'] = -cap_val
                capped.append(adj)
            stats = compute_stats(capped, f"{rname}_{cap_name}")
            regime_result[cap_name] = stats

        phase1[rname] = regime_result

    print(f"\n    {'Regime':<14} {'N':>6}  {'Fixed$35':>10} {'P_0.01':>10} {'P_0.0075':>10} {'P_0.0125':>10} {'NoCap':>10}")
    print("    " + "-" * 80)
    for rname, data in phase1.items():
        if data['n'] == 0: continue
        f35  = data.get('Fixed_35', {}).get('sharpe', 0)
        p10  = data.get('Price_0.01', {}).get('sharpe', 0)
        p75  = data.get('Price_0.0075', {}).get('sharpe', 0)
        p125 = data.get('Price_0.0125', {}).get('sharpe', 0)
        nc   = data.get('No_Cap', {}).get('sharpe', 0)
        print(f"    {rname:<14} {data['n']:>6}  {f35:>10.3f} {p10:>10.3f} {p75:>10.3f} {p125:>10.3f} {nc:>10.3f}",
              flush=True)

    results['phase1_regime'] = phase1

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Full sample - sweep ratio coefficients
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 2: Full Sample Ratio Sweep")
    print("=" * 80, flush=True)

    ratios = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025]
    phase2 = {}

    print(f"\n    {'Ratio':<8} {'N':>6} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>8} {'WR':>6} {'Worst':>10} {'Cap@4700':>10}")
    print("    " + "-" * 78)

    for ratio in ratios:
        capped = []
        for t in portfolio_raw:
            adj = dict(t)
            cap_val = ratio * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
            if adj['pnl'] < -cap_val:
                adj['pnl'] = -cap_val
            capped.append(adj)
        stats = compute_stats(capped, f"ratio_{ratio}")
        stats['ratio'] = ratio
        stats['cap_at_4700'] = round(ratio * 4700, 2)
        phase2[str(ratio)] = stats
        print(f"    {ratio:<8} {stats['n']:>6} {stats['sharpe']:>8.3f} {stats['pnl']:>12.2f} "
              f"{stats['max_dd']:>8.2f} {stats['wr']:>5.1f}% {stats['worst_trade']:>10.2f} "
              f"${stats['cap_at_4700']:>8.2f}", flush=True)

    # Add fixed $35 for comparison
    capped_35 = apply_fixed_cap(portfolio_raw, 35)
    stats_35 = compute_stats(capped_35, "Fixed_$35")
    stats_35['cap_at_4700'] = 35.0
    phase2['fixed_35'] = stats_35
    print(f"    {'Fix$35':<8} {stats_35['n']:>6} {stats_35['sharpe']:>8.3f} {stats_35['pnl']:>12.2f} "
          f"{stats_35['max_dd']:>8.2f} {stats_35['wr']:>5.1f}% {stats_35['worst_trade']:>10.2f} "
          f"${stats_35['cap_at_4700']:>8.2f}", flush=True)

    results['phase2_ratio_sweep'] = phase2

    # ══════════════════════════════════════════════════════════════
    # Phase 3: High price zone only ($3000+)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 3: $3,000+ Zone Only")
    print("=" * 80, flush=True)

    high_trades = [t for t in portfolio_raw if t['entry'] >= 3000]
    print(f"  Trades with entry >= $3,000: {len(high_trades)}", flush=True)

    phase3 = {'n_total': len(high_trades)}
    if high_trades:
        avg_price = np.mean([t['entry'] for t in high_trades])
        phase3['avg_entry_price'] = round(avg_price, 2)

        print(f"  Average entry price: ${avg_price:.2f}")
        print(f"\n    {'Method':<14} {'Sharpe':>8} {'PnL':>10} {'MaxDD':>8} {'Worst':>10}")
        print("    " + "-" * 56)

        for ratio in [0.0075, 0.01, 0.0125, 0.015]:
            capped = []
            for t in high_trades:
                adj = dict(t)
                cap_val = ratio * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
                if adj['pnl'] < -cap_val: adj['pnl'] = -cap_val
                capped.append(adj)
            stats = compute_stats(capped, f"P_{ratio}")
            phase3[f'ratio_{ratio}'] = stats
            print(f"    P_{ratio:<10} {stats['sharpe']:>8.3f} {stats['pnl']:>10.2f} "
                  f"{stats['max_dd']:>8.2f} {stats['worst_trade']:>10.2f}", flush=True)

        # Fixed $35
        capped_35h = []
        for t in high_trades:
            adj = dict(t)
            cap_val = 35 * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
            if adj['pnl'] < -cap_val: adj['pnl'] = -cap_val
            capped_35h.append(adj)
        stats_35h = compute_stats(capped_35h, "Fixed_$35")
        phase3['fixed_35'] = stats_35h
        print(f"    {'Fixed_$35':<14} {stats_35h['sharpe']:>8.3f} {stats_35h['pnl']:>10.2f} "
              f"{stats_35h['max_dd']:>8.2f} {stats_35h['worst_trade']:>10.2f}", flush=True)

    results['phase3_high_price'] = phase3

    # ══════════════════════════════════════════════════════════════
    # Phase 4: $4,000+ zone (most recent data, closest to current)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 4: $4,000+ Zone (Current Market)")
    print("=" * 80, flush=True)

    ultra_trades = [t for t in portfolio_raw if t['entry'] >= 4000]
    print(f"  Trades with entry >= $4,000: {len(ultra_trades)}", flush=True)

    phase4 = {'n_total': len(ultra_trades)}
    if ultra_trades:
        avg_price4 = np.mean([t['entry'] for t in ultra_trades])
        min_price4 = min(t['entry'] for t in ultra_trades)
        max_price4 = max(t['entry'] for t in ultra_trades)
        phase4['avg_entry_price'] = round(avg_price4, 2)
        phase4['min_entry_price'] = round(min_price4, 2)
        phase4['max_entry_price'] = round(max_price4, 2)

        print(f"  Price range: ${min_price4:.2f} - ${max_price4:.2f}, avg ${avg_price4:.2f}")

        print(f"\n    {'Method':<14} {'Cap@avg':>8} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'MaxDD':>8} {'WR':>6} {'AvgPnL':>8} {'Worst':>10}")
        print("    " + "-" * 90)

        for ratio in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]:
            capped = []
            for t in ultra_trades:
                adj = dict(t)
                cap_val = ratio * t['entry'] * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
                if adj['pnl'] < -cap_val: adj['pnl'] = -cap_val
                capped.append(adj)
            stats = compute_stats(capped, f"P_{ratio}")
            stats['cap_at_avg'] = round(ratio * avg_price4, 2)
            phase4[f'ratio_{ratio}'] = stats
            print(f"    P_{ratio:<10} ${stats['cap_at_avg']:>6.1f} {stats['n']:>6} {stats['sharpe']:>8.3f} "
                  f"{stats['pnl']:>10.2f} {stats['max_dd']:>8.2f} {stats['wr']:>5.1f}% "
                  f"{stats['avg_pnl']:>8.4f} {stats['worst_trade']:>10.2f}", flush=True)

        # Fixed $35
        capped_35u = []
        for t in ultra_trades:
            adj = dict(t)
            cap_val = 35 * (R89_LOTS.get(t.get('strategy', 'L8_MAX'), 0.02) / UNIT_LOT)
            if adj['pnl'] < -cap_val: adj['pnl'] = -cap_val
            capped_35u.append(adj)
        stats_35u = compute_stats(capped_35u, "Fixed_$35")
        stats_35u['cap_at_avg'] = 35.0
        phase4['fixed_35'] = stats_35u
        print(f"    {'Fixed_$35':<14} ${35.0:>6.1f} {stats_35u['n']:>6} {stats_35u['sharpe']:>8.3f} "
              f"{stats_35u['pnl']:>10.2f} {stats_35u['max_dd']:>8.2f} {stats_35u['wr']:>5.1f}% "
              f"{stats_35u['avg_pnl']:>8.4f} {stats_35u['worst_trade']:>10.2f}", flush=True)

        # No cap
        stats_nc = compute_stats(ultra_trades, "No_Cap")
        phase4['no_cap'] = stats_nc
        print(f"    {'No_Cap':<14} {'inf':>8} {stats_nc['n']:>6} {stats_nc['sharpe']:>8.3f} "
              f"{stats_nc['pnl']:>10.2f} {stats_nc['max_dd']:>8.2f} {stats_nc['wr']:>5.1f}% "
              f"{stats_nc['avg_pnl']:>8.4f} {stats_nc['worst_trade']:>10.2f}", flush=True)

    results['phase4_ultra_high'] = phase4

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Loss analysis at $4,000+
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 5: Loss Distribution at $4,000+")
    print("=" * 80, flush=True)

    if ultra_trades:
        losses = [t for t in ultra_trades if t['pnl'] < 0]
        loss_pnls = [t['pnl'] for t in losses]
        print(f"  Total trades: {len(ultra_trades)}, Losses: {len(losses)}", flush=True)
        if losses:
            loss_arr = np.array(loss_pnls)
            loss_entries = [t['entry'] for t in losses]
            phase5 = {
                'n_losses': len(losses),
                'avg_loss': round(float(loss_arr.mean()), 2),
                'median_loss': round(float(np.median(loss_arr)), 2),
                'worst_loss': round(float(loss_arr.min()), 2),
                'p10_loss': round(float(np.percentile(loss_arr, 10)), 2),
                'p25_loss': round(float(np.percentile(loss_arr, 25)), 2),
                'avg_loss_entry_price': round(float(np.mean(loss_entries)), 2),
            }
            print(f"  Avg loss:    ${phase5['avg_loss']}")
            print(f"  Median loss: ${phase5['median_loss']}")
            print(f"  Worst loss:  ${phase5['worst_loss']}")
            print(f"  P10 (tail):  ${phase5['p10_loss']}")
            print(f"  P25:         ${phase5['p25_loss']}", flush=True)

            # What cap would clip the worst 10%?
            p90_loss = abs(np.percentile(loss_arr, 10))
            optimal_ratio = p90_loss / np.mean(loss_entries)
            phase5['p90_loss_abs'] = round(p90_loss, 2)
            phase5['implied_ratio_p90'] = round(optimal_ratio, 6)
            print(f"\n  To clip worst 10% of losses:")
            print(f"    P90 abs loss = ${p90_loss:.2f}")
            print(f"    Implied ratio = {optimal_ratio:.4f} (at avg entry ${np.mean(loss_entries):.0f})")
            print(f"    This means cap = ${optimal_ratio * 4700:.2f} at $4,700", flush=True)
        else:
            phase5 = {'n_losses': 0}
    else:
        phase5 = {'n_trades': 0}
    results['phase5_loss_distribution'] = phase5

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Recommendation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  Phase 6: Recommendation for $4,700 Gold")
    print("=" * 80, flush=True)

    if phase4.get('n_total', 0) > 0:
        best_ratio = None; best_sharpe = 0
        for ratio in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]:
            s = phase4.get(f'ratio_{ratio}', {}).get('sharpe', 0)
            if s > best_sharpe:
                best_sharpe = s; best_ratio = ratio
        
        fixed_sharpe = phase4.get('fixed_35', {}).get('sharpe', 0)
        
        results['recommendation'] = {
            'best_ratio': best_ratio,
            'best_sharpe_at_4000plus': best_sharpe,
            'fixed_35_sharpe_at_4000plus': fixed_sharpe,
            'improvement_pct': round((best_sharpe / fixed_sharpe - 1) * 100, 1) if fixed_sharpe > 0 else 0,
            'cap_at_4700': round(best_ratio * 4700, 2) if best_ratio else 0,
        }
        print(f"  Best ratio at $4,000+: {best_ratio}")
        print(f"  Cap at $4,700: ${best_ratio * 4700:.2f}" if best_ratio else "  No best ratio found")
        print(f"  Sharpe: {best_sharpe:.3f} vs Fixed $35: {fixed_sharpe:.3f}")
        print(f"  Improvement: +{results['recommendation']['improvement_pct']:.1f}%", flush=True)

    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n  R166b completed in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    with open(OUTPUT_DIR / "r166b_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r166b_results.json", flush=True)


if __name__ == "__main__":
    main()
