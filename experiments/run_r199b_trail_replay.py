#!/usr/bin/env python3
"""
R199b - 实盘 Keltner 交易 M15 Trail 回放对比
=============================================
读取 gold_trade_log.json 中所有 Keltner OPEN/CLOSE 对，
用 M15 数据逐 bar 模拟 3 套 trail 参数的 exit 行为。

不修改任何实盘代码，纯分析脚本。
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

warnings.filterwarnings('ignore')

TRADE_LOG = Path(r"c:\Users\hlin2\gold-quant-trading\data\gold_trade_log.json")
M15_FILE = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
M15_UPDATE = Path("data/download/temp_m15_update/xauusd-m15-bid-2026-04-24-2026-05-05T14-57.csv")

PV = 100
SPREAD = 0.30

TRAIL_CONFIGS = [
    {'name': 'current_0.02_0.005', 'act': 0.02, 'dist': 0.005},
    {'name': 'old_0.06_0.01',      'act': 0.06, 'dist': 0.01},
    {'name': 'cand_0.08_0.003',    'act': 0.08, 'dist': 0.003},
]

SL_ATR = 6.0
TP_ATR = 8.0
CAP = 70.0
MAX_HOLD_M15 = 8  # 2 H1 bars = 8 M15 bars


def load_m15():
    df = pd.read_csv(M15_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    if M15_UPDATE.exists():
        df2 = pd.read_csv(M15_UPDATE)
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], unit='ms', utc=True)
        df2 = df2.set_index('timestamp')
        df2.index = df2.index.tz_localize(None)
        df2.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        df2 = df2[['Open', 'High', 'Low', 'Close']].copy()
        df = pd.concat([df, df2[~df2.index.isin(df.index)]]).sort_index()

    print(f"M15: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})")
    return df


def load_trades():
    with open(TRADE_LOG, encoding='utf-8') as f:
        log = json.load(f)

    opens = {}
    for t in log:
        if t.get('strategy') not in ('keltner', 'l8_max'):
            continue
        if t.get('action') == 'OPEN':
            key = t.get('time', '')
            opens[key] = t

    pairs = []
    used_opens = set()
    for t in log:
        if t.get('strategy') not in ('keltner', 'l8_max'):
            continue
        if t.get('action') != 'CLOSE':
            continue

        entry_price = t.get('open_price')
        close_price = t.get('close_price')
        profit = t.get('profit')
        direction = t.get('direction')
        close_time = t.get('time', '')
        reason = t.get('reason', '')
        hold_hours = t.get('hold_hours', t.get('hold_days', 0))
        lots = t.get('lots', 0.04)
        ticket = t.get('ticket')

        if not entry_price or not close_price:
            continue

        best_open = None
        best_dist = float('inf')
        for okey, o in opens.items():
            if okey in used_opens:
                continue
            op = o.get('price', o.get('entry_price', 0))
            if abs(op - entry_price) < best_dist:
                best_dist = abs(op - entry_price)
                best_open = (okey, o)

        atr = 0
        entry_time_str = close_time
        if best_open:
            okey, o = best_open
            used_opens.add(okey)
            atr = o.get('factors', {}).get('ATR', 0)
            entry_time_str = o.get('time', close_time)
            if not direction:
                direction = o.get('direction')

        if not direction:
            direction = 'BUY' if profit and profit > 0 and close_price > entry_price else 'SELL'

        if atr <= 0:
            atr = 19.0  # approximate current ATR

        pairs.append({
            'entry_price': entry_price,
            'close_price': close_price,
            'direction': direction,
            'actual_profit': profit,
            'actual_reason': reason,
            'atr': atr,
            'lots': lots,
            'entry_time': entry_time_str,
            'close_time': close_time,
            'hold_hours': hold_hours,
            'ticket': ticket,
        })

    print(f"Keltner trade pairs: {len(pairs)}")
    return pairs


def simulate_exit_m15(m15, entry_time_str, entry_price, direction, atr, lots, trail_act, trail_dist):
    """Simulate exit on M15 bars for one trade with given trail params."""
    try:
        entry_time = pd.Timestamp(entry_time_str)
    except:
        return None

    if entry_time.tzinfo:
        entry_time = entry_time.tz_localize(None)

    start_idx = m15.index.searchsorted(entry_time)
    if start_idx >= len(m15) - 1:
        return None

    sl_pts = SL_ATR * atr
    tp_pts = TP_ATR * atr
    cap_pts = CAP / (lots * PV) if lots > 0 else 999

    extreme = entry_price
    trail_price = 0

    for j in range(start_idx + 1, min(start_idx + MAX_HOLD_M15 + 1, len(m15))):
        h = m15['High'].iloc[j]
        lo = m15['Low'].iloc[j]
        c = m15['Close'].iloc[j]
        bar_time = m15.index[j]

        if direction == 'BUY':
            pnl_pts = c - entry_price - SPREAD
            pnl_hi = h - entry_price - SPREAD
            pnl_lo = lo - entry_price - SPREAD
        else:
            pnl_pts = entry_price - c - SPREAD
            pnl_hi = entry_price - lo - SPREAD
            pnl_lo = entry_price - h - SPREAD

        pnl_dollar = pnl_pts * lots * PV

        # TP
        if pnl_hi * lots * PV >= tp_pts * lots * PV:
            return {'reason': 'TP', 'pnl': tp_pts * lots * PV, 'bars': j - start_idx, 'time': str(bar_time)}

        # SL
        if pnl_lo * lots * PV <= -sl_pts * lots * PV:
            return {'reason': 'SL', 'pnl': -sl_pts * lots * PV, 'bars': j - start_idx, 'time': str(bar_time)}

        # Cap
        if pnl_dollar < -CAP:
            return {'reason': 'Cap', 'pnl': -CAP, 'bars': j - start_idx, 'time': str(bar_time)}

        # Trail
        ad = trail_act * atr
        tdd = trail_dist * atr

        if direction == 'BUY':
            extreme = max(extreme, h)
            if extreme - entry_price >= ad:
                new_trail = extreme - tdd
                trail_price = max(trail_price, new_trail)
                if lo <= trail_price:
                    trail_pnl = (trail_price - entry_price - SPREAD) * lots * PV
                    return {'reason': 'Trail', 'pnl': round(trail_pnl, 2), 'bars': j - start_idx, 'time': str(bar_time)}
        else:
            extreme = min(extreme, lo) if extreme > 0 else lo
            if entry_price - extreme >= ad:
                new_trail = extreme + tdd
                if trail_price == 0:
                    trail_price = new_trail
                else:
                    trail_price = min(trail_price, new_trail)
                if h >= trail_price:
                    trail_pnl = (entry_price - trail_price - SPREAD) * lots * PV
                    return {'reason': 'Trail', 'pnl': round(trail_pnl, 2), 'bars': j - start_idx, 'time': str(bar_time)}

    # Timeout
    end_j = min(start_idx + MAX_HOLD_M15, len(m15) - 1)
    c_end = m15['Close'].iloc[end_j]
    if direction == 'BUY':
        pnl = (c_end - entry_price - SPREAD) * lots * PV
    else:
        pnl = (entry_price - c_end - SPREAD) * lots * PV
    return {'reason': 'Timeout', 'pnl': round(pnl, 2), 'bars': MAX_HOLD_M15, 'time': str(m15.index[end_j])}


def main():
    print("=" * 80)
    print("  R199b - M15 Trail 回放对比 (实盘入场点)")
    print("=" * 80)
    print()

    m15 = load_m15()
    trades = load_trades()
    print()

    results = {cfg['name']: [] for cfg in TRAIL_CONFIGS}
    actual_pnls = []
    detail_lines = []

    for i, trade in enumerate(trades):
        ep = trade['entry_price']
        d = trade['direction']
        atr = trade['atr']
        lots = trade['lots']
        et = trade['entry_time']
        actual_pnl = trade['actual_profit'] or 0
        actual_reason = trade['actual_reason']
        actual_pnls.append(actual_pnl)

        # Classify actual reason
        ar_short = 'Other'
        if 'Trailing' in actual_reason: ar_short = 'Trail'
        elif 'MaxLoss' in actual_reason or 'Cap' in actual_reason: ar_short = 'Cap'
        elif 'SL' in actual_reason or '止损' in actual_reason: ar_short = 'SL'
        elif '时间' in actual_reason or 'Timeout' in actual_reason: ar_short = 'Timeout'
        elif '出场' in actual_reason: ar_short = 'Signal'

        line = f"  #{i+1:>3} {et[:16]} {d:>4}@{ep:.2f} ATR={atr:.1f}  actual: ${actual_pnl:>7.2f} ({ar_short})"

        for cfg in TRAIL_CONFIGS:
            sim = simulate_exit_m15(m15, et, ep, d, atr, lots, cfg['act'], cfg['dist'])
            if sim:
                results[cfg['name']].append(sim)
                line += f"  | {cfg['name'][:12]:>12}: ${sim['pnl']:>7.2f} ({sim['reason']:>7}, {sim['bars']}b)"
            else:
                results[cfg['name']].append({'reason': 'NoData', 'pnl': 0, 'bars': 0})
                line += f"  | {cfg['name'][:12]:>12}: NO DATA"

        detail_lines.append(line)

    # Print details
    print("=" * 80)
    print("  TRADE-BY-TRADE DETAIL")
    print("=" * 80)
    for line in detail_lines:
        print(line)

    # Summary
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    n = len(trades)
    act_total = sum(actual_pnls)
    act_wins = sum(1 for p in actual_pnls if p > 0)
    act_losses = [p for p in actual_pnls if p < 0]
    act_wins_list = [p for p in actual_pnls if p > 0]
    act_avg_win = np.mean(act_wins_list) if act_wins_list else 0
    act_avg_loss = np.mean(act_losses) if act_losses else 0
    act_rr = abs(act_avg_win / act_avg_loss) if act_avg_loss != 0 else 0

    print(f"\n  {'':>20} {'Total PnL':>10} {'AvgPnL':>8} {'WR%':>6} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'Trail%':>7} {'Cap%':>6} {'Timeout%':>9}")
    print(f"  {'-'*100}")

    # Actual
    act_trail_n = sum(1 for p in actual_pnls for t in trades if 'Trailing' in t.get('actual_reason',''))
    print(f"  {'ACTUAL (live)':>20} ${act_total:>9.2f} ${np.mean(actual_pnls):>7.2f} {act_wins/n*100:>5.1f}% ${act_avg_win:>7.2f} ${act_avg_loss:>7.2f} {act_rr:>5.2f} {'':>7} {'':>6} {'':>9}")

    for cfg in TRAIL_CONFIGS:
        name = cfg['name']
        pnls = [r['pnl'] for r in results[name]]
        reasons = Counter(r['reason'] for r in results[name])
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [p for p in pnls if p < 0]
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        trail_pct = reasons.get('Trail', 0) / n * 100
        cap_pct = reasons.get('Cap', 0) / n * 100
        timeout_pct = reasons.get('Timeout', 0) / n * 100
        valid = sum(1 for r in results[name] if r['reason'] != 'NoData')

        print(f"  {name:>20} ${total_pnl:>9.2f} ${np.mean(pnls):>7.2f} {wins/n*100:>5.1f}% ${avg_win:>7.2f} ${avg_loss:>7.2f} {rr:>5.2f} {trail_pct:>6.1f}% {cap_pct:>5.1f}% {timeout_pct:>8.1f}%")

    # Per-trade comparison: how often each config beats actual
    print(f"\n  === PER-TRADE COMPARISON vs ACTUAL ===")
    for cfg in TRAIL_CONFIGS:
        name = cfg['name']
        better = sum(1 for j in range(n) if results[name][j]['pnl'] > actual_pnls[j])
        same = sum(1 for j in range(n) if abs(results[name][j]['pnl'] - actual_pnls[j]) < 0.01)
        worse = n - better - same
        better_sum = sum(results[name][j]['pnl'] - actual_pnls[j] for j in range(n) if results[name][j]['pnl'] > actual_pnls[j])
        worse_sum = sum(results[name][j]['pnl'] - actual_pnls[j] for j in range(n) if results[name][j]['pnl'] < actual_pnls[j])
        print(f"  {name:>20}: better {better}/{n} (+${better_sum:.2f}), worse {worse}/{n} (${worse_sum:.2f}), same {same}/{n}")

    # Trail-only analysis (only trades where actual was Trail exit)
    trail_trades = [j for j in range(n) if 'Trailing' in trades[j].get('actual_reason', '')]
    if trail_trades:
        print(f"\n  === TRAIL-EXIT TRADES ONLY ({len(trail_trades)} trades) ===")
        act_trail_pnls = [actual_pnls[j] for j in trail_trades]
        print(f"  {'ACTUAL':>20}: total=${sum(act_trail_pnls):.2f}, avg=${np.mean(act_trail_pnls):.2f}")
        for cfg in TRAIL_CONFIGS:
            name = cfg['name']
            sim_pnls = [results[name][j]['pnl'] for j in trail_trades]
            sim_reasons = Counter(results[name][j]['reason'] for j in trail_trades)
            delta = sum(sim_pnls) - sum(act_trail_pnls)
            print(f"  {name:>20}: total=${sum(sim_pnls):.2f}, avg=${np.mean(sim_pnls):.2f}, delta=${delta:+.2f}, exits={dict(sim_reasons)}")

    # Cap/SL trades analysis (these should be identical across configs)
    cap_trades = [j for j in range(n) if 'MaxLoss' in trades[j].get('actual_reason', '') or 'Cap' in trades[j].get('actual_reason', '')]
    if cap_trades:
        print(f"\n  === CAP/SL TRADES ({len(cap_trades)} trades) ===")
        act_cap_pnls = [actual_pnls[j] for j in cap_trades]
        print(f"  {'ACTUAL':>20}: total=${sum(act_cap_pnls):.2f}, avg=${np.mean(act_cap_pnls):.2f}")
        for cfg in TRAIL_CONFIGS:
            name = cfg['name']
            sim_pnls = [results[name][j]['pnl'] for j in cap_trades]
            print(f"  {name:>20}: total=${sum(sim_pnls):.2f}, avg=${np.mean(sim_pnls):.2f}")

    # Hold duration comparison
    print(f"\n  === AVERAGE HOLD DURATION (M15 bars) ===")
    for cfg in TRAIL_CONFIGS:
        name = cfg['name']
        bars = [r['bars'] for r in results[name] if r['reason'] != 'NoData']
        if bars:
            print(f"  {name:>20}: avg={np.mean(bars):.1f} bars ({np.mean(bars)*15:.0f} min), median={np.median(bars):.0f}")

    print(f"\n{'='*80}")
    print(f"  DONE - {n} trades analyzed")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
