#!/usr/bin/env python3
"""R211: Live EA Trade Log vs Backtest Reconciliation
======================================================
Deep comparison of actual EA trades against backtest predictions.
Uses gold_trade_log.json from the live trading system.

Phase 1 — Trade Statistics Overview
  Per-strategy summary: n_trades, PnL, win_rate, avg_pnl, exit reasons

Phase 2 — Slippage Analysis
  For trades with signal_price vs open_price: entry slippage distribution
  For trades with close_signal_price vs close_price: exit slippage

Phase 3 — Per-Strategy Deep Dive
  Keltner: avg hold time, exit reason distribution, MaxLoss Cap frequency
  TSMOM: signal quality, filter blocks
  Other strategies: PSAR, Chandelier, DualThrust, SESS_BO

Phase 4 — Backtest vs Live PnL Comparison
  Run backtest over the same date range as live data (2026-03-25 to 2026-05-13)
  Compare per-strategy PnL

Phase 5 — Loss Profile Analysis
  Worst trades, clustering of losses, drawdown episodes

Run: python experiments/run_r211_live_reconciliation.py
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path("results/r211_live_reconciliation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG_PATHS = [
    Path(r"c:\Users\hlin2\gold-quant-trading\data\gold_trade_log.json"),
    Path("data/gold_trade_log.json"),
    Path("/root/gold-quant-research/data/gold_trade_log.json"),
]


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')


def load_trade_log() -> List[Dict]:
    for p in TRADE_LOG_PATHS:
        if p.exists():
            print(f'  Found trade log: {p}')
            with open(p, encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"Trade log not found in: {TRADE_LOG_PATHS}")


def main():
    t_start = time.time()
    print('=' * 80)
    print('R211: Live EA Trade Log vs Backtest Reconciliation')
    print('=' * 80)

    print('\nLoading trade log...')
    log = load_trade_log()
    print(f'  Total entries: {len(log)}')

    opens = [e for e in log if e.get('action') == 'OPEN']
    closes = [e for e in log if e.get('action') == 'CLOSE']
    print(f'  OPEN: {len(opens)}, CLOSE: {len(closes)}')

    if closes:
        first_close = closes[0].get('time', '')[:10]
        last_close = closes[-1].get('time', '')[:10]
        print(f'  Date range: {first_close} -> {last_close}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Trade Statistics Overview
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Trade Statistics Overview')
    print('=' * 80)

    strat_stats = defaultdict(lambda: {
        'n': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0.0,
        'pnls': [], 'reasons': defaultdict(int), 'hold_hours': [],
    })

    for t in closes:
        strat = t.get('strategy', 'unknown')
        pnl = t.get('profit', 0)
        reason = t.get('reason', 'unknown')

        s = strat_stats[strat]
        s['n'] += 1
        s['total_pnl'] += pnl
        s['pnls'].append(pnl)
        if pnl > 0:
            s['wins'] += 1
        else:
            s['losses'] += 1

        # Categorize exit reason
        reason_lower = reason.lower()
        if 'maxloss' in reason_lower or 'cap' in reason_lower:
            s['reasons']['MaxLoss_Cap'] += 1
        elif 'trailing' in reason_lower or 'trail' in reason_lower:
            s['reasons']['Trail'] += 1
        elif 'sl' in reason_lower or 'stop' in reason_lower:
            s['reasons']['SL'] += 1
        elif 'tp' in reason_lower or 'take' in reason_lower:
            s['reasons']['TP'] += 1
        elif 'timeout' in reason_lower or '超时' in reason_lower:
            s['reasons']['Timeout'] += 1
        elif 'keltner' in reason_lower and ('中轨' in reason or '出场' in reason):
            s['reasons']['KC_Mid_Exit'] += 1
        elif 'chandelier' in reason_lower:
            s['reasons']['Chandelier_Exit'] += 1
        elif 'friday' in reason_lower or '周五' in reason_lower:
            s['reasons']['Friday_Close'] += 1
        else:
            s['reasons']['Other'] += 1

        hh = t.get('hold_hours', t.get('hold_days', 0))
        if isinstance(hh, (int, float)):
            s['hold_hours'].append(abs(hh))

    phase1 = {}
    print(f'\n  {"Strategy":<15} {"N":>5} {"PnL":>10} {"WR%":>7} {"AvgPnL":>8} {"AvgHold":>8} {"Cap%":>6}')
    for strat in sorted(strat_stats.keys()):
        s = strat_stats[strat]
        avg_pnl = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
        wr = 100 * s['wins'] / s['n'] if s['n'] > 0 else 0
        avg_hold = np.mean(s['hold_hours']) if s['hold_hours'] else 0
        cap_pct = 100 * s['reasons'].get('MaxLoss_Cap', 0) / s['n'] if s['n'] > 0 else 0

        print(f'  {strat:<15} {s["n"]:>5} {s["total_pnl"]:>10.2f} {wr:>6.1f}% '
              f'{avg_pnl:>8.2f} {avg_hold:>7.1f}h {cap_pct:>5.1f}%')

        phase1[strat] = {
            'n': s['n'],
            'total_pnl': round(s['total_pnl'], 2),
            'win_rate': round(wr, 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_hold_hours': round(avg_hold, 2),
            'cap_pct': round(cap_pct, 2),
            'reasons': dict(s['reasons']),
        }

    total_pnl = sum(s['total_pnl'] for s in strat_stats.values())
    total_n = sum(s['n'] for s in strat_stats.values())
    print(f'\n  TOTAL: {total_n} trades  PnL=${total_pnl:.2f}')

    phase1['_total'] = {'n': total_n, 'pnl': round(total_pnl, 2)}
    save('phase1_overview', phase1)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Slippage Analysis
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Slippage Analysis')
    print('=' * 80)

    entry_slippages = defaultdict(list)
    for t in log:
        if t.get('action') != 'OPEN':
            continue
        sig_price = t.get('signal_price')
        actual_price = t.get('price')
        if sig_price and actual_price:
            slip = abs(actual_price - sig_price)
            entry_slippages[t.get('strategy', 'unknown')].append(slip)

    exit_slippages = defaultdict(list)
    for t in closes:
        slip_val = t.get('entry_slippage')
        if slip_val is not None:
            exit_slippages[t.get('strategy', 'unknown')].append(abs(slip_val))

    phase2 = {}
    print(f'\n  Entry slippage (signal_price vs actual_price):')
    for strat, slips in sorted(entry_slippages.items()):
        arr = np.array(slips)
        stats = {
            'n': len(slips),
            'mean': round(float(arr.mean()), 4),
            'median': round(float(np.median(arr)), 4),
            'p90': round(float(np.percentile(arr, 90)), 4),
            'max': round(float(arr.max()), 4),
        }
        phase2[f'{strat}_entry'] = stats
        print(f'    {strat:<15} n={stats["n"]:>4}  mean={stats["mean"]:.4f}  '
              f'p90={stats["p90"]:.4f}  max={stats["max"]:.4f}')

    print(f'\n  Exit slippage (recorded entry_slippage field):')
    for strat, slips in sorted(exit_slippages.items()):
        arr = np.array(slips)
        stats = {
            'n': len(slips),
            'mean': round(float(arr.mean()), 4),
            'median': round(float(np.median(arr)), 4),
            'p90': round(float(np.percentile(arr, 90)), 4),
            'max': round(float(arr.max()), 4),
        }
        phase2[f'{strat}_exit_slip'] = stats
        print(f'    {strat:<15} n={stats["n"]:>4}  mean={stats["mean"]:.4f}  '
              f'p90={stats["p90"]:.4f}  max={stats["max"]:.4f}')

    save('phase2_slippage', phase2)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Per-Strategy Deep Dive
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 3: Per-Strategy Deep Dive')
    print('=' * 80)

    phase3 = {}
    for strat in sorted(strat_stats.keys()):
        s = strat_stats[strat]
        pnls = np.array(s['pnls'])
        if len(pnls) == 0:
            continue

        print(f'\n  === {strat.upper()} ({s["n"]} trades) ===')

        # PnL distribution
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        detail = {
            'n': s['n'],
            'total_pnl': round(float(pnls.sum()), 2),
            'avg_win': round(float(wins.mean()), 2) if len(wins) > 0 else 0,
            'avg_loss': round(float(losses.mean()), 2) if len(losses) > 0 else 0,
            'max_win': round(float(wins.max()), 2) if len(wins) > 0 else 0,
            'max_loss': round(float(losses.min()), 2) if len(losses) > 0 else 0,
            'win_rate': round(100 * len(wins) / len(pnls), 2),
            'reasons': dict(s['reasons']),
        }

        # Cumulative equity curve stats
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        detail['max_dd'] = round(float(dd.max()), 2)
        detail['final_equity'] = round(float(cum[-1]), 2)

        # Weekly PnL breakdown
        weekly_pnl = defaultdict(float)
        weekly_n = defaultdict(int)
        for t in closes:
            if t.get('strategy') != strat:
                continue
            dt_str = t.get('time', '')[:10]
            try:
                dt = datetime.strptime(dt_str, '%Y-%m-%d')
                week = dt.strftime('%Y-W%W')
                weekly_pnl[week] += t.get('profit', 0)
                weekly_n[week] += 1
            except ValueError:
                pass

        detail['weekly'] = {
            w: {'pnl': round(weekly_pnl[w], 2), 'n': weekly_n[w]}
            for w in sorted(weekly_pnl.keys())
        }

        print(f'    PnL: ${detail["total_pnl"]:.2f}  WR={detail["win_rate"]:.1f}%  '
              f'AvgWin=${detail["avg_win"]:.2f}  AvgLoss=${detail["avg_loss"]:.2f}')
        print(f'    MaxWin=${detail["max_win"]:.2f}  MaxLoss=${detail["max_loss"]:.2f}  '
              f'MaxDD=${detail["max_dd"]:.2f}')
        print(f'    Exit reasons: {dict(s["reasons"])}')

        # Worst 5 trades
        worst_idx = np.argsort(pnls)[:5]
        detail['worst_trades'] = []
        strat_closes = [t for t in closes if t.get('strategy') == strat]
        for idx in worst_idx:
            if idx < len(strat_closes):
                t = strat_closes[idx]
                detail['worst_trades'].append({
                    'time': t.get('time', '')[:16],
                    'pnl': t.get('profit', 0),
                    'reason': t.get('reason', '')[:60],
                    'open_price': t.get('open_price', 0),
                    'close_price': t.get('close_price', 0),
                })

        if detail['worst_trades']:
            print(f'    Worst trades:')
            for wt in detail['worst_trades']:
                reason_safe = wt["reason"][:40].encode('ascii', 'replace').decode()
                print(f'      {wt["time"]}  ${wt["pnl"]:.2f}  {reason_safe}')

        phase3[strat] = detail

    save('phase3_strategy_deep_dive', phase3)

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Backtest vs Live PnL Comparison
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 4: Backtest vs Live Comparison')
    print('=' * 80)

    # Try to run a Keltner backtest over the same period
    phase4 = {}
    try:
        from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
        from backtest.engine import BacktestEngine
        from backtest.stats import calc_stats as bt_calc_stats
        import indicators as signals_mod
        from indicators import get_orb_strategy

        live_start = closes[0].get('time', '')[:10] if closes else '2026-03-25'
        live_end = closes[-1].get('time', '')[:10] if closes else '2026-05-13'
        print(f'\n  Live period: {live_start} -> {live_end}')

        data = DataBundle.load_default()
        get_orb_strategy().reset_daily()
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False

        engine = BacktestEngine(data.m15_df, data.h1_df, **LIVE_PARITY_KWARGS)
        bt_trades = engine.run()

        # Filter backtest trades to live period
        ts_start = pd.Timestamp(live_start, tz='UTC')
        ts_end = pd.Timestamp(live_end, tz='UTC') + pd.Timedelta(days=1)
        bt_period = [t for t in bt_trades
                     if ts_start <= pd.Timestamp(t.entry_time) <= ts_end]

        bt_pnl = sum(t.pnl for t in bt_period)
        live_keltner_pnl = strat_stats.get('keltner', {}).get('total_pnl', 0)

        phase4['keltner'] = {
            'live_n': strat_stats.get('keltner', {}).get('n', 0),
            'live_pnl': round(live_keltner_pnl, 2),
            'bt_n': len(bt_period),
            'bt_pnl': round(bt_pnl, 2),
            'delta_pnl': round(live_keltner_pnl - bt_pnl, 2),
            'delta_n': strat_stats.get('keltner', {}).get('n', 0) - len(bt_period),
        }
        print(f'\n  Keltner comparison ({live_start} -> {live_end}):')
        print(f'    Live:     n={phase4["keltner"]["live_n"]:>4}  PnL=${live_keltner_pnl:>8.2f}')
        print(f'    Backtest: n={len(bt_period):>4}  PnL=${bt_pnl:>8.2f}')
        print(f'    Delta:    n={phase4["keltner"]["delta_n"]:>4}  PnL=${phase4["keltner"]["delta_pnl"]:>8.2f}')

        if bt_pnl != 0:
            slippage_pct = (live_keltner_pnl - bt_pnl) / abs(bt_pnl) * 100
            phase4['keltner']['slippage_pct'] = round(slippage_pct, 2)
            print(f'    Execution gap: {slippage_pct:+.1f}%')

    except Exception as e:
        print(f'  Backtest comparison failed: {e}')
        phase4['error'] = str(e)

    save('phase4_bt_vs_live', phase4)

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Loss Profile & Drawdown Episodes
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 5: Loss Profile & Drawdown Episodes')
    print('=' * 80)

    all_pnls = []
    for t in closes:
        all_pnls.append({
            'time': t.get('time', ''),
            'strategy': t.get('strategy', 'unknown'),
            'pnl': t.get('profit', 0),
            'reason': t.get('reason', '')[:60],
        })

    all_pnls.sort(key=lambda x: x['time'])
    pnl_arr = np.array([t['pnl'] for t in all_pnls])

    if len(pnl_arr) > 0:
        cum = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum

        # Find drawdown episodes (peaks to troughs)
        episodes = []
        in_dd = False
        dd_start = 0
        for i in range(len(dd)):
            if dd[i] > 0 and not in_dd:
                in_dd = True
                dd_start = i
            elif dd[i] == 0 and in_dd:
                in_dd = False
                dd_max = float(dd[dd_start:i].max())
                if dd_max > 20:
                    dd_peak_idx = dd_start + np.argmax(dd[dd_start:i])
                    episodes.append({
                        'start': all_pnls[dd_start]['time'][:16],
                        'trough': all_pnls[dd_peak_idx]['time'][:16],
                        'end': all_pnls[i]['time'][:16],
                        'max_dd': round(dd_max, 2),
                        'n_trades': i - dd_start,
                        'trades': [
                            {'time': all_pnls[j]['time'][:16],
                             'strategy': all_pnls[j]['strategy'],
                             'pnl': all_pnls[j]['pnl']}
                            for j in range(dd_start, min(i, dd_start + 10))
                        ],
                    })

        episodes.sort(key=lambda x: x['max_dd'], reverse=True)

        print(f'\n  Total trades: {len(all_pnls)}')
        print(f'  Total PnL: ${float(pnl_arr.sum()):.2f}')
        print(f'  Max Drawdown: ${float(dd.max()):.2f}')
        print(f'  Drawdown episodes (>{20}):')
        for ep in episodes[:5]:
            print(f'    {ep["start"]} -> {ep["end"]}: DD=${ep["max_dd"]:.0f} '
                  f'({ep["n_trades"]} trades)')

        # Loss clustering: consecutive losses
        loss_streaks = []
        streak = 0
        streak_pnl = 0
        streak_start = ''
        for t in all_pnls:
            if t['pnl'] <= 0:
                if streak == 0:
                    streak_start = t['time'][:16]
                streak += 1
                streak_pnl += t['pnl']
            else:
                if streak >= 3:
                    loss_streaks.append({
                        'start': streak_start,
                        'length': streak,
                        'total_loss': round(streak_pnl, 2),
                    })
                streak = 0
                streak_pnl = 0

        print(f'\n  Loss streaks (3+ consecutive):')
        for ls in loss_streaks:
            print(f'    {ls["start"]}: {ls["length"]} trades, ${ls["total_loss"]:.2f}')

        # Day-of-week PnL
        dow_pnl = defaultdict(lambda: {'pnl': 0, 'n': 0})
        for t in all_pnls:
            try:
                dt = datetime.strptime(t['time'][:10], '%Y-%m-%d')
                dow = dt.strftime('%A')
                dow_pnl[dow]['pnl'] += t['pnl']
                dow_pnl[dow]['n'] += 1
            except ValueError:
                pass

        print(f'\n  Day-of-week PnL:')
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            d = dow_pnl.get(day, {'pnl': 0, 'n': 0})
            print(f'    {day:<12} n={d["n"]:>4}  PnL=${d["pnl"]:>8.2f}')

        # Hour-of-day PnL (entry hour)
        hour_pnl = defaultdict(lambda: {'pnl': 0, 'n': 0})
        for t in all_pnls:
            try:
                dt = datetime.strptime(t['time'][:19], '%Y-%m-%dT%H:%M:%S')
                h = dt.hour
                hour_pnl[h]['pnl'] += t['pnl']
                hour_pnl[h]['n'] += 1
            except ValueError:
                pass

        phase5 = {
            'total_pnl': round(float(pnl_arr.sum()), 2),
            'max_dd': round(float(dd.max()), 2),
            'episodes': episodes[:10],
            'loss_streaks': loss_streaks,
            'dow_pnl': {k: {'pnl': round(v['pnl'], 2), 'n': v['n']}
                        for k, v in dow_pnl.items()},
            'hour_pnl': {str(k): {'pnl': round(v['pnl'], 2), 'n': v['n']}
                         for k, v in sorted(hour_pnl.items())},
        }
    else:
        phase5 = {'error': 'no trades'}

    save('phase5_loss_profile', phase5)

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('FINAL SUMMARY')
    print('=' * 80)

    summary = {
        'period': f'{first_close} -> {last_close}' if closes else 'N/A',
        'total_trades': total_n,
        'total_pnl': round(total_pnl, 2),
        'per_strategy': phase1,
        'bt_comparison': phase4,
        'max_dd': phase5.get('max_dd', 0),
    }

    # Key findings
    findings = []
    for strat, s in phase1.items():
        if strat.startswith('_'):
            continue
        if s.get('total_pnl', 0) < -50:
            findings.append(f'{strat}: significant loss ${s["total_pnl"]:.0f}')
        if s.get('cap_pct', 0) > 20:
            findings.append(f'{strat}: high MaxLoss Cap rate {s["cap_pct"]:.0f}%')

    summary['findings'] = findings
    print(f'  Period: {summary["period"]}')
    print(f'  Total: {total_n} trades, PnL=${total_pnl:.2f}')
    if findings:
        print(f'  Key findings:')
        for f in findings:
            print(f'    - {f}')

    save('R211_summary', summary)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
