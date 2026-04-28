"""R46: L8 + S4 Chandelier 双策略组合 — 多维度压力测试
=====================================================
Phase A: 危机时段压力测试 (6大危机)
Phase B: 滚动窗口 Walk-Forward (22个半年窗口)
Phase C: 年度分解 + 月度分布
Phase D: 组合 Spread 敏感度 (含真实 spread 模拟)
Phase E: Monte Carlo 参数扰动 (CH 参数±20%)
Phase F: 连续亏损/回撤分析
Phase G: BUY/SELL 方向分解
Phase H: 最终结论汇总

USAGE
-----
    python -m experiments.run_round46_combo_stress
"""
import sys, os, time, json, traceback, random
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats
from backtest.engine import TradeRecord
from experiments.run_round45_new_signals import (
    backtest_signals, trades_to_stats, daily_pnl_correlation,
    SimpleTrade, Tee, chandelier_signals,
)

OUT_DIR = ROOT / "results" / "round46_combo_stress"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARATHON_START = time.time()

L8_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'keltner_adx_threshold': 14,
    'regime_config': {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    'keltner_max_hold_m15': 20,
    'time_decay_tp': False,
    'min_entry_gap_hours': 1.0,
}

CH_PARAMS = {'period': 10, 'mult': 3.0, 'ema_filter': False}
CH_BT = {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
          'trail_act': 0.28, 'trail_dist': 0.06}


def elapsed():
    return f"[{(time.time()-MARATHON_START)/60:.1f} min]"

def phase_header(name, desc):
    print(f"\n{'='*70}")
    print(f"  {name}: {desc}")
    print(f"  {elapsed()}")
    print(f"{'='*70}\n", flush=True)

def save_json(data, filename):
    path = OUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def run_combo(data, h1_df, start=None, end=None, spread_cost=0.0, ch_params=None, ch_bt=None, label=""):
    """Run L8 + Chandelier combo, return (l8_stats, ch_stats, combo_stats, l8_daily, ch_daily)."""
    if ch_params is None:
        ch_params = CH_PARAMS
    if ch_bt is None:
        ch_bt = CH_BT

    if start and end:
        slice_data = data.slice(start, end)
        slice_h1 = h1_df[start:end]
    else:
        slice_data = data
        slice_h1 = h1_df

    if len(slice_h1) < 100:
        return None, None, None, {}, {}

    l8_stats = run_variant(slice_data, f'L8_{label}', verbose=False, **L8_KWARGS)
    l8_trades = l8_stats.get('_trades', [])
    l8_daily = {}
    for t in l8_trades:
        d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
        l8_daily[d] = l8_daily.get(d, 0) + t.pnl

    sig, atr = chandelier_signals(slice_h1, **ch_params)
    ch_trades = backtest_signals(slice_h1, sig, atr, spread_cost=spread_cost, **ch_bt)
    ch_stats = trades_to_stats(ch_trades, f'CH_{label}')
    ch_daily = ch_stats['daily_pnl']

    combo_daily = {}
    for d in set(list(l8_daily.keys()) + list(ch_daily.keys())):
        combo_daily[d] = l8_daily.get(d, 0) + ch_daily.get(d, 0)

    dates = sorted(combo_daily.keys())
    pnls = [combo_daily[d] for d in dates]
    total = sum(pnls)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
    else:
        sharpe = 0
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    combo_stats = {
        'label': f'Combo_{label}',
        'sharpe': round(sharpe, 2),
        'total_pnl': round(total, 2),
        'max_dd': round(max_dd, 2),
        'n_days': len(dates),
    }
    return l8_stats, ch_stats, combo_stats, l8_daily, ch_daily


# ══════════════════════════════════════════════════════════════
# Phase A: 危机时段压力测试
# ══════════════════════════════════════════════════════════════

def phase_a(data, h1_df):
    phase_header("Phase A", "危机时段压力测试")

    crises = [
        ("Brexit_2016",       "2016-06-20", "2016-07-20"),
        ("Trade_War_2018",    "2018-09-01", "2018-12-31"),
        ("COVID_Crash_2020",  "2020-02-20", "2020-04-15"),
        ("Inflation_2022",    "2022-03-01", "2022-06-30"),
        ("SVB_2023",          "2023-03-01", "2023-04-15"),
        ("Liberation_Day_25", "2025-03-15", "2025-05-15"),
        ("Tariff_2026",       "2026-01-01", "2026-04-25"),
    ]

    results = []
    for name, start, end in crises:
        l8_s, ch_s, combo_s, l8_d, ch_d = run_combo(data, h1_df, start, end, label=name)
        if combo_s is None:
            print(f"  {name}: insufficient data, skipped")
            continue

        corr = daily_pnl_correlation(l8_d, ch_d)
        row = {
            'crisis': name, 'period': f"{start}~{end}",
            'l8_sharpe': round(l8_s['sharpe'], 2),
            'l8_pnl': round(l8_s['total_pnl'], 0),
            'ch_sharpe': ch_s['sharpe'],
            'ch_pnl': round(ch_s['total_pnl'], 0),
            'combo_sharpe': combo_s['sharpe'],
            'combo_pnl': combo_s['total_pnl'],
            'combo_maxdd': combo_s['max_dd'],
            'correlation': corr,
        }
        results.append(row)
        print(f"  {name}: L8={l8_s['sharpe']:.2f}/${l8_s['total_pnl']:.0f}, "
              f"CH={ch_s['sharpe']}/${ch_s['total_pnl']:.0f}, "
              f"Combo={combo_s['sharpe']}/${combo_s['total_pnl']}, "
              f"DD=${combo_s['max_dd']}, corr={corr}")

    profitable = sum(1 for r in results if r['combo_pnl'] > 0)
    print(f"\n  ★ {profitable}/{len(results)} 危机时段组合盈利")

    save_json(results, 'A_crisis_stress.json')
    print(f"  Phase A complete. {elapsed()}")
    return results


# ══════════════════════════════════════════════════════════════
# Phase B: 滚动窗口 Walk-Forward
# ══════════════════════════════════════════════════════════════

def phase_b(data, h1_df):
    phase_header("Phase B", "滚动窗口 Walk-Forward (半年窗口)")

    windows = []
    for year in range(2015, 2027):
        windows.append((f"{year}H1", f"{year}-01-01", f"{year}-07-01"))
        if year < 2026:
            windows.append((f"{year}H2", f"{year}-07-01", f"{year+1}-01-01"))
        else:
            windows.append((f"{year}H2", f"{year}-04-01", f"{year}-04-25"))

    results = []
    for wname, start, end in windows:
        l8_s, ch_s, combo_s, l8_d, ch_d = run_combo(data, h1_df, start, end, label=wname)
        if combo_s is None:
            continue
        if l8_s['n'] < 10:
            continue

        row = {
            'window': wname, 'period': f"{start}~{end}",
            'l8_sharpe': round(l8_s['sharpe'], 2),
            'l8_pnl': round(l8_s['total_pnl'], 0),
            'ch_sharpe': ch_s['sharpe'],
            'ch_pnl': round(ch_s['total_pnl'], 0),
            'combo_sharpe': combo_s['sharpe'],
            'combo_pnl': combo_s['total_pnl'],
            'combo_maxdd': combo_s['max_dd'],
        }
        results.append(row)

    l8_sharpes = [r['l8_sharpe'] for r in results]
    combo_sharpes = [r['combo_sharpe'] for r in results]
    l8_positive = sum(1 for s in l8_sharpes if s > 0)
    combo_positive = sum(1 for s in combo_sharpes if s > 0)

    print(f"  {len(results)} 半年窗口:")
    print(f"    L8 alone:  {l8_positive}/{len(results)} positive, "
          f"mean={np.mean(l8_sharpes):.2f}, min={min(l8_sharpes):.2f}")
    print(f"    L8+CH:     {combo_positive}/{len(results)} positive, "
          f"mean={np.mean(combo_sharpes):.2f}, min={min(combo_sharpes):.2f}")

    combo_better = sum(1 for i in range(len(results)) if combo_sharpes[i] >= l8_sharpes[i])
    print(f"    Combo >= L8 在 {combo_better}/{len(results)} 窗口")

    worst_5 = sorted(results, key=lambda r: r['combo_sharpe'])[:5]
    print(f"\n  最差5个窗口:")
    for r in worst_5:
        print(f"    {r['window']}: L8={r['l8_sharpe']}, Combo={r['combo_sharpe']}, "
              f"PnL=${r['combo_pnl']}, MaxDD=${r['combo_maxdd']}")

    save_json({'windows': results,
               'summary': {
                   'n_windows': len(results),
                   'l8_positive': l8_positive,
                   'combo_positive': combo_positive,
                   'l8_mean': round(np.mean(l8_sharpes), 2),
                   'combo_mean': round(np.mean(combo_sharpes), 2),
                   'combo_better_count': combo_better,
               }},
              'B_walk_forward.json')
    print(f"  Phase B complete. {elapsed()}")
    return results


# ══════════════════════════════════════════════════════════════
# Phase C: 年度/月度分解
# ══════════════════════════════════════════════════════════════

def phase_c(data, h1_df):
    phase_header("Phase C", "年度分解 + 月度分布")

    l8_full, ch_full, combo_full, l8_daily, ch_daily = run_combo(data, h1_df, label="full")

    # 年度
    combo_daily_all = {}
    for d in set(list(l8_daily.keys()) + list(ch_daily.keys())):
        combo_daily_all[d] = l8_daily.get(d, 0) + ch_daily.get(d, 0)

    year_pnl_l8 = defaultdict(float)
    year_pnl_ch = defaultdict(float)
    year_pnl_combo = defaultdict(float)
    month_pnl_combo = defaultdict(float)

    for d, pnl in l8_daily.items():
        year_pnl_l8[d[:4]] += pnl
    for d, pnl in ch_daily.items():
        year_pnl_ch[d[:4]] += pnl
    for d, pnl in combo_daily_all.items():
        year_pnl_combo[d[:4]] += pnl
        month_pnl_combo[d[:7]] += pnl

    years = sorted(year_pnl_combo.keys())
    print("  年度分解:")
    print(f"  {'Year':>6} | {'L8 PnL':>10} | {'CH PnL':>10} | {'Combo PnL':>10} | {'CH贡献%':>8}")
    for y in years:
        l8_p = year_pnl_l8.get(y, 0)
        ch_p = year_pnl_ch.get(y, 0)
        co_p = year_pnl_combo.get(y, 0)
        ch_pct = ch_p / co_p * 100 if co_p != 0 else 0
        print(f"  {y:>6} | ${l8_p:>9.0f} | ${ch_p:>9.0f} | ${co_p:>9.0f} | {ch_pct:>7.1f}%")

    # 月度
    months = sorted(month_pnl_combo.keys())
    monthly_pnls = [month_pnl_combo[m] for m in months]
    neg_months = sum(1 for p in monthly_pnls if p < 0)

    print(f"\n  月度分布:")
    print(f"    总月数: {len(months)}")
    print(f"    负月数: {neg_months} ({neg_months/len(months)*100:.1f}%)")
    print(f"    月均PnL: ${np.mean(monthly_pnls):.0f}")
    print(f"    最差月: ${min(monthly_pnls):.0f}")
    print(f"    最好月: ${max(monthly_pnls):.0f}")

    # 日内小时分布 (CH trades only)
    sig, atr = chandelier_signals(h1_df, **CH_PARAMS)
    ch_trades_full = backtest_signals(h1_df, sig, atr, **CH_BT)
    hour_pnl = defaultdict(list)
    for t in ch_trades_full:
        h = t.entry_time.hour if hasattr(t.entry_time, 'hour') else 0
        hour_pnl[h].append(t.pnl)

    print(f"\n  Chandelier 入场小时分布 (Top 5):")
    hour_stats = [(h, len(ps), sum(ps), np.mean(ps)) for h, ps in hour_pnl.items()]
    hour_stats.sort(key=lambda x: x[2], reverse=True)
    for h, n, total, avg in hour_stats[:5]:
        print(f"    UTC {h:02d}:00: N={n}, PnL=${total:.0f}, Avg=${avg:.2f}")

    year_data = []
    for y in years:
        year_data.append({
            'year': y,
            'l8_pnl': round(year_pnl_l8.get(y, 0), 0),
            'ch_pnl': round(year_pnl_ch.get(y, 0), 0),
            'combo_pnl': round(year_pnl_combo.get(y, 0), 0),
        })

    save_json({
        'annual': year_data,
        'monthly': {'n': len(months), 'neg': neg_months, 'mean': round(np.mean(monthly_pnls), 0),
                    'worst': round(min(monthly_pnls), 0), 'best': round(max(monthly_pnls), 0)},
        'ch_hour_dist': [{
            'hour': h, 'n': n, 'total_pnl': round(total, 0), 'avg_pnl': round(avg, 2)
        } for h, n, total, avg in hour_stats],
    }, 'C_annual_monthly.json')
    print(f"  Phase C complete. {elapsed()}")
    return year_data


# ══════════════════════════════════════════════════════════════
# Phase D: 组合 Spread 敏感度
# ══════════════════════════════════════════════════════════════

def phase_d(data, h1_df):
    phase_header("Phase D", "组合 Spread 敏感度 (含 L8 + CH 各自 spread)")

    spreads = [0.0, 0.30, 0.50, 0.80, 1.00, 1.50]
    results = []

    for sp in spreads:
        _, ch_s, combo_s, l8_d, ch_d = run_combo(data, h1_df, spread_cost=sp, label=f"sp{sp}")
        results.append({
            'spread': sp,
            'combo_sharpe': combo_s['sharpe'],
            'combo_pnl': combo_s['total_pnl'],
            'combo_maxdd': combo_s['max_dd'],
            'ch_sharpe': ch_s['sharpe'],
            'ch_pnl': round(ch_s['total_pnl'], 0),
        })
        print(f"    spread=${sp:.2f}: Combo Sharpe={combo_s['sharpe']}, PnL=${combo_s['total_pnl']}, "
              f"MaxDD=${combo_s['max_dd']}, CH_Sharpe={ch_s['sharpe']}")

    # L8 本身不受这里的 spread 影响 (它有自己的 spread model)
    s0 = results[0]['combo_sharpe']
    s50 = next((r['combo_sharpe'] for r in results if r['spread'] == 0.50), 0)
    degradation = (s0 - s50) / s0 * 100 if s0 > 0 else 0
    print(f"\n  组合 Sharpe 衰减 @$0.50: {degradation:.1f}%")

    save_json(results, 'D_spread_sensitivity.json')
    print(f"  Phase D complete. {elapsed()}")
    return results


# ══════════════════════════════════════════════════════════════
# Phase E: Monte Carlo 参数扰动
# ══════════════════════════════════════════════════════════════

def phase_e(data, h1_df):
    phase_header("Phase E", "Monte Carlo 参数扰动 (CH参数±20%, 100次)")

    random.seed(42)
    np.random.seed(42)

    base_period = CH_PARAMS['period']
    base_mult = CH_PARAMS['mult']
    base_sl = CH_BT['sl_mult']
    base_tp = CH_BT['tp_mult']
    base_mh = CH_BT['max_hold']
    base_ta = CH_BT['trail_act']
    base_td = CH_BT['trail_dist']

    n_trials = 100
    results = []

    for i in range(n_trials):
        p = max(5, int(base_period * np.random.uniform(0.8, 1.2)))
        m = max(1.0, base_mult * np.random.uniform(0.8, 1.2))
        sl = max(1.0, base_sl * np.random.uniform(0.8, 1.2))
        tp = max(2.0, base_tp * np.random.uniform(0.8, 1.2))
        mh = max(5, int(base_mh * np.random.uniform(0.8, 1.2)))
        ta = max(0.05, base_ta * np.random.uniform(0.8, 1.2))
        td = max(0.01, base_td * np.random.uniform(0.8, 1.2))

        ch_p = {'period': p, 'mult': round(m, 2), 'ema_filter': False}
        ch_b = {'sl_mult': round(sl, 2), 'tp_mult': round(tp, 2),
                'max_hold': mh, 'trail_act': round(ta, 3), 'trail_dist': round(td, 4)}

        _, ch_s, combo_s, _, _ = run_combo(data, h1_df, ch_params=ch_p, ch_bt=ch_b,
                                            label=f"mc{i}")
        if combo_s:
            results.append({
                'trial': i,
                'params': {**ch_p, **ch_b},
                'combo_sharpe': combo_s['sharpe'],
                'combo_pnl': combo_s['total_pnl'],
                'combo_maxdd': combo_s['max_dd'],
                'ch_sharpe': ch_s['sharpe'],
            })

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_trials} trials done... {elapsed()}")

    combo_sharpes = [r['combo_sharpe'] for r in results]
    ch_sharpes = [r['ch_sharpe'] for r in results]

    print(f"\n  Monte Carlo 结果 ({len(results)} trials):")
    print(f"    Combo Sharpe: mean={np.mean(combo_sharpes):.2f}, "
          f"std={np.std(combo_sharpes):.2f}, "
          f"min={min(combo_sharpes):.2f}, max={max(combo_sharpes):.2f}")
    print(f"    CH Sharpe:    mean={np.mean(ch_sharpes):.2f}, "
          f"std={np.std(ch_sharpes):.2f}")
    pct_above_10 = sum(1 for s in combo_sharpes if s >= 10) / len(combo_sharpes) * 100
    pct_above_8 = sum(1 for s in combo_sharpes if s >= 8) / len(combo_sharpes) * 100
    print(f"    Combo Sharpe >= 10: {pct_above_10:.0f}%")
    print(f"    Combo Sharpe >= 8:  {pct_above_8:.0f}%")

    save_json({
        'n_trials': len(results),
        'combo_sharpe_mean': round(np.mean(combo_sharpes), 2),
        'combo_sharpe_std': round(np.std(combo_sharpes), 2),
        'combo_sharpe_min': round(min(combo_sharpes), 2),
        'combo_sharpe_max': round(max(combo_sharpes), 2),
        'pct_above_10': round(pct_above_10, 1),
        'pct_above_8': round(pct_above_8, 1),
        'worst_5': sorted(results, key=lambda r: r['combo_sharpe'])[:5],
        'best_5': sorted(results, key=lambda r: r['combo_sharpe'], reverse=True)[:5],
    }, 'E_monte_carlo.json')
    print(f"  Phase E complete. {elapsed()}")
    return results


# ══════════════════════════════════════════════════════════════
# Phase F: 连续亏损/回撤分析
# ══════════════════════════════════════════════════════════════

def phase_f(data, h1_df):
    phase_header("Phase F", "连续亏损 / 回撤分析")

    _, _, _, l8_daily, ch_daily = run_combo(data, h1_df, label="dd_analysis")

    combo_daily = {}
    for d in set(list(l8_daily.keys()) + list(ch_daily.keys())):
        combo_daily[d] = l8_daily.get(d, 0) + ch_daily.get(d, 0)

    dates = sorted(combo_daily.keys())
    pnls = [combo_daily[d] for d in dates]

    # L8 daily
    l8_dates = sorted(l8_daily.keys())
    l8_pnls = [l8_daily[d] for d in l8_dates]

    def analyze_streaks(daily_pnls, label):
        streak = 0
        max_loss_streak = 0
        loss_streaks = []
        for p in daily_pnls:
            if p < 0:
                streak += 1
                max_loss_streak = max(max_loss_streak, streak)
            else:
                if streak > 0:
                    loss_streaks.append(streak)
                streak = 0
        if streak > 0:
            loss_streaks.append(streak)

        cumsum = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = running_max - cumsum

        dd_peak_idx = np.argmax(drawdowns)
        dd_start_idx = np.argmax(cumsum[:dd_peak_idx+1]) if dd_peak_idx > 0 else 0

        result = {
            'label': label,
            'max_loss_streak_days': max_loss_streak,
            'avg_loss_streak_days': round(np.mean(loss_streaks), 1) if loss_streaks else 0,
            'max_drawdown': round(float(np.max(drawdowns)), 2),
            'neg_days': sum(1 for p in daily_pnls if p < 0),
            'total_days': len(daily_pnls),
            'neg_day_pct': round(sum(1 for p in daily_pnls if p < 0) / len(daily_pnls) * 100, 1),
        }
        return result

    l8_dd = analyze_streaks(l8_pnls, "L8_alone")
    combo_dd = analyze_streaks(pnls, "L8+CH_combo")

    print(f"  {'Metric':<30} | {'L8 Alone':>12} | {'L8+CH Combo':>12}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*12}")
    for key in ['max_loss_streak_days', 'avg_loss_streak_days', 'max_drawdown', 'neg_days', 'neg_day_pct']:
        l8_val = l8_dd[key]
        co_val = combo_dd[key]
        unit = '%' if 'pct' in key else ('$' if 'drawdown' in key else '')
        if unit == '$':
            print(f"  {key:<30} | {unit}{l8_val:>11} | {unit}{co_val:>11}")
        else:
            print(f"  {key:<30} | {l8_val:>12} | {co_val:>12}")

    # Top 10 worst combo days
    worst_days = sorted(zip(dates, pnls), key=lambda x: x[1])[:10]
    print(f"\n  组合最差10天:")
    for d, p in worst_days:
        l8_p = l8_daily.get(d, 0)
        ch_p = ch_daily.get(d, 0)
        print(f"    {d}: Combo=${p:.2f} (L8=${l8_p:.2f}, CH=${ch_p:.2f})")

    save_json({
        'l8_alone': l8_dd,
        'combo': combo_dd,
        'worst_10_days': [{'date': d, 'combo_pnl': round(p, 2),
                           'l8_pnl': round(l8_daily.get(d, 0), 2),
                           'ch_pnl': round(ch_daily.get(d, 0), 2)} for d, p in worst_days],
    }, 'F_drawdown_analysis.json')
    print(f"  Phase F complete. {elapsed()}")
    return combo_dd


# ══════════════════════════════════════════════════════════════
# Phase G: BUY/SELL 方向分解
# ══════════════════════════════════════════════════════════════

def phase_g(h1_df):
    phase_header("Phase G", "Chandelier BUY/SELL 方向分解")

    sig, atr = chandelier_signals(h1_df, **CH_PARAMS)
    all_trades = backtest_signals(h1_df, sig, atr, **CH_BT)

    buy_trades = [t for t in all_trades if t.direction == 'BUY']
    sell_trades = [t for t in all_trades if t.direction == 'SELL']

    buy_stats = trades_to_stats(buy_trades, "CH_BUY")
    sell_stats = trades_to_stats(sell_trades, "CH_SELL")
    all_stats = trades_to_stats(all_trades, "CH_ALL")

    # Exit reason breakdown
    exit_reasons = defaultdict(lambda: {'buy': 0, 'sell': 0, 'buy_pnl': 0, 'sell_pnl': 0})
    for t in all_trades:
        key = t.exit_reason
        if t.direction == 'BUY':
            exit_reasons[key]['buy'] += 1
            exit_reasons[key]['buy_pnl'] += t.pnl
        else:
            exit_reasons[key]['sell'] += 1
            exit_reasons[key]['sell_pnl'] += t.pnl

    print(f"  {'Metric':<20} | {'BUY':>10} | {'SELL':>10} | {'ALL':>10}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'N trades':<20} | {buy_stats['n']:>10} | {sell_stats['n']:>10} | {all_stats['n']:>10}")
    print(f"  {'Total PnL':<20} | ${buy_stats['total_pnl']:>9.0f} | ${sell_stats['total_pnl']:>9.0f} | ${all_stats['total_pnl']:>9.0f}")
    print(f"  {'Sharpe':<20} | {buy_stats['sharpe']:>10} | {sell_stats['sharpe']:>10} | {all_stats['sharpe']:>10}")
    print(f"  {'Win Rate':<20} | {buy_stats['win_rate']:>9}% | {sell_stats['win_rate']:>9}% | {all_stats['win_rate']:>9}%")
    print(f"  {'Avg PnL':<20} | ${buy_stats['avg_pnl']:>9} | ${sell_stats['avg_pnl']:>9} | ${all_stats['avg_pnl']:>9}")
    print(f"  {'Avg Bars':<20} | {buy_stats['avg_bars']:>10} | {sell_stats['avg_bars']:>10} | {all_stats['avg_bars']:>10}")

    print(f"\n  出场原因分解:")
    print(f"  {'Reason':<10} | {'BUY_N':>6} | {'BUY_PnL':>10} | {'SELL_N':>6} | {'SELL_PnL':>10}")
    for reason in sorted(exit_reasons.keys()):
        r = exit_reasons[reason]
        print(f"  {reason:<10} | {r['buy']:>6} | ${r['buy_pnl']:>9.0f} | {r['sell']:>6} | ${r['sell_pnl']:>9.0f}")

    save_json({
        'buy': {k: v for k, v in buy_stats.items() if k != 'daily_pnl'},
        'sell': {k: v for k, v in sell_stats.items() if k != 'daily_pnl'},
        'all': {k: v for k, v in all_stats.items() if k != 'daily_pnl'},
        'exit_reasons': {k: v for k, v in exit_reasons.items()},
    }, 'G_direction_analysis.json')
    print(f"  Phase G complete. {elapsed()}")
    return {'buy': buy_stats, 'sell': sell_stats}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_file = open(OUT_DIR / "00_master_log.txt", 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"R46 L8+CH Combo Stress Test — Started at {datetime.now()}")
    print(f"Output: {OUT_DIR}\n")

    print("Loading data...")
    t0 = time.time()
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"Data loaded in {time.time()-t0:.1f}s")
    print(f"  H1: {len(h1_df)} bars, {h1_df.index[0]} ~ {h1_df.index[-1]}")

    phases = [
        ("A", phase_a, (data, h1_df)),
        ("B", phase_b, (data, h1_df)),
        ("C", phase_c, (data, h1_df)),
        ("D", phase_d, (data, h1_df)),
        ("E", phase_e, (data, h1_df)),
        ("F", phase_f, (data, h1_df)),
        ("G", phase_g, (h1_df,)),
    ]

    completed = []
    for pname, pfunc, pargs in phases:
        try:
            t_phase = time.time()
            result = pfunc(*pargs)
            dt = time.time() - t_phase
            completed.append((pname, dt, result))
            print(f"\n  Phase {pname} took {dt/60:.1f} min")
        except Exception as e:
            print(f"\n  Phase {pname} FAILED: {e}")
            traceback.print_exc()
            completed.append((pname, -1, None))

    total_elapsed = time.time() - MARATHON_START
    print(f"\n\n{'='*70}")
    print(f"  R46 COMPLETE — {total_elapsed/60:.0f} minutes")
    print(f"{'='*70}")
    for pname, dt, _ in completed:
        status = f"{dt/60:.1f} min" if dt > 0 else "FAILED"
        print(f"  Phase {pname}: {status}")
    print(f"\n  Results: {OUT_DIR}")
    log_file.close()
