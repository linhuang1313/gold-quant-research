#!/usr/bin/env python3
"""
Round 47 — "全策略组合终极验证 + 新出场研究" (24h Server Marathon)
================================================================
预计总耗时: ~20-24 小时 (服务器 CPU)

=== Phase A: L8+CH 四策略全组合验证 (~3h) ===
A1: L8_BASE 独立全样本基线
A2: S4 Chandelier 独立全样本基线
A3: S3 Dual Thrust 独立全样本基线
A4: L8+CH 双策略组合 (已确认, 基线)
A5: L8+DT 双策略组合
A6: L8+CH+DT 三策略组合
A7: 所有组合 K-Fold 6/6 验证

=== Phase B: Chandelier 出场参数全扫描 (~4h) ===
B1: SL 扫描 (1.5/2.0/2.5/3.0/3.5/4.0/5.0) × TP 扫描 (4/6/8/10/12)
B2: MaxHold 扫描 (8/12/16/20/30/50) on best SL/TP
B3: Trail 参数扫描 (act 0.10~0.50 × dist 0.02~0.15)
B4: 最优出场 K-Fold 6/6 验证
B5: 最优出场 + L8 组合 vs 原始 combo

=== Phase C: Chandelier 入场增强 (~3h) ===
C1: ADX 趋势强度过滤 (ADX>14/18/22/25/30)
C2: ATR 波动率 regime 过滤 (percentile 25-75 / 20-80 / 不过滤)
C3: H1 KC 同向过滤 (与 L8 的 H1 filter 相同)
C4: 入场间隔 (1h/2h/4h) 防止同区间重复开仓
C5: 最优过滤叠加 K-Fold 6/6

=== Phase D: 多资产 Chandelier (~3h) ===
D1: XAGUSD (白银) Chandelier 独立回测 + 参数扫描
D2: EURUSD Chandelier 独立回测 + 参数扫描
D3: 跨资产 K-Fold 验证
D4: 跨资产日收益相关性矩阵

=== Phase E: L8 引擎参数微调确认 (~4h) ===
E1: L8_BASE ADX 14 vs 18 对比 (R43 结论确认)
E2: L8 MaxHold 扫描 (8/12/16/20) 在 L8_BASE 配置
E3: L8 Trail regime 微调 (当前3档 vs 5档 vs 无regime)
E4: L8 + EqCurve LB=10 全面确认
E5: L8 + H1 KC 同向过滤 (EMA20/M2.0 vs EMA25/M1.2)
E6: 最优 L8 K-Fold 6/6

=== Phase F: 终极组合优化 (~4h) ===
F1: L8(best) + CH(best) 组合 lot 比例扫描 (0.5x/0.75x/1.0x/1.5x)
F2: 全组合 Spread 压力 ($0.30/$0.50/$0.75/$1.00/$1.50)
F3: 全组合 Walk-Forward 24半年窗口
F4: 全组合危机时段 (7大危机)
F5: 全组合 Monte Carlo 50x 参数扰动 (±20%)
F6: 全组合年度逐年分解
F7: 全组合 BUY/SELL 方向分析

=== Phase G: 最终报告 ===
G1: 汇总所有结果, 输出最终推荐配置
"""
import sys, os, time, json, traceback, random, gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
from backtest.stats import calc_stats
from backtest.engine import TradeRecord
from indicators import (
    calc_chandelier, calc_dual_thrust_range, prepare_indicators
)

OUT_DIR = ROOT / "results" / "round47_marathon"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARATHON_START = time.time()

# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            try: f.flush()
            except: pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except: pass

def save_json(data, filename):
    path = OUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")

def elapsed():
    return f"[{(time.time()-MARATHON_START)/60:.1f} min]"

def phase_header(name, desc):
    print(f"\n{'='*70}")
    print(f"  {name}: {desc}")
    print(f"  {elapsed()}")
    print(f"{'='*70}\n", flush=True)


# ═══════════════════════════════════════════════════════════════
# Strategy Configs
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Generic signal backtester (from R45)
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimpleTrade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    atr: pd.Series,
    sl_mult: float = 3.0,
    tp_mult: float = 8.0,
    max_hold: int = 20,
    trail_act: float = 0.28,
    trail_dist: float = 0.06,
    spread_cost: float = 0.0,
    min_gap_bars: int = 0,
    label: str = "",
) -> List[SimpleTrade]:
    trades = []
    pos = None
    last_entry_bar = -9999

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    times = df.index
    sig_vals = signals.values
    atr_vals = atr.values

    for i in range(1, len(df)):
        if pos is not None:
            direction, entry_price, entry_bar, sl, tp, trail_price, entry_atr = pos
            bars_held = i - entry_bar
            h, l, c = highs[i], lows[i], closes[i]

            if direction == 'BUY':
                float_profit = (h - entry_price) / entry_atr if entry_atr > 0 else 0
                if float_profit >= trail_act and trail_price is None:
                    trail_price = h - trail_dist * entry_atr
                if trail_price is not None:
                    trail_price = max(trail_price, h - trail_dist * entry_atr)
            else:
                float_profit = (entry_price - l) / entry_atr if entry_atr > 0 else 0
                if float_profit >= trail_act and trail_price is None:
                    trail_price = l + trail_dist * entry_atr
                if trail_price is not None:
                    trail_price = min(trail_price, l + trail_dist * entry_atr)

            exit_price = None
            exit_reason = None

            if direction == 'BUY':
                if l <= sl:
                    exit_price, exit_reason = sl, 'SL'
                elif h >= tp:
                    exit_price, exit_reason = tp, 'TP'
                elif trail_price is not None and l <= trail_price:
                    exit_price, exit_reason = trail_price, 'TRAIL'
                elif bars_held >= max_hold:
                    exit_price, exit_reason = c, 'TIMEOUT'
            else:
                if h >= sl:
                    exit_price, exit_reason = sl, 'SL'
                elif l <= tp:
                    exit_price, exit_reason = tp, 'TP'
                elif trail_price is not None and h >= trail_price:
                    exit_price, exit_reason = trail_price, 'TRAIL'
                elif bars_held >= max_hold:
                    exit_price, exit_reason = c, 'TIMEOUT'

            if exit_price is not None:
                if direction == 'BUY':
                    pnl = exit_price - entry_price - spread_cost
                else:
                    pnl = entry_price - exit_price - spread_cost
                trades.append(SimpleTrade(
                    entry_time=times[entry_bar],
                    exit_time=times[i],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    bars_held=bars_held,
                    exit_reason=exit_reason,
                ))
                pos = None
            else:
                pos = (direction, entry_price, entry_bar, sl, tp, trail_price, entry_atr)

        if pos is None and i < len(df) - 1:
            sig = sig_vals[i]
            if sig == 0 or np.isnan(sig):
                continue
            if i - last_entry_bar < min_gap_bars:
                continue
            entry_price = opens[i + 1]
            entry_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0

            if sig > 0:
                sl_price = entry_price - sl_mult * entry_atr
                tp_price = entry_price + tp_mult * entry_atr
                pos = ('BUY', entry_price, i + 1, sl_price, tp_price, None, entry_atr)
                last_entry_bar = i
            elif sig < 0:
                sl_price = entry_price + sl_mult * entry_atr
                tp_price = entry_price - tp_mult * entry_atr
                pos = ('SELL', entry_price, i + 1, sl_price, tp_price, None, entry_atr)
                last_entry_bar = i

    return trades


def trades_to_stats(trades: List[SimpleTrade], label: str = "") -> Dict:
    if not trades:
        return {'label': label, 'n': 0, 'total_pnl': 0, 'sharpe': 0,
                'win_rate': 0, 'max_dd': 0, 'avg_pnl': 0, 'daily_pnl': {}}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    daily_pnl = {}
    for t in trades:
        d = t.exit_time.date() if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
        d = str(d)
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl
    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    return {
        'label': label,
        'n': len(trades),
        'total_pnl': round(sum(pnls), 2),
        'sharpe': round(sharpe, 2),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'max_dd': round(max_dd, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'avg_bars': round(np.mean([t.bars_held for t in trades]), 1),
        'exit_reasons': exit_reasons,
        'daily_pnl': daily_pnl,
    }


def daily_pnl_correlation(daily_a: Dict, daily_b: Dict) -> float:
    all_dates = sorted(set(daily_a.keys()) | set(daily_b.keys()))
    a = [daily_a.get(d, 0) for d in all_dates]
    b = [daily_b.get(d, 0) for d in all_dates]
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def combine_daily_pnl(*daily_dicts) -> Dict[str, float]:
    combined = {}
    for d in daily_dicts:
        for date, pnl in d.items():
            combined[date] = combined.get(date, 0) + pnl
    return combined


def stats_from_daily(daily_pnl: Dict[str, float], label: str = "") -> Dict:
    if not daily_pnl:
        return {'label': label, 'sharpe': 0, 'total_pnl': 0, 'max_dd': 0, 'n_days': 0}
    dates = sorted(daily_pnl.keys())
    pnls = [daily_pnl[d] for d in dates]
    total = sum(pnls)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
    else:
        sharpe = 0
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
    neg_months = 0
    monthly = defaultdict(float)
    for d, p in daily_pnl.items():
        monthly[d[:7]] += p
    neg_months = sum(1 for v in monthly.values() if v < 0)
    return {
        'label': label, 'sharpe': round(sharpe, 2), 'total_pnl': round(total, 2),
        'max_dd': round(max_dd, 2), 'n_days': len(dates), 'neg_months': neg_months,
    }


# ═══════════════════════════════════════════════════════════════
# Signal generators
# ═══════════════════════════════════════════════════════════════

def chandelier_signals(df, period=10, mult=3.0, ema_filter=False,
                       adx_min=0, atr_pct_low=0, atr_pct_high=100,
                       h1_kc_filter=False, **kw):
    ch = calc_chandelier(df, period, mult)
    atr = (df['High'] - df['Low']).rolling(14).mean()
    ema100 = df['Close'].ewm(span=100).mean()
    close = df['Close']

    above_long = close > ch['Chand_long']
    below_short = close < ch['Chand_short']

    flip_bull = above_long & (~above_long.shift(1).fillna(False))
    flip_bear = below_short & (~below_short.shift(1).fillna(False))

    sig = pd.Series(0, index=df.index)
    if ema_filter:
        sig[flip_bull & (close > ema100)] = 1
        sig[flip_bear & (close < ema100)] = -1
    else:
        sig[flip_bull] = 1
        sig[flip_bear] = -1

    if adx_min > 0 and 'ADX' in df.columns:
        adx = df['ADX']
        sig[adx < adx_min] = 0

    if atr_pct_low > 0 or atr_pct_high < 100:
        if 'atr_percentile' in df.columns:
            pct = df['atr_percentile'] * 100
            sig[(pct < atr_pct_low) | (pct > atr_pct_high)] = 0

    if h1_kc_filter and 'KC_upper' in df.columns and 'KC_lower' in df.columns:
        kc_upper = df['KC_upper']
        kc_lower = df['KC_lower']
        bull_filter = close > kc_upper
        bear_filter = close < kc_lower
        buy_mask = sig == 1
        sell_mask = sig == -1
        sig[buy_mask & ~bull_filter] = 0
        sig[sell_mask & ~bear_filter] = 0

    return sig, atr


def dual_thrust_signals(df, k_buy=0.6, k_sell=0.6, lookback=1,
                        sl_mult=3.0, tp_mult=8.0, max_hold=20,
                        trail_act=0.28, trail_dist=0.06, **kw):
    dr = calc_dual_thrust_range(df, lookback)
    atr = (df['High'] - df['Low']).rolling(14).mean()

    daily_open = df.groupby(df.index.date)['Open'].transform('first')
    buy_line = daily_open + k_buy * dr
    sell_line = daily_open - k_sell * dr

    sig = pd.Series(0, index=df.index)
    sig[df['Close'] > buy_line] = 1
    sig[df['Close'] < sell_line] = -1

    return sig, atr


def kfold_test_signal(h1_df, signal_func, sig_params, bt_params, label):
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-05-01"),
    ]
    results = []
    for fname, start, end in folds:
        fold_df = h1_df[start:end]
        if len(fold_df) < 200:
            continue
        signals, atr = signal_func(fold_df, **sig_params)
        trades = backtest_signals(fold_df, signals, atr, **bt_params)
        stats = trades_to_stats(trades, f"{label}_{fname}")
        stats['fold'] = fname
        results.append(stats)
    sharpes = [r['sharpe'] for r in results]
    pass_count = sum(1 for s in sharpes if s > 0)
    return {
        'label': label, 'folds': [{k: v for k, v in r.items() if k != 'daily_pnl'} for r in results],
        'sharpes': sharpes, 'mean_sharpe': round(np.mean(sharpes), 2) if sharpes else 0,
        'min_sharpe': round(min(sharpes), 2) if sharpes else 0,
        'pass': f"{pass_count}/{len(results)}",
    }


def kfold_combo(data, h1_df, l8_kw, ch_sig, ch_bt, label):
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-05-01"),
    ]
    results = []
    for fname, start, end in folds:
        slice_data = data.slice(start, end)
        fold_h1 = h1_df[start:end]
        if len(fold_h1) < 200 or len(slice_data.m15_df) < 500:
            continue
        l8_stats = run_variant(slice_data, f'L8_{fname}', verbose=False, **l8_kw)
        l8_trades = l8_stats.get('_trades', [])
        l8_daily = {}
        for t in l8_trades:
            d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
            l8_daily[d] = l8_daily.get(d, 0) + t.pnl

        sig, atr = chandelier_signals(fold_h1, **ch_sig)
        ch_trades = backtest_signals(fold_h1, sig, atr, **ch_bt)
        ch_stats = trades_to_stats(ch_trades, f'CH_{fname}')

        combo = combine_daily_pnl(l8_daily, ch_stats['daily_pnl'])
        cs = stats_from_daily(combo, f'Combo_{fname}')
        cs['fold'] = fname
        cs['l8_sharpe'] = round(l8_stats['sharpe'], 2)
        cs['ch_sharpe'] = ch_stats['sharpe']
        results.append(cs)

    sharpes = [r['sharpe'] for r in results]
    pass_count = sum(1 for s in sharpes if s > 0)
    return {
        'label': label, 'folds': results, 'sharpes': sharpes,
        'mean_sharpe': round(np.mean(sharpes), 2) if sharpes else 0,
        'min_sharpe': round(min(sharpes), 2) if sharpes else 0,
        'pass': f"{pass_count}/{len(results)}",
    }


def run_l8(data, label="L8", **extra_kw):
    kw = {**L8_KWARGS, **extra_kw}
    stats = run_variant(data, label, verbose=False, **kw)
    l8_daily = {}
    for t in stats.get('_trades', []):
        d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
        l8_daily[d] = l8_daily.get(d, 0) + t.pnl
    return stats, l8_daily


def run_ch(h1_df, ch_params=None, ch_bt=None, spread_cost=0.0, label="CH"):
    if ch_params is None: ch_params = CH_PARAMS
    if ch_bt is None: ch_bt = CH_BT
    sig, atr = chandelier_signals(h1_df, **ch_params)
    trades = backtest_signals(h1_df, sig, atr, spread_cost=spread_cost, **ch_bt)
    stats = trades_to_stats(trades, label)
    return stats, stats['daily_pnl']


def run_dt(h1_df, dt_params=None, dt_bt=None, spread_cost=0.0, label="DT"):
    if dt_params is None: dt_params = {'k_buy': 0.6, 'k_sell': 0.6, 'lookback': 1}
    if dt_bt is None: dt_bt = {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
                                 'trail_act': 0.28, 'trail_dist': 0.06}
    sig, atr = dual_thrust_signals(h1_df, **dt_params)
    trades = backtest_signals(h1_df, sig, atr, spread_cost=spread_cost, **dt_bt)
    stats = trades_to_stats(trades, label)
    return stats, stats['daily_pnl']


# ═══════════════════════════════════════════════════════════════
# Phase A: 全组合验证
# ═══════════════════════════════════════════════════════════════

def phase_a(data, h1_df):
    phase_header("Phase A", "全策略组合验证")

    l8_s, l8_d = run_l8(data, "L8_BASE")
    print(f"  L8_BASE: Sharpe={l8_s['sharpe']:.2f}, PnL=${l8_s['total_pnl']:.0f}, N={l8_s['n']}")

    ch_s, ch_d = run_ch(h1_df, label="CH_p10_m3.0")
    print(f"  CH:      Sharpe={ch_s['sharpe']:.2f}, PnL=${ch_s['total_pnl']:.0f}, N={ch_s['n']}")

    dt_s, dt_d = run_dt(h1_df, label="DT_k0.6")
    print(f"  DT:      Sharpe={dt_s['sharpe']:.2f}, PnL=${dt_s['total_pnl']:.0f}, N={dt_s['n']}")

    combos = {}
    for name, dicts in [
        ("L8+CH",    [l8_d, ch_d]),
        ("L8+DT",    [l8_d, dt_d]),
        ("CH+DT",    [ch_d, dt_d]),
        ("L8+CH+DT", [l8_d, ch_d, dt_d]),
    ]:
        combined = combine_daily_pnl(*dicts)
        cs = stats_from_daily(combined, name)
        corr_info = {}
        if 'L8' in name and 'CH' in name:
            corr_info['l8_ch'] = daily_pnl_correlation(l8_d, ch_d)
        if 'L8' in name and 'DT' in name:
            corr_info['l8_dt'] = daily_pnl_correlation(l8_d, dt_d)
        if 'CH' in name and 'DT' in name:
            corr_info['ch_dt'] = daily_pnl_correlation(ch_d, dt_d)
        cs['correlations'] = corr_info
        combos[name] = cs
        print(f"  {name}: Sharpe={cs['sharpe']}, PnL=${cs['total_pnl']:.0f}, "
              f"MaxDD=${cs['max_dd']:.0f}, NegMonths={cs['neg_months']}, corr={corr_info}")

    # K-Fold on best combo
    print(f"\n  K-Fold verification...")
    kf_l8ch = kfold_combo(data, h1_df, L8_KWARGS, CH_PARAMS, CH_BT, "L8+CH")
    print(f"  L8+CH K-Fold: {kf_l8ch['pass']}, mean={kf_l8ch['mean_sharpe']}, min={kf_l8ch['min_sharpe']}")

    kf_ch = kfold_test_signal(h1_df, chandelier_signals, CH_PARAMS, CH_BT, "CH_solo")
    print(f"  CH solo K-Fold: {kf_ch['pass']}, mean={kf_ch['mean_sharpe']}, min={kf_ch['min_sharpe']}")

    results = {
        'baselines': {
            'L8': {k: v for k, v in l8_s.items() if k not in ('_trades', '_equity_curve', 'daily_pnl')},
            'CH': {k: v for k, v in ch_s.items() if k != 'daily_pnl'},
            'DT': {k: v for k, v in dt_s.items() if k != 'daily_pnl'},
        },
        'combos': combos,
        'kfold_l8ch': kf_l8ch,
        'kfold_ch': kf_ch,
    }
    save_json(results, 'A_combo_baselines.json')
    print(f"  Phase A complete. {elapsed()}")
    return l8_d, ch_d, dt_d


# ═══════════════════════════════════════════════════════════════
# Phase B: Chandelier 出场参数全扫描
# ═══════════════════════════════════════════════════════════════

def phase_b(h1_df, l8_daily):
    phase_header("Phase B", "Chandelier 出场参数全扫描")

    # B1: SL×TP grid
    print("  B1: SL×TP 网格扫描...")
    sl_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    tp_range = [4.0, 6.0, 8.0, 10.0, 12.0]
    grid_results = []
    for sl in sl_range:
        for tp in tp_range:
            bt_kw = {'sl_mult': sl, 'tp_mult': tp, 'max_hold': 20,
                     'trail_act': 0.28, 'trail_dist': 0.06}
            ch_s, ch_d = run_ch(h1_df, ch_bt=bt_kw, label=f"CH_sl{sl}_tp{tp}")
            combo = combine_daily_pnl(l8_daily, ch_d)
            cs = stats_from_daily(combo, f"Combo_sl{sl}_tp{tp}")
            row = {'sl': sl, 'tp': tp, 'ch_sharpe': ch_s['sharpe'], 'ch_n': ch_s['n'],
                   'ch_wr': ch_s['win_rate'], 'combo_sharpe': cs['sharpe'],
                   'combo_pnl': cs['total_pnl'], 'combo_maxdd': cs['max_dd']}
            grid_results.append(row)
            if ch_s['n'] > 0:
                print(f"    SL={sl}, TP={tp}: CH Sharpe={ch_s['sharpe']:.2f}, "
                      f"Combo Sharpe={cs['sharpe']:.2f}, N={ch_s['n']}")

    grid_results.sort(key=lambda x: x['combo_sharpe'], reverse=True)
    best_exit = grid_results[0]
    print(f"\n  ★ Best SL/TP: SL={best_exit['sl']}, TP={best_exit['tp']}, "
          f"Combo Sharpe={best_exit['combo_sharpe']}")

    # B2: MaxHold sweep with best SL/TP
    print(f"\n  B2: MaxHold 扫描 (SL={best_exit['sl']}, TP={best_exit['tp']})...")
    mh_results = []
    for mh in [8, 12, 16, 20, 30, 50]:
        bt_kw = {'sl_mult': best_exit['sl'], 'tp_mult': best_exit['tp'],
                 'max_hold': mh, 'trail_act': 0.28, 'trail_dist': 0.06}
        ch_s, ch_d = run_ch(h1_df, ch_bt=bt_kw, label=f"CH_mh{mh}")
        combo = combine_daily_pnl(l8_daily, ch_d)
        cs = stats_from_daily(combo, f"Combo_mh{mh}")
        row = {'max_hold': mh, 'ch_sharpe': ch_s['sharpe'], 'ch_n': ch_s['n'],
               'combo_sharpe': cs['sharpe'], 'combo_maxdd': cs['max_dd']}
        mh_results.append(row)
        print(f"    MH={mh}: CH Sharpe={ch_s['sharpe']:.2f}, Combo={cs['sharpe']:.2f}")

    best_mh = max(mh_results, key=lambda x: x['combo_sharpe'])

    # B3: Trail param sweep
    print(f"\n  B3: Trail 参数扫描...")
    trail_results = []
    for act in [0.10, 0.15, 0.20, 0.28, 0.35, 0.45]:
        for dist in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
            if dist >= act:
                continue
            bt_kw = {'sl_mult': best_exit['sl'], 'tp_mult': best_exit['tp'],
                     'max_hold': best_mh['max_hold'], 'trail_act': act, 'trail_dist': dist}
            ch_s, ch_d = run_ch(h1_df, ch_bt=bt_kw, label=f"CH_t{act}_{dist}")
            combo = combine_daily_pnl(l8_daily, ch_d)
            cs = stats_from_daily(combo, f"Combo_t{act}_{dist}")
            row = {'trail_act': act, 'trail_dist': dist, 'ch_sharpe': ch_s['sharpe'],
                   'combo_sharpe': cs['sharpe'], 'combo_maxdd': cs['max_dd'], 'ch_n': ch_s['n']}
            trail_results.append(row)

    trail_results.sort(key=lambda x: x['combo_sharpe'], reverse=True)
    best_trail = trail_results[0]
    print(f"  ★ Best Trail: act={best_trail['trail_act']}, dist={best_trail['trail_dist']}, "
          f"Combo={best_trail['combo_sharpe']}")

    # B4: K-Fold on best config
    best_bt = {
        'sl_mult': best_exit['sl'], 'tp_mult': best_exit['tp'],
        'max_hold': best_mh['max_hold'],
        'trail_act': best_trail['trail_act'], 'trail_dist': best_trail['trail_dist'],
    }
    print(f"\n  B4: K-Fold on best exit config: {best_bt}")
    kf = kfold_test_signal(h1_df, chandelier_signals, CH_PARAMS, best_bt, "CH_bestExit")
    print(f"  K-Fold: {kf['pass']}, mean={kf['mean_sharpe']}, min={kf['min_sharpe']}")

    # B5: Compare best vs original
    ch_best_s, ch_best_d = run_ch(h1_df, ch_bt=best_bt, label="CH_optimized")
    combo_best = stats_from_daily(combine_daily_pnl(l8_daily, ch_best_d), "Combo_optimized")
    ch_orig_s, ch_orig_d = run_ch(h1_df, label="CH_original")
    combo_orig = stats_from_daily(combine_daily_pnl(l8_daily, ch_orig_d), "Combo_original")
    print(f"\n  B5: Original vs Optimized:")
    print(f"    Original:  CH Sharpe={ch_orig_s['sharpe']}, Combo Sharpe={combo_orig['sharpe']}, MaxDD=${combo_orig['max_dd']}")
    print(f"    Optimized: CH Sharpe={ch_best_s['sharpe']}, Combo Sharpe={combo_best['sharpe']}, MaxDD=${combo_best['max_dd']}")

    results = {
        'sl_tp_grid': grid_results[:20],
        'maxhold_sweep': mh_results,
        'trail_sweep': trail_results[:20],
        'best_exit': best_bt,
        'kfold': kf,
        'comparison': {
            'original': {'ch_sharpe': ch_orig_s['sharpe'], 'combo_sharpe': combo_orig['sharpe']},
            'optimized': {'ch_sharpe': ch_best_s['sharpe'], 'combo_sharpe': combo_best['sharpe']},
        },
    }
    save_json(results, 'B_exit_optimization.json')
    print(f"  Phase B complete. {elapsed()}")
    return best_bt


# ═══════════════════════════════════════════════════════════════
# Phase C: Chandelier 入场增强
# ═══════════════════════════════════════════════════════════════

def phase_c(h1_df, l8_daily, ch_bt):
    phase_header("Phase C", "Chandelier 入场增强")

    baseline_s, baseline_d = run_ch(h1_df, ch_bt=ch_bt, label="CH_baseline")
    baseline_combo = stats_from_daily(combine_daily_pnl(l8_daily, baseline_d), "baseline")
    print(f"  Baseline: CH Sharpe={baseline_s['sharpe']}, Combo={baseline_combo['sharpe']}")

    # C1: ADX filter
    print(f"\n  C1: ADX 过滤...")
    adx_results = []
    for adx_min in [0, 14, 18, 22, 25, 30]:
        sig_p = {**CH_PARAMS, 'adx_min': adx_min}
        ch_s, ch_d = run_ch(h1_df, ch_params=sig_p, ch_bt=ch_bt, label=f"CH_adx{adx_min}")
        combo = stats_from_daily(combine_daily_pnl(l8_daily, ch_d), f"Combo_adx{adx_min}")
        row = {'adx_min': adx_min, 'ch_sharpe': ch_s['sharpe'], 'ch_n': ch_s['n'],
               'combo_sharpe': combo['sharpe'], 'combo_maxdd': combo['max_dd']}
        adx_results.append(row)
        print(f"    ADX>={adx_min}: CH N={ch_s['n']}, Sharpe={ch_s['sharpe']:.2f}, Combo={combo['sharpe']:.2f}")

    # C2: ATR percentile regime filter
    print(f"\n  C2: ATR Regime 过滤...")
    atr_results = []
    for lo, hi in [(0, 100), (10, 90), (20, 80), (25, 75), (30, 70)]:
        sig_p = {**CH_PARAMS, 'atr_pct_low': lo, 'atr_pct_high': hi}
        ch_s, ch_d = run_ch(h1_df, ch_params=sig_p, ch_bt=ch_bt, label=f"CH_atr{lo}_{hi}")
        combo = stats_from_daily(combine_daily_pnl(l8_daily, ch_d), f"Combo_atr{lo}_{hi}")
        row = {'atr_lo': lo, 'atr_hi': hi, 'ch_sharpe': ch_s['sharpe'], 'ch_n': ch_s['n'],
               'combo_sharpe': combo['sharpe']}
        atr_results.append(row)
        print(f"    ATR[{lo}-{hi}]: N={ch_s['n']}, CH Sharpe={ch_s['sharpe']:.2f}, Combo={combo['sharpe']:.2f}")

    # C3: H1 KC 同向过滤
    print(f"\n  C3: H1 KC 同向过滤...")
    kc_results = []
    for use_kc in [False, True]:
        sig_p = {**CH_PARAMS, 'h1_kc_filter': use_kc}
        ch_s, ch_d = run_ch(h1_df, ch_params=sig_p, ch_bt=ch_bt, label=f"CH_kc{use_kc}")
        combo = stats_from_daily(combine_daily_pnl(l8_daily, ch_d), f"Combo_kc{use_kc}")
        row = {'h1_kc_filter': use_kc, 'ch_sharpe': ch_s['sharpe'], 'ch_n': ch_s['n'],
               'combo_sharpe': combo['sharpe']}
        kc_results.append(row)
        print(f"    KC_filter={use_kc}: N={ch_s['n']}, CH Sharpe={ch_s['sharpe']:.2f}, Combo={combo['sharpe']:.2f}")

    # C4: Entry gap
    print(f"\n  C4: 入场间隔...")
    gap_results = []
    for gap in [0, 1, 2, 4, 6]:
        ch_s_r, ch_d_r = run_ch_with_gap(h1_df, ch_bt, gap, label=f"CH_gap{gap}")
        combo = stats_from_daily(combine_daily_pnl(l8_daily, ch_d_r), f"Combo_gap{gap}")
        row = {'gap_bars': gap, 'ch_sharpe': ch_s_r['sharpe'], 'ch_n': ch_s_r['n'],
               'combo_sharpe': combo['sharpe']}
        gap_results.append(row)
        print(f"    Gap={gap}h: N={ch_s_r['n']}, CH Sharpe={ch_s_r['sharpe']:.2f}, Combo={combo['sharpe']:.2f}")

    # C5: Best combo K-Fold
    best_adx = max(adx_results, key=lambda x: x['combo_sharpe'])
    best_atr = max(atr_results, key=lambda x: x['combo_sharpe'])
    best_gap = max(gap_results, key=lambda x: x['combo_sharpe'])

    best_sig = {**CH_PARAMS}
    if best_adx['adx_min'] > 0 and best_adx['combo_sharpe'] > baseline_combo['sharpe'] + 0.1:
        best_sig['adx_min'] = best_adx['adx_min']
    if best_atr['atr_lo'] > 0:
        if best_atr['combo_sharpe'] > baseline_combo['sharpe'] + 0.1:
            best_sig['atr_pct_low'] = best_atr['atr_lo']
            best_sig['atr_pct_high'] = best_atr['atr_hi']

    print(f"\n  C5: Best entry config: {best_sig}")
    kf = kfold_test_signal(h1_df, chandelier_signals, best_sig, ch_bt, "CH_bestEntry")
    print(f"  K-Fold: {kf['pass']}, mean={kf['mean_sharpe']}, min={kf['min_sharpe']}")

    results = {
        'adx_sweep': adx_results,
        'atr_regime_sweep': atr_results,
        'kc_filter': kc_results,
        'gap_sweep': gap_results,
        'best_entry': best_sig,
        'kfold': kf,
    }
    save_json(results, 'C_entry_enhancement.json')
    print(f"  Phase C complete. {elapsed()}")
    return best_sig


def run_ch_with_gap(h1_df, ch_bt, gap_bars, label="CH"):
    sig, atr = chandelier_signals(h1_df, **CH_PARAMS)
    trades = backtest_signals(h1_df, sig, atr, min_gap_bars=gap_bars, **ch_bt)
    stats = trades_to_stats(trades, label)
    return stats, stats['daily_pnl']


# ═══════════════════════════════════════════════════════════════
# Phase D: 多资产 Chandelier
# ═══════════════════════════════════════════════════════════════

def phase_d(xau_h1_daily):
    phase_header("Phase D", "多资产 Chandelier")

    from backtest.runner import load_csv

    assets = {
        'XAGUSD': [
            "data/download/xagusd-h1-bid-2015-01-01-2026-04-27.csv",
            "data/download/xagusd-h1-bid-2015-01-01-2026-04-10.csv",
        ],
        'EURUSD': [
            "data/download/eurusd-h1-bid-2015-01-01-2026-04-27.csv",
            "data/download/eurusd-h1-bid-2015-01-01-2026-04-10.csv",
        ],
    }

    all_dailies = {'XAUUSD': xau_h1_daily}
    asset_results = {}

    for asset_name, candidates in assets.items():
        csv_path = None
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break
        if csv_path is None:
            print(f"  {asset_name}: data not found, skipping")
            continue

        print(f"\n  {asset_name}: Loading from {csv_path}...")
        df = load_csv(csv_path)
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        from indicators import calc_adx as _calc_adx
        df['ADX'] = _calc_adx(df, 14)
        df['atr_percentile'] = df['ATR'].rolling(500, min_periods=50).rank(pct=True)
        print(f"    {len(df)} H1 bars loaded")

        # Parameter scan
        best_sharpe = -999
        best_cfg = None
        scan_results = []
        for period in [10, 15, 22]:
            for mult in [2.0, 3.0, 4.0]:
                sig_p = {'period': period, 'mult': mult, 'ema_filter': False}
                bt_p = {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
                        'trail_act': 0.28, 'trail_dist': 0.06}
                ch_s, ch_d = run_ch(df, ch_params=sig_p, ch_bt=bt_p,
                                     label=f"{asset_name}_p{period}_m{mult}")
                row = {'period': period, 'mult': mult, 'sharpe': ch_s['sharpe'],
                       'n': ch_s['n'], 'pnl': ch_s['total_pnl'], 'wr': ch_s['win_rate']}
                scan_results.append(row)
                if ch_s['sharpe'] > best_sharpe:
                    best_sharpe = ch_s['sharpe']
                    best_cfg = row.copy()

        scan_results.sort(key=lambda x: x['sharpe'], reverse=True)
        print(f"  {asset_name} Best: p={best_cfg['period']}, m={best_cfg['mult']}, "
              f"Sharpe={best_cfg['sharpe']}, N={best_cfg['n']}")

        # K-Fold on best
        best_sig = {'period': best_cfg['period'], 'mult': best_cfg['mult'], 'ema_filter': False}
        best_bt = {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
                   'trail_act': 0.28, 'trail_dist': 0.06}
        kf = kfold_test_signal(df, chandelier_signals, best_sig, best_bt, f"{asset_name}_CH")
        print(f"  K-Fold: {kf['pass']}, mean={kf['mean_sharpe']}, min={kf['min_sharpe']}")

        # Collect daily PnL for cross-asset correlation
        ch_s_best, ch_d_best = run_ch(df, ch_params=best_sig, ch_bt=best_bt, label=f"{asset_name}_best")
        all_dailies[asset_name] = ch_d_best

        asset_results[asset_name] = {
            'scan': scan_results[:10],
            'best': best_cfg,
            'kfold': kf,
        }

    # D4: Cross-asset correlation
    print(f"\n  D4: 跨资产相关性矩阵...")
    corr_matrix = {}
    asset_names = list(all_dailies.keys())
    for i, a1 in enumerate(asset_names):
        for a2 in asset_names[i+1:]:
            corr = daily_pnl_correlation(all_dailies[a1], all_dailies[a2])
            key = f"{a1}_vs_{a2}"
            corr_matrix[key] = corr
            print(f"    {key}: r={corr}")

    asset_results['cross_asset_correlation'] = corr_matrix
    save_json(asset_results, 'D_multi_asset.json')
    print(f"  Phase D complete. {elapsed()}")
    return asset_results


# ═══════════════════════════════════════════════════════════════
# Phase E: L8 引擎参数微调确认
# ═══════════════════════════════════════════════════════════════

def phase_e(data, h1_df):
    phase_header("Phase E", "L8 引擎参数微调确认")

    # E1: ADX 14 vs 18
    print("  E1: L8 ADX threshold...")
    adx_results = []
    for adx in [14, 16, 18, 20]:
        kw = {**L8_KWARGS, 'keltner_adx_threshold': adx}
        s, d = run_l8(data, f"L8_adx{adx}", **{'keltner_adx_threshold': adx})
        row = {'adx': adx, 'sharpe': round(s['sharpe'], 2), 'n': s['n'],
               'pnl': round(s['total_pnl'], 0), 'wr': round(s['win_rate'], 1)}
        adx_results.append(row)
        print(f"    ADX={adx}: Sharpe={s['sharpe']:.2f}, N={s['n']}, PnL=${s['total_pnl']:.0f}")

    # E2: MaxHold sweep for L8
    print(f"\n  E2: L8 MaxHold...")
    mh_results = []
    for mh in [8, 12, 16, 20]:
        s, d = run_l8(data, f"L8_mh{mh}", **{'keltner_max_hold_m15': mh})
        row = {'max_hold': mh, 'sharpe': round(s['sharpe'], 2), 'n': s['n'],
               'pnl': round(s['total_pnl'], 0)}
        mh_results.append(row)
        print(f"    MH={mh}: Sharpe={s['sharpe']:.2f}, N={s['n']}, PnL=${s['total_pnl']:.0f}")

    # E3: Regime trail variants
    print(f"\n  E3: Trail Regime 对比...")
    regime_configs = {
        '3-tier_current': {
            'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
        },
        'flat_normal': {
            'low':    {'trail_act': 0.14, 'trail_dist': 0.025},
            'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
            'high':   {'trail_act': 0.14, 'trail_dist': 0.025},
        },
        '3-tier_tighter': {
            'low':    {'trail_act': 0.18, 'trail_dist': 0.03},
            'normal': {'trail_act': 0.10, 'trail_dist': 0.02},
            'high':   {'trail_act': 0.04, 'trail_dist': 0.005},
        },
    }
    regime_results = []
    for name, rc in regime_configs.items():
        s, d = run_l8(data, f"L8_{name}", **{'regime_config': rc})
        row = {'regime': name, 'sharpe': round(s['sharpe'], 2), 'n': s['n'],
               'pnl': round(s['total_pnl'], 0), 'max_dd': round(s['max_dd'], 0)}
        regime_results.append(row)
        print(f"    {name}: Sharpe={s['sharpe']:.2f}, PnL=${s['total_pnl']:.0f}")

    # E4: EqCurve
    print(f"\n  E4: L8 + EqCurve...")
    for lb in [10, 20, 30, 50]:
        kw_eq = {'equity_curve_filter': True, 'equity_ma_period': lb}
        s, d = run_l8(data, f"L8_eq{lb}", **kw_eq)
        print(f"    EqCurve MA={lb}: Sharpe={s['sharpe']:.2f}, N={s['n']}, PnL=${s['total_pnl']:.0f}")

    # E5: KC Bandwidth filter
    print(f"\n  E5: L8 + KC Bandwidth Filter...")
    for bw_bars in [0, 3, 5, 8]:
        s, d = run_l8(data, f"L8_kcbw{bw_bars}", **{'kc_bw_filter_bars': bw_bars})
        print(f"    KCBW={bw_bars}: Sharpe={s['sharpe']:.2f}, N={s['n']}, PnL=${s['total_pnl']:.0f}")

    results = {
        'adx_sweep': adx_results,
        'maxhold_sweep': mh_results,
        'regime_trail': regime_results,
    }
    save_json(results, 'E_l8_tuning.json')
    print(f"  Phase E complete. {elapsed()}")


# ═══════════════════════════════════════════════════════════════
# Phase F: 终极组合优化
# ═══════════════════════════════════════════════════════════════

def phase_f(data, h1_df, best_ch_bt, best_ch_sig=None):
    phase_header("Phase F", "终极组合优化")

    if best_ch_sig is None:
        best_ch_sig = CH_PARAMS

    # F1: Lot ratio
    print("  F1: CH Lot 比例扫描...")
    l8_s, l8_d = run_l8(data, "L8_base_f")
    lot_results = []
    for ratio in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        ch_s, ch_d = run_ch(h1_df, ch_params=best_ch_sig, ch_bt=best_ch_bt, label=f"CH_r{ratio}")
        # Scale CH PnL by ratio
        scaled_d = {d: p * ratio for d, p in ch_d.items()}
        combo = combine_daily_pnl(l8_d, scaled_d)
        cs = stats_from_daily(combo, f"Combo_r{ratio}")
        row = {'ratio': ratio, 'combo_sharpe': cs['sharpe'], 'combo_pnl': cs['total_pnl'],
               'combo_maxdd': cs['max_dd'], 'neg_months': cs['neg_months']}
        lot_results.append(row)
        print(f"    Ratio={ratio}x: Sharpe={cs['sharpe']:.2f}, PnL=${cs['total_pnl']:.0f}, MaxDD=${cs['max_dd']:.0f}")

    best_ratio = max(lot_results, key=lambda x: x['combo_sharpe'])

    # F2: Spread stress
    print(f"\n  F2: Spread 压力测试...")
    spread_results = []
    for sp in [0.0, 0.30, 0.50, 0.75, 1.00, 1.50]:
        ch_s, ch_d = run_ch(h1_df, ch_params=best_ch_sig, ch_bt=best_ch_bt,
                             spread_cost=sp, label=f"CH_sp{sp}")
        combo = combine_daily_pnl(l8_d, ch_d)
        cs = stats_from_daily(combo, f"Combo_sp{sp}")
        row = {'spread': sp, 'ch_sharpe': ch_s['sharpe'], 'combo_sharpe': cs['sharpe'],
               'combo_maxdd': cs['max_dd']}
        spread_results.append(row)
        print(f"    Spread=${sp}: CH Sharpe={ch_s['sharpe']:.2f}, Combo={cs['sharpe']:.2f}")

    # F3: Walk-Forward 24 windows
    print(f"\n  F3: Walk-Forward...")
    wf_results = []
    for year in range(2015, 2027):
        for half, (start, end) in [("H1", (f"{year}-01-01", f"{year}-07-01")),
                                    ("H2", (f"{year}-07-01", f"{year+1}-01-01") if year < 2026
                                     else (f"{year}-04-01", f"{year}-04-28"))]:
            name = f"{year}{half}"
            try:
                slice_d = data.slice(start, end)
                fold_h1 = h1_df[start:end]
                if len(fold_h1) < 100 or len(slice_d.m15_df) < 300:
                    continue
                l8_fold = run_variant(slice_d, f'L8_{name}', verbose=False, **L8_KWARGS)
                l8_fd = {}
                for t in l8_fold.get('_trades', []):
                    d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
                    l8_fd[d] = l8_fd.get(d, 0) + t.pnl
                sig, atr = chandelier_signals(fold_h1, **best_ch_sig)
                ch_trades = backtest_signals(fold_h1, sig, atr, **best_ch_bt)
                ch_fold = trades_to_stats(ch_trades, f'CH_{name}')
                combo = combine_daily_pnl(l8_fd, ch_fold['daily_pnl'])
                cs = stats_from_daily(combo, name)
                row = {'window': name, 'l8_sharpe': round(l8_fold['sharpe'], 2),
                       'ch_sharpe': ch_fold['sharpe'], 'combo_sharpe': cs['sharpe'],
                       'combo_pnl': cs['total_pnl']}
                wf_results.append(row)
                print(f"    {name}: L8={l8_fold['sharpe']:.2f}, CH={ch_fold['sharpe']}, Combo={cs['sharpe']}")
            except Exception as e:
                print(f"    {name}: error - {e}")

    profitable_wf = sum(1 for r in wf_results if r['combo_pnl'] > 0)
    print(f"  ★ {profitable_wf}/{len(wf_results)} windows profitable")

    # F4: Crisis periods
    print(f"\n  F4: 危机时段...")
    crises = [
        ("Brexit_2016",       "2016-06-20", "2016-07-20"),
        ("Trade_War_2018",    "2018-09-01", "2018-12-31"),
        ("COVID_Crash_2020",  "2020-02-20", "2020-04-15"),
        ("Inflation_2022",    "2022-03-01", "2022-06-30"),
        ("SVB_2023",          "2023-03-01", "2023-04-15"),
        ("Liberation_Day_25", "2025-03-15", "2025-05-15"),
        ("Tariff_2026",       "2026-01-01", "2026-04-25"),
    ]
    crisis_results = []
    for cname, start, end in crises:
        try:
            slice_d = data.slice(start, end)
            fold_h1 = h1_df[start:end]
            if len(fold_h1) < 50:
                continue
            l8_fold = run_variant(slice_d, f'L8_{cname}', verbose=False, **L8_KWARGS)
            l8_fd = {}
            for t in l8_fold.get('_trades', []):
                d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
                l8_fd[d] = l8_fd.get(d, 0) + t.pnl
            sig, atr = chandelier_signals(fold_h1, **best_ch_sig)
            ch_trades = backtest_signals(fold_h1, sig, atr, **best_ch_bt)
            ch_fold = trades_to_stats(ch_trades, f'CH_{cname}')
            combo = combine_daily_pnl(l8_fd, ch_fold['daily_pnl'])
            cs = stats_from_daily(combo, cname)
            row = {'crisis': cname, 'l8_sharpe': round(l8_fold['sharpe'], 2),
                   'ch_sharpe': ch_fold['sharpe'], 'combo_sharpe': cs['sharpe'],
                   'combo_pnl': cs['total_pnl'], 'combo_maxdd': cs['max_dd']}
            crisis_results.append(row)
            print(f"    {cname}: Combo Sharpe={cs['sharpe']}, PnL=${cs['total_pnl']:.0f}")
        except Exception as e:
            print(f"    {cname}: error - {e}")

    # F5: Monte Carlo
    print(f"\n  F5: Monte Carlo 50x 参数扰动...")
    mc_results = []
    rng = random.Random(42)
    for trial in range(50):
        pct_mult = 1.0 + (rng.random() - 0.5) * 0.4  # ±20%
        pct_period = max(5, int(CH_PARAMS['period'] * (1.0 + (rng.random() - 0.5) * 0.4)))
        mc_sig = {'period': pct_period, 'mult': CH_PARAMS['mult'] * pct_mult, 'ema_filter': False}
        mc_bt = {k: v * (1.0 + (rng.random() - 0.5) * 0.4) if isinstance(v, float) else v
                 for k, v in best_ch_bt.items()}
        mc_bt['max_hold'] = max(5, int(mc_bt.get('max_hold', 20)))

        try:
            ch_s, ch_d = run_ch(h1_df, ch_params=mc_sig, ch_bt=mc_bt, label=f"MC_{trial}")
            combo = combine_daily_pnl(l8_d, ch_d)
            cs = stats_from_daily(combo, f"MC_{trial}")
            mc_results.append({'trial': trial, 'combo_sharpe': cs['sharpe'],
                                'ch_sharpe': ch_s['sharpe'], 'ch_n': ch_s['n']})
        except:
            pass

    if mc_results:
        mc_sharpes = [r['combo_sharpe'] for r in mc_results]
        print(f"  MC: mean={np.mean(mc_sharpes):.2f}, min={min(mc_sharpes):.2f}, "
              f"max={max(mc_sharpes):.2f}, %positive={sum(1 for s in mc_sharpes if s > 0)/len(mc_sharpes)*100:.0f}%")

    # F6: Annual decomposition
    print(f"\n  F6: 年度分解...")
    annual_results = []
    for year in range(2015, 2027):
        start = f"{year}-01-01"
        end = f"{year+1}-01-01" if year < 2026 else f"{year}-04-28"
        try:
            slice_d = data.slice(start, end)
            fold_h1 = h1_df[start:end]
            if len(fold_h1) < 100:
                continue
            l8_fold = run_variant(slice_d, f'L8_{year}', verbose=False, **L8_KWARGS)
            l8_fd = {}
            for t in l8_fold.get('_trades', []):
                d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
                l8_fd[d] = l8_fd.get(d, 0) + t.pnl
            sig, atr = chandelier_signals(fold_h1, **best_ch_sig)
            ch_trades = backtest_signals(fold_h1, sig, atr, **best_ch_bt)
            ch_fold = trades_to_stats(ch_trades, f'CH_{year}')
            combo = combine_daily_pnl(l8_fd, ch_fold['daily_pnl'])
            cs = stats_from_daily(combo, str(year))
            row = {'year': year, 'l8_pnl': round(sum(l8_fd.values()), 0),
                   'ch_pnl': round(ch_fold['total_pnl'], 0),
                   'combo_pnl': cs['total_pnl'], 'combo_sharpe': cs['sharpe']}
            annual_results.append(row)
            print(f"    {year}: L8=${sum(l8_fd.values()):.0f}, CH=${ch_fold['total_pnl']:.0f}, "
                  f"Combo=${cs['total_pnl']:.0f}, Sharpe={cs['sharpe']}")
        except Exception as e:
            print(f"    {year}: error - {e}")

    # F7: BUY/SELL direction analysis
    print(f"\n  F7: 方向分析...")
    sig, atr = chandelier_signals(h1_df, **best_ch_sig)
    all_trades = backtest_signals(h1_df, sig, atr, **best_ch_bt)
    buy_trades = [t for t in all_trades if t.direction == 'BUY']
    sell_trades = [t for t in all_trades if t.direction == 'SELL']
    buy_s = trades_to_stats(buy_trades, "CH_BUY")
    sell_s = trades_to_stats(sell_trades, "CH_SELL")
    print(f"    BUY:  N={buy_s['n']}, Sharpe={buy_s['sharpe']}, PnL=${buy_s['total_pnl']:.0f}, WR={buy_s['win_rate']}%")
    print(f"    SELL: N={sell_s['n']}, Sharpe={sell_s['sharpe']}, PnL=${sell_s['total_pnl']:.0f}, WR={sell_s['win_rate']}%")

    results = {
        'lot_ratio': lot_results,
        'best_ratio': best_ratio,
        'spread_stress': spread_results,
        'walk_forward': wf_results,
        'crisis': crisis_results,
        'monte_carlo_summary': {
            'n_trials': len(mc_results),
            'mean_sharpe': round(np.mean([r['combo_sharpe'] for r in mc_results]), 2) if mc_results else 0,
            'min_sharpe': round(min([r['combo_sharpe'] for r in mc_results]), 2) if mc_results else 0,
            'pct_positive': round(sum(1 for r in mc_results if r['combo_sharpe'] > 0) / max(len(mc_results), 1) * 100, 1),
        },
        'annual': annual_results,
        'direction': {
            'buy': {k: v for k, v in buy_s.items() if k != 'daily_pnl'},
            'sell': {k: v for k, v in sell_s.items() if k != 'daily_pnl'},
        },
    }
    save_json(results, 'F_ultimate_combo.json')
    print(f"  Phase F complete. {elapsed()}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase G: Final Report
# ═══════════════════════════════════════════════════════════════

def phase_g():
    phase_header("Phase G", "最终报告")

    json_files = sorted(OUT_DIR.glob('*.json'))
    summary = {}
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            summary[jf.stem] = data
        except:
            pass

    total_hours = (time.time() - MARATHON_START) / 3600
    summary['_meta'] = {
        'total_runtime_hours': round(total_hours, 2),
        'timestamp': datetime.now().isoformat(),
        'phases_completed': len(json_files),
    }

    save_json(summary, 'G_final_summary.json')
    print(f"\n  ★ R47 Marathon 完成!")
    print(f"    总耗时: {total_hours:.1f} 小时")
    print(f"    输出目录: {OUT_DIR}")
    print(f"    结果文件: {len(json_files)} JSON")
    print(f"\n{'='*70}")
    print(f"  R47 Marathon 结束. {elapsed()}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    log_path = OUT_DIR / "r47_stdout.txt"
    import io
    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    print(f"{'='*70}")
    print(f"  R47 MARATHON — 全策略组合终极验证")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Output:  {OUT_DIR}")
    print(f"{'='*70}\n")

    try:
        # Check for resume mode: skip phases whose JSON already exists
        resume = os.environ.get('R47_RESUME', '0') == '1'

        data = DataBundle.load_default()
        h1_df = data.h1_df.copy()
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).rank(pct=True)

        if resume and (OUT_DIR / 'A_combo_baselines.json').exists():
            print("  [RESUME] Phase A already done, recomputing baselines for daily PnL...")
            l8_s, l8_daily = run_l8(data, "L8_BASE")
            ch_s, ch_daily = run_ch(h1_df, label="CH_p10_m3.0")
            dt_s, dt_daily = run_dt(h1_df, label="DT_k0.6")
        else:
            l8_daily, ch_daily, dt_daily = phase_a(data, h1_df)
        gc.collect()

        if resume and (OUT_DIR / 'B_exit_optimization.json').exists():
            print("  [RESUME] Phase B already done, loading best_exit...")
            with open(OUT_DIR / 'B_exit_optimization.json') as f:
                b_data = json.load(f)
            best_ch_bt = b_data['best_exit']
        else:
            best_ch_bt = phase_b(h1_df, l8_daily)
        gc.collect()

        if resume and (OUT_DIR / 'C_entry_enhancement.json').exists():
            print("  [RESUME] Phase C already done, loading best_entry...")
            with open(OUT_DIR / 'C_entry_enhancement.json') as f:
                c_data = json.load(f)
            best_ch_sig = c_data['best_entry']
        else:
            best_ch_sig = phase_c(h1_df, l8_daily, best_ch_bt)
        gc.collect()

        if resume and (OUT_DIR / 'D_multi_asset.json').exists():
            print("  [RESUME] Phase D already done, skipping...")
        else:
            phase_d(ch_daily)
        gc.collect()

        # Phase E
        phase_e(data, h1_df)
        gc.collect()

        # Phase F
        phase_f(data, h1_df, best_ch_bt, best_ch_sig)
        gc.collect()

        # Phase G
        phase_g()

    except Exception as e:
        print(f"\n!!! FATAL ERROR: {e}")
        traceback.print_exc()

    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_path}")


if __name__ == '__main__':
    main()
