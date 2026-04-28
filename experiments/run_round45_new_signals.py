"""R45: 全新信号源探索 — 6 类独立策略
=====================================
S1: Donchian Channel Breakout (趋势追踪)
S2: Bollinger Band Squeeze-to-Expansion (波动率周期)
S3: Dual Thrust (开盘区间突破)
S4: Chandelier Exit Flip (ATR 极值反转)
S5: Z-Score Mean Reversion (反趋势)
S6: Range Contraction Filter (叠加到 L8)

每个策略: 参数扫描 → K-Fold 6/6 → 与 L7/L8 相关性 → Spread 敏感度

USAGE
-----
    python -m experiments.run_round45_new_signals
"""
import sys, os, time, json, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from backtest.runner import (
    DataBundle, run_variant, run_kfold,
    LIVE_PARITY_KWARGS, sanitize_for_json
)
from backtest.stats import calc_stats
from backtest.engine import TradeRecord
from indicators import (
    calc_donchian, calc_chandelier, calc_zscore,
    calc_dual_thrust_range, calc_range_contraction, prepare_indicators
)

OUT_DIR = ROOT / "results" / "round45_new_signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARATHON_START = time.time()


# ══════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# Generic signal backtester (not using engine.py)
# ══════════════════════════════════════════════════════════════

@dataclass
class SimpleTrade:
    entry_time: datetime
    exit_time: datetime
    direction: str       # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str

def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,        # +1 = BUY, -1 = SELL, 0 = no signal
    atr: pd.Series,
    sl_mult: float = 3.0,
    tp_mult: float = 8.0,
    max_hold: int = 20,
    trail_act: float = 0.28,
    trail_dist: float = 0.06,
    spread_cost: float = 0.0,
    label: str = "",
) -> List[SimpleTrade]:
    """Simple vectorized-ish backtester for new signal evaluation."""
    trades = []
    pos = None  # (direction, entry_price, entry_bar, sl, tp, trail_price, atr_at_entry)

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

            # Trailing stop update
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
            entry_price = opens[i + 1]
            entry_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0

            if sig > 0:
                sl_price = entry_price - sl_mult * entry_atr
                tp_price = entry_price + tp_mult * entry_atr
                pos = ('BUY', entry_price, i + 1, sl_price, tp_price, None, entry_atr)
            elif sig < 0:
                sl_price = entry_price + sl_mult * entry_atr
                tp_price = entry_price - tp_mult * entry_atr
                pos = ('SELL', entry_price, i + 1, sl_price, tp_price, None, entry_atr)

    return trades


def trades_to_stats(trades: List[SimpleTrade], label: str = "") -> Dict:
    """Convert SimpleTrade list to stats dict compatible with our framework."""
    if not trades:
        return {'label': label, 'n': 0, 'total_pnl': 0, 'sharpe': 0,
                'win_rate': 0, 'max_dd': 0, 'avg_pnl': 0, 'daily_pnl': {}}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

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

    return {
        'label': label,
        'n': len(trades),
        'total_pnl': sum(pnls),
        'sharpe': round(sharpe, 2),
        'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'max_dd': round(max_dd, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'avg_bars': round(np.mean([t.bars_held for t in trades]), 1),
        'daily_pnl': daily_pnl,
    }


def kfold_test(h1_df: pd.DataFrame, signal_func, params: dict, label: str) -> Dict:
    """Run 6-fold K-Fold on a signal function."""
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
        signals, atr = signal_func(fold_df, **params)
        trades = backtest_signals(fold_df, signals, atr, **params.get('bt_kwargs', {}))
        stats = trades_to_stats(trades, f"{label}_{fname}")
        stats['fold'] = fname
        results.append(stats)

    sharpes = [r['sharpe'] for r in results]
    pass_count = sum(1 for s in sharpes if s > 0)
    return {
        'label': label,
        'folds': results,
        'sharpes': sharpes,
        'mean_sharpe': round(np.mean(sharpes), 2) if sharpes else 0,
        'min_sharpe': round(min(sharpes), 2) if sharpes else 0,
        'pass': f"{pass_count}/{len(results)}",
    }


def daily_pnl_correlation(daily_a: Dict, daily_b: Dict) -> float:
    """Compute correlation between two daily PnL dicts."""
    all_dates = sorted(set(daily_a.keys()) | set(daily_b.keys()))
    a = [daily_a.get(d, 0) for d in all_dates]
    b = [daily_b.get(d, 0) for d in all_dates]
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


# ══════════════════════════════════════════════════════════════
# S1: Donchian Channel Breakout
# ══════════════════════════════════════════════════════════════

def donchian_signals(df, lookback=20, ema_filter=True, sl_mult=3.0, tp_mult=8.0,
                     max_hold=20, trail_act=0.28, trail_dist=0.06, **kw):
    dc = calc_donchian(df, lookback)
    atr = (df['High'] - df['Low']).rolling(14).mean()
    ema100 = df['Close'].ewm(span=100).mean()

    sig = pd.Series(0, index=df.index)
    close = df['Close']
    prev_close = close.shift(1)

    buy_cond = (close > dc['DC_upper'].shift(1)) & (prev_close <= dc['DC_upper'].shift(1))
    sell_cond = (close < dc['DC_lower'].shift(1)) & (prev_close >= dc['DC_lower'].shift(1))

    if ema_filter:
        buy_cond = buy_cond & (close > ema100)
        sell_cond = sell_cond & (close < ema100)

    sig[buy_cond] = 1
    sig[sell_cond] = -1

    return sig, atr


def phase_s1(h1_df, l8_daily):
    phase_header("S1", "Donchian Channel Breakout")

    param_grid = []
    for lb in [10, 20, 30, 50]:
        for ema in [True, False]:
            for sl in [2.0, 3.0, 4.0]:
                param_grid.append({
                    'lookback': lb, 'ema_filter': ema,
                    'bt_kwargs': {'sl_mult': sl, 'tp_mult': 8.0, 'max_hold': lb,
                                  'trail_act': 0.28, 'trail_dist': 0.06},
                })

    results = []
    for p in param_grid:
        signals, atr = donchian_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"DC_lb{p['lookback']}_ema{p['ema_filter']}_sl{bt_kw['sl_mult']}")
        stats['params'] = {k: v for k, v in p.items() if k != 'bt_kwargs'}
        stats['params'].update(bt_kw)
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 5 Donchian configs:")
    for r in results[:5]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    best = results[0] if results else None
    kfold = None
    if best and best['sharpe'] > 1.0:
        best_params = {k: v for k, v in best['params'].items()}
        bt_kw_kf = {k: v for k, v in best_params.items()
                    if k in ('sl_mult', 'tp_mult', 'max_hold', 'trail_act', 'trail_dist')}
        sig_kw = {k: v for k, v in best_params.items() if k not in bt_kw_kf}
        sig_kw['bt_kwargs'] = bt_kw_kf
        kfold = kfold_test(h1_df, donchian_signals, sig_kw, "DC_best")
        print(f"  K-Fold: {kfold['pass']}, mean={kfold['mean_sharpe']}, min={kfold['min_sharpe']}")

    save_json({'grid': [{k: v for k, v in r.items() if k != 'daily_pnl'} for r in results[:20]],
               'kfold': kfold}, 'S1_donchian.json')
    print(f"  S1 complete. {elapsed()}")
    return results[0] if results else None


# ══════════════════════════════════════════════════════════════
# S2: BB Squeeze-to-Expansion
# ══════════════════════════════════════════════════════════════

def bb_squeeze_signals(df, bb_period=20, bb_std=2.0, squeeze_bars=3,
                       bw_pct_threshold=20, adx_min=15,
                       sl_mult=3.0, tp_mult=8.0, max_hold=15,
                       trail_act=0.28, trail_dist=0.06, **kw):
    close = df['Close']
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std
    bandwidth = (bb_upper - bb_lower) / sma.replace(0, np.nan)
    bw_pct = bandwidth.rolling(120).rank(pct=True) * 100

    in_squeeze = bw_pct < bw_pct_threshold
    squeeze_count = in_squeeze.rolling(squeeze_bars).sum()
    was_squeezed = squeeze_count.shift(1) >= squeeze_bars

    atr = (df['High'] - df['Low']).rolling(14).mean()
    adx = df['ADX'] if 'ADX' in df.columns else pd.Series(25, index=df.index)

    sig = pd.Series(0, index=df.index)
    buy_cond = was_squeezed & (close > bb_upper) & (adx >= adx_min)
    sell_cond = was_squeezed & (close < bb_lower) & (adx >= adx_min)
    sig[buy_cond] = 1
    sig[sell_cond] = -1

    return sig, atr


def phase_s2(h1_df, l8_daily):
    phase_header("S2", "BB Squeeze-to-Expansion")

    param_grid = []
    for bb_std in [2.0, 2.5]:
        for sq_bars in [3, 5, 8]:
            for bw_pct in [10, 20, 30]:
                for adx_min in [12, 18]:
                    param_grid.append({
                        'bb_std': bb_std, 'squeeze_bars': sq_bars,
                        'bw_pct_threshold': bw_pct, 'adx_min': adx_min,
                        'bt_kwargs': {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 15,
                                      'trail_act': 0.28, 'trail_dist': 0.06},
                    })

    results = []
    for p in param_grid:
        signals, atr = bb_squeeze_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"BBS_std{p['bb_std']}_sq{p['squeeze_bars']}_bw{p['bw_pct_threshold']}_adx{p['adx_min']}")
        stats['params'] = {k: v for k, v in p.items() if k != 'bt_kwargs'}
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 5 BB Squeeze configs:")
    for r in results[:5]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    best = results[0] if results else None
    kfold = None
    if best and best['sharpe'] > 1.0:
        sig_kw = {k: v for k, v in best['params'].items()}
        sig_kw['bt_kwargs'] = best.get('params', {}).get('bt_kwargs', 
                               {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 15,
                                'trail_act': 0.28, 'trail_dist': 0.06})
        kfold = kfold_test(h1_df, bb_squeeze_signals, sig_kw, "BBS_best")
        print(f"  K-Fold: {kfold['pass']}, mean={kfold['mean_sharpe']}, min={kfold['min_sharpe']}")

    save_json({'grid': [{k: v for k, v in r.items() if k != 'daily_pnl'} for r in results[:20]],
               'kfold': kfold}, 'S2_bb_squeeze.json')
    print(f"  S2 complete. {elapsed()}")
    return best


# ══════════════════════════════════════════════════════════════
# S3: Dual Thrust
# ══════════════════════════════════════════════════════════════

def dual_thrust_signals(df, n_bars=6, k_up=0.5, k_down=0.5,
                        sl_mult=3.0, tp_mult=6.0, max_hold=10,
                        trail_act=0.28, trail_dist=0.06, **kw):
    dt_range = calc_dual_thrust_range(df, n_bars)
    atr = (df['High'] - df['Low']).rolling(14).mean()

    dates = pd.Series(df.index.date, index=df.index)
    daily_open = df.groupby(dates)['Open'].transform('first')

    sig = pd.Series(0, index=df.index)
    buy_cond = df['Close'] > (daily_open + k_up * dt_range)
    sell_cond = df['Close'] < (daily_open - k_down * dt_range)
    sig[buy_cond] = 1
    sig[sell_cond] = -1

    first_signal = sig.groupby(dates).cumsum().abs()
    sig[first_signal > 1] = 0

    return sig, atr


def phase_s3(h1_df, l8_daily):
    phase_header("S3", "Dual Thrust")

    param_grid = []
    for n in [4, 6, 8]:
        for k in [0.3, 0.5, 0.7, 0.9]:
            param_grid.append({
                'n_bars': n, 'k_up': k, 'k_down': k,
                'bt_kwargs': {'sl_mult': 3.0, 'tp_mult': 6.0, 'max_hold': 10,
                              'trail_act': 0.28, 'trail_dist': 0.06},
            })

    results = []
    for p in param_grid:
        signals, atr = dual_thrust_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"DT_n{p['n_bars']}_k{p['k_up']}")
        stats['params'] = {k_: v for k_, v in p.items() if k_ != 'bt_kwargs'}
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 5 Dual Thrust configs:")
    for r in results[:5]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    best = results[0] if results else None
    kfold = None
    if best and best['sharpe'] > 1.0:
        sig_kw = {k_: v for k_, v in best['params'].items()}
        sig_kw['bt_kwargs'] = {'sl_mult': 3.0, 'tp_mult': 6.0, 'max_hold': 10,
                               'trail_act': 0.28, 'trail_dist': 0.06}
        kfold = kfold_test(h1_df, dual_thrust_signals, sig_kw, "DT_best")
        print(f"  K-Fold: {kfold['pass']}, mean={kfold['mean_sharpe']}, min={kfold['min_sharpe']}")

    save_json({'grid': [{k_: v for k_, v in r.items() if k_ != 'daily_pnl'} for r in results[:20]],
               'kfold': kfold}, 'S3_dual_thrust.json')
    print(f"  S3 complete. {elapsed()}")
    return best


# ══════════════════════════════════════════════════════════════
# S4: Chandelier Exit Flip
# ══════════════════════════════════════════════════════════════

def chandelier_signals(df, period=22, mult=3.0, ema_filter=True,
                       sl_mult=3.0, tp_mult=8.0, max_hold=20,
                       trail_act=0.28, trail_dist=0.06, **kw):
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

    return sig, atr


def phase_s4(h1_df, l8_daily):
    phase_header("S4", "Chandelier Exit Flip")

    param_grid = []
    for period in [10, 15, 22]:
        for mult in [2.0, 3.0, 4.0]:
            for ema in [True, False]:
                param_grid.append({
                    'period': period, 'mult': mult, 'ema_filter': ema,
                    'bt_kwargs': {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
                                  'trail_act': 0.28, 'trail_dist': 0.06},
                })

    results = []
    for p in param_grid:
        signals, atr = chandelier_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"CH_p{p['period']}_m{p['mult']}_ema{p['ema_filter']}")
        stats['params'] = {k: v for k, v in p.items() if k != 'bt_kwargs'}
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 5 Chandelier configs:")
    for r in results[:5]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    best = results[0] if results else None
    kfold = None
    if best and best['sharpe'] > 1.0:
        sig_kw = {k: v for k, v in best['params'].items()}
        sig_kw['bt_kwargs'] = {'sl_mult': 3.0, 'tp_mult': 8.0, 'max_hold': 20,
                               'trail_act': 0.28, 'trail_dist': 0.06}
        kfold = kfold_test(h1_df, chandelier_signals, sig_kw, "CH_best")
        print(f"  K-Fold: {kfold['pass']}, mean={kfold['mean_sharpe']}, min={kfold['min_sharpe']}")

    save_json({'grid': [{k: v for k, v in r.items() if k != 'daily_pnl'} for r in results[:20]],
               'kfold': kfold}, 'S4_chandelier.json')
    print(f"  S4 complete. {elapsed()}")
    return best


# ══════════════════════════════════════════════════════════════
# S5: Z-Score Mean Reversion
# ══════════════════════════════════════════════════════════════

def zscore_signals(df, sma_period=50, z_threshold=2.0, adx_cap=25,
                   sl_mult=2.0, tp_mult=4.0, max_hold=10,
                   trail_act=0.5, trail_dist=0.1, **kw):
    z = calc_zscore(df['Close'], sma_period)
    atr = (df['High'] - df['Low']).rolling(14).mean()
    adx = df['ADX'] if 'ADX' in df.columns else pd.Series(20, index=df.index)

    sig = pd.Series(0, index=df.index)
    buy_cond = (z < -z_threshold) & (adx < adx_cap)
    sell_cond = (z > z_threshold) & (adx < adx_cap)
    sig[buy_cond] = 1
    sig[sell_cond] = -1

    return sig, atr


def phase_s5(h1_df, l8_daily):
    phase_header("S5", "Z-Score Mean Reversion")

    param_grid = []
    for period in [20, 50, 100]:
        for z_th in [1.5, 2.0, 2.5, 3.0]:
            for adx_cap in [20, 25, 30]:
                param_grid.append({
                    'sma_period': period, 'z_threshold': z_th, 'adx_cap': adx_cap,
                    'bt_kwargs': {'sl_mult': 2.0, 'tp_mult': 4.0, 'max_hold': 10,
                                  'trail_act': 0.5, 'trail_dist': 0.1},
                })

    results = []
    for p in param_grid:
        signals, atr = zscore_signals(h1_df, **p)
        bt_kw = p.get('bt_kwargs', {})
        trades = backtest_signals(h1_df, signals, atr, **bt_kw)
        stats = trades_to_stats(trades, f"ZS_p{p['sma_period']}_z{p['z_threshold']}_adx{p['adx_cap']}")
        stats['params'] = {k: v for k, v in p.items() if k != 'bt_kwargs'}
        results.append(stats)

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("  Top 5 Z-Score configs:")
    for r in results[:5]:
        corr = daily_pnl_correlation(r['daily_pnl'], l8_daily)
        print(f"    {r['label']}: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.0f}, "
              f"N={r['n']}, WR={r['win_rate']}%, L8_corr={corr}")
        r['l8_corr'] = corr

    best = results[0] if results else None
    kfold = None
    if best and best['sharpe'] > 0.5:
        sig_kw = {k: v for k, v in best['params'].items()}
        sig_kw['bt_kwargs'] = {'sl_mult': 2.0, 'tp_mult': 4.0, 'max_hold': 10,
                               'trail_act': 0.5, 'trail_dist': 0.1}
        kfold = kfold_test(h1_df, zscore_signals, sig_kw, "ZS_best")
        print(f"  K-Fold: {kfold['pass']}, mean={kfold['mean_sharpe']}, min={kfold['min_sharpe']}")

    save_json({'grid': [{k: v for k, v in r.items() if k != 'daily_pnl'} for r in results[:20]],
               'kfold': kfold}, 'S5_zscore.json')
    print(f"  S5 complete. {elapsed()}")
    return best


# ══════════════════════════════════════════════════════════════
# S6: Range Contraction Filter on L8_BASE
# ══════════════════════════════════════════════════════════════

def phase_s6(data, l8_daily):
    phase_header("S6", "Range Contraction Filter on L8_BASE")

    L8_BASE = {
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

    print("  Running L8_BASE baseline...")
    base_stats = run_variant(data, 'L8_BASE_ref', verbose=False, **L8_BASE)
    print(f"  Baseline: Sharpe={base_stats['sharpe']}, PnL=${base_stats['total_pnl']:.0f}, N={base_stats['n']}")

    base_trades = base_stats.get('_trades', [])
    if not base_trades:
        print("  No trades available for post-hoc filtering, skipping S6")
        save_json({'error': 'no trades'}, 'S6_range_contraction.json')
        return None

    rc_short = calc_range_contraction(data.h1_df, 7, 28)
    rc_values = {}
    for idx in rc_short.index:
        rc_values[idx] = rc_short.loc[idx] if not pd.isna(rc_short.loc[idx]) else 999

    results = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        filtered_trades = []
        for t in base_trades:
            entry_ts = t.entry_time
            h1_ts = entry_ts.floor('h') if hasattr(entry_ts, 'floor') else entry_ts
            rc_val = rc_values.get(h1_ts, 999)
            if rc_val < threshold:
                filtered_trades.append(t)

        if filtered_trades:
            from backtest.stats import calc_stats as official_stats
            equity = [0]
            for t in filtered_trades:
                equity.append(equity[-1] + t.pnl)
            stats = official_stats(filtered_trades, equity)
            stats['label'] = f"L8+RC_{threshold}"
            stats['rc_threshold'] = threshold
            stats['n_filtered'] = len(base_trades) - len(filtered_trades)
            results.append(stats)
            delta = stats['sharpe'] - base_stats['sharpe']
            print(f"    RC<{threshold}: Sharpe={stats['sharpe']:.2f} (delta={delta:+.2f}), "
                  f"PnL=${stats['total_pnl']:.0f}, N={stats['n']} "
                  f"(filtered out {stats['n_filtered']})")

    save_json({'baseline': {k: v for k, v in base_stats.items() if not k.startswith('_')},
               'rc_variants': [{k: v for k, v in r.items() if not k.startswith('_')}
                               for r in results]},
              'S6_range_contraction.json')
    print(f"  S6 complete. {elapsed()}")
    return results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_file = open(OUT_DIR / "00_master_log.txt", 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"R45 New Signal Sources — Started at {datetime.now()}")
    print(f"Output: {OUT_DIR}")

    print("\nLoading data...")
    t0 = time.time()
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"Data loaded in {time.time()-t0:.1f}s")
    print(f"  H1: {len(h1_df)} bars, {h1_df.index[0]} ~ {h1_df.index[-1]}")

    # L8_BASE reference for correlation
    print("\nRunning L8_BASE reference for correlation baseline...")
    L8_BASE = {
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
    l8_stats = run_variant(data, 'L8_BASE_corr_ref', verbose=False, **L8_BASE)
    l8_trades = l8_stats.get('_trades', [])
    l8_daily = {}
    for t in l8_trades:
        d = str(t.exit_time.date()) if hasattr(t.exit_time, 'date') else str(t.exit_time)[:10]
        l8_daily[d] = l8_daily.get(d, 0) + t.pnl
    print(f"  L8 ref: Sharpe={l8_stats['sharpe']}, PnL=${l8_stats['total_pnl']:.0f}, N={l8_stats['n']}")

    phases = [
        ("S1", phase_s1, (h1_df, l8_daily)),
        ("S2", phase_s2, (h1_df, l8_daily)),
        ("S3", phase_s3, (h1_df, l8_daily)),
        ("S4", phase_s4, (h1_df, l8_daily)),
        ("S5", phase_s5, (h1_df, l8_daily)),
        ("S6", phase_s6, (data, l8_daily)),
    ]

    completed = []
    for pname, pfunc, pargs in phases:
        try:
            t_phase = time.time()
            result = pfunc(*pargs)
            dt = time.time() - t_phase
            completed.append((pname, dt, result))
            print(f"\n  {pname} took {dt/60:.1f} min")
        except Exception as e:
            print(f"\n  {pname} FAILED: {e}")
            traceback.print_exc()
            completed.append((pname, -1, None))

    total_elapsed = time.time() - MARATHON_START
    print(f"\n\n{'='*70}")
    print(f"  R45 COMPLETE — {total_elapsed/60:.0f} minutes")
    print(f"{'='*70}")
    for pname, dt, result in completed:
        status = f"{dt/60:.1f} min" if dt > 0 else "FAILED"
        sharpe = result.get('sharpe', '?') if isinstance(result, dict) else '?'
        print(f"  {pname}: {status}, best Sharpe={sharpe}")

    print(f"\n  Results: {OUT_DIR}")
    log_file.close()
