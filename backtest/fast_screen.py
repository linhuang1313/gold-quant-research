"""
Fast Screening Backtester
==========================
Lightweight, NumPy-only signal backtester for rapid parameter scanning.

~10-20x faster than BacktestEngine because it:
  - Operates on a single timeframe (H1) with pre-vectorized signals
  - Uses raw NumPy arrays instead of DataFrame row access
  - No M15 sub-bar resolution, no multi-position logic
  - No regime gating, no intraday scoring

Usage pattern (two-tier screening):
  1. Generate many param combos → run through fast_backtest_signals()
  2. Rank by Sharpe → take top-K
  3. Run top-K through full BacktestEngine for final validation

This module is extracted and generalized from the R45/R47 backtest_signals().
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


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


def fast_backtest_signals(
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
    """Run a fast single-pass backtest on pre-computed signal series.

    Args:
        df: OHLC DataFrame (H1 or M15) with DatetimeIndex
        signals: Series aligned with df; +1=BUY, -1=SELL, 0=no signal
        atr: ATR Series aligned with df
        sl_mult: stop-loss distance in ATR multiples
        tp_mult: take-profit distance in ATR multiples
        max_hold: max bars to hold before timeout exit
        trail_act: trailing stop activation threshold (ATR multiples of float profit)
        trail_dist: trailing stop distance (ATR multiples)
        spread_cost: per-trade transaction cost in price units
        min_gap_bars: minimum bars between entries

    Returns:
        List of SimpleTrade results
    """
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
    """Compute summary statistics from a list of SimpleTrade."""
    if not trades:
        return {'label': label, 'n': 0, 'total_pnl': 0, 'sharpe': 0,
                'win_rate': 0, 'max_dd': 0, 'avg_pnl': 0, 'daily_pnl': {}}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    daily_pnl: Dict[str, float] = {}
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
    exit_reasons: Dict[str, int] = {}
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
    """Compute annualized Pearson correlation between two daily PnL dicts."""
    all_dates = sorted(set(daily_a.keys()) | set(daily_b.keys()))
    a = [daily_a.get(d, 0) for d in all_dates]
    b = [daily_b.get(d, 0) for d in all_dates]
    if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def combine_daily_pnl(*daily_dicts) -> Dict[str, float]:
    """Merge multiple daily PnL dicts by summing same-date values."""
    combined: Dict[str, float] = {}
    for d in daily_dicts:
        for date, pnl in d.items():
            combined[date] = combined.get(date, 0) + pnl
    return combined


def stats_from_daily(daily_pnl: Dict[str, float], label: str = "") -> Dict:
    """Compute Sharpe, PnL, MaxDD from a daily PnL dict."""
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
    monthly: Dict[str, float] = defaultdict(float)
    for d, p in daily_pnl.items():
        monthly[d[:7]] += p
    neg_months = sum(1 for v in monthly.values() if v < 0)
    return {
        'label': label, 'sharpe': round(sharpe, 2), 'total_pnl': round(total, 2),
        'max_dd': round(max_dd, 2), 'n_days': len(dates), 'neg_months': neg_months,
    }


def screen_grid(
    df: pd.DataFrame,
    signal_func,
    param_grid: List[Dict],
    bt_defaults: Optional[Dict] = None,
    top_k: int = 3,
    rank_by: str = 'sharpe',
    min_sharpe: Optional[float] = None,
    label_prefix: str = "",
    verbose: bool = True,
) -> List[Dict]:
    """Screen a grid of signal+backtest parameter combos.

    Two modes:
      - Eliminate mode (min_sharpe is set): return ALL combos with sharpe >= min_sharpe,
        sorted by rank_by. Use this to avoid accidentally discarding good strategies.
      - Top-K mode (min_sharpe is None): return only the top_k results by rank_by.

    Args:
        df: OHLC DataFrame (H1) with indicators
        signal_func: callable(df, **sig_params) -> (signals_series, atr_series)
        param_grid: list of dicts, each with 'sig_params' and optionally 'bt_params'
            - sig_params: kwargs for signal_func
            - bt_params: kwargs overrides for fast_backtest_signals
        bt_defaults: default bt params (sl_mult, tp_mult, etc.)
        top_k: number of top results to return (ignored in eliminate mode)
        rank_by: metric to rank by ('sharpe', 'total_pnl', 'win_rate')
        min_sharpe: if set, return all combos with sharpe >= this threshold (eliminate mode)
        label_prefix: prefix for labels
        verbose: print progress

    Returns:
        List of dicts sorted by rank_by descending, each containing:
          - 'stats': the stats dict
          - 'sig_params': the signal params used
          - 'bt_params': the backtest params used
          - 'rank': 1-based rank
    """
    if bt_defaults is None:
        bt_defaults = {}

    results = []
    n_total = len(param_grid)
    import time
    t0 = time.time()

    for idx, combo in enumerate(param_grid):
        sig_params = combo.get('sig_params', {})
        bt_params = {**bt_defaults, **combo.get('bt_params', {})}
        lbl = combo.get('label', f"{label_prefix}C{idx}")

        signals, atr = signal_func(df, **sig_params)
        trades = fast_backtest_signals(df, signals, atr, **bt_params, label=lbl)
        stats = trades_to_stats(trades, lbl)

        results.append({
            'stats': stats,
            'sig_params': sig_params,
            'bt_params': bt_params,
            'label': lbl,
        })

        if verbose and (idx + 1) % max(1, n_total // 10) == 0:
            elapsed = time.time() - t0
            print(f"    Screen: {idx+1}/{n_total} ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: x['stats'].get(rank_by, 0), reverse=True)

    for i, r in enumerate(results):
        r['rank'] = i + 1

    # Eliminate mode: keep all with sharpe >= threshold
    if min_sharpe is not None:
        survivors = [r for r in results if r['stats'].get('sharpe', 0) >= min_sharpe]
        eliminated = len(results) - len(survivors)
        if verbose:
            elapsed = time.time() - t0
            print(f"  Screen complete: {n_total} combos in {elapsed:.1f}s", flush=True)
            print(f"  Eliminate mode (min_sharpe={min_sharpe}): "
                  f"{len(survivors)} survivors, {eliminated} eliminated "
                  f"({eliminated/n_total*100:.0f}% reduction)")
            if survivors:
                s = survivors[0]['stats']
                print(f"  Best: {survivors[0]['label']}: "
                      f"Sharpe={s['sharpe']:.2f}, PnL=${s['total_pnl']:.0f}")
        return survivors

    # Top-K mode
    if verbose:
        elapsed = time.time() - t0
        print(f"  Screen complete: {n_total} combos in {elapsed:.1f}s", flush=True)
        print(f"  Top {min(top_k, len(results))} by {rank_by}:")
        for r in results[:top_k]:
            s = r['stats']
            print(f"    #{r['rank']} {r['label']}: "
                  f"Sharpe={s['sharpe']:.2f}, PnL=${s['total_pnl']:.0f}, "
                  f"N={s['n']}, WR={s['win_rate']:.1f}%, MaxDD=${s['max_dd']:.0f}")

    return results[:top_k]


def kfold_screen(
    df: pd.DataFrame,
    signal_func,
    sig_params: Dict,
    bt_params: Dict,
    label: str = "",
    n_folds: int = 6,
) -> Dict:
    """Run K-Fold validation using fast screener on a single combo.

    Returns dict with per-fold stats, mean/min sharpe, pass count.
    """
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-05-01"),
    ][:n_folds]

    results = []
    for fname, start, end in folds:
        fold_df = df[start:end]
        if len(fold_df) < 200:
            continue
        signals, atr = signal_func(fold_df, **sig_params)
        trades = fast_backtest_signals(fold_df, signals, atr, **bt_params)
        stats = trades_to_stats(trades, f"{label}_{fname}")
        stats['fold'] = fname
        results.append(stats)

    sharpes = [r['sharpe'] for r in results]
    pass_count = sum(1 for s in sharpes if s > 0)
    return {
        'label': label,
        'folds': [{k: v for k, v in r.items() if k != 'daily_pnl'} for r in results],
        'sharpes': sharpes,
        'mean_sharpe': round(np.mean(sharpes), 2) if sharpes else 0,
        'min_sharpe': round(min(sharpes), 2) if sharpes else 0,
        'pass': f"{pass_count}/{len(results)}",
    }
