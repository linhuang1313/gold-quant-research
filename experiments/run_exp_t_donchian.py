# -*- coding: utf-8 -*-
"""
Experiment T: Donchian Channel breakout (M15) — post-hoc signal study vs Keltner baseline.
"""
import sys
import io
import time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

import research_config as config
import indicators as signals_mod
from indicators import check_keltner_signal
from backtest.engine import TradeRecord
from backtest.runner import (
    DataBundle,
    LIVE_PARITY_KWARGS,
    calc_stats,
    run_variant,
    run_kfold,
)

# ── Constants ─────────────────────────────────────────────────
DC_PERIODS = [20, 40, 60, 80, 100]
HOLD_BARS = [4, 8, 12, 20]
SPREAD_PER_TRADE = 0.30
LOTS = 0.01
M15_LOOKBACK = 150
KELTNER_ADX = int(LIVE_PARITY_KWARGS.get('keltner_adx_threshold', 18))
OVERLAP_BARS = 2


def section(title: str):
    print('\n' + '=' * 72)
    print(f'  {title}')
    print('=' * 72)


from backtest.engine import BacktestEngine  # noqa: E402


def h1_closed_window(m15_time: pd.Timestamp, h1_df: pd.DataFrame, h1_lookup: dict) -> pd.DataFrame | None:
    """Mirror BacktestEngine._get_h1_window(..., closed_only=True)."""
    h1_time = m15_time.floor('h')
    if h1_time in h1_lookup:
        h1_idx = h1_lookup[h1_time]
    else:
        h1_times = h1_df.index
        mask = h1_times <= m15_time
        if not mask.any():
            return None
        h1_idx = int(mask.sum()) - 1
    h1_len = len(h1_df)
    if h1_idx >= h1_len:
        h1_idx = h1_len - 1
    h1_idx -= 1
    if h1_idx < 0:
        return None
    start = max(0, h1_idx - BacktestEngine.H1_WINDOW + 1)
    return h1_df.iloc[start : h1_idx + 1]


def collect_keltner_signal_bar_indices(m15_df: pd.DataFrame, h1_df: pd.DataFrame) -> list[int]:
    """
    Raw H1 Keltner signals (check_keltner_signal on last closed H1 bar), aligned to M15 bar index
    at hour boundaries — same ADX threshold as LIVE_PARITY, no intraday choppy gate.
    """
    h1_lookup = BacktestEngine._build_h1_lookup(h1_df)
    old_adx = signals_mod.ADX_TREND_THRESHOLD
    signals_mod.ADX_TREND_THRESHOLD = KELTNER_ADX
    out: list[int] = []
    try:
        for i in range(M15_LOOKBACK, len(m15_df)):
            ts = m15_df.index[i]
            if ts.minute != 0:
                continue
            hw = h1_closed_window(ts, h1_df, h1_lookup)
            if hw is None or len(hw) < 105:
                continue
            sig = check_keltner_signal(hw)
            if sig:
                out.append(i)
    finally:
        signals_mod.ADX_TREND_THRESHOLD = old_adx
    return out


def donchian_pnl(
    direction: str,
    entry_price: float,
    exit_price: float,
    lots: float,
    spread: float,
) -> float:
    if direction == 'BUY':
        pts = exit_price - entry_price
    else:
        pts = entry_price - exit_price
    gross = pts * lots * config.POINT_VALUE_PER_LOT
    cost = spread * lots * config.POINT_VALUE_PER_LOT
    return round(gross - cost, 2)


def simulate_donchian_trades(
    m15_df: pd.DataFrame,
    period: int,
    n_hold: int,
) -> list[TradeRecord]:
    close = m15_df['Close'].astype(float)
    ema100 = m15_df['EMA100'].astype(float)
    is_flat = m15_df.get('is_flat', pd.Series(False, index=m15_df.index))

    dc_high_prev = close.rolling(period).max().shift(1)
    dc_low_prev = close.rolling(period).min().shift(1)

    trades: list[TradeRecord] = []
    warmup = max(period + 2, M15_LOOKBACK)
    n = len(m15_df)

    for i in range(warmup, n - 1 - n_hold):
        if bool(is_flat.iloc[i]):
            continue
        ph = dc_high_prev.iloc[i]
        pl = dc_low_prev.iloc[i]
        c = close.iloc[i]
        ema = ema100.iloc[i]
        if pd.isna(ph) or pd.isna(pl) or pd.isna(ema):
            continue

        direction: str | None = None
        if c > ph and c > ema:
            direction = 'BUY'
        elif c < pl and c < ema:
            direction = 'SELL'
        else:
            continue

        ent_i = i + 1
        ex_i = i + 1 + n_hold
        if ex_i >= n:
            break
        entry_open = float(m15_df['Open'].iloc[ent_i])
        exit_close = float(m15_df['Close'].iloc[ex_i])
        entry_time = m15_df.index[ent_i].to_pydatetime()
        exit_time = m15_df.index[ex_i].to_pydatetime()

        pnl = donchian_pnl(direction, entry_open, exit_close, LOTS, SPREAD_PER_TRADE)
        trades.append(
            TradeRecord(
                strategy='donchian',
                direction=direction,
                entry_price=entry_open,
                exit_price=exit_close,
                entry_time=entry_time,
                exit_time=exit_time,
                lots=LOTS,
                pnl=pnl,
                exit_reason='fixed_hold',
                bars_held=n_hold,
            )
        )
    return trades


def equity_from_trades(trades: list[TradeRecord]) -> list[float]:
    eq = [float(config.CAPITAL)]
    for t in trades:
        eq.append(eq[-1] + t.pnl)
    return eq


def daily_pnl_series(trades: list[TradeRecord]) -> pd.Series:
    d: dict = {}
    for t in trades:
        day = pd.Timestamp(t.exit_time).date()
        d[day] = d.get(day, 0.0) + t.pnl
    return pd.Series(d).sort_index()


def overlap_stats(
    donchian_signal_indices: list[int],
    keltner_ixs: set[int],
    window: int = OVERLAP_BARS,
) -> tuple[float, int]:
    if not donchian_signal_indices:
        return 0.0, 0
    hits = 0
    for ix in donchian_signal_indices:
        for k in range(ix - window, ix + window + 1):
            if k in keltner_ixs:
                hits += 1
                break
    return hits / len(donchian_signal_indices), hits


def collect_donchian_signal_indices(m15_df: pd.DataFrame, period: int) -> list[int]:
    close = m15_df['Close'].astype(float)
    ema100 = m15_df['EMA100'].astype(float)
    is_flat = m15_df.get('is_flat', pd.Series(False, index=m15_df.index))
    dc_high_prev = close.rolling(period).max().shift(1)
    dc_low_prev = close.rolling(period).min().shift(1)
    warmup = max(period + 2, M15_LOOKBACK)
    n = len(m15_df)
    sigs: list[int] = []
    for i in range(warmup, n - 1):
        if bool(is_flat.iloc[i]):
            continue
        ph = dc_high_prev.iloc[i]
        pl = dc_low_prev.iloc[i]
        c = close.iloc[i]
        ema = ema100.iloc[i]
        if pd.isna(ph) or pd.isna(pl) or pd.isna(ema):
            continue
        if (c > ph and c > ema) or (c < pl and c < ema):
            sigs.append(i)
    return sigs


def run_donchian_on_bundle(data: DataBundle, period: int, n_hold: int, keltner_ixs: set[int]) -> dict:
    trades = simulate_donchian_trades(data.m15_df, period, n_hold)
    eq = equity_from_trades(trades)
    stats = calc_stats(trades, eq)
    stats['period'] = period
    stats['n_hold'] = n_hold
    stats['_trades'] = trades

    sig_ixs = collect_donchian_signal_indices(data.m15_df, period)
    overlap_pct, overlap_n = overlap_stats(sig_ixs, keltner_ixs)
    stats['donchian_signal_count'] = len(sig_ixs)
    stats['overlap_pct'] = overlap_pct * 100
    stats['overlap_n'] = overlap_n

    kset = keltner_ixs
    non_overlap_trades = []
    # Map: signal bar index = entry index - 1 for our sim (signal at i, entry i+1)
    warmup = max(period + 2, M15_LOOKBACK)
    close = data.m15_df['Close'].astype(float)
    ema100 = data.m15_df['EMA100'].astype(float)
    is_flat = data.m15_df.get('is_flat', pd.Series(False, index=data.m15_df.index))
    dc_high_prev = close.rolling(period).max().shift(1)
    dc_low_prev = close.rolling(period).min().shift(1)
    n = len(data.m15_df)
    for i in range(warmup, n - 1 - n_hold):
        if bool(is_flat.iloc[i]):
            continue
        ph = dc_high_prev.iloc[i]
        pl = dc_low_prev.iloc[i]
        c = close.iloc[i]
        ema = ema100.iloc[i]
        if pd.isna(ph) or pd.isna(pl) or pd.isna(ema):
            continue
        direction = None
        if c > ph and c > ema:
            direction = 'BUY'
        elif c < pl and c < ema:
            direction = 'SELL'
        else:
            continue
        near_k = any((i + d) in kset for d in range(-OVERLAP_BARS, OVERLAP_BARS + 1))
        if near_k:
            continue
        ent_i = i + 1
        ex_i = i + 1 + n_hold
        entry_open = float(data.m15_df['Open'].iloc[ent_i])
        exit_close = float(data.m15_df['Close'].iloc[ex_i])
        pnl = donchian_pnl(direction, entry_open, exit_close, LOTS, SPREAD_PER_TRADE)
        non_overlap_trades.append(
            TradeRecord(
                strategy='donchian_no_k_overlap',
                direction=direction,
                entry_price=entry_open,
                exit_price=exit_close,
                entry_time=data.m15_df.index[ent_i].to_pydatetime(),
                exit_time=data.m15_df.index[ex_i].to_pydatetime(),
                lots=LOTS,
                pnl=pnl,
                exit_reason='fixed_hold',
                bars_held=n_hold,
            )
        )
    stats['non_overlap_n'] = len(non_overlap_trades)
    stats['non_overlap_pnl'] = sum(t.pnl for t in non_overlap_trades)
    no_eq = equity_from_trades(non_overlap_trades)
    no_st = calc_stats(non_overlap_trades, no_eq)
    stats['non_overlap_sharpe'] = no_st['sharpe']
    return stats


def donchian_kfold(data: DataBundle, period: int, n_hold: int) -> list[dict]:
    """6-fold time-series CV on sliced bundles (Donchian sim only)."""
    folds = [
        ('Fold1', '2015-01-01', '2017-01-01'),
        ('Fold2', '2017-01-01', '2019-01-01'),
        ('Fold3', '2019-01-01', '2021-01-01'),
        ('Fold4', '2021-01-01', '2023-01-01'),
        ('Fold5', '2023-01-01', '2025-01-01'),
        ('Fold6', '2025-01-01', '2026-04-01'),
    ]
    results = []
    for fold_name, start, end in folds:
        sl = data.slice(start, end)
        if len(sl.m15_df) < 1000:
            continue
        # Keltner indices local to slice: recompute on sliced M15
        local_k = set(collect_keltner_signal_bar_indices(sl.m15_df, sl.h1_df))
        st = run_donchian_on_bundle(sl, period, n_hold, local_k)
        st['fold'] = fold_name
        st['test_start'] = start
        st['test_end'] = end
        results.append(st)
    return results


def main():
    t0 = time.time()
    section('Load DataBundle.load_default()')
    data = DataBundle.load_default()

    section('Collect raw H1 Keltner signal bar indices (M15, hour boundaries)')
    k_ix_list = collect_keltner_signal_bar_indices(data.m15_df, data.h1_df)
    k_ix_set = set(k_ix_list)
    print(f'  Keltner raw signal bars: {len(k_ix_list)} (ADX>={KELTNER_ADX}, KC+EMA100, closed H1)')

    section('Keltner / LIVE_PARITY full-engine baseline')
    base_kw = {**LIVE_PARITY_KWARGS, 'spread_cost': SPREAD_PER_TRADE, 'spread_model': 'fixed'}
    base = run_variant(data, 'LIVE_PARITY_baseline', verbose=True, **base_kw)
    k_trades = [t for t in base['_trades'] if t.strategy == 'keltner']
    k_eq = equity_from_trades(k_trades)
    k_stats = calc_stats(k_trades, k_eq)
    print(
        f"  Keltner-only (from baseline run): n={k_stats['n']}, "
        f"Sharpe={k_stats['sharpe']:.3f}, PnL=${k_stats['total_pnl']:.0f}"
    )

    section('Donchian grid: period × hold (fixed spread ${:.2f}, {:.2f} lot)'.format(SPREAD_PER_TRADE, LOTS))
    grid_rows: list[dict] = []
    for period in DC_PERIODS:
        for h in HOLD_BARS:
            st = run_donchian_on_bundle(data, period, h, k_ix_set)
            grid_rows.append(
                {
                    'period': period,
                    'hold': h,
                    'n': st['n'],
                    'sharpe': st['sharpe'],
                    'total_pnl': st['total_pnl'],
                    'overlap_pct': st['overlap_pct'],
                    'non_overlap_pnl': st['non_overlap_pnl'],
                    'non_overlap_sharpe': st['non_overlap_sharpe'],
                }
            )
            print(
                f"  DC N={period:3d} hold={h:2d} | trades={st['n']:5d} Sharpe={st['sharpe']:6.3f} "
                f"PnL=${st['total_pnl']:8.0f} | overlap={st['overlap_pct']:5.1f}% "
                f"| no-overlap PnL=${st['non_overlap_pnl']:.0f} Sharpe={st['non_overlap_sharpe']:.3f}"
            )

    section('Overlap summary (Donchian signal vs Keltner ±{} bars)'.format(OVERLAP_BARS))
    for period in DC_PERIODS:
        sigs = collect_donchian_signal_indices(data.m15_df, period)
        op, on = overlap_stats(sigs, k_ix_set)
        print(f'  Period {period:3d}: Donchian signals={len(sigs):6d}, overlap={op*100:5.1f}% ({on} hits)')

    section('Daily PnL correlation: Donchian vs Keltner')
    best_row = max(grid_rows, key=lambda r: r['sharpe'])
    best_p, best_h = best_row['period'], best_row['hold']
    best_st = run_donchian_on_bundle(data, best_p, best_h, k_ix_set)
    dc_trades = best_st['_trades']
    s_dc = daily_pnl_series(dc_trades)
    s_k = daily_pnl_series(k_trades)
    al_dc, al_k = s_dc.align(s_k, join='outer', fill_value=0.0)
    if len(al_dc) > 2 and al_dc.std(ddof=1) > 0 and al_k.std(ddof=1) > 0:
        corr = float(al_dc.corr(al_k))
    else:
        corr = float('nan')
    print(f'  Best grid Sharpe combo (full sample): period={best_p}, hold={best_h}, Sharpe={best_row["sharpe"]:.3f}')
    print(f'  Pearson corr(daily PnL Donchian, daily PnL Keltner): {corr:.4f}')

    section('K-Fold (6-fold) for Sharpe>0 configs')
    positive = [r for r in grid_rows if r['sharpe'] > 0]
    if not positive:
        print('  No period/hold with positive full-sample Sharpe — skipping K-fold.')
    else:
        positive.sort(key=lambda r: -r['sharpe'])
        for r in positive[:5]:
            print(
                f"\n  --- K-fold: DC period={r['period']} hold={r['hold']} (full Sharpe={r['sharpe']:.3f}) ---"
            )
            folds = donchian_kfold(data, r['period'], r['hold'])
            sharps = [x['sharpe'] for x in folds]
            pnls = [x['total_pnl'] for x in folds]
            if sharps:
                print(
                    f"     folds={len(folds)}  mean Sharpe={np.mean(sharps):.3f}  "
                    f"mean PnL=${np.mean(pnls):.0f}  min Sharpe={np.min(sharps):.3f}"
                )
                for x in folds:
                    print(
                        f"       {x['fold']}: Sharpe={x['sharpe']:.3f} PnL=${x['total_pnl']:.0f} "
                        f"n={x['n']}"
                    )

    section('SUMMARY TABLE — Donchian grid')
    print(f"{'period':>6} {'hold':>4} {'n':>6} {'Sharpe':>8} {'PnL':>10} {'Ovlp%':>7} {'NO-PnL':>10} {'NO-Shrp':>8}")
    print('-' * 72)
    for r in sorted(grid_rows, key=lambda x: (-x['sharpe'], -x['total_pnl'])):
        print(
            f"{r['period']:6d} {r['hold']:4d} {r['n']:6d} {r['sharpe']:8.3f} {r['total_pnl']:10.0f} "
            f"{r['overlap_pct']:7.1f} {r['non_overlap_pnl']:10.0f} {r['non_overlap_sharpe']:8.3f}"
        )

    print('\n  Baseline LIVE_PARITY: Sharpe={:.3f} PnL=${:.0f} (all strategies)'.format(base['sharpe'], base['total_pnl']))
    print(
        f"  Keltner-only baseline: Sharpe={k_stats['sharpe']:.3f} PnL=${k_stats['total_pnl']:.0f} "
        f"n={k_stats['n']}"
    )
    print(f"  Daily PnL corr (best Donchian vs Keltner): {corr:.4f}")
    print(f"\n  Done in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
