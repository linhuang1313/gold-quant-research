"""R204: Keltner Trail M1-Resolution Replay
==========================================
Validates R202 winners (D: td=0.015, F: td=0.008) at M1 resolution.

Background:
  R202 ran Keltner backtest at M15 resolution. The winner-D trail params
  (ta=0.06/td=0.015) showed +0.65 Sharpe vs baseline. But M15 OHLC ordering
  bias may overstate tight-trail performance.

Method:
  1. Load full Keltner M15 backtest at baseline params -> get trade list
  2. For each trade, take entry_time/entry_price/entry_atr
  3. Replay exit using M1 data (15x higher resolution) under multiple
     trail param sets:
       - Baseline (ta=0.14, td=0.025)
       - D (ta=0.06, td=0.015)
       - F (ta=0.06, td=0.008)
  4. Compare:
       - Aggregate PnL per config
       - Trail trigger timing (which bar)
       - Hair-trigger rate (trail exit on same M15 bar as entry)
       - Sharpe estimate

Note:
  M1 is still bar-resolution (not tick), so a residual ordering bias
  remains. But it's 15x finer than M15, which is the main risk we want
  to quantify before deploying td=0.008.

Run: python experiments/run_r204_keltner_m1_replay.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtest.runner import (
    DataBundle, load_csv, LIVE_PARITY_KWARGS,
    M15_CSV_PATH, H1_CSV_PATH,
)
from backtest.engine import BacktestEngine


M1_CANDIDATES = [
    Path('data/download/xauusd-m1-bid-2015-01-01-2026-05-06.csv'),
    Path('data/download/xauusd-m1-bid-2015-01-01-2026-04-27.csv'),
    Path('data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv'),
]


def find_m1() -> Path:
    for p in M1_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f'No M1 CSV found in {M1_CANDIDATES}')


def load_m1_dukascopy(path: Path) -> pd.DataFrame:
    """Load Dukascopy-format M1 CSV.

    Header: 'Gmt time,Open,High,Low,Close,Volume'
    Time format: '01.01.2015 23:01:00.000' (UTC, day-first)
    Falls back to timestamp(ms) format used by backtest.runner.load_csv.
    """
    # Peek the header
    with open(path, 'r') as f:
        header = f.readline().strip()

    if 'timestamp' in header.lower():
        return load_csv(str(path))

    print(f'  Detected Dukascopy format: header="{header}"')
    df = pd.read_csv(path)
    if 'Gmt time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Gmt time'],
                                         format='%d.%m.%Y %H:%M:%S.%f',
                                         utc=True)
        df.drop(columns=['Gmt time'], inplace=True)
    elif 'GMT Time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['GMT Time'],
                                         format='%d.%m.%Y %H:%M:%S.%f',
                                         utc=True)
        df.drop(columns=['GMT Time'], inplace=True)
    else:
        raise ValueError(f'Unknown M1 format. Header: {df.columns.tolist()}')

    df.set_index('timestamp', inplace=True)
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # Drop "weekend gap" rows where OHLC are all equal (flat bars)
    df['is_flat'] = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])
    return df


# ----- Trail simulation -----
def simulate_trail_exit_m1(
    m1: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    direction: str,
    entry_atr: float,
    trail_act_atr: float,
    trail_dist_atr: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    max_hold_m1: int,
) -> dict:
    """Replay one position's exit using M1 bars.

    Returns dict with exit_price, exit_time, exit_reason, bars_held_m1.
    """
    sl_dist = sl_atr_mult * entry_atr
    tp_dist = tp_atr_mult * entry_atr
    act_dist = trail_act_atr * entry_atr
    trl_dist = trail_dist_atr * entry_atr

    is_buy = direction == 'BUY'
    sl_price = entry_price - sl_dist if is_buy else entry_price + sl_dist
    tp_price = entry_price + tp_dist if is_buy else entry_price - tp_dist

    # Take m1 bars FROM entry_time (inclusive) forward
    sub = m1[m1.index > entry_time]
    if len(sub) == 0:
        return {'exit_price': entry_price, 'exit_time': entry_time,
                'reason': 'no_data', 'bars_held_m1': 0, 'pnl': 0.0,
                'trail_activated': False, 'trail_activated_bar': -1}

    sub = sub.iloc[:max_hold_m1]
    high = sub['High'].values
    low = sub['Low'].values
    close = sub['Close'].values
    times = sub.index

    extreme = entry_price
    trail_price = 0.0
    trail_activated = False
    trail_activated_bar = -1

    for j in range(len(sub)):
        hi = high[j]
        lo = low[j]
        c = close[j]

        # SL/TP take priority (cross detection)
        if is_buy:
            if lo <= sl_price:
                return {'exit_price': sl_price, 'exit_time': times[j],
                        'reason': 'SL', 'bars_held_m1': j + 1,
                        'pnl': (sl_price - entry_price),
                        'trail_activated': trail_activated,
                        'trail_activated_bar': trail_activated_bar}
            if hi >= tp_price:
                return {'exit_price': tp_price, 'exit_time': times[j],
                        'reason': 'TP', 'bars_held_m1': j + 1,
                        'pnl': (tp_price - entry_price),
                        'trail_activated': trail_activated,
                        'trail_activated_bar': trail_activated_bar}
        else:
            if hi >= sl_price:
                return {'exit_price': sl_price, 'exit_time': times[j],
                        'reason': 'SL', 'bars_held_m1': j + 1,
                        'pnl': (entry_price - sl_price),
                        'trail_activated': trail_activated,
                        'trail_activated_bar': trail_activated_bar}
            if lo <= tp_price:
                return {'exit_price': tp_price, 'exit_time': times[j],
                        'reason': 'TP', 'bars_held_m1': j + 1,
                        'pnl': (entry_price - tp_price),
                        'trail_activated': trail_activated,
                        'trail_activated_bar': trail_activated_bar}

        # Update extreme
        if is_buy:
            extreme = max(extreme, hi)
            if extreme - entry_price >= act_dist:
                if not trail_activated:
                    trail_activated = True
                    trail_activated_bar = j + 1
                new_trail = extreme - trl_dist
                trail_price = max(trail_price, new_trail) if trail_price > 0 else new_trail
                if lo <= trail_price:
                    return {'exit_price': trail_price, 'exit_time': times[j],
                            'reason': 'Trail', 'bars_held_m1': j + 1,
                            'pnl': (trail_price - entry_price),
                            'trail_activated': True,
                            'trail_activated_bar': trail_activated_bar}
        else:
            extreme = min(extreme, lo)
            if entry_price - extreme >= act_dist:
                if not trail_activated:
                    trail_activated = True
                    trail_activated_bar = j + 1
                new_trail = extreme + trl_dist
                trail_price = min(trail_price, new_trail) if trail_price > 0 else new_trail
                if hi >= trail_price:
                    return {'exit_price': trail_price, 'exit_time': times[j],
                            'reason': 'Trail', 'bars_held_m1': j + 1,
                            'pnl': (entry_price - trail_price),
                            'trail_activated': True,
                            'trail_activated_bar': trail_activated_bar}

    # Timeout
    last_idx = min(len(sub) - 1, max_hold_m1 - 1)
    last_close = close[last_idx]
    pnl = (last_close - entry_price) if is_buy else (entry_price - last_close)
    return {'exit_price': last_close, 'exit_time': times[last_idx],
            'reason': 'Timeout', 'bars_held_m1': last_idx + 1,
            'pnl': pnl, 'trail_activated': trail_activated,
            'trail_activated_bar': trail_activated_bar}


# ----- Configs to compare -----
CONFIGS = {
    'baseline_M15': {'ta': 0.14, 'td': 0.025, 'sl_atr': 3.5, 'tp_atr': 8.0,
                     'max_hold_m1': 20 * 15},   # 20 M15 = 300 M1
    'D_M15':       {'ta': 0.06, 'td': 0.015, 'sl_atr': 3.5, 'tp_atr': 8.0,
                     'max_hold_m1': 20 * 15},
    'F_M15':       {'ta': 0.06, 'td': 0.008, 'sl_atr': 3.5, 'tp_atr': 8.0,
                     'max_hold_m1': 20 * 15},
}


def main():
    print('=' * 70)
    print('R204: Keltner Trail M1-Resolution Replay')
    print('=' * 70)

    print('\nStep 1: Load M15+H1 data, run baseline Keltner backtest...')
    data = DataBundle.load_default()
    print(f'  M15: {len(data.m15_df):,}  H1: {len(data.h1_df):,}')

    # Baseline backtest (current production params)
    kw = dict(LIVE_PARITY_KWARGS)
    engine = BacktestEngine(data.m15_df, data.h1_df, **kw)
    trades = engine.run()
    print(f'  Baseline trades: {len(trades)}')

    # Extract entry context for each trade
    entries = []
    for t in trades:
        entries.append({
            'entry_time': pd.Timestamp(t.entry_time, tz='UTC') if t.entry_time.tzinfo is None
                          else pd.Timestamp(t.entry_time),
            'entry_price': t.entry_price,
            'direction': t.direction,
            'm15_exit_price': t.exit_price,
            'm15_exit_time': pd.Timestamp(t.exit_time, tz='UTC') if t.exit_time.tzinfo is None
                             else pd.Timestamp(t.exit_time),
            'm15_pnl': t.pnl,
            'm15_reason': t.exit_reason,
            'm15_bars_held': t.bars_held,
        })

    # Compute entry ATR from M15 (snap to entry_time bar)
    print('  Computing entry ATRs...')
    m15 = data.m15_df.copy()
    if 'ATR' not in m15.columns:
        # Wilder ATR 14 on M15
        high = m15['High'].values
        low = m15['Low'].values
        close = m15['Close'].values
        tr = np.zeros(len(m15))
        for i in range(1, len(m15)):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        atr = pd.Series(tr, index=m15.index).rolling(14, min_periods=1).mean()
        m15['ATR'] = atr

    for e in entries:
        try:
            e['entry_atr'] = float(m15.loc[m15.index <= e['entry_time'], 'ATR'].iloc[-1])
        except Exception:
            e['entry_atr'] = float('nan')

    entries = [e for e in entries if not pd.isna(e['entry_atr']) and e['entry_atr'] > 0]
    print(f'  Trades with valid ATR: {len(entries)}')

    print('\nStep 2: Load M1 data...')
    m1_path = find_m1()
    print(f'  Path: {m1_path}')
    m1 = load_m1_dukascopy(m1_path)
    print(f'  M1 bars: {len(m1):,}  Range: {m1.index[0]} -> {m1.index[-1]}')

    # Filter entries to those within M1 coverage
    m1_start = m1.index[0]
    m1_end = m1.index[-1]
    entries = [e for e in entries if m1_start <= e['entry_time'] <= m1_end]
    print(f'  Trades in M1 range: {len(entries)}')

    print('\nStep 3: Replay each config at M1 resolution...')

    PV = 100  # Pip value -> per-unit PnL in $ for XAUUSD lot=1.0 / 100USD per unit
    LOT = 0.04  # standard test lot

    results = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f'\n  -- {cfg_name}: ta={cfg["ta"]} td={cfg["td"]} --')
        replay_trades = []
        n_total = len(entries)
        for k, e in enumerate(entries):
            if (k + 1) % 5000 == 0:
                print(f'    progress {k+1}/{n_total}')
            out = simulate_trail_exit_m1(
                m1, e['entry_time'], e['entry_price'], e['direction'],
                e['entry_atr'],
                trail_act_atr=cfg['ta'], trail_dist_atr=cfg['td'],
                sl_atr_mult=cfg['sl_atr'], tp_atr_mult=cfg['tp_atr'],
                max_hold_m1=cfg['max_hold_m1'])
            out['pnl_usd'] = out['pnl'] * LOT * PV
            replay_trades.append(out)

        pnls = np.array([t['pnl_usd'] for t in replay_trades])
        wins = (pnls > 0).sum()
        reasons = pd.Series([t['reason'] for t in replay_trades]).value_counts().to_dict()
        sharpe = pnls.mean() / max(pnls.std(), 1e-9) * np.sqrt(252 * 6)  # approx

        # Hair trigger: trail exited within 15 M1 bars of entry (same M15 bar)
        trail_trades = [t for t in replay_trades if t['reason'] == 'Trail']
        hair_15 = sum(1 for t in trail_trades if t['bars_held_m1'] <= 15)
        hair_30 = sum(1 for t in trail_trades if t['bars_held_m1'] <= 30)

        # Trail-activation bar distribution
        activated = [t['trail_activated_bar'] for t in trail_trades
                     if t['trail_activated_bar'] > 0]

        results[cfg_name] = {
            'n_trades': len(replay_trades),
            'pnl_total': round(float(pnls.sum()), 2),
            'pnl_mean': round(float(pnls.mean()), 4),
            'pnl_std': round(float(pnls.std()), 4),
            'win_rate_pct': round(100 * wins / max(len(pnls), 1), 2),
            'sharpe_approx': round(float(sharpe), 3),
            'max_dd_est': round(float((np.minimum.accumulate(np.cumsum(pnls)) - np.cumsum(pnls)).min()), 2),
            'exit_reasons': reasons,
            'trail_trades': len(trail_trades),
            'hair_trigger_le15_m1': hair_15,
            'hair_trigger_le15_pct': round(100 * hair_15 / max(len(trail_trades), 1), 2),
            'hair_trigger_le30_m1': hair_30,
            'hair_trigger_le30_pct': round(100 * hair_30 / max(len(trail_trades), 1), 2),
            'trail_act_bar_median': float(np.median(activated)) if activated else 0,
            'trail_act_bar_p25': float(np.percentile(activated, 25)) if activated else 0,
            'trail_act_bar_p75': float(np.percentile(activated, 75)) if activated else 0,
        }

        r = results[cfg_name]
        print(f'    PnL total:      ${r["pnl_total"]:.2f}')
        print(f'    Mean per trade: ${r["pnl_mean"]:.4f}')
        print(f'    Win rate:       {r["win_rate_pct"]:.2f}%')
        print(f'    Sharpe (approx):{r["sharpe_approx"]}')
        print(f'    Trail trades:   {r["trail_trades"]}')
        print(f'    Hair-trigger <=15 M1 bars (<=1 M15): {r["hair_trigger_le15_pct"]}%')
        print(f'    Hair-trigger <=30 M1 bars (<=2 M15): {r["hair_trigger_le30_pct"]}%')
        print(f'    Exit reasons:   {r["exit_reasons"]}')

    # Comparison table
    print('\n' + '=' * 70)
    print('COMPARISON (M1 replay vs M15 baseline)')
    print('=' * 70)
    print(f'{"Config":<14} {"PnL Total":>12} {"Win%":>7} {"Sharpe":>8} '
          f'{"TrailN":>8} {"Hair15%":>9} {"Hair30%":>9}')
    for n, r in results.items():
        print(f'{n:<14} {r["pnl_total"]:>12.2f} {r["win_rate_pct"]:>7.2f} '
              f'{r["sharpe_approx"]:>8.3f} {r["trail_trades"]:>8} '
              f'{r["hair_trigger_le15_pct"]:>9.2f} {r["hair_trigger_le30_pct"]:>9.2f}')

    print('\nDeltas vs baseline_M15:')
    base = results['baseline_M15']
    for n, r in results.items():
        if n == 'baseline_M15':
            continue
        d_pnl = r['pnl_total'] - base['pnl_total']
        d_sharpe = r['sharpe_approx'] - base['sharpe_approx']
        d_winr = r['win_rate_pct'] - base['win_rate_pct']
        print(f'  {n}: ΔPnL ${d_pnl:+.2f}  ΔSharpe {d_sharpe:+.3f}  ΔWinRate {d_winr:+.2f}%')

    # Save
    out = Path('results/r204_keltner_m1_replay')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'R204_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Results saved to: {out / "R204_results.json"}')


if __name__ == '__main__':
    main()
