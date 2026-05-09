#!/usr/bin/env python3
"""
R95 — L8_MAX (Keltner) Multi-Position Exploration
===================================================
Tests different max_positions (1, 2, 3) with various add-position conditions.

Variants:
  Baseline: max_positions=1 (current live)
  A: max_positions=2, ADX>=28, profit>=1.0*ATR (current _can_add_position logic)
  B: max_positions=2, ADX>=25, profit>=0.5*ATR (relaxed)
  C: max_positions=3, ADX>=30, profit>=1.5*ATR (strict)
  D: max_positions=2, same direction only, min distance 0.5*ATR

ML Exit filter applied to ALL entries (including add-positions).
K-Fold validation on best variants.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r95_keltner_multipos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01
CAPITAL = 5000

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).tz_localize(None).normalize() if hasattr(pd.Timestamp(t['exit_time']), 'tz_localize') else pd.Timestamp(t['exit_time']).normalize()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))

def sharpe(daily_pnl):
    if len(daily_pnl) < 10: return 0.0
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    if arr.std() == 0: return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))

def max_dd(daily_pnl):
    arr = np.array(daily_pnl) if not isinstance(daily_pnl, np.ndarray) else daily_pnl
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) > 0 else 0.0


def run_l8_multipos(bundle, max_positions, adx_min, profit_min_atr, min_distance_atr,
                    same_direction_only=True, spread=SPREAD, lot=UNIT_LOT, maxloss_cap=35):
    """
    Run L8_MAX with multi-position logic.
    
    First position: normal L8_MAX entry.
    Additional positions: require ADX >= adx_min, existing position profit >= profit_min_atr * ATR,
                          and min price distance from last entry.
    """
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])

    if max_positions <= 1:
        trades = []
        for t in raw_trades:
            trades.append({
                'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'pnl': t.pnl, 'reason': t.exit_reason,
            })
        return trades

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    h1_df['ATR'] = compute_atr(h1_df)
    h1_df['ADX'] = compute_adx(h1_df)

    all_entries = []
    for t in raw_trades:
        all_entries.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason,
        })

    base_trades = list(all_entries)
    add_trades = []

    open_positions = []
    for trade in all_entries:
        entry_t = pd.Timestamp(trade['entry_time'])
        exit_t = pd.Timestamp(trade['exit_time'])

        expired = [p for p in open_positions if pd.Timestamp(p['exit_time']) <= entry_t]
        for p in expired:
            open_positions.remove(p)

        if len(open_positions) >= max_positions:
            open_positions.append(trade)
            continue

        if len(open_positions) == 0:
            open_positions.append(trade)
            continue

        if entry_t.tz is not None:
            entry_naive = entry_t.tz_localize(None)
        else:
            entry_naive = entry_t

        h1_idx = h1_df.index
        if h1_idx.tz is not None:
            h1_idx_naive = h1_idx.tz_localize(None)
        else:
            h1_idx_naive = h1_idx

        mask = h1_idx_naive <= entry_naive
        if mask.sum() == 0:
            open_positions.append(trade)
            continue

        bar_loc = mask.sum() - 1
        current_atr = h1_df['ATR'].iloc[bar_loc]
        current_adx = h1_df['ADX'].iloc[bar_loc]

        if pd.isna(current_adx) or current_adx < adx_min:
            open_positions.append(trade)
            continue

        last_pos = open_positions[-1]
        last_entry_price = last_pos['entry']
        current_entry_price = trade['entry']

        if same_direction_only and trade['dir'] != last_pos['dir']:
            open_positions.append(trade)
            continue

        distance = abs(current_entry_price - last_entry_price)
        min_dist = min_distance_atr * current_atr if not pd.isna(current_atr) else 999
        if distance < min_dist:
            open_positions.append(trade)
            continue

        current_price = current_entry_price
        if last_pos['dir'] == 'BUY':
            float_profit = current_price - last_entry_price
        else:
            float_profit = last_entry_price - current_price

        profit_threshold = profit_min_atr * current_atr if not pd.isna(current_atr) else 999
        if float_profit < profit_threshold:
            open_positions.append(trade)
            continue

        add_trades.append(trade)
        open_positions.append(trade)

    combined = base_trades + add_trades
    return combined


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  R95 — L8_MAX Multi-Position Exploration", flush=True)
    print("=" * 80, flush=True)

    from backtest.runner import DataBundle
    print("\n  Loading data...", flush=True)
    bundle = DataBundle.load_custom()

    variants = {
        'Baseline': {'max_positions': 1, 'adx_min': 0, 'profit_min_atr': 0, 'min_distance_atr': 0, 'same_direction_only': False},
        'A_ADX28_Profit1.0': {'max_positions': 2, 'adx_min': 28, 'profit_min_atr': 1.0, 'min_distance_atr': 0.5, 'same_direction_only': True},
        'B_ADX25_Profit0.5': {'max_positions': 2, 'adx_min': 25, 'profit_min_atr': 0.5, 'min_distance_atr': 0.5, 'same_direction_only': True},
        'C_ADX30_Profit1.5_3pos': {'max_positions': 3, 'adx_min': 30, 'profit_min_atr': 1.5, 'min_distance_atr': 0.5, 'same_direction_only': True},
        'D_ADX25_Dist0.5_2pos': {'max_positions': 2, 'adx_min': 25, 'profit_min_atr': 0.3, 'min_distance_atr': 0.5, 'same_direction_only': True},
    }

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Run all variants on full data
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: Full-sample comparison", flush=True)
    print("=" * 70, flush=True)

    results = {}
    variant_trades = {}

    for name, params in variants.items():
        print(f"\n    Running {name}...", flush=True)
        trades = run_l8_multipos(bundle, **params)
        ds = trades_to_daily_series(trades)
        sh = sharpe(ds)
        dd = max_dd(ds)
        pnl = sum(t['pnl'] for t in trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0
        avg_pnl = pnl / len(trades) if trades else 0

        results[name] = {
            'params': params,
            'n_trades': len(trades),
            'sharpe': round(sh, 3),
            'max_dd': round(dd, 2),
            'pnl': round(pnl, 2),
            'wr': round(wr, 1),
            'avg_pnl': round(avg_pnl, 2),
        }
        variant_trades[name] = trades
        print(f"      {len(trades)} trades, Sharpe={sh:.3f}, MaxDD=${dd:.1f}, WR={wr:.1f}%, Avg=${avg_pnl:.2f}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Phase 2: K-Fold Validation on variants that beat baseline
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: K-Fold Validation", flush=True)
    print("=" * 70, flush=True)

    baseline_sharpe = results['Baseline']['sharpe']
    candidates = [name for name, r in results.items()
                  if r['sharpe'] > baseline_sharpe and name != 'Baseline']
    candidates.append('Baseline')

    kfold_results = {}
    for name in candidates:
        params = variants[name]
        fold_sharpes = []
        print(f"\n    K-Fold: {name}", flush=True)

        for fold_name, start, end in FOLDS:
            fold_bundle = bundle.slice(start, end) if hasattr(bundle, 'slice') else bundle
            try:
                trades = run_l8_multipos(fold_bundle, **params, maxloss_cap=35)
                ds = trades_to_daily_series(trades)
                fs = sharpe(ds)
            except Exception:
                fs = 0.0
            fold_sharpes.append(fs)

        positive = sum(1 for s in fold_sharpes if s > 0)
        kfold_results[name] = {
            'fold_sharpes': [round(s, 2) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(np.mean(fold_sharpes), 3),
            'pass_4of6': positive >= 4,
        }
        print(f"      {positive}/6 positive, mean={np.mean(fold_sharpes):.3f}, "
              f"folds={[f'{s:.2f}' for s in fold_sharpes]}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    output = {
        'experiment': 'R95 L8_MAX Multi-Position Exploration',
        'elapsed_s': round(elapsed, 1),
        'full_sample': results,
        'kfold': kfold_results,
        'recommendation': '',
    }

    best_kfold = max(kfold_results.items(), key=lambda x: x[1]['mean_sharpe'])
    if best_kfold[0] != 'Baseline' and best_kfold[1]['pass_4of6']:
        output['recommendation'] = f"Upgrade to {best_kfold[0]} (mean Sharpe {best_kfold[1]['mean_sharpe']:.3f} vs Baseline)"
    else:
        output['recommendation'] = "Keep Baseline (max_positions=1) - no significant improvement from multi-position"

    print(f"\n{'='*80}", flush=True)
    print(f"  R95 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Recommendation: {output['recommendation']}", flush=True)
    print(f"{'='*80}", flush=True)

    with open(OUTPUT_DIR / "r95_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR}/r95_results.json", flush=True)


if __name__ == "__main__":
    main()
