#!/usr/bin/env python3
"""
R158 — Order Flow / Volume Signal Research for Keltner (L8_MAX)
================================================================
We have Yahoo Finance H1 data with real GC futures volume (2023-12 ~ 2026-05).
Dukascopy H1 has no volume. This experiment evaluates whether volume-based
signals can improve Keltner strategy performance.

Signals tested:
  1. Volume Spike Confirmation  (Vol > N*MA20 at entry bar)
  2. VWAP deviation             (price vs rolling VWAP)
  3. OBV trend confirmation     (OBV slope agrees with direction)
  4. Volume Climax detection    (extreme volume + reversal candle)
  5. Pseudo CVD (delta proxy)   ((Close-Low)/(High-Low) cumulative)

Phases:
  1: Data alignment — merge YF volume with Dukascopy prices
  2: Signal calculation + IC analysis (forward return prediction)
  3: Keltner entry filter test (Volume Spike + OBV)
  4: Keltner exit signal test (Volume Climax + OBV Divergence)
  5: Independent VWAP reversion strategy
  6: K-Fold validation (3 folds, ~8 months each)
  7: Tick volume proxy test (bar range as volume substitute)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r158_volume_signals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()

PV = 100
SPREAD = 0.30
LOT = 0.01


def fmt(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    if s == 0: return 0.0
    return float(np.mean(arr) / s * np.sqrt(252))


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def trades_to_daily(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0}
    pnls = [t['pnl'] for t in trades]
    ds = trades_to_daily(trades)
    n = len(trades)
    return {
        'n': n, 'sharpe': round(sharpe(ds.values), 2),
        'pnl': round(sum(pnls), 2), 'max_dd': round(max_dd(ds.values), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'avg_pnl': round(np.mean(pnls), 2),
    }


def print_row(label, s):
    print(f"  {label:<55} n={s['n']:>5} Sh={s['sharpe']:>5.2f} PnL={fmt(s['pnl'])} "
          f"DD={fmt(s['max_dd'])} WR={s['wr']:.0f}% Avg={s['avg_pnl']:.2f}", flush=True)


def load_yf_volume():
    """Load Yahoo Finance H1 data with real volume."""
    yf_path = Path("data/xauusd_h1_yf.csv")
    if not yf_path.exists():
        raise FileNotFoundError(f"YF data not found: {yf_path}")
    df = pd.read_csv(yf_path, parse_dates=['Datetime'], index_col='Datetime')
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    df = df[df['volume'] > 0].copy()
    return df


def compute_volume_signals(df):
    """Compute all 5 volume-based signals on H1 DataFrame."""
    df = df.copy()

    # 1. Volume Spike: Vol / MA20
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] / df['vol_ma20']

    # 2. Rolling VWAP (20-bar)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_20'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_dev'] = (df['close'] - df['vwap_20']) / df['vwap_20'] * 100  # % deviation

    # 3. OBV + slope
    direction = np.sign(df['close'].diff())
    df['obv'] = (direction * df['volume']).cumsum()
    df['obv_slope_5'] = df['obv'].diff(5)
    df['obv_slope_10'] = df['obv'].diff(10)
    df['price_slope_5'] = df['close'].diff(5)

    # 4. Volume Climax: extreme volume + reversal candle
    df['vol_z'] = (df['volume'] - df['vol_ma20']) / df['volume'].rolling(20).std()
    body = abs(df['close'] - df['open'])
    full_range = df['high'] - df['low']
    df['body_ratio'] = body / full_range.replace(0, np.nan)
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / full_range.replace(0, np.nan)
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / full_range.replace(0, np.nan)
    df['climax_bull'] = (df['vol_z'] > 2) & (df['lower_wick'] > 0.5)
    df['climax_bear'] = (df['vol_z'] > 2) & (df['upper_wick'] > 0.5)

    # 5. Pseudo CVD (delta proxy)
    df['bar_delta'] = np.where(full_range > 0,
                               (df['close'] - df['low']) / full_range * 2 - 1, 0)
    df['pseudo_cvd'] = (df['bar_delta'] * df['volume']).cumsum()
    df['cvd_slope_5'] = df['pseudo_cvd'].diff(5)

    # Forward returns for IC analysis
    df['fwd_ret_1h'] = df['close'].shift(-1) / df['close'] - 1
    df['fwd_ret_4h'] = df['close'].shift(-4) / df['close'] - 1

    return df.dropna(subset=['vol_ma20', 'vwap_20', 'obv_slope_5'])


def run_l8_with_data(bundle, start_date, end_date):
    """Run Keltner backtest on a specific date range, return trades."""
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    sub = bundle.slice(start_date, end_date)
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = 35
    kw['spread_cost'] = SPREAD
    kw['initial_capital'] = 2000
    kw['min_lot_size'] = LOT
    kw['max_lot_size'] = LOT
    kw['keltner_max_hold_m15'] = 8
    result = run_variant(sub, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        entry_ts = pd.Timestamp(t.entry_time)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize('UTC')
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars_held': t.bars_held,
            'lots': t.lots,
        })
    return trades


def main():
    results = {}

    print("=" * 100, flush=True)
    print("  R158 — Order Flow / Volume Signal Research", flush=True)
    print(f"  Started: {datetime.now()}", flush=True)
    print("=" * 100, flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Data Alignment
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 1: Data Alignment (YF Volume + Dukascopy Price)", flush=True)
    print(f"{'='*100}\n", flush=True)

    yf = load_yf_volume()
    print(f"  YF data: {len(yf)} H1 bars, {yf.index[0]} ~ {yf.index[-1]}", flush=True)
    print(f"  Volume: mean={yf['volume'].mean():.0f}, median={yf['volume'].median():.0f}, "
          f"max={yf['volume'].max():.0f}", flush=True)

    # Load Dukascopy data for price alignment
    from backtest.runner import DataBundle
    print("\n  Loading DataBundle (Dukascopy)...", flush=True)
    bundle = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
    print("  Bundle ready.", flush=True)

    # Find overlap period
    dk_start = bundle.h1_df.index[0]
    dk_end = bundle.h1_df.index[-1]
    overlap_start = max(yf.index[0], dk_start)
    overlap_end = min(yf.index[-1], dk_end)
    print(f"\n  Dukascopy range: {dk_start} ~ {dk_end}", flush=True)
    print(f"  Overlap period:  {overlap_start} ~ {overlap_end}", flush=True)

    # Merge YF volume onto Dukascopy H1 index
    dk_h1 = bundle.h1_df.loc[overlap_start:overlap_end].copy()
    yf_vol = yf['volume'].reindex(dk_h1.index, method=None)
    matched = yf_vol.notna().sum()
    print(f"  Dukascopy bars in overlap: {len(dk_h1)}", flush=True)
    print(f"  YF volume matched: {matched} ({matched/len(dk_h1)*100:.1f}%)", flush=True)

    # For unmatched, try nearest-hour matching (YF timestamps may differ by minutes)
    if matched < len(dk_h1) * 0.5:
        print("  Low match rate - trying fuzzy merge (nearest hour)...", flush=True)
        yf_resampled = yf['volume'].resample('1h').sum()
        yf_vol = yf_resampled.reindex(dk_h1.index, method='nearest', tolerance=pd.Timedelta('30min'))
        matched = yf_vol.notna().sum()
        print(f"  After fuzzy merge: {matched} ({matched/len(dk_h1)*100:.1f}%)", flush=True)

    dk_h1['real_volume'] = yf_vol.fillna(0)
    dk_h1 = dk_h1[dk_h1['real_volume'] > 0].copy()
    print(f"  Final merged bars with volume: {len(dk_h1)}", flush=True)

    results['phase1'] = {
        'yf_bars': len(yf),
        'overlap_start': str(overlap_start),
        'overlap_end': str(overlap_end),
        'dk_bars_overlap': int(len(bundle.h1_df.loc[overlap_start:overlap_end])),
        'matched_volume_bars': int(len(dk_h1)),
    }

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Signal Calculation + IC Analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 2: Volume Signal Computation + IC Analysis", flush=True)
    print(f"{'='*100}\n", flush=True)

    # Build signal DataFrame from YF data directly (prices are close enough)
    sig_df = yf.copy()
    sig_df.rename(columns={'volume': 'volume'}, inplace=True)
    sig_df = compute_volume_signals(sig_df)
    print(f"  Signal DataFrame: {len(sig_df)} bars after dropna", flush=True)

    # IC analysis: correlation between each signal and forward returns
    signal_cols = ['vol_spike', 'vwap_dev', 'obv_slope_5', 'obv_slope_10',
                   'vol_z', 'bar_delta', 'cvd_slope_5']
    fwd_cols = ['fwd_ret_1h', 'fwd_ret_4h']

    ic_results = {}
    print("\n  Information Coefficient (Pearson) with forward returns:", flush=True)
    print(f"  {'Signal':<20} {'IC (1h)':>10} {'IC (4h)':>10} {'|IC| rank':>10}", flush=True)
    print(f"  {'-'*50}", flush=True)

    for sig in signal_cols:
        ics = {}
        for fwd in fwd_cols:
            valid = sig_df[[sig, fwd]].dropna()
            if len(valid) > 50:
                ic = valid[sig].corr(valid[fwd])
            else:
                ic = 0.0
            ics[fwd] = round(ic, 4)
        ic_results[sig] = ics
        print(f"  {sig:<20} {ics['fwd_ret_1h']:>10.4f} {ics['fwd_ret_4h']:>10.4f}", flush=True)

    # Rank by absolute IC for 4h forward
    ranked = sorted(ic_results.items(), key=lambda x: abs(x[1]['fwd_ret_4h']), reverse=True)
    print(f"\n  Top signals by |IC(4h)|:", flush=True)
    for i, (sig, ics) in enumerate(ranked):
        print(f"    #{i+1} {sig}: IC(4h) = {ics['fwd_ret_4h']:.4f}", flush=True)

    results['phase2_ic'] = ic_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Volume as Keltner Entry Filter
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 3: Volume as Keltner Entry Filter", flush=True)
    print(f"{'='*100}\n", flush=True)

    overlap_start_str = str(overlap_start.date())
    overlap_end_str = str(overlap_end.date())

    # Baseline: Keltner on overlap period only
    base_trades = run_l8_with_data(bundle, overlap_start_str, overlap_end_str)
    base_stats = stats(base_trades)
    print_row(f"Baseline Keltner (MH=8, {overlap_start_str[:7]}~{overlap_end_str[:7]})", base_stats)
    results['phase3_baseline'] = base_stats

    # Build volume lookup for filtering
    vol_lookup = {}
    for ts, row in sig_df.iterrows():
        vol_lookup[ts] = {
            'vol_spike': row.get('vol_spike', 1.0),
            'obv_slope_5': row.get('obv_slope_5', 0),
            'vwap_dev': row.get('vwap_dev', 0),
            'cvd_slope_5': row.get('cvd_slope_5', 0),
            'vol_z': row.get('vol_z', 0),
        }

    def get_vol_signal(entry_time, field):
        """Find the nearest H1 bar's volume signal for a trade entry time."""
        et = pd.Timestamp(entry_time)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        h1_bar = et.floor('h')
        if h1_bar in vol_lookup:
            return vol_lookup[h1_bar].get(field, None)
        # Try 1 hour back
        h1_prev = h1_bar - pd.Timedelta(hours=1)
        if h1_prev in vol_lookup:
            return vol_lookup[h1_prev].get(field, None)
        return None

    # Annotate trades with volume signals
    for t in base_trades:
        for field in ['vol_spike', 'obv_slope_5', 'vwap_dev', 'cvd_slope_5', 'vol_z']:
            t[field] = get_vol_signal(t['entry_time'], field)

    annotated = [t for t in base_trades if t.get('vol_spike') is not None]
    not_annotated = len(base_trades) - len(annotated)
    print(f"\n  Trades with volume data: {len(annotated)} / {len(base_trades)} "
          f"({not_annotated} unmatched)", flush=True)

    # Filter tests
    filter_results = []

    # 3a: Volume Spike filter (only enter when vol > Nx MA20)
    print("\n  3a: Volume Spike Filter (require Vol > N*MA20 at entry):", flush=True)
    for threshold in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        filtered = [t for t in annotated if t['vol_spike'] >= threshold]
        rejected = [t for t in annotated if t['vol_spike'] < threshold]
        s_pass = stats(filtered)
        s_rej = stats(rejected)
        label = f"VolSpike >= {threshold:.1f}"
        print_row(f"  PASS {label}", s_pass)
        if rejected:
            print_row(f"  REJECT {label}", s_rej)
        filter_results.append({
            'filter': 'vol_spike', 'threshold': threshold,
            'pass': s_pass, 'reject': s_rej,
            'retention': round(len(filtered) / max(len(annotated), 1) * 100, 1),
        })

    # 3b: OBV slope alignment (only enter BUY when OBV rising, SELL when falling)
    print("\n  3b: OBV Slope Alignment Filter:", flush=True)
    obv_aligned = [t for t in annotated
                   if t.get('obv_slope_5') is not None and (
                       (t['dir'] == 'BUY' and t['obv_slope_5'] > 0) or
                       (t['dir'] == 'SELL' and t['obv_slope_5'] < 0)
                   )]
    obv_rejected = [t for t in annotated if t not in obv_aligned]
    s_pass = stats(obv_aligned)
    s_rej = stats(obv_rejected)
    print_row("  PASS OBV slope aligned", s_pass)
    print_row("  REJECT OBV slope opposed", s_rej)
    filter_results.append({
        'filter': 'obv_slope_aligned',
        'pass': s_pass, 'reject': s_rej,
        'retention': round(len(obv_aligned) / max(len(annotated), 1) * 100, 1),
    })

    # 3c: CVD slope alignment
    print("\n  3c: CVD Slope Alignment Filter:", flush=True)
    cvd_aligned = [t for t in annotated
                   if t.get('cvd_slope_5') is not None and (
                       (t['dir'] == 'BUY' and t['cvd_slope_5'] > 0) or
                       (t['dir'] == 'SELL' and t['cvd_slope_5'] < 0)
                   )]
    cvd_rejected = [t for t in annotated if t not in cvd_aligned]
    s_pass = stats(cvd_aligned)
    s_rej = stats(cvd_rejected)
    print_row("  PASS CVD slope aligned", s_pass)
    print_row("  REJECT CVD slope opposed", s_rej)
    filter_results.append({
        'filter': 'cvd_slope_aligned',
        'pass': s_pass, 'reject': s_rej,
        'retention': round(len(cvd_aligned) / max(len(annotated), 1) * 100, 1),
    })

    # 3d: Combined filter (VolSpike >= 1.2 AND OBV aligned)
    print("\n  3d: Combined Filter (VolSpike>=1.2 + OBV aligned):", flush=True)
    combined = [t for t in annotated
                if t.get('vol_spike', 0) >= 1.2
                and t.get('obv_slope_5') is not None
                and ((t['dir'] == 'BUY' and t['obv_slope_5'] > 0) or
                     (t['dir'] == 'SELL' and t['obv_slope_5'] < 0))]
    combined_rej = [t for t in annotated if t not in combined]
    s_pass = stats(combined)
    s_rej = stats(combined_rej)
    print_row("  PASS Combined", s_pass)
    print_row("  REJECT Combined", s_rej)
    filter_results.append({
        'filter': 'vol_spike_1.2_and_obv',
        'pass': s_pass, 'reject': s_rej,
        'retention': round(len(combined) / max(len(annotated), 1) * 100, 1),
    })

    results['phase3_filters'] = filter_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Volume as Exit Signal
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 4: Volume Climax as Exit Signal Analysis", flush=True)
    print(f"{'='*100}\n", flush=True)

    # Analyze: do winning trades exit near volume climax?
    # Check volume conditions at exit time
    exit_analysis = {'climax_at_exit': 0, 'no_climax': 0, 'climax_pnl': [], 'no_climax_pnl': []}
    for t in annotated:
        exit_vol_z = get_vol_signal(t['exit_time'], 'vol_z')
        if exit_vol_z is not None and exit_vol_z > 2:
            exit_analysis['climax_at_exit'] += 1
            exit_analysis['climax_pnl'].append(t['pnl'])
        else:
            exit_analysis['no_climax'] += 1
            exit_analysis['no_climax_pnl'].append(t['pnl'])

    print(f"  Trades exiting at volume climax (vol_z > 2): {exit_analysis['climax_at_exit']}", flush=True)
    if exit_analysis['climax_pnl']:
        print(f"    Avg PnL: ${np.mean(exit_analysis['climax_pnl']):.2f}, "
              f"WR: {sum(1 for p in exit_analysis['climax_pnl'] if p>0)/len(exit_analysis['climax_pnl'])*100:.0f}%", flush=True)
    print(f"  Trades exiting WITHOUT climax: {exit_analysis['no_climax']}", flush=True)
    if exit_analysis['no_climax_pnl']:
        print(f"    Avg PnL: ${np.mean(exit_analysis['no_climax_pnl']):.2f}, "
              f"WR: {sum(1 for p in exit_analysis['no_climax_pnl'] if p>0)/len(exit_analysis['no_climax_pnl'])*100:.0f}%", flush=True)

    # OBV divergence at exit
    print("\n  OBV divergence analysis (would it help exit earlier?):", flush=True)
    losing_trades = [t for t in annotated if t['pnl'] < 0]
    winning_trades = [t for t in annotated if t['pnl'] > 0]

    losing_obv_opposed = sum(1 for t in losing_trades
                             if t.get('obv_slope_5') is not None and (
                                 (t['dir'] == 'BUY' and t['obv_slope_5'] < 0) or
                                 (t['dir'] == 'SELL' and t['obv_slope_5'] > 0)
                             ))
    print(f"  Losing trades with OBV opposing entry dir: {losing_obv_opposed}/{len(losing_trades)}", flush=True)

    results['phase4_exit'] = {
        'climax_at_exit': exit_analysis['climax_at_exit'],
        'climax_avg_pnl': round(np.mean(exit_analysis['climax_pnl']), 2) if exit_analysis['climax_pnl'] else 0,
        'no_climax': exit_analysis['no_climax'],
        'no_climax_avg_pnl': round(np.mean(exit_analysis['no_climax_pnl']), 2) if exit_analysis['no_climax_pnl'] else 0,
        'losing_with_obv_opposed': losing_obv_opposed,
        'total_losing': len(losing_trades),
    }

    # ═══════════════════════════════════════════════════════════════
    # Phase 5: Independent VWAP Reversion Strategy
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 5: Independent VWAP Reversion Strategy", flush=True)
    print(f"{'='*100}\n", flush=True)

    vwap_strat_results = []
    for dev_thresh in [0.1, 0.15, 0.2, 0.3, 0.5]:
        for hold_bars in [2, 4, 8, 12]:
            trades_vwap = []
            for i in range(20, len(sig_df) - hold_bars):
                row = sig_df.iloc[i]
                vwap_d = row.get('vwap_dev', 0)
                if pd.isna(vwap_d):
                    continue
                direction = None
                if vwap_d < -dev_thresh:
                    direction = 'BUY'
                elif vwap_d > dev_thresh:
                    direction = 'SELL'
                if direction is None:
                    continue

                entry_price = row['close'] + SPREAD / 2
                exit_row = sig_df.iloc[i + hold_bars]
                exit_price = exit_row['close'] - SPREAD / 2

                if direction == 'BUY':
                    pnl = (exit_price - entry_price) * LOT * PV
                else:
                    pnl = (entry_price - exit_price) * LOT * PV

                trades_vwap.append({
                    'dir': direction, 'entry': entry_price, 'exit': exit_price,
                    'entry_time': sig_df.index[i], 'exit_time': sig_df.index[i + hold_bars],
                    'pnl': round(pnl, 2), 'reason': 'timeout', 'bars_held': hold_bars,
                })

            s = stats(trades_vwap)
            s['dev_thresh'] = dev_thresh
            s['hold_bars'] = hold_bars
            if s['n'] > 0:
                label = f"VWAP dev={dev_thresh}% hold={hold_bars}h"
                print_row(label, s)
                vwap_strat_results.append(s)

    results['phase5_vwap'] = vwap_strat_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 6: K-Fold Validation (3 folds)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 6: K-Fold Validation (3 folds, ~8 months each)", flush=True)
    print(f"{'='*100}\n", flush=True)

    total_days = (pd.Timestamp(overlap_end_str) - pd.Timestamp(overlap_start_str)).days
    n_folds = 3
    fold_days = total_days // n_folds

    folds = []
    for i in range(n_folds):
        fs = pd.Timestamp(overlap_start_str) + pd.Timedelta(days=i * fold_days)
        fe = fs + pd.Timedelta(days=fold_days) if i < n_folds - 1 else pd.Timestamp(overlap_end_str)
        folds.append((str(fs.date()), str(fe.date())))

    # Find best entry filter from Phase 3
    best_filter_label = "VolSpike >= 1.2"
    best_thresh = 1.2

    kfold_results = {}

    # Baseline K-Fold
    print("  Baseline (no volume filter):", flush=True)
    base_folds = []
    for fi, (fs, fe) in enumerate(folds):
        t_fold = run_l8_with_data(bundle, fs, fe)
        s = stats(t_fold)
        base_folds.append(s['sharpe'])
        print(f"    Fold {fi+1} ({fs} ~ {fe}): Sh={s['sharpe']:.2f} n={s['n']} PnL={fmt(s['pnl'])}", flush=True)
    kfold_results['baseline'] = {
        'folds_sharpe': base_folds,
        'mean_sharpe': round(np.mean(base_folds), 2),
    }

    # VolSpike filtered K-Fold
    print(f"\n  {best_filter_label} filter:", flush=True)
    vs_folds = []
    for fi, (fs, fe) in enumerate(folds):
        t_fold = run_l8_with_data(bundle, fs, fe)
        # Annotate with volume
        t_annotated = []
        for t in t_fold:
            vs = get_vol_signal(t['entry_time'], 'vol_spike')
            if vs is not None and vs >= best_thresh:
                t_annotated.append(t)
        s = stats(t_annotated)
        vs_folds.append(s['sharpe'])
        print(f"    Fold {fi+1} ({fs} ~ {fe}): Sh={s['sharpe']:.2f} n={s['n']} PnL={fmt(s['pnl'])}", flush=True)
    kfold_results['vol_spike_1.2'] = {
        'folds_sharpe': vs_folds,
        'mean_sharpe': round(np.mean(vs_folds), 2),
    }

    # OBV aligned K-Fold
    print(f"\n  OBV slope aligned filter:", flush=True)
    obv_folds = []
    for fi, (fs, fe) in enumerate(folds):
        t_fold = run_l8_with_data(bundle, fs, fe)
        t_annotated = []
        for t in t_fold:
            obv = get_vol_signal(t['entry_time'], 'obv_slope_5')
            if obv is not None and (
                (t['dir'] == 'BUY' and obv > 0) or
                (t['dir'] == 'SELL' and obv < 0)
            ):
                t_annotated.append(t)
        s = stats(t_annotated)
        obv_folds.append(s['sharpe'])
        print(f"    Fold {fi+1} ({fs} ~ {fe}): Sh={s['sharpe']:.2f} n={s['n']} PnL={fmt(s['pnl'])}", flush=True)
    kfold_results['obv_aligned'] = {
        'folds_sharpe': obv_folds,
        'mean_sharpe': round(np.mean(obv_folds), 2),
    }

    results['phase6_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Tick Volume Proxy Test
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}", flush=True)
    print("  Phase 7: Tick Volume Proxy (bar range as volume substitute)", flush=True)
    print(f"{'='*100}\n", flush=True)

    # Hypothesis: H-L range correlates with real volume
    # If so, we can use range as volume proxy in live trading
    merged_for_proxy = dk_h1[dk_h1['real_volume'] > 0].copy()
    h_col = 'High' if 'High' in merged_for_proxy.columns else 'high'
    l_col = 'Low' if 'Low' in merged_for_proxy.columns else 'low'
    merged_for_proxy['bar_range'] = merged_for_proxy[h_col] - merged_for_proxy[l_col]
    merged_for_proxy['bar_range_ma20'] = merged_for_proxy['bar_range'].rolling(20).mean()
    merged_for_proxy['range_spike'] = merged_for_proxy['bar_range'] / merged_for_proxy['bar_range_ma20']

    if len(merged_for_proxy.dropna()) > 50:
        valid = merged_for_proxy.dropna(subset=['real_volume', 'bar_range'])
        range_vol_corr = valid['bar_range'].corr(valid['real_volume'])
        range_spike_corr = valid['range_spike'].corr(
            valid['real_volume'] / valid['real_volume'].rolling(20).mean()
        )
        print(f"  Bar Range vs Real Volume correlation: {range_vol_corr:.4f}", flush=True)
        print(f"  Range Spike vs Volume Spike correlation: {range_spike_corr:.4f}", flush=True)

        # If correlation is decent (> 0.3), range can proxy volume in live EA
        if abs(range_vol_corr) > 0.3:
            print(f"  -> Moderate correlation! Range can serve as volume proxy.", flush=True)
        else:
            print(f"  -> Weak correlation. Range is not a reliable volume proxy.", flush=True)

        # Test range-based filter on full Dukascopy data
        print("\n  Testing range spike as entry filter (full 11-year backtest):", flush=True)
        full_trades = run_l8_with_data(bundle, "2015-01-01", "2026-05-01")
        full_base = stats(full_trades)
        print_row("Full baseline (2015-2026)", full_base)

        # Build range spike lookup from Dukascopy H1
        dk_full_h1 = bundle.h1_df.copy()
        h_col2 = 'High' if 'High' in dk_full_h1.columns else 'high'
        l_col2 = 'Low' if 'Low' in dk_full_h1.columns else 'low'
        dk_full_h1['bar_range'] = dk_full_h1[h_col2] - dk_full_h1[l_col2]
        dk_full_h1['range_ma20'] = dk_full_h1['bar_range'].rolling(20).mean()
        dk_full_h1['range_spike'] = dk_full_h1['bar_range'] / dk_full_h1['range_ma20']
        range_lookup = dk_full_h1['range_spike'].to_dict()

        for rng_thresh in [1.0, 1.2, 1.5, 2.0]:
            filtered = []
            for t in full_trades:
                et = pd.Timestamp(t['entry_time'])
                if et.tzinfo is None:
                    et = et.tz_localize('UTC')
                h1_bar = et.floor('h')
                rs = range_lookup.get(h1_bar, None)
                if rs is not None and rs >= rng_thresh:
                    filtered.append(t)
            s = stats(filtered)
            s['range_thresh'] = rng_thresh
            s['retention'] = round(len(filtered) / max(len(full_trades), 1) * 100, 1)
            print_row(f"Range spike >= {rng_thresh:.1f} ({s['retention']:.0f}% kept)", s)

        results['phase7_proxy'] = {
            'range_vol_corr': round(range_vol_corr, 4),
            'range_spike_corr': round(range_spike_corr, 4) if not pd.isna(range_spike_corr) else 0,
        }
    else:
        print("  Not enough merged data for proxy analysis.", flush=True)
        results['phase7_proxy'] = {'error': 'insufficient_data'}

    # ═══════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    out_file = OUTPUT_DIR / "r158_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"  R158 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"{'='*100}\n", flush=True)
    print(f"  Saved: {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
