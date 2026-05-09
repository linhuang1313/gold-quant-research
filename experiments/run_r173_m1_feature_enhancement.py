#!/usr/bin/env python3
"""
R173 — M1 Microstructure Feature Enhancement for H1 Entry
============================================================
Tests whether M1-level features can improve H1 Keltner entry signal quality
(NOT replace them, just confirm/filter).

Phase 1: M1 Feature Engineering
  - Load M1 data (all years, ~260MB)
  - Resample M1→H1 to compute per-H1-bar microstructure features
  - Merge features onto H1 DataFrame

Phase 2: Feature IC Analysis
  - Information Coefficient of each M1 feature vs next-bar H1 return
  - IC_IR = mean(IC) / std(IC) over rolling windows

Phase 3: Entry Signal Quality Analysis
  - Run L8_MAX baseline to get all Keltner entry signals
  - Discriminate good vs bad trades using M1 features (Cohen's d)

Phase 4: XGBoost Entry Filter
  - Walk-forward ML filter using M1 + H1 features
  - Compare M1-only, H1-only, and combined filters

Phase 5: Backtest Impact
  - Apply best filter to L8_MAX strategy
  - K-Fold 6-fold validation
"""
import sys, os, time, json, glob, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_dir)
warnings.filterwarnings('ignore')

from backtest.runner import (
    DataBundle, run_variant, LIVE_PARITY_KWARGS,
    load_m15, load_h1_aligned, H1_CSV_PATH,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

OUTPUT_DIR = Path("results/r173_m1_feature_enhancement")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

M1_DATA_DIR = Path("data/m1_raw")
PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

M1_FEATURES = [
    'm1_range_ratio', 'm1_acceleration', 'm1_momentum_15',
    'm1_trend_consistency', 'm1_reversal_count', 'm1_late_surge',
    'm1_opening_gap', 'm1_body_vs_wick', 'm1_vwap_dist',
    'm1_volume_profile',
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def elapsed():
    return f"[{time.time() - t0:.0f}s]"


def save_json(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}", flush=True)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 5 or n2 < 5:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (m1 - m2) / pooled_std


# ═══════════════════════════════════════════════════════════════
# Phase 1: M1 Feature Engineering
# ═══════════════════════════════════════════════════════════════

def load_m1_year(filepath: str) -> pd.DataFrame:
    """Load a single M1 CSV file (semicolon-separated, no header)."""
    df = pd.read_csv(filepath, sep=';', header=None,
                     names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S', utc=True)
    df.set_index('datetime', inplace=True)
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    return df


def compute_m1_features_for_h1_group(m1_group: pd.DataFrame, prev_h1_close: float) -> Dict:
    """Compute microstructure features for M1 bars within one H1 bar."""
    n = len(m1_group)
    if n < 5:
        return {feat: np.nan for feat in M1_FEATURES}

    opens = m1_group['Open'].values
    highs = m1_group['High'].values
    lows = m1_group['Low'].values
    closes = m1_group['Close'].values
    volumes = m1_group['Volume'].values if 'Volume' in m1_group.columns else np.zeros(n)

    h1_high = highs.max()
    h1_low = lows.min()
    h1_range = h1_high - h1_low
    h1_open = opens[0]
    h1_close = closes[-1]

    features = {}

    # m1_range_ratio: max single-bar range / H1 range
    m1_ranges = highs - lows
    if h1_range > 1e-6:
        features['m1_range_ratio'] = m1_ranges.max() / h1_range
    else:
        features['m1_range_ratio'] = 0.0

    # m1_acceleration: 2nd derivative of close at end of bar
    if n >= 3:
        d1 = np.diff(closes)
        d2 = np.diff(d1)
        features['m1_acceleration'] = d2[-1] if len(d2) > 0 else 0.0
    else:
        features['m1_acceleration'] = 0.0

    # m1_momentum_15: momentum in last 15 minutes
    last_15 = min(15, n)
    features['m1_momentum_15'] = closes[-1] - closes[-last_15]

    # m1_trend_consistency: % of bars moving in same direction as H1 bar
    h1_direction = np.sign(h1_close - h1_open)
    if h1_direction == 0:
        features['m1_trend_consistency'] = 0.5
    else:
        bar_directions = np.sign(closes - opens)
        features['m1_trend_consistency'] = np.mean(bar_directions == h1_direction)

    # m1_reversal_count: direction changes
    close_diffs = np.diff(closes)
    signs = np.sign(close_diffs)
    signs_nonzero = signs[signs != 0]
    if len(signs_nonzero) > 1:
        features['m1_reversal_count'] = np.sum(np.diff(signs_nonzero) != 0)
    else:
        features['m1_reversal_count'] = 0

    # m1_late_surge: (last 15min range) / (first 45min range)
    split_idx = max(1, n - 15)
    late_range = highs[split_idx:].max() - lows[split_idx:].min() if split_idx < n else 0
    early_range = highs[:split_idx].max() - lows[:split_idx].min() if split_idx > 0 else 1e-6
    features['m1_late_surge'] = late_range / max(early_range, 1e-6)

    # m1_opening_gap: |first M1 open - previous H1 close|
    if prev_h1_close > 0:
        features['m1_opening_gap'] = abs(opens[0] - prev_h1_close)
    else:
        features['m1_opening_gap'] = 0.0

    # m1_body_vs_wick: body / range from M1 perspective
    if h1_range > 1e-6:
        features['m1_body_vs_wick'] = abs(h1_close - h1_open) / h1_range
    else:
        features['m1_body_vs_wick'] = 0.0

    # m1_vwap_dist: VWAP distance from close (if volume > 0)
    total_vol = volumes.sum()
    if total_vol > 0:
        typical_price = (highs + lows + closes) / 3.0
        vwap = np.sum(typical_price * volumes) / total_vol
        features['m1_vwap_dist'] = (h1_close - vwap) / max(h1_range, 1e-6)
    else:
        features['m1_vwap_dist'] = 0.0

    # m1_volume_profile: volume in last 15min / total volume
    if total_vol > 0:
        late_vol = volumes[split_idx:].sum()
        features['m1_volume_profile'] = late_vol / total_vol
    else:
        features['m1_volume_profile'] = 0.5

    return features


def build_m1_features(m1_all: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """Build M1 microstructure features aligned to H1 bars."""
    print(f"\n{elapsed()} Building M1 features for {len(h1_df)} H1 bars...", flush=True)

    m1_all['h1_bar'] = m1_all.index.floor('h')

    h1_times = h1_df.index
    feature_records = []
    prev_close = 0.0
    processed = 0
    total = len(h1_times)

    for h1_time in h1_times:
        m1_group = m1_all[m1_all['h1_bar'] == h1_time]
        features = compute_m1_features_for_h1_group(m1_group, prev_close)
        features['h1_time'] = h1_time
        feature_records.append(features)

        if len(m1_group) > 0:
            prev_close = m1_group['Close'].iloc[-1]

        processed += 1
        if processed % 5000 == 0:
            print(f"    {processed}/{total} H1 bars processed ({processed/total*100:.1f}%)", flush=True)

    feat_df = pd.DataFrame(feature_records)
    feat_df.set_index('h1_time', inplace=True)
    feat_df.index = feat_df.index.tz_localize('UTC') if feat_df.index.tz is None else feat_df.index

    coverage = feat_df[M1_FEATURES[0]].notna().mean()
    print(f"  Feature coverage: {coverage:.1%} of H1 bars have M1 data", flush=True)
    print(f"  Feature shape: {feat_df.shape}", flush=True)

    return feat_df


def phase1_m1_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 1: Load M1 data and compute features."""
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 1: M1 Feature Engineering", flush=True)
    print(f"{'='*70}", flush=True)

    m1_files = sorted(glob.glob(str(M1_DATA_DIR / "xauusd_*.csv")))
    if not m1_files:
        print(f"  ERROR: No M1 files found in {M1_DATA_DIR}", flush=True)
        return None, None

    print(f"  Found {len(m1_files)} M1 files:", flush=True)
    for f in m1_files:
        print(f"    {Path(f).name}", flush=True)

    # Load M1 data year by year to manage memory
    print(f"\n{elapsed()} Loading M1 data...", flush=True)
    m1_dfs = []
    for f in m1_files:
        print(f"    Loading {Path(f).name}...", end='', flush=True)
        df = load_m1_year(f)
        print(f" {len(df)} bars", flush=True)
        m1_dfs.append(df)

    m1_all = pd.concat(m1_dfs).sort_index()
    del m1_dfs
    gc.collect()
    print(f"  Total M1 bars: {len(m1_all):,}", flush=True)
    print(f"  M1 range: {m1_all.index[0]} -> {m1_all.index[-1]}", flush=True)
    print(f"  Memory: ~{m1_all.memory_usage(deep=True).sum() / 1e6:.0f} MB", flush=True)

    # Load H1 data
    print(f"\n{elapsed()} Loading H1 data...", flush=True)
    data = DataBundle.load_default()
    h1_df = data.h1_df

    # Compute M1 features
    feat_df = build_m1_features(m1_all, h1_df)

    # Merge onto H1
    h1_with_m1 = h1_df.join(feat_df, how='left')
    print(f"\n  Merged H1 shape: {h1_with_m1.shape}", flush=True)

    # Save feature stats
    feat_stats = {}
    for col in M1_FEATURES:
        if col in h1_with_m1.columns:
            s = h1_with_m1[col].dropna()
            feat_stats[col] = {
                'count': int(len(s)),
                'mean': float(s.mean()),
                'std': float(s.std()),
                'min': float(s.min()),
                'max': float(s.max()),
                'median': float(s.median()),
                'coverage': float(s.notna().mean() if col in h1_with_m1.columns else 0),
            }
    save_json(feat_stats, "phase1_feature_stats.json")

    # Free M1 memory
    del m1_all
    gc.collect()
    print(f"\n{elapsed()} Phase 1 complete. M1 memory freed.", flush=True)

    return h1_with_m1, data


# ═══════════════════════════════════════════════════════════════
# Phase 2: Feature IC Analysis
# ═══════════════════════════════════════════════════════════════

def phase2_ic_analysis(h1_with_m1: pd.DataFrame):
    """Compute Information Coefficient of M1 features vs forward returns."""
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 2: Feature IC Analysis", flush=True)
    print(f"{'='*70}", flush=True)

    # Forward 1-bar return
    h1_with_m1['fwd_return'] = h1_with_m1['Close'].shift(-1) / h1_with_m1['Close'] - 1.0

    ic_results = {}
    rolling_window = 500

    for feat in M1_FEATURES:
        if feat not in h1_with_m1.columns:
            continue
        valid = h1_with_m1[[feat, 'fwd_return']].dropna()
        if len(valid) < 100:
            ic_results[feat] = {'ic': 0.0, 'ic_ir': 0.0, 'n': 0}
            continue

        # Full-sample IC (Spearman)
        ic_full, p_val = scipy_stats.spearmanr(valid[feat], valid['fwd_return'])

        # Rolling IC for IC_IR
        rolling_ics = []
        for start_idx in range(0, len(valid) - rolling_window, rolling_window // 2):
            chunk = valid.iloc[start_idx:start_idx + rolling_window]
            if len(chunk) >= 50:
                r, _ = scipy_stats.spearmanr(chunk[feat], chunk['fwd_return'])
                if not np.isnan(r):
                    rolling_ics.append(r)

        ic_mean = np.mean(rolling_ics) if rolling_ics else ic_full
        ic_std = np.std(rolling_ics) if len(rolling_ics) > 1 else 1.0
        ic_ir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

        ic_results[feat] = {
            'ic': float(ic_full),
            'ic_pval': float(p_val),
            'ic_mean_rolling': float(ic_mean),
            'ic_std_rolling': float(ic_std),
            'ic_ir': float(ic_ir),
            'n': int(len(valid)),
        }
        print(f"  {feat:25s}: IC={ic_full:+.4f} (p={p_val:.3f}), IC_IR={ic_ir:+.3f}, N={len(valid)}", flush=True)

    # Rank by |IC|
    ranked = sorted(ic_results.items(), key=lambda x: abs(x[1].get('ic', 0)), reverse=True)
    print(f"\n  Top features by |IC|:", flush=True)
    for feat, vals in ranked[:5]:
        print(f"    {feat:25s}: |IC|={abs(vals['ic']):.4f}, IC_IR={vals['ic_ir']:+.3f}", flush=True)

    # Compare with standard H1 features
    h1_features = ['ATR', 'ADX', 'RSI14', 'squeeze']
    print(f"\n  H1-only feature ICs for comparison:", flush=True)
    for feat in h1_features:
        if feat in h1_with_m1.columns:
            valid = h1_with_m1[[feat, 'fwd_return']].dropna()
            if len(valid) > 100:
                ic, p = scipy_stats.spearmanr(valid[feat], valid['fwd_return'])
                print(f"    {feat:25s}: IC={ic:+.4f} (p={p:.3f})", flush=True)
                ic_results[f"h1_{feat}"] = {'ic': float(ic), 'ic_pval': float(p), 'n': int(len(valid))}

    save_json(ic_results, "phase2_ic_analysis.json")
    print(f"\n{elapsed()} Phase 2 complete.", flush=True)
    return ic_results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Entry Signal Quality Analysis
# ═══════════════════════════════════════════════════════════════

def phase3_entry_quality(h1_with_m1: pd.DataFrame, data: DataBundle):
    """Analyze which M1 features discriminate good vs bad Keltner entries."""
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 3: Entry Signal Quality Analysis", flush=True)
    print(f"{'='*70}", flush=True)

    # Run L8_MAX baseline to get trades
    print(f"\n{elapsed()} Running L8_MAX baseline...", flush=True)
    baseline_kwargs = dict(LIVE_PARITY_KWARGS)
    baseline_kwargs['min_lot_size'] = 0.02
    baseline_kwargs['max_lot_size'] = 0.02
    baseline_kwargs['maxloss_cap'] = 35

    stats = run_variant(data, "L8_MAX_baseline", **baseline_kwargs)
    trades = stats.get('_trades', [])
    print(f"  Baseline: {len(trades)} trades, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}", flush=True)

    if not trades:
        print("  ERROR: No trades from baseline", flush=True)
        return {}

    # Match each trade's entry time to H1 M1 features
    trade_features = []
    for trade in trades:
        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
        pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
        strategy = trade.strategy if hasattr(trade, 'strategy') else trade.get('strategy', '')

        if 'L8' not in str(strategy) and 'Keltner' not in str(strategy):
            continue

        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        if entry_time is None:
            continue

        # Find nearest H1 bar
        h1_time = pd.Timestamp(entry_time).floor('h')
        if h1_time.tzinfo is None:
            h1_time = h1_time.tz_localize('UTC')

        if h1_time in h1_with_m1.index:
            row = h1_with_m1.loc[h1_time]
            feat_vals = {feat: row.get(feat, np.nan) for feat in M1_FEATURES if feat in h1_with_m1.columns}
            feat_vals['pnl'] = pnl
            feat_vals['is_good'] = 1 if pnl > 0 else 0
            trade_features.append(feat_vals)

    if not trade_features:
        # Fallback: try matching with tolerance
        for trade in trades:
            entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
            pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)

            if isinstance(entry_time, str):
                entry_time = pd.Timestamp(entry_time)
            if entry_time is None:
                continue

            h1_time = pd.Timestamp(entry_time).floor('h')
            if h1_time.tzinfo is None:
                h1_time = h1_time.tz_localize('UTC')

            idx = h1_with_m1.index.get_indexer([h1_time], method='nearest')
            if idx[0] >= 0 and idx[0] < len(h1_with_m1):
                row = h1_with_m1.iloc[idx[0]]
                feat_vals = {feat: row.get(feat, np.nan) for feat in M1_FEATURES if feat in h1_with_m1.columns}
                feat_vals['pnl'] = pnl
                feat_vals['is_good'] = 1 if pnl > 0 else 0
                trade_features.append(feat_vals)

    trade_df = pd.DataFrame(trade_features)
    print(f"  Matched {len(trade_df)} Keltner entries with M1 features", flush=True)

    if len(trade_df) < 20:
        print("  WARNING: Too few matched trades for meaningful analysis", flush=True)
        save_json({'n_trades': len(trade_df), 'error': 'too_few_trades'}, "phase3_entry_quality.json")
        return {}

    good_trades = trade_df[trade_df['is_good'] == 1]
    bad_trades = trade_df[trade_df['is_good'] == 0]
    print(f"  Good trades: {len(good_trades)}, Bad trades: {len(bad_trades)}", flush=True)

    # Cohen's d for each feature
    discrimination_results = {}
    print(f"\n  Feature discrimination (Cohen's d):", flush=True)
    for feat in M1_FEATURES:
        if feat not in trade_df.columns:
            continue
        good_vals = good_trades[feat].dropna().values
        bad_vals = bad_trades[feat].dropna().values
        d = cohens_d(good_vals, bad_vals)
        discrimination_results[feat] = {
            'cohens_d': float(d),
            'good_mean': float(np.mean(good_vals)) if len(good_vals) > 0 else 0,
            'bad_mean': float(np.mean(bad_vals)) if len(bad_vals) > 0 else 0,
            'good_n': int(len(good_vals)),
            'bad_n': int(len(bad_vals)),
        }
        effect = "LARGE" if abs(d) > 0.8 else "MEDIUM" if abs(d) > 0.5 else "small"
        print(f"    {feat:25s}: d={d:+.3f} ({effect})", flush=True)

    # Rank by |d|
    ranked = sorted(discrimination_results.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)
    print(f"\n  Top discriminating M1 features:", flush=True)
    for feat, vals in ranked[:5]:
        print(f"    {feat:25s}: d={vals['cohens_d']:+.3f}", flush=True)

    save_json(discrimination_results, "phase3_entry_quality.json")
    print(f"\n{elapsed()} Phase 3 complete.", flush=True)
    return discrimination_results


# ═══════════════════════════════════════════════════════════════
# Phase 4: XGBoost Entry Filter
# ═══════════════════════════════════════════════════════════════

def phase4_xgb_filter(h1_with_m1: pd.DataFrame, data: DataBundle):
    """Walk-forward XGBoost filter using M1 and H1 features."""
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 4: XGBoost Entry Filter", flush=True)
    print(f"{'='*70}", flush=True)

    if not HAS_XGB:
        print("  WARNING: XGBoost not installed. Skipping Phase 4.", flush=True)
        save_json({'error': 'xgboost_not_installed'}, "phase4_xgb_filter.json")
        return {}
    if not HAS_SKLEARN:
        print("  WARNING: scikit-learn not installed. Skipping Phase 4.", flush=True)
        save_json({'error': 'sklearn_not_installed'}, "phase4_xgb_filter.json")
        return {}

    # Run full baseline to get all trades with entry times
    print(f"\n{elapsed()} Running full baseline for trade extraction...", flush=True)
    baseline_kwargs = dict(LIVE_PARITY_KWARGS)
    baseline_kwargs['min_lot_size'] = 0.02
    baseline_kwargs['max_lot_size'] = 0.02
    baseline_kwargs['maxloss_cap'] = 35

    stats = run_variant(data, "L8_MAX_full", **baseline_kwargs)
    trades = stats.get('_trades', [])

    # Build training dataset: features at each entry time, target = profitable
    h1_features_cols = ['ATR', 'ADX', 'RSI14']
    if 'squeeze' in h1_with_m1.columns:
        h1_features_cols.append('squeeze')

    # Compute kc_strength if not present
    if 'kc_strength' not in h1_with_m1.columns and 'KC_upper' in h1_with_m1.columns:
        kc_range = h1_with_m1['KC_upper'] - h1_with_m1['KC_lower']
        h1_with_m1['kc_strength'] = kc_range / h1_with_m1['ATR'].replace(0, np.nan)
        h1_features_cols.append('kc_strength')
    elif 'kc_strength' in h1_with_m1.columns:
        h1_features_cols.append('kc_strength')

    all_features = M1_FEATURES + h1_features_cols
    available_features = [f for f in all_features if f in h1_with_m1.columns]
    m1_available = [f for f in M1_FEATURES if f in h1_with_m1.columns]
    h1_available = [f for f in h1_features_cols if f in h1_with_m1.columns]

    print(f"  Available features: {len(available_features)} (M1: {len(m1_available)}, H1: {len(h1_available)})", flush=True)

    # Build labeled dataset
    samples = []
    for trade in trades:
        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
        pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)

        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        if entry_time is None:
            continue

        h1_time = pd.Timestamp(entry_time).floor('h')
        if h1_time.tzinfo is None:
            h1_time = h1_time.tz_localize('UTC')

        idx = h1_with_m1.index.get_indexer([h1_time], method='nearest')
        if idx[0] >= 0 and idx[0] < len(h1_with_m1):
            row = h1_with_m1.iloc[idx[0]]
            sample = {feat: row.get(feat, np.nan) for feat in available_features}
            sample['target'] = 1 if pnl > 0 else 0
            sample['pnl'] = pnl
            sample['entry_time'] = entry_time
            sample['year'] = pd.Timestamp(entry_time).year
            samples.append(sample)

    sample_df = pd.DataFrame(samples)
    print(f"  Total labeled samples: {len(sample_df)}", flush=True)
    print(f"  Win rate: {sample_df['target'].mean():.3f}", flush=True)

    if len(sample_df) < 100:
        print("  ERROR: Too few samples for ML training", flush=True)
        save_json({'error': 'too_few_samples', 'n': len(sample_df)}, "phase4_xgb_filter.json")
        return {}

    # Walk-forward: train on years 1..N, test on N+1
    years = sorted(sample_df['year'].unique())
    print(f"  Years: {years}", flush=True)

    filter_configs = {
        'M1_only': m1_available,
        'H1_only': h1_available,
        'H1_M1_combined': available_features,
    }

    wf_results = {config_name: [] for config_name in filter_configs}

    for test_year_idx in range(2, len(years)):
        test_year = years[test_year_idx]
        train_years = years[:test_year_idx]

        train_mask = sample_df['year'].isin(train_years)
        test_mask = sample_df['year'] == test_year
        train_data = sample_df[train_mask]
        test_data = sample_df[test_mask]

        if len(train_data) < 50 or len(test_data) < 20:
            continue

        print(f"\n  Walk-forward: train={train_years} -> test={test_year} "
              f"(train={len(train_data)}, test={len(test_data)})", flush=True)

        for config_name, feature_list in filter_configs.items():
            feat_cols = [f for f in feature_list if f in sample_df.columns]
            if len(feat_cols) < 2:
                continue

            X_train = train_data[feat_cols].fillna(0).values
            y_train = train_data['target'].values
            X_test = test_data[feat_cols].fillna(0).values
            y_test = test_data['target'].values

            # Handle class imbalance
            pos_count = y_train.sum()
            neg_count = len(y_train) - pos_count
            scale_weight = neg_count / max(pos_count, 1)

            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
            )

            try:
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)

                # Filter impact: if we only take predicted-good trades
                filtered_pnl = test_data[y_pred == 1]['pnl'].sum()
                all_pnl = test_data['pnl'].sum()
                n_filtered = y_pred.sum()
                n_total = len(y_pred)

                result = {
                    'test_year': int(test_year),
                    'auc': float(auc),
                    'precision': float(precision),
                    'recall': float(recall),
                    'n_train': int(len(train_data)),
                    'n_test': int(len(test_data)),
                    'n_filtered': int(n_filtered),
                    'n_total': int(n_total),
                    'filtered_pnl': float(filtered_pnl),
                    'all_pnl': float(all_pnl),
                }
                wf_results[config_name].append(result)

                print(f"    {config_name:20s}: AUC={auc:.3f}, Prec={precision:.3f}, "
                      f"Recall={recall:.3f}, Filtered={n_filtered}/{n_total}", flush=True)

            except Exception as e:
                print(f"    {config_name:20s}: ERROR - {e}", flush=True)

    # Summary
    print(f"\n  === Walk-Forward Summary ===", flush=True)
    summary = {}
    for config_name, results in wf_results.items():
        if not results:
            continue
        avg_auc = np.mean([r['auc'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        total_filtered_pnl = sum(r['filtered_pnl'] for r in results)
        total_all_pnl = sum(r['all_pnl'] for r in results)
        summary[config_name] = {
            'avg_auc': float(avg_auc),
            'avg_precision': float(avg_precision),
            'avg_recall': float(avg_recall),
            'total_filtered_pnl': float(total_filtered_pnl),
            'total_all_pnl': float(total_all_pnl),
            'n_years': len(results),
            'yearly_results': results,
        }
        print(f"  {config_name:20s}: Avg AUC={avg_auc:.3f}, Prec={avg_precision:.3f}, "
              f"PnL filtered=${total_filtered_pnl:.0f} vs all=${total_all_pnl:.0f}", flush=True)

    # Determine if M1 adds value
    m1_only_auc = summary.get('M1_only', {}).get('avg_auc', 0.5)
    h1_only_auc = summary.get('H1_only', {}).get('avg_auc', 0.5)
    combined_auc = summary.get('H1_M1_combined', {}).get('avg_auc', 0.5)

    m1_adds_value = combined_auc > h1_only_auc + 0.01
    print(f"\n  M1 adds value? {'YES' if m1_adds_value else 'NO'} "
          f"(Combined AUC {combined_auc:.3f} vs H1-only {h1_only_auc:.3f})", flush=True)

    summary['m1_adds_value'] = m1_adds_value
    summary['m1_auc_improvement'] = float(combined_auc - h1_only_auc)

    save_json(summary, "phase4_xgb_filter.json")
    print(f"\n{elapsed()} Phase 4 complete.", flush=True)
    return summary


# ═══════════════════════════════════════════════════════════════
# Phase 5: Backtest Impact (K-Fold Validation)
# ═══════════════════════════════════════════════════════════════

def phase5_backtest_impact(h1_with_m1: pd.DataFrame, data: DataBundle, phase4_results: Dict):
    """Apply best ML filter to L8_MAX and validate with K-Fold."""
    print(f"\n{'='*70}", flush=True)
    print(f"PHASE 5: Backtest Impact (K-Fold Validation)", flush=True)
    print(f"{'='*70}", flush=True)

    if not HAS_XGB or not HAS_SKLEARN:
        print("  SKIP: Missing XGBoost or sklearn", flush=True)
        save_json({'error': 'missing_dependencies'}, "phase5_backtest_impact.json")
        return

    # Determine best feature set from Phase 4
    best_config = 'H1_M1_combined'
    if phase4_results:
        aucs = {k: v.get('avg_auc', 0) for k, v in phase4_results.items() if isinstance(v, dict) and 'avg_auc' in v}
        if aucs:
            best_config = max(aucs, key=aucs.get)
    print(f"  Using best config: {best_config}", flush=True)

    h1_features_cols = ['ATR', 'ADX', 'RSI14']
    if 'squeeze' in h1_with_m1.columns:
        h1_features_cols.append('squeeze')
    if 'kc_strength' in h1_with_m1.columns:
        h1_features_cols.append('kc_strength')

    if best_config == 'M1_only':
        feature_list = [f for f in M1_FEATURES if f in h1_with_m1.columns]
    elif best_config == 'H1_only':
        feature_list = [f for f in h1_features_cols if f in h1_with_m1.columns]
    else:
        feature_list = [f for f in (M1_FEATURES + h1_features_cols) if f in h1_with_m1.columns]

    print(f"  Feature list ({len(feature_list)}): {feature_list}", flush=True)

    # Run K-Fold validation
    baseline_kwargs = dict(LIVE_PARITY_KWARGS)
    baseline_kwargs['min_lot_size'] = 0.02
    baseline_kwargs['max_lot_size'] = 0.02
    baseline_kwargs['maxloss_cap'] = 35

    fold_results = []
    filtered_better_count = 0

    for fold_name, fold_start, fold_end in FOLDS:
        print(f"\n  --- {fold_name}: {fold_start} to {fold_end} ---", flush=True)

        fold_data = data.slice(fold_start, fold_end)
        if len(fold_data.m15_df) < 1000:
            print(f"    SKIP: insufficient data ({len(fold_data.m15_df)} M15 bars)", flush=True)
            continue

        # Run unfiltered baseline
        stats_baseline = run_variant(fold_data, f"{fold_name}_baseline", verbose=False, **baseline_kwargs)
        trades_baseline = stats_baseline.get('_trades', [])

        # Build ML filter using data BEFORE this fold (walk-forward)
        train_end = pd.Timestamp(fold_start, tz='UTC')
        train_h1 = h1_with_m1[h1_with_m1.index < train_end]

        if len(train_h1) < 200:
            print(f"    SKIP: insufficient training data ({len(train_h1)} H1 bars)", flush=True)
            fold_results.append({
                'fold': fold_name, 'status': 'skip_insufficient_train'
            })
            continue

        # Build training labels from pre-fold trades
        train_data_slice = data.slice("2015-01-01", fold_start)
        if len(train_data_slice.m15_df) < 1000:
            print(f"    SKIP: insufficient train data slice", flush=True)
            continue

        stats_train = run_variant(train_data_slice, f"{fold_name}_train", verbose=False, **baseline_kwargs)
        train_trades = stats_train.get('_trades', [])

        # Build training samples
        train_samples = []
        for trade in train_trades:
            entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
            pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
            if entry_time is None:
                continue

            h1_time = pd.Timestamp(entry_time).floor('h')
            if h1_time.tzinfo is None:
                h1_time = h1_time.tz_localize('UTC')

            idx = h1_with_m1.index.get_indexer([h1_time], method='nearest')
            if idx[0] >= 0 and idx[0] < len(h1_with_m1):
                row = h1_with_m1.iloc[idx[0]]
                sample = {feat: row.get(feat, np.nan) for feat in feature_list}
                sample['target'] = 1 if pnl > 0 else 0
                train_samples.append(sample)

        if len(train_samples) < 50:
            print(f"    SKIP: too few training samples ({len(train_samples)})", flush=True)
            fold_results.append({'fold': fold_name, 'status': 'skip_few_train_samples'})
            continue

        train_df = pd.DataFrame(train_samples)
        X_train = train_df[feature_list].fillna(0).values
        y_train = train_df['target'].values

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_weight = neg_count / max(pos_count, 1)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Apply filter to test-fold trades
        test_trade_features = []
        for trade in trades_baseline:
            entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
            pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
            if entry_time is None:
                continue

            h1_time = pd.Timestamp(entry_time).floor('h')
            if h1_time.tzinfo is None:
                h1_time = h1_time.tz_localize('UTC')

            idx = h1_with_m1.index.get_indexer([h1_time], method='nearest')
            if idx[0] >= 0 and idx[0] < len(h1_with_m1):
                row = h1_with_m1.iloc[idx[0]]
                feat_vals = {feat: row.get(feat, np.nan) for feat in feature_list}
                feat_vals['pnl'] = pnl
                test_trade_features.append(feat_vals)

        if not test_trade_features:
            print(f"    No test trades matched", flush=True)
            fold_results.append({'fold': fold_name, 'status': 'no_test_trades'})
            continue

        test_tf_df = pd.DataFrame(test_trade_features)
        X_test = test_tf_df[feature_list].fillna(0).values
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute filtered stats
        pnls_all = test_tf_df['pnl'].values
        pnls_filtered = pnls_all[y_pred == 1]

        baseline_pnl = pnls_all.sum()
        filtered_pnl = pnls_filtered.sum()
        baseline_n = len(pnls_all)
        filtered_n = len(pnls_filtered)

        baseline_sharpe = (np.mean(pnls_all) / np.std(pnls_all) * np.sqrt(252)) if np.std(pnls_all) > 0 else 0
        filtered_sharpe = (np.mean(pnls_filtered) / np.std(pnls_filtered) * np.sqrt(252)) if len(pnls_filtered) > 1 and np.std(pnls_filtered) > 0 else 0

        # Max drawdown approximation
        def max_dd(pnls):
            if len(pnls) == 0:
                return 0
            cum = np.cumsum(pnls)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            return dd.max()

        baseline_dd = max_dd(pnls_all)
        filtered_dd = max_dd(pnls_filtered)

        filtered_better = filtered_sharpe > baseline_sharpe
        if filtered_better:
            filtered_better_count += 1

        fold_result = {
            'fold': fold_name,
            'status': 'ok',
            'baseline_pnl': float(baseline_pnl),
            'filtered_pnl': float(filtered_pnl),
            'baseline_n': int(baseline_n),
            'filtered_n': int(filtered_n),
            'baseline_sharpe': float(baseline_sharpe),
            'filtered_sharpe': float(filtered_sharpe),
            'baseline_maxdd': float(baseline_dd),
            'filtered_maxdd': float(filtered_dd),
            'filtered_better': bool(filtered_better),
        }
        fold_results.append(fold_result)

        print(f"    Baseline: N={baseline_n}, PnL=${baseline_pnl:.0f}, Sharpe={baseline_sharpe:.2f}, MaxDD=${baseline_dd:.0f}", flush=True)
        print(f"    Filtered: N={filtered_n}, PnL=${filtered_pnl:.0f}, Sharpe={filtered_sharpe:.2f}, MaxDD=${filtered_dd:.0f}", flush=True)
        status = "BETTER" if filtered_better else "worse"
        print(f"    -> {status}", flush=True)

    # Final verdict
    n_valid_folds = sum(1 for r in fold_results if r.get('status') == 'ok')
    pass_criterion = filtered_better_count >= 4

    print(f"\n  === K-Fold Summary ===", flush=True)
    print(f"  Valid folds: {n_valid_folds}/6", flush=True)
    print(f"  Filtered better: {filtered_better_count}/{n_valid_folds}", flush=True)
    print(f"  PASS (>=4/6)? {'YES' if pass_criterion else 'NO'}", flush=True)

    summary = {
        'best_config': best_config,
        'features_used': feature_list,
        'n_valid_folds': n_valid_folds,
        'filtered_better_count': filtered_better_count,
        'pass_criterion': pass_criterion,
        'fold_results': fold_results,
    }
    save_json(summary, "phase5_backtest_impact.json")
    print(f"\n{elapsed()} Phase 5 complete.", flush=True)
    return summary


# ═══════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════

def write_final_summary(phase2_results, phase3_results, phase4_results, phase5_results):
    """Write consolidated summary of all phases."""
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    summary = {
        'experiment': 'R173_M1_Feature_Enhancement',
        'total_runtime_s': round(time.time() - t0, 1),
        'phases': {},
    }

    # Phase 2 highlights
    if phase2_results:
        top_ic = sorted(
            [(k, v) for k, v in phase2_results.items() if not k.startswith('h1_')],
            key=lambda x: abs(x[1].get('ic', 0)), reverse=True
        )[:3]
        summary['phases']['phase2_ic'] = {
            'top_features': [(k, v.get('ic', 0)) for k, v in top_ic],
            'any_significant': any(abs(v.get('ic', 0)) > 0.03 for _, v in phase2_results.items()),
        }

    # Phase 3 highlights
    if phase3_results:
        top_d = sorted(phase3_results.items(), key=lambda x: abs(x[1].get('cohens_d', 0)), reverse=True)[:3]
        summary['phases']['phase3_discrimination'] = {
            'top_features': [(k, v['cohens_d']) for k, v in top_d],
            'any_medium_effect': any(abs(v.get('cohens_d', 0)) > 0.5 for v in phase3_results.values()),
        }

    # Phase 4 highlights
    if phase4_results and isinstance(phase4_results, dict):
        summary['phases']['phase4_ml'] = {
            'm1_adds_value': phase4_results.get('m1_adds_value', False),
            'm1_auc_improvement': phase4_results.get('m1_auc_improvement', 0),
        }

    # Phase 5 highlights
    if phase5_results and isinstance(phase5_results, dict):
        summary['phases']['phase5_kfold'] = {
            'pass_criterion': phase5_results.get('pass_criterion', False),
            'filtered_better_count': phase5_results.get('filtered_better_count', 0),
            'n_valid_folds': phase5_results.get('n_valid_folds', 0),
        }

    # Overall verdict
    m1_useful = False
    if phase4_results and phase4_results.get('m1_adds_value'):
        m1_useful = True
    if phase5_results and phase5_results.get('pass_criterion'):
        m1_useful = True

    summary['verdict'] = {
        'm1_features_useful': m1_useful,
        'recommendation': (
            "M1 features add value to entry filtering. Consider integration into live system."
            if m1_useful else
            "M1 features do NOT significantly improve H1 entry quality. "
            "Stick with H1-level features. M1 data overhead not justified."
        ),
    }

    save_json(summary, "final_summary.json")

    print(f"\n  Overall Verdict: M1 features {'USEFUL' if m1_useful else 'NOT USEFUL'}", flush=True)
    print(f"  Recommendation: {summary['verdict']['recommendation']}", flush=True)
    print(f"\n  Total runtime: {time.time() - t0:.0f}s ({(time.time() - t0)/3600:.1f} hours)", flush=True)
    print(f"  All results saved to: {OUTPUT_DIR}/", flush=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"R173 — M1 Microstructure Feature Enhancement for H1 Entry", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Started: {pd.Timestamp.now()}", flush=True)
    print(f"Output: {OUTPUT_DIR}/", flush=True)
    print(f"M1 data: {M1_DATA_DIR}/", flush=True)

    # Phase 1
    h1_with_m1, data = phase1_m1_features()
    if h1_with_m1 is None or data is None:
        print("\nFATAL: Phase 1 failed. Cannot continue.", flush=True)
        return

    # Phase 2
    phase2_results = phase2_ic_analysis(h1_with_m1)

    # Phase 3
    phase3_results = phase3_entry_quality(h1_with_m1, data)

    # Phase 4
    phase4_results = phase4_xgb_filter(h1_with_m1, data)

    # Phase 5
    phase5_results = phase5_backtest_impact(h1_with_m1, data, phase4_results)

    # Final
    write_final_summary(phase2_results, phase3_results, phase4_results, phase5_results)

    print(f"\n{'='*70}", flush=True)
    print(f"R173 COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
