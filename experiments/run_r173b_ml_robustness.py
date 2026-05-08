#!/usr/bin/env python3
"""
R173b — M1+H1 Entry Filter: Robustness Suite + Threshold Sweep + Pipeline Check
================================================================================

Adapts R173 Phase 4 ML filter (M1 microstructure + H1 baseline features for L8_MAX
Keltner entry filtering) and subjects it to the 5 mandatory robustness tests
per .cursor/rules/ml-robustness-tests.md.

Additionally:
  - Threshold sweep (à la R98): test [0.40 .. 0.70] with K-Fold validation.
  - M1 data pipeline verification: validates the chain
    MT4 → CSV → 60-bar rolling → feature compute → model feed.

Feature set (up to 15 dims):
  M1: m1_range_ratio, m1_acceleration, m1_momentum_15, m1_trend_consistency,
      m1_reversal_count, m1_late_surge, m1_opening_gap, m1_body_vs_wick,
      m1_vwap_dist, m1_volume_profile
  H1: ATR, ADX, RSI14, squeeze, kc_strength

Strategy: L8_MAX (Keltner) only.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

from backtest.runner import (
    DataBundle, run_variant, LIVE_PARITY_KWARGS,
)

OUTPUT_DIR = Path('results/r173b_ml_robustness')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

PV = 100
SPREAD = 0.30

HOLDOUT_SPLIT = '2023-01-01'
N_SHUFFLE = 20
RNG = np.random.RandomState(42)

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

H1_FEATURES = ['ATR', 'ADX', 'RSI14', 'squeeze', 'kc_strength']

ALL_FEATURES_R173 = M1_FEATURES + H1_FEATURES

XGB_BASE_PARAMS = {
    'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'use_label_encoder': False,
    'eval_metric': 'logloss', 'verbosity': 0,
}

PARAM_GRID = [
    {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100},
    {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 150},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 100},
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200},
    {'max_depth': 4, 'learning_rate': 0.08, 'n_estimators': 100},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200},
    {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 100},
]

THRESHOLD_SWEEP = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]

t0 = time.time()


def elapsed():
    return f"[{time.time() - t0:.0f}s]"


def save_json(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}", flush=True)


# ═══════════════════════════════════════════════════════════════
# M1 Data Loading & Feature Engineering
# ═══════════════════════════════════════════════════════════════

def load_m1_dukascopy(filepath: str) -> pd.DataFrame:
    """Load dukascopy-format M1 CSV (comma-sep, 'Gmt time' col, DD.MM.YYYY format)."""
    df = pd.read_csv(filepath)
    time_col = 'Gmt time' if 'Gmt time' in df.columns else df.columns[0]
    df.rename(columns={time_col: 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S.%f',
                                    utc=True, dayfirst=True)
    df.set_index('datetime', inplace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    return df


def load_m1_semicolon(filepath: str) -> pd.DataFrame:
    """Load semicolon-separated M1 CSV (server format, no header)."""
    df = pd.read_csv(filepath, sep=';', header=None,
                     names=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S', utc=True)
    df.set_index('datetime', inplace=True)
    df['Volume'] = df['Volume'].fillna(0).astype(float)
    return df


def find_and_load_m1() -> pd.DataFrame:
    """Find M1 data from either dukascopy download or m1_raw folder."""
    # Option 1: dukascopy download (single large CSV)
    dukascopy_files = sorted(glob.glob("data/download/xauusd-m1-bid-*.csv"))
    if dukascopy_files:
        filepath = dukascopy_files[-1]
        print(f"  Loading M1 from dukascopy: {filepath}", flush=True)
        return load_m1_dukascopy(filepath)

    # Option 2: m1_raw folder (server-style semicolon CSVs)
    raw_files = sorted(glob.glob("data/m1_raw/xauusd_*.csv"))
    if raw_files:
        print(f"  Loading {len(raw_files)} M1 files from data/m1_raw/", flush=True)
        dfs = []
        for f in raw_files:
            dfs.append(load_m1_semicolon(f))
        return pd.concat(dfs).sort_index()

    raise FileNotFoundError("No M1 data found in data/download/ or data/m1_raw/")


def compute_m1_features_for_h1(m1_group: pd.DataFrame, prev_h1_close: float) -> Dict:
    """Compute 10 microstructure features for M1 bars within one H1 bar."""
    n = len(m1_group)
    if n < 5:
        return {feat: np.nan for feat in M1_FEATURES}

    opens = m1_group['Open'].values
    highs = m1_group['High'].values
    lows = m1_group['Low'].values
    closes = m1_group['Close'].values
    volumes = m1_group['Volume'].values

    h1_high = highs.max()
    h1_low = lows.min()
    h1_range = h1_high - h1_low
    h1_open = opens[0]
    h1_close = closes[-1]

    features: Dict = {}

    m1_ranges = highs - lows
    features['m1_range_ratio'] = float(m1_ranges.max() / h1_range) if h1_range > 1e-6 else 0.0

    if n >= 3:
        d1 = np.diff(closes)
        d2 = np.diff(d1)
        features['m1_acceleration'] = float(d2[-1]) if len(d2) > 0 else 0.0
    else:
        features['m1_acceleration'] = 0.0

    last_15 = min(15, n)
    features['m1_momentum_15'] = float(closes[-1] - closes[-last_15])

    h1_direction = np.sign(h1_close - h1_open)
    if h1_direction == 0:
        features['m1_trend_consistency'] = 0.5
    else:
        bar_directions = np.sign(closes - opens)
        features['m1_trend_consistency'] = float(np.mean(bar_directions == h1_direction))

    signs = np.sign(np.diff(closes))
    signs_nonzero = signs[signs != 0]
    features['m1_reversal_count'] = int(np.sum(np.diff(signs_nonzero) != 0)) if len(signs_nonzero) > 1 else 0

    split_idx = max(1, int(n * 0.75))
    late_range = highs[split_idx:].max() - lows[split_idx:].min() if split_idx < n else 0
    early_range = highs[:split_idx].max() - lows[:split_idx].min() if split_idx > 0 else 1e-6
    features['m1_late_surge'] = float(late_range / max(early_range, 1e-6))

    features['m1_opening_gap'] = float(abs(opens[0] - prev_h1_close)) if prev_h1_close > 0 else 0.0

    features['m1_body_vs_wick'] = float(abs(h1_close - h1_open) / h1_range) if h1_range > 1e-6 else 0.0

    total_vol = volumes.sum()
    if total_vol > 0:
        typical_price = (highs + lows + closes) / 3.0
        vwap = np.sum(typical_price * volumes) / total_vol
        features['m1_vwap_dist'] = float((h1_close - vwap) / max(h1_range, 1e-6))
    else:
        features['m1_vwap_dist'] = 0.0

    if total_vol > 0:
        late_vol = volumes[split_idx:].sum()
        features['m1_volume_profile'] = float(late_vol / total_vol)
    else:
        features['m1_volume_profile'] = 0.5

    return features


def build_m1_features_aligned(m1_all: pd.DataFrame, h1_df: pd.DataFrame) -> pd.DataFrame:
    """Build M1 features aligned to H1 bar timestamps."""
    print(f"  {elapsed()} Building M1 features for {len(h1_df)} H1 bars...", flush=True)

    m1_all['h1_bar'] = m1_all.index.floor('h')
    h1_times = h1_df.index
    records = []
    prev_close = 0.0
    total = len(h1_times)

    for i, h1_time in enumerate(h1_times):
        m1_group = m1_all[m1_all['h1_bar'] == h1_time]
        feats = compute_m1_features_for_h1(m1_group, prev_close)
        feats['h1_time'] = h1_time
        records.append(feats)
        if len(m1_group) > 0:
            prev_close = m1_group['Close'].iloc[-1]
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{total} ({(i+1)/total*100:.0f}%)", flush=True)

    feat_df = pd.DataFrame(records).set_index('h1_time')
    if feat_df.index.tz is None:
        feat_df.index = feat_df.index.tz_localize('UTC')

    coverage = feat_df[M1_FEATURES[0]].notna().mean()
    print(f"  M1 feature coverage: {coverage:.1%}", flush=True)
    return feat_df


def prepare_h1_with_m1(data: DataBundle, m1_all: pd.DataFrame) -> pd.DataFrame:
    """Merge M1 features onto H1 DataFrame and add derived H1 features."""
    h1_df = data.h1_df.copy()

    # Add squeeze and kc_strength from existing indicators
    if 'squeeze' not in h1_df.columns:
        if 'KC_upper' in h1_df.columns and 'KC_lower' in h1_df.columns:
            h1_df['squeeze'] = (h1_df['KC_upper'] - h1_df['KC_lower']) / h1_df['ATR'].replace(0, np.nan)
        else:
            h1_df['squeeze'] = 0.0

    if 'kc_strength' not in h1_df.columns:
        if 'KC_upper' in h1_df.columns and 'KC_lower' in h1_df.columns:
            kc_range = h1_df['KC_upper'] - h1_df['KC_lower']
            kc_mid = (h1_df['KC_upper'] + h1_df['KC_lower']) / 2
            h1_df['kc_strength'] = (h1_df['Close'] - kc_mid) / kc_range.replace(0, np.nan)
        else:
            h1_df['kc_strength'] = 0.0

    # Compute RSI14 if not present
    if 'RSI14' not in h1_df.columns:
        delta = h1_df['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        h1_df['RSI14'] = 100 - 100 / (1 + rs)

    # Build M1 features
    m1_feat = build_m1_features_aligned(m1_all, h1_df)

    # Merge
    h1_with_m1 = h1_df.join(m1_feat, how='left')
    return h1_with_m1


def build_labeled_dataset(trades: list, h1_with_m1: pd.DataFrame,
                          feature_cols: List[str]) -> pd.DataFrame:
    """Build labeled feature dataset from L8_MAX trades."""
    samples = []
    for trade in trades:
        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
        pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
        exit_time = trade.exit_time if hasattr(trade, 'exit_time') else trade.get('exit_time', None)

        if entry_time is None:
            continue
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)

        h1_time = pd.Timestamp(entry_time).floor('h')
        if h1_time.tzinfo is None:
            h1_time = h1_time.tz_localize('UTC')

        idx = h1_with_m1.index.get_indexer([h1_time], method='nearest')
        if idx[0] < 0 or idx[0] >= len(h1_with_m1):
            continue

        row = h1_with_m1.iloc[idx[0]]
        sample = {feat: row.get(feat, np.nan) for feat in feature_cols if feat in row.index}
        sample['label'] = 1 if pnl > 0 else 0
        sample['pnl'] = pnl
        sample['entry_time'] = entry_time
        sample['exit_time'] = exit_time
        samples.append(sample)

    return pd.DataFrame(samples)


# ═══════════════════════════════════════════════════════════════
# ML Utilities
# ═══════════════════════════════════════════════════════════════

def make_xgb(**overrides) -> XGBClassifier:
    kw = {**XGB_BASE_PARAMS, **overrides}
    return XGBClassifier(**kw)


def _median_impute(train: pd.DataFrame, test: pd.DataFrame):
    med = train.median()
    Xt = train.fillna(med)
    Xv = test.fillna(med)
    const = [c for c in Xt.columns if Xt[c].nunique() <= 1]
    if const:
        Xt = Xt.drop(columns=const)
        Xv = Xv.drop(columns=const)
    return Xt, Xv


def train_walkforward_oos(X: pd.DataFrame, y: np.ndarray,
                          entry_times: pd.Series, model) -> dict:
    """Expanding-window walk-forward, returns overall + per-fold AUC."""
    oos_pred = np.full(len(y), np.nan)
    fold_aucs: List[float] = []

    for fold_name, fold_start, fold_end in FOLDS:
        fs = pd.Timestamp(fold_start, tz='UTC')
        fe = pd.Timestamp(fold_end, tz='UTC')
        train_m = entry_times < fs
        test_m = (entry_times >= fs) & (entry_times < fe)
        if train_m.sum() < 40 or test_m.sum() < 8:
            continue
        X_tr, X_te = _median_impute(X.loc[train_m], X.loc[test_m])
        y_tr, y_te = y[train_m.values], y[test_m.values]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        if X_tr.shape[1] == 0:
            continue
        try:
            m = deepcopy(model)
            m.fit(X_tr.values, y_tr)
            prob = m.predict_proba(X_te.values)[:, 1]
            idx = np.where(test_m.values)[0]
            oos_pred[idx] = prob
            fold_aucs.append(float(roc_auc_score(y_te, prob)))
        except Exception:
            continue

    valid = ~np.isnan(oos_pred)
    overall = None
    if valid.sum() > 20 and len(np.unique(y[valid])) > 1:
        overall = float(roc_auc_score(y[valid], oos_pred[valid]))
    return {'overall_auc': overall, 'fold_aucs': fold_aucs, 'oos_pred': oos_pred}


def _sharpe(daily: np.ndarray) -> float:
    if len(daily) < 10 or np.std(daily, ddof=1) == 0:
        return 0.0
    return float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252))


def _trades_to_daily(trades_pnl: List[Tuple]) -> np.ndarray:
    """From list of (pnl, exit_time) tuples compute daily PnL array."""
    if not trades_pnl:
        return np.array([0.0])
    daily: Dict = {}
    for pnl, exit_t in trades_pnl:
        d = pd.Timestamp(exit_t).date()
        daily[d] = daily.get(d, 0) + pnl
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


# ═══════════════════════════════════════════════════════════════
# 5 Robustness Tests
# ═══════════════════════════════════════════════════════════════

def test1_holdout(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series,
                  feat_df: pd.DataFrame) -> dict:
    """Test 1: Independent Holdout (train <2023, test >=2023)."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  TEST 1: Independent Holdout", flush=True)
    print(f"  {'='*60}", flush=True)

    split_ts = pd.Timestamp(HOLDOUT_SPLIT, tz='UTC')
    train_m = entry_times < split_ts
    test_m = entry_times >= split_ts

    if train_m.sum() < 40 or test_m.sum() < 15:
        print(f"    SKIP: insufficient split (train={train_m.sum()}, test={test_m.sum()})", flush=True)
        return {'passed': False, 'reason': 'insufficient_split'}

    X_tr, X_te = _median_impute(X.loc[train_m], X.loc[test_m])
    y_tr, y_te = y[train_m.values], y[test_m.values]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or X_tr.shape[1] == 0:
        return {'passed': False, 'reason': 'degenerate_split'}

    model = make_xgb()
    model.fit(X_tr.values, y_tr)
    prob = model.predict_proba(X_te.values)[:, 1]
    auc = float(roc_auc_score(y_te, prob))

    # Sharpe comparison
    test_df = feat_df.loc[test_m].reset_index(drop=True)
    baseline_data = [(row['pnl'], row['exit_time']) for _, row in test_df.iterrows()]
    filtered_data = [(test_df.iloc[i]['pnl'], test_df.iloc[i]['exit_time'])
                     for i in range(len(prob)) if prob[i] >= 0.5]

    b_sh = _sharpe(_trades_to_daily(baseline_data))
    f_sh = _sharpe(_trades_to_daily(filtered_data))

    passed = auc > 0.65 and f_sh > b_sh
    print(f"    AUC = {auc:.4f} (need > 0.65)", flush=True)
    print(f"    Baseline Sharpe = {b_sh:.3f}, Filtered Sharpe = {f_sh:.3f}", flush=True)
    print(f"    >>> {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        'passed': passed,
        'holdout_auc': round(auc, 4),
        'baseline_sharpe': round(b_sh, 4),
        'filtered_sharpe': round(f_sh, 4),
        'n_train': int(train_m.sum()),
        'n_test': int(test_m.sum()),
    }


def test2_param_perturb(X: pd.DataFrame, y: np.ndarray,
                        entry_times: pd.Series) -> dict:
    """Test 2: Parameter Perturbation (≥8 variants, std < 0.05)."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  TEST 2: Parameter Perturbation ({len(PARAM_GRID)} variants)", flush=True)
    print(f"  {'='*60}", flush=True)

    aucs = []
    for i, params in enumerate(PARAM_GRID):
        model = make_xgb(**params)
        r = train_walkforward_oos(X, y, entry_times, model)
        if r['overall_auc'] is not None:
            aucs.append(r['overall_auc'])
            print(f"    [{i+1}] depth={params.get('max_depth')}, lr={params.get('learning_rate')}, "
                  f"n={params.get('n_estimators')} -> AUC={r['overall_auc']:.4f}", flush=True)

    if len(aucs) < 4:
        print(f"    SKIP: insufficient variants ({len(aucs)})", flush=True)
        return {'passed': False, 'reason': 'insufficient_variants'}

    std = float(np.std(aucs))
    passed = std < 0.05
    print(f"    std(AUC) = {std:.4f} (need < 0.05) -> {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        'passed': passed,
        'std_auc': round(std, 4),
        'mean_auc': round(float(np.mean(aucs)), 4),
        'min_auc': round(float(np.min(aucs)), 4),
        'n_variants': len(aucs),
        'all_aucs': [round(a, 4) for a in aucs],
    }


def test3_shuffle(X: pd.DataFrame, y: np.ndarray,
                  entry_times: pd.Series) -> dict:
    """Test 3: Random Label Shuffle (20 perms, z > 3, p < 0.05)."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  TEST 3: Random Label Shuffle ({N_SHUFFLE} permutations)", flush=True)
    print(f"  {'='*60}", flush=True)

    real = train_walkforward_oos(X, y, entry_times, make_xgb())
    real_auc = real['overall_auc']
    if real_auc is None:
        return {'passed': False, 'reason': 'no_real_auc'}
    print(f"    Real AUC = {real_auc:.4f}", flush=True)

    shuffled = []
    for i in range(N_SHUFFLE):
        y_perm = RNG.permutation(y)
        r = train_walkforward_oos(X, y_perm, entry_times, make_xgb())
        if r['overall_auc'] is not None:
            shuffled.append(r['overall_auc'])
        if (i + 1) % 5 == 0:
            print(f"      shuffle {i+1}/{N_SHUFFLE}...", flush=True)

    if len(shuffled) < 8:
        return {'passed': False, 'reason': 'insufficient_shuffle_aucs'}

    mu, sd = float(np.mean(shuffled)), float(np.std(shuffled))
    z = (real_auc - mu) / sd if sd > 1e-12 else 0.0
    p_emp = sum(1 for s in shuffled if s >= real_auc) / len(shuffled)
    passed = z > 3.0 and p_emp < 0.05

    print(f"    Shuffle mean={mu:.4f}, std={sd:.4f}", flush=True)
    print(f"    z={z:.2f}, p={p_emp:.4f} -> {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        'passed': passed,
        'real_auc': round(real_auc, 4),
        'shuffle_mean': round(mu, 4),
        'shuffle_std': round(sd, 4),
        'z_score': round(z, 2),
        'p_value': round(p_emp, 4),
    }


def test4_ablation(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series,
                   feature_cols: List[str]) -> dict:
    """Test 4: Feature Ablation (M1-only / H1-only / full)."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  TEST 4: Feature Ablation", flush=True)
    print(f"  {'='*60}", flush=True)

    groups = {
        'm1_only': [c for c in M1_FEATURES if c in feature_cols],
        'h1_only': [c for c in H1_FEATURES if c in feature_cols],
        'full': feature_cols,
    }
    out = {}
    for name, cols in groups.items():
        if not cols:
            out[name] = None
            print(f"    {name}: NO FEATURES AVAILABLE", flush=True)
            continue
        X_sub = X[cols].copy()
        model = make_xgb()
        r = train_walkforward_oos(X_sub, y, entry_times, model)
        out[name] = r['overall_auc']
        auc_str = f"{r['overall_auc']:.4f}" if r['overall_auc'] else "N/A"
        print(f"    {name:12s}: AUC={auc_str} ({len(cols)} features)", flush=True)

    full_a = out.get('full')
    h1_a = out.get('h1_only')
    m1_a = out.get('m1_only')

    # R173 specific: full (M1+H1) should beat H1-only (M1 adds value)
    m1_adds_value = full_a is not None and h1_a is not None and full_a > h1_a
    # h1-only should not be near-random
    h1_informative = h1_a is not None and h1_a > 0.55

    passed = m1_adds_value and h1_informative
    print(f"    M1 adds value? {'YES' if m1_adds_value else 'NO'} "
          f"(full={full_a}, h1_only={h1_a})", flush=True)
    print(f"    >>> {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        'passed': passed,
        'full_auc': round(full_a, 4) if full_a else None,
        'h1_only_auc': round(h1_a, 4) if h1_a else None,
        'm1_only_auc': round(m1_a, 4) if m1_a else None,
        'm1_adds_value': m1_adds_value,
        'h1_informative': h1_informative,
    }


def test5_fold_stability(X: pd.DataFrame, y: np.ndarray,
                         entry_times: pd.Series) -> dict:
    """Test 5: Per-Fold AUC Stability (min > 0.58, CV < 0.20)."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  TEST 5: Per-Fold AUC Stability", flush=True)
    print(f"  {'='*60}", flush=True)

    r = train_walkforward_oos(X, y, entry_times, make_xgb())
    folds = r['fold_aucs']

    if len(folds) < 3:
        print(f"    Only {len(folds)} folds", flush=True)
        return {'passed': False, 'reason': f'only_{len(folds)}_folds', 'fold_aucs': folds}

    for i, auc in enumerate(folds):
        print(f"    Fold {i+1}: AUC = {auc:.4f}", flush=True)

    mean_a = float(np.mean(folds))
    std_a = float(np.std(folds))
    cv = std_a / mean_a if mean_a > 0 else 999.0
    min_a = float(np.min(folds))

    passed = min_a > 0.58 and cv < 0.20
    print(f"    mean={mean_a:.4f}, std={std_a:.4f}, CV={cv:.4f}, min={min_a:.4f}", flush=True)
    print(f"    >>> {'PASS' if passed else 'FAIL'}", flush=True)

    return {
        'passed': passed,
        'fold_aucs': [round(x, 4) for x in folds],
        'mean_auc': round(mean_a, 4),
        'cv': round(cv, 4),
        'min_auc': round(min_a, 4),
    }


# ═══════════════════════════════════════════════════════════════
# Threshold Sweep (R98-style)
# ═══════════════════════════════════════════════════════════════

def threshold_sweep(X: pd.DataFrame, y: np.ndarray, entry_times: pd.Series,
                    feat_df: pd.DataFrame) -> dict:
    """Sweep thresholds and K-Fold validate the best."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  THRESHOLD SWEEP: {THRESHOLD_SWEEP}", flush=True)
    print(f"  {'='*60}", flush=True)

    # Get full OOS predictions via walk-forward
    r = train_walkforward_oos(X, y, entry_times, make_xgb())
    oos_pred = r['oos_pred']
    valid_mask = ~np.isnan(oos_pred)

    if valid_mask.sum() < 50:
        print(f"    Not enough OOS predictions ({valid_mask.sum()})", flush=True)
        return {'status': 'insufficient_oos'}

    valid_df = feat_df[valid_mask].reset_index(drop=True)
    valid_probs = oos_pred[valid_mask]

    results = {}
    print(f"\n    {'Threshold':>10}  {'N_pass':>7}  {'Filter%':>8}  {'Sharpe':>8}  {'PnL':>10}", flush=True)

    for thr in THRESHOLD_SWEEP:
        mask = valid_probs >= thr
        n_pass = mask.sum()
        filter_rate = round((1 - n_pass / len(valid_probs)) * 100, 1) if len(valid_probs) > 0 else 0
        passed_df = valid_df[mask]
        if len(passed_df) > 0:
            trades_data = [(row['pnl'], row['exit_time']) for _, row in passed_df.iterrows()]
            daily = _trades_to_daily(trades_data)
            sh = _sharpe(daily)
            pnl = float(passed_df['pnl'].sum())
        else:
            sh, pnl = 0.0, 0.0

        results[str(thr)] = {
            'threshold': thr,
            'n_pass': int(n_pass),
            'filter_rate': filter_rate,
            'sharpe': round(sh, 3),
            'pnl': round(pnl, 1),
        }
        print(f"    {thr:>10.2f}  {n_pass:>7}  {filter_rate:>7.1f}%  {sh:>8.3f}  ${pnl:>9.0f}", flush=True)

    # Baseline (no filter)
    baseline_data = [(row['pnl'], row['exit_time']) for _, row in valid_df.iterrows()]
    baseline_sharpe = _sharpe(_trades_to_daily(baseline_data))
    baseline_pnl = float(valid_df['pnl'].sum())
    print(f"    {'baseline':>10}  {len(valid_df):>7}  {'0.0':>8}%  {baseline_sharpe:>8.3f}  ${baseline_pnl:>9.0f}", flush=True)

    # Find peak
    ranked = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    best_thr = float(ranked[0][0]) if ranked else 0.50
    best_sharpe = ranked[0][1]['sharpe'] if ranked else 0.0

    # K-Fold on best 3 thresholds
    best_3 = [float(k) for k, _ in ranked[:3]]
    print(f"\n    K-Fold validation on best 3: {best_3}", flush=True)

    kfold_results = {}
    for thr in best_3:
        fold_sharpes = []
        for fold_name, fold_start, fold_end in FOLDS:
            fs = pd.Timestamp(fold_start, tz='UTC')
            fe = pd.Timestamp(fold_end, tz='UTC')
            fold_m = (entry_times >= fs) & (entry_times < fe) & valid_mask
            if fold_m.sum() < 10:
                continue
            fold_df = feat_df[fold_m].reset_index(drop=True)
            fold_probs = oos_pred[fold_m.values]
            f_mask = fold_probs >= thr
            if f_mask.sum() < 3:
                continue
            f_data = [(fold_df.iloc[i]['pnl'], fold_df.iloc[i]['exit_time'])
                      for i in range(len(fold_df)) if f_mask[i]]
            fold_sharpes.append(_sharpe(_trades_to_daily(f_data)))

        positive = sum(1 for s in fold_sharpes if s > 0)
        kfold_pass = positive >= 4
        kfold_results[str(thr)] = {
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': positive,
            'pass_4of6': kfold_pass,
        }
        print(f"      thr={thr:.2f}: {positive}/{len(fold_sharpes)} folds positive -> "
              f"{'PASS' if kfold_pass else 'FAIL'}", flush=True)

    return {
        'sweep_results': results,
        'baseline_sharpe': round(baseline_sharpe, 3),
        'best_threshold': best_thr,
        'best_sharpe': best_sharpe,
        'kfold_validation': kfold_results,
    }


# ═══════════════════════════════════════════════════════════════
# M1 Pipeline Verification
# ═══════════════════════════════════════════════════════════════

def verify_m1_pipeline(m1_all: pd.DataFrame, h1_with_m1: pd.DataFrame) -> dict:
    """Verify M1 data pipeline integrity for live deployment."""
    print(f"\n  {'='*60}", flush=True)
    print(f"  M1 DATA PIPELINE VERIFICATION", flush=True)
    print(f"  {'='*60}", flush=True)

    checks = {}

    # Check 1: M1 bar count per H1 bar (expect ~60)
    m1_all_copy = m1_all.copy()
    m1_all_copy['h1_bar'] = m1_all_copy.index.floor('h')
    bars_per_h1 = m1_all_copy.groupby('h1_bar').size()
    median_bars = int(bars_per_h1.median())
    pct_full = float((bars_per_h1 >= 55).mean())
    checks['bars_per_h1'] = {
        'median': median_bars,
        'pct_full_bars_gte55': round(pct_full, 3),
        'pass': median_bars >= 55 and pct_full > 0.80,
    }
    print(f"    [1] Bars per H1: median={median_bars}, "
          f"≥55 bars in {pct_full:.1%} of hours -> "
          f"{'PASS' if checks['bars_per_h1']['pass'] else 'FAIL'}", flush=True)

    # Check 2: M1 data continuity (gaps)
    m1_index = m1_all.index
    diffs = m1_index[1:] - m1_index[:-1]
    expected_gap = pd.Timedelta(minutes=1)
    large_gaps = (diffs > pd.Timedelta(minutes=5)).sum()
    weekend_gaps = (diffs > pd.Timedelta(hours=24)).sum()
    checks['continuity'] = {
        'total_bars': len(m1_all),
        'large_gaps_gt5min': int(large_gaps),
        'weekend_gaps_gt24h': int(weekend_gaps),
        'pass': large_gaps < len(m1_all) * 0.01,
    }
    print(f"    [2] Continuity: {len(m1_all):,} bars, "
          f"gaps>5min={large_gaps}, weekends={weekend_gaps} -> "
          f"{'PASS' if checks['continuity']['pass'] else 'FAIL'}", flush=True)

    # Check 3: Feature coverage (M1 features not NaN)
    m1_coverage = {}
    for feat in M1_FEATURES:
        if feat in h1_with_m1.columns:
            cov = float(h1_with_m1[feat].notna().mean())
            m1_coverage[feat] = round(cov, 3)
    avg_coverage = np.mean(list(m1_coverage.values())) if m1_coverage else 0
    checks['feature_coverage'] = {
        'per_feature': m1_coverage,
        'avg_coverage': round(avg_coverage, 3),
        'pass': avg_coverage > 0.85,
    }
    print(f"    [3] Feature coverage: avg={avg_coverage:.1%} -> "
          f"{'PASS' if checks['feature_coverage']['pass'] else 'FAIL'}", flush=True)

    # Check 4: Feature value ranges (no infinities, reasonable bounds)
    range_ok = True
    for feat in M1_FEATURES:
        if feat in h1_with_m1.columns:
            vals = h1_with_m1[feat].dropna()
            if vals.isin([np.inf, -np.inf]).any():
                range_ok = False
                break
    checks['value_ranges'] = {'pass': range_ok}
    print(f"    [4] Value ranges (no inf): -> {'PASS' if range_ok else 'FAIL'}", flush=True)

    # Check 5: Live-mode simulation (rolling 60-bar window computation)
    # Simulate computing features for the last 100 H1 bars as if real-time
    last_100_h1 = h1_with_m1.tail(100)
    computed_ok = 0
    for h1_time in last_100_h1.index[-20:]:
        m1_group = m1_all_copy[m1_all_copy['h1_bar'] == h1_time]
        if len(m1_group) >= 5:
            feats = compute_m1_features_for_h1(m1_group, 0.0)
            if not any(np.isnan(v) for v in feats.values()):
                computed_ok += 1
    checks['live_simulation'] = {
        'tested_bars': 20,
        'computed_ok': computed_ok,
        'pass': computed_ok >= 15,
    }
    print(f"    [5] Live simulation (20 bars): {computed_ok}/20 computed OK -> "
          f"{'PASS' if checks['live_simulation']['pass'] else 'FAIL'}", flush=True)

    n_pass = sum(1 for c in checks.values() if c.get('pass'))
    total = len(checks)
    print(f"\n    Pipeline verdict: {n_pass}/{total} checks passed", flush=True)

    return {
        'checks': checks,
        'n_pass': n_pass,
        'n_total': total,
        'pipeline_ok': n_pass >= 4,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print('=' * 70, flush=True)
    print('  R173b — M1+H1 Entry Filter: Robustness + Threshold + Pipeline', flush=True)
    print('=' * 70, flush=True)

    # Load M1 data
    print(f"\n{elapsed()} Loading M1 data...", flush=True)
    m1_all = find_and_load_m1()
    print(f"  M1: {len(m1_all):,} bars ({m1_all.index[0]} -> {m1_all.index[-1]})", flush=True)
    print(f"  Memory: ~{m1_all.memory_usage(deep=True).sum() / 1e6:.0f} MB", flush=True)

    # Load H1 data via DataBundle
    print(f"\n{elapsed()} Loading H1 data (DataBundle)...", flush=True)
    data = DataBundle.load_default()
    h1_df = data.h1_df
    print(f"  H1: {len(h1_df)} bars", flush=True)

    # Build merged H1+M1 features
    print(f"\n{elapsed()} Building M1 features & merging...", flush=True)
    h1_with_m1 = prepare_h1_with_m1(data, m1_all)

    # Determine available features
    available = [f for f in ALL_FEATURES_R173 if f in h1_with_m1.columns]
    print(f"  Available features: {len(available)} / {len(ALL_FEATURES_R173)}", flush=True)
    print(f"    M1: {[f for f in M1_FEATURES if f in available]}", flush=True)
    print(f"    H1: {[f for f in H1_FEATURES if f in available]}", flush=True)

    # Run L8_MAX to get trades
    print(f"\n{elapsed()} Running L8_MAX backtest...", flush=True)
    baseline_kwargs = dict(LIVE_PARITY_KWARGS)
    baseline_kwargs['min_lot_size'] = 0.02
    baseline_kwargs['max_lot_size'] = 0.02
    baseline_kwargs['maxloss_cap'] = 35

    stats = run_variant(data, "R173b_L8MAX", **baseline_kwargs)
    trades = stats.get('_trades', [])
    print(f"  L8_MAX: {len(trades)} trades, Sharpe={stats.get('sharpe', 0):.2f}, "
          f"PnL=${stats.get('total_pnl', 0):.0f}", flush=True)

    # Build labeled feature matrix
    print(f"\n{elapsed()} Building labeled dataset...", flush=True)
    feat_df = build_labeled_dataset(trades, h1_with_m1, available)
    print(f"  Samples: {len(feat_df)}, win_rate={feat_df['label'].mean()*100:.1f}%", flush=True)

    if len(feat_df) < 100:
        print("  ERROR: Too few samples, aborting.", flush=True)
        return

    # Prepare X, y, entry_times
    feature_cols = [c for c in available if c in feat_df.columns]
    X = feat_df[feature_cols].copy()
    y = feat_df['label'].values
    entry_times = pd.to_datetime(feat_df['entry_time'])
    if entry_times.dt.tz is None:
        entry_times = entry_times.dt.tz_localize('UTC')

    # ══════════════════════════════════════
    # Part 1: 5 Robustness Tests
    # ══════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"  PART 1: 5 MANDATORY ROBUSTNESS TESTS", flush=True)
    print(f"{'='*70}", flush=True)

    t1 = test1_holdout(X, y, entry_times, feat_df)
    t2 = test2_param_perturb(X, y, entry_times)
    t3 = test3_shuffle(X, y, entry_times)
    t4 = test4_ablation(X, y, entry_times, feature_cols)
    t5 = test5_fold_stability(X, y, entry_times)

    tests = {'test1_holdout': t1, 'test2_param': t2, 'test3_shuffle': t3,
             'test4_ablation': t4, 'test5_fold': t5}
    n_pass = sum(1 for v in tests.values() if v.get('passed'))
    deploy_band = 'robust' if n_pass >= 4 else ('caution' if n_pass == 3 else 'do_not_deploy')

    # ══════════════════════════════════════
    # Part 2: Threshold Sweep
    # ══════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"  PART 2: THRESHOLD SWEEP", flush=True)
    print(f"{'='*70}", flush=True)

    sweep = threshold_sweep(X, y, entry_times, feat_df)

    # ══════════════════════════════════════
    # Part 3: M1 Pipeline Verification
    # ══════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print(f"  PART 3: M1 PIPELINE VERIFICATION", flush=True)
    print(f"{'='*70}", flush=True)

    pipeline = verify_m1_pipeline(m1_all, h1_with_m1)

    # ══════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════
    elapsed_s = round(time.time() - t0, 1)

    summary = {
        'elapsed_s': elapsed_s,
        'strategy': 'L8_MAX',
        'n_trades': len(trades),
        'n_samples': len(feat_df),
        'features_used': feature_cols,
        'robustness_tests': tests,
        'n_pass': n_pass,
        'deploy_band': deploy_band,
        'threshold_sweep': sweep,
        'pipeline_verification': pipeline,
    }

    save_json(summary, 'r173b_summary.json')

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Robustness: {n_pass}/5 PASS -> {deploy_band}", flush=True)
    for key, res in tests.items():
        status = 'PASS' if res.get('passed') else 'FAIL'
        print(f"    {key}: {status}", flush=True)
    if sweep.get('best_threshold'):
        print(f"  Best threshold: {sweep['best_threshold']} (Sharpe={sweep['best_sharpe']:.3f})", flush=True)
    print(f"  Pipeline: {'OK' if pipeline.get('pipeline_ok') else 'ISSUES'} "
          f"({pipeline['n_pass']}/{pipeline['n_total']} checks)", flush=True)
    print(f"  Elapsed: {elapsed_s}s", flush=True)
    print(f"  Output: {OUTPUT_DIR / 'r173b_summary.json'}", flush=True)

    # Free memory
    del m1_all, h1_with_m1
    gc.collect()


if __name__ == '__main__':
    main()
