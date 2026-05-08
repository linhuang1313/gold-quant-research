#!/usr/bin/env python3
"""
Export R173 H1-only Entry Filter Model (4 features: ATR, ADX, RSI14, squeeze)
=============================================================================
Trains XGBoost on ALL historical L8_MAX trades (expanding window up to 2026-05-01)
using the 4 H1 features validated in R173b (5/5 robustness PASS, AUC 0.69-0.76).

Outputs to gold-quant-trading/data/r173_ml_filter.json for live shadow-mode deployment.

Run from gold-quant-research root:
    python scripts/export_r173_model.py
"""
import sys, os, io, json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_PATH = Path(r"C:\Users\hlin2\gold-quant-trading\data\r173_ml_filter.json")
TRAIN_CUTOFF = '2026-05-01'

FEATURE_COLS = ['ATR', 'ADX', 'RSI14', 'squeeze']

XGB_PARAMS = {
    'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 1,
}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Export R173 H1 Entry Filter (4 features)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Train cutoff: {TRAIN_CUTOFF}")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

    print("\n  Loading data (DataBundle)...", flush=True)
    data = DataBundle.load_default()
    h1_df = data.h1_df
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    for col in FEATURE_COLS:
        if col not in h1_df.columns:
            raise ValueError(f"Feature '{col}' not found in H1 data. Available: {list(h1_df.columns)}")

    print("\n  Running L8_MAX backtest...", flush=True)
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': 35,
          'min_lot_size': 0.02, 'max_lot_size': 0.02}
    result = run_variant(data, "R173_EXPORT", verbose=False, **kw)
    trades = result.get('_trades', [])
    print(f"  {len(trades)} trades, Sharpe={result.get('sharpe', 0):.2f}, "
          f"PnL=${result.get('total_pnl', 0):.0f}")

    print("\n  Building labeled feature matrix...", flush=True)
    samples = []
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF, tz='UTC')

    for trade in trades:
        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
        pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)

        if entry_time is None:
            continue
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize('UTC')
        if entry_time >= cutoff_ts:
            continue

        h1_time = entry_time.floor('h')
        idx = h1_df.index.get_indexer([h1_time], method='nearest')
        if idx[0] < 0 or idx[0] >= len(h1_df):
            continue

        row = h1_df.iloc[idx[0]]
        sample = {feat: float(row[feat]) for feat in FEATURE_COLS}
        sample['label'] = 1 if pnl > 0 else 0
        sample['pnl'] = pnl
        samples.append(sample)

    feat_df = pd.DataFrame(samples)
    print(f"  Training samples: {len(feat_df)} (cutoff < {TRAIN_CUTOFF})")
    print(f"  Win rate: {feat_df['label'].mean()*100:.1f}%")

    X = feat_df[FEATURE_COLS].copy()
    y = feat_df['label'].values

    med = X.median()
    X_filled = X.fillna(med)
    print(f"  NaN rows filled: {X.isna().any(axis=1).sum()}")

    print("\n  Training XGBoost...", flush=True)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_filled.values, y)

    train_proba = model.predict_proba(X_filled.values)[:, 1]
    train_auc = roc_auc_score(y, train_proba)
    train_acc = float((model.predict(X_filled.values) == y).mean())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_PATH))

    meta = {
        'model': 'R173_H1_EntryFilter',
        'features': FEATURE_COLS,
        'threshold': 0.70,
        'xgb_params': XGB_PARAMS,
        'train_cutoff': TRAIN_CUTOFF,
        'n_samples': len(feat_df),
        'win_rate': round(float(feat_df['label'].mean()), 4),
        'train_auc': round(train_auc, 4),
        'train_acc': round(train_acc, 4),
        'exported_at': datetime.now().isoformat(),
        'robustness': '5/5 PASS (R173b)',
        'median_fill': {k: round(v, 6) for k, v in med.to_dict().items()},
    }
    meta_path = OUTPUT_PATH.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  {'='*50}")
    print(f"  Model saved: {OUTPUT_PATH}")
    print(f"  Meta saved:  {meta_path}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Train samples: {len(feat_df)}")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Recommended threshold: 0.70")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()
