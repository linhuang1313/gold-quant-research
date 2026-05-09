#!/usr/bin/env python3
"""
Export production ML Exit model for L8_MAX (Keltner).
=====================================================
Trains XGBoost on ALL historical L8_MAX trades using R92-B validated features,
then saves model to gold-quant-trading/data/ for live use.

Run from gold-quant-research root:
    python scripts/export_ml_model.py

Prerequisites:
    pip install xgboost scikit-learn
"""
import sys, os, io
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_PATH = Path(r"C:\Users\hlin2\gold-quant-trading\data\l8_ml_exit_model.json")

USE_FEATURES = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2',
    'kc_breakout_strength', 'volume_ratio', 'atr_percentile',
    'ema9_ema21_cross', 'close_ema100_dist',
    'hour_of_day', 'day_of_week', 'direction',
]


def main():
    print("=" * 70)
    print("  Export ML Exit Model for L8_MAX")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    from experiments.run_r92b_multi_strategy import (
        compute_h1_indicators, build_features_for_trades
    )

    print("\n  Loading data and running L8_MAX backtest...", flush=True)
    bundle = DataBundle.load_custom()

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': 35,
          'spread_cost': 0.30, 'initial_capital': 2000,
          'min_lot_size': 0.02, 'max_lot_size': 0.02}
    result = run_variant(bundle, "L8_MAX", verbose=True, **kw)
    trades_raw = result['_trades']
    print(f"  {len(trades_raw)} trades from backtest", flush=True)

    print("  Computing H1 indicators...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    h1_indicators = compute_h1_indicators(h1_df)

    trade_dicts = []
    for t in trades_raw:
        trade_dicts.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason,
        })

    print("  Building feature matrix...", flush=True)
    X, y, valid_indices = build_features_for_trades(trade_dicts, h1_indicators, None)

    available = [c for c in USE_FEATURES if c in X.columns]
    if len(available) < len(USE_FEATURES):
        missing = set(USE_FEATURES) - set(available)
        print(f"  [WARN] Missing features: {missing}")
    X_use = X[available]

    print(f"  Features: {X_use.shape[1]} cols, {len(y)} samples", flush=True)
    print(f"  Win rate: {y.mean()*100:.1f}%", flush=True)

    print("\n  Training XGBoost on full dataset...", flush=True)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, verbosity=1,
    )

    med = X_use.median()
    X_filled = X_use.fillna(med)
    model.fit(X_filled, y)

    train_acc = float((model.predict(X_filled) == y).mean())
    train_proba = model.predict_proba(X_filled)[:, 1]
    from sklearn.metrics import roc_auc_score
    train_auc = roc_auc_score(y, train_proba)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_PATH))

    print(f"\n  {'='*50}")
    print(f"  Model saved: {OUTPUT_PATH}")
    print(f"  Features: {len(available)} ({', '.join(available[:5])}...)")
    print(f"  Train samples: {len(y)}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()
