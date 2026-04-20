"""
Round 20: New Search Space Exploration
=======================================
Phase 2: External data signal quality scoring (A1-A4)
Phase 3: Meta-Labeling ML model (B1-B3)

Core idea: enrich the existing L7 Keltner system with external information
sources to improve signal quality, without changing the entry logic itself.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import (
    DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
)

OUT_DIR = Path("results/round20_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_DATA_PATH = Path("data/external/aligned_daily.csv")

ULTRA2 = {
    'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
    'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
    'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
}


def get_base():
    return {**LIVE_PARITY_KWARGS}


def get_l6():
    base = get_base()
    base['regime_config'] = ULTRA2
    return base


def get_l7():
    kw = get_l6()
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    return kw


def load_external_data():
    """Load aligned daily external data."""
    df = pd.read_csv(EXTERNAL_DATA_PATH, parse_dates=["Date"], index_col="Date")
    return df


def extract_trade_features(trades, external_df):
    """Join trade records with external daily data to build feature matrix."""
    rows = []
    for t in trades:
        entry_date = pd.Timestamp(t.entry_time).normalize().tz_localize(None)

        row = {
            'entry_time': t.entry_time,
            'entry_date': entry_date,
            'direction': t.direction,
            'strategy': t.strategy,
            'pnl': t.pnl,
            'bars_held': t.bars_held,
            'exit_reason': t.exit_reason,
            'entry_atr': getattr(t, 'entry_atr', 0),
            'profitable': 1 if t.pnl > 0 else 0,
        }

        # Merge external data (use previous day to avoid look-ahead)
        prev_date = entry_date - pd.Timedelta(days=1)
        # Find closest available date
        if prev_date in external_df.index:
            ext_row = external_df.loc[prev_date]
        else:
            mask = external_df.index <= prev_date
            if mask.any():
                ext_row = external_df.loc[external_df.index[mask][-1]]
            else:
                ext_row = None

        if ext_row is not None:
            for col in external_df.columns:
                row[f'ext_{col}'] = ext_row[col]

        rows.append(row)

    return pd.DataFrame(rows)


def phase2_signal_quality_analysis(data, out):
    """Phase 2: Analyze how external data correlates with trade outcomes."""
    print("=" * 70)
    print("R20 Phase 2: External Data Signal Quality Analysis")
    print("=" * 70)

    external_df = load_external_data()
    l7_kw = get_l7()

    # Run L7 baseline to get all trades
    for sp_label, sp in [("sp0.3", 0.30)]:
        print(f"\n--- Running L7 baseline ({sp_label}) ---")
        stats = run_variant(data, f"L7_{sp_label}",
                            spread_cost=sp, **l7_kw)
        trades = stats['_trades']
        keltner_trades = [t for t in trades if t.strategy == 'keltner']
        print(f"  Total trades: {len(trades)}, Keltner: {len(keltner_trades)}")

        # Build feature matrix
        feat_df = extract_trade_features(keltner_trades, external_df)
        print(f"  Feature matrix: {feat_df.shape}")

        # --- A1: COT Regime Analysis ---
        print("\n=== A1: COT Managed Money Regime ===")
        if 'ext_COT_MM_Net_Pct' in feat_df.columns:
            feat_df['cot_regime'] = pd.cut(
                feat_df['ext_COT_MM_Net_Pct'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['VeryShort', 'Short', 'Neutral', 'Long', 'VeryLong']
            )
            cot_stats = feat_df.groupby('cot_regime', observed=True).agg(
                n=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                win_rate=('profitable', 'mean'),
                total_pnl=('pnl', 'sum'),
            ).round(3)
            print(cot_stats.to_string())

            # Split by direction
            for direction in ['BUY', 'SELL']:
                sub = feat_df[feat_df['direction'] == direction]
                if len(sub) > 50:
                    sub_stats = sub.groupby('cot_regime', observed=True).agg(
                        n=('pnl', 'count'),
                        avg_pnl=('pnl', 'mean'),
                        win_rate=('profitable', 'mean'),
                    ).round(3)
                    print(f"\n  {direction} trades by COT regime:")
                    print(sub_stats.to_string())
        else:
            print("  [SKIP] COT data not available in features")

        # --- A2: VIX Regime Analysis ---
        print("\n=== A2: VIX Regime ===")
        if 'ext_VIX_Close' in feat_df.columns:
            feat_df['vix_regime'] = pd.cut(
                feat_df['ext_VIX_Close'],
                bins=[0, 15, 20, 25, 35, 100],
                labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
            )
            vix_stats = feat_df.groupby('vix_regime', observed=True).agg(
                n=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                win_rate=('profitable', 'mean'),
                total_pnl=('pnl', 'sum'),
            ).round(3)
            print(vix_stats.to_string())

            # VIX z-score analysis
            if 'ext_VIX_Zscore' in feat_df.columns:
                feat_df['vix_z_regime'] = pd.cut(
                    feat_df['ext_VIX_Zscore'],
                    bins=[-10, -1, -0.5, 0.5, 1, 10],
                    labels=['VeryLow_Z', 'Low_Z', 'Normal_Z', 'High_Z', 'VeryHigh_Z']
                )
                vz_stats = feat_df.groupby('vix_z_regime', observed=True).agg(
                    n=('pnl', 'count'),
                    avg_pnl=('pnl', 'mean'),
                    win_rate=('profitable', 'mean'),
                ).round(3)
                print(f"\n  VIX Z-score regimes:")
                print(vz_stats.to_string())

        # --- A3: DXY Momentum Analysis ---
        print("\n=== A3: DXY Momentum ===")
        if 'ext_DXY_Mom5' in feat_df.columns:
            feat_df['dxy_mom_regime'] = pd.cut(
                feat_df['ext_DXY_Mom5'],
                bins=[-1, -0.01, -0.003, 0.003, 0.01, 1],
                labels=['StrongDown', 'Down', 'Flat', 'Up', 'StrongUp']
            )
            dxy_stats = feat_df.groupby('dxy_mom_regime', observed=True).agg(
                n=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                win_rate=('profitable', 'mean'),
                total_pnl=('pnl', 'sum'),
            ).round(3)
            print(dxy_stats.to_string())

            # DXY momentum x direction interaction
            for direction in ['BUY', 'SELL']:
                sub = feat_df[feat_df['direction'] == direction]
                if len(sub) > 50:
                    sub_stats = sub.groupby('dxy_mom_regime', observed=True).agg(
                        n=('pnl', 'count'),
                        avg_pnl=('pnl', 'mean'),
                        win_rate=('profitable', 'mean'),
                    ).round(3)
                    print(f"\n  {direction} trades by DXY momentum:")
                    print(sub_stats.to_string())

        # --- A4: GLD Volume Spike ---
        print("\n=== A4: GLD Volume Spike ===")
        if 'ext_GLD_Vol_Ratio' in feat_df.columns:
            feat_df['gld_vol_regime'] = pd.cut(
                feat_df['ext_GLD_Vol_Ratio'],
                bins=[0, 0.7, 1.0, 1.5, 2.0, 100],
                labels=['VeryLow', 'Low', 'Normal', 'High', 'Spike']
            )
            gld_stats = feat_df.groupby('gld_vol_regime', observed=True).agg(
                n=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                win_rate=('profitable', 'mean'),
                total_pnl=('pnl', 'sum'),
            ).round(3)
            print(gld_stats.to_string())

        # --- Correlation matrix ---
        print("\n=== Factor-PnL Correlations (Top 15) ===")
        ext_cols = [c for c in feat_df.columns if c.startswith('ext_')]
        if ext_cols:
            corrs = feat_df[ext_cols + ['pnl']].corr()['pnl'].drop('pnl')
            corrs = corrs.dropna().abs().sort_values(ascending=False)
            print(corrs.head(15).to_string())

        # --- IC (Information Coefficient) by factor ---
        print("\n=== Rank IC (Spearman) with PnL ===")
        from scipy.stats import spearmanr
        ic_results = {}
        for col in ext_cols:
            valid = feat_df[[col, 'pnl']].dropna()
            if len(valid) > 100:
                ic, pval = spearmanr(valid[col], valid['pnl'])
                ic_results[col] = {'IC': round(ic, 4), 'p-value': round(pval, 4),
                                   'n': len(valid)}
        if ic_results:
            ic_df = pd.DataFrame(ic_results).T.sort_values('IC', key=abs, ascending=False)
            print(ic_df.head(15).to_string())

        # Save full feature matrix for Phase 3
        feat_path = OUT_DIR / f"trade_features_{sp_label}.csv"
        feat_df.to_csv(feat_path, index=False)
        print(f"\n  Saved feature matrix to {feat_path}")

    return feat_df


def phase3_meta_labeling(data, out):
    """Phase 3: Meta-Labeling ML model to predict trade profitability."""
    print("\n" + "=" * 70)
    print("R20 Phase 3: Meta-Labeling ML Model")
    print("=" * 70)

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import lightgbm as lgb

    feat_path = OUT_DIR / "trade_features_sp0.3.csv"
    if not feat_path.exists():
        print("  [ERROR] Run Phase 2 first to generate trade features")
        return

    feat_df = pd.read_csv(feat_path)
    feat_df['entry_time'] = pd.to_datetime(feat_df['entry_time'])
    feat_df['entry_date'] = pd.to_datetime(feat_df['entry_date'])

    # Filter to keltner only
    feat_df = feat_df[feat_df['strategy'] == 'keltner'].copy()
    print(f"  Keltner trades: {len(feat_df)}")
    print(f"  Win rate: {feat_df['profitable'].mean():.3f}")
    print(f"  Date range: {feat_df['entry_date'].min()} ~ {feat_df['entry_date'].max()}")

    # Build feature columns
    ext_cols = [c for c in feat_df.columns if c.startswith('ext_')]
    # Add time features
    feat_df['hour'] = feat_df['entry_time'].dt.hour
    feat_df['dow'] = feat_df['entry_time'].dt.dayofweek
    feat_df['month'] = feat_df['entry_time'].dt.month
    feat_df['is_buy'] = (feat_df['direction'] == 'BUY').astype(int)

    feature_cols = ext_cols + ['hour', 'dow', 'month', 'is_buy']
    # Drop any with too many NaNs
    feature_cols = [c for c in feature_cols if feat_df[c].notna().sum() > len(feat_df) * 0.8]

    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Drop rows with NaN in features
    valid_mask = feat_df[feature_cols].notna().all(axis=1)
    df = feat_df[valid_mask].copy().reset_index(drop=True)
    print(f"  Valid samples after NaN drop: {len(df)}")

    X = df[feature_cols].values
    y = df['profitable'].values

    # --- B1: Purged Time Series K-Fold ---
    print("\n=== B1: Purged Time Series Cross-Validation ===")
    n_splits = 6
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=50)

    fold_results = []
    feature_importances = np.zeros(len(feature_cols))

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=8,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        # Simulate: skip low-confidence signals
        test_df = df.iloc[test_idx].copy()
        test_df['pred_prob'] = y_prob

        # Strategy: only trade when model says prob > 0.5
        high_conf = test_df[test_df['pred_prob'] > 0.5]
        low_conf = test_df[test_df['pred_prob'] <= 0.5]

        baseline_pnl = test_df['pnl'].sum()
        filtered_pnl = high_conf['pnl'].sum()
        skipped_pnl = low_conf['pnl'].sum()
        baseline_wr = test_df['profitable'].mean()
        filtered_wr = high_conf['profitable'].mean() if len(high_conf) > 0 else 0

        train_period = f"{df.iloc[train_idx[0]]['entry_date']:%Y-%m} ~ {df.iloc[train_idx[-1]]['entry_date']:%Y-%m}"
        test_period = f"{df.iloc[test_idx[0]]['entry_date']:%Y-%m} ~ {df.iloc[test_idx[-1]]['entry_date']:%Y-%m}"

        fold_results.append({
            'fold': fold_idx + 1,
            'train': train_period,
            'test': test_period,
            'n_test': len(test_idx),
            'acc': acc,
            'auc': auc,
            'baseline_pnl': baseline_pnl,
            'filtered_pnl': filtered_pnl,
            'skipped_pnl': skipped_pnl,
            'delta_pnl': filtered_pnl - baseline_pnl,
            'n_traded': len(high_conf),
            'n_skipped': len(low_conf),
            'baseline_wr': baseline_wr,
            'filtered_wr': filtered_wr,
        })

        feature_importances += model.feature_importances_

        print(f"\n  Fold {fold_idx+1}: train={train_period} test={test_period}")
        print(f"    Accuracy: {acc:.3f}, AUC: {auc:.3f}")
        print(f"    Baseline: {len(test_idx)} trades, PnL=${baseline_pnl:.0f}, WR={baseline_wr:.1%}")
        print(f"    Filtered: {len(high_conf)} trades, PnL=${filtered_pnl:.0f}, WR={filtered_wr:.1%}")
        print(f"    Skipped:  {len(low_conf)} trades, PnL=${skipped_pnl:.0f}")
        print(f"    Delta PnL: ${filtered_pnl - baseline_pnl:.0f}")

    # Summary
    print("\n=== ML Model Summary ===")
    fold_df = pd.DataFrame(fold_results)
    print(fold_df[['fold', 'n_test', 'acc', 'auc', 'baseline_pnl', 'filtered_pnl',
                    'delta_pnl', 'n_traded', 'n_skipped', 'filtered_wr']].to_string(index=False))

    total_baseline = fold_df['baseline_pnl'].sum()
    total_filtered = fold_df['filtered_pnl'].sum()
    total_skipped = fold_df['skipped_pnl'].sum()
    avg_auc = fold_df['auc'].mean()

    print(f"\n  Total Baseline PnL: ${total_baseline:.0f}")
    print(f"  Total Filtered PnL: ${total_filtered:.0f}")
    print(f"  Total Skipped PnL:  ${total_skipped:.0f}")
    print(f"  Net Delta:          ${total_filtered - total_baseline:.0f}")
    print(f"  Avg AUC:            {avg_auc:.3f}")

    positive_delta = sum(1 for r in fold_results if r['delta_pnl'] >= 0)
    print(f"  Folds with positive delta: {positive_delta}/{n_splits}")

    # --- B2: Feature Importance ---
    print("\n=== B2: Feature Importance (avg across folds) ===")
    feature_importances /= n_splits
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    print(imp_df.head(20).to_string(index=False))

    # --- B3: Confidence threshold sweep ---
    print("\n=== B3: Confidence Threshold Sweep ===")
    # Retrain on first 70% of data, test on last 30%
    split_idx = int(len(df) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_subset = df.iloc[split_idx:].copy()

    model_full = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        num_leaves=8, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model_full.fit(X_train, y_train)
    test_subset['pred_prob'] = model_full.predict_proba(X_test)[:, 1]

    print(f"{'Threshold':>10} {'N_Trade':>8} {'N_Skip':>7} {'PnL':>10} {'Base_PnL':>10} "
          f"{'Delta':>8} {'WR':>6} {'Avg$/t':>8}")
    baseline_total = test_subset['pnl'].sum()
    baseline_wr = test_subset['profitable'].mean()

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        traded = test_subset[test_subset['pred_prob'] > threshold]
        skipped = test_subset[test_subset['pred_prob'] <= threshold]
        if len(traded) == 0:
            continue
        pnl = traded['pnl'].sum()
        wr = traded['profitable'].mean()
        avg_pnl = traded['pnl'].mean()
        print(f"{threshold:>10.2f} {len(traded):>8} {len(skipped):>7} ${pnl:>9.0f} ${baseline_total:>9.0f} "
              f"${pnl - baseline_total:>7.0f} {wr:>5.1%} ${avg_pnl:>7.2f}")

    print(f"{'Baseline':>10} {len(test_subset):>8} {0:>7} ${baseline_total:>9.0f} ${baseline_total:>9.0f} "
          f"${'0':>7} {baseline_wr:>5.1%} ${test_subset['pnl'].mean():>7.2f}")


def main():
    t_start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Setup output
    out_path = OUT_DIR / "R20_full_output.txt"
    out = open(out_path, 'w', encoding='utf-8')

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                try:
                    f.write(data)
                except UnicodeEncodeError:
                    f.write(data.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    import sys as _sys
    old_stdout = _sys.stdout
    _sys.stdout = Tee(old_stdout, out)

    print(f"# R20: New Search Space Exploration")
    print(f"# Started: {ts}")
    print(f"# Workers: 1 (local)")

    # Load data
    data = DataBundle.load_default()
    print(f"# Data: M15={len(data.m15_df)} bars, H1={len(data.h1_df)} bars")

    # Phase 2
    feat_df = phase2_signal_quality_analysis(data, out)

    # Phase 3
    phase3_meta_labeling(data, out)

    elapsed = time.time() - t_start
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    _sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
