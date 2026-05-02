#!/usr/bin/env python3
"""
R90-C: ML Direction Prediction Model (GPU-Accelerated)
=======================================================
Trains XGBoost and LightGBM models to predict gold price direction
using 94 external features + gold-derived features.

Walk-forward: 12-fold expanding window validation.
Target: next-day and next-5-day direction prediction.
"""
import sys, os, io, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r90_external_data/r90c_ml_direction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100

# ═══════════════════════════════════════════════════════════════
# GPU detection helpers
# ═══════════════════════════════════════════════════════════════

def detect_xgb_device():
    """Try GPU, fall back to CPU."""
    import xgboost as xgb
    try:
        test = xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=2, verbosity=0)
        X_t = np.random.randn(20, 3).astype(np.float32)
        y_t = (X_t[:, 0] > 0).astype(int)
        test.fit(X_t, y_t)
        print("  XGBoost: CUDA GPU detected")
        return 'cuda'
    except Exception:
        print("  XGBoost: GPU not available, using CPU")
        return 'cpu'


def detect_lgb_device():
    """Try GPU, fall back to CPU."""
    import lightgbm as lgb
    try:
        test = lgb.LGBMClassifier(device='gpu', n_estimators=2, verbose=-1)
        X_t = np.random.randn(20, 3).astype(np.float32)
        y_t = (X_t[:, 0] > 0).astype(int)
        test.fit(X_t, y_t)
        print("  LightGBM: GPU detected")
        return 'gpu'
    except Exception:
        print("  LightGBM: GPU not available, using CPU")
        return 'cpu'


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_external_data():
    """Load aligned_daily.csv with Date index."""
    path = Path("data/external/aligned_daily.csv")
    if not path.exists():
        raise FileNotFoundError(f"External data not found: {path}")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    print(f"  External data: {len(df)} rows, {df.shape[1]} columns")
    print(f"  Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def load_gold_daily():
    """Load H1 gold bid data, resample to daily OHLC."""
    _candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
    ]
    h1_path = next((p for p in _candidates if p.exists()), _candidates[0])
    if not h1_path.exists():
        raise FileNotFoundError(f"H1 data not found: {h1_path}")

    df = pd.read_csv(h1_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    daily = df.resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Open'])

    daily.index = daily.index.tz_localize(None)
    print(f"  Gold daily: {len(daily)} bars, {daily.index[0].date()} -> {daily.index[-1].date()}")
    return daily


def merge_data(gold_daily, ext_df):
    """Merge gold daily OHLC with external data on Date."""
    gold_daily.index.name = 'Date'
    merged = gold_daily.join(ext_df, how='inner', rsuffix='_ext')
    print(f"  Merged: {len(merged)} rows ({merged.index[0].date()} -> {merged.index[-1].date()})")
    return merged


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def build_gold_features(df):
    """Compute gold-specific technical features from daily OHLC."""
    feat = pd.DataFrame(index=df.index)
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    for p in [1, 2, 5, 10, 20, 60]:
        feat[f'gold_ret_{p}d'] = close.pct_change(p)
        feat[f'gold_logret_{p}d'] = np.log(close / close.shift(p))

    atr14 = compute_atr(df, 14)
    feat['gold_atr14'] = atr14
    feat['gold_atr14_pctile'] = atr14.rolling(100).rank(pct=True)

    feat['gold_rsi14'] = compute_rsi(close, 14)
    feat['gold_rsi2'] = compute_rsi(close, 2)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    feat['gold_mom_sma20'] = close / sma20 - 1
    feat['gold_mom_sma50'] = close / sma50 - 1

    for w in [5, 20, 60]:
        feat[f'gold_vol_{w}d'] = close.pct_change().rolling(w).std()

    feat['gold_range_pctile'] = ((high - low) / atr14).rolling(100).rank(pct=True)

    vol_sma20 = vol.rolling(20).mean()
    feat['gold_vol_ratio'] = vol / vol_sma20.replace(0, np.nan)

    feat['gold_hl_range'] = (high - low) / close

    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    feat['gold_dist_20d_high'] = (close - high20) / close
    feat['gold_dist_20d_low'] = (close - low20) / close

    return feat


def build_interaction_features(df):
    """Build interaction features from external data columns."""
    feat = pd.DataFrame(index=df.index)

    pairs = [
        ('REAL_YIELD_DFII10', 'DXY_Mom5', 'ix_ryield_dxy'),
        ('VIX_Zscore', 'CREDIT_STRESS', 'ix_vix_credit'),
        ('COPPER_GOLD_RATIO', 'CRUDE_Mom5', 'ix_copper_crude'),
        ('DXY_Mom20', 'US10Y_Change20', 'ix_dxy_us10y'),
        ('M2_YoY', 'FED_FUNDS_DFF', 'ix_m2_fed'),
        ('USDCNH_Mom20', 'USDJPY_Mom5', 'ix_cnh_jpy'),
    ]
    for col_a, col_b, name in pairs:
        if col_a in df.columns and col_b in df.columns:
            feat[name] = df[col_a] * df[col_b]
        else:
            print(f"  [WARN] Interaction skipped — missing {col_a} or {col_b}")

    return feat


def build_targets(df):
    """Build target variables: dir_1d, dir_5d, ret_5d_quintile."""
    close = df['Close']
    targets = pd.DataFrame(index=df.index)

    targets['dir_1d'] = (close.shift(-1) > close).astype(int)
    targets['dir_5d'] = (close.shift(-5) > close).astype(int)

    fwd_5d_ret = close.shift(-5) / close - 1
    targets['ret_5d_quintile'] = pd.qcut(fwd_5d_ret, 5, labels=False, duplicates='drop')

    return targets


def build_full_dataset(gold_daily, ext_df):
    """Assemble all features and targets into a single DataFrame."""
    merged = merge_data(gold_daily, ext_df)

    ext_numeric = merged.select_dtypes(include=[np.number])
    # Drop gold OHLCV from external set (we keep our own)
    drop_cols = [c for c in ext_numeric.columns if c in ['Open', 'High', 'Low', 'Close', 'Volume']]
    ext_features = ext_numeric.drop(columns=drop_cols, errors='ignore')

    gold_feat = build_gold_features(merged)
    interaction_feat = build_interaction_features(merged)
    targets = build_targets(merged)

    features = pd.concat([ext_features, gold_feat, interaction_feat], axis=1)

    features = features.ffill().bfill()

    # Drop columns that are still entirely NaN, then fill remaining with 0
    nan_cols = features.columns[features.isna().all()]
    if len(nan_cols) > 0:
        print(f"  Dropping {len(nan_cols)} all-NaN columns: {list(nan_cols[:5])}...")
        features = features.drop(columns=nan_cols)
    features = features.fillna(0)

    dataset = pd.concat([features, targets], axis=1)
    before = len(dataset)
    dataset = dataset.dropna(subset=['dir_1d', 'dir_5d'])
    dataset = dataset.iloc[100:]  # drop early rows where rolling indicators are NaN
    print(f"  Final dataset: {len(dataset)} rows (dropped {before - len(dataset)} early/NaN rows)")
    print(f"  Features: {features.shape[1]}, Targets: {targets.shape[1]}")

    feature_cols = [c for c in features.columns if c in dataset.columns]
    target_cols = ['dir_1d', 'dir_5d', 'ret_5d_quintile']

    return dataset, feature_cols, target_cols


# ═══════════════════════════════════════════════════════════════
# Walk-Forward Protocol
# ═══════════════════════════════════════════════════════════════

def generate_folds(dataset_index):
    """Generate ~12 expanding-window folds with 18-month test windows."""
    all_starts = pd.date_range('2009-01-01', '2026-01-01', freq='6MS')
    fold_starts = all_starts[::3]  # every 3rd → ~12 folds

    folds = []
    for i, fs in enumerate(fold_starts):
        fe = fs + pd.DateOffset(months=18)
        if fs < dataset_index.min() + pd.DateOffset(years=3):
            continue
        if fs > dataset_index.max():
            break
        folds.append((f"Fold{i+1}", fs, fe))

    print(f"  Walk-forward folds: {len(folds)}")
    for name, s, e in folds:
        print(f"    {name}: test {s.date()} -> {e.date()}")
    return folds


# ═══════════════════════════════════════════════════════════════
# Model Training & Evaluation
# ═══════════════════════════════════════════════════════════════

def train_evaluate_fold(X_train, y_train, X_test, y_test, xgb_device, lgb_device, target_name):
    """Train XGBoost + LightGBM on one fold, return metrics + predictions."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    results = {}

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', device=xgb_device,
        eval_metric='logloss', random_state=42,
        use_label_encoder=False, verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)

    results['xgb'] = {
        'accuracy': float(accuracy_score(y_test, xgb_pred)),
        'precision': float(precision_score(y_test, xgb_pred, zero_division=0)),
        'recall': float(recall_score(y_test, xgb_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, xgb_prob)),
        'probs': xgb_prob,
        'model': xgb_model,
    }

    # --- LightGBM ---
    lgb_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        verbose=-1, random_state=42,
    )
    if lgb_device == 'gpu':
        lgb_params['device'] = 'gpu'

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)],
    )
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)

    results['lgb'] = {
        'accuracy': float(accuracy_score(y_test, lgb_pred)),
        'precision': float(precision_score(y_test, lgb_pred, zero_division=0)),
        'recall': float(recall_score(y_test, lgb_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, lgb_prob)),
        'probs': lgb_prob,
        'model': lgb_model,
    }

    # --- Ensemble (soft voting) ---
    ens_prob = (xgb_prob + lgb_prob) / 2
    ens_pred = (ens_prob >= 0.5).astype(int)
    results['ensemble'] = {
        'accuracy': float(accuracy_score(y_test, ens_pred)),
        'precision': float(precision_score(y_test, ens_pred, zero_division=0)),
        'recall': float(recall_score(y_test, ens_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_test, ens_prob)),
        'probs': ens_prob,
    }

    return results


def run_walk_forward(dataset, feature_cols, target_col, folds, xgb_device, lgb_device):
    """Run full walk-forward validation for a single target."""
    all_fold_results = []
    all_predictions = []

    for fold_name, fold_start, fold_end in folds:
        t0 = time.time()

        train_mask = dataset.index < fold_start
        test_mask = (dataset.index >= fold_start) & (dataset.index < fold_end)

        X_train = dataset.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = dataset.loc[train_mask, target_col].values.astype(int)
        X_test = dataset.loc[test_mask, feature_cols].values.astype(np.float32)
        y_test = dataset.loc[test_mask, target_col].values.astype(int)

        if len(X_train) < 500 or len(X_test) < 20:
            print(f"  {fold_name}: skipped (train={len(X_train)}, test={len(X_test)})")
            continue

        results = train_evaluate_fold(X_train, y_train, X_test, y_test,
                                      xgb_device, lgb_device, target_col)
        elapsed = time.time() - t0

        fold_rec = {
            'fold': fold_name,
            'train_start': str(dataset.index[train_mask][0].date()),
            'train_end': str(dataset.index[train_mask][-1].date()),
            'test_start': str(fold_start.date()),
            'test_end': str(min(fold_end, dataset.index[test_mask][-1]).date()),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'elapsed_sec': round(elapsed, 1),
        }
        for model_name in ['xgb', 'lgb', 'ensemble']:
            for metric in ['accuracy', 'precision', 'recall', 'auc']:
                fold_rec[f'{model_name}_{metric}'] = results[model_name][metric]
        all_fold_results.append(fold_rec)

        pred_df = pd.DataFrame({
            'date': dataset.index[test_mask],
            f'actual_{target_col}': y_test,
            f'xgb_prob_{target_col}': results['xgb']['probs'],
            f'lgb_prob_{target_col}': results['lgb']['probs'],
            f'ensemble_prob_{target_col}': results['ensemble']['probs'],
        })
        all_predictions.append(pred_df)

        print(f"  {fold_name} ({target_col}): "
              f"train={len(X_train):,} test={len(X_test):,} | "
              f"XGB acc={results['xgb']['accuracy']:.3f} AUC={results['xgb']['auc']:.3f} | "
              f"LGB acc={results['lgb']['accuracy']:.3f} AUC={results['lgb']['auc']:.3f} | "
              f"ENS acc={results['ensemble']['accuracy']:.3f} AUC={results['ensemble']['auc']:.3f} | "
              f"{elapsed:.1f}s")

    last_fold_models = None
    if all_fold_results:
        # Re-train on last fold to extract feature importance & SHAP
        last_fold_start = folds[-1][1]
        last_fold_end = folds[-1][2]
        train_mask = dataset.index < last_fold_start
        test_mask = (dataset.index >= last_fold_start) & (dataset.index < last_fold_end)
        X_train = dataset.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = dataset.loc[train_mask, target_col].values.astype(int)
        X_test = dataset.loc[test_mask, feature_cols].values.astype(np.float32)
        y_test = dataset.loc[test_mask, target_col].values.astype(int)
        if len(X_test) > 0:
            last_results = train_evaluate_fold(X_train, y_train, X_test, y_test,
                                               xgb_device, lgb_device, target_col)
            last_fold_models = {
                'xgb': last_results['xgb']['model'],
                'lgb': last_results['lgb']['model'],
                'X_test': X_test,
                'y_test': y_test,
            }

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return all_fold_results, predictions_df, last_fold_models


# ═══════════════════════════════════════════════════════════════
# Feature Importance & SHAP
# ═══════════════════════════════════════════════════════════════

def extract_feature_importance(last_fold_models, feature_cols, top_n=30):
    """Get gain-based feature importance from XGBoost; try SHAP on last fold."""
    importance = {}

    if last_fold_models is None:
        return importance

    xgb_model = last_fold_models['xgb']
    xgb_imp = xgb_model.feature_importances_
    imp_pairs = sorted(zip(feature_cols, xgb_imp.tolist()), key=lambda x: -x[1])
    importance['xgb_gain_top30'] = [{'feature': f, 'importance': round(v, 6)}
                                     for f, v in imp_pairs[:top_n]]

    lgb_model = last_fold_models['lgb']
    lgb_imp = lgb_model.feature_importances_
    lgb_pairs = sorted(zip(feature_cols, lgb_imp.tolist()), key=lambda x: -x[1])
    importance['lgb_gain_top30'] = [{'feature': f, 'importance': int(v)}
                                     for f, v in lgb_pairs[:top_n]]

    # SHAP (best-effort on last fold only)
    try:
        import shap
        print("\n  Computing SHAP values (last fold only)...")
        t0 = time.time()
        explainer = shap.TreeExplainer(xgb_model)
        X_test = last_fold_models['X_test']
        # Limit to 500 samples for speed
        if len(X_test) > 500:
            idx = np.random.choice(len(X_test), 500, replace=False)
            X_shap = X_test[idx]
        else:
            X_shap = X_test
        shap_values = explainer.shap_values(X_shap)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_pairs = sorted(zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: -x[1])
        importance['shap_top30'] = [{'feature': f, 'mean_abs_shap': round(v, 6)}
                                     for f, v in shap_pairs[:top_n]]
        print(f"  SHAP computed in {time.time() - t0:.1f}s")
    except ImportError:
        print("  [INFO] shap not installed, skipping SHAP analysis")
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")

    return importance


# ═══════════════════════════════════════════════════════════════
# Signal-Following Backtest
# ═══════════════════════════════════════════════════════════════

def signal_following_backtest(predictions_df, gold_daily, target_col='dir_1d'):
    """Long when P(up) > 0.55, short when P(up) < 0.45, flat otherwise."""
    prob_col = f'ensemble_prob_{target_col}'
    actual_col = f'actual_{target_col}'

    if prob_col not in predictions_df.columns or len(predictions_df) == 0:
        return {}

    bt = predictions_df.copy()
    bt['date'] = pd.to_datetime(bt['date'])
    bt = bt.sort_values('date').reset_index(drop=True)

    bt = bt.merge(
        gold_daily[['Close']].reset_index().rename(columns={'Date': 'date', 'Close': 'gold_close'}),
        on='date', how='left'
    )
    bt['gold_ret'] = bt['gold_close'].pct_change()

    bt['signal'] = 0
    bt.loc[bt[prob_col] > 0.55, 'signal'] = 1
    bt.loc[bt[prob_col] < 0.45, 'signal'] = -1

    bt['daily_pnl'] = bt['signal'] * bt['gold_ret'] * PV
    bt['cum_pnl'] = bt['daily_pnl'].cumsum()

    equity = bt['cum_pnl']
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    trading_days = bt[bt['signal'] != 0]
    wins = trading_days[trading_days['daily_pnl'] > 0]
    win_rate = len(wins) / len(trading_days) if len(trading_days) > 0 else 0

    daily_returns = bt['daily_pnl']
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0)

    result = {
        'total_pnl': round(float(equity.iloc[-1]), 2),
        'sharpe': round(float(sharpe), 3),
        'max_dd': round(float(max_dd), 2),
        'win_rate': round(float(win_rate), 4),
        'n_trading_days': int(len(trading_days)),
        'n_long': int((bt['signal'] == 1).sum()),
        'n_short': int((bt['signal'] == -1).sum()),
        'n_flat': int((bt['signal'] == 0).sum()),
        'avg_daily_pnl': round(float(daily_returns.mean()), 4),
    }
    return result


# ═══════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════

def aggregate_fold_metrics(fold_results):
    """Compute mean ± std of per-fold metrics."""
    if not fold_results:
        return {}
    metrics_keys = ['accuracy', 'precision', 'recall', 'auc']
    agg = {}
    for model in ['xgb', 'lgb', 'ensemble']:
        agg[model] = {}
        for m in metrics_keys:
            key = f'{model}_{m}'
            vals = [f[key] for f in fold_results if key in f]
            if vals:
                agg[model][f'{m}_mean'] = round(float(np.mean(vals)), 4)
                agg[model][f'{m}_std'] = round(float(np.std(vals)), 4)
    return agg


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("R90-C: ML Direction Prediction Model (GPU-Accelerated)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_total = time.time()

    # --- Device detection ---
    print("\n--- Device Detection ---")
    xgb_device = detect_xgb_device()
    lgb_device = detect_lgb_device()

    # --- Data Loading ---
    print("\n--- Loading Data ---")
    ext_df = load_external_data()
    gold_daily = load_gold_daily()

    # --- Feature Engineering ---
    print("\n--- Building Features & Targets ---")
    dataset, feature_cols, target_cols = build_full_dataset(gold_daily, ext_df)
    print(f"  Feature columns ({len(feature_cols)}):")
    for i in range(0, len(feature_cols), 8):
        print(f"    {', '.join(feature_cols[i:i+8])}")

    # --- Walk-Forward ---
    folds = generate_folds(dataset.index)

    all_results = {}
    all_predictions = {}
    all_importances = {}

    for target in ['dir_1d', 'dir_5d']:
        print(f"\n{'='*70}")
        print(f"  Walk-Forward: target = {target}")
        print(f"{'='*70}")

        fold_results, preds_df, last_models = run_walk_forward(
            dataset, feature_cols, target, folds, xgb_device, lgb_device
        )

        agg = aggregate_fold_metrics(fold_results)
        all_results[target] = {
            'per_fold': fold_results,
            'aggregate': agg,
        }
        all_predictions[target] = preds_df

        if target == 'dir_1d':
            importance = extract_feature_importance(last_models, feature_cols)
            all_importances = importance

        # Summary
        print(f"\n  --- {target} Aggregate ---")
        for model in ['xgb', 'lgb', 'ensemble']:
            if model in agg:
                m = agg[model]
                print(f"    {model:>8s}: Acc={m.get('accuracy_mean',0):.4f}±{m.get('accuracy_std',0):.4f}  "
                      f"AUC={m.get('auc_mean',0):.4f}±{m.get('auc_std',0):.4f}  "
                      f"Prec={m.get('precision_mean',0):.4f}  Rec={m.get('recall_mean',0):.4f}")

    # --- Signal-Following Backtest ---
    print(f"\n{'='*70}")
    print("  Signal-Following Backtest (ensemble, dir_1d)")
    print(f"{'='*70}")

    bt_1d = signal_following_backtest(all_predictions.get('dir_1d', pd.DataFrame()),
                                      gold_daily, 'dir_1d')
    bt_5d = signal_following_backtest(all_predictions.get('dir_5d', pd.DataFrame()),
                                      gold_daily, 'dir_5d')

    print(f"\n  dir_1d backtest:")
    if bt_1d:
        print(f"    Sharpe:   {bt_1d['sharpe']:.3f}")
        print(f"    Total PnL: {bt_1d['total_pnl']:.2f}")
        print(f"    MaxDD:    {bt_1d['max_dd']:.2f}")
        print(f"    Win Rate: {bt_1d['win_rate']:.4f}")
        print(f"    Trading days: {bt_1d['n_trading_days']} (long={bt_1d['n_long']}, "
              f"short={bt_1d['n_short']}, flat={bt_1d['n_flat']})")

    print(f"\n  dir_5d backtest:")
    if bt_5d:
        print(f"    Sharpe:   {bt_5d['sharpe']:.3f}")
        print(f"    Total PnL: {bt_5d['total_pnl']:.2f}")
        print(f"    MaxDD:    {bt_5d['max_dd']:.2f}")
        print(f"    Win Rate: {bt_5d['win_rate']:.4f}")

    # --- Ensemble comparison ---
    print(f"\n{'='*70}")
    print("  Ensemble vs Individual Model Comparison")
    print(f"{'='*70}")

    for target in ['dir_1d', 'dir_5d']:
        agg = all_results[target].get('aggregate', {})
        if not agg:
            continue
        print(f"\n  Target: {target}")
        print(f"    {'Model':<10s} {'Acc':>8s} {'AUC':>8s} {'Prec':>8s} {'Rec':>8s}")
        print(f"    {'-'*42}")
        for model in ['xgb', 'lgb', 'ensemble']:
            m = agg.get(model, {})
            print(f"    {model:<10s} "
                  f"{m.get('accuracy_mean',0):>8.4f} "
                  f"{m.get('auc_mean',0):>8.4f} "
                  f"{m.get('precision_mean',0):>8.4f} "
                  f"{m.get('recall_mean',0):>8.4f}")

    # --- Feature Importance Summary ---
    if all_importances.get('xgb_gain_top30'):
        print(f"\n{'='*70}")
        print("  XGBoost Feature Importance (gain, top 30)")
        print(f"{'='*70}")
        for i, item in enumerate(all_importances['xgb_gain_top30']):
            print(f"    {i+1:>2d}. {item['feature']:<35s} {item['importance']:.6f}")

    # --- Save outputs ---
    print(f"\n{'='*70}")
    print("  Saving outputs")
    print(f"{'='*70}")

    elapsed_total = time.time() - t_total

    output_json = {
        'experiment': 'R90-C ML Direction Prediction',
        'timestamp': datetime.now().isoformat(),
        'runtime_sec': round(elapsed_total, 1),
        'devices': {'xgboost': xgb_device, 'lightgbm': lgb_device},
        'dataset': {
            'n_rows': len(dataset),
            'n_features': len(feature_cols),
            'date_range': f"{dataset.index[0].date()} -> {dataset.index[-1].date()}",
        },
        'n_folds': len(folds),
        'dir_1d': all_results.get('dir_1d', {}),
        'dir_5d': all_results.get('dir_5d', {}),
        'backtest_dir_1d': bt_1d,
        'backtest_dir_5d': bt_5d,
        'feature_importance': all_importances,
    }

    json_path = OUTPUT_DIR / "r90c_results.json"
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2, default=str)
    print(f"  Results: {json_path}")

    # Merge predictions from both targets
    pred_1d = all_predictions.get('dir_1d', pd.DataFrame())
    pred_5d = all_predictions.get('dir_5d', pd.DataFrame())

    if not pred_1d.empty and not pred_5d.empty:
        pred_merged = pred_1d.merge(
            pred_5d[['date', 'actual_dir_5d', 'xgb_prob_dir_5d',
                      'lgb_prob_dir_5d', 'ensemble_prob_dir_5d']],
            on='date', how='outer'
        )
    elif not pred_1d.empty:
        pred_merged = pred_1d
    elif not pred_5d.empty:
        pred_merged = pred_5d
    else:
        pred_merged = pd.DataFrame()

    if not pred_merged.empty:
        csv_path = OUTPUT_DIR / "r90c_predictions.csv"
        pred_merged.to_csv(csv_path, index=False)
        print(f"  Predictions: {csv_path} ({len(pred_merged)} rows)")

    # --- Final Summary ---
    print(f"\n{'='*80}")
    print("R90-C SUMMARY")
    print(f"{'='*80}")
    print(f"  Runtime:    {elapsed_total/60:.1f} min ({elapsed_total:.0f}s)")
    print(f"  Devices:    XGB={xgb_device}, LGB={lgb_device}")
    print(f"  Dataset:    {len(dataset):,} rows × {len(feature_cols)} features")
    print(f"  Folds:      {len(folds)}")
    for target in ['dir_1d', 'dir_5d']:
        agg = all_results[target].get('aggregate', {})
        ens = agg.get('ensemble', {})
        print(f"\n  {target}:")
        print(f"    Ensemble Acc:  {ens.get('accuracy_mean', 0):.4f} ± {ens.get('accuracy_std', 0):.4f}")
        print(f"    Ensemble AUC:  {ens.get('auc_mean', 0):.4f} ± {ens.get('auc_std', 0):.4f}")
    if bt_1d:
        print(f"\n  Backtest (dir_1d): Sharpe={bt_1d['sharpe']:.3f}, "
              f"PnL={bt_1d['total_pnl']:.2f}, MaxDD={bt_1d['max_dd']:.2f}, "
              f"WinRate={bt_1d['win_rate']:.4f}")
    if bt_5d:
        print(f"  Backtest (dir_5d): Sharpe={bt_5d['sharpe']:.3f}, "
              f"PnL={bt_5d['total_pnl']:.2f}, MaxDD={bt_5d['max_dd']:.2f}, "
              f"WinRate={bt_5d['win_rate']:.4f}")

    print(f"\n{'='*80}")
    print(f"R90-C completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
