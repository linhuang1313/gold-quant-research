#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R137 — Temporal Fusion Transformer Entry Prediction
=====================================================
Deep ML experiment using PyTorch (or XGBoost fallback) to predict gold
direction and magnitude on a 4-bar forward horizon.

Features (~20):
  H1: Close, returns(1/4/8/24), ATR14, RSI14, RSI2, KC_position, MACD_hist, volume
  D1 (resampled): EMA20 trend, daily range, daily momentum
  Macro (aligned_daily.csv): VIX, DXY, US10Y, real_yield

Phases:
  1. Load data & build feature matrix
  2. Target: 4-bar forward return sign (+1/-1) and magnitude quintile (1-5)
  3. Model: simplified TFT-style (multi-head attention over lookback=24)
     - Fallback: XGBoost with lagged features
  4. Walk-Forward: 6-month train / 2-month test, sliding every 2 months (2019-2026)
  5. Evaluation: AUC for direction, IC for magnitude
  6. Convert to trading signals (P(dir) > 0.6 AND mag > median)
  7. Backtest filtered signals vs baseline
  8. K-Fold validation

Install: pip install torch  OR  pip install xgboost scikit-learn
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv

OUTPUT_DIR = Path("results/r137_tft_entry")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
LOOKBACK = 24
FORWARD_BARS = 4
ALIGNED_CSV = Path("data/external/aligned_daily.csv")
H1_CSV = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv")

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# PyTorch availability
# ═══════════════════════════════════════════════════════════════
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    print("[INFO] PyTorch available — using TFT-style model")
except ImportError:
    print("[INFO] PyTorch not available — falling back to XGBoost")

if not HAS_TORCH:
    try:
        import xgboost as xgb
        from sklearn.model_selection import KFold
        print("[INFO] XGBoost available")
    except ImportError:
        print("[ERROR] Neither torch nor xgboost available. Install one:")
        print("  pip install torch")
        print("  pip install xgboost scikit-learn")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_kc_position(close, period=25, atr_mult=1.2, atr_period=14):
    kc_mid = close.ewm(span=period).mean()
    atr = (close.rolling(atr_period).max() - close.rolling(atr_period).min()).rolling(atr_period).mean()
    kc_upper = kc_mid + atr_mult * atr
    kc_lower = kc_mid - atr_mult * atr
    width = kc_upper - kc_lower
    pos = (close - kc_lower) / width.replace(0, np.nan)
    return pos.fillna(0.5)


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    return {
        'n': n,
        'sharpe': round(_sharpe(daily), 3),
        'pnl': round(sum(pnls), 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(_max_dd(daily), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1: Load data & build feature matrix
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("R137 — TFT Entry Prediction")
print("=" * 70)

print("\n[Phase 1] Loading data & building features...")

h1 = load_csv(str(H1_CSV))
print(f"  H1 bars: {len(h1)} ({h1.index[0].date()} to {h1.index[-1].date()})")

macro_df = None
if ALIGNED_CSV.exists():
    macro_df = pd.read_csv(str(ALIGNED_CSV), index_col=0, parse_dates=True)
    if macro_df.index.tz is None:
        macro_df.index = macro_df.index.tz_localize(None)
    else:
        macro_df.index = macro_df.index.tz_localize(None)
    print(f"  Macro data: {len(macro_df)} days, cols={list(macro_df.columns[:8])}")
else:
    print("  [WARN] Macro CSV not found, using synthetic proxies")

# H1 features
h1['ret_1'] = h1['Close'].pct_change(1)
h1['ret_4'] = h1['Close'].pct_change(4)
h1['ret_8'] = h1['Close'].pct_change(8)
h1['ret_24'] = h1['Close'].pct_change(24)
h1['ATR14'] = compute_atr(h1, 14)
h1['RSI14'] = compute_rsi(h1['Close'], 14)
h1['RSI2'] = compute_rsi(h1['Close'], 2)
h1['KC_pos'] = compute_kc_position(h1['Close'])
ema12 = h1['Close'].ewm(span=12).mean()
ema26 = h1['Close'].ewm(span=26).mean()
h1['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
h1['volume'] = h1['Volume'] if 'Volume' in h1.columns else 0

# D1 features (resampled from H1)
h1_daily = h1.resample('1D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
h1_daily['EMA20'] = h1_daily['Close'].ewm(span=20).mean()
h1_daily['daily_range'] = (h1_daily['High'] - h1_daily['Low']) / h1_daily['Close']
h1_daily['daily_mom'] = h1_daily['Close'].pct_change(5)
d1_features = h1_daily[['EMA20', 'daily_range', 'daily_mom']].copy()
d1_features['EMA20_trend'] = (h1_daily['Close'] - h1_daily['EMA20']) / h1_daily['EMA20']

# Merge D1 features into H1 (forward fill daily)
h1['date'] = h1.index.date
d1_features.index = d1_features.index.date
for col in ['EMA20_trend', 'daily_range', 'daily_mom']:
    h1[col] = h1['date'].map(d1_features[col].to_dict())

# Macro features
macro_cols = ['VIX', 'DXY', 'US10Y', 'real_yield']
if macro_df is not None:
    macro_daily = macro_df[macro_cols].copy() if all(c in macro_df.columns for c in macro_cols) else None
    if macro_daily is not None:
        macro_daily.index = macro_daily.index.date
        for col in macro_cols:
            h1[col] = h1['date'].map(macro_daily[col].to_dict())
    else:
        for col in macro_cols:
            h1[col] = 0.0
else:
    for col in macro_cols:
        h1[col] = 0.0

# Categorical features
h1['hour'] = h1.index.hour
h1['dow'] = h1.index.dayofweek

# Feature list
CONTINUOUS_FEATURES = [
    'ret_1', 'ret_4', 'ret_8', 'ret_24', 'ATR14', 'RSI14', 'RSI2',
    'KC_pos', 'MACD_hist', 'volume', 'EMA20_trend', 'daily_range',
    'daily_mom', 'VIX', 'DXY', 'US10Y', 'real_yield'
]
CATEGORICAL_FEATURES = ['hour', 'dow']
ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

h1 = h1.dropna(subset=CONTINUOUS_FEATURES)
h1[CONTINUOUS_FEATURES] = h1[CONTINUOUS_FEATURES].fillna(0)

# ═══════════════════════════════════════════════════════════════
# Phase 2: Target construction
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 2] Building targets...")

h1['fwd_ret'] = h1['Close'].shift(-FORWARD_BARS) / h1['Close'] - 1
h1['target_dir'] = (h1['fwd_ret'] > 0).astype(int)
h1['target_mag'] = h1['fwd_ret'].abs()

h1 = h1.dropna(subset=['fwd_ret'])
mag_quantiles = h1['target_mag'].quantile([0.2, 0.4, 0.6, 0.8]).values
h1['target_mag_q'] = np.digitize(h1['target_mag'].values, mag_quantiles) + 1

print(f"  Samples: {len(h1)}")
print(f"  Direction balance: {h1['target_dir'].mean():.3f} (up ratio)")
print(f"  Magnitude quintile distribution: {h1['target_mag_q'].value_counts().sort_index().to_dict()}")


# ═══════════════════════════════════════════════════════════════
# Phase 3: Model definition
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 3] Model definition...")

if HAS_TORCH:
    class SimpleTFT(nn.Module):
        """Simplified TFT: embeddings for categoricals + linear for continuous,
        multi-head attention over lookback window, dual output heads."""

        def __init__(self, n_continuous, n_hours=24, n_dows=7, embed_dim=8,
                     hidden_dim=64, n_heads=4, lookback=LOOKBACK):
            super().__init__()
            self.lookback = lookback
            self.hour_embed = nn.Embedding(n_hours, embed_dim)
            self.dow_embed = nn.Embedding(n_dows, embed_dim)
            input_dim = n_continuous + embed_dim * 2
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )
            self.dir_head = nn.Linear(hidden_dim // 2, 1)
            self.mag_head = nn.Linear(hidden_dim // 2, 1)

        def forward(self, x_cont, x_hour, x_dow):
            # x_cont: (batch, lookback, n_continuous)
            # x_hour, x_dow: (batch, lookback)
            h_emb = self.hour_embed(x_hour)
            d_emb = self.dow_embed(x_dow)
            x = torch.cat([x_cont, h_emb, d_emb], dim=-1)
            x = self.input_proj(x)
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            x = x[:, -1, :]  # take last timestep
            x = self.ff(x)
            dir_logit = self.dir_head(x).squeeze(-1)
            mag_pred = self.mag_head(x).squeeze(-1)
            return dir_logit, mag_pred

    def create_sequences(X_cont, X_hour, X_dow, y_dir, y_mag, lookback=LOOKBACK):
        """Create windowed sequences for TFT input."""
        n = len(X_cont)
        seqs_c, seqs_h, seqs_d, dirs, mags = [], [], [], [], []
        for i in range(lookback, n):
            seqs_c.append(X_cont[i - lookback:i])
            seqs_h.append(X_hour[i - lookback:i])
            seqs_d.append(X_dow[i - lookback:i])
            dirs.append(y_dir[i])
            mags.append(y_mag[i])
        return (np.array(seqs_c), np.array(seqs_h), np.array(seqs_d),
                np.array(dirs), np.array(mags))

    def train_tft(X_cont_train, X_hour_train, X_dow_train, y_dir_train, y_mag_train,
                  epochs=30, lr=1e-3, batch_size=256):
        """Train TFT model on training data."""
        sc_c, sh, sd, yd, ym = create_sequences(
            X_cont_train, X_hour_train, X_dow_train, y_dir_train, y_mag_train)
        if len(sc_c) < 100:
            return None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleTFT(n_continuous=X_cont_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        dataset = TensorDataset(
            torch.FloatTensor(sc_c), torch.LongTensor(sh), torch.LongTensor(sd),
            torch.FloatTensor(yd), torch.FloatTensor(ym))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch in loader:
                xc, xh, xd, yd_b, ym_b = [b.to(device) for b in batch]
                optimizer.zero_grad()
                dir_logit, mag_pred = model(xc, xh, xd)
                loss = bce(dir_logit, yd_b) + 0.5 * mse(mag_pred, ym_b)
                loss.backward()
                optimizer.step()
        model.eval()
        return model

    def predict_tft(model, X_cont_test, X_hour_test, X_dow_test):
        """Generate predictions from TFT model."""
        sc_c, sh, sd, _, _ = create_sequences(
            X_cont_test, X_hour_test, X_dow_test,
            np.zeros(len(X_cont_test)), np.zeros(len(X_cont_test)))
        if len(sc_c) == 0:
            return np.array([]), np.array([])
        device = next(model.parameters()).device
        with torch.no_grad():
            xc = torch.FloatTensor(sc_c).to(device)
            xh = torch.LongTensor(sh).to(device)
            xd = torch.LongTensor(sd).to(device)
            dir_logit, mag_pred = model(xc, xh, xd)
            dir_prob = torch.sigmoid(dir_logit).cpu().numpy()
            mag_val = mag_pred.cpu().numpy()
        return dir_prob, mag_val

    print("  TFT model: MultiheadAttention(4 heads), hidden=64, lookback=24")

else:
    def train_xgb_model(X_train, y_dir_train, y_mag_train, n_lags=8):
        """XGBoost fallback: add lag features for temporal context."""
        X_lagged = _add_lag_features(X_train, n_lags)
        valid = ~np.isnan(X_lagged).any(axis=1)
        X_lagged = X_lagged[valid]
        y_d = y_dir_train[valid]
        y_m = y_mag_train[valid]
        dir_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
            use_label_encoder=False, verbosity=0)
        dir_model.fit(X_lagged, y_d)
        mag_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbosity=0)
        mag_model.fit(X_lagged, y_m)
        return dir_model, mag_model

    def predict_xgb(dir_model, mag_model, X_test, n_lags=8):
        """Predict with XGBoost models."""
        X_lagged = _add_lag_features(X_test, n_lags)
        valid = ~np.isnan(X_lagged).any(axis=1)
        dir_prob = np.full(len(X_test), 0.5)
        mag_pred = np.full(len(X_test), 0.0)
        if valid.sum() > 0:
            dir_prob[valid] = dir_model.predict_proba(X_lagged[valid])[:, 1]
            mag_pred[valid] = mag_model.predict(X_lagged[valid])
        return dir_prob, mag_pred

    def _add_lag_features(X, n_lags):
        """Create lagged feature matrix for XGBoost temporal context."""
        n, d = X.shape
        X_out = np.full((n, d * (n_lags + 1)), np.nan)
        for lag in range(n_lags + 1):
            if lag == 0:
                X_out[:, :d] = X
            else:
                X_out[lag:, lag * d:(lag + 1) * d] = X[:-lag]
        return X_out

    print("  XGBoost fallback: 200 trees, lag features (n_lags=8)")


# ═══════════════════════════════════════════════════════════════
# Phase 4: Walk-Forward evaluation
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 4] Walk-Forward (6mo train / 2mo test, sliding 2mo)...")

X_all = h1[CONTINUOUS_FEATURES].values.astype(np.float32)
X_hour = h1['hour'].values.astype(np.int64)
X_dow = h1['dow'].values.astype(np.int64)
y_dir = h1['target_dir'].values.astype(np.float32)
y_mag = h1['target_mag'].values.astype(np.float32)
timestamps = h1.index

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Walk-forward windows
wf_start = pd.Timestamp('2019-01-01', tz='UTC')
wf_end = timestamps[-1]
train_months = 6
test_months = 2

results_wf = []
current_start = wf_start

while current_start + pd.DateOffset(months=train_months + test_months) <= wf_end:
    train_end = current_start + pd.DateOffset(months=train_months)
    test_end = train_end + pd.DateOffset(months=test_months)

    train_mask = (timestamps >= current_start) & (timestamps < train_end)
    test_mask = (timestamps >= train_end) & (timestamps < test_end)

    if train_mask.sum() < 200 or test_mask.sum() < 50:
        current_start += pd.DateOffset(months=test_months)
        continue

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train = X_all_scaled[train_idx]
    X_test = X_all_scaled[test_idx]
    y_dir_train = y_dir[train_idx]
    y_dir_test = y_dir[test_idx]
    y_mag_train = y_mag[train_idx]
    y_mag_test = y_mag[test_idx]

    if HAS_TORCH:
        X_hour_train = X_hour[train_idx]
        X_hour_test = X_hour[test_idx]
        X_dow_train = X_dow[train_idx]
        X_dow_test = X_dow[test_idx]
        model = train_tft(X_train, X_hour_train, X_dow_train, y_dir_train, y_mag_train)
        if model is None:
            current_start += pd.DateOffset(months=test_months)
            continue
        dir_prob, mag_pred = predict_tft(model, X_test, X_hour_test, X_dow_test)
        offset = LOOKBACK
    else:
        dir_model, mag_model = train_xgb_model(X_train, y_dir_train, y_mag_train)
        dir_prob, mag_pred = predict_xgb(dir_model, mag_model, X_test)
        offset = 8  # n_lags

    # Compute metrics (align predictions with targets)
    if len(dir_prob) > 0:
        y_dir_eval = y_dir_test[offset:] if HAS_TORCH else y_dir_test
        y_mag_eval = y_mag_test[offset:] if HAS_TORCH else y_mag_test
        min_len = min(len(dir_prob), len(y_dir_eval))
        dir_prob = dir_prob[:min_len]
        mag_pred = mag_pred[:min_len]
        y_dir_eval = y_dir_eval[:min_len]
        y_mag_eval = y_mag_eval[:min_len]

        try:
            auc = roc_auc_score(y_dir_eval, dir_prob)
        except ValueError:
            auc = 0.5
        ic = np.corrcoef(mag_pred, y_mag_eval)[0, 1] if len(mag_pred) > 10 else 0.0
        if np.isnan(ic):
            ic = 0.0
    else:
        auc = 0.5
        ic = 0.0

    period_str = f"{current_start.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}"
    results_wf.append({
        'period': period_str,
        'train_n': int(train_mask.sum()),
        'test_n': int(test_mask.sum()),
        'auc': round(auc, 4),
        'ic': round(ic, 4),
        'dir_prob_mean': round(float(dir_prob.mean()), 4) if len(dir_prob) > 0 else 0.5,
    })
    print(f"  {period_str}: AUC={auc:.4f}  IC={ic:.4f}  n_test={test_mask.sum()}")
    current_start += pd.DateOffset(months=test_months)

print(f"\n  Walk-Forward folds: {len(results_wf)}")
avg_auc = np.mean([r['auc'] for r in results_wf]) if results_wf else 0.5
avg_ic = np.mean([r['ic'] for r in results_wf]) if results_wf else 0.0
print(f"  Mean AUC: {avg_auc:.4f}  Mean IC: {avg_ic:.4f}")


# ═══════════════════════════════════════════════════════════════
# Phase 5: Full model — signal generation & backtest
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 5] Full model predictions & signal generation...")

# Train on 2015-2021, predict 2021-2026
split_date = pd.Timestamp('2021-01-01', tz='UTC')
train_mask = timestamps < split_date
test_mask = timestamps >= split_date

X_train_full = X_all_scaled[train_mask]
X_test_full = X_all_scaled[test_mask]
y_dir_train_full = y_dir[train_mask]
y_mag_train_full = y_mag[train_mask]
y_dir_test_full = y_dir[test_mask]
y_mag_test_full = y_mag[test_mask]
test_times = timestamps[test_mask]

if HAS_TORCH:
    model_full = train_tft(
        X_train_full, X_hour[train_mask], X_dow[train_mask],
        y_dir_train_full, y_mag_train_full, epochs=50)
    if model_full is not None:
        dir_prob_full, mag_pred_full = predict_tft(
            model_full, X_test_full, X_hour[test_mask], X_dow[test_mask])
        signal_offset = LOOKBACK
    else:
        dir_prob_full = np.full(test_mask.sum(), 0.5)
        mag_pred_full = np.zeros(test_mask.sum())
        signal_offset = 0
else:
    dir_model_full, mag_model_full = train_xgb_model(
        X_train_full, y_dir_train_full, y_mag_train_full)
    dir_prob_full, mag_pred_full = predict_xgb(dir_model_full, mag_model_full, X_test_full)
    signal_offset = 0

# Phase 6: Convert predictions to trading signals
print("\n[Phase 6] Signal filtering (P(dir) > 0.6 AND mag > median)...")

h1_test = h1[test_mask].iloc[signal_offset:].copy()
dp = dir_prob_full[:len(h1_test)]
mp = mag_pred_full[:len(h1_test)]

if len(dp) > 0:
    mag_median = np.median(mp[mp > 0]) if (mp > 0).sum() > 0 else np.median(mp)
    buy_signal = (dp > 0.6) & (mp > mag_median)
    sell_signal = (dp < 0.4) & (mp > mag_median)
    print(f"  Buy signals: {buy_signal.sum()}")
    print(f"  Sell signals: {sell_signal.sum()}")
    print(f"  Signal rate: {(buy_signal.sum() + sell_signal.sum()) / len(dp) * 100:.1f}%")
else:
    buy_signal = np.array([])
    sell_signal = np.array([])

# Phase 7: Backtest filtered signals vs baseline
print("\n[Phase 7] Backtest: ML-filtered vs baseline (all entries)...")

def backtest_signals(h1_slice, buy_mask, sell_mask, spread=SPREAD, lot=UNIT_LOT,
                     sl_atr=4.5, tp_atr=6.0, max_hold=20):
    """Simple backtest from directional signals with ATR-based exits."""
    df = h1_slice.copy()
    if 'ATR14' not in df.columns:
        df['ATR14'] = compute_atr(df, 14)
    c = df['Close'].values; hi = df['High'].values; lo = df['Low'].values
    atr = df['ATR14'].values; times = df.index
    n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            bars_held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_now = (c[i] - pos['entry'] - spread) * lot * PV
                if lo[i] <= pos['entry'] - pos['atr'] * sl_atr:
                    trades.append({'dir': 'BUY', 'entry': pos['entry'], 'exit': c[i],
                                   'entry_time': pos['time'], 'exit_time': times[i],
                                   'pnl': -pos['atr'] * sl_atr * lot * PV, 'reason': 'SL'})
                    pos = None; last_exit = i; continue
                if hi[i] >= pos['entry'] + pos['atr'] * tp_atr:
                    trades.append({'dir': 'BUY', 'entry': pos['entry'], 'exit': c[i],
                                   'entry_time': pos['time'], 'exit_time': times[i],
                                   'pnl': pos['atr'] * tp_atr * lot * PV, 'reason': 'TP'})
                    pos = None; last_exit = i; continue
                if bars_held >= max_hold:
                    trades.append({'dir': 'BUY', 'entry': pos['entry'], 'exit': c[i],
                                   'entry_time': pos['time'], 'exit_time': times[i],
                                   'pnl': pnl_now, 'reason': 'TimeExit'})
                    pos = None; last_exit = i; continue
            else:
                pnl_now = (pos['entry'] - c[i] - spread) * lot * PV
                if hi[i] >= pos['entry'] + pos['atr'] * sl_atr:
                    trades.append({'dir': 'SELL', 'entry': pos['entry'], 'exit': c[i],
                                   'entry_time': pos['time'], 'exit_time': times[i],
                                   'pnl': -pos['atr'] * sl_atr * lot * PV, 'reason': 'SL'})
                    pos = None; last_exit = i; continue
                if lo[i] <= pos['entry'] - pos['atr'] * tp_atr:
                    trades.append({'dir': 'SELL', 'entry': pos['entry'], 'exit': c[i],
                                   'entry_time': pos['time'], 'exit_time': times[i],
                                   'pnl': pos['atr'] * tp_atr * lot * PV, 'reason': 'TP'})
                    pos = None; last_exit = i; continue
                if bars_held >= max_hold:
                    trades.append({'dir': 'SELL', 'entry': pos['entry'], 'exit': c[i],
                                   'entry_time': pos['time'], 'exit_time': times[i],
                                   'pnl': pnl_now, 'reason': 'TimeExit'})
                    pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if i < len(buy_mask) and buy_mask[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif i < len(sell_mask) and sell_mask[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades

if len(dp) > 0:
    # ML-filtered backtest
    ml_trades = backtest_signals(h1_test, buy_signal, sell_signal)
    ml_stats = _compute_stats(ml_trades)

    # Baseline: enter on every bar where ret_1 gives a signal (random-like)
    np.random.seed(42)
    baseline_buy = np.random.rand(len(h1_test)) < (buy_signal.sum() / len(buy_signal)) * 2
    baseline_sell = np.random.rand(len(h1_test)) < (sell_signal.sum() / len(sell_signal)) * 2
    baseline_trades = backtest_signals(h1_test, baseline_buy, baseline_sell)
    baseline_stats = _compute_stats(baseline_trades)

    print(f"\n  ML-filtered: {ml_stats}")
    print(f"  Baseline:    {baseline_stats}")
else:
    ml_stats = _compute_stats([])
    baseline_stats = _compute_stats([])
    print("  [WARN] No predictions available for backtest")


# ═══════════════════════════════════════════════════════════════
# Phase 8: K-Fold validation
# ═══════════════════════════════════════════════════════════════
print("\n[Phase 8] K-Fold validation (5 folds)...")

n_folds = 5
fold_size = len(X_all_scaled) // n_folds
kfold_results = []

for fold in range(n_folds):
    test_start = fold * fold_size
    test_end_idx = min((fold + 1) * fold_size, len(X_all_scaled))
    train_indices = np.concatenate([np.arange(0, test_start), np.arange(test_end_idx, len(X_all_scaled))])
    test_indices = np.arange(test_start, test_end_idx)

    X_tr = X_all_scaled[train_indices]
    X_te = X_all_scaled[test_indices]
    y_dir_tr = y_dir[train_indices]
    y_dir_te = y_dir[test_indices]
    y_mag_tr = y_mag[train_indices]
    y_mag_te = y_mag[test_indices]

    if HAS_TORCH:
        model_k = train_tft(X_tr, X_hour[train_indices], X_dow[train_indices], y_dir_tr, y_mag_tr, epochs=20)
        if model_k is None:
            kfold_results.append({'fold': fold, 'auc': 0.5, 'ic': 0.0})
            continue
        dp_k, mp_k = predict_tft(model_k, X_te, X_hour[test_indices], X_dow[test_indices])
        y_dir_k = y_dir_te[LOOKBACK:]
        y_mag_k = y_mag_te[LOOKBACK:]
    else:
        dm_k, mm_k = train_xgb_model(X_tr, y_dir_tr, y_mag_tr)
        dp_k, mp_k = predict_xgb(dm_k, mm_k, X_te)
        y_dir_k = y_dir_te
        y_mag_k = y_mag_te

    min_len = min(len(dp_k), len(y_dir_k))
    dp_k = dp_k[:min_len]; y_dir_k = y_dir_k[:min_len]
    mp_k = mp_k[:min_len]; y_mag_k = y_mag_k[:min_len]

    try:
        auc_k = roc_auc_score(y_dir_k, dp_k)
    except ValueError:
        auc_k = 0.5
    ic_k = np.corrcoef(mp_k, y_mag_k)[0, 1] if len(mp_k) > 10 else 0.0
    if np.isnan(ic_k):
        ic_k = 0.0
    kfold_results.append({'fold': fold, 'auc': round(auc_k, 4), 'ic': round(ic_k, 4)})
    print(f"  Fold {fold}: AUC={auc_k:.4f}  IC={ic_k:.4f}")

kfold_mean_auc = np.mean([r['auc'] for r in kfold_results])
kfold_mean_ic = np.mean([r['ic'] for r in kfold_results])
print(f"\n  K-Fold Mean: AUC={kfold_mean_auc:.4f}  IC={kfold_mean_ic:.4f}")


# ═══════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"R137 complete in {elapsed:.1f}s")

results = {
    'experiment': 'R137_TFT_Entry',
    'model_type': 'TFT (PyTorch)' if HAS_TORCH else 'XGBoost (fallback)',
    'features': ALL_FEATURES,
    'n_features': len(ALL_FEATURES),
    'lookback': LOOKBACK,
    'forward_bars': FORWARD_BARS,
    'walk_forward': {
        'folds': results_wf,
        'mean_auc': round(avg_auc, 4),
        'mean_ic': round(avg_ic, 4),
    },
    'backtest': {
        'ml_filtered': ml_stats,
        'baseline': baseline_stats,
    },
    'kfold': {
        'folds': kfold_results,
        'mean_auc': round(kfold_mean_auc, 4),
        'mean_ic': round(kfold_mean_ic, 4),
    },
    'elapsed_s': round(elapsed, 1),
}

with open(OUTPUT_DIR / "r137_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR / 'r137_results.json'}")
print("=" * 70)
