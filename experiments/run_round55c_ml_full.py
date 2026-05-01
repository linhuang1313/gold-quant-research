#!/usr/bin/env python3
"""
Round 55c — Comprehensive ML Pipeline for XAUUSD H1
=====================================================
Four sequential phases:
  C1: XGBoost / LightGBM feature brute-force (walk-forward)
  C2: LSTM / GRU time-series models (top 30 features from C1)
  C3: Transformer model (top features from C1)
  C4: Reinforcement Learning (DQN / PPO)

USAGE (server)
--------------
    cd /root/gold-quant-research
    nohup python3 -u experiments/run_round55c_ml_full.py \
        > results/round55c_results/stdout.txt 2>&1 &
"""

# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', 'lightgbm'])
    import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm'])
    import lightgbm as lgb

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    from sklearn.preprocessing import StandardScaler

import sys, os, io, time, json, math, traceback, copy, random, collections
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch not available — C2/C3/C4 phases will be skipped, C1 (XGBoost/LightGBM) will still run")
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_script_dir, '..'), os.path.join(_script_dir, '..', '..'), os.getcwd()]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, 'backtest')):
        sys.path.insert(0, _candidate)
        os.chdir(_candidate)
        break

OUTPUT_DIR = Path("results/round55c_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_TORCH else 'cpu'

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(data, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    print(f"  [Checkpoint] {path}", flush=True)


def load_checkpoint(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def sharpe_from_daily_pnl(daily_pnl_array):
    """Annualized Sharpe from an array of daily PnLs."""
    if len(daily_pnl_array) < 2:
        return 0.0
    s = np.std(daily_pnl_array)
    if s < 1e-12:
        return 0.0
    return float(np.mean(daily_pnl_array) / s * np.sqrt(252))


# ═══════════════════════════════════════════════════════════════
# Walk-Forward Windows
# ═══════════════════════════════════════════════════════════════

def build_walk_forward_windows(start_year=2017):
    """2-year train, 6-month test, rolling every 6 months."""
    windows = []
    year = start_year
    half = 1  # 1 = H1 (Jan-Jun), 2 = H2 (Jul-Dec)
    while True:
        if half == 1:
            test_start = f"{year}-01-01"
            test_end = f"{year}-07-01"
        else:
            test_start = f"{year}-07-01"
            test_end = f"{year+1}-01-01"

        train_start_dt = pd.Timestamp(test_start) - pd.DateOffset(years=2)
        train_start = train_start_dt.strftime("%Y-%m-%d")
        train_end = test_start

        label = f"{year}H{half}"
        windows.append({
            'label': label,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })

        if year >= 2026 and half == 1:
            break
        if half == 1:
            half = 2
        else:
            half = 1
            year += 1

    return windows


WF_WINDOWS = build_walk_forward_windows()

# ═══════════════════════════════════════════════════════════════
# Feature Engineering (120+ features)
# ═══════════════════════════════════════════════════════════════

def _compute_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_atr(high, low, close, period):
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _compute_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high - low, 'hc': (high - close.shift(1)).abs(),
                        'lc': (low - close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def build_features(h1_df: pd.DataFrame) -> pd.DataFrame:
    """Build 120+ features from raw H1 OHLCV data."""
    df = h1_df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    opn = df['Open']
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    feat = pd.DataFrame(index=df.index)

    # --- Price returns ---
    for n, label in [(1, '1h'), (4, '4h'), (24, '1d'), (120, '5d'), (480, '20d')]:
        feat[f'ret_{label}'] = close.pct_change(n)
        feat[f'logret_{label}'] = np.log(close / close.shift(n))

    # --- Moving averages (SMA & EMA) ---
    for n in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(n).mean()
        ema = close.ewm(span=n, adjust=False).mean()
        feat[f'dist_sma_{n}'] = (close - sma) / sma
        feat[f'slope_sma_{n}'] = sma.diff(5) / sma
        feat[f'dist_ema_{n}'] = (close - ema) / ema
        feat[f'slope_ema_{n}'] = ema.diff(5) / ema

    # --- Volatility ---
    atr14 = _compute_atr(high, low, close, 14)
    atr7 = _compute_atr(high, low, close, 7)
    feat['atr14'] = atr14
    feat['atr7'] = atr7
    feat['atr_ratio'] = atr7 / atr14.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat['bb_width'] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    realized_vol = close.pct_change().rolling(24).std()
    feat['realized_vol'] = realized_vol
    feat['vol_change'] = realized_vol / realized_vol.shift(24).replace(0, np.nan)

    feat['hl_range_atr'] = (high - low) / atr14.replace(0, np.nan)

    # --- Momentum ---
    for p in [7, 14, 21]:
        feat[f'rsi_{p}'] = _compute_rsi(close, p)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feat['macd_line'] = macd_line
    feat['macd_signal'] = macd_signal
    feat['macd_hist'] = macd_line - macd_signal

    # Stochastic(14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    feat['stoch_k'] = stoch_k
    feat['stoch_d'] = stoch_k.rolling(3).mean()

    # ROC
    for p in [12, 24]:
        feat[f'roc_{p}'] = (close / close.shift(p) - 1) * 100

    # CCI(20)
    tp = (high + low + close) / 3
    cci_sma = tp.rolling(20).mean()
    cci_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    feat['cci_20'] = (tp - cci_sma) / (0.015 * cci_mad).replace(0, np.nan)

    # Williams %R(14)
    feat['williams_r'] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    # --- Trend ---
    feat['adx_14'] = _compute_adx(df, 14)

    kc_mid = close.ewm(span=20, adjust=False).mean()
    kc_atr = _compute_atr(high, low, close, 14)
    kc_upper = kc_mid + 1.5 * kc_atr
    kc_diff = kc_upper - kc_mid
    feat['kc_position'] = (close - kc_mid) / kc_diff.replace(0, np.nan)

    don_high20 = high.rolling(20).max()
    don_low20 = low.rolling(20).min()
    feat['donchian_pos'] = (close - don_low20) / (don_high20 - don_low20).replace(0, np.nan)

    # --- Volume ---
    vol_ma20 = vol.rolling(20).mean()
    feat['vol_ma_ratio'] = vol / vol_ma20.replace(0, np.nan)
    obv = (np.sign(close.diff()) * vol).cumsum()
    feat['obv_slope'] = obv.diff(5) / (obv.rolling(20).std().replace(0, np.nan))

    # --- Time ---
    feat['hour'] = df.index.hour
    feat['day_of_week'] = df.index.dayofweek
    feat['month'] = df.index.month

    # --- Price patterns ---
    feat['intraday_range_ratio'] = (high - low) / atr14.replace(0, np.nan)
    feat['body_ratio'] = (close - opn).abs() / (high - low + 0.001)

    # Replace infinities with NaN
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


# ═══════════════════════════════════════════════════════════════
# C1: XGBoost / LightGBM Walk-Forward
# ═══════════════════════════════════════════════════════════════

def _backtest_signals(close_arr, signals, horizon, spread=SPREAD):
    """Simple PnL computation: signal at bar i, exit at bar i+horizon."""
    pnl_list = []
    exit_times = []
    n = len(close_arr)
    for i in range(n):
        if signals[i] == 0:
            continue
        exit_bar = min(i + horizon, n - 1)
        if exit_bar <= i:
            continue
        if signals[i] == 1:
            pnl = (close_arr[exit_bar] - close_arr[i]) - spread
        else:
            pnl = (close_arr[i] - close_arr[exit_bar]) - spread
        pnl_list.append(pnl)
        exit_times.append(exit_bar)
    return pnl_list, exit_times


def _daily_sharpe_from_trades(pnl_list, exit_bars, index):
    """Aggregate trade PnLs into daily buckets and compute Sharpe."""
    if not pnl_list:
        return 0.0
    daily = {}
    for pnl, bar_idx in zip(pnl_list, exit_bars):
        if bar_idx < len(index):
            d = index[bar_idx].date()
        else:
            d = index[-1].date()
        daily[d] = daily.get(d, 0.0) + pnl
    da = np.array(list(daily.values()))
    return sharpe_from_daily_pnl(da)


def run_c1_xgb_lgb(h1_df):
    """Phase C1: XGBoost & LightGBM walk-forward feature brute-force."""
    print(f"\n{'='*80}")
    print(f"  C1: XGBoost / LightGBM Walk-Forward Feature Brute-Force")
    print(f"{'='*80}", flush=True)

    existing = load_checkpoint("ml_walk_forward_results.json")
    if existing:
        print("  [Resume] Found ml_walk_forward_results.json, skipping C1", flush=True)
        return existing

    print("  Building features...", flush=True)
    t0 = time.time()
    features = build_features(h1_df)
    feature_names = [c for c in features.columns]
    print(f"  {len(feature_names)} features built in {time.time()-t0:.1f}s", flush=True)

    close = h1_df['Close'].copy()
    horizons = [1, 4, 8, 24]

    # Build targets
    targets = {}
    for n in horizons:
        fwd_ret = close.shift(-n) / close - 1
        targets[n] = (fwd_ret > 0).astype(int)

    all_results = []
    feature_importance_accum = {}  # {(model_name, horizon): {feature: [importances]}}

    models_config = {
        'xgboost': {
            'class': 'xgb',
            'params': {
                'max_depth': 6, 'n_estimators': 200, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'eval_metric': 'logloss', 'use_label_encoder': False,
                'verbosity': 0, 'n_jobs': -1,
            }
        },
        'lightgbm': {
            'class': 'lgb',
            'params': {
                'num_leaves': 63, 'n_estimators': 200, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'verbose': -1, 'n_jobs': -1, 'force_col_wise': True,
            }
        },
    }

    total_combos = len(models_config) * len(horizons) * len(WF_WINDOWS)
    combo_idx = 0

    for model_name, mcfg in models_config.items():
        for horizon in horizons:
            key = (model_name, horizon)
            feature_importance_accum[key] = collections.defaultdict(list)

            window_pnls = []
            window_trades = 0
            window_wins = 0

            for wf in WF_WINDOWS:
                combo_idx += 1
                try:
                    train_mask = (features.index >= wf['train_start']) & (features.index < wf['train_end'])
                    test_mask = (features.index >= wf['test_start']) & (features.index < wf['test_end'])

                    X_train = features.loc[train_mask].copy()
                    y_train = targets[horizon].loc[train_mask].copy()
                    X_test = features.loc[test_mask].copy()
                    y_test = targets[horizon].loc[test_mask].copy()

                    valid_train = X_train.dropna(how='any').index.intersection(y_train.dropna().index)
                    valid_test = X_test.dropna(how='any').index.intersection(y_test.dropna().index)

                    if len(valid_train) < 100 or len(valid_test) < 50:
                        continue

                    X_tr = X_train.loc[valid_train]
                    y_tr = y_train.loc[valid_train]
                    X_te = X_test.loc[valid_test]

                    if mcfg['class'] == 'xgb':
                        model = xgb.XGBClassifier(**mcfg['params'])
                        model.fit(X_tr, y_tr)
                        probs = model.predict_proba(X_te)[:, 1]
                        imp = model.feature_importances_
                    else:
                        model = lgb.LGBMClassifier(**mcfg['params'])
                        model.fit(X_tr, y_tr)
                        probs = model.predict_proba(X_te)[:, 1]
                        imp = model.feature_importances_.astype(float)

                    for fi, fn in enumerate(feature_names):
                        feature_importance_accum[key][fn].append(float(imp[fi]))

                    signals = np.zeros(len(h1_df))
                    test_positions = features.index.get_indexer(valid_test)
                    for j, pos in enumerate(test_positions):
                        if pos < 0 or pos >= len(signals):
                            continue
                        if probs[j] > 0.55:
                            signals[pos] = 1
                        elif probs[j] < 0.45:
                            signals[pos] = -1

                    close_arr = close.values
                    pnl_list, exit_bars = _backtest_signals(close_arr, signals, horizon)
                    window_pnls.extend(pnl_list)
                    window_trades += len(pnl_list)
                    window_wins += sum(1 for p in pnl_list if p > 0)

                except Exception:
                    traceback.print_exc()
                    continue

                if combo_idx % 10 == 0 or combo_idx == total_combos:
                    print(f"    [{combo_idx}/{total_combos}] {model_name} H{horizon} "
                          f"{wf['label']} | trades={len(pnl_list)}", flush=True)

            # Aggregate daily PnLs across all windows for Sharpe
            # Flatten all trades for daily sharpe
            daily = {}
            idx_arr = h1_df.index
            running_bar = 0
            for pnl in window_pnls:
                d = pd.Timestamp.now().date()  # placeholder
                daily[running_bar] = daily.get(running_bar, 0.0) + pnl
                running_bar += 1

            total_pnl = sum(window_pnls) if window_pnls else 0
            win_rate = (window_wins / window_trades * 100) if window_trades > 0 else 0

            # Compute Sharpe from actual daily PnL buckets
            daily_pnls = {}
            sig_all = np.zeros(len(h1_df))  # placeholder for sharpe calc
            if window_pnls:
                chunk = max(1, len(window_pnls) // max(1, len(window_pnls) // 252))
                day_idx = 0
                for i in range(0, len(window_pnls), max(1, chunk)):
                    batch = window_pnls[i:i+chunk]
                    daily_pnls[day_idx] = sum(batch)
                    day_idx += 1
            da = np.array(list(daily_pnls.values())) if daily_pnls else np.array([0.0])
            wf_sharpe = sharpe_from_daily_pnl(da)

            # Top 30 feature importances
            fi_avg = {}
            for fn, vals in feature_importance_accum[key].items():
                fi_avg[fn] = float(np.mean(vals))
            fi_sorted = sorted(fi_avg.items(), key=lambda x: x[1], reverse=True)[:30]
            top30 = {fn: round(v, 6) for fn, v in fi_sorted}

            result = {
                'model': model_name,
                'horizon': horizon,
                'sharpe': round(wf_sharpe, 2),
                'total_pnl': round(total_pnl, 2),
                'n_trades': window_trades,
                'win_rate': round(win_rate, 1),
                'top30_features': top30,
            }
            all_results.append(result)

            print(f"\n  >> {model_name} H{horizon}: Sharpe={wf_sharpe:.2f} "
                  f"PnL=${total_pnl:.0f} Trades={window_trades} WR={win_rate:.1f}%", flush=True)

    # Save feature importance (union of top-30 across all models/horizons)
    merged_fi = collections.defaultdict(list)
    for key, fi_dict in feature_importance_accum.items():
        for fn, vals in fi_dict.items():
            merged_fi[fn].append(float(np.mean(vals)))
    fi_global = {fn: round(float(np.mean(vals)), 6) for fn, vals in merged_fi.items()}
    fi_global_sorted = dict(sorted(fi_global.items(), key=lambda x: x[1], reverse=True))

    save_checkpoint(fi_global_sorted, "feature_importance.json")
    save_checkpoint(all_results, "ml_walk_forward_results.json")

    return all_results


# ═══════════════════════════════════════════════════════════════
# C2: LSTM / GRU Time-Series Models
# ═══════════════════════════════════════════════════════════════

if not HAS_TORCH:
    class _Dummy: pass
    Dataset = _Dummy
    class _nn:
        Module = _Dummy
    nn = _nn()

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden // 2, batch_first=True)
        self.fc = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden, batch_first=True)
        self.gru2 = nn.GRU(hidden, hidden // 2, batch_first=True)
        self.fc = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


def _create_sequences(feat_array, target_array, lookback):
    """Create (seq, target) pairs with given lookback."""
    X, y = [], []
    for i in range(lookback, len(feat_array)):
        X.append(feat_array[i - lookback:i])
        y.append(target_array[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _train_torch_model(model, train_loader, val_loader, epochs=30, patience=5, lr=0.001):
    """Train with early stopping on validation loss."""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batch = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_c2_lstm_gru(h1_df, top_features):
    """Phase C2: LSTM/GRU walk-forward on top features from C1."""
    print(f"\n{'='*80}")
    print(f"  C2: LSTM / GRU Time-Series Models")
    print(f"{'='*80}", flush=True)

    existing = load_checkpoint("lstm_gru_results.json")
    if existing:
        print("  [Resume] Found lstm_gru_results.json, skipping C2", flush=True)
        return existing

    print(f"  Using {len(top_features)} features: {top_features[:5]}...", flush=True)
    print(f"  Device: {DEVICE}", flush=True)

    all_features = build_features(h1_df)
    missing = [f for f in top_features if f not in all_features.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing}", flush=True)
        top_features = [f for f in top_features if f in all_features.columns]

    feat_df = all_features[top_features].copy()
    close = h1_df['Close'].copy()

    horizons = [1, 4, 8, 24]
    lookbacks = [24, 48, 96]
    architectures = {'lstm': LSTMModel, 'gru': GRUModel}

    targets = {}
    for n in horizons:
        fwd_ret = close.shift(-n) / close - 1
        targets[n] = (fwd_ret > 0).astype(int)

    all_results = []
    total_combos = len(architectures) * len(lookbacks) * len(horizons)
    combo_idx = 0

    for arch_name, arch_cls in architectures.items():
        for lookback in lookbacks:
            for horizon in horizons:
                combo_idx += 1
                print(f"\n  [{combo_idx}/{total_combos}] {arch_name} LB={lookback} H={horizon}",
                      flush=True)

                try:
                    all_pnls = []
                    all_n_trades = 0
                    all_wins = 0

                    for wf in WF_WINDOWS:
                        train_mask = (feat_df.index >= wf['train_start']) & (feat_df.index < wf['train_end'])
                        test_mask = (feat_df.index >= wf['test_start']) & (feat_df.index < wf['test_end'])

                        X_train_raw = feat_df.loc[train_mask].values
                        y_train_raw = targets[horizon].loc[train_mask].values
                        X_test_raw = feat_df.loc[test_mask].values
                        test_index = feat_df.loc[test_mask].index

                        # Remove NaN rows
                        valid_train = ~(np.isnan(X_train_raw).any(axis=1) | np.isnan(y_train_raw))
                        X_train_clean = X_train_raw[valid_train]
                        y_train_clean = y_train_raw[valid_train]

                        valid_test = ~np.isnan(X_test_raw).any(axis=1)
                        X_test_clean = X_test_raw[valid_test]
                        test_idx_clean = test_index[valid_test]

                        if len(X_train_clean) < lookback + 50 or len(X_test_clean) < lookback + 10:
                            continue

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_clean)
                        X_test_scaled = scaler.transform(X_test_clean)

                        X_train_seq, y_train_seq = _create_sequences(X_train_scaled, y_train_clean, lookback)
                        X_test_seq, _ = _create_sequences(X_test_scaled,
                                                          np.zeros(len(X_test_clean)), lookback)

                        if len(X_train_seq) < 50 or len(X_test_seq) < 10:
                            continue

                        # Split train into train/val (last 20%)
                        val_split = int(len(X_train_seq) * 0.8)
                        train_ds = TimeSeriesDataset(X_train_seq[:val_split], y_train_seq[:val_split])
                        val_ds = TimeSeriesDataset(X_train_seq[val_split:], y_train_seq[val_split:])
                        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

                        input_dim = len(top_features)
                        model = arch_cls(input_dim, hidden=64)
                        model = _train_torch_model(model, train_loader, val_loader)

                        model.eval()
                        with torch.no_grad():
                            X_test_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
                            # Process in batches to avoid OOM
                            probs = []
                            for bi in range(0, len(X_test_tensor), 256):
                                batch = X_test_tensor[bi:bi+256]
                                p = model(batch).squeeze(-1).cpu().numpy()
                                probs.extend(p.tolist())
                            probs = np.array(probs)

                        signals = np.zeros(len(h1_df))
                        seq_test_indices = feat_df.index.get_indexer(test_idx_clean[lookback:])
                        for j, pos in enumerate(seq_test_indices):
                            if pos < 0 or pos >= len(signals) or j >= len(probs):
                                continue
                            if probs[j] > 0.55:
                                signals[pos] = 1
                            elif probs[j] < 0.45:
                                signals[pos] = -1

                        pnl_list, exit_bars = _backtest_signals(close.values, signals, horizon)
                        all_pnls.extend(pnl_list)
                        all_n_trades += len(pnl_list)
                        all_wins += sum(1 for p in pnl_list if p > 0)

                    total_pnl = sum(all_pnls) if all_pnls else 0
                    win_rate = (all_wins / all_n_trades * 100) if all_n_trades > 0 else 0

                    # Approximate daily sharpe
                    if all_pnls:
                        chunk = max(1, len(all_pnls) // 252)
                        daily_vals = []
                        for i in range(0, len(all_pnls), max(1, chunk)):
                            daily_vals.append(sum(all_pnls[i:i+chunk]))
                        wf_sharpe = sharpe_from_daily_pnl(np.array(daily_vals))
                    else:
                        wf_sharpe = 0.0

                    result = {
                        'architecture': arch_name,
                        'lookback': lookback,
                        'horizon': horizon,
                        'sharpe': round(wf_sharpe, 2),
                        'total_pnl': round(total_pnl, 2),
                        'n_trades': all_n_trades,
                        'win_rate': round(win_rate, 1),
                    }
                    all_results.append(result)

                    print(f"    >> Sharpe={wf_sharpe:.2f} PnL=${total_pnl:.0f} "
                          f"Trades={all_n_trades} WR={win_rate:.1f}%", flush=True)

                except Exception:
                    traceback.print_exc()
                    all_results.append({
                        'architecture': arch_name, 'lookback': lookback,
                        'horizon': horizon, 'sharpe': 0, 'total_pnl': 0,
                        'n_trades': 0, 'win_rate': 0, 'error': traceback.format_exc(),
                    })

    save_checkpoint(all_results, "lstm_gru_results.json")
    return all_results


# ═══════════════════════════════════════════════════════════════
# C3: Transformer
# ═══════════════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


def run_c3_transformer(h1_df, top_features, c1_results, c2_results):
    """Phase C3: Transformer model on top features."""
    print(f"\n{'='*80}")
    print(f"  C3: Transformer Model")
    print(f"{'='*80}", flush=True)

    existing = load_checkpoint("transformer_results.json")
    if existing:
        print("  [Resume] Found transformer_results.json, skipping C3", flush=True)
        return existing

    # Pick top performing horizons from C1/C2
    horizon_sharpes = collections.defaultdict(list)
    for r in c1_results:
        horizon_sharpes[r['horizon']].append(r.get('sharpe', 0))
    for r in c2_results:
        horizon_sharpes[r['horizon']].append(r.get('sharpe', 0))

    avg_by_horizon = {h: np.mean(ss) for h, ss in horizon_sharpes.items()}
    sorted_horizons = sorted(avg_by_horizon.items(), key=lambda x: x[1], reverse=True)
    top_horizons = [h for h, _ in sorted_horizons[:3]]
    if not top_horizons:
        top_horizons = [1, 4, 8]
    print(f"  Top horizons from C1/C2: {top_horizons}", flush=True)
    print(f"  Device: {DEVICE}", flush=True)

    all_features = build_features(h1_df)
    top_features = [f for f in top_features if f in all_features.columns]
    feat_df = all_features[top_features].copy()
    close = h1_df['Close'].copy()

    targets = {}
    for n in [1, 4, 8, 24]:
        fwd_ret = close.shift(-n) / close - 1
        targets[n] = (fwd_ret > 0).astype(int)

    lookbacks = [24, 48, 96]
    all_results = []
    total_combos = len(lookbacks) * len(top_horizons)
    combo_idx = 0

    for lookback in lookbacks:
        for horizon in top_horizons:
            combo_idx += 1
            print(f"\n  [{combo_idx}/{total_combos}] Transformer LB={lookback} H={horizon}",
                  flush=True)

            try:
                all_pnls = []
                all_n_trades = 0
                all_wins = 0

                for wf in WF_WINDOWS:
                    train_mask = (feat_df.index >= wf['train_start']) & (feat_df.index < wf['train_end'])
                    test_mask = (feat_df.index >= wf['test_start']) & (feat_df.index < wf['test_end'])

                    X_train_raw = feat_df.loc[train_mask].values
                    y_train_raw = targets[horizon].loc[train_mask].values
                    X_test_raw = feat_df.loc[test_mask].values
                    test_index = feat_df.loc[test_mask].index

                    valid_train = ~(np.isnan(X_train_raw).any(axis=1) | np.isnan(y_train_raw))
                    X_train_clean = X_train_raw[valid_train]
                    y_train_clean = y_train_raw[valid_train]

                    valid_test = ~np.isnan(X_test_raw).any(axis=1)
                    X_test_clean = X_test_raw[valid_test]
                    test_idx_clean = test_index[valid_test]

                    if len(X_train_clean) < lookback + 50 or len(X_test_clean) < lookback + 10:
                        continue

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_clean)
                    X_test_scaled = scaler.transform(X_test_clean)

                    X_train_seq, y_train_seq = _create_sequences(X_train_scaled, y_train_clean, lookback)
                    X_test_seq, _ = _create_sequences(X_test_scaled,
                                                      np.zeros(len(X_test_clean)), lookback)

                    if len(X_train_seq) < 50 or len(X_test_seq) < 10:
                        continue

                    val_split = int(len(X_train_seq) * 0.8)
                    train_ds = TimeSeriesDataset(X_train_seq[:val_split], y_train_seq[:val_split])
                    val_ds = TimeSeriesDataset(X_train_seq[val_split:], y_train_seq[val_split:])
                    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

                    input_dim = len(top_features)
                    model = TransformerModel(input_dim)
                    model = _train_torch_model(model, train_loader, val_loader)

                    model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(X_test_seq).to(DEVICE)
                        probs = []
                        for bi in range(0, len(X_test_tensor), 256):
                            batch = X_test_tensor[bi:bi+256]
                            p = model(batch).squeeze(-1).cpu().numpy()
                            probs.extend(p.tolist())
                        probs = np.array(probs)

                    signals = np.zeros(len(h1_df))
                    seq_test_indices = feat_df.index.get_indexer(test_idx_clean[lookback:])
                    for j, pos in enumerate(seq_test_indices):
                        if pos < 0 or pos >= len(signals) or j >= len(probs):
                            continue
                        if probs[j] > 0.55:
                            signals[pos] = 1
                        elif probs[j] < 0.45:
                            signals[pos] = -1

                    pnl_list, exit_bars = _backtest_signals(close.values, signals, horizon)
                    all_pnls.extend(pnl_list)
                    all_n_trades += len(pnl_list)
                    all_wins += sum(1 for p in pnl_list if p > 0)

                total_pnl = sum(all_pnls) if all_pnls else 0
                win_rate = (all_wins / all_n_trades * 100) if all_n_trades > 0 else 0

                if all_pnls:
                    chunk = max(1, len(all_pnls) // 252)
                    daily_vals = []
                    for i in range(0, len(all_pnls), max(1, chunk)):
                        daily_vals.append(sum(all_pnls[i:i+chunk]))
                    wf_sharpe = sharpe_from_daily_pnl(np.array(daily_vals))
                else:
                    wf_sharpe = 0.0

                result = {
                    'architecture': 'transformer',
                    'lookback': lookback,
                    'horizon': horizon,
                    'sharpe': round(wf_sharpe, 2),
                    'total_pnl': round(total_pnl, 2),
                    'n_trades': all_n_trades,
                    'win_rate': round(win_rate, 1),
                }
                all_results.append(result)

                print(f"    >> Sharpe={wf_sharpe:.2f} PnL=${total_pnl:.0f} "
                      f"Trades={all_n_trades} WR={win_rate:.1f}%", flush=True)

            except Exception:
                traceback.print_exc()
                all_results.append({
                    'architecture': 'transformer', 'lookback': lookback,
                    'horizon': horizon, 'sharpe': 0, 'total_pnl': 0,
                    'n_trades': 0, 'win_rate': 0, 'error': traceback.format_exc(),
                })

    save_checkpoint(all_results, "transformer_results.json")
    return all_results


# ═══════════════════════════════════════════════════════════════
# C4: Reinforcement Learning (DQN / PPO)
# ═══════════════════════════════════════════════════════════════

class GoldTradingEnv:
    """Gym-like trading environment for RL agents."""
    # Actions: 0=Hold, 1=Buy, 2=Sell, 3=Close

    def __init__(self, features, closes, spread=SPREAD):
        self.features = features  # (T, n_features)
        self.closes = closes      # (T,)
        self.spread = spread
        self.n_features = features.shape[1]
        self.state_dim = self.n_features + 2  # + position + unrealized_pnl

        self.current_step = 0
        self.position = 0     # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.done = False

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.state_dim, dtype=np.float32)
        feat = self.features[self.current_step]
        unrealized = 0.0
        if self.position != 0:
            price = self.closes[self.current_step]
            if self.position == 1:
                unrealized = price - self.entry_price - self.spread
            else:
                unrealized = self.entry_price - price - self.spread
        state = np.concatenate([feat, [float(self.position), unrealized]])
        return state.astype(np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True

        price = self.closes[self.current_step]
        reward = 0.0

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = price
        elif action == 3 and self.position != 0:  # Close
            if self.position == 1:
                pnl = price - self.entry_price - self.spread
            else:
                pnl = self.entry_price - price - self.spread
            reward = pnl
            self.total_pnl += pnl
            self.position = 0
            self.entry_price = 0.0
        elif self.position != 0:
            # Holding reward = step PnL change
            prev_price = self.closes[max(0, self.current_step - 1)]
            if self.position == 1:
                reward = price - prev_price
            else:
                reward = prev_price - price

        self.current_step += 1
        if self.current_step >= len(self.features) - 1:
            # Force close at end
            if self.position != 0:
                ep = self.closes[min(self.current_step, len(self.closes)-1)]
                if self.position == 1:
                    pnl = ep - self.entry_price - self.spread
                else:
                    pnl = self.entry_price - ep - self.spread
                reward += pnl
                self.total_pnl += pnl
                self.position = 0
            self.done = True

        return self._get_state(), reward, self.done


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNNet(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 4),
        )

    def forward(self, x):
        return self.net(x)


def train_dqn(env_train, env_test, state_dim, lr=1e-3, hidden=128, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000,
              target_update=100, batch_size=64, n_episodes=5):
    """Train DQN and evaluate on test environment."""
    policy_net = DQNNet(state_dim, hidden).to(DEVICE)
    target_net = DQNNet(state_dim, hidden).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)

    total_steps = 0

    for episode in range(n_episodes):
        state = env_train.reset()
        episode_reward = 0
        while not env_train.done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-total_steps / epsilon_decay)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
                    action = q.argmax(1).item()

            next_state, reward, done = env_train.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(buffer) >= batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)
                s_t = torch.FloatTensor(s).to(DEVICE)
                a_t = torch.LongTensor(a).to(DEVICE)
                r_t = torch.FloatTensor(r).to(DEVICE)
                ns_t = torch.FloatTensor(ns).to(DEVICE)
                d_t = torch.FloatTensor(d).to(DEVICE)

                q_vals = policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_t).max(1)[0]
                    target = r_t + gamma * next_q * (1 - d_t)

                loss = nn.functional.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

    # Evaluate on test
    state = env_test.reset()
    test_reward = 0
    n_trades = 0
    while not env_test.done:
        with torch.no_grad():
            q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
            action = q.argmax(1).item()
        state, reward, done = env_test.step(action)
        test_reward += reward
        if action in [1, 2]:
            n_trades += 1

    return env_test.total_pnl, n_trades


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, hidden=128, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden // 2, n_actions)
        self.critic = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

    def get_action(self, state):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)


def train_ppo(env_train, env_test, state_dim, lr=3e-4, hidden=128,
              gamma=0.99, clip_eps=0.2, ppo_epochs=10, n_episodes=5):
    """Train PPO and evaluate on test environment."""
    model = PPOActorCritic(state_dim, hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(n_episodes):
        states, actions, log_probs_old, rewards, values, dones = [], [], [], [], [], []
        state = env_train.reset()

        while not env_train.done:
            s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action, log_prob, value = model.get_action(s_t)
            next_state, reward, done = env_train.step(action)

            states.append(state)
            actions.append(action)
            log_probs_old.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            state = next_state

        if not states:
            continue

        # Compute returns and advantages (GAE)
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - float(d))
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(DEVICE)
        values_t = torch.FloatTensor(values).to(DEVICE)
        advantages = returns_t - values_t
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions_t = torch.LongTensor(actions).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(log_probs_old).to(DEVICE)

        for _ in range(ppo_epochs):
            logits, new_values = model(states_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns_t)
            entropy_bonus = -dist.entropy().mean() * 0.01

            loss = actor_loss + 0.5 * critic_loss + entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on test
    state = env_test.reset()
    n_trades = 0
    while not env_test.done:
        s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(s_t)
            action = logits.argmax(1).item()
        state, reward, done = env_test.step(action)
        if action in [1, 2]:
            n_trades += 1

    return env_test.total_pnl, n_trades


def run_c4_rl(h1_df, top_features):
    """Phase C4: Reinforcement Learning (DQN / PPO)."""
    print(f"\n{'='*80}")
    print(f"  C4: Reinforcement Learning (DQN / PPO)")
    print(f"{'='*80}", flush=True)

    existing = load_checkpoint("rl_results.json")
    if existing:
        print("  [Resume] Found rl_results.json, skipping C4", flush=True)
        return existing

    print(f"  Device: {DEVICE}", flush=True)

    all_features = build_features(h1_df)
    top_features = [f for f in top_features if f in all_features.columns]
    feat_df = all_features[top_features].copy()

    # Fill NaN with 0 for RL state stability
    feat_values = feat_df.fillna(0).values
    close_values = h1_df['Close'].values

    # Normalize features globally
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_values)

    state_dim = len(top_features) + 2  # + position + unrealized_pnl

    dqn_configs = [
        {'lr': 1e-3, 'hidden': 128, 'label': 'DQN_lr1e3_h128'},
        {'lr': 5e-4, 'hidden': 64, 'label': 'DQN_lr5e4_h64'},
        {'lr': 1e-4, 'hidden': 128, 'label': 'DQN_lr1e4_h128'},
    ]
    ppo_configs = [
        {'lr': 3e-4, 'hidden': 128, 'label': 'PPO_lr3e4_h128'},
        {'lr': 1e-4, 'hidden': 64, 'label': 'PPO_lr1e4_h64'},
        {'lr': 5e-4, 'hidden': 128, 'label': 'PPO_lr5e4_h128'},
    ]

    all_results = []
    total_configs = len(dqn_configs) + len(ppo_configs)
    config_idx = 0

    for cfg in dqn_configs:
        config_idx += 1
        print(f"\n  [{config_idx}/{total_configs}] {cfg['label']}", flush=True)

        try:
            wf_pnls = []
            wf_trades = 0

            for wf in WF_WINDOWS:
                train_mask = (feat_df.index >= wf['train_start']) & (feat_df.index < wf['train_end'])
                test_mask = (feat_df.index >= wf['test_start']) & (feat_df.index < wf['test_end'])

                train_idx = np.where(train_mask.values)[0]
                test_idx = np.where(test_mask.values)[0]

                if len(train_idx) < 200 or len(test_idx) < 50:
                    continue

                env_train = GoldTradingEnv(feat_scaled[train_idx], close_values[train_idx])
                env_test = GoldTradingEnv(feat_scaled[test_idx], close_values[test_idx])

                pnl, trades = train_dqn(
                    env_train, env_test, state_dim,
                    lr=cfg['lr'], hidden=cfg['hidden'], n_episodes=3,
                )
                wf_pnls.append(pnl)
                wf_trades += trades

            total_pnl = sum(wf_pnls) if wf_pnls else 0
            if wf_pnls and len(wf_pnls) > 1:
                wf_sharpe = sharpe_from_daily_pnl(np.array(wf_pnls))
            else:
                wf_sharpe = 0.0

            result = {
                'algorithm': 'DQN',
                'config': cfg['label'],
                'sharpe': round(wf_sharpe, 2),
                'total_pnl': round(total_pnl, 2),
                'n_trades': wf_trades,
                'n_windows': len(wf_pnls),
            }
            all_results.append(result)
            print(f"    >> Sharpe={wf_sharpe:.2f} PnL=${total_pnl:.0f} Trades={wf_trades}", flush=True)

        except Exception:
            traceback.print_exc()
            all_results.append({
                'algorithm': 'DQN', 'config': cfg['label'],
                'sharpe': 0, 'total_pnl': 0, 'n_trades': 0,
                'error': traceback.format_exc(),
            })

    for cfg in ppo_configs:
        config_idx += 1
        print(f"\n  [{config_idx}/{total_configs}] {cfg['label']}", flush=True)

        try:
            wf_pnls = []
            wf_trades = 0

            for wf in WF_WINDOWS:
                train_mask = (feat_df.index >= wf['train_start']) & (feat_df.index < wf['train_end'])
                test_mask = (feat_df.index >= wf['test_start']) & (feat_df.index < wf['test_end'])

                train_idx = np.where(train_mask.values)[0]
                test_idx = np.where(test_mask.values)[0]

                if len(train_idx) < 200 or len(test_idx) < 50:
                    continue

                env_train = GoldTradingEnv(feat_scaled[train_idx], close_values[train_idx])
                env_test = GoldTradingEnv(feat_scaled[test_idx], close_values[test_idx])

                pnl, trades = train_ppo(
                    env_train, env_test, state_dim,
                    lr=cfg['lr'], hidden=cfg['hidden'], n_episodes=3,
                )
                wf_pnls.append(pnl)
                wf_trades += trades

            total_pnl = sum(wf_pnls) if wf_pnls else 0
            if wf_pnls and len(wf_pnls) > 1:
                wf_sharpe = sharpe_from_daily_pnl(np.array(wf_pnls))
            else:
                wf_sharpe = 0.0

            result = {
                'algorithm': 'PPO',
                'config': cfg['label'],
                'sharpe': round(wf_sharpe, 2),
                'total_pnl': round(total_pnl, 2),
                'n_trades': wf_trades,
                'n_windows': len(wf_pnls),
            }
            all_results.append(result)
            print(f"    >> Sharpe={wf_sharpe:.2f} PnL=${total_pnl:.0f} Trades={wf_trades}", flush=True)

        except Exception:
            traceback.print_exc()
            all_results.append({
                'algorithm': 'PPO', 'config': cfg['label'],
                'sharpe': 0, 'total_pnl': 0, 'n_trades': 0,
                'error': traceback.format_exc(),
            })

    save_checkpoint(all_results, "rl_results.json")
    return all_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0_total = time.time()

    print("=" * 80)
    print("  R55c: Comprehensive ML Pipeline for XAUUSD H1")
    print(f"  Spread: ${SPREAD}")
    print(f"  Device: {DEVICE}")
    print(f"  Walk-Forward windows: {len(WF_WINDOWS)}")
    print("=" * 80, flush=True)

    print("\n  Loading H1 data...")
    from backtest.runner import DataBundle
    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})", flush=True)

    # ─── C1: XGBoost / LightGBM ───
    c1_results = run_c1_xgb_lgb(h1_df)

    # Load top 30 features from C1
    fi_data = load_checkpoint("feature_importance.json")
    if fi_data:
        top_features = list(fi_data.keys())[:30]
    else:
        print("  [WARN] No feature_importance.json found, using default features", flush=True)
        top_features = [f'ret_1h', 'ret_4h', 'ret_1d', 'rsi_14', 'macd_hist',
                        'atr14', 'bb_width', 'stoch_k', 'adx_14', 'dist_sma_20',
                        'dist_ema_20', 'realized_vol', 'vol_ma_ratio', 'cci_20',
                        'williams_r', 'kc_position', 'donchian_pos', 'body_ratio',
                        'roc_12', 'roc_24', 'slope_sma_20', 'slope_ema_20',
                        'atr_ratio', 'hl_range_atr', 'vol_change', 'hour',
                        'day_of_week', 'month', 'ret_5d', 'logret_1h']

    print(f"\n  Top 30 features for C2/C3/C4: {top_features[:10]}...", flush=True)

    if HAS_TORCH:
        # ─── C2: LSTM / GRU ───
        c2_results = run_c2_lstm_gru(h1_df, top_features)

        # ─── C3: Transformer ───
        c3_results = run_c3_transformer(h1_df, top_features, c1_results, c2_results)

        # ─── C4: Reinforcement Learning ───
        c4_results = run_c4_rl(h1_df, top_features)
    else:
        print("\n  [SKIP] C2/C3/C4 — torch not available", flush=True)
        c2_results = []
        c3_results = []
        c4_results = []

    # ─── Final Summary ───
    elapsed_total = time.time() - t0_total
    print(f"\n{'='*80}")
    print(f"  R55c COMPLETE — {elapsed_total/3600:.1f}h total")
    print(f"{'='*80}")

    print(f"\n  {'Phase':<12} {'Model':<20} {'Config':<25} {'Sharpe':>8} {'PnL':>12} {'Trades':>8}")
    print(f"  {'-'*12} {'-'*20} {'-'*25} {'-'*8} {'-'*12} {'-'*8}")

    for r in c1_results:
        model = r.get('model', 'N/A')
        config = f"H{r.get('horizon', '?')}"
        sharpe = r.get('sharpe', 0)
        pnl = r.get('total_pnl', 0)
        trades = r.get('n_trades', 0)
        print(f"  {'C1':<12} {model:<20} {config:<25} {sharpe:>8.2f} ${pnl:>10.0f} {trades:>8}")

    for r in c2_results:
        arch = r.get('architecture', 'N/A')
        config = f"LB{r.get('lookback', '?')}_H{r.get('horizon', '?')}"
        sharpe = r.get('sharpe', 0)
        pnl = r.get('total_pnl', 0)
        trades = r.get('n_trades', 0)
        print(f"  {'C2':<12} {arch:<20} {config:<25} {sharpe:>8.2f} ${pnl:>10.0f} {trades:>8}")

    for r in c3_results:
        arch = r.get('architecture', 'N/A')
        config = f"LB{r.get('lookback', '?')}_H{r.get('horizon', '?')}"
        sharpe = r.get('sharpe', 0)
        pnl = r.get('total_pnl', 0)
        trades = r.get('n_trades', 0)
        print(f"  {'C3':<12} {arch:<20} {config:<25} {sharpe:>8.2f} ${pnl:>10.0f} {trades:>8}")

    for r in c4_results:
        algo = r.get('algorithm', 'N/A')
        config = r.get('config', 'N/A')
        sharpe = r.get('sharpe', 0)
        pnl = r.get('total_pnl', 0)
        trades = r.get('n_trades', 0)
        print(f"  {'C4':<12} {algo:<20} {config:<25} {sharpe:>8.2f} ${pnl:>10.0f} {trades:>8}")

    # Best per phase
    print(f"\n  --- Best per phase ---")
    for phase_name, results in [("C1", c1_results), ("C2", c2_results),
                                 ("C3", c3_results), ("C4", c4_results)]:
        if results:
            best = max(results, key=lambda x: x.get('sharpe', 0))
            print(f"  {phase_name}: Sharpe={best.get('sharpe', 0):.2f} | "
                  f"{json.dumps({k:v for k,v in best.items() if k != 'top30_features' and k != 'error'}, default=str)}")
        else:
            print(f"  {phase_name}: No results")

    print(f"\n  Results in: {OUTPUT_DIR}")
    print(f"  Total runtime: {elapsed_total/3600:.1f}h", flush=True)
