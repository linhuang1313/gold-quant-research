"""
M1 微观结构 Alpha Research
============================
目标: 从数据出发, 找到 M1 级别真正有预测力的特征,
然后用 XGBoost 建模, 最后用 Walk-Forward 回测。

Phase 1: Alpha Research (本脚本)
  - 构建 50+ 微观特征
  - 对每个特征计算 IC (信息系数)
  - 条件概率分析: 特征极端时, 未来 N bars 价格方向的概率
  - 找出 IC > 0.02 的有效因子

Phase 2: ML Model (本脚本)
  - 用有效因子训练 XGBoost
  - Walk-Forward: 训练 6 个月, 预测 1 个月, 滚动
  - 输出信号: BUY / SELL / FLAT + 置信度

Phase 3: Backtest (本脚本)
  - 用 ML 信号驱动 scalper
  - TP/SL 自适应
  - 完整绩效分析
"""
import sys
import time
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.engine import TradeRecord
from backtest.stats import aggregate_daily_pnl


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_m1(path, start=None, end=None):
    df = pd.read_csv(path)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.000')
    df = df.set_index('Gmt time').sort_index()
    if start: df = df[df.index >= start]
    if end: df = df[df.index <= end]
    return df


# ═══════════════════════════════════════════════════════════════
# Feature Engineering (50+ features)
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建 M1 微观结构特征。"""
    f = pd.DataFrame(index=df.index)
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    o = df['Open'].values
    v = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))

    # --- 价格动量 ---
    for n in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
        f[f'ret_{n}'] = pd.Series(c, index=df.index).pct_change(n)
        f[f'move_{n}'] = pd.Series(c - np.roll(c, n), index=df.index)

    # --- 波动率 ---
    for n in [5, 10, 20, 30, 60]:
        f[f'vol_{n}'] = pd.Series(c, index=df.index).rolling(n).std()
        f[f'range_{n}'] = pd.Series(h - lo, index=df.index).rolling(n).mean()
        f[f'range_ratio_{n}'] = (h - lo) / f[f'range_{n}'].values

    # --- 价格偏离 ---
    for n in [10, 20, 30, 60]:
        sma = pd.Series(c, index=df.index).rolling(n).mean()
        std = pd.Series(c, index=df.index).rolling(n).std()
        f[f'zscore_{n}'] = (pd.Series(c, index=df.index) - sma) / std.replace(0, np.nan)
        f[f'dist_sma_{n}'] = pd.Series(c, index=df.index) - sma

    # --- EMA 特征 ---
    for fast, slow in [(3, 8), (5, 13), (8, 21)]:
        ema_f = pd.Series(c, index=df.index).ewm(span=fast, adjust=False).mean()
        ema_s = pd.Series(c, index=df.index).ewm(span=slow, adjust=False).mean()
        f[f'ema_diff_{fast}_{slow}'] = ema_f - ema_s
        f[f'ema_slope_{fast}'] = ema_f.diff()

    # --- 高低价特征 ---
    for n in [10, 20, 30, 60]:
        hh = pd.Series(h, index=df.index).rolling(n).max()
        ll = pd.Series(lo, index=df.index).rolling(n).min()
        rng = hh - ll
        f[f'price_pos_{n}'] = (pd.Series(c, index=df.index) - ll) / rng.replace(0, np.nan)
        f[f'dist_high_{n}'] = hh - pd.Series(c, index=df.index)
        f[f'dist_low_{n}'] = pd.Series(c, index=df.index) - ll

    # --- Bar 特征 ---
    f['bar_size'] = pd.Series(h - lo, index=df.index)
    f['bar_body'] = pd.Series(np.abs(c - o), index=df.index)
    f['bar_body_ratio'] = f['bar_body'] / f['bar_size'].replace(0, np.nan)
    f['bar_direction'] = pd.Series(np.sign(c - o), index=df.index)
    f['upper_wick'] = pd.Series(h - np.maximum(c, o), index=df.index)
    f['lower_wick'] = pd.Series(np.minimum(c, o) - lo, index=df.index)
    f['wick_ratio'] = (f['upper_wick'] - f['lower_wick']) / f['bar_size'].replace(0, np.nan)

    # --- 连续方向 ---
    direction = pd.Series(np.sign(c - o), index=df.index)
    consec = direction.copy()
    for i in range(1, len(consec)):
        if consec.iloc[i] == consec.iloc[i - 1] and consec.iloc[i] != 0:
            consec.iloc[i] = consec.iloc[i - 1] + np.sign(consec.iloc[i])
    f['consecutive_dir'] = consec

    # --- RSI ---
    for n in [5, 10, 14]:
        delta = pd.Series(c, index=df.index).diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        f[f'rsi_{n}'] = 100 - 100 / (1 + rs)

    # --- 时间特征 ---
    f['hour'] = df.index.hour
    f['minute'] = df.index.minute
    f['day_of_week'] = df.index.dayofweek
    f['is_asia'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    f['is_london'] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
    f['is_ny'] = ((df.index.hour >= 13) & (df.index.hour < 22)).astype(int)

    # --- 累计日内收益 ---
    dates = df.index.date
    day_open = pd.Series(o, index=df.index).groupby(dates).transform('first')
    f['intraday_ret'] = (pd.Series(c, index=df.index) - day_open) / day_open

    # --- 成交量相关 (如果有) ---
    if v.sum() > 0:
        for n in [5, 10, 20]:
            f[f'vol_ratio_{n}'] = pd.Series(v, index=df.index) / pd.Series(v, index=df.index).rolling(n).mean()

    return f


# ═══════════════════════════════════════════════════════════════
# Label Construction
# ═══════════════════════════════════════════════════════════════

def build_labels(df: pd.DataFrame, tp_points: float = 2.0, sl_points: float = 3.0,
                 max_bars: int = 30, spread: float = 0.30) -> pd.DataFrame:
    """
    构建标签: 在未来 max_bars 内, 先触达 TP 还是 SL?

    返回:
      label_buy:  1=买入盈利(先达TP), 0=买入亏损(先达SL/-timeout)
      label_sell: 1=卖出盈利, 0=卖出亏损
      best_dir:   1=BUY好, -1=SELL好, 0=都不好
    """
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    n = len(df)

    label_buy = np.full(n, np.nan)
    label_sell = np.full(n, np.nan)
    buy_pnl = np.full(n, np.nan)
    sell_pnl = np.full(n, np.nan)

    for i in range(n - max_bars):
        entry = c[i]

        # BUY scenario
        buy_entry = entry + spread / 2
        buy_tp_px = buy_entry + tp_points
        buy_sl_px = buy_entry - sl_points
        buy_result = 0
        buy_p = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if h[j] >= buy_tp_px:
                buy_result = 1
                buy_p = tp_points * 100 * 0.01
                break
            if lo[j] <= buy_sl_px:
                buy_result = 0
                buy_p = -sl_points * 100 * 0.01
                break
        else:
            buy_p = (c[min(i + max_bars, n - 1)] - buy_entry - spread / 2) * 100 * 0.01
            buy_result = 1 if buy_p > 0 else 0

        # SELL scenario
        sell_entry = entry - spread / 2
        sell_tp_px = sell_entry - tp_points
        sell_sl_px = sell_entry + sl_points
        sell_result = 0
        sell_p = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lo[j] <= sell_tp_px:
                sell_result = 1
                sell_p = tp_points * 100 * 0.01
                break
            if h[j] >= sell_sl_px:
                sell_result = 0
                sell_p = -sl_points * 100 * 0.01
                break
        else:
            sell_p = (sell_entry - c[min(i + max_bars, n - 1)] - spread / 2) * 100 * 0.01
            sell_result = 1 if sell_p > 0 else 0

        label_buy[i] = buy_result
        label_sell[i] = sell_result
        buy_pnl[i] = buy_p
        sell_pnl[i] = sell_p

    labels = pd.DataFrame({
        'label_buy': label_buy,
        'label_sell': label_sell,
        'buy_pnl': buy_pnl,
        'sell_pnl': sell_pnl,
    }, index=df.index)

    labels['best_dir'] = 0
    labels.loc[(labels['label_buy'] == 1) & (labels['label_sell'] == 0), 'best_dir'] = 1
    labels.loc[(labels['label_sell'] == 1) & (labels['label_buy'] == 0), 'best_dir'] = -1

    return labels


# ═══════════════════════════════════════════════════════════════
# Alpha Research: IC Analysis
# ═══════════════════════════════════════════════════════════════

def alpha_ic_analysis(features: pd.DataFrame, labels: pd.DataFrame):
    """计算每个特征与 label 的 IC (Spearman相关系数)。"""
    print("\n" + "=" * 60)
    print("Alpha IC Analysis")
    print("=" * 60)

    target_buy = labels['label_buy'].dropna()
    target_sell = labels['label_sell'].dropna()

    results = []
    for col in features.columns:
        feat = features[col].reindex(target_buy.index).dropna()
        common = feat.index.intersection(target_buy.dropna().index)
        if len(common) < 1000:
            continue

        ic_buy = feat.loc[common].corr(target_buy.loc[common], method='spearman')
        ic_sell = feat.loc[common].corr(target_sell.loc[common], method='spearman')
        ic_best = max(abs(ic_buy), abs(ic_sell))

        results.append({
            'feature': col,
            'ic_buy': ic_buy,
            'ic_sell': ic_sell,
            'ic_abs_max': ic_best,
            'direction': 'BUY' if abs(ic_buy) > abs(ic_sell) else 'SELL',
        })

    results.sort(key=lambda x: -x['ic_abs_max'])

    print(f"\n  {'Feature':<30} {'IC_Buy':>8} {'IC_Sell':>8} {'|IC|_Max':>8} {'Best':>5}")
    print(f"  {'-'*65}")
    for r in results[:40]:
        marker = " ***" if r['ic_abs_max'] > 0.03 else " **" if r['ic_abs_max'] > 0.02 else ""
        print(f"  {r['feature']:<30} {r['ic_buy']:>8.4f} {r['ic_sell']:>8.4f} {r['ic_abs_max']:>8.4f} {r['direction']:>5}{marker}")

    good = [r for r in results if r['ic_abs_max'] > 0.015]
    print(f"\n  Features with |IC| > 0.015: {len(good)} / {len(results)}")
    return results


# ═══════════════════════════════════════════════════════════════
# ML Pipeline: Walk-Forward XGBoost
# ═══════════════════════════════════════════════════════════════

def train_ml_model(df: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame,
                   top_features: List[str], train_months: int = 6, test_months: int = 1):
    """
    Walk-Forward ML 训练。

    流程:
      1. 滚动窗口: 训练 6 个月 → 预测 1 个月 → 滑动
      2. 两个模型: buy_model (预测做多TP概率), sell_model (预测做空TP概率)
      3. 信号: max(p_buy, p_sell) > threshold → 入场
    """
    import xgboost as xgb

    print("\n" + "=" * 60)
    print("Walk-Forward ML Training")
    print("=" * 60)

    feat_cols = top_features
    X = features[feat_cols].copy()
    y_buy = labels['label_buy'].copy()
    y_sell = labels['label_sell'].copy()

    # Drop NaN
    valid = X.dropna().index.intersection(y_buy.dropna().index)
    X = X.loc[valid]
    y_buy = y_buy.loc[valid]
    y_sell = y_sell.loc[valid]

    print(f"  Valid samples: {len(X):,}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Buy rate: {y_buy.mean():.3f}, Sell rate: {y_sell.mean():.3f}")

    # Walk-Forward splits
    dates = X.index.to_series()
    min_date = dates.min()
    max_date = dates.max()

    predictions = pd.DataFrame(index=X.index, columns=['p_buy', 'p_sell'], dtype=float)
    predictions[:] = np.nan

    current = min_date + pd.DateOffset(months=train_months)
    fold = 0

    xgb_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    while current < max_date:
        train_end = current
        test_end = current + pd.DateOffset(months=test_months)

        train_mask = (dates >= min_date) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        if len(X_train) < 5000 or len(X_test) < 500:
            current += pd.DateOffset(months=test_months)
            continue

        y_buy_train = y_buy.loc[train_mask]
        y_sell_train = y_sell.loc[train_mask]

        # Train buy model
        model_buy = xgb.XGBClassifier(**xgb_params)
        model_buy.fit(X_train, y_buy_train, eval_set=[(X_test, y_buy.loc[test_mask])],
                      verbose=False)

        # Train sell model
        model_sell = xgb.XGBClassifier(**xgb_params)
        model_sell.fit(X_train, y_sell_train, eval_set=[(X_test, y_sell.loc[test_mask])],
                       verbose=False)

        # Predict
        p_buy = model_buy.predict_proba(X_test)[:, 1]
        p_sell = model_sell.predict_proba(X_test)[:, 1]

        predictions.loc[test_mask, 'p_buy'] = p_buy
        predictions.loc[test_mask, 'p_sell'] = p_sell

        # Stats
        y_buy_test = y_buy.loc[test_mask]
        y_sell_test = y_sell.loc[test_mask]
        acc_buy = accuracy_score(y_buy_test, (p_buy > 0.5).astype(int))
        acc_sell = accuracy_score(y_sell_test, (p_sell > 0.5).astype(int))

        try:
            auc_buy = roc_auc_score(y_buy_test, p_buy)
            auc_sell = roc_auc_score(y_sell_test, p_sell)
        except:
            auc_buy = auc_sell = 0.5

        fold += 1
        period = f"{train_end.strftime('%Y-%m')} -> {min(test_end, max_date).strftime('%Y-%m')}"
        print(f"  Fold {fold:>2}: {period}  "
              f"Buy(Acc={acc_buy:.3f} AUC={auc_buy:.3f}) "
              f"Sell(Acc={acc_sell:.3f} AUC={auc_sell:.3f}) "
              f"N={len(X_test):,}")

        current += pd.DateOffset(months=test_months)

    # Feature importance (from last fold)
    print(f"\n  Top 15 Features (Buy model):")
    imp = pd.Series(model_buy.feature_importances_, index=feat_cols).sort_values(ascending=False)
    for feat, score in imp.head(15).items():
        print(f"    {feat:<30} {score:.4f}")

    print(f"\n  Top 15 Features (Sell model):")
    imp_s = pd.Series(model_sell.feature_importances_, index=feat_cols).sort_values(ascending=False)
    for feat, score in imp_s.head(15).items():
        print(f"    {feat:<30} {score:.4f}")

    return predictions


# ═══════════════════════════════════════════════════════════════
# ML-Driven Backtest
# ═══════════════════════════════════════════════════════════════

def ml_backtest(df: pd.DataFrame, predictions: pd.DataFrame,
                tp: float = 2.0, sl: float = 3.5, hard_sl: float = 8.0,
                threshold: float = 0.55, spread: float = 0.30,
                cooldown: int = 3, max_hold: int = 30,
                session_hours: set = None, max_per_day: int = 20,
                scale_after: int = 4, lots: tuple = (0.01, 0.02, 0.03),
                ) -> Tuple[List[TradeRecord], List[float]]:
    """
    用 ML 预测概率驱动 scalper。

    入场: p_buy > threshold → BUY, p_sell > threshold → SELL
         如果两个都 > threshold, 选概率更高的
    出场: TP / SL / Timeout
    """
    if session_hours is None:
        session_hours = set(range(0, 13))

    valid = predictions.dropna().index
    idx_set = set(valid)

    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    times = df.index
    n = len(df)

    # Map time to index
    time_to_idx = {t: i for i, t in enumerate(times)}

    trades = []
    equity = [2000.0]
    pos = None
    last_close_idx = -999
    consec_wins = 0
    lot_idx = 0
    daily_cnt = {}

    for i in range(n):
        ts = times[i]
        c = close[i]; h_val = high[i]; lo_val = low[i]
        day = str(ts.date())

        # ── Exit ──
        if pos is not None:
            p = pos
            lm = p['lots'] / 0.01

            if p['dir'] == 'BUY':
                pnl_best = (h_val - p['entry'] - spread) * p['lots'] * 100
                pnl_worst = (lo_val - p['entry'] - spread) * p['lots'] * 100
                pnl_c = (c - p['entry'] - spread) * p['lots'] * 100
            else:
                pnl_best = (p['entry'] - lo_val - spread) * p['lots'] * 100
                pnl_worst = (p['entry'] - h_val - spread) * p['lots'] * 100
                pnl_c = (p['entry'] - c - spread) * p['lots'] * 100

            tp_lvl = tp * lm
            sl_lvl = sl * lm
            hsl_lvl = hard_sl * lm
            held = i - p['idx']

            if pnl_best >= tp_lvl:
                _add_trade(trades, equity, p, c, ts, "TP", i, tp_lvl)
                pos = None; last_close_idx = i; consec_wins += 1
                if consec_wins >= scale_after:
                    lot_idx = min(lot_idx + 1, len(lots) - 1)
                continue
            if pnl_worst <= -sl_lvl:
                _add_trade(trades, equity, p, c, ts, "SL", i, -sl_lvl)
                pos = None; last_close_idx = i; consec_wins = 0
                lot_idx = max(0, lot_idx - 1)
                continue
            if pnl_worst <= -hsl_lvl:
                _add_trade(trades, equity, p, c, ts, "HardSL", i, -hsl_lvl)
                pos = None; last_close_idx = i; consec_wins = 0; lot_idx = 0
                continue
            if held >= max_hold:
                _add_trade(trades, equity, p, c, ts, "Timeout", i, pnl_c)
                pos = None; last_close_idx = i
                if pnl_c > 0: consec_wins += 1
                else: consec_wins = 0; lot_idx = max(0, lot_idx - 1)
                continue

        # ── Entry ──
        if pos is not None: continue
        if i - last_close_idx < cooldown: continue
        if ts.hour not in session_hours: continue
        if daily_cnt.get(day, 0) >= max_per_day: continue
        if ts not in idx_set: continue

        row = predictions.loc[ts]
        p_buy = row['p_buy']
        p_sell = row['p_sell']

        if pd.isna(p_buy) or pd.isna(p_sell):
            continue

        direction = None
        if p_buy > threshold and p_sell > threshold:
            direction = 'BUY' if p_buy > p_sell else 'SELL'
        elif p_buy > threshold:
            direction = 'BUY'
        elif p_sell > threshold:
            direction = 'SELL'

        if direction is None:
            continue

        entry_px = c + spread / 2 if direction == 'BUY' else c - spread / 2
        pos = {
            'dir': direction,
            'entry': entry_px,
            'time': ts,
            'lots': lots[lot_idx],
            'idx': i,
        }
        daily_cnt[day] = daily_cnt.get(day, 0) + 1

    if pos:
        pnl = ((close[-1] - pos['entry'] - spread) if pos['dir'] == 'BUY'
               else (pos['entry'] - close[-1] - spread)) * pos['lots'] * 100
        _add_trade(trades, equity, pos, close[-1], times[-1], "EOD", n - 1, pnl)

    return trades, equity


def _add_trade(trades, equity, pos, ep, ts, reason, idx, pnl):
    trades.append(TradeRecord(
        strategy="MLScalp", direction=pos['dir'],
        entry_price=pos['entry'], exit_price=ep,
        entry_time=pos['time'], exit_time=ts,
        lots=pos['lots'], pnl=pnl, exit_reason=reason,
        bars_held=idx - pos['idx'],
    ))
    equity.append(equity[-1] + pnl)


def report(trades, equity, label):
    if not trades:
        print(f"  {label}: No trades"); return {}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    wr = len(wins) / len(pnls) * 100
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0
    daily = aggregate_daily_pnl(trades)
    sh = 0
    if len(daily) > 1 and np.std(daily, ddof=1) > 0:
        sh = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252)
    pk = equity[0]; dd = 0
    for e in equity:
        if e > pk: pk = e
        if pk - e > dd: dd = pk - e

    by_reason = {}
    for t in trades:
        r = t.exit_reason
        if r not in by_reason: by_reason[r] = {'n': 0, 'pnl': 0}
        by_reason[r]['n'] += 1; by_reason[r]['pnl'] += t.pnl

    by_lots = {}
    for t in trades:
        k = f"{t.lots:.2f}"
        if k not in by_lots: by_lots[k] = {'n': 0, 'pnl': 0, 'w': 0}
        by_lots[k]['n'] += 1; by_lots[k]['pnl'] += t.pnl
        if t.pnl > 0: by_lots[k]['w'] += 1

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades):,}")
    print(f"  Total PnL: ${total:,.2f}")
    print(f"  Sharpe: {sh:.2f}")
    print(f"  Win Rate: {wr:.1f}%")
    rr = avg_w / avg_l if avg_l > 0 else 0
    print(f"  Avg Win: ${avg_w:.2f} | Avg Loss: ${avg_l:.2f} | RR: {rr:.2f}")
    print(f"  Max DD: ${dd:,.2f}")
    print(f"  Avg bars held: {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"  Exit reasons:")
    for r, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"    {r:>10}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}")
    print(f"  By lot size:")
    for k in sorted(by_lots.keys()):
        v = by_lots[k]
        print(f"    {k}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}, WR={v['w']/max(1,v['n'])*100:.1f}%")

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        if y not in year_pnl: year_pnl[y] = [0, 0.0]
        year_pnl[y][0] += 1; year_pnl[y][1] += t.pnl
    print(f"  Year-by-Year:")
    for y in sorted(year_pnl.keys()):
        ny, p = year_pnl[y]
        print(f"    {y}: N={ny:>5}, PnL=${p:>10,.2f}")

    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("# M1 ML Scalper — Alpha Research + Walk-Forward Backtest")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"

    # 先用 2020-2026 做研究 (减少内存和时间)
    df = load_m1(m1_path, start="2020-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Feature Engineering
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("Phase 1: Feature Engineering")
    print("=" * 60)

    features = build_features(df)
    print(f"  Built {len(features.columns)} features")
    print(f"  Sample features: {list(features.columns[:10])}")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Label Construction
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 2: Label Construction")
    print("=" * 60)

    # 测试多种 TP/SL 组合
    for tp_p, sl_p, mh in [(2.0, 3.0, 20), (2.0, 3.5, 30), (3.0, 5.0, 30)]:
        labels = build_labels(df, tp_points=tp_p, sl_points=sl_p, max_bars=mh)
        buy_rate = labels['label_buy'].dropna().mean()
        sell_rate = labels['label_sell'].dropna().mean()
        both_rate = ((labels['label_buy'] == 1) & (labels['label_sell'] == 1)).mean()
        neither_rate = ((labels['label_buy'] == 0) & (labels['label_sell'] == 0)).mean()
        print(f"  TP=${tp_p} SL=${sl_p} MH={mh}: Buy={buy_rate:.3f} Sell={sell_rate:.3f} "
              f"Both={both_rate:.3f} Neither={neither_rate:.3f}")

    # Use best label config for ML
    labels = build_labels(df, tp_points=2.0, sl_points=3.5, max_bars=30)

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Alpha IC Analysis
    # ═══════════════════════════════════════════════════════════
    ic_results = alpha_ic_analysis(features, labels)
    top_feats = [r['feature'] for r in ic_results if r['ic_abs_max'] > 0.01][:35]
    print(f"\n  Selected {len(top_feats)} features for ML")

    if len(top_feats) < 5:
        print("  WARNING: Too few features with IC > 0.01, using top 20 by |IC|")
        top_feats = [r['feature'] for r in ic_results[:20]]

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Walk-Forward ML Training
    # ═══════════════════════════════════════════════════════════
    predictions = train_ml_model(df, features, labels, top_feats,
                                  train_months=6, test_months=1)

    pred_valid = predictions.dropna()
    print(f"\n  Predictions: {len(pred_valid):,} samples")
    print(f"  p_buy  mean={pred_valid['p_buy'].mean():.3f} std={pred_valid['p_buy'].std():.3f}")
    print(f"  p_sell mean={pred_valid['p_sell'].mean():.3f} std={pred_valid['p_sell'].std():.3f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: ML-Driven Backtest
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 5: ML-Driven Backtest")
    print("=" * 60)

    configs = [
        dict(tp=2.0, sl=3.5, threshold=0.55, max_hold=30, cooldown=3,
             label="ML: TP=$2, SL=$3.5, Conf>55%"),
        dict(tp=2.0, sl=3.5, threshold=0.60, max_hold=30, cooldown=3,
             label="ML: TP=$2, SL=$3.5, Conf>60%"),
        dict(tp=2.0, sl=3.5, threshold=0.65, max_hold=30, cooldown=5,
             label="ML: TP=$2, SL=$3.5, Conf>65%"),
        dict(tp=1.5, sl=3.0, threshold=0.55, max_hold=20, cooldown=2,
             label="ML: TP=$1.5, SL=$3, Conf>55%"),
        dict(tp=1.5, sl=3.0, threshold=0.60, max_hold=20, cooldown=2,
             label="ML: TP=$1.5, SL=$3, Conf>60%"),
        dict(tp=3.0, sl=5.0, threshold=0.55, max_hold=30, cooldown=3,
             label="ML: TP=$3, SL=$5, Conf>55%"),
        dict(tp=3.0, sl=5.0, threshold=0.60, max_hold=30, cooldown=5,
             label="ML: TP=$3, SL=$5, Conf>60%"),
        dict(tp=2.0, sl=3.5, threshold=0.50, max_hold=30, cooldown=3,
             label="ML: TP=$2, SL=$3.5, Conf>50%"),
    ]

    all_results = []
    for cfg in configs:
        lbl = cfg.pop('label')
        print(f"\n  Running: {lbl}")
        tr, eq = ml_backtest(df, predictions, **cfg, spread=0.30,
                              session_hours=set(range(0, 13)), max_per_day=20)
        res = report(tr, eq, lbl)
        all_results.append((lbl, res))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — ML Backtest Results")
    print("=" * 60)
    all_results.sort(key=lambda x: x[1].get('sharpe', -999), reverse=True)
    print(f"{'Strategy':<45} {'N':>6} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'DD':>8}")
    print("-" * 85)
    for lbl, res in all_results:
        if res:
            print(f"{lbl:<45} {res['n']:>6,} ${res['pnl']:>9,.2f} {res['sharpe']:>7.2f} "
                  f"{res['wr']:>5.1f}% ${res['dd']:>7,.2f}")

    elapsed = time.time() - t0
    print(f"\n\nTotal: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
