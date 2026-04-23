"""
M1 ML Scalper v2 — 修复内存问题, 使用 2022-2026 数据
=====================================================
改进:
  1. 只用 2022-2026 (减半数据量)
  2. 精简特征到 Top 20
  3. Walk-Forward: 训练3个月, 预测1个月 (更快迭代)
  4. 多种 TP/SL + 阈值组合回测
"""
import sys
import time
import warnings
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.engine import TradeRecord
from backtest.stats import aggregate_daily_pnl


def load_m1(path, start=None, end=None):
    df = pd.read_csv(path)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.000')
    df = df.set_index('Gmt time').sort_index()
    if start: df = df[df.index >= start]
    if end: df = df[df.index <= end]
    return df


def build_features(df):
    """精简版特征: Top 20 by IC."""
    f = pd.DataFrame(index=df.index)
    c = df['Close']; h = df['High']; lo = df['Low']; o = df['Open']

    # 波动率/Range (IC > 0.13) — 最重要的特征组
    for n in [5, 10, 20, 30]:
        f[f'range_{n}'] = (h - lo).rolling(n).mean()
        f[f'vol_{n}'] = c.rolling(n).std()

    # Bar 特征 (IC ~0.09)
    f['bar_size'] = h - lo
    f['bar_body'] = (c - o).abs()

    # 价格位置 (IC ~0.08-0.11)
    for n in [10, 20, 30]:
        hh = h.rolling(n).max()
        ll = lo.rolling(n).min()
        rng = hh - ll
        f[f'dist_high_{n}'] = hh - c
        f[f'dist_low_{n}'] = c - ll
        f[f'price_pos_{n}'] = (c - ll) / rng.replace(0, np.nan)

    # Z-score (IC ~0.02)
    for n in [10, 20]:
        sma = c.rolling(n).mean()
        std = c.rolling(n).std()
        f[f'zscore_{n}'] = (c - sma) / std.replace(0, np.nan)

    # 时间 (IC ~0.03-0.06)
    f['hour'] = df.index.hour
    f['is_london'] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
    f['is_asia'] = (df.index.hour < 8).astype(int)

    # 动量
    for n in [5, 10, 20]:
        f[f'move_{n}'] = c - c.shift(n)

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    return f


def build_labels(df, tp_pts=2.0, sl_pts=3.5, max_bars=30, spread=0.30):
    """向量化标签生成 — 更快更省内存."""
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    n = len(df)

    label_buy = np.full(n, np.nan)
    label_sell = np.full(n, np.nan)

    # 分块处理减少内存
    chunk = 100000
    for start in range(0, n - max_bars, chunk):
        end = min(start + chunk, n - max_bars)
        for i in range(start, end):
            entry = c[i]
            buy_entry = entry + spread / 2
            sell_entry = entry - spread / 2

            # BUY
            buy_result = 0
            for j in range(i + 1, min(i + max_bars + 1, n)):
                if h[j] >= buy_entry + tp_pts:
                    buy_result = 1; break
                if lo[j] <= buy_entry - sl_pts:
                    buy_result = 0; break

            # SELL
            sell_result = 0
            for j in range(i + 1, min(i + max_bars + 1, n)):
                if lo[j] <= sell_entry - tp_pts:
                    sell_result = 1; break
                if h[j] >= sell_entry + sl_pts:
                    sell_result = 0; break

            label_buy[i] = buy_result
            label_sell[i] = sell_result

        pct = (end * 100) // (n - max_bars)
        print(f"    labels {pct}%...", end=" ", flush=True)

    print("done!", flush=True)
    return pd.DataFrame({'label_buy': label_buy, 'label_sell': label_sell}, index=df.index)


def train_wf(df, features, labels, train_m=3, test_m=1):
    """Walk-Forward XGBoost, 内存优化版."""
    feat_cols = [c for c in features.columns if features[c].notna().sum() > 1000]
    X = features[feat_cols]
    y_buy = labels['label_buy']
    y_sell = labels['label_sell']

    valid = X.dropna().index.intersection(y_buy.dropna().index)
    X = X.loc[valid]
    y_buy = y_buy.loc[valid]
    y_sell = y_sell.loc[valid]

    print(f"  Valid: {len(X):,} samples, {len(feat_cols)} features")
    print(f"  Buy rate: {y_buy.mean():.3f}, Sell rate: {y_sell.mean():.3f}")

    dates = X.index.to_series()
    min_d = dates.min()
    max_d = dates.max()

    preds = pd.DataFrame(index=X.index, columns=['p_buy', 'p_sell'], dtype=float)
    preds[:] = np.nan

    params = {
        'n_estimators': 150,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 100,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',
    }

    cur = min_d + pd.DateOffset(months=train_m)
    fold = 0
    last_buy_model = None
    last_sell_model = None

    while cur < max_d:
        test_end = cur + pd.DateOffset(months=test_m)
        train_mask = (dates >= cur - pd.DateOffset(months=train_m)) & (dates < cur)
        test_mask = (dates >= cur) & (dates < test_end)

        X_tr = X.loc[train_mask]
        X_te = X.loc[test_mask]

        if len(X_tr) < 3000 or len(X_te) < 500:
            cur += pd.DateOffset(months=test_m)
            continue

        y_b_tr = y_buy.loc[train_mask]
        y_s_tr = y_sell.loc[train_mask]

        model_b = xgb.XGBClassifier(**params)
        model_b.fit(X_tr, y_b_tr, verbose=False)

        model_s = xgb.XGBClassifier(**params)
        model_s.fit(X_tr, y_s_tr, verbose=False)

        p_b = model_b.predict_proba(X_te)[:, 1]
        p_s = model_s.predict_proba(X_te)[:, 1]
        preds.loc[test_mask, 'p_buy'] = p_b
        preds.loc[test_mask, 'p_sell'] = p_s

        acc_b = accuracy_score(y_buy.loc[test_mask], (p_b > 0.5).astype(int))
        acc_s = accuracy_score(y_sell.loc[test_mask], (p_s > 0.5).astype(int))
        try:
            auc_b = roc_auc_score(y_buy.loc[test_mask], p_b)
            auc_s = roc_auc_score(y_sell.loc[test_mask], p_s)
        except:
            auc_b = auc_s = 0.5

        fold += 1
        p = f"{cur.strftime('%Y-%m')} -> {min(test_end, max_d).strftime('%Y-%m')}"
        print(f"  F{fold:>2}: {p}  B(Acc={acc_b:.3f} AUC={auc_b:.3f}) S(Acc={acc_s:.3f} AUC={auc_s:.3f}) N={len(X_te):,}")

        last_buy_model = model_b
        last_sell_model = model_s
        cur += pd.DateOffset(months=test_m)
        gc.collect()

    # Feature importance
    if last_buy_model:
        imp = pd.Series(last_buy_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print(f"\n  Top 10 Buy Features:")
        for f_name, s in imp.head(10).items():
            print(f"    {f_name:<25} {s:.4f}")

    if last_sell_model:
        imp_s = pd.Series(last_sell_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
        print(f"\n  Top 10 Sell Features:")
        for f_name, s in imp_s.head(10).items():
            print(f"    {f_name:<25} {s:.4f}")

    return preds


def ml_backtest(df, preds, tp=2.0, sl=3.5, threshold=0.55, spread=0.30,
                cooldown=3, max_hold=30, session_hours=None, max_per_day=20,
                scale_after=4, lots=(0.01, 0.02, 0.03)):
    if session_hours is None:
        session_hours = set(range(0, 13))

    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    times = df.index; n = len(df)
    valid_set = set(preds.dropna().index)

    trades = []; equity = [2000.0]
    pos = None; last_close = -999; cw = 0; li = 0; dc = {}

    for i in range(n):
        ts = times[i]; c = close[i]; h_v = high[i]; lo_v = low[i]
        day = str(ts.date())

        if pos is not None:
            lm = pos['lots'] / 0.01
            if pos['dir'] == 'BUY':
                pb = (h_v - pos['e'] - spread) * pos['lots'] * 100
                pw = (lo_v - pos['e'] - spread) * pos['lots'] * 100
                pc = (c - pos['e'] - spread) * pos['lots'] * 100
            else:
                pb = (pos['e'] - lo_v - spread) * pos['lots'] * 100
                pw = (pos['e'] - h_v - spread) * pos['lots'] * 100
                pc = (pos['e'] - c - spread) * pos['lots'] * 100

            tp_l = tp * lm; sl_l = sl * lm; held = i - pos['i']

            if pb >= tp_l:
                _at(trades, equity, pos, c, ts, "TP", i, tp_l)
                pos = None; last_close = i; cw += 1
                if cw >= scale_after: li = min(li + 1, len(lots) - 1)
                continue
            if pw <= -sl_l:
                _at(trades, equity, pos, c, ts, "SL", i, -sl_l)
                pos = None; last_close = i; cw = 0; li = max(0, li - 1)
                continue
            if held >= max_hold:
                _at(trades, equity, pos, c, ts, "Timeout", i, pc)
                pos = None; last_close = i
                if pc > 0: cw += 1
                else: cw = 0; li = max(0, li - 1)
                continue

        if pos is not None: continue
        if i - last_close < cooldown: continue
        if ts.hour not in session_hours: continue
        if dc.get(day, 0) >= max_per_day: continue
        if ts not in valid_set: continue

        row = preds.loc[ts]
        pb_v = row['p_buy']; ps_v = row['p_sell']
        if pd.isna(pb_v) or pd.isna(ps_v): continue

        d = None
        if pb_v > threshold and ps_v > threshold:
            d = 'BUY' if pb_v > ps_v else 'SELL'
        elif pb_v > threshold: d = 'BUY'
        elif ps_v > threshold: d = 'SELL'
        if d is None: continue

        ep = c + spread / 2 if d == 'BUY' else c - spread / 2
        pos = {'dir': d, 'e': ep, 'time': ts, 'lots': lots[li], 'i': i}
        dc[day] = dc.get(day, 0) + 1

    if pos:
        pnl = ((close[-1] - pos['e'] - spread) if pos['dir'] == 'BUY'
               else (pos['e'] - close[-1] - spread)) * pos['lots'] * 100
        _at(trades, equity, pos, close[-1], times[-1], "EOD", n - 1, pnl)

    return trades, equity


def _at(trades, equity, pos, ep, ts, reason, idx, pnl):
    trades.append(TradeRecord(
        strategy="MLScalp", direction=pos['dir'],
        entry_price=pos['e'], exit_price=ep,
        entry_time=pos['time'], exit_time=ts,
        lots=pos['lots'], pnl=pnl, exit_reason=reason,
        bars_held=idx - pos['i'],
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
    rr = avg_w / avg_l if avg_l > 0 else 0

    by_reason = {}
    for t in trades:
        r = t.exit_reason
        if r not in by_reason: by_reason[r] = {'n': 0, 'pnl': 0}
        by_reason[r]['n'] += 1; by_reason[r]['pnl'] += t.pnl

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades):,}")
    print(f"  Total PnL: ${total:,.2f}")
    print(f"  Sharpe: {sh:.2f}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Avg Win: ${avg_w:.2f} | Avg Loss: ${avg_l:.2f} | RR: {rr:.2f}")
    print(f"  Max DD: ${dd:,.2f}")
    print(f"  Avg bars held: {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"  Exit reasons:")
    for r, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"    {r:>10}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}")

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        if y not in year_pnl: year_pnl[y] = [0, 0.0]
        year_pnl[y][0] += 1; year_pnl[y][1] += t.pnl
    print(f"  Year-by-Year:")
    for y in sorted(year_pnl.keys()):
        ny, p = year_pnl[y]
        print(f"    {y}: N={ny:>5}, PnL=${p:>10,.2f}")

    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd, 'rr': rr}


def main():
    t0 = time.time()
    print("# M1 ML Scalper v2")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"
    df = load_m1(m1_path, start="2022-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    print("Phase 1: Features")
    features = build_features(df)
    print(f"  {len(features.columns)} features\n")

    print("Phase 2: Labels")
    labels = build_labels(df, tp_pts=2.0, sl_pts=3.5, max_bars=30)

    print(f"\n  Buy rate: {labels['label_buy'].dropna().mean():.3f}")
    print(f"  Sell rate: {labels['label_sell'].dropna().mean():.3f}")

    print("\nPhase 3: Walk-Forward Training")
    preds = train_wf(df, features, labels, train_m=3, test_m=1)

    valid = preds.dropna()
    print(f"\n  Predictions: {len(valid):,}")
    if len(valid) == 0:
        print("  No predictions! Exiting.")
        return

    print(f"  p_buy  mean={valid['p_buy'].mean():.3f} std={valid['p_buy'].std():.3f}")
    print(f"  p_sell mean={valid['p_sell'].mean():.3f} std={valid['p_sell'].std():.3f}")

    print("\nPhase 4: Backtest")
    configs = [
        dict(tp=2.0, sl=3.5, threshold=0.52, cooldown=3, max_hold=30, label="ML: TP$2 SL$3.5 C>52%"),
        dict(tp=2.0, sl=3.5, threshold=0.55, cooldown=3, max_hold=30, label="ML: TP$2 SL$3.5 C>55%"),
        dict(tp=2.0, sl=3.5, threshold=0.58, cooldown=3, max_hold=30, label="ML: TP$2 SL$3.5 C>58%"),
        dict(tp=2.0, sl=3.5, threshold=0.60, cooldown=5, max_hold=30, label="ML: TP$2 SL$3.5 C>60%"),
        dict(tp=1.5, sl=3.0, threshold=0.55, cooldown=2, max_hold=20, label="ML: TP$1.5 SL$3 C>55%"),
        dict(tp=1.5, sl=3.0, threshold=0.58, cooldown=2, max_hold=20, label="ML: TP$1.5 SL$3 C>58%"),
        dict(tp=3.0, sl=5.0, threshold=0.55, cooldown=3, max_hold=30, label="ML: TP$3 SL$5 C>55%"),
        dict(tp=3.0, sl=5.0, threshold=0.58, cooldown=5, max_hold=30, label="ML: TP$3 SL$5 C>58%"),
        dict(tp=2.0, sl=3.5, threshold=0.55, cooldown=3, max_hold=30,
             session_hours=set(range(0, 24)), max_per_day=30, label="ML: TP$2 SL$3.5 C>55% (24h)"),
    ]

    all_res = []
    for cfg in configs:
        lbl = cfg.pop('label')
        tr, eq = ml_backtest(df, preds, **cfg, spread=0.30)
        res = report(tr, eq, lbl)
        all_res.append((lbl, res))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_res.sort(key=lambda x: x[1].get('sharpe', -999), reverse=True)
    print(f"{'Strategy':<45} {'N':>6} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'RR':>5} {'DD':>8}")
    print("-" * 90)
    for lbl, res in all_res:
        if res:
            print(f"{lbl:<45} {res['n']:>6,} ${res['pnl']:>9,.2f} {res['sharpe']:>7.2f} "
                  f"{res['wr']:>5.1f}% {res.get('rr',0):>5.2f} ${res['dd']:>7,.2f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
