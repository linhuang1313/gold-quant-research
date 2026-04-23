"""
M1 ML Scalper v5-Final — 直接用最优参数出详细报告
====================================================
基于 v5 Phase 1-3 结论:
  Best TP/SL: TP=$5.0, SL=$4.0 (Sharpe=-1.38)
  Best QCut: 10b/15% (Sharpe=-1.08)
  Best Threshold: C>60%

本脚本:
1. 训练 6 个月窗口 Ensemble
2. 用 numpy array 替代 set lookup, 节省内存
3. 只跑 ~10 个关键配置 + 详细 champion 报告
"""
import sys
import time
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

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
    f = pd.DataFrame(index=df.index)
    c = df['Close']; h = df['High']; lo = df['Low']; o = df['Open']
    for n in [5, 10, 20, 30]:
        f[f'range_{n}'] = (h - lo).rolling(n).mean()
        f[f'vol_{n}'] = c.rolling(n).std()
    f['bar_size'] = h - lo; f['bar_body'] = (c - o).abs()
    for n in [10, 20, 30]:
        hh = h.rolling(n).max(); ll = lo.rolling(n).min(); rng = hh - ll
        f[f'dist_high_{n}'] = hh - c; f[f'dist_low_{n}'] = c - ll
        f[f'price_pos_{n}'] = (c - ll) / rng.replace(0, np.nan)
    for n in [10, 20]:
        sma = c.rolling(n).mean(); std = c.rolling(n).std()
        f[f'zscore_{n}'] = (c - sma) / std.replace(0, np.nan)
    f['hour'] = df.index.hour
    f['is_london'] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
    f['is_asia'] = (df.index.hour < 8).astype(int)
    for n in [5, 10, 20]: f[f'move_{n}'] = c - c.shift(n)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    atr14 = pd.DataFrame({'hl': h - lo, 'hc': (h - c.shift(1)).abs(), 'lc': (lo - c.shift(1)).abs()}).max(axis=1).rolling(14).mean()
    f['atr_14'] = atr14
    f['atr_pct'] = atr14.rolling(240).rank(pct=True)
    f['atr_change'] = atr14 / atr14.shift(10) - 1
    f['vol_ratio'] = f['vol_5'] / f['vol_30'].replace(0, np.nan)
    f['range_squeeze'] = f['range_5'] / f['range_30'].replace(0, np.nan)
    return f


def build_labels(df, tp_pts=3.0, sl_pts=5.0, max_bars=30, spread=0.30):
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values; n = len(df)
    lb = np.full(n, np.nan); ls = np.full(n, np.nan)
    for i in range(n - max_bars):
        entry = c[i]; be = entry + spread / 2; se = entry - spread / 2
        br = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if h[j] >= be + tp_pts: br = 1; break
            if lo[j] <= be - sl_pts: break
        sr = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lo[j] <= se - tp_pts: sr = 1; break
            if h[j] >= se + sl_pts: break
        lb[i] = br; ls[i] = sr
    return pd.DataFrame({'label_buy': lb, 'label_sell': ls}, index=df.index)


def train_ensemble_wf(df, features, labels, train_m=6, test_m=1):
    feat_cols = [c for c in features.columns if features[c].notna().sum() > 1000]
    X = features[feat_cols]; y_b = labels['label_buy']; y_s = labels['label_sell']
    valid = X.dropna().index.intersection(y_b.dropna().index)
    X = X.loc[valid]; y_b = y_b.loc[valid]; y_s = y_s.loc[valid]
    print(f"  Valid: {len(X):,}, {len(feat_cols)} feats")
    dates = X.index.to_series(); min_d = dates.min(); max_d = dates.max()

    # Use numpy arrays for memory efficiency
    p_buy = np.full(len(X), np.nan, dtype=np.float32)
    p_sell = np.full(len(X), np.nan, dtype=np.float32)
    idx_map = {ts: i for i, ts in enumerate(X.index)}

    xgb_p = dict(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.7,
                 colsample_bytree=0.7, min_child_weight=100, reg_alpha=1.0, reg_lambda=3.0,
                 random_state=42, n_jobs=-1, verbosity=0, tree_method='hist')
    lgb_p = dict(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.7,
                 colsample_bytree=0.7, min_child_samples=100, reg_alpha=1.0, reg_lambda=3.0,
                 random_state=42, n_jobs=-1, verbose=-1)

    cur = min_d + pd.DateOffset(months=train_m); fold = 0
    while cur < max_d:
        test_end = cur + pd.DateOffset(months=test_m)
        tr_mask = (dates >= cur - pd.DateOffset(months=train_m)) & (dates < cur)
        te_mask = (dates >= cur) & (dates < test_end)
        X_tr = X.loc[tr_mask]; X_te = X.loc[te_mask]
        if len(X_tr) < 5000 or len(X_te) < 500:
            cur += pd.DateOffset(months=test_m); continue

        xgb_b = xgb.XGBClassifier(**xgb_p); xgb_b.fit(X_tr, y_b.loc[tr_mask], verbose=False)
        xgb_s = xgb.XGBClassifier(**xgb_p); xgb_s.fit(X_tr, y_s.loc[tr_mask], verbose=False)
        lgb_b = lgb.LGBMClassifier(**lgb_p); lgb_b.fit(X_tr, y_b.loc[tr_mask])
        lgb_s = lgb.LGBMClassifier(**lgb_p); lgb_s.fit(X_tr, y_s.loc[tr_mask])

        pb = (xgb_b.predict_proba(X_te)[:, 1] + lgb_b.predict_proba(X_te)[:, 1]) / 2
        ps = (xgb_s.predict_proba(X_te)[:, 1] + lgb_s.predict_proba(X_te)[:, 1]) / 2

        te_indices = [idx_map[ts] for ts in X_te.index]
        p_buy[te_indices] = pb.astype(np.float32)
        p_sell[te_indices] = ps.astype(np.float32)

        try:
            auc_b = roc_auc_score(y_b.loc[te_mask], pb)
            auc_s = roc_auc_score(y_s.loc[te_mask], ps)
        except: auc_b = auc_s = 0.5
        fold += 1
        print(f"  F{fold:>2}: {cur.strftime('%Y-%m')} -> {min(test_end, max_d).strftime('%Y-%m')}  B={auc_b:.3f} S={auc_s:.3f} N={len(X_te):,}")
        cur += pd.DateOffset(months=test_m)
        del xgb_b, xgb_s, lgb_b, lgb_s; gc.collect()

    return X.index, p_buy, p_sell


def backtest_fast(df, pred_index, p_buy, p_sell, atr_pct_arr, tp=5.0, sl=4.0,
                  threshold=0.60, spread=0.30, cooldown=5, max_hold=25,
                  session_hours=None, max_per_day=15,
                  scale_after=4, lots=(0.01, 0.02, 0.03),
                  max_atr_pct=0.50, qcut_bars=10, qcut_loss=0.15):
    """Memory-efficient backtest using numpy arrays only."""
    if session_hours is None:
        session_hours = set(range(0, 13))

    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    times = df.index; n = len(df)

    # Build lookup: for each bar in df, what's the pred index (-1 if no pred)
    pred_ts_set = set(pred_index)
    pred_map = {}
    for i, ts in enumerate(pred_index):
        pred_map[ts] = i

    trades = []; equity = [2000.0]
    pos = None; last_close = -999; cw = 0; li = 0; dc = {}

    for i in range(n):
        ts = times[i]; c = close[i]; hv = high[i]; lv = low[i]
        day = str(ts.date())

        if pos is not None:
            lm = pos['lots'] / 0.01
            if pos['dir'] == 'BUY':
                pnl_cur = (c - pos['e'] - spread) * pos['lots'] * 100
                pnl_best = (hv - pos['e'] - spread) * pos['lots'] * 100
                pnl_worst = (lv - pos['e'] - spread) * pos['lots'] * 100
            else:
                pnl_cur = (pos['e'] - c - spread) * pos['lots'] * 100
                pnl_best = (pos['e'] - lv - spread) * pos['lots'] * 100
                pnl_worst = (pos['e'] - hv - spread) * pos['lots'] * 100
            cur_tp = tp * lm; cur_sl = sl * lm; held = i - pos['i']

            if pnl_best >= cur_tp:
                _at(trades, equity, pos, c, ts, "TP", i, cur_tp); pos = None; last_close = i
                cw += 1; li = min(li + 1, len(lots) - 1) if cw >= scale_after else li; continue
            if pnl_worst <= -cur_sl:
                _at(trades, equity, pos, c, ts, "SL", i, -cur_sl); pos = None; last_close = i
                cw = 0; li = max(0, li - 1); continue
            if held >= qcut_bars and pnl_cur <= -cur_sl * qcut_loss:
                _at(trades, equity, pos, c, ts, "QCut", i, pnl_cur); pos = None; last_close = i
                cw = 0; li = max(0, li - 1); continue
            if held >= max_hold:
                _at(trades, equity, pos, c, ts, "TO", i, pnl_cur); pos = None; last_close = i
                if pnl_cur > 0: cw += 1
                else: cw = 0; li = max(0, li - 1)
                continue

        if pos is not None: continue
        if i - last_close < cooldown: continue
        if ts.hour not in session_hours: continue
        if dc.get(day, 0) >= max_per_day: continue
        if ts not in pred_ts_set: continue

        ap = atr_pct_arr[i]
        if np.isnan(ap) or ap > max_atr_pct: continue

        pi = pred_map[ts]
        pbv = p_buy[pi]; psv = p_sell[pi]
        if np.isnan(pbv) or np.isnan(psv): continue

        d = None
        if pbv > threshold and psv > threshold:
            d = 'BUY' if pbv > psv else 'SELL'
        elif pbv > threshold: d = 'BUY'
        elif psv > threshold: d = 'SELL'
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
        strategy="MLv5", direction=pos['dir'], entry_price=pos['e'], exit_price=ep,
        entry_time=pos['time'], exit_time=ts, lots=pos['lots'], pnl=pnl,
        exit_reason=reason, bars_held=idx - pos['i']))
    equity.append(equity[-1] + pnl)


def calc_stats(trades, equity):
    if not trades: return {'n': 0, 'pnl': 0, 'sharpe': -999, 'wr': 0, 'dd': 0, 'rr': 0}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    total = sum(pnls); wr = len(wins) / len(pnls) * 100
    avg_w = np.mean(wins) if wins else 0; avg_l = abs(np.mean(losses)) if losses else 0
    daily = aggregate_daily_pnl(trades)
    sh = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252) if len(daily) > 1 and np.std(daily, ddof=1) > 0 else 0
    pk = equity[0]; dd = 0
    for e in equity:
        if e > pk: pk = e
        if pk - e > dd: dd = pk - e
    rr = avg_w / avg_l if avg_l > 0 else 0
    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd, 'rr': rr,
            'avg_w': avg_w, 'avg_l': avg_l}


def print_detailed(trades, equity, label):
    r = calc_stats(trades, equity)
    if r['n'] == 0: print(f"  {label}: No trades"); return r

    by_reason = {}
    for t in trades:
        rv = t.exit_reason
        if rv not in by_reason: by_reason[rv] = {'n': 0, 'pnl': 0}
        by_reason[rv]['n'] += 1; by_reason[rv]['pnl'] += t.pnl

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Trades: {r['n']:,}  |  PnL: ${r['pnl']:,.2f}  |  Sharpe: {r['sharpe']:.2f}")
    print(f"  WR: {r['wr']:.1f}%  |  AvgW: ${r['avg_w']:.2f}  AvgL: ${r['avg_l']:.2f}  RR: {r['rr']:.2f}")
    print(f"  MaxDD: ${r['dd']:,.2f}  |  Avg bars: {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"  Exits: ", end="")
    for rv, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"{rv}={v['n']}(${v['pnl']:+,.0f}) ", end="")
    print()

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        if y not in year_pnl: year_pnl[y] = [0, 0.0]
        year_pnl[y][0] += 1; year_pnl[y][1] += t.pnl
    print(f"  Years: ", end="")
    for y in sorted(year_pnl.keys()):
        ny, p = year_pnl[y]
        m = "+" if p > 0 else ""
        print(f"{y}={ny}(${m}{p:,.0f}) ", end="")
    print()

    monthly = {}
    for t in trades:
        m = pd.Timestamp(t.exit_time).strftime('%Y-%m')
        if m not in monthly: monthly[m] = [0, 0.0]
        monthly[m][0] += 1; monthly[m][1] += t.pnl
    print(f"\n  Monthly PnL:")
    pos_m = 0; neg_m = 0
    for m in sorted(monthly.keys()):
        nm, p = monthly[m]
        marker = "+" if p >= 0 else ""
        bar = "█" * max(1, int(abs(p) / 10))
        side = "↑" if p >= 0 else "↓"
        print(f"    {m}: {nm:>3} trades  ${marker}{p:>7,.2f}  {side}{bar}")
        if p >= 0: pos_m += 1
        else: neg_m += 1
    if pos_m + neg_m > 0:
        print(f"\n  Profitable months: {pos_m}/{pos_m+neg_m} ({pos_m/(pos_m+neg_m)*100:.0f}%)")

    return r


def main():
    t0 = time.time()
    print("# M1 ML Scalper v5-Final")
    print(f"# {pd.Timestamp.now()}\n")

    df = load_m1("data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv", start="2022-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    features = build_features(df)
    print(f"  Features: {len(features.columns)}")

    print("\n  Building labels...")
    labels = build_labels(df, tp_pts=3.0, sl_pts=5.0, max_bars=30)

    print("\n  Training Ensemble WF (6m/1m)...")
    pred_index, p_buy, p_sell = train_ensemble_wf(df, features, labels, train_m=6, test_m=1)
    valid_n = np.sum(~np.isnan(p_buy))
    print(f"\n  Valid preds: {valid_n:,}")

    atr_pct_arr = features['atr_pct'].values

    all_res = []

    configs = [
        # From v4/v5 research: best configurations
        ("Best: TP$5/SL$4 QCut(10/15%) C>60% CD=5 MH=25 VF=50%",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=5, max_hold=25, max_atr_pct=0.50, qcut_bars=10, qcut_loss=0.15)),
        ("Alt1: TP$6/SL$3.5 QCut(10/15%) C>60%",
         dict(tp=6.0, sl=3.5, threshold=0.60, cooldown=5, max_hold=25, max_atr_pct=0.50, qcut_bars=10, qcut_loss=0.15)),
        ("Alt2: TP$5/SL$4 QCut(10/15%) C>60% CD=3",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=3, max_hold=25, max_atr_pct=0.50, qcut_bars=10, qcut_loss=0.15)),
        ("Alt3: TP$5/SL$4 QCut(10/15%) C>60% VF=60%",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=5, max_hold=25, max_atr_pct=0.60, qcut_bars=10, qcut_loss=0.15)),
        ("Alt4: TP$5/SL$4 QCut(10/15%) C>60% VF=40%",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=5, max_hold=25, max_atr_pct=0.40, qcut_bars=10, qcut_loss=0.15)),
        ("Alt5: TP$5/SL$4 C>64% (high conf)",
         dict(tp=5.0, sl=4.0, threshold=0.64, cooldown=5, max_hold=25, max_atr_pct=0.50, qcut_bars=10, qcut_loss=0.15)),
        ("Alt6: TP$5/SL$4 QCut(5/50%)",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=5, max_hold=25, max_atr_pct=0.50, qcut_bars=5, qcut_loss=0.50)),
        ("Alt7: TP$5/SL$4 MH=15",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=5, max_hold=15, max_atr_pct=0.50, qcut_bars=10, qcut_loss=0.15)),
        ("NoVF: TP$5/SL$4 QCut(10/15%) C>60% NoVF",
         dict(tp=5.0, sl=4.0, threshold=0.60, cooldown=5, max_hold=25, max_atr_pct=1.00, qcut_bars=10, qcut_loss=0.15)),
    ]

    print("\n" + "=" * 65)
    print("Config Comparison")
    print("=" * 65)

    for lbl, params in configs:
        tr, eq = backtest_fast(df, pred_index, p_buy, p_sell, atr_pct_arr, **params)
        r = calc_stats(tr, eq)
        all_res.append((lbl, r, tr, eq))
        print(f"  {lbl[:52]:52s} N={r['n']:>5,} PnL=${r['pnl']:>8,.2f} SR={r['sharpe']:>6.2f} "
              f"WR={r['wr']:>5.1f}% RR={r['rr']:.2f} DD=${r['dd']:>7,.2f}")

    # Sort and identify champion
    all_res.sort(key=lambda x: x[1]['sharpe'], reverse=True)
    champ_lbl, champ_r, champ_tr, champ_eq = all_res[0]

    print(f"\n  CHAMPION: {champ_lbl}")

    # Detailed champion report
    print_detailed(champ_tr, champ_eq, f"CHAMPION: {champ_lbl[:40]}")

    # Second best
    if len(all_res) > 1:
        lbl2, r2, tr2, eq2 = all_res[1]
        print_detailed(tr2, eq2, f"RUNNER-UP: {lbl2[:40]}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
