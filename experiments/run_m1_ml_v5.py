"""
M1 ML Scalper v5 — 精细网格搜索
=================================
基于 v4 发现:
- 最优配置: TP$5/SL$3.5 + QCut (Sharpe=-0.85, RR=0.96)
- C>70% 小样本正 Sharpe
- QCut 单独比 Lock+QCut 更好

本轮:
1. TP/SL 精细网格: TP [4.0-6.0, step=0.5] × SL [2.5-4.5, step=0.5]
2. QCut 参数网格: bars [5,8,10,15] × loss [0.2,0.3,0.4,0.5]
3. Threshold 网格: [0.55-0.72, step=0.02]
4. Cooldown / MaxHold 微调
5. 用更长训练窗口(6m) 提高模型稳定性
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

    f['bar_size'] = h - lo
    f['bar_body'] = (c - o).abs()

    for n in [10, 20, 30]:
        hh = h.rolling(n).max(); ll = lo.rolling(n).min()
        rng = hh - ll
        f[f'dist_high_{n}'] = hh - c
        f[f'dist_low_{n}'] = c - ll
        f[f'price_pos_{n}'] = (c - ll) / rng.replace(0, np.nan)

    for n in [10, 20]:
        sma = c.rolling(n).mean(); std = c.rolling(n).std()
        f[f'zscore_{n}'] = (c - sma) / std.replace(0, np.nan)

    f['hour'] = df.index.hour
    f['is_london'] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
    f['is_asia'] = (df.index.hour < 8).astype(int)

    for n in [5, 10, 20]:
        f[f'move_{n}'] = c - c.shift(n)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    atr14 = pd.DataFrame({
        'hl': h - lo,
        'hc': (h - c.shift(1)).abs(),
        'lc': (lo - c.shift(1)).abs(),
    }).max(axis=1).rolling(14).mean()
    f['atr_14'] = atr14
    f['atr_pct'] = atr14.rolling(240).rank(pct=True)
    f['atr_change'] = atr14 / atr14.shift(10) - 1
    f['vol_ratio'] = f['vol_5'] / f['vol_30'].replace(0, np.nan)
    f['range_squeeze'] = f['range_5'] / f['range_30'].replace(0, np.nan)

    return f


def build_labels(df, tp_pts=3.0, sl_pts=5.0, max_bars=30, spread=0.30):
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    n = len(df)
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
    """6-month training window for more stability."""
    feat_cols = [c for c in features.columns if features[c].notna().sum() > 1000]
    X = features[feat_cols]; y_b = labels['label_buy']; y_s = labels['label_sell']
    valid = X.dropna().index.intersection(y_b.dropna().index)
    X = X.loc[valid]; y_b = y_b.loc[valid]; y_s = y_s.loc[valid]

    print(f"  Valid: {len(X):,}, {len(feat_cols)} feats, BuyR={y_b.mean():.3f}, SellR={y_s.mean():.3f}")

    dates = X.index.to_series()
    min_d = dates.min(); max_d = dates.max()
    preds = pd.DataFrame(index=X.index, columns=['p_buy', 'p_sell'], dtype=float)
    preds[:] = np.nan

    xgb_p = dict(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.7,
                 colsample_bytree=0.7, min_child_weight=100, reg_alpha=1.0, reg_lambda=3.0,
                 random_state=42, n_jobs=-1, verbosity=0, tree_method='hist')
    lgb_p = dict(n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.7,
                 colsample_bytree=0.7, min_child_samples=100, reg_alpha=1.0, reg_lambda=3.0,
                 random_state=42, n_jobs=-1, verbose=-1)

    cur = min_d + pd.DateOffset(months=train_m)
    fold = 0

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

        preds.loc[te_mask, 'p_buy'] = pb
        preds.loc[te_mask, 'p_sell'] = ps

        try:
            auc_b = roc_auc_score(y_b.loc[te_mask], pb)
            auc_s = roc_auc_score(y_s.loc[te_mask], ps)
        except: auc_b = auc_s = 0.5

        fold += 1
        p = f"{cur.strftime('%Y-%m')} -> {min(test_end, max_d).strftime('%Y-%m')}"
        print(f"  F{fold:>2}: {p}  B={auc_b:.3f} S={auc_s:.3f} N={len(X_te):,}")

        cur += pd.DateOffset(months=test_m)
        del xgb_b, xgb_s, lgb_b, lgb_s; gc.collect()

    return preds


def backtest(df, preds, features, tp=5.0, sl=3.5, threshold=0.60,
             spread=0.30, cooldown=5, max_hold=25,
             session_hours=None, max_per_day=15,
             scale_after=4, lots=(0.01, 0.02, 0.03),
             vol_filter=True, max_atr_pct=0.50,
             qcut=True, qcut_bars=10, qcut_loss=0.3,
             ):
    if session_hours is None:
        session_hours = set(range(0, 13))

    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    times = df.index; n = len(df)
    valid_set = set(preds.dropna().index)
    atr_pct_vals = features['atr_pct'].values if 'atr_pct' in features.columns else None

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

            cur_tp = tp * lm; cur_sl = sl * lm
            held = i - pos['i']

            if pnl_best >= cur_tp:
                _at(trades, equity, pos, c, ts, "TP", i, cur_tp)
                pos = None; last_close = i; cw += 1
                if cw >= scale_after: li = min(li + 1, len(lots) - 1)
                continue
            if pnl_worst <= -cur_sl:
                _at(trades, equity, pos, c, ts, "SL", i, -cur_sl)
                pos = None; last_close = i; cw = 0; li = max(0, li - 1)
                continue
            if qcut and held >= qcut_bars and pnl_cur <= -cur_sl * qcut_loss:
                _at(trades, equity, pos, c, ts, "QCut", i, pnl_cur)
                pos = None; last_close = i; cw = 0; li = max(0, li - 1)
                continue
            if held >= max_hold:
                _at(trades, equity, pos, c, ts, "Timeout", i, pnl_cur)
                pos = None; last_close = i
                if pnl_cur > 0: cw += 1
                else: cw = 0; li = max(0, li - 1)
                continue

        if pos is not None: continue
        if i - last_close < cooldown: continue
        if ts.hour not in session_hours: continue
        if dc.get(day, 0) >= max_per_day: continue
        if ts not in valid_set: continue

        if vol_filter and atr_pct_vals is not None:
            ap = atr_pct_vals[i]
            if np.isnan(ap) or ap > max_atr_pct: continue

        row = preds.loc[ts]
        pbv = row['p_buy']; psv = row['p_sell']
        if pd.isna(pbv) or pd.isna(psv): continue

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
        strategy="MLv5", direction=pos['dir'],
        entry_price=pos['e'], exit_price=ep,
        entry_time=pos['time'], exit_time=ts,
        lots=pos['lots'], pnl=pnl, exit_reason=reason,
        bars_held=idx - pos['i'],
    ))
    equity.append(equity[-1] + pnl)


def report(trades, equity, label):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': -999, 'wr': 0, 'dd': 0, 'rr': 0}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    total = sum(pnls); wr = len(wins) / len(pnls) * 100
    avg_w = np.mean(wins) if wins else 0; avg_l = abs(np.mean(losses)) if losses else 0
    daily = aggregate_daily_pnl(trades)
    sh = 0
    if len(daily) > 1 and np.std(daily, ddof=1) > 0:
        sh = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252)
    pk = equity[0]; dd = 0
    for e in equity:
        if e > pk: pk = e
        if pk - e > dd: dd = pk - e
    rr = avg_w / avg_l if avg_l > 0 else 0
    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd, 'rr': rr,
            'avg_w': avg_w, 'avg_l': avg_l}


def report_detailed(trades, equity, label):
    r = report(trades, equity, label)
    if r['n'] == 0:
        print(f"  {label}: No trades"); return r

    by_reason = {}
    for t in trades:
        rv = t.exit_reason
        if rv not in by_reason: by_reason[rv] = {'n': 0, 'pnl': 0}
        by_reason[rv]['n'] += 1; by_reason[rv]['pnl'] += t.pnl

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
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
        marker = "+" if p > 0 else ""
        print(f"{y}={ny}(${marker}{p:,.0f}) ", end="")
    print()

    return r


def main():
    t0 = time.time()
    print("# M1 ML Scalper v5 — Fine Grid Search")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"
    df = load_m1(m1_path, start="2022-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    features = build_features(df)
    print(f"  Features: {len(features.columns)}")

    print("\n  Building labels (TP=$3, SL=$5, MH=30)...")
    labels = build_labels(df, tp_pts=3.0, sl_pts=5.0, max_bars=30)
    print(f"  BuyR={labels['label_buy'].dropna().mean():.3f}")

    print("\n  Training Ensemble WF (6m train, 1m test)...")
    preds = train_ensemble_wf(df, features, labels, train_m=6, test_m=1)
    valid_p = preds.dropna()
    print(f"\n  Preds: {len(valid_p):,}")

    all_res = []

    # ════════════════════════════════════════════
    # Phase 1: TP/SL Grid (精细)
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 1: TP/SL Grid (QCut ON, VF50%, C>60%)")
    print("=" * 60)

    for tp_v in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        for sl_v in [2.5, 3.0, 3.5, 4.0, 4.5]:
            if tp_v / sl_v < 0.8 or tp_v / sl_v > 3.0: continue  # skip extreme ratios
            lbl = f"P1: TP${tp_v}/SL${sl_v}"
            tr, eq = backtest(df, preds, features, tp=tp_v, sl=sl_v, threshold=0.60,
                              cooldown=5, max_hold=25, qcut=True, qcut_bars=10, qcut_loss=0.3)
            r = report(tr, eq, lbl)
            all_res.append((lbl, r))
            print(f"  {lbl:30s} N={r['n']:>5,} PnL=${r['pnl']:>9,.2f} SR={r['sharpe']:>6.2f} "
                  f"WR={r['wr']:>5.1f}% RR={r['rr']:.2f} DD=${r['dd']:>7,.2f}")

    # Find top 5
    phase1_top = sorted(all_res, key=lambda x: x[1]['sharpe'], reverse=True)[:5]
    print(f"\n  Top-5 TP/SL:")
    for lbl, r in phase1_top:
        print(f"    {lbl:30s} SR={r['sharpe']:.2f} PnL=${r['pnl']:,.2f}")

    best_tp, best_sl = 5.0, 3.5  # default
    for lbl, r in phase1_top[:1]:
        parts = lbl.split("TP$")[1].split("/SL$")
        best_tp = float(parts[0]); best_sl = float(parts[1])
    print(f"\n  >>> Best: TP=${best_tp}, SL=${best_sl}")

    # ════════════════════════════════════════════
    # Phase 2: QCut Grid
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Phase 2: QCut Grid (TP${best_tp}/SL${best_sl})")
    print("=" * 60)

    qcut_res = []
    for qb in [5, 8, 10, 12, 15]:
        for ql in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            lbl = f"P2: QCut({qb}b/{ql:.0%})"
            tr, eq = backtest(df, preds, features, tp=best_tp, sl=best_sl, threshold=0.60,
                              cooldown=5, max_hold=25, qcut=True, qcut_bars=qb, qcut_loss=ql)
            r = report(tr, eq, lbl)
            qcut_res.append((lbl, r))
            print(f"  {lbl:30s} N={r['n']:>5,} PnL=${r['pnl']:>9,.2f} SR={r['sharpe']:>6.2f} "
                  f"WR={r['wr']:>5.1f}%")

    qcut_top = sorted(qcut_res, key=lambda x: x[1]['sharpe'], reverse=True)[:3]
    print(f"\n  Top-3 QCut:")
    for lbl, r in qcut_top:
        print(f"    {lbl:30s} SR={r['sharpe']:.2f} PnL=${r['pnl']:,.2f}")

    best_qb, best_ql = 10, 0.3
    for lbl, r in qcut_top[:1]:
        # parse
        p = lbl.split("QCut(")[1].replace(")", "").replace("%", "")
        parts = p.split("b/")
        best_qb = int(parts[0]); best_ql = float(parts[1]) / 100.0 if float(parts[1]) > 1 else float(parts[1])
    print(f"\n  >>> Best QCut: bars={best_qb}, loss={best_ql:.0%}")

    # ════════════════════════════════════════════
    # Phase 3: Threshold Grid
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Phase 3: Threshold Grid (TP${best_tp}/SL${best_sl}, QCut({best_qb}/{best_ql:.0%}))")
    print("=" * 60)

    thresh_res = []
    for th in [0.52, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72]:
        lbl = f"P3: C>{th:.0%}"
        tr, eq = backtest(df, preds, features, tp=best_tp, sl=best_sl, threshold=th,
                          cooldown=5, max_hold=25, qcut=True, qcut_bars=best_qb, qcut_loss=best_ql)
        r = report(tr, eq, lbl)
        thresh_res.append((lbl, r))
        print(f"  {lbl:30s} N={r['n']:>5,} PnL=${r['pnl']:>9,.2f} SR={r['sharpe']:>6.2f} "
              f"WR={r['wr']:>5.1f}% RR={r['rr']:.2f}")

    thresh_top = sorted(thresh_res, key=lambda x: x[1]['sharpe'], reverse=True)[:3]
    best_th = 0.60
    for lbl, r in thresh_top[:1]:
        p = lbl.split("C>")[1].replace("%", "")
        best_th = float(p) / 100.0 if float(p) > 1 else float(p)
    print(f"\n  >>> Best Threshold: {best_th:.0%}")

    # ════════════════════════════════════════════
    # Phase 4: Cooldown / MaxHold / VolFilter
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"Phase 4: Meta Params (TP${best_tp}/SL${best_sl}, QCut({best_qb}/{best_ql:.0%}), C>{best_th:.0%})")
    print("=" * 60)

    meta_res = []
    for cd in [3, 5, 8]:
        for mh in [15, 25]:
            for vf_pct in [0.40, 0.50, 0.60]:
                lbl = f"P4: CD={cd} MH={mh} VF={vf_pct:.0%}"
                tr, eq = backtest(df, preds, features, tp=best_tp, sl=best_sl, threshold=best_th,
                                  cooldown=cd, max_hold=mh, qcut=True, qcut_bars=best_qb, qcut_loss=best_ql,
                                  max_atr_pct=vf_pct)
                r = report(tr, eq, lbl)
                meta_res.append((lbl, r))
                print(f"  {lbl:35s} N={r['n']:>5,} PnL=${r['pnl']:>8,.2f} SR={r['sharpe']:>6.2f}")

    meta_top = sorted(meta_res, key=lambda x: x[1]['sharpe'], reverse=True)[:5]
    print(f"\n  Top-5 Meta:")
    for lbl, r in meta_top:
        print(f"    {lbl:40s} N={r['n']:>5,} PnL=${r['pnl']:>8,.2f} SR={r['sharpe']:>6.2f} "
              f"WR={r['wr']:>5.1f}% RR={r['rr']:.2f} DD=${r['dd']:>7,.2f}")

    # ════════════════════════════════════════════
    # Phase 5: Final Champion
    # ════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 5: Final Champion (detailed)")
    print("=" * 60)

    # Get best meta params
    best_meta_lbl = meta_top[0][0]
    parts = best_meta_lbl.split("CD=")[1].split(" MH=")
    best_cd = int(parts[0])
    parts2 = parts[1].split(" VF=")
    best_mh = int(parts2[0])
    best_vf = float(parts2[1].replace("%", "")) / 100

    print(f"\n  Champion config:")
    print(f"    TP=${best_tp}, SL=${best_sl}, C>{best_th:.0%}")
    print(f"    QCut: bars={best_qb}, loss={best_ql:.0%}")
    print(f"    CD={best_cd}, MH={best_mh}, VF={best_vf:.0%}")

    tr, eq = backtest(df, preds, features, tp=best_tp, sl=best_sl, threshold=best_th,
                      cooldown=best_cd, max_hold=best_mh,
                      qcut=True, qcut_bars=best_qb, qcut_loss=best_ql,
                      max_atr_pct=best_vf)
    r = report_detailed(tr, eq, "CHAMPION")

    # Also show no-QCut variant
    tr2, eq2 = backtest(df, preds, features, tp=best_tp, sl=best_sl, threshold=best_th,
                        cooldown=best_cd, max_hold=best_mh,
                        qcut=False, max_atr_pct=best_vf)
    r2 = report_detailed(tr2, eq2, "CHAMPION (no QCut)")

    # Also show 2024+ only performance
    print("\n  --- 2024+ Trades Only ---")
    recent_tr = [t for t in tr if pd.Timestamp(t.entry_time).year >= 2024]
    recent_pnls = [t.pnl for t in recent_tr]
    if recent_pnls:
        wins = [p for p in recent_pnls if p > 0]
        losses = [p for p in recent_pnls if p <= 0]
        print(f"  N={len(recent_tr)}, PnL=${sum(recent_pnls):,.2f}, "
              f"WR={len(wins)/len(recent_pnls)*100:.1f}%, "
              f"AvgW=${np.mean(wins) if wins else 0:.2f}, "
              f"AvgL=${abs(np.mean(losses)) if losses else 0:.2f}")

    # Monthly breakdown
    monthly = {}
    for t in tr:
        m = pd.Timestamp(t.exit_time).strftime('%Y-%m')
        if m not in monthly: monthly[m] = [0, 0.0]
        monthly[m][0] += 1; monthly[m][1] += t.pnl

    print(f"\n  Monthly PnL:")
    pos_m = 0; neg_m = 0
    for m in sorted(monthly.keys()):
        nm, p = monthly[m]
        marker = "+" if p >= 0 else ""
        bar = "█" * max(1, int(abs(p) / 20))
        side = "↑" if p >= 0 else "↓"
        print(f"    {m}: {nm:>4} trades  ${marker}{p:>8,.2f}  {side}{bar}")
        if p >= 0: pos_m += 1
        else: neg_m += 1

    print(f"\n  Profitable months: {pos_m}/{pos_m+neg_m} ({pos_m/(pos_m+neg_m)*100:.0f}%)")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
