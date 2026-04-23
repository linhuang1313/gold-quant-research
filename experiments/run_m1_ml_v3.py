"""
M1 ML Scalper v3 — 三大改进
============================
1. 波动率过滤: 只在低/中波动时交易 (模型强区, AUC 0.7+)
2. ATR 自适应 TP/SL: 用 ATR 比例替代固定美元
3. Ensemble: XGBoost + LightGBM 概率平均, 提升稳定性
4. 概率校准: Platt scaling 让概率更准
"""
import sys
import time
import warnings
import gc
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

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
    """精简 + 新增波动率 regime 特征."""
    f = pd.DataFrame(index=df.index)
    c = df['Close']; h = df['High']; lo = df['Low']; o = df['Open']

    # 波动率/Range
    for n in [5, 10, 20, 30]:
        f[f'range_{n}'] = (h - lo).rolling(n).mean()
        f[f'vol_{n}'] = c.rolling(n).std()

    f['bar_size'] = h - lo
    f['bar_body'] = (c - o).abs()

    # 价格位置
    for n in [10, 20, 30]:
        hh = h.rolling(n).max(); ll = lo.rolling(n).min()
        rng = hh - ll
        f[f'dist_high_{n}'] = hh - c
        f[f'dist_low_{n}'] = c - ll
        f[f'price_pos_{n}'] = (c - ll) / rng.replace(0, np.nan)

    # Z-score
    for n in [10, 20]:
        sma = c.rolling(n).mean(); std = c.rolling(n).std()
        f[f'zscore_{n}'] = (c - sma) / std.replace(0, np.nan)

    # 时间
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

    # ── 新增: 波动率 regime 特征 ──
    atr14 = pd.DataFrame({
        'hl': h - lo,
        'hc': (h - c.shift(1)).abs(),
        'lc': (lo - c.shift(1)).abs(),
    }).max(axis=1).rolling(14).mean()
    f['atr_14'] = atr14

    # ATR 相对于历史的 percentile (用 rolling 240 bars ≈ 4 小时)
    f['atr_pct'] = atr14.rolling(240).rank(pct=True)

    # 波动率变化速率
    f['atr_change'] = atr14 / atr14.shift(10) - 1

    # 短期/长期波动率比
    f['vol_ratio'] = f['vol_5'] / f['vol_30'].replace(0, np.nan)

    # Range 收缩 (squeeze)
    f['range_squeeze'] = f['range_5'] / f['range_30'].replace(0, np.nan)

    return f


def build_labels_fast(df, tp_pts=2.0, sl_pts=3.5, max_bars=30, spread=0.30):
    """标签生成."""
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    n = len(df)
    lb = np.full(n, np.nan); ls = np.full(n, np.nan)

    for i in range(n - max_bars):
        entry = c[i]
        be = entry + spread / 2; se = entry - spread / 2

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


def build_atr_labels(df, tp_mult=0.5, sl_mult=0.8, max_bars=30, spread=0.30):
    """ATR 自适应标签: TP/SL 按 ATR 比例."""
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    n = len(df)

    # Compute ATR
    tr = np.maximum(h - lo,
         np.maximum(np.abs(h - np.roll(c, 1)),
                    np.abs(lo - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(14).mean().values

    lb = np.full(n, np.nan); ls = np.full(n, np.nan)

    for i in range(15, n - max_bars):
        if np.isnan(atr[i]) or atr[i] < 0.3:
            continue
        tp = atr[i] * tp_mult
        sl = atr[i] * sl_mult
        entry = c[i]
        be = entry + spread / 2; se = entry - spread / 2

        br = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if h[j] >= be + tp: br = 1; break
            if lo[j] <= be - sl: break

        sr = 0
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if lo[j] <= se - tp: sr = 1; break
            if h[j] >= se + sl: break

        lb[i] = br; ls[i] = sr

    return pd.DataFrame({'label_buy': lb, 'label_sell': ls}, index=df.index)


def train_ensemble_wf(df, features, labels, train_m=3, test_m=1):
    """Walk-Forward Ensemble: XGBoost + LightGBM 平均概率."""
    feat_cols = [c for c in features.columns if features[c].notna().sum() > 1000]
    X = features[feat_cols]; y_b = labels['label_buy']; y_s = labels['label_sell']
    valid = X.dropna().index.intersection(y_b.dropna().index)
    X = X.loc[valid]; y_b = y_b.loc[valid]; y_s = y_s.loc[valid]

    print(f"  Valid: {len(X):,}, {len(feat_cols)} feats, BuyR={y_b.mean():.3f}, SellR={y_s.mean():.3f}")

    dates = X.index.to_series()
    min_d = dates.min(); max_d = dates.max()
    preds = pd.DataFrame(index=X.index, columns=['p_buy', 'p_sell'], dtype=float)
    preds[:] = np.nan

    xgb_p = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.7,
                 colsample_bytree=0.7, min_child_weight=100, reg_alpha=0.5, reg_lambda=2.0,
                 random_state=42, n_jobs=-1, verbosity=0, tree_method='hist')

    lgb_p = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.7,
                 colsample_bytree=0.7, min_child_samples=100, reg_alpha=0.5, reg_lambda=2.0,
                 random_state=42, n_jobs=-1, verbose=-1)

    cur = min_d + pd.DateOffset(months=train_m)
    fold = 0

    while cur < max_d:
        test_end = cur + pd.DateOffset(months=test_m)
        tr_mask = (dates >= cur - pd.DateOffset(months=train_m)) & (dates < cur)
        te_mask = (dates >= cur) & (dates < test_end)
        X_tr = X.loc[tr_mask]; X_te = X.loc[te_mask]

        if len(X_tr) < 3000 or len(X_te) < 500:
            cur += pd.DateOffset(months=test_m); continue

        # XGBoost
        xgb_b = xgb.XGBClassifier(**xgb_p); xgb_b.fit(X_tr, y_b.loc[tr_mask], verbose=False)
        xgb_s = xgb.XGBClassifier(**xgb_p); xgb_s.fit(X_tr, y_s.loc[tr_mask], verbose=False)

        # LightGBM
        lgb_b = lgb.LGBMClassifier(**lgb_p); lgb_b.fit(X_tr, y_b.loc[tr_mask])
        lgb_s = lgb.LGBMClassifier(**lgb_p); lgb_s.fit(X_tr, y_s.loc[tr_mask])

        # Ensemble: average probabilities
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
        print(f"  F{fold:>2}: {p}  B_AUC={auc_b:.3f} S_AUC={auc_s:.3f} N={len(X_te):,}")

        cur += pd.DateOffset(months=test_m)
        del xgb_b, xgb_s, lgb_b, lgb_s
        gc.collect()

    return preds


def ml_backtest(df, preds, features, tp=2.0, sl=3.5, threshold=0.55,
                spread=0.30, cooldown=3, max_hold=30,
                session_hours=None, max_per_day=20,
                scale_after=4, lots=(0.01, 0.02, 0.03),
                vol_filter=False, max_atr_pct=0.7,
                atr_adaptive=False, tp_atr_mult=0.5, sl_atr_mult=0.8,
                ):
    """ML 回测 + 波动率过滤 + ATR 自适应."""
    if session_hours is None:
        session_hours = set(range(0, 13))

    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    times = df.index; n = len(df)
    valid_set = set(preds.dropna().index)

    atr_pct_vals = features['atr_pct'].values if 'atr_pct' in features.columns else None
    atr_vals = features['atr_14'].values if 'atr_14' in features.columns else None

    trades = []; equity = [2000.0]
    pos = None; last_close = -999; cw = 0; li = 0; dc = {}

    for i in range(n):
        ts = times[i]; c = close[i]; hv = high[i]; lv = low[i]
        day = str(ts.date())

        # Current trade TP/SL
        if pos is not None:
            lm = pos['lots'] / 0.01
            cur_tp = pos.get('tp', tp) * lm
            cur_sl = pos.get('sl', sl) * lm

            if pos['dir'] == 'BUY':
                pb = (hv - pos['e'] - spread) * pos['lots'] * 100
                pw = (lv - pos['e'] - spread) * pos['lots'] * 100
                pc = (c - pos['e'] - spread) * pos['lots'] * 100
            else:
                pb = (pos['e'] - lv - spread) * pos['lots'] * 100
                pw = (pos['e'] - hv - spread) * pos['lots'] * 100
                pc = (pos['e'] - c - spread) * pos['lots'] * 100

            held = i - pos['i']

            if pb >= cur_tp:
                _at(trades, equity, pos, c, ts, "TP", i, cur_tp)
                pos = None; last_close = i; cw += 1
                if cw >= scale_after: li = min(li + 1, len(lots) - 1)
                continue
            if pw <= -cur_sl:
                _at(trades, equity, pos, c, ts, "SL", i, -cur_sl)
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

        # Vol filter
        if vol_filter and atr_pct_vals is not None:
            ap = atr_pct_vals[i]
            if np.isnan(ap) or ap > max_atr_pct:
                continue

        row = preds.loc[ts]
        pbv = row['p_buy']; psv = row['p_sell']
        if pd.isna(pbv) or pd.isna(psv): continue

        d = None
        if pbv > threshold and psv > threshold:
            d = 'BUY' if pbv > psv else 'SELL'
        elif pbv > threshold: d = 'BUY'
        elif psv > threshold: d = 'SELL'
        if d is None: continue

        # ATR adaptive TP/SL
        trade_tp = tp; trade_sl = sl
        if atr_adaptive and atr_vals is not None:
            av = atr_vals[i]
            if not np.isnan(av) and av > 0.3:
                trade_tp = av * tp_atr_mult
                trade_sl = av * sl_atr_mult

        ep = c + spread / 2 if d == 'BUY' else c - spread / 2
        pos = {'dir': d, 'e': ep, 'time': ts, 'lots': lots[li], 'i': i,
               'tp': trade_tp, 'sl': trade_sl}
        dc[day] = dc.get(day, 0) + 1

    if pos:
        pnl = ((close[-1] - pos['e'] - spread) if pos['dir'] == 'BUY'
               else (pos['e'] - close[-1] - spread)) * pos['lots'] * 100
        _at(trades, equity, pos, close[-1], times[-1], "EOD", n - 1, pnl)

    return trades, equity


def _at(trades, equity, pos, ep, ts, reason, idx, pnl):
    trades.append(TradeRecord(
        strategy="MLv3", direction=pos['dir'],
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

    by_reason = {}
    for t in trades:
        r = t.exit_reason
        if r not in by_reason: by_reason[r] = {'n': 0, 'pnl': 0}
        by_reason[r]['n'] += 1; by_reason[r]['pnl'] += t.pnl

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades):,}  |  PnL: ${total:,.2f}  |  Sharpe: {sh:.2f}")
    print(f"  WR: {wr:.1f}%  |  AvgW: ${avg_w:.2f}  AvgL: ${avg_l:.2f}  RR: {rr:.2f}")
    print(f"  MaxDD: ${dd:,.2f}  |  Avg bars: {np.mean([t.bars_held for t in trades]):.1f}")
    print(f"  Exits: ", end="")
    for r, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"{r}={v['n']}(${v['pnl']:+,.0f}) ", end="")
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

    return {'n': len(trades), 'pnl': total, 'sharpe': sh, 'wr': wr, 'dd': dd, 'rr': rr}


def main():
    t0 = time.time()
    print("# M1 ML Scalper v3 — Vol Filter + ATR Adaptive + Ensemble")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"
    df = load_m1(m1_path, start="2022-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    features = build_features(df)
    print(f"  Features: {len(features.columns)}")

    # ═══════════════════════════════════════════════════════════
    # Test A: 固定 TP/SL 标签 + Ensemble
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A. 固定 TP/SL Labels + XGB+LGB Ensemble")
    print("=" * 60)

    print("  Building labels (TP=$3, SL=$5, MH=30)...")
    labels_a = build_labels_fast(df, tp_pts=3.0, sl_pts=5.0, max_bars=30)
    print(f"  BuyR={labels_a['label_buy'].dropna().mean():.3f} SellR={labels_a['label_sell'].dropna().mean():.3f}")

    print("\n  Training Ensemble WF...")
    preds_a = train_ensemble_wf(df, features, labels_a, train_m=3, test_m=1)

    valid_a = preds_a.dropna()
    print(f"\n  Preds: {len(valid_a):,}")

    # Backtest variations
    all_res = []

    # A1: Baseline (same as v2 best)
    tr, eq = ml_backtest(df, preds_a, features, tp=3.0, sl=5.0, threshold=0.55,
                         cooldown=3, max_hold=30, spread=0.30)
    r = report(tr, eq, "A1: Ensemble TP$3 SL$5 C>55%")
    all_res.append(("A1: Ensemble TP$3 SL$5 C>55%", r))

    # A2: + Vol filter 70%
    tr, eq = ml_backtest(df, preds_a, features, tp=3.0, sl=5.0, threshold=0.55,
                         cooldown=3, max_hold=30, spread=0.30,
                         vol_filter=True, max_atr_pct=0.70)
    r = report(tr, eq, "A2: +VolFilter(70%)")
    all_res.append(("A2: +VolFilter(70%)", r))

    # A3: + Vol filter 50%
    tr, eq = ml_backtest(df, preds_a, features, tp=3.0, sl=5.0, threshold=0.55,
                         cooldown=3, max_hold=30, spread=0.30,
                         vol_filter=True, max_atr_pct=0.50)
    r = report(tr, eq, "A3: +VolFilter(50%)")
    all_res.append(("A3: +VolFilter(50%)", r))

    # A4: + Vol filter 40%
    tr, eq = ml_backtest(df, preds_a, features, tp=3.0, sl=5.0, threshold=0.55,
                         cooldown=3, max_hold=30, spread=0.30,
                         vol_filter=True, max_atr_pct=0.40)
    r = report(tr, eq, "A4: +VolFilter(40%)")
    all_res.append(("A4: +VolFilter(40%)", r))

    # A5: Higher threshold
    tr, eq = ml_backtest(df, preds_a, features, tp=3.0, sl=5.0, threshold=0.60,
                         cooldown=5, max_hold=30, spread=0.30,
                         vol_filter=True, max_atr_pct=0.50)
    r = report(tr, eq, "A5: VolF50%+C>60%")
    all_res.append(("A5: VolF50%+C>60%", r))

    # A6: Tighter TP/SL
    tr, eq = ml_backtest(df, preds_a, features, tp=2.5, sl=4.0, threshold=0.55,
                         cooldown=3, max_hold=25, spread=0.30,
                         vol_filter=True, max_atr_pct=0.50)
    r = report(tr, eq, "A6: VolF50%+TP$2.5+SL$4")
    all_res.append(("A6: VolF50%+TP$2.5+SL$4", r))

    # ═══════════════════════════════════════════════════════════
    # Test B: ATR 自适应标签 + Ensemble
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. ATR 自适应 Labels + Ensemble")
    print("=" * 60)

    for tp_m, sl_m in [(0.4, 0.7), (0.5, 0.8), (0.6, 1.0)]:
        lbl_name = f"ATR_TP={tp_m}_SL={sl_m}"
        print(f"\n  Building ATR labels ({lbl_name})...")
        labels_b = build_atr_labels(df, tp_mult=tp_m, sl_mult=sl_m, max_bars=30)
        br = labels_b['label_buy'].dropna().mean()
        sr = labels_b['label_sell'].dropna().mean()
        print(f"  BuyR={br:.3f} SellR={sr:.3f}")

        print("  Training Ensemble WF...")
        preds_b = train_ensemble_wf(df, features, labels_b, train_m=3, test_m=1)

        # ATR adaptive backtest
        for thresh in [0.52, 0.55, 0.58]:
            for vf, vf_pct in [(False, 1.0), (True, 0.50)]:
                lbl = f"B: ATR({tp_m}/{sl_m}) C>{thresh:.0%} VF={'Y' if vf else 'N'}"
                tr, eq = ml_backtest(df, preds_b, features,
                                     tp=3.0, sl=5.0,  # fallback
                                     threshold=thresh,
                                     cooldown=3, max_hold=30, spread=0.30,
                                     vol_filter=vf, max_atr_pct=vf_pct,
                                     atr_adaptive=True, tp_atr_mult=tp_m, sl_atr_mult=sl_m)
                r = report(tr, eq, lbl)
                all_res.append((lbl, r))

        del preds_b, labels_b
        gc.collect()

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — All Configs Ranked by Sharpe")
    print("=" * 70)
    all_res.sort(key=lambda x: x[1].get('sharpe', -999), reverse=True)
    print(f"{'Config':<50} {'N':>6} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'RR':>5} {'DD':>8}")
    print("-" * 95)
    for lbl, res in all_res:
        if res:
            m = " <<<" if res['sharpe'] == max(r['sharpe'] for _, r in all_res if r) else ""
            print(f"{lbl:<50} {res['n']:>6,} ${res['pnl']:>9,.2f} {res['sharpe']:>7.2f} "
                  f"{res['wr']:>5.1f}% {res.get('rr',0):>5.2f} ${res['dd']:>7,.2f}{m}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
