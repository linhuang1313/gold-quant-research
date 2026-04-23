"""
M1 ML Scalper v4 — RR 修复 + 智能出场
========================================
从 v3 A5 (最佳配置) 出发, 专注解决 RR 问题:
1. 缩窄 SL ($5 -> $3.5/$4), 接近 TP 的 1.2x
2. 加入 trailing profit-lock 出场: 盈利超过 50% TP 后, 回撤到峰值 -30% 就平仓
3. Mean-reversion exit: 价格回到入场价附近时快速小亏平仓, 不等 SL
4. 更高 threshold 精选信号 (0.58-0.70)
5. 缩短 max hold (30 -> 15/20), 避免超时亏损
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


def train_ensemble_wf(df, features, labels, train_m=3, test_m=1):
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

        if len(X_tr) < 3000 or len(X_te) < 500:
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
        print(f"  F{fold:>2}: {p}  B_AUC={auc_b:.3f} S_AUC={auc_s:.3f} N={len(X_te):,}")

        cur += pd.DateOffset(months=test_m)
        del xgb_b, xgb_s, lgb_b, lgb_s; gc.collect()

    return preds


def ml_backtest_v4(df, preds, features, tp=3.0, sl=3.5, threshold=0.60,
                   spread=0.30, cooldown=5, max_hold=20,
                   session_hours=None, max_per_day=15,
                   scale_after=4, lots=(0.01, 0.02, 0.03),
                   vol_filter=True, max_atr_pct=0.50,
                   # v4 smart exit
                   trailing_lock=True, lock_trigger=0.5, lock_trail=0.35,
                   quick_cut=True, quick_cut_bars=10, quick_cut_loss=0.3,
                   ):
    """
    v4 backtest: 智能出场
    - trailing_lock: 盈利达到 TP*lock_trigger 后, 如果回撤到峰值*(1-lock_trail), 锁利平仓
    - quick_cut: 持仓超过 quick_cut_bars 且浮亏超过 quick_cut_loss*SL, 快速止损
    """
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

            cur_tp = tp * lm
            cur_sl = sl * lm
            held = i - pos['i']

            # Track peak profit
            if pnl_best > pos.get('peak_pnl', 0):
                pos['peak_pnl'] = pnl_best

            # Exit 1: TP hit
            if pnl_best >= cur_tp:
                _at(trades, equity, pos, c, ts, "TP", i, cur_tp)
                pos = None; last_close = i; cw += 1
                if cw >= scale_after: li = min(li + 1, len(lots) - 1)
                continue

            # Exit 2: SL hit
            if pnl_worst <= -cur_sl:
                _at(trades, equity, pos, c, ts, "SL", i, -cur_sl)
                pos = None; last_close = i; cw = 0; li = max(0, li - 1)
                continue

            # Exit 3: Trailing profit lock
            if trailing_lock:
                peak = pos.get('peak_pnl', 0)
                if peak >= cur_tp * lock_trigger and pnl_cur <= peak * (1 - lock_trail):
                    _at(trades, equity, pos, c, ts, "Lock", i, pnl_cur)
                    pos = None; last_close = i
                    if pnl_cur > 0: cw += 1
                    else: cw = 0
                    continue

            # Exit 4: Quick cut (long hold + still losing)
            if quick_cut and held >= quick_cut_bars:
                if pnl_cur <= -cur_sl * quick_cut_loss:
                    _at(trades, equity, pos, c, ts, "QCut", i, pnl_cur)
                    pos = None; last_close = i; cw = 0; li = max(0, li - 1)
                    continue

            # Exit 5: Max hold timeout
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

        ep = c + spread / 2 if d == 'BUY' else c - spread / 2
        pos = {'dir': d, 'e': ep, 'time': ts, 'lots': lots[li], 'i': i, 'peak_pnl': 0}
        dc[day] = dc.get(day, 0) + 1

    if pos:
        pnl = ((close[-1] - pos['e'] - spread) if pos['dir'] == 'BUY'
               else (pos['e'] - close[-1] - spread)) * pos['lots'] * 100
        _at(trades, equity, pos, close[-1], times[-1], "EOD", n - 1, pnl)

    return trades, equity


def _at(trades, equity, pos, ep, ts, reason, idx, pnl):
    trades.append(TradeRecord(
        strategy="MLv4", direction=pos['dir'],
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
    print("# M1 ML Scalper v4 — RR Fix + Smart Exit")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"
    df = load_m1(m1_path, start="2022-01-01")
    print(f"  Data: {len(df):,} M1 bars\n")

    features = build_features(df)
    print(f"  Features: {len(features.columns)}")

    print("\n  Building labels (TP=$3, SL=$5, MH=30)...")
    labels = build_labels(df, tp_pts=3.0, sl_pts=5.0, max_bars=30)
    print(f"  BuyR={labels['label_buy'].dropna().mean():.3f} SellR={labels['label_sell'].dropna().mean():.3f}")

    print("\n  Training Ensemble WF (3m train, 1m test)...")
    preds = train_ensemble_wf(df, features, labels, train_m=3, test_m=1)

    valid_p = preds.dropna()
    print(f"\n  Preds: {len(valid_p):,}")

    all_res = []

    # ═══════════════════════════════════════════════════════════
    # Group 1: RR 修复 — 缩小 SL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Group 1: SL 缩窄 (RR 修复)")
    print("=" * 60)

    for sl_val in [5.0, 4.0, 3.5, 3.0]:
        lbl = f"G1: TP$3 SL${sl_val} C>60% VF50%"
        tr, eq = ml_backtest_v4(df, preds, features,
                                tp=3.0, sl=sl_val, threshold=0.60,
                                cooldown=5, max_hold=20, spread=0.30,
                                vol_filter=True, max_atr_pct=0.50,
                                trailing_lock=False, quick_cut=False)
        r = report(tr, eq, lbl)
        all_res.append((lbl, r))

    # ═══════════════════════════════════════════════════════════
    # Group 2: 智能出场 (Lock + QCut)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Group 2: 智能出场")
    print("=" * 60)

    for sl_val in [4.0, 3.5]:
        # Lock only
        lbl = f"G2a: SL${sl_val} +Lock(50%/35%)"
        tr, eq = ml_backtest_v4(df, preds, features,
                                tp=3.0, sl=sl_val, threshold=0.60,
                                cooldown=5, max_hold=20, spread=0.30,
                                vol_filter=True, max_atr_pct=0.50,
                                trailing_lock=True, lock_trigger=0.5, lock_trail=0.35,
                                quick_cut=False)
        r = report(tr, eq, lbl)
        all_res.append((lbl, r))

        # QCut only
        lbl = f"G2b: SL${sl_val} +QCut(10b/30%)"
        tr, eq = ml_backtest_v4(df, preds, features,
                                tp=3.0, sl=sl_val, threshold=0.60,
                                cooldown=5, max_hold=20, spread=0.30,
                                vol_filter=True, max_atr_pct=0.50,
                                trailing_lock=False,
                                quick_cut=True, quick_cut_bars=10, quick_cut_loss=0.3)
        r = report(tr, eq, lbl)
        all_res.append((lbl, r))

        # Both
        lbl = f"G2c: SL${sl_val} +Lock+QCut"
        tr, eq = ml_backtest_v4(df, preds, features,
                                tp=3.0, sl=sl_val, threshold=0.60,
                                cooldown=5, max_hold=20, spread=0.30,
                                vol_filter=True, max_atr_pct=0.50,
                                trailing_lock=True, lock_trigger=0.5, lock_trail=0.35,
                                quick_cut=True, quick_cut_bars=10, quick_cut_loss=0.3)
        r = report(tr, eq, lbl)
        all_res.append((lbl, r))

    # ═══════════════════════════════════════════════════════════
    # Group 3: 更高 threshold
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Group 3: 高 Threshold 精选")
    print("=" * 60)

    for thresh in [0.62, 0.65, 0.70]:
        for sl_val in [4.0, 3.5]:
            lbl = f"G3: SL${sl_val} C>{thresh:.0%} +Lock+QCut"
            tr, eq = ml_backtest_v4(df, preds, features,
                                    tp=3.0, sl=sl_val, threshold=thresh,
                                    cooldown=5, max_hold=20, spread=0.30,
                                    vol_filter=True, max_atr_pct=0.50,
                                    trailing_lock=True, lock_trigger=0.5, lock_trail=0.35,
                                    quick_cut=True, quick_cut_bars=10, quick_cut_loss=0.3)
            r = report(tr, eq, lbl)
            all_res.append((lbl, r))

    # ═══════════════════════════════════════════════════════════
    # Group 4: TP 放大
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Group 4: TP 放大")
    print("=" * 60)

    for tp_val in [4.0, 5.0]:
        for sl_val in [4.0, 3.5]:
            lbl = f"G4: TP${tp_val} SL${sl_val} C>60% +Lock+QCut"
            tr, eq = ml_backtest_v4(df, preds, features,
                                    tp=tp_val, sl=sl_val, threshold=0.60,
                                    cooldown=5, max_hold=25, spread=0.30,
                                    vol_filter=True, max_atr_pct=0.50,
                                    trailing_lock=True, lock_trigger=0.5, lock_trail=0.35,
                                    quick_cut=True, quick_cut_bars=12, quick_cut_loss=0.3)
            r = report(tr, eq, lbl)
            all_res.append((lbl, r))

    # ═══════════════════════════════════════════════════════════
    # Group 5: 不同 Lock 参数
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Group 5: Lock 参数调优")
    print("=" * 60)

    for lt, ltr in [(0.4, 0.3), (0.3, 0.25), (0.6, 0.4), (0.5, 0.5)]:
        lbl = f"G5: SL$4 Lock({lt}/{ltr}) +QCut"
        tr, eq = ml_backtest_v4(df, preds, features,
                                tp=3.0, sl=4.0, threshold=0.60,
                                cooldown=5, max_hold=20, spread=0.30,
                                vol_filter=True, max_atr_pct=0.50,
                                trailing_lock=True, lock_trigger=lt, lock_trail=ltr,
                                quick_cut=True, quick_cut_bars=10, quick_cut_loss=0.3)
        r = report(tr, eq, lbl)
        all_res.append((lbl, r))

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL SUMMARY v4 — All Configs Ranked by Sharpe")
    print("=" * 80)
    all_res.sort(key=lambda x: x[1].get('sharpe', -999), reverse=True)
    print(f"{'Config':<50} {'N':>6} {'PnL':>10} {'Sharpe':>7} {'WR':>6} {'RR':>5} {'DD':>8}")
    print("-" * 95)
    for lbl, res in all_res:
        if res:
            m = " <<<" if res.get('sharpe', -999) == max(r.get('sharpe', -999) for _, r in all_res if r) else ""
            print(f"{lbl:<50} {res['n']:>6,} ${res['pnl']:>9,.2f} {res['sharpe']:>7.2f} "
                  f"{res['wr']:>5.1f}% {res.get('rr',0):>5.2f} ${res['dd']:>7,.2f}{m}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
