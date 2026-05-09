#!/usr/bin/env python3
"""
R116-C: H1 Overnight Strategy — Entry/Exit Timing + ML Filter
==============================================================
Uses 11-year Dukascopy H1 data (2015-2026).

Phase 1: Find optimal UTC entry hour & exit hour for Thu/Fri overnight BUY
Phase 2: Test various entry/exit combinations + SL/TP in H1 precision
Phase 3: XGBoost ML filter — can ML improve the overnight signal?
Phase 4: K-Fold validation (5 folds)
Phase 5: Portfolio integration (6th strategy alongside existing 5)
"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r116c_h1_overnight_ml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
t0 = time.time()

FOLDS = [
    ("Fold1", "2015-01-01", "2017-03-01"),
    ("Fold2", "2017-03-01", "2019-06-01"),
    ("Fold3", "2019-06-01", "2021-09-01"),
    ("Fold4", "2021-09-01", "2023-12-01"),
    ("Fold5", "2023-12-01", "2027-01-01"),
]


def sharpe(arr, ann=252):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(ann)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def metrics_arr(pnl_arr):
    if len(pnl_arr) < 5:
        return {'n': len(pnl_arr), 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg': 0}
    wins = (pnl_arr > 0).sum()
    return {
        'n': len(pnl_arr), 'sharpe': round(sharpe(pnl_arr), 3),
        'pnl': round(float(pnl_arr.sum()), 2), 'max_dd': round(max_dd(pnl_arr), 2),
        'wr': round(wins / len(pnl_arr) * 100, 1),
        'avg': round(float(pnl_arr.mean()), 4),
    }


def load_h1():
    df = pd.read_csv(DATA_DIR / "download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[df['Close'] > 100]

    tr = pd.concat([df['High'] - df['Low'],
                     (df['High'] - df['Close'].shift()).abs(),
                     (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_pct'] = df['ATR14'] / df['Close'] * 100

    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['date'] = df.index.date

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()

    df['ret_1h'] = df['Close'].pct_change()
    df['vol_20h'] = df['ret_1h'].rolling(20).std()
    df['atr_rank'] = df['ATR_pct'].rolling(24*252).rank(pct=True)

    return df.dropna(subset=['ATR14'])


def main():
    print("=" * 80)
    print("  R116-C: H1 Overnight Strategy — Entry/Exit + ML Filter")
    print("=" * 80)

    h1 = load_h1()
    print(f"  H1 data: {len(h1)} bars ({h1.index[0]} ~ {h1.index[-1]})")
    print(f"  Date range: {h1['date'].min()} ~ {h1['date'].max()}")

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Entry/Exit Hour Heatmap
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 1: Entry Hour → Exit Hour Return Heatmap")
    print("  (Thu+Fri BUY only, entry at close of hour, exit at close of hour)")
    print("=" * 70)

    thu_fri = h1[h1['dow'].isin([3, 4])]
    dates_list = sorted(thu_fri['date'].unique())

    heatmap = {}
    for entry_h in range(0, 24, 2):
        for exit_h in range(0, 24, 2):
            if exit_h == entry_h: continue
            pnls = []
            for d in dates_list:
                day_data = thu_fri[thu_fri['date'] == d]
                entry_bar = day_data[day_data['hour'] == entry_h]
                if entry_h < exit_h:
                    exit_bar = day_data[day_data['hour'] == exit_h]
                else:
                    next_dates = [dd for dd in dates_list if dd > d]
                    if not next_dates: continue
                    next_d = next_dates[0]
                    next_day = thu_fri[thu_fri['date'] == next_d]
                    exit_bar = next_day[next_day['hour'] == exit_h]

                if len(entry_bar) == 0 or len(exit_bar) == 0: continue
                entry_price = float(entry_bar['Close'].iloc[0])
                exit_price = float(exit_bar['Close'].iloc[0])
                pnl = (exit_price - entry_price - SPREAD) * UNIT_LOT * PV
                pnls.append(pnl)

            if len(pnls) >= 20:
                arr = np.array(pnls)
                heatmap[(entry_h, exit_h)] = {
                    'sharpe': round(sharpe(arr), 3), 'n': len(arr),
                    'pnl': round(arr.sum(), 2), 'wr': round((arr > 0).sum() / len(arr) * 100, 1),
                }

    top = sorted(heatmap.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:15]
    print(f"\n  Top 15 entry→exit combos (Thu+Fri BUY):")
    print(f"  {'Entry':>6s} {'Exit':>6s}  {'Sharpe':>7s}  {'n':>5s}  {'PnL':>10s}  {'WR':>6s}")
    for (eh, xh), m in top:
        next_day = "next" if xh <= eh else "same"
        print(f"  UTC{eh:02d}→UTC{xh:02d}  {m['sharpe']:7.3f}  {m['n']:5d}  ${m['pnl']:>9.0f}  {m['wr']:5.1f}%  ({next_day} day)")

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Detailed Overnight Strategy Backtest (H1 precision)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 2: H1-Precision Overnight Backtest with SL/TP")
    print("=" * 70)

    def bt_overnight_h1(data, entry_hour=20, exit_hour=8, dow_filter=[3, 4],
                        sl_atr=2.0, tp_atr=0, trend_filter=False):
        """
        BUY at entry_hour close, hold overnight, exit at exit_hour close next day.
        Intra-bar SL checking using High/Low.
        """
        dates = sorted(data['date'].unique())
        trades = []

        for i, d in enumerate(dates):
            day_data = data[data['date'] == d]
            if len(day_data) == 0: continue
            if day_data['dow'].iloc[0] not in dow_filter: continue

            entry_bars = day_data[day_data['hour'] == entry_hour]
            if len(entry_bars) == 0: continue

            entry_bar = entry_bars.iloc[0]
            atr = float(entry_bar['ATR14'])
            if np.isnan(atr) or atr < 0.1: continue

            if trend_filter:
                sma = float(entry_bar['SMA200']) if 'SMA200' in entry_bar.index else np.nan
                if np.isnan(sma) or float(entry_bar['Close']) < sma: continue

            entry_price = float(entry_bar['Close']) + SPREAD / 2
            sl_price = entry_price - atr * sl_atr
            tp_price = entry_price + atr * tp_atr if tp_atr > 0 else 1e9

            exit_found = False

            if exit_hour > entry_hour:
                check_bars = day_data[day_data['hour'] > entry_hour]
                next_dates = [dd for dd in dates if dd > d]
            else:
                check_bars = day_data[day_data['hour'] > entry_hour]
                next_dates = [dd for dd in dates if dd > d]

            for _, bar in check_bars.iterrows():
                low = float(bar['Low']); high = float(bar['High'])
                if low <= sl_price:
                    pnl = -(atr * sl_atr) * UNIT_LOT * PV
                    trades.append({'pnl': pnl, 'reason': 'SL', 'date': str(d)})
                    exit_found = True; break
                if high >= tp_price and tp_atr > 0:
                    pnl = (atr * tp_atr) * UNIT_LOT * PV
                    trades.append({'pnl': pnl, 'reason': 'TP', 'date': str(d)})
                    exit_found = True; break

            if exit_found: continue

            if next_dates:
                next_d = next_dates[0]
                next_data = data[data['date'] == next_d]
                if exit_hour <= entry_hour:
                    check_next = next_data[next_data['hour'] <= exit_hour]
                else:
                    check_next = next_data

                for _, bar in check_next.iterrows():
                    low = float(bar['Low']); high = float(bar['High'])
                    if low <= sl_price:
                        pnl = -(atr * sl_atr) * UNIT_LOT * PV
                        trades.append({'pnl': pnl, 'reason': 'SL', 'date': str(d)})
                        exit_found = True; break
                    if high >= tp_price and tp_atr > 0:
                        pnl = (atr * tp_atr) * UNIT_LOT * PV
                        trades.append({'pnl': pnl, 'reason': 'TP', 'date': str(d)})
                        exit_found = True; break

                if not exit_found:
                    exit_bars = next_data[next_data['hour'] == exit_hour]
                    if len(exit_bars) > 0:
                        exit_price = float(exit_bars.iloc[0]['Close'])
                        pnl = (exit_price - entry_price - SPREAD) * UNIT_LOT * PV
                        trades.append({'pnl': pnl, 'reason': 'Time', 'date': str(d)})

        return trades

    configs = [
        ('E20_X06_SL2', 20, 6, [3, 4], 2.0, 0, False),
        ('E20_X08_SL2', 20, 8, [3, 4], 2.0, 0, False),
        ('E20_X10_SL2', 20, 10, [3, 4], 2.0, 0, False),
        ('E22_X06_SL2', 22, 6, [3, 4], 2.0, 0, False),
        ('E22_X08_SL2', 22, 8, [3, 4], 2.0, 0, False),
        ('E22_X10_SL2', 22, 10, [3, 4], 2.0, 0, False),
        ('E18_X06_SL2', 18, 6, [3, 4], 2.0, 0, False),
        ('E18_X08_SL2', 18, 8, [3, 4], 2.0, 0, False),
        ('E20_X08_SL3', 20, 8, [3, 4], 3.0, 0, False),
        ('E20_X08_SL2_TP2', 20, 8, [3, 4], 2.0, 2.0, False),
        ('E20_X08_SL2_TP3', 20, 8, [3, 4], 2.0, 3.0, False),
        ('E20_X08_SL2_trend', 20, 8, [3, 4], 2.0, 0, True),
        ('E20_X08_SL2_MonThuFri', 20, 8, [0, 3, 4], 2.0, 0, False),
        ('E22_X08_SL2_MonThuFri', 22, 8, [0, 3, 4], 2.0, 0, False),
    ]

    phase2_results = {}
    print(f"\n  {'Config':<28s}  {'n':>5s}  {'Sharpe':>7s}  {'PnL':>10s}  {'WR':>6s}  {'MaxDD':>8s}  {'SL%':>5s}  {'TP%':>5s}")
    print("  " + "-" * 85)

    for label, eh, xh, dows, sl, tp, tf in configs:
        trades = bt_overnight_h1(h1, entry_hour=eh, exit_hour=xh, dow_filter=dows,
                                 sl_atr=sl, tp_atr=tp, trend_filter=tf)
        if not trades:
            print(f"  {label:<28s}  NO TRADES")
            continue
        pnls = np.array([t['pnl'] for t in trades])
        m = metrics_arr(pnls)
        sl_pct = sum(1 for t in trades if t['reason'] == 'SL') / len(trades) * 100
        tp_pct = sum(1 for t in trades if t['reason'] == 'TP') / len(trades) * 100
        print(f"  {label:<28s}  {m['n']:5d}  {m['sharpe']:7.3f}  ${m['pnl']:>9.0f}  {m['wr']:5.1f}%  ${m['max_dd']:>7.0f}  {sl_pct:4.1f}%  {tp_pct:4.1f}%")
        phase2_results[label] = {**m, 'trades': trades}

    # ════════════════════════════════════════════════════════════════
    # Phase 3: ML Filter (XGBoost)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 3: ML Filter — Can XGBoost Improve Overnight Entry?")
    print("=" * 70)

    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        has_xgb = True
    except ImportError:
        print("  WARNING: XGBoost not available, skipping ML phase")
        has_xgb = False

    ml_results = {}

    if has_xgb:
        best_label = 'E20_X08_SL2'
        if best_label not in phase2_results:
            best_label = list(phase2_results.keys())[0]
        best_trades = phase2_results[best_label]['trades']

        entry_hour = 20
        exit_hour = 8
        dow_filter = [3, 4]

        print(f"  Base strategy: {best_label} ({len(best_trades)} trades)")
        print("  Building feature matrix from H1 bars at entry time...")

        features = []
        labels = []

        dates_used = set()
        for t in best_trades:
            d = t['date']
            if d in dates_used: continue
            dates_used.add(d)

            day_data = h1[h1['date'] == pd.Timestamp(d).date()]
            entry_bars = day_data[day_data['hour'] == entry_hour]
            if len(entry_bars) == 0: continue

            eb = entry_bars.iloc[0]

            atr = float(eb['ATR14']) if not pd.isna(eb['ATR14']) else 0
            close = float(eb['Close'])
            sma50 = float(eb['SMA50']) if not pd.isna(eb.get('SMA50', np.nan)) else close
            sma200 = float(eb['SMA200']) if not pd.isna(eb.get('SMA200', np.nan)) else close
            ema20 = float(eb['EMA20']) if not pd.isna(eb.get('EMA20', np.nan)) else close
            vol = float(eb['vol_20h']) if not pd.isna(eb.get('vol_20h', np.nan)) else 0

            recent = day_data[day_data['hour'] <= entry_hour]
            if len(recent) < 3: continue

            intraday_ret = (close - float(recent['Open'].iloc[0])) / float(recent['Open'].iloc[0]) * 100
            intraday_range = (float(recent['High'].max()) - float(recent['Low'].min())) / close * 100
            last_3h_ret = (close - float(recent['Close'].iloc[-min(3, len(recent))])) / close * 100

            feat = {
                'atr_pct': atr / close * 100 if close > 0 else 0,
                'close_vs_sma50': (close - sma50) / sma50 * 100 if sma50 > 0 else 0,
                'close_vs_sma200': (close - sma200) / sma200 * 100 if sma200 > 0 else 0,
                'close_vs_ema20': (close - ema20) / ema20 * 100 if ema20 > 0 else 0,
                'vol_20h': vol * 100,
                'intraday_ret': intraday_ret,
                'intraday_range': intraday_range,
                'last_3h_ret': last_3h_ret,
                'dow': float(day_data['dow'].iloc[0]),
                'hour': float(entry_hour),
            }

            features.append(feat)
            labels.append(1 if t['pnl'] > 0 else 0)

        X = pd.DataFrame(features)
        y = np.array(labels)
        print(f"  Features: {X.shape}, Win rate: {y.mean()*100:.1f}%")

        tscv = TimeSeriesSplit(n_splits=5)
        fold_aucs = []
        fold_improvements = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0, random_state=42
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5
            fold_aucs.append(round(auc, 3))

            base_trades_fold = [best_trades[i] for i in test_idx]
            base_pnls = np.array([t['pnl'] for t in base_trades_fold])
            base_sharpe = sharpe(base_pnls)

            for threshold in [0.50, 0.55, 0.60, 0.65]:
                mask = proba >= threshold
                if mask.sum() < 5: continue
                filtered_pnls = base_pnls[mask]
                filtered_sharpe = sharpe(filtered_pnls)
                fold_improvements.append({
                    'fold': fold_idx, 'threshold': threshold,
                    'base_sharpe': round(base_sharpe, 3),
                    'ml_sharpe': round(filtered_sharpe, 3),
                    'improvement': round(filtered_sharpe - base_sharpe, 3),
                    'kept_pct': round(mask.sum() / len(mask) * 100, 1),
                    'n_base': len(base_pnls), 'n_ml': int(mask.sum()),
                    'auc': round(auc, 3),
                })

        print(f"\n  Walk-Forward AUCs: {fold_aucs}")
        print(f"  Mean AUC: {np.mean(fold_aucs):.3f}")

        print(f"\n  ML Filter Impact by Threshold:")
        print(f"  {'Thresh':>7s}  {'Fold':>5s}  {'Base':>7s}  {'ML':>7s}  {'Delta':>7s}  {'Kept%':>6s}  {'AUC':>5s}")
        for r in fold_improvements:
            marker = "+" if r['improvement'] > 0 else " "
            print(f"  {r['threshold']:7.2f}  {r['fold']:5d}  {r['base_sharpe']:7.3f}  {r['ml_sharpe']:7.3f}  "
                  f"{marker}{r['improvement']:6.3f}  {r['kept_pct']:5.1f}%  {r['auc']:.3f}")

        by_thresh = {}
        for r in fold_improvements:
            t = r['threshold']
            if t not in by_thresh:
                by_thresh[t] = {'improvements': [], 'kept': [], 'aucs': []}
            by_thresh[t]['improvements'].append(r['improvement'])
            by_thresh[t]['kept'].append(r['kept_pct'])
            by_thresh[t]['aucs'].append(r['auc'])

        print(f"\n  Summary by Threshold:")
        for t, v in sorted(by_thresh.items()):
            avg_imp = np.mean(v['improvements'])
            avg_kept = np.mean(v['kept'])
            pos = sum(1 for x in v['improvements'] if x > 0)
            marker = "BETTER" if avg_imp > 0 and pos >= 3 else "WORSE" if avg_imp < 0 else "MIXED"
            print(f"    Threshold={t:.2f}: avg_improvement={avg_imp:+.3f}, kept={avg_kept:.0f}%, "
                  f"positive={pos}/{len(v['improvements'])} [{marker}]")

        importances = model.feature_importances_
        feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
        print(f"\n  Feature Importances (last fold):")
        for fname, imp in feat_imp:
            bar = "#" * int(imp * 50)
            print(f"    {fname:20s}: {imp:.3f} {bar}")

        ml_results = {
            'fold_aucs': fold_aucs, 'mean_auc': round(np.mean(fold_aucs), 3),
            'improvements': fold_improvements,
            'feature_importances': {f: round(float(i), 4) for f, i in feat_imp},
        }

    # ════════════════════════════════════════════════════════════════
    # Phase 4: K-Fold Validation of Best H1 Configs
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 4: K-Fold Validation (H1 precision)")
    print("=" * 70)

    kf_configs = [
        ('E20_X08_SL2_ThuFri', 20, 8, [3, 4], 2.0, 0, False),
        ('E22_X08_SL2_ThuFri', 22, 8, [3, 4], 2.0, 0, False),
        ('E20_X08_SL2_MonThuFri', 20, 8, [0, 3, 4], 2.0, 0, False),
        ('E20_X08_SL2_ThuFri_trend', 20, 8, [3, 4], 2.0, 0, True),
        ('E20_X10_SL2_ThuFri', 20, 10, [3, 4], 2.0, 0, False),
        ('E20_X06_SL2_ThuFri', 20, 6, [3, 4], 2.0, 0, False),
    ]

    kf_results = {}
    for label, eh, xh, dows, sl, tp, tf in kf_configs:
        fold_sharpes = []
        fold_pnls = []
        fold_trades = []
        for fname, start, end in FOLDS:
            sub = h1[(h1.index >= start) & (h1.index < end)]
            trades = bt_overnight_h1(sub, entry_hour=eh, exit_hour=xh, dow_filter=dows,
                                     sl_atr=sl, tp_atr=tp, trend_filter=tf)
            if not trades:
                fold_sharpes.append(0.0); fold_pnls.append(0.0); fold_trades.append(0)
                continue
            pnls = np.array([t['pnl'] for t in trades])
            fold_sharpes.append(round(sharpe(pnls), 3))
            fold_pnls.append(round(pnls.sum(), 2))
            fold_trades.append(len(trades))

        pos = sum(1 for s in fold_sharpes if s > 0)
        status = "PASS" if pos >= 3 else "FAIL"
        mean_s = round(np.mean(fold_sharpes), 3)
        print(f"  {label:<35s}: {fold_sharpes} -> {pos}/5 [{status}] mean={mean_s} (trades: {fold_trades})")
        kf_results[label] = {
            'fold_sharpes': fold_sharpes, 'fold_pnls': fold_pnls,
            'fold_trades': fold_trades,
            'positive': pos, 'mean': mean_s, 'pass': pos >= 3,
        }

    # ════════════════════════════════════════════════════════════════
    # Phase 5: Portfolio Integration Analysis
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Phase 5: Portfolio Integration — 6th Strategy Viability")
    print("=" * 70)

    best_config = 'E20_X08_SL2_ThuFri'
    if best_config in phase2_results:
        bt = phase2_results[best_config]
        trades_per_year = bt['n'] / 11.0
        print(f"\n  {best_config}:")
        print(f"    Total trades: {bt['n']} over ~11 years")
        print(f"    Trades/year: {trades_per_year:.1f}")
        print(f"    Sharpe: {bt['sharpe']}")
        print(f"    PnL at 0.01 lot: ${bt['pnl']:.0f}")
        print(f"    Max DD: ${bt['max_dd']:.0f}")
        print(f"    Win rate: {bt['wr']}%")

        for lot in [0.01, 0.02, 0.03, 0.05]:
            scale = lot / 0.01
            print(f"    At {lot:.2f} lot: PnL=${bt['pnl']*scale:.0f}, MaxDD=${bt['max_dd']*scale:.0f}")

    # Correlation check
    print(f"\n  Trade Timing Overlap Check:")
    print(f"    Entry: UTC 20:00 (Thu/Fri)")
    print(f"    Exit:  UTC 08:00 (next day)")
    print(f"    Duration: ~12 hours overnight")
    print(f"    Overlap with existing strategies:")
    print(f"      PSAR: Entry any time -> may overlap")
    print(f"      TSMOM: Entry any time -> may overlap")
    print(f"      SESS_BO: NY session -> entry before overnight exit, LOW overlap")
    print(f"      L8_MAX: Any time -> may overlap")
    print(f"      Donchian: Any time -> may overlap")
    print(f"    Direction: BUY only -> no conflict with SELL signals")

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    all_results = {
        'phase2_backtest': {k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in phase2_results.items()},
        'phase3_ml': ml_results,
        'phase4_kfold': kf_results,
        'elapsed_s': round(elapsed, 1),
    }

    out_file = OUTPUT_DIR / "r116c_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  Best H1 Config: Entry UTC20, Exit UTC08, Thu+Fri, SL=2x ATR")
    print(f"  K-Fold Results:")
    for k, v in kf_results.items():
        status = "PASS" if v['pass'] else "FAIL"
        print(f"    {k:<35s}: {v['positive']}/5 [{status}] mean={v['mean']}")

    if ml_results:
        print(f"\n  ML Filter: Mean AUC={ml_results['mean_auc']}")
        best_thresh = max(by_thresh.items(), key=lambda x: np.mean(x[1]['improvements']))
        avg_imp = np.mean(best_thresh[1]['improvements'])
        print(f"  Best threshold: {best_thresh[0]:.2f} (avg improvement: {avg_imp:+.3f})")

    print(f"\n  Saved: {out_file}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
