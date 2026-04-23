"""
R26: Daily/H4 Keltner K-Fold Validation + Correlation with L7
===============================================================
Phase A: K-Fold validation on top Daily Keltner configs
Phase B: K-Fold validation on top H4 Keltner configs
Phase C: Correlation analysis with L7 (daily PnL overlap)
Phase D: Combined portfolio simulation (L7 + best new strategy)
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUT_DIR = Path("results/round26_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except UnicodeEncodeError: f.write(data.encode('ascii', errors='replace').decode('ascii'))
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


L7_KWARGS = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
}

# ── Keltner helpers (from R25) ──

def compute_adx(df, period=14):
    high = df['High']; low = df['Low']; close = df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.DataFrame({'hl': high-low, 'hc': (high-close.shift(1)).abs(),
                        'lc': (low-close.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_s)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_s)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(period).mean()


def add_kc(df, ema_period=20, atr_period=14, mult=1.5):
    df = df.copy()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    df['ATR'] = tr.rolling(atr_period).mean()
    df['KC_upper'] = df['EMA'] + mult * df['ATR']
    df['KC_lower'] = df['EMA'] - mult * df['ATR']
    df['ADX'] = compute_adx(df, atr_period)
    return df


def backtest_kc(df, label, ema=20, atr_p=14, mult=1.5, adx_thresh=18,
                sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06,
                max_hold=20, spread=0.30, lot=0.03, return_trades=False):
    """KC backtest returning stats dict + optionally raw trades."""
    df = add_kc(df, ema, atr_p, mult)
    df = df.dropna()

    trades = []; pos = None; equity = [2000.0]
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
    atr = df['ATR'].values; adx_arr = df['ADX'].values
    times = df.index; n = len(df)
    last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]
        cur_atr = atr[i]; cur_adx = adx_arr[i]

        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_high = (h - pos['entry'] - spread) * lot * 100
                pnl_low = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_cur = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_high = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_low = (pos['entry'] - h - spread) * lot * 100
                pnl_cur = (pos['entry'] - c - spread) * lot * 100

            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100

            exited = False
            if pnl_high >= tp_val:
                _at(trades, equity, pos, c, times[i], "TP", i, tp_val)
                exited = True
            elif pnl_low <= -sl_val:
                _at(trades, equity, pos, c, times[i], "SL", i, -sl_val)
                exited = True
            else:
                act_dist = trail_act_atr * pos['atr']
                trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_price = h - trail_d
                    if lo_v <= ts_price:
                        _at(trades, equity, pos, c, times[i], "Trail", i,
                            (ts_price - pos['entry'] - spread) * lot * 100)
                        exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_price = lo_v + trail_d
                    if h >= ts_price:
                        _at(trades, equity, pos, c, times[i], "Trail", i,
                            (pos['entry'] - ts_price - spread) * lot * 100)
                        exited = True

                if not exited and held >= max_hold:
                    _at(trades, equity, pos, c, times[i], "Timeout", i, pnl_cur)
                    exited = True

            if exited:
                pos = None; last_exit = i
                continue

        if pos is not None:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(cur_adx) or cur_adx < adx_thresh:
            continue
        if np.isnan(cur_atr) or cur_atr < 0.1:
            continue

        prev_c = close[i-1]
        if prev_c > kc_up[i-1]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i,
                   'time': times[i], 'atr': cur_atr}
        elif prev_c < kc_lo[i-1]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i,
                   'time': times[i], 'atr': cur_atr}

    if not trades:
        r = {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0,
             'win_rate': 0, 'max_dd': 0}
        if return_trades: r['_trades'] = []
        return r

    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls)
    dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()

    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

    r = {'label': label, 'n': len(trades), 'sharpe': sh,
         'total_pnl': sum(pnls), 'win_rate': wins/len(trades)*100, 'max_dd': dd}
    if return_trades:
        r['_trades'] = trades
        r['_daily_pnl'] = daily
    return r


def _at(trades, equity, pos, exit_p, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
                   'entry_time': pos['time'], 'exit_time': exit_time,
                   'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})
    equity.append(equity[-1] + pnl)


def kc_kfold(df_full, label, n_folds=6, **kw):
    """Run K-Fold on KC backtest by time-splitting the dataframe."""
    df_full = df_full.dropna()
    n = len(df_full)
    fold_size = n // n_folds
    results = []

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        df_fold = df_full.iloc[start:end]
        if len(df_fold) < 50:
            continue
        fl = f"{label}Fold{i+1}"
        r = backtest_kc(df_fold, fl, **kw)
        print(f"    {fl}: N={r['n']}, Sharpe={r['sharpe']:.2f}, "
              f"PnL=${r['total_pnl']:.0f}, WR={r['win_rate']:.1f}%, MaxDD=${r['max_dd']:.0f}")
        results.append(r)

    if not results:
        print(f"  K-Fold [{label}]: No results")
        return results

    sharpes = [r['sharpe'] for r in results]
    pos = sum(1 for s in sharpes if s > 0)
    print(f"\n  K-Fold [{label}]: {pos}/{len(results)} positive, "
          f"mean={np.mean(sharpes):.2f}, min={min(sharpes):.2f}, max={max(sharpes):.2f}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase A: Daily Keltner K-Fold
# ═══════════════════════════════════════════════════════════════

def run_phase_A(h1_df):
    print("\n" + "=" * 80)
    print("Phase A: Daily Keltner K-Fold Validation")
    print("=" * 80)

    d1 = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    print(f"  D1: {len(d1):,} bars ({d1.index[0]} -> {d1.index[-1]})")

    configs = [
        ("D1_EMA10_M2.0_ADX22", dict(ema=10, mult=2.0, adx_thresh=22, max_hold=15,
                                       trail_act_atr=0.40, trail_dist_atr=0.10)),
        ("D1_EMA20_M2.0_ADX18", dict(ema=20, mult=2.0, adx_thresh=18, max_hold=15,
                                       trail_act_atr=0.40, trail_dist_atr=0.10)),
        ("D1_EMA20_M2.0_ADX15", dict(ema=20, mult=2.0, adx_thresh=15, max_hold=15,
                                       trail_act_atr=0.40, trail_dist_atr=0.10)),
        ("D1_EMA15_M1.5_ADX15", dict(ema=15, mult=1.5, adx_thresh=15, max_hold=15,
                                       trail_act_atr=0.40, trail_dist_atr=0.10)),
    ]

    d1_best = {}
    for label, kw in configs:
        print(f"\n  --- {label} ---")
        # Full backtest first
        r_full = backtest_kc(d1, f"{label}_FULL", return_trades=True, **kw)
        print(f"  FULL: N={r_full['n']}, Sharpe={r_full['sharpe']:.2f}, "
              f"PnL=${r_full['total_pnl']:.0f}, WR={r_full['win_rate']:.1f}%, MaxDD=${r_full['max_dd']:.0f}")

        # K-Fold
        kf = kc_kfold(d1, label, n_folds=6, **kw)
        if kf:
            d1_best[label] = {'full': r_full, 'kfold': kf}

    return d1_best


# ═══════════════════════════════════════════════════════════════
# Phase B: H4 Keltner K-Fold
# ═══════════════════════════════════════════════════════════════

def run_phase_B(h1_df):
    print("\n" + "=" * 80)
    print("Phase B: H4 Keltner K-Fold Validation")
    print("=" * 80)

    h4 = h1_df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    print(f"  H4: {len(h4):,} bars ({h4.index[0]} -> {h4.index[-1]})")

    configs = [
        ("H4_EMA20_M2.0_ADX18", dict(ema=20, mult=2.0, adx_thresh=18, max_hold=30,
                                       trail_act_atr=0.28, trail_dist_atr=0.06)),
        ("H4_EMA20_M1.5_ADX22", dict(ema=20, mult=1.5, adx_thresh=22, max_hold=30,
                                       trail_act_atr=0.28, trail_dist_atr=0.06)),
    ]

    h4_best = {}
    for label, kw in configs:
        print(f"\n  --- {label} ---")
        r_full = backtest_kc(h4, f"{label}_FULL", return_trades=True, **kw)
        print(f"  FULL: N={r_full['n']}, Sharpe={r_full['sharpe']:.2f}, "
              f"PnL=${r_full['total_pnl']:.0f}, WR={r_full['win_rate']:.1f}%, MaxDD=${r_full['max_dd']:.0f}")
        kf = kc_kfold(h4, label, n_folds=6, **kw)
        if kf:
            h4_best[label] = {'full': r_full, 'kfold': kf}

    return h4_best


# ═══════════════════════════════════════════════════════════════
# Phase C: Correlation Analysis with L7
# ═══════════════════════════════════════════════════════════════

def run_phase_C(data, d1_best, h4_best):
    print("\n" + "=" * 80)
    print("Phase C: Correlation Analysis with L7")
    print("=" * 80)

    # Run L7 with MH=8 (best from R25)
    kw = copy.deepcopy(L7_KWARGS)
    kw['keltner_max_hold_m15'] = 8
    l7 = run_variant(data, "L7_MH8", **kw)
    l7_trades = l7['_trades']

    # Build L7 daily PnL
    l7_daily = {}
    for t in l7_trades:
        d = pd.Timestamp(t.exit_time).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += t.pnl

    print(f"\n  L7 (MH=8): {len(l7_trades)} trades, Sharpe={l7['sharpe']:.2f}, PnL=${l7['total_pnl']:.0f}")

    # Compare with each new strategy
    all_strats = {}
    all_strats.update(d1_best)
    all_strats.update(h4_best)

    print(f"\n  {'Strategy':<30s} {'Corr':>6} {'L7 Sharpe':>10} {'New Sharpe':>11} {'Combined':>10} {'Combo DD':>9}")
    print(f"  {'-'*76}")

    for name, info in all_strats.items():
        new_daily = info['full'].get('_daily_pnl', {})
        if not new_daily:
            continue

        # Align dates
        all_dates = sorted(set(l7_daily.keys()) | set(new_daily.keys()))
        l7_arr = np.array([l7_daily.get(d, 0) for d in all_dates])
        new_arr = np.array([new_daily.get(d, 0) for d in all_dates])

        # Correlation
        mask = (l7_arr != 0) | (new_arr != 0)
        if mask.sum() > 30:
            corr = np.corrcoef(l7_arr[mask], new_arr[mask])[0, 1]
        else:
            corr = np.nan

        # Combined portfolio
        combo = l7_arr + new_arr
        combo_sh = combo.mean() / combo.std() * np.sqrt(252) if combo.std() > 0 else 0
        combo_eq = np.cumsum(combo)
        combo_dd = (np.maximum.accumulate(combo_eq) - combo_eq).max()

        l7_sh_daily = l7_arr.mean() / l7_arr.std() * np.sqrt(252) if l7_arr.std() > 0 else 0
        new_sh_daily = new_arr.mean() / new_arr.std() * np.sqrt(252) if new_arr.std() > 0 else 0

        print(f"  {name:<30s} {corr:>6.3f} {l7_sh_daily:>10.2f} {new_sh_daily:>11.2f} {combo_sh:>10.2f} ${combo_dd:>8.0f}")

    # Detailed: best combo candidate
    best_name = None
    best_combo_sh = 0
    for name, info in all_strats.items():
        new_daily = info['full'].get('_daily_pnl', {})
        if not new_daily: continue
        all_dates = sorted(set(l7_daily.keys()) | set(new_daily.keys()))
        l7_arr = np.array([l7_daily.get(d, 0) for d in all_dates])
        new_arr = np.array([new_daily.get(d, 0) for d in all_dates])
        combo = l7_arr + new_arr
        sh = combo.mean() / combo.std() * np.sqrt(252) if combo.std() > 0 else 0
        if sh > best_combo_sh:
            best_combo_sh = sh; best_name = name

    if best_name:
        print(f"\n  >>> Best combo: L7_MH8 + {best_name}, Combined Sharpe={best_combo_sh:.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase D: Year-by-Year Combined Analysis
# ═══════════════════════════════════════════════════════════════

def run_phase_D(data, d1_best, h4_best):
    print("\n" + "=" * 80)
    print("Phase D: Year-by-Year Combined Portfolio")
    print("=" * 80)

    kw = copy.deepcopy(L7_KWARGS)
    kw['keltner_max_hold_m15'] = 8
    l7 = run_variant(data, "L7_MH8_D", verbose=False, **kw)
    l7_trades = l7['_trades']

    l7_daily = {}
    for t in l7_trades:
        d = pd.Timestamp(t.exit_time).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += t.pnl

    # Pick best daily and H4
    for strat_name in ["D1_EMA20_M2.0_ADX18", "D1_EMA10_M2.0_ADX22",
                        "H4_EMA20_M2.0_ADX18"]:
        if strat_name not in {**d1_best, **h4_best}:
            continue
        info = {**d1_best, **h4_best}[strat_name]
        new_daily = info['full'].get('_daily_pnl', {})
        if not new_daily:
            continue

        all_dates = sorted(set(l7_daily.keys()) | set(new_daily.keys()))

        print(f"\n  --- L7_MH8 + {strat_name} ---")
        print(f"  {'Year':>6} {'L7 PnL':>10} {'New PnL':>10} {'Combined':>10} {'L7 Sharpe':>10} {'New Sharpe':>11} {'Combo Sharpe':>13}")
        print(f"  {'-'*70}")

        years = sorted(set(d.year for d in all_dates))
        for y in years:
            y_dates = [d for d in all_dates if d.year == y]
            l7_y = np.array([l7_daily.get(d, 0) for d in y_dates])
            new_y = np.array([new_daily.get(d, 0) for d in y_dates])
            combo_y = l7_y + new_y

            def _sh(arr):
                return arr.mean() / arr.std() * np.sqrt(252) if len(arr) > 5 and arr.std() > 0 else 0

            print(f"  {y:>6} ${l7_y.sum():>9.0f} ${new_y.sum():>9.0f} ${combo_y.sum():>9.0f} "
                  f"{_sh(l7_y):>10.2f} {_sh(new_y):>11.2f} {_sh(combo_y):>13.2f}")

        # Totals
        l7_all = np.array([l7_daily.get(d, 0) for d in all_dates])
        new_all = np.array([new_daily.get(d, 0) for d in all_dates])
        combo_all = l7_all + new_all

        def _sh(arr):
            return arr.mean() / arr.std() * np.sqrt(252) if len(arr) > 5 and arr.std() > 0 else 0

        eq = np.cumsum(combo_all)
        dd = (np.maximum.accumulate(eq) - eq).max()
        print(f"  {'TOTAL':>6} ${l7_all.sum():>9.0f} ${new_all.sum():>9.0f} ${combo_all.sum():>9.0f} "
              f"{_sh(l7_all):>10.2f} {_sh(new_all):>11.2f} {_sh(combo_all):>13.2f}")
        print(f"  Combined MaxDD: ${dd:.0f}")


def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R26_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R26: Daily/H4 Keltner K-Fold + Correlation with L7")
    print(f"# Started: {ts}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    for phase_fn, name, args in [
        (run_phase_A, "A", (h1_df,)),
        (run_phase_B, "B", (h1_df,)),
    ]:
        try:
            result = phase_fn(*args)
            out.flush()
            print(f"\n# Phase {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
            if name == "A": d1_best = result
            elif name == "B": h4_best = result
        except Exception as e:
            print(f"\n# Phase {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            out.flush()
            if name == "A": d1_best = {}
            elif name == "B": h4_best = {}

    for phase_fn, name, args in [
        (run_phase_C, "C", (data, d1_best, h4_best)),
        (run_phase_D, "D", (data, d1_best, h4_best)),
    ]:
        try:
            phase_fn(*args)
            out.flush()
            print(f"\n# Phase {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
        except Exception as e:
            print(f"\n# Phase {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            out.flush()

    elapsed = time.time() - t0
    print(f"\n# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Elapsed: {elapsed/60:.1f} minutes")

    sys.stdout = old_stdout
    out.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
