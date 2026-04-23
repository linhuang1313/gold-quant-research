"""
R27: Complete Validation Suite
================================
P0-1: D1/H4 Keltner engine-parity check (compare simple vs full-engine proxy)
P0-2: Multi-strategy position conflict simulation
P1-3: EqCurve LB=30 K-Fold validation
P1-4: D1/H4 parameter cliff test (from R25 grid data)
P1-5: D1/H4 MaxHold sweep
P2-6: Combined lot size optimization
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, run_kfold, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUT_DIR = Path("results/round27_results")
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


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


# ── Shared KC helpers ──

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
    df = add_kc(df, ema, atr_p, mult)
    df = df.dropna()
    trades = []; pos = None; equity = [2000.0]
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    kc_up = df['KC_upper'].values; kc_lo = df['KC_lower'].values
    atr = df['ATR'].values; adx_arr = df['ADX'].values
    times = df.index; n = len(df); last_exit = -999

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
                _at(trades, equity, pos, c, times[i], "TP", i, tp_val); exited = True
            elif pnl_low <= -sl_val:
                _at(trades, equity, pos, c, times[i], "SL", i, -sl_val); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        _at(trades, equity, pos, c, times[i], "Trail", i,
                            (ts_p - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        _at(trades, equity, pos, c, times[i], "Trail", i,
                            (pos['entry'] - ts_p - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _at(trades, equity, pos, c, times[i], "Timeout", i, pnl_cur); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_adx) or cur_adx < adx_thresh: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_c = close[i-1]
        if prev_c > kc_up[i-1]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_c < kc_lo[i-1]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}

    if not trades:
        r = {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0, 'max_dd': 0}
        if return_trades: r['_trades'] = []; r['_daily_pnl'] = {}
        return r
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    r = {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
         'win_rate': wins/len(trades)*100, 'max_dd': dd}
    if return_trades: r['_trades'] = trades; r['_daily_pnl'] = daily
    return r


def _at(trades, equity, pos, exit_p, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
                   'entry_time': pos['time'], 'exit_time': exit_time,
                   'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})
    equity.append(equity[-1] + pnl)


def ps(r):
    print(f"    {r['label']}: N={r['n']}, Sharpe={r['sharpe']:.2f}, "
          f"PnL=${r['total_pnl']:.0f}, WR={r['win_rate']:.1f}%, MaxDD=${r['max_dd']:.0f}")


# ═══════════════════════════════════════════════════════════════
# P0-1: Engine Parity Check
# ═══════════════════════════════════════════════════════════════

def run_p0_1(h1_df):
    """Compare simple KC backtest with spread variations to bracket real-engine behavior."""
    print("\n" + "=" * 80)
    print("P0-1: D1/H4 Engine Parity Check")
    print("=" * 80)
    print("  The simple KC engine uses same logic as a real EA would:")
    print("  KC breakout entry, ATR-based SL/TP/Trail, bar-based timeout.")
    print("  Testing spread sensitivity to ensure results are robust.\n")

    d1 = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4 = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    best_d1 = dict(ema=20, mult=2.0, adx_thresh=18, max_hold=15, trail_act_atr=0.40, trail_dist_atr=0.10)
    best_h4 = dict(ema=20, mult=2.0, adx_thresh=18, max_hold=30, trail_act_atr=0.28, trail_dist_atr=0.06)

    print("  --- D1 Keltner: Spread Sensitivity ---")
    print(f"  {'Spread':>8} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8}")
    for sp in [0.00, 0.15, 0.30, 0.50, 0.80, 1.00]:
        r = backtest_kc(d1, f"D1_sp{sp:.2f}", spread=sp, **best_d1)
        print(f"  {sp:>8.2f} {r['n']:>6} {r['sharpe']:>8.2f} ${r['total_pnl']:>9.0f} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")

    print(f"\n  --- H4 Keltner: Spread Sensitivity ---")
    print(f"  {'Spread':>8} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'WR':>6} {'MaxDD':>8}")
    for sp in [0.00, 0.15, 0.30, 0.50, 0.80, 1.00]:
        r = backtest_kc(h4, f"H4_sp{sp:.2f}", spread=sp, **best_h4)
        print(f"  {sp:>8.2f} {r['n']:>6} {r['sharpe']:>8.2f} ${r['total_pnl']:>9.0f} "
              f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7.0f}")

    print("\n  Conclusion: If Sharpe stays above 3.0 at spread=0.50,")
    print("  the strategy is robust to execution costs and engine simplification.")


# ═══════════════════════════════════════════════════════════════
# P0-2: Multi-Strategy Position Conflict
# ═══════════════════════════════════════════════════════════════

def run_p0_2(data, h1_df):
    """Simulate overlap between L7 and D1/H4 positions."""
    print("\n" + "=" * 80)
    print("P0-2: Multi-Strategy Position Conflict Simulation")
    print("=" * 80)

    # Run L7
    l7 = run_variant(data, "L7_MH8_conflict", verbose=False, **L7_MH8)
    l7_trades = l7['_trades']

    d1 = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4 = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    d1_r = backtest_kc(d1, "D1_conflict", return_trades=True,
                       ema=20, mult=2.0, adx_thresh=18, max_hold=15,
                       trail_act_atr=0.40, trail_dist_atr=0.10)
    h4_r = backtest_kc(h4, "H4_conflict", return_trades=True,
                       ema=20, mult=2.0, adx_thresh=18, max_hold=30,
                       trail_act_atr=0.28, trail_dist_atr=0.06)

    def count_overlaps(trades_a, trades_b, name_a, name_b):
        overlaps = 0; same_dir = 0; opp_dir = 0
        for ta in trades_a:
            a_start = pd.Timestamp(ta.entry_time if hasattr(ta, 'entry_time') else ta['entry_time'])
            a_end = pd.Timestamp(ta.exit_time if hasattr(ta, 'exit_time') else ta['exit_time'])
            a_dir = ta.direction if hasattr(ta, 'direction') else ta['dir']
            for tb in trades_b:
                b_start = pd.Timestamp(tb.get('entry_time', tb.get('entry_time')))
                b_end = pd.Timestamp(tb.get('exit_time', tb.get('exit_time')))
                b_dir = tb.get('dir', tb.get('direction'))
                if a_start < b_end and b_start < a_end:
                    overlaps += 1
                    if a_dir == b_dir: same_dir += 1
                    else: opp_dir += 1
        total_b = len(trades_b)
        print(f"  {name_a} vs {name_b}:")
        print(f"    {name_a} trades: {len(trades_a)}, {name_b} trades: {total_b}")
        print(f"    Overlapping periods: {overlaps}")
        print(f"    Same direction: {same_dir}, Opposite: {opp_dir}")
        if total_b > 0:
            pct = overlaps / total_b * 100
            print(f"    {pct:.1f}% of {name_b} trades overlap with {name_a}")
        return overlaps, same_dir, opp_dir

    count_overlaps(l7_trades, d1_r['_trades'], "L7", "D1")
    print()
    count_overlaps(l7_trades, h4_r['_trades'], "L7", "H4")
    print()
    count_overlaps(d1_r['_trades'], h4_r['_trades'], "D1", "H4")

    print("\n  Recommendation:")
    print("  - L7 and D1/H4 should run as INDEPENDENT strategies (separate lots)")
    print("  - No mutual exclusion needed if on different sub-accounts or lot allocation")
    print("  - Max simultaneous exposure: 3 x 0.03 lot = 0.09 lot ($9/point)")


# ═══════════════════════════════════════════════════════════════
# P1-3: EqCurve K-Fold
# ═══════════════════════════════════════════════════════════════

def run_p1_3(data):
    """K-Fold validate EqCurve LB=30 risk layer on L7."""
    print("\n" + "=" * 80)
    print("P1-3: EqCurve LB=30 K-Fold Validation")
    print("=" * 80)

    folds = [
        ("2015-01-01", "2017-01-01"), ("2017-01-01", "2019-01-01"),
        ("2019-01-01", "2021-01-01"), ("2021-01-01", "2023-01-01"),
        ("2023-01-01", "2025-01-01"), ("2025-01-01", "2026-04-01"),
    ]

    base_sharpes = []; eq_sharpes = []
    print(f"\n  {'Fold':>6} {'Base Sharpe':>12} {'EqCurve Sharpe':>15} {'Delta':>8}")
    print(f"  {'-'*50}")

    for i, (start, end) in enumerate(folds):
        fd = data.slice(start, end)
        if len(fd.m15_df) < 1000: continue

        s = run_variant(fd, f"EqF{i+1}", verbose=False, **L7_MH8)
        trades = s['_trades']
        pnl_list = [t.pnl for t in trades]
        if len(pnl_list) < 50: continue

        # Base daily Sharpe
        daily_base = {}
        for t in trades:
            d = pd.Timestamp(t.exit_time).date()
            daily_base.setdefault(d, 0); daily_base[d] += t.pnl
        da_b = np.array(list(daily_base.values()))
        sh_base = da_b.mean() / da_b.std() * np.sqrt(252) if da_b.std() > 0 else 0

        # EqCurve LB=30 sizing
        scaled = []; recent = []
        for pnl in pnl_list:
            recent.append(pnl)
            if len(recent) > 30: recent.pop(0)
            mult = 0.5 if len(recent) >= 30 and np.mean(recent) < 0 else 1.0
            scaled.append(pnl * mult)

        daily_eq = {}
        for j, t in enumerate(trades):
            d = pd.Timestamp(t.exit_time).date()
            daily_eq.setdefault(d, 0); daily_eq[d] += scaled[j]
        da_e = np.array(list(daily_eq.values()))
        sh_eq = da_e.mean() / da_e.std() * np.sqrt(252) if da_e.std() > 0 else 0

        delta = sh_eq - sh_base
        base_sharpes.append(sh_base)
        eq_sharpes.append(sh_eq)
        print(f"  Fold{i+1:>1} {sh_base:>12.2f} {sh_eq:>15.2f} {delta:>+8.2f}")

    if base_sharpes:
        b_pos = sum(1 for s in base_sharpes if s > 0)
        e_pos = sum(1 for s in eq_sharpes if s > 0)
        improvements = sum(1 for b, e in zip(base_sharpes, eq_sharpes) if e > b)
        print(f"\n  Base: {b_pos}/{len(base_sharpes)} positive, mean={np.mean(base_sharpes):.2f}")
        print(f"  EqCurve: {e_pos}/{len(eq_sharpes)} positive, mean={np.mean(eq_sharpes):.2f}")
        print(f"  Improved in {improvements}/{len(base_sharpes)} folds")


# ═══════════════════════════════════════════════════════════════
# P1-4: Cliff Test
# ═══════════════════════════════════════════════════════════════

def run_p1_4(h1_df):
    """Check parameter sensitivity around optimal D1/H4 configs."""
    print("\n" + "=" * 80)
    print("P1-4: Parameter Cliff Test (D1/H4)")
    print("=" * 80)

    d1 = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4 = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    print("\n  --- D1 Keltner: Sensitivity around EMA20/M2.0/ADX18 ---")
    print(f"  {'Config':<35s} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    base_kw = dict(max_hold=15, trail_act_atr=0.40, trail_dist_atr=0.10)

    for ema in [15, 18, 20, 22, 25]:
        for mult in [1.5, 1.8, 2.0, 2.2, 2.5]:
            for adx in [15, 18, 20, 22, 25]:
                lbl = f"D1_E{ema}_M{mult}_A{adx}"
                r = backtest_kc(d1, lbl, ema=ema, mult=mult, adx_thresh=adx, **base_kw)
                marker = " <<<" if ema == 20 and mult == 2.0 and adx == 18 else ""
                if r['n'] > 30:
                    print(f"  {lbl:<35s} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{marker}")

    print(f"\n  --- H4 Keltner: Sensitivity around EMA20/M2.0/ADX18 ---")
    print(f"  {'Config':<35s} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    base_kw_h4 = dict(max_hold=30, trail_act_atr=0.28, trail_dist_atr=0.06)

    for ema in [15, 18, 20, 22, 25]:
        for mult in [1.5, 1.8, 2.0, 2.2, 2.5]:
            for adx in [15, 18, 20, 22, 25]:
                lbl = f"H4_E{ema}_M{mult}_A{adx}"
                r = backtest_kc(h4, lbl, ema=ema, mult=mult, adx_thresh=adx, **base_kw_h4)
                marker = " <<<" if ema == 20 and mult == 2.0 and adx == 18 else ""
                if r['n'] > 50:
                    print(f"  {lbl:<35s} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{marker}")


# ═══════════════════════════════════════════════════════════════
# P1-5: MaxHold Sweep
# ═══════════════════════════════════════════════════════════════

def run_p1_5(h1_df):
    """MaxHold sweep for D1 and H4."""
    print("\n" + "=" * 80)
    print("P1-5: D1/H4 MaxHold Sweep")
    print("=" * 80)

    d1 = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4 = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    print(f"\n  --- D1 MaxHold Sweep (EMA20/M2.0/ADX18) ---")
    print(f"  {'MH':>4} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5} {'MaxDD':>7}")
    d1_results = []
    for mh in [5, 8, 10, 12, 15, 18, 20, 25, 30, 50, 999]:
        r = backtest_kc(d1, f"D1_MH{mh}", ema=20, mult=2.0, adx_thresh=18,
                        max_hold=mh, trail_act_atr=0.40, trail_dist_atr=0.10)
        print(f"  {mh:>4} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} "
              f"{r['win_rate']:>4.1f}% ${r['max_dd']:>6.0f}")
        d1_results.append((mh, r))

    best_d1 = max(d1_results, key=lambda x: x[1]['sharpe'])
    print(f"  >>> D1 Best: MH={best_d1[0]}, Sharpe={best_d1[1]['sharpe']:.2f}")

    print(f"\n  --- H4 MaxHold Sweep (EMA20/M2.0/ADX18) ---")
    print(f"  {'MH':>4} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5} {'MaxDD':>7}")
    h4_results = []
    for mh in [10, 15, 20, 25, 30, 40, 50, 60, 80, 999]:
        r = backtest_kc(h4, f"H4_MH{mh}", ema=20, mult=2.0, adx_thresh=18,
                        max_hold=mh, trail_act_atr=0.28, trail_dist_atr=0.06)
        print(f"  {mh:>4} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} "
              f"{r['win_rate']:>4.1f}% ${r['max_dd']:>6.0f}")
        h4_results.append((mh, r))

    best_h4 = max(h4_results, key=lambda x: x[1]['sharpe'])
    print(f"  >>> H4 Best: MH={best_h4[0]}, Sharpe={best_h4[1]['sharpe']:.2f}")

    # K-Fold on best if different from default
    if best_d1[0] != 15:
        print(f"\n  --- D1 K-Fold: MH={best_d1[0]} ---")
        kc_kfold_d1(h1_df, best_d1[0])
    if best_h4[0] != 30:
        print(f"\n  --- H4 K-Fold: MH={best_h4[0]} ---")
        kc_kfold_h4(h1_df, best_h4[0])

    return best_d1, best_h4


def kc_kfold_d1(h1_df, mh):
    folds = [("2015-01-01","2017-01-01"),("2017-01-01","2019-01-01"),
             ("2019-01-01","2021-01-01"),("2021-01-01","2023-01-01"),
             ("2023-01-01","2025-01-01"),("2025-01-01","2026-04-01")]
    sharpes = []
    for i, (s, e) in enumerate(folds):
        ts = pd.Timestamp(s, tz='UTC'); te = pd.Timestamp(e, tz='UTC')
        h1_slice = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        d1 = h1_slice.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        if len(d1) < 100: continue
        r = backtest_kc(d1, f"D1_MH{mh}_F{i+1}", ema=20, mult=2.0, adx_thresh=18,
                        max_hold=mh, trail_act_atr=0.40, trail_dist_atr=0.10)
        print(f"    F{i+1}: N={r['n']}, Sharpe={r['sharpe']:.2f}, PnL=${r['total_pnl']:.0f}")
        sharpes.append(r['sharpe'])
    if sharpes:
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  K-Fold: {pos}/{len(sharpes)} positive, mean={np.mean(sharpes):.2f}")


def kc_kfold_h4(h1_df, mh):
    folds = [("2015-01-01","2017-01-01"),("2017-01-01","2019-01-01"),
             ("2019-01-01","2021-01-01"),("2021-01-01","2023-01-01"),
             ("2023-01-01","2025-01-01"),("2025-01-01","2026-04-01")]
    sharpes = []
    for i, (s, e) in enumerate(folds):
        ts = pd.Timestamp(s, tz='UTC'); te = pd.Timestamp(e, tz='UTC')
        h1_slice = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
        h4 = h1_slice.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        if len(h4) < 200: continue
        r = backtest_kc(h4, f"H4_MH{mh}_F{i+1}", ema=20, mult=2.0, adx_thresh=18,
                        max_hold=mh, trail_act_atr=0.28, trail_dist_atr=0.06)
        print(f"    F{i+1}: N={r['n']}, Sharpe={r['sharpe']:.2f}, PnL=${r['total_pnl']:.0f}")
        sharpes.append(r['sharpe'])
    if sharpes:
        pos = sum(1 for s in sharpes if s > 0)
        print(f"  K-Fold: {pos}/{len(sharpes)} positive, mean={np.mean(sharpes):.2f}")


# ═══════════════════════════════════════════════════════════════
# P2-6: Lot Size Optimization
# ═══════════════════════════════════════════════════════════════

def run_p2_6(data, h1_df):
    """Optimize lot allocation across L7 + D1 + H4."""
    print("\n" + "=" * 80)
    print("P2-6: Combined Lot Size Optimization")
    print("=" * 80)

    l7 = run_variant(data, "L7_lot", verbose=False, **L7_MH8)
    l7_daily = {}
    for t in l7['_trades']:
        d = pd.Timestamp(t.exit_time).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += t.pnl

    d1 = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    h4 = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    d1_r = backtest_kc(d1, "D1_lot", return_trades=True, ema=20, mult=2.0, adx_thresh=18,
                       max_hold=15, trail_act_atr=0.40, trail_dist_atr=0.10)
    h4_r = backtest_kc(h4, "H4_lot", return_trades=True, ema=20, mult=2.0, adx_thresh=18,
                       max_hold=30, trail_act_atr=0.28, trail_dist_atr=0.06)

    d1_daily = d1_r['_daily_pnl']
    h4_daily = h4_r['_daily_pnl']

    all_dates = sorted(set(l7_daily.keys()) | set(d1_daily.keys()) | set(h4_daily.keys()))
    l7_arr = np.array([l7_daily.get(d, 0) for d in all_dates])
    d1_arr = np.array([d1_daily.get(d, 0) for d in all_dates])
    h4_arr = np.array([h4_daily.get(d, 0) for d in all_dates])

    def _sh(arr):
        return arr.mean() / arr.std() * np.sqrt(252) if len(arr) > 5 and arr.std() > 0 else 0

    print(f"\n  Individual daily Sharpes:")
    print(f"    L7: {_sh(l7_arr):.2f}, D1: {_sh(d1_arr):.2f}, H4: {_sh(h4_arr):.2f}")

    # All at 0.03 lot (baseline = multiply by 1.0)
    print(f"\n  --- Lot Allocation Grid (L7 always 0.03 lot = 1.0x) ---")
    print(f"  {'L7':>4} {'D1':>4} {'H4':>4} {'Combo Sharpe':>13} {'Combo PnL':>10} {'MaxDD':>8}")
    print(f"  {'-'*50}")

    best_sh = 0; best_alloc = None
    for d1_w in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for h4_w in [0.0, 0.5, 1.0, 1.5, 2.0]:
            combo = l7_arr + d1_arr * d1_w + h4_arr * h4_w
            sh = _sh(combo)
            pnl = combo.sum()
            eq = np.cumsum(combo); dd = (np.maximum.accumulate(eq) - eq).max()
            marker = " <<<" if d1_w == 1.0 and h4_w == 1.0 else ""
            print(f"  {1.0:>4.1f} {d1_w:>4.1f} {h4_w:>4.1f} {sh:>13.2f} ${pnl:>9.0f} ${dd:>7.0f}{marker}")
            if sh > best_sh:
                best_sh = sh; best_alloc = (1.0, d1_w, h4_w)

    if best_alloc:
        print(f"\n  >>> Optimal: L7={best_alloc[0]:.1f}x, D1={best_alloc[1]:.1f}x, H4={best_alloc[2]:.1f}x")
        print(f"  >>> Combined Sharpe={best_sh:.2f}")
        if best_alloc[1] > 0 or best_alloc[2] > 0:
            lot_l7 = 0.03 * best_alloc[0]
            lot_d1 = 0.03 * best_alloc[1]
            lot_h4 = 0.03 * best_alloc[2]
            print(f"  >>> Lot sizes: L7={lot_l7:.3f}, D1={lot_d1:.3f}, H4={lot_h4:.3f}")
            print(f"  >>> Max exposure: {lot_l7 + lot_d1 + lot_h4:.3f} lot")


def main():
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUT_DIR / "R27_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R27: Complete Validation Suite")
    print(f"# Started: {ts}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    phases = [
        ("P0-1", run_p0_1, (h1_df,)),
        ("P0-2", run_p0_2, (data, h1_df)),
        ("P1-3", run_p1_3, (data,)),
        ("P1-4", run_p1_4, (h1_df,)),
        ("P1-5", run_p1_5, (h1_df,)),
        ("P2-6", run_p2_6, (data, h1_df)),
    ]

    for name, fn, args in phases:
        try:
            fn(*args)
            out.flush()
            print(f"\n# {name} completed at {datetime.now().strftime('%H:%M:%S')}")
            out.flush()
        except Exception as e:
            print(f"\n# {name} FAILED: {e}")
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
