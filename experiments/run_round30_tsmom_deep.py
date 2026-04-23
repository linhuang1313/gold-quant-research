"""
R30: TSMOM Deep Validation
============================
A: Correlation analysis — TSMOM vs L7, D1, H4 daily PnL
B: Parameter cliff test — lookback/SL/TP/MH/trail sensitivity
C: Optimal-param K-Fold — SL3.5/TP12/MH50 on 6 folds
D: Multi-strategy portfolio optimization — L7+TSMOM+D1+H4
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUT_DIR = Path("results/round30_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data)
            except: pass
            f.flush()
    def flush(self):
        for f in self.files: f.flush()


def get_l7_mh8():
    kw = {**LIVE_PARITY_KWARGS}
    kw['regime_config'] = {
        'low':    {'trail_act': 0.30, 'trail_dist': 0.06},
        'normal': {'trail_act': 0.20, 'trail_dist': 0.04},
        'high':   {'trail_act': 0.08, 'trail_dist': 0.01},
    }
    kw['time_adaptive_trail'] = True
    kw['time_adaptive_trail_start'] = 2
    kw['time_adaptive_trail_decay'] = 0.75
    kw['time_adaptive_trail_floor'] = 0.003
    kw['min_entry_gap_hours'] = 1.0
    kw['keltner_max_hold_m15'] = 8
    return kw


# ── Shared KC/TSMOM helpers ──

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


def _at(trades, equity, pos, close, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'],
                   'entry_time': pos['time'], 'exit_time': exit_time,
                   'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})
    equity.append(equity[-1] + pnl)


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


def _run_tsmom(df, label, weights, sl_atr=3.5, tp_atr=12.0, trail_act=0.28,
               trail_dist=0.06, max_hold=50, spread=0.30, lot=0.03):
    """Multi-period weighted TSMOM."""
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values
    times = df.index; n = len(close)
    max_lb = max(lb for lb, _ in weights)
    trades = []; pos = None; last_exit = -999; equity = [2000.0]

    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0
        for lb, w in weights:
            if i >= lb: s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s

    for i in range(max_lb + 1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i] - pos['entry'] - spread) * lot * 100
                pnl_l = (low[i] - pos['entry'] - spread) * lot * 100
                pnl_c = (close[i] - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low[i] - spread) * lot * 100
                pnl_l = (pos['entry'] - high[i] - spread) * lot * 100
                pnl_c = (pos['entry'] - close[i] - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _at(trades, equity, pos, close[i], times[i], "TP", i, tp_val); exited = True
            elif pnl_l <= -sl_val:
                _at(trades, equity, pos, close[i], times[i], "SL", i, -sl_val); exited = True
            else:
                ad = trail_act * pos['atr']; td = trail_dist * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        _at(trades, equity, pos, close[i], times[i], "Trail", i,
                            (ts - pos['entry'] - spread) * lot * 100); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        _at(trades, equity, pos, close[i], times[i], "Trail", i,
                            (pos['entry'] - ts - spread) * lot * 100); exited = True
                if not exited and held >= max_hold:
                    _at(trades, equity, pos, close[i], times[i], "Timeout", i, pnl_c); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    _at(trades, equity, pos, close[i], times[i], "Reversal", i, pnl_c); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    _at(trades, equity, pos, close[i], times[i], "Reversal", i, pnl_c); exited = True
            if exited: pos = None; last_exit = i; continue

        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(atr[i]) or atr[i] < 0.1: continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': close[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': close[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}

    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0, 'max_dd': 0,
                '_trades': [], '_daily_pnl': {}}
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
            'win_rate': wins/len(trades)*100, 'max_dd': dd, '_trades': trades, '_daily_pnl': daily}


def make_daily(trades):
    """Convert trade list to daily PnL series."""
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    return pd.Series(daily).sort_index()


TSMOM_WEIGHTS = [(480, 0.5), (1440, 0.5)]  # 20d + 60d


# ═══════════════════════════════════════════════════════════════
# Phase A: Correlation Analysis
# ═══════════════════════════════════════════════════════════════

def run_phase_A(data, h1_df):
    """Cross-strategy daily PnL correlation."""
    print("\n" + "=" * 80)
    print("Phase A: Cross-Strategy Correlation Analysis")
    print("=" * 80)

    # L7(MH=8)
    l7 = run_variant(data, "L7MH8_corr", verbose=False, **get_l7_mh8())
    l7_daily = make_daily(l7['_trades'])
    print(f"  L7(MH=8): {len(l7['_trades'])} trades, Sharpe={l7['sharpe']:.2f}")

    # D1 Keltner
    d1 = h1_df.resample('D').agg({'Open':'first','High':'max','Low':'min',
                                   'Close':'last','Volume':'sum'}).dropna()
    d1_r = backtest_kc(d1, "D1", ema=20, mult=2.0, adx_thresh=18,
                       trail_act_atr=0.40, trail_dist_atr=0.10, max_hold=8,
                       return_trades=True)
    d1_daily = make_daily(d1_r['_trades'])
    print(f"  D1 KC:    {d1_r['n']} trades, Sharpe={d1_r['sharpe']:.2f}")

    # H4 Keltner
    h4 = h1_df.resample('4h').agg({'Open':'first','High':'max','Low':'min',
                                    'Close':'last','Volume':'sum'}).dropna()
    h4_r = backtest_kc(h4, "H4", ema=20, mult=2.0, adx_thresh=18,
                       trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=20,
                       return_trades=True)
    h4_daily = make_daily(h4_r['_trades'])
    print(f"  H4 KC:    {h4_r['n']} trades, Sharpe={h4_r['sharpe']:.2f}")

    # TSMOM (20d+60d, optimal params from R29)
    h1c = h1_df.copy()
    tr = pd.DataFrame({
        'hl': h1c['High'] - h1c['Low'],
        'hc': (h1c['High'] - h1c['Close'].shift(1)).abs(),
        'lc': (h1c['Low'] - h1c['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1c['ATR'] = tr.rolling(14).mean()
    tsmom_r = _run_tsmom(h1c, "TSMOM", TSMOM_WEIGHTS, sl_atr=3.5, tp_atr=12.0,
                         trail_act=0.28, trail_dist=0.06, max_hold=50)
    tsmom_daily = make_daily(tsmom_r['_trades'])
    print(f"  TSMOM:    {tsmom_r['n']} trades, Sharpe={tsmom_r['sharpe']:.2f}")

    # Correlation matrix
    all_dates = sorted(set(l7_daily.index) | set(d1_daily.index) |
                       set(h4_daily.index) | set(tsmom_daily.index))
    combined = pd.DataFrame({
        'L7': l7_daily.reindex(all_dates, fill_value=0),
        'D1': d1_daily.reindex(all_dates, fill_value=0),
        'H4': h4_daily.reindex(all_dates, fill_value=0),
        'TSMOM': tsmom_daily.reindex(all_dates, fill_value=0),
    })

    print(f"\n  --- A1: Daily PnL Correlation Matrix ---")
    print(f"  {'':>8}  {'L7':>8}  {'D1':>8}  {'H4':>8}  {'TSMOM':>8}")
    corr = combined.corr()
    for s in ['L7', 'D1', 'H4', 'TSMOM']:
        row = "  " + f"{s:>8}"
        for t in ['L7', 'D1', 'H4', 'TSMOM']:
            row += f"  {corr.loc[s, t]:>8.3f}"
        print(row)

    # Rolling 60-day correlation
    print(f"\n  --- A2: Rolling 60-Day Correlation (TSMOM vs others) ---")
    for other in ['L7', 'D1', 'H4']:
        roll_corr = combined['TSMOM'].rolling(60).corr(combined[other]).dropna()
        if len(roll_corr) < 60: continue
        print(f"  TSMOM-{other}: mean={roll_corr.mean():.3f}, "
              f"std={roll_corr.std():.3f}, "
              f"min={roll_corr.min():.3f}, max={roll_corr.max():.3f}, "
              f"pct>0.3={100 * (roll_corr > 0.3).mean():.0f}%")

    # Yearly breakdown
    print(f"\n  --- A3: Yearly Correlation (TSMOM vs L7) ---")
    combined['year'] = pd.to_datetime(combined.index).year
    for yr in sorted(combined['year'].unique()):
        sub = combined[combined['year'] == yr]
        if len(sub) < 30: continue
        c = sub['TSMOM'].corr(sub['L7'])
        print(f"  {yr}: corr={c:+.3f}, N={len(sub)}")

    return combined


# ═══════════════════════════════════════════════════════════════
# Phase B: TSMOM Parameter Cliff Test
# ═══════════════════════════════════════════════════════════════

def run_phase_B(h1_df):
    """Test TSMOM parameter stability across all dimensions."""
    print("\n" + "=" * 80)
    print("Phase B: TSMOM Parameter Cliff Test")
    print("=" * 80)

    h1c = h1_df.copy()
    tr = pd.DataFrame({
        'hl': h1c['High'] - h1c['Low'],
        'hc': (h1c['High'] - h1c['Close'].shift(1)).abs(),
        'lc': (h1c['Low'] - h1c['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1c['ATR'] = tr.rolling(14).mean()

    # B1: Lookback period sensitivity
    print(f"\n  --- B1: Lookback Period Sensitivity ---")
    print(f"  {'Fast':>6} {'Slow':>6} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    base_sh = None
    for fast in [240, 360, 480, 600]:
        for slow in [960, 1200, 1440, 1680, 1920]:
            if fast >= slow: continue
            w = [(fast, 0.5), (slow, 0.5)]
            r = _run_tsmom(h1c, f"LB_{fast}_{slow}", w, sl_atr=3.5, tp_atr=12.0,
                           trail_act=0.28, trail_dist=0.06, max_hold=50)
            tag = " ***" if fast == 480 and slow == 1440 else ""
            print(f"  {fast:>6} {slow:>6} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{tag}")
            if fast == 480 and slow == 1440: base_sh = r['sharpe']
    if base_sh:
        print(f"\n  Base (480/1440): Sharpe={base_sh:.2f}")

    # B2: SL sensitivity
    print(f"\n  --- B2: SL Multiplier Sensitivity ---")
    print(f"  {'SL':>6} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for sl in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        r = _run_tsmom(h1c, f"SL{sl}", TSMOM_WEIGHTS, sl_atr=sl, tp_atr=12.0,
                       trail_act=0.28, trail_dist=0.06, max_hold=50)
        tag = " ***" if sl == 3.5 else ""
        print(f"  {sl:>6.1f} {r['n']:>5} {r['sharpe']:>7.2f} "
              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{tag}")

    # B3: TP sensitivity
    print(f"\n  --- B3: TP Multiplier Sensitivity ---")
    print(f"  {'TP':>6} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for tp in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 999.0]:
        r = _run_tsmom(h1c, f"TP{tp}", TSMOM_WEIGHTS, sl_atr=3.5, tp_atr=tp,
                       trail_act=0.28, trail_dist=0.06, max_hold=50)
        tag = " ***" if tp == 12.0 else ""
        print(f"  {tp:>6.1f} {r['n']:>5} {r['sharpe']:>7.2f} "
              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{tag}")

    # B4: MaxHold sensitivity
    print(f"\n  --- B4: MaxHold Sensitivity ---")
    print(f"  {'MH':>6} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for mh in [10, 15, 20, 30, 40, 50, 60, 80, 100, 999]:
        r = _run_tsmom(h1c, f"MH{mh}", TSMOM_WEIGHTS, sl_atr=3.5, tp_atr=12.0,
                       trail_act=0.28, trail_dist=0.06, max_hold=mh)
        tag = " ***" if mh == 50 else ""
        print(f"  {mh:>6} {r['n']:>5} {r['sharpe']:>7.2f} "
              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{tag}")

    # B5: Trail sensitivity
    print(f"\n  --- B5: Trail Parameters Sensitivity ---")
    print(f"  {'Act':>6} {'Dist':>6} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for act in [0.12, 0.20, 0.28, 0.40, 0.60]:
        for dist in [0.02, 0.04, 0.06, 0.10, 0.15]:
            r = _run_tsmom(h1c, f"T_{act}_{dist}", TSMOM_WEIGHTS, sl_atr=3.5, tp_atr=12.0,
                           trail_act=act, trail_dist=dist, max_hold=50)
            tag = " ***" if act == 0.28 and dist == 0.06 else ""
            print(f"  {act:>6.2f} {dist:>6.2f} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%{tag}")

    # B6: Spread sensitivity
    print(f"\n  --- B6: Spread Sensitivity ---")
    print(f"  {'Spread':>8} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    for sp in [0.0, 0.15, 0.30, 0.50, 0.80, 1.00]:
        r = _run_tsmom(h1c, f"Sp{sp}", TSMOM_WEIGHTS, sl_atr=3.5, tp_atr=12.0,
                       trail_act=0.28, trail_dist=0.06, max_hold=50, spread=sp)
        print(f"  ${sp:>7.2f} {r['n']:>5} {r['sharpe']:>7.2f} "
              f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")


# ═══════════════════════════════════════════════════════════════
# Phase C: TSMOM Optimal Param K-Fold
# ═══════════════════════════════════════════════════════════════

def run_phase_C(h1_df):
    """K-Fold for TSMOM with SL3.5/TP12/MH50 (optimal from R29)."""
    print("\n" + "=" * 80)
    print("Phase C: TSMOM Optimal-Param K-Fold Validation")
    print("=" * 80)

    folds = [("2015-01-01","2017-01-01"),("2017-01-01","2019-01-01"),
             ("2019-01-01","2021-01-01"),("2021-01-01","2023-01-01"),
             ("2023-01-01","2025-01-01"),("2025-01-01","2026-04-01")]

    # Two configs to compare
    configs = [
        ("TSMOM_base", {"sl_atr": 3.5, "tp_atr": 8.0, "trail_act": 0.28,
                        "trail_dist": 0.06, "max_hold": 30}),
        ("TSMOM_opt",  {"sl_atr": 3.5, "tp_atr": 12.0, "trail_act": 0.28,
                        "trail_dist": 0.06, "max_hold": 50}),
    ]

    for cfg_name, cfg in configs:
        print(f"\n  --- {cfg_name}: SL={cfg['sl_atr']}, TP={cfg['tp_atr']}, MH={cfg['max_hold']} ---")
        sharpes = []; pnls = []
        print(f"  {'Fold':>6} {'Period':>25} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
        for i, (s, e) in enumerate(folds):
            ts = pd.Timestamp(s, tz='UTC'); te = pd.Timestamp(e, tz='UTC')
            h1s = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
            if len(h1s) < 500: continue
            h1s = h1s.copy()
            tr = pd.DataFrame({
                'hl': h1s['High'] - h1s['Low'],
                'hc': (h1s['High'] - h1s['Close'].shift(1)).abs(),
                'lc': (h1s['Low'] - h1s['Close'].shift(1)).abs(),
            }).max(axis=1)
            h1s['ATR'] = tr.rolling(14).mean()
            r = _run_tsmom(h1s, f"{cfg_name}_F{i+1}", TSMOM_WEIGHTS, **cfg)
            print(f"  F{i+1:>4} {s} - {e} {r['n']:>5} {r['sharpe']:>7.2f} "
                  f"${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")
            sharpes.append(r['sharpe']); pnls.append(r['total_pnl'])
        if sharpes:
            pos = sum(1 for s in sharpes if s > 0)
            print(f"\n  K-Fold: {pos}/{len(sharpes)} positive")
            print(f"  Mean Sharpe: {np.mean(sharpes):.2f}, Std: {np.std(sharpes):.2f}")
            print(f"  Min: {min(sharpes):.2f}, Max: {max(sharpes):.2f}")
            print(f"  Total PnL: ${sum(pnls):.0f}")
            if pos == len(sharpes):
                print(f"  >>> PASS")
            else:
                print(f"  >>> FAIL ({len(sharpes)-pos} negative folds)")


# ═══════════════════════════════════════════════════════════════
# Phase D: Multi-Strategy Portfolio Optimization
# ═══════════════════════════════════════════════════════════════

def run_phase_D(combined_df):
    """Optimize lot weights for L7 + TSMOM + D1 + H4."""
    print("\n" + "=" * 80)
    print("Phase D: Multi-Strategy Portfolio Optimization")
    print("=" * 80)

    if combined_df is None:
        print("  No data from Phase A!"); return

    def portfolio_sharpe(df, weights):
        daily = df['L7'] * weights[0] + df['D1'] * weights[1] + \
                df['H4'] * weights[2] + df['TSMOM'] * weights[3]
        if daily.std() == 0: return 0
        return daily.mean() / daily.std() * np.sqrt(252)

    def portfolio_stats(df, weights, label):
        daily = df['L7'] * weights[0] + df['D1'] * weights[1] + \
                df['H4'] * weights[2] + df['TSMOM'] * weights[3]
        eq = daily.cumsum()
        dd = (eq.cummax() - eq).max()
        sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        return {'label': label, 'sharpe': sh, 'pnl': daily.sum(),
                'max_dd': dd, 'weights': weights}

    print(f"\n  --- D1: Individual Strategy Performance ---")
    for name in ['L7', 'D1', 'H4', 'TSMOM']:
        d = combined_df[name]
        sh = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0
        eq = d.cumsum(); dd = (eq.cummax() - eq).max()
        print(f"  {name:>8}: Sharpe={sh:.2f}, PnL=${d.sum():.0f}, MaxDD=${dd:.0f}")

    print(f"\n  --- D2: Two-Strategy Combos ---")
    print(f"  {'Combo':>20} {'W1':>4} {'W2':>4} {'Sharpe':>7} {'PnL':>9} {'MaxDD':>7}")
    pairs = [('L7','TSMOM'), ('L7','D1'), ('L7','H4'), ('TSMOM','D1'),
             ('TSMOM','H4'), ('D1','H4')]
    for s1, s2 in pairs:
        best_sh = 0; best_w = None
        for w in np.arange(0.2, 1.01, 0.1):
            daily = combined_df[s1] * w + combined_df[s2] * (1.0 - w + 0.2)
            sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
            if sh > best_sh: best_sh = sh; best_w = w
        if best_w is not None:
            w2 = 1.0 - best_w + 0.2
            daily = combined_df[s1] * best_w + combined_df[s2] * w2
            pnl = daily.sum()
            eq = daily.cumsum(); dd = (eq.cummax() - eq).max()
            print(f"  {s1}+{s2:>8} {best_w:>4.1f} {w2:>4.1f} {best_sh:>7.2f} "
                  f"${pnl:>8.0f} ${dd:>6.0f}")

    # D3: Full 4-strategy grid
    print(f"\n  --- D3: Full 4-Strategy Grid (top 20) ---")
    print(f"  {'L7w':>4} {'D1w':>4} {'H4w':>4} {'TSw':>4} {'Sharpe':>7} {'PnL':>9} {'MaxDD':>7}")
    results = []
    for l7w in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for d1w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for h4w in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for tw in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    if l7w + d1w + h4w + tw < 0.5: continue
                    r = portfolio_stats(combined_df, [l7w, d1w, h4w, tw],
                                       f"L7={l7w},D1={d1w},H4={h4w},TS={tw}")
                    results.append(r)
    results.sort(key=lambda x: -x['sharpe'])
    for r in results[:20]:
        w = r['weights']
        print(f"  {w[0]:>4.2f} {w[1]:>4.2f} {w[2]:>4.2f} {w[3]:>4.2f} "
              f"{r['sharpe']:>7.2f} ${r['pnl']:>8.0f} ${r['max_dd']:>6.0f}")

    # D4: Practical lot allocation (total risk = 0.1 lot)
    print(f"\n  --- D4: Practical Lot Allocation (total ~0.1 lot) ---")
    print(f"  {'Config':>30} {'L7':>6} {'D1':>6} {'H4':>6} {'TSMOM':>6} {'Sharpe':>7} {'PnL':>9}")

    lot_configs = [
        ("Current L7 only", [0.03, 0, 0, 0]),
        ("L7(0.03)+H4(0.015)", [0.03, 0, 0.015, 0]),
        ("L7(0.03)+TSMOM(0.015)", [0.03, 0, 0, 0.015]),
        ("L7(0.03)+H4(0.01)+TSMOM(0.01)", [0.03, 0, 0.01, 0.01]),
        ("L7(0.03)+D1(0.01)+H4(0.01)+TS(0.01)", [0.03, 0.01, 0.01, 0.01]),
        ("Equal 4x0.025", [0.025, 0.025, 0.025, 0.025]),
        ("L7(0.04)+TS(0.02)", [0.04, 0, 0, 0.02]),
        ("L7(0.03)+D1(0.005)+H4(0.01)+TS(0.015)", [0.03, 0.005, 0.01, 0.015]),
    ]

    # Normalize: combined_df is at 0.03 lot scale for L7
    for name, lots in lot_configs:
        w = [lots[0]/0.03, lots[1]/0.03, lots[2]/0.03, lots[3]/0.03]
        daily = combined_df['L7'] * w[0] + combined_df['D1'] * w[1] + \
                combined_df['H4'] * w[2] + combined_df['TSMOM'] * w[3]
        sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        pnl = daily.sum()
        print(f"  {name:>30} {lots[0]:>6.3f} {lots[1]:>6.3f} {lots[2]:>6.3f} {lots[3]:>6.3f} "
              f"{sh:>7.2f} ${pnl:>8.0f}")

    # D5: K-Fold of best portfolio
    print(f"\n  --- D5: Best Portfolio K-Fold ---")
    if len(results) > 0:
        best = results[0]
        w = best['weights']
        print(f"  Best portfolio: L7={w[0]}, D1={w[1]}, H4={w[2]}, TSMOM={w[3]}")
        folds = [("2015","2017"),("2017","2019"),("2019","2021"),
                 ("2021","2023"),("2023","2025"),("2025","2027")]
        combined_df_dt = combined_df.copy()
        combined_df_dt.index = pd.to_datetime(combined_df_dt.index)
        sharpes = []
        for i, (s, e) in enumerate(folds):
            sub = combined_df_dt[(combined_df_dt.index >= s) & (combined_df_dt.index < e)]
            if len(sub) < 30: continue
            daily = sub['L7'] * w[0] + sub['D1'] * w[1] + sub['H4'] * w[2] + sub['TSMOM'] * w[3]
            sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
            sharpes.append(sh)
            print(f"    F{i+1} ({s}-{e}): Sharpe={sh:.2f}, PnL=${daily.sum():.0f}")
        if sharpes:
            pos = sum(1 for s in sharpes if s > 0)
            print(f"  Portfolio K-Fold: {pos}/{len(sharpes)} positive, "
                  f"mean={np.mean(sharpes):.2f}")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R30_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R30: TSMOM Deep Validation")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    phases = [
        ("A", lambda: run_phase_A(data, h1_df)),
        ("B", lambda: run_phase_B(h1_df)),
        ("C", lambda: run_phase_C(h1_df)),
        ("D", lambda: None),  # placeholder, needs Phase A result
    ]

    combined = None
    for name, fn in phases:
        try:
            if name == "A":
                combined = run_phase_A(data, h1_df)
            elif name == "D":
                run_phase_D(combined)
            else:
                fn()
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
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
