"""
R36: Practical Edge Refinement
===============================
A: Session filter — Asian/London/NY signal quality + filter test
B: Day-of-Week effect — Mon-Fri signal quality + filter
C: ATR Percentile lot sizing — high vol reduce / low vol increase
D: Spread avoidance — skip high-spread hours
E: Cross-strategy EqCurve — joint L7+TSMOM risk control
F: 12-month rolling Walk-Forward stress test
G: Random Spread Monte Carlo — slippage simulation
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.runner import (DataBundle, run_variant, LIVE_PARITY_KWARGS)

OUT_DIR = Path("results/round36_results")
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


L7_MH8 = {
    **LIVE_PARITY_KWARGS,
    'time_adaptive_trail': {'start': 2, 'decay': 0.75, 'floor': 0.003},
    'min_entry_gap_hours': 1.0,
    'keltner_max_hold_m15': 8,
}


def get_pnl(t):
    return t.pnl if hasattr(t, 'pnl') else t['pnl']

def get_exit_time(t):
    return t.exit_time if hasattr(t, 'exit_time') else t['exit_time']

def get_entry_time(t):
    return t.entry_time if hasattr(t, 'entry_time') else t['entry_time']

def get_direction(t):
    return t.direction if hasattr(t, 'direction') else t.get('dir', '')

def get_exit_reason(t):
    return t.exit_reason if hasattr(t, 'exit_reason') else t.get('reason', '')

def get_bars_held(t):
    return t.bars_held if hasattr(t, 'bars_held') else t.get('bars', 0)

def get_entry_atr(t):
    return t.entry_atr if hasattr(t, 'entry_atr') else t.get('atr', 0)


def sharpe_from_trades(trades):
    if not trades: return 0, 0, 0, 0
    daily = {}
    for t in trades:
        d = pd.Timestamp(get_exit_time(t)).date()
        daily.setdefault(d, 0); daily[d] += get_pnl(t)
    da = np.array(list(daily.values()))
    sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
    pnl = da.sum()
    wr = sum(1 for t in trades if get_pnl(t) > 0) / len(trades) * 100
    return sh, pnl, wr, len(trades)


def add_h1_kc(h1_df, ema_period=20, mult=2.0):
    h1 = h1_df.copy()
    h1['EMA_kc'] = h1['Close'].ewm(span=ema_period, adjust=False).mean()
    tr = pd.DataFrame({
        'hl': h1['High'] - h1['Low'],
        'hc': (h1['High'] - h1['Close'].shift(1)).abs(),
        'lc': (h1['Low'] - h1['Close'].shift(1)).abs(),
    }).max(axis=1)
    h1['ATR_kc'] = tr.rolling(14).mean()
    h1['KC_U'] = h1['EMA_kc'] + mult * h1['ATR_kc']
    h1['KC_L'] = h1['EMA_kc'] - mult * h1['ATR_kc']
    h1['kc_dir'] = 'NEUTRAL'
    h1.loc[h1['Close'] > h1['KC_U'], 'kc_dir'] = 'BULL'
    h1.loc[h1['Close'] < h1['KC_L'], 'kc_dir'] = 'BEAR'
    return h1


def filter_by_h1_kc(trades, h1_kc):
    kept = []
    for t in trades:
        et = get_entry_time(t)
        td = get_direction(t)
        et_ts = pd.Timestamp(et)
        h1_mask = h1_kc.index <= et_ts
        if not h1_mask.any(): continue
        kc_d = h1_kc.loc[h1_kc.index[h1_mask][-1], 'kc_dir']
        if (td == 'BUY' and kc_d == 'BULL') or (td == 'SELL' and kc_d == 'BEAR'):
            kept.append(t)
    return kept


def apply_eqcurve(pnl_list, lb=10, cut=0, red=0.0):
    scaled = []; recent = []; triggers = 0
    for pnl in pnl_list:
        recent.append(pnl)
        if len(recent) > lb: recent.pop(0)
        active = len(recent) >= lb and np.mean(recent) < cut
        if active: triggers += 1
        mult = red if active else 1.0
        scaled.append(pnl * mult)
    return scaled, triggers


def _rec(trades, pos, exit_time, reason, bar_idx, pnl):
    trades.append({'dir': pos['dir'], 'entry': pos['entry'], 'entry_time': pos['time'],
                   'exit_time': exit_time, 'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']})


def run_tsmom(df, label, weights, sl_atr=4.0, tp_atr=12.0,
              trail_act=0.28, trail_dist=0.06, max_hold=80,
              spread=0.30, lot=0.03):
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    tr = pd.DataFrame({'hl': df['High']-df['Low'],
                        'hc': (df['High']-df['Close'].shift(1)).abs(),
                        'lc': (df['Low']-df['Close'].shift(1)).abs()}).max(axis=1)
    atr = tr.rolling(14).mean().values
    times = df.index; n = len(close)
    max_lb = max(lb for lb, _ in weights)
    trades = []; pos = None; last_exit = -999

    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = sum(w * np.sign(close[i] / close[i-lb] - 1.0) for lb, w in weights if i >= lb)
        score[i] = s

    for i in range(max_lb + 1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i]-pos['entry']-spread)*lot*100
                pnl_l = (low[i]-pos['entry']-spread)*lot*100
                pnl_c = (close[i]-pos['entry']-spread)*lot*100
            else:
                pnl_h = (pos['entry']-low[i]-spread)*lot*100
                pnl_l = (pos['entry']-high[i]-spread)*lot*100
                pnl_c = (pos['entry']-close[i]-spread)*lot*100
            tp_val = tp_atr*pos['atr']*lot*100; sl_val = sl_atr*pos['atr']*lot*100
            exited = False
            if pnl_h >= tp_val: _rec(trades,pos,times[i],"TP",i,tp_val); exited=True
            elif pnl_l <= -sl_val: _rec(trades,pos,times[i],"SL",i,-sl_val); exited=True
            else:
                ad=trail_act*pos['atr']; td=trail_dist*pos['atr']
                if pos['dir']=='BUY' and high[i]-pos['entry']>=ad:
                    ts=high[i]-td
                    if low[i]<=ts: _rec(trades,pos,times[i],"Trail",i,(ts-pos['entry']-spread)*lot*100); exited=True
                elif pos['dir']=='SELL' and pos['entry']-low[i]>=ad:
                    ts=low[i]+td
                    if high[i]>=ts: _rec(trades,pos,times[i],"Trail",i,(pos['entry']-ts-spread)*lot*100); exited=True
                if not exited and held>=max_hold: _rec(trades,pos,times[i],"Timeout",i,pnl_c); exited=True
            if not exited and not np.isnan(score[i]):
                if pos['dir']=='BUY' and score[i]<0: _rec(trades,pos,times[i],"Rev",i,pnl_c); exited=True
                elif pos['dir']=='SELL' and score[i]>0: _rec(trades,pos,times[i],"Rev",i,pnl_c); exited=True
            if exited: pos=None; last_exit=i; continue
        if pos is not None: continue
        if i-last_exit<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1 or np.isnan(score[i]): continue
        if score[i]>0: pos={'dir':'BUY','entry':close[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i]<0: pos={'dir':'SELL','entry':close[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# Phase A: Session Filter
# ═══════════════════════════════════════════════════════════════

def run_phase_A(data):
    print("\n" + "=" * 80)
    print("Phase A: Session / Time-of-Day Filter")
    print("=" * 80)

    base = run_variant(data, "A_base", verbose=False, **L7_MH8)
    trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    filtered = filter_by_h1_kc(trades, h1_kc)

    # Session definitions (UTC hours)
    sessions = {
        'Asian':  list(range(22, 24)) + list(range(0, 7)),   # 22:00-07:00 UTC
        'London': list(range(7, 13)),                          # 07:00-13:00 UTC
        'NY':     list(range(13, 17)),                         # 13:00-17:00 UTC
        'Late':   list(range(17, 22)),                         # 17:00-22:00 UTC
    }

    # A1: Session breakdown (base L7)
    print(f"\n  --- A1: Session Breakdown (L7 base) ---")
    print(f"  {'Session':>8} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'WR':>5} {'AvgPnL':>8}")
    for sname, hours in sessions.items():
        st = [t for t in trades if pd.Timestamp(get_entry_time(t)).hour in hours]
        sh, pnl, wr, n = sharpe_from_trades(st)
        avg = pnl/n if n > 0 else 0
        print(f"  {sname:>8} {n:>6} {sh:>7.2f} ${pnl:>8.0f} {wr:>4.1f}% ${avg:>7.2f}")

    # A2: Session breakdown (L7 + H1 filter)
    print(f"\n  --- A2: Session Breakdown (L7 + H1 filter) ---")
    print(f"  {'Session':>8} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'WR':>5} {'AvgPnL':>8}")
    for sname, hours in sessions.items():
        st = [t for t in filtered if pd.Timestamp(get_entry_time(t)).hour in hours]
        sh, pnl, wr, n = sharpe_from_trades(st)
        avg = pnl/n if n > 0 else 0
        print(f"  {sname:>8} {n:>6} {sh:>7.2f} ${pnl:>8.0f} {wr:>4.1f}% ${avg:>7.2f}")

    # A3: Session filter — skip worst session
    print(f"\n  --- A3: Session Exclusion Test ---")
    base_sh, base_pnl, _, base_n = sharpe_from_trades(filtered)
    print(f"  Full: Sharpe={base_sh:.2f}, N={base_n}, PnL=${base_pnl:.0f}")
    for excl_name, excl_hours in sessions.items():
        kept = [t for t in filtered if pd.Timestamp(get_entry_time(t)).hour not in excl_hours]
        sh, pnl, wr, n = sharpe_from_trades(kept)
        print(f"  Excl {excl_name}: Sharpe={sh:.2f}, N={n}, PnL=${pnl:.0f} (delta={sh-base_sh:+.2f})")

    # A4: Hourly heatmap (top/bottom hours)
    print(f"\n  --- A4: Hourly Heatmap (L7+H1filt, top/bottom) ---")
    hourly = {}
    for t in filtered:
        h = pd.Timestamp(get_entry_time(t)).hour
        hourly.setdefault(h, []).append(get_pnl(t))
    print(f"  {'Hour':>5} {'N':>5} {'PnL':>8} {'WR':>5} {'AvgPnL':>8}")
    for h in sorted(hourly.keys()):
        pnls = hourly[h]
        n = len(pnls); pnl = sum(pnls)
        wr = sum(1 for p in pnls if p > 0)/n*100
        print(f"  {h:>5} {n:>5} ${pnl:>7.0f} {wr:>4.1f}% ${pnl/n:>7.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase B: Day-of-Week Effect
# ═══════════════════════════════════════════════════════════════

def run_phase_B(data):
    print("\n" + "=" * 80)
    print("Phase B: Day-of-Week Effect")
    print("=" * 80)

    base = run_variant(data, "B_base", verbose=False, **L7_MH8)
    trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    filtered = filter_by_h1_kc(trades, h1_kc)

    days = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}

    # B1: DOW breakdown
    print(f"\n  --- B1: Day-of-Week Breakdown (L7 + H1 filter) ---")
    print(f"  {'Day':>5} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'WR':>5} {'AvgPnL':>8}")
    for dow, dname in days.items():
        dt = [t for t in filtered if pd.Timestamp(get_entry_time(t)).dayofweek == dow]
        sh, pnl, wr, n = sharpe_from_trades(dt)
        avg = pnl/n if n > 0 else 0
        print(f"  {dname:>5} {n:>6} {sh:>7.2f} ${pnl:>8.0f} {wr:>4.1f}% ${avg:>7.2f}")

    # B2: Skip worst day
    print(f"\n  --- B2: Day Exclusion Test ---")
    base_sh, base_pnl, _, base_n = sharpe_from_trades(filtered)
    print(f"  Full: Sharpe={base_sh:.2f}, N={base_n}, PnL=${base_pnl:.0f}")
    for dow, dname in days.items():
        kept = [t for t in filtered if pd.Timestamp(get_entry_time(t)).dayofweek != dow]
        sh, pnl, wr, n = sharpe_from_trades(kept)
        print(f"  Excl {dname}: Sharpe={sh:.2f}, N={n}, PnL=${pnl:.0f} (delta={sh-base_sh:+.2f})")

    # B3: K-Fold on best DOW filter if any improvement > 0.5
    print(f"\n  --- B3: DOW Filter K-Fold (if useful) ---")
    best_excl = None; best_delta = 0
    for dow, dname in days.items():
        kept = [t for t in filtered if pd.Timestamp(get_entry_time(t)).dayofweek != dow]
        sh, _, _, _ = sharpe_from_trades(kept)
        if sh - base_sh > best_delta:
            best_delta = sh - base_sh; best_excl = (dow, dname)
    if best_delta > 0.3 and best_excl:
        print(f"  Best: Exclude {best_excl[1]} (delta={best_delta:+.2f})")
        folds = [("F1","2015-01-01","2017-01-01"),("F2","2017-01-01","2019-01-01"),
                 ("F3","2019-01-01","2021-01-01"),("F4","2021-01-01","2023-01-01"),
                 ("F5","2023-01-01","2025-01-01"),("F6","2025-01-01","2026-04-01")]
        pass_ct = 0
        for fn, s, e in folds:
            sd = pd.Timestamp(s).date(); ed = pd.Timestamp(e).date()
            ft = [t for t in filtered if sd <= pd.Timestamp(get_exit_time(t)).date() < ed
                  and pd.Timestamp(get_entry_time(t)).dayofweek != best_excl[0]]
            sh, pnl, _, n = sharpe_from_trades(ft)
            if sh > 0: pass_ct += 1
            print(f"    {fn}: Sharpe={sh:.2f}, N={n}, PnL=${pnl:.0f}")
        print(f"  K-Fold pass: {pass_ct}/6")
    else:
        print(f"  No DOW filter improves Sharpe by > 0.3, skipping K-Fold")


# ═══════════════════════════════════════════════════════════════
# Phase C: ATR Percentile Lot Sizing
# ═══════════════════════════════════════════════════════════════

def run_phase_C(data):
    print("\n" + "=" * 80)
    print("Phase C: ATR Percentile Lot Sizing")
    print("=" * 80)

    base = run_variant(data, "C_base", verbose=False, **L7_MH8)
    trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    filtered = filter_by_h1_kc(trades, h1_kc)

    h1_atr = data.h1_df['ATR'].values if 'ATR' in data.h1_df.columns else None
    h1_times = data.h1_df.index

    def get_atr_pct_at_time(entry_ts):
        et = pd.Timestamp(entry_ts)
        mask = h1_times <= et
        if not mask.any(): return 50
        idx = np.where(np.array(mask))[0][-1]
        start = max(0, idx - 50)
        window = h1_atr[start:idx+1]
        window = window[~np.isnan(window)]
        if len(window) < 10: return 50
        current = h1_atr[idx]
        return np.searchsorted(np.sort(window), current) / len(window) * 100

    # C1: Performance by ATR quartile
    print(f"\n  --- C1: Performance by ATR Quartile ---")
    quartiles = {'Q1_Low': (0, 25), 'Q2_Med': (25, 50), 'Q3_High': (50, 75), 'Q4_VHigh': (75, 100)}
    print(f"  {'Quartile':>10} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'WR':>5} {'AvgPnL':>8}")
    for qname, (lo, hi) in quartiles.items():
        qt = [t for t in filtered if lo <= get_atr_pct_at_time(get_entry_time(t)) < hi]
        sh, pnl, wr, n = sharpe_from_trades(qt)
        avg = pnl/n if n > 0 else 0
        print(f"  {qname:>10} {n:>6} {sh:>7.2f} ${pnl:>8.0f} {wr:>4.1f}% ${avg:>7.2f}")

    # C2: Lot sizing schemes
    print(f"\n  --- C2: ATR-Based Lot Sizing Schemes ---")
    base_pnls = [get_pnl(t) for t in filtered]
    base_sh, base_pnl, _, _ = sharpe_from_trades(filtered)
    print(f"  Base (flat lot): Sharpe={base_sh:.2f}, PnL=${base_pnl:.0f}")

    schemes = {
        'InverseVol': lambda pct: 1.5 if pct < 25 else (1.0 if pct < 75 else 0.5),
        'MildInv':    lambda pct: 1.2 if pct < 25 else (1.0 if pct < 75 else 0.8),
        'HighOnly':   lambda pct: 1.0 if pct < 75 else 0.5,
        'LowBoost':   lambda pct: 1.5 if pct < 25 else 1.0,
        'Skip_Q4':    lambda pct: 0.0 if pct >= 75 else 1.0,
    }

    for sname, fn in schemes.items():
        scaled_pnls = []
        for t in filtered:
            pct = get_atr_pct_at_time(get_entry_time(t))
            mult = fn(pct)
            scaled_pnls.append(get_pnl(t) * mult)
        daily = {}
        for i, t in enumerate(filtered):
            d = pd.Timestamp(get_exit_time(t)).date()
            daily.setdefault(d, 0); daily[d] += scaled_pnls[i]
        da = np.array(list(daily.values()))
        sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
        eq = np.cumsum(da)
        dd = (np.maximum.accumulate(eq+2000)-(eq+2000)).max()
        print(f"  {sname:>12}: Sharpe={sh:.2f}, PnL=${sum(scaled_pnls):.0f}, MaxDD=${dd:.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase D: Spread Dynamic Avoidance
# ═══════════════════════════════════════════════════════════════

def run_phase_D(data):
    print("\n" + "=" * 80)
    print("Phase D: Spread Dynamic Avoidance")
    print("=" * 80)

    base = run_variant(data, "D_base", verbose=False, **L7_MH8)
    trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    filtered = filter_by_h1_kc(trades, h1_kc)

    # Known high-spread hours from R34: Asian session open, NFP
    high_spread_hours = [21, 22, 23, 0, 1, 2]

    # D1: PnL in high-spread vs normal hours
    print(f"\n  --- D1: High-Spread Hour Analysis ---")
    hs_trades = [t for t in filtered if pd.Timestamp(get_entry_time(t)).hour in high_spread_hours]
    normal_trades = [t for t in filtered if pd.Timestamp(get_entry_time(t)).hour not in high_spread_hours]
    sh_hs, pnl_hs, wr_hs, n_hs = sharpe_from_trades(hs_trades)
    sh_nm, pnl_nm, wr_nm, n_nm = sharpe_from_trades(normal_trades)
    print(f"  High-spread hours ({high_spread_hours}): N={n_hs}, Sharpe={sh_hs:.2f}, PnL=${pnl_hs:.0f}, WR={wr_hs:.1f}%")
    print(f"  Normal hours: N={n_nm}, Sharpe={sh_nm:.2f}, PnL=${pnl_nm:.0f}, WR={wr_nm:.1f}%")

    # D2: Impact of different spread avoidance windows
    print(f"\n  --- D2: Spread Avoidance Windows ---")
    base_sh, base_pnl, _, base_n = sharpe_from_trades(filtered)
    print(f"  Base: Sharpe={base_sh:.2f}, N={base_n}")

    avoidance_windows = {
        'Skip_21-2': [21, 22, 23, 0, 1],
        'Skip_22-1': [22, 23, 0],
        'Skip_23_only': [23],
        'Skip_21-4': [21, 22, 23, 0, 1, 2, 3],
        'Skip_NFP_Fri13': [],  # placeholder, check below
    }
    for wname, hours in avoidance_windows.items():
        if wname == 'Skip_NFP_Fri13':
            kept = [t for t in filtered if not (pd.Timestamp(get_entry_time(t)).dayofweek == 4
                    and pd.Timestamp(get_entry_time(t)).hour in [12, 13, 14])]
        else:
            kept = [t for t in filtered if pd.Timestamp(get_entry_time(t)).hour not in hours]
        sh, pnl, wr, n = sharpe_from_trades(kept)
        print(f"  {wname:>15}: Sharpe={sh:.2f}, N={n}, PnL=${pnl:.0f} (delta={sh-base_sh:+.2f})")

    # D3: Simulated variable spread impact
    print(f"\n  --- D3: Variable Spread Impact (per-trade) ---")
    base_pnls_sum = sum(get_pnl(t) for t in filtered)
    for spread_penalty_map in [
        ('Flat $0.30', lambda h: 0),
        ('Asian +$0.30', lambda h: 0.30 if h in high_spread_hours else 0),
        ('Asian +$0.50', lambda h: 0.50 if h in high_spread_hours else 0),
        ('Asian +$1.00', lambda h: 1.00 if h in high_spread_hours else 0),
    ]:
        name, fn = spread_penalty_map
        adj_pnls = []
        for t in filtered:
            h = pd.Timestamp(get_entry_time(t)).hour
            penalty = fn(h) * 0.03 * 100  # lot*100 conversion
            adj_pnls.append(get_pnl(t) - penalty)
        daily = {}
        for i, t in enumerate(filtered):
            d = pd.Timestamp(get_exit_time(t)).date()
            daily.setdefault(d, 0); daily[d] += adj_pnls[i]
        da = np.array(list(daily.values()))
        sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
        print(f"  {name:>18}: Sharpe={sh:.2f}, PnL=${sum(adj_pnls):.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase E: Cross-Strategy Joint EqCurve
# ═══════════════════════════════════════════════════════════════

def run_phase_E(data):
    print("\n" + "=" * 80)
    print("Phase E: Cross-Strategy Joint EqCurve")
    print("=" * 80)

    base = run_variant(data, "E_base", verbose=False, **L7_MH8)
    l7_trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    l7_filtered = filter_by_h1_kc(l7_trades, h1_kc)

    weights = [(480, 1.0), (1440, 0.5)]
    ts_trades = run_tsmom(data.h1_df, "E_ts", weights)

    # Build daily PnL
    l7_daily = {}
    for t in l7_filtered:
        d = pd.Timestamp(get_exit_time(t)).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += get_pnl(t)
    ts_daily = {}
    for t in ts_trades:
        d = pd.Timestamp(t['exit_time']).date()
        ts_daily.setdefault(d, 0); ts_daily[d] += t['pnl']

    l7_ds = pd.Series(l7_daily).sort_index()
    ts_ds = pd.Series(ts_daily).sort_index()
    all_idx = sorted(set(l7_ds.index) | set(ts_ds.index))

    # E1: Independent EqCurve
    print(f"\n  --- E1: Independent EqCurve (current approach) ---")
    l7_pnls = [get_pnl(t) for t in l7_filtered]
    l7_eq, l7_trigs = apply_eqcurve(l7_pnls, lb=10, cut=0, red=0.0)
    ts_pnls = [t['pnl'] for t in ts_trades]
    ts_eq, ts_trigs = apply_eqcurve(ts_pnls, lb=5, cut=0, red=0.0)

    daily_indep = {}
    for i, t in enumerate(l7_filtered):
        d = pd.Timestamp(get_exit_time(t)).date()
        daily_indep.setdefault(d, 0); daily_indep[d] += l7_eq[i]
    for i, t in enumerate(ts_trades):
        d = pd.Timestamp(t['exit_time']).date()
        daily_indep.setdefault(d, 0); daily_indep[d] += ts_eq[i] * 0.25
    da = np.array(list(daily_indep.values()))
    sh_indep = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
    print(f"  L7 EqCurve triggers: {l7_trigs}, TSMOM EqCurve triggers: {ts_trigs}")
    print(f"  Combined Sharpe: {sh_indep:.2f}, PnL: ${da.sum():.0f}")

    # E2: Joint EqCurve — pause ALL strategies when joint daily PnL is negative
    print(f"\n  --- E2: Joint Portfolio EqCurve ---")
    combo = l7_ds.reindex(all_idx, fill_value=0) + ts_ds.reindex(all_idx, fill_value=0) * 0.25
    for lb in [5, 10, 20, 30]:
        joint_eq = combo.copy()
        rolling_mean = combo.rolling(lb, min_periods=lb).mean()
        skip_mask = rolling_mean < 0
        joint_eq[skip_mask] = 0
        da = joint_eq.values
        sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
        trigs = skip_mask.sum()
        print(f"  Joint EqCurve LB={lb}: Sharpe={sh:.2f}, PnL=${da.sum():.0f}, Triggers={trigs}")

    # E3: Drawdown-based pause
    print(f"\n  --- E3: Drawdown-Based Pause ---")
    combo_raw = l7_ds.reindex(all_idx, fill_value=0) + ts_ds.reindex(all_idx, fill_value=0) * 0.25
    eq = combo_raw.cumsum()
    for dd_thresh in [50, 100, 150, 200]:
        paused = combo_raw.copy()
        peak = 0; paused_days = 0
        for i in range(len(eq)):
            peak = max(peak, eq.iloc[i])
            dd = peak - eq.iloc[i]
            if dd > dd_thresh:
                paused.iloc[i] = 0
                paused_days += 1
        da = paused.values
        sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
        print(f"  DD threshold ${dd_thresh}: Sharpe={sh:.2f}, PnL=${da.sum():.0f}, Paused={paused_days} days")


# ═══════════════════════════════════════════════════════════════
# Phase F: 12-Month Rolling Walk-Forward
# ═══════════════════════════════════════════════════════════════

def run_phase_F(data):
    print("\n" + "=" * 80)
    print("Phase F: 12-Month Rolling Walk-Forward Stress Test")
    print("=" * 80)

    base = run_variant(data, "F_base", verbose=False, **L7_MH8)
    trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    filtered = filter_by_h1_kc(trades, h1_kc)

    # F1: 12-month rolling Sharpe
    print(f"\n  --- F1: 12-Month Rolling Sharpe ---")
    daily = {}
    for t in filtered:
        d = pd.Timestamp(get_exit_time(t)).date()
        daily.setdefault(d, 0); daily[d] += get_pnl(t)
    ds = pd.Series(daily).sort_index()

    print(f"  {'Window':>20} {'Sharpe':>7} {'PnL':>9} {'Days':>5}")
    min_sh = 999; max_sh = -999; all_sh = []
    for year in range(2015, 2026):
        for start_month in [1, 7]:
            start = pd.Timestamp(f"{year}-{start_month:02d}-01").date()
            end = pd.Timestamp(f"{year}-{start_month+6:02d}-01").date() if start_month == 1 else \
                  pd.Timestamp(f"{year+1}-01-01").date()
            try:
                end = pd.Timestamp(f"{year}-{start_month+6:02d}-01").date()
            except:
                end = pd.Timestamp(f"{year+1}-{start_month+6-12:02d}-01").date()
            window = ds[(ds.index >= start) & (ds.index < end)]
            if len(window) < 30: continue
            sh = window.mean()/window.std()*np.sqrt(252) if window.std()>0 else 0
            min_sh = min(min_sh, sh); max_sh = max(max_sh, sh); all_sh.append(sh)
            print(f"  {str(start):>10}~{str(end):>10} {sh:>7.2f} ${window.sum():>8.0f} {len(window):>5}")

    if all_sh:
        print(f"\n  Summary: Min={min_sh:.2f}, Max={max_sh:.2f}, Mean={np.mean(all_sh):.2f}, "
              f"Positive={sum(1 for s in all_sh if s > 0)}/{len(all_sh)}")

    # F2: Worst drawdown periods
    print(f"\n  --- F2: Worst Drawdown Periods ---")
    eq = ds.cumsum()
    peak = eq.cummax()
    dd = peak - eq
    worst_dd = dd.nlargest(5)
    for d, val in worst_dd.items():
        peak_date = eq[:d].idxmax()
        print(f"  DD=${val:.0f} at {d} (peak at {peak_date})")

    # F3: Consecutive losing days
    print(f"\n  --- F3: Consecutive Losing Day Analysis ---")
    losing = (ds < 0).astype(int)
    max_streak = 0; current = 0; streaks = []
    for v in losing:
        if v: current += 1
        else:
            if current > 0: streaks.append(current)
            current = 0
    if current > 0: streaks.append(current)
    if streaks:
        print(f"  Max consecutive losing days: {max(streaks)}")
        print(f"  Mean losing streak: {np.mean(streaks):.1f}")
        print(f"  Streaks > 3 days: {sum(1 for s in streaks if s > 3)}")
        print(f"  Streaks > 5 days: {sum(1 for s in streaks if s > 5)}")


# ═══════════════════════════════════════════════════════════════
# Phase G: Random Spread Monte Carlo
# ═══════════════════════════════════════════════════════════════

def run_phase_G(data):
    print("\n" + "=" * 80)
    print("Phase G: Random Spread Monte Carlo (Slippage Simulation)")
    print("=" * 80)

    base = run_variant(data, "G_base", verbose=False, **L7_MH8)
    trades = base['_trades']
    h1_kc = add_h1_kc(data.h1_df, 20, 2.0)
    filtered = filter_by_h1_kc(trades, h1_kc)

    base_pnls = [get_pnl(t) for t in filtered]
    base_sh, base_pnl, _, _ = sharpe_from_trades(filtered)

    np.random.seed(42)
    lot = 0.03
    n_trials = 200

    # G1: Random spread scenarios
    print(f"\n  --- G1: Monte Carlo Spread Scenarios ({n_trials} trials each) ---")
    scenarios = {
        'Fixed $0.30':     {'mean': 0.30, 'std': 0.0},
        'Mean $0.40±0.15': {'mean': 0.40, 'std': 0.15},
        'Mean $0.50±0.20': {'mean': 0.50, 'std': 0.20},
        'Mean $0.60±0.30': {'mean': 0.60, 'std': 0.30},
        'Mean $0.80±0.40': {'mean': 0.80, 'std': 0.40},
        'Worst $1.00±0.50':{'mean': 1.00, 'std': 0.50},
    }

    for sname, params in scenarios.items():
        sharpes = []
        for trial in range(n_trials):
            extra_spreads = np.random.normal(params['mean'], params['std'], len(filtered))
            extra_spreads = np.maximum(extra_spreads, 0.10)
            base_spread = 0.30
            penalty = (extra_spreads - base_spread) * lot * 100

            adj_pnls = [get_pnl(filtered[i]) - penalty[i] for i in range(len(filtered))]
            daily = {}
            for i, t in enumerate(filtered):
                d = pd.Timestamp(get_exit_time(t)).date()
                daily.setdefault(d, 0); daily[d] += adj_pnls[i]
            da = np.array(list(daily.values()))
            sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
            sharpes.append(sh)

        sharpes = np.array(sharpes)
        print(f"  {sname:>22}: Mean={sharpes.mean():.2f}, Min={sharpes.min():.2f}, "
              f"P5={np.percentile(sharpes,5):.2f}, P25={np.percentile(sharpes,25):.2f}, "
              f"P(>0)={sum(sharpes>0)/len(sharpes)*100:.0f}%")

    # G2: Spike spread (occasional $1-3 spikes)
    print(f"\n  --- G2: Occasional Spike Spread ---")
    for spike_pct in [0.05, 0.10, 0.20, 0.30]:
        sharpes = []
        for trial in range(n_trials):
            spreads = np.full(len(filtered), 0.30)
            spike_mask = np.random.random(len(filtered)) < spike_pct
            spreads[spike_mask] = np.random.uniform(1.0, 3.0, spike_mask.sum())
            penalty = (spreads - 0.30) * lot * 100

            adj_pnls = [get_pnl(filtered[i]) - penalty[i] for i in range(len(filtered))]
            daily = {}
            for i, t in enumerate(filtered):
                d = pd.Timestamp(get_exit_time(t)).date()
                daily.setdefault(d, 0); daily[d] += adj_pnls[i]
            da = np.array(list(daily.values()))
            sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
            sharpes.append(sh)

        sharpes = np.array(sharpes)
        print(f"  Spike {spike_pct*100:.0f}%: Mean={sharpes.mean():.2f}, Min={sharpes.min():.2f}, "
              f"P5={np.percentile(sharpes,5):.2f}, P(>0)={sum(sharpes>0)/len(sharpes)*100:.0f}%")

    # G3: Session-aware spread
    print(f"\n  --- G3: Session-Aware Spread Model ---")
    for trial_count in [n_trials]:
        sharpes = []
        for trial in range(trial_count):
            adj_pnls = []
            for t in filtered:
                h = pd.Timestamp(get_entry_time(t)).hour
                if h in [22, 23, 0, 1, 2]:  # Asian
                    sp = np.random.normal(0.80, 0.30)
                elif h in [12, 13, 14] and pd.Timestamp(get_entry_time(t)).dayofweek == 4:  # NFP Fri
                    sp = np.random.normal(1.50, 0.50)
                else:
                    sp = np.random.normal(0.35, 0.10)
                sp = max(sp, 0.10)
                penalty = (sp - 0.30) * lot * 100
                adj_pnls.append(get_pnl(t) - penalty)
            daily = {}
            for i, t in enumerate(filtered):
                d = pd.Timestamp(get_exit_time(t)).date()
                daily.setdefault(d, 0); daily[d] += adj_pnls[i]
            da = np.array(list(daily.values()))
            sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
            sharpes.append(sh)
        sharpes = np.array(sharpes)
        print(f"  Session-aware: Mean={sharpes.mean():.2f}, Min={sharpes.min():.2f}, "
              f"P5={np.percentile(sharpes,5):.2f}, P(>0)={sum(sharpes>0)/len(sharpes)*100:.0f}%")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R36_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R36: Practical Edge Refinement")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()

    for name, fn in [("A", lambda: run_phase_A(data)),
                     ("B", lambda: run_phase_B(data)),
                     ("C", lambda: run_phase_C(data)),
                     ("D", lambda: run_phase_D(data)),
                     ("E", lambda: run_phase_E(data)),
                     ("F", lambda: run_phase_F(data)),
                     ("G", lambda: run_phase_G(data))]:
        try:
            fn()
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
