"""
R35: Deep Validation of All Promising Strategies
=================================================
A: H1 KC Multi-TF Filter — parameter cliff, spread robustness, yearly, +EqCurve
B: TSMOM — exit optimization, EqCurve overlay, spread robustness, yearly
C: Full Portfolio Optimization — L7+H1Filter+EqCurve+TSMOM+D1+H4
D: SuperTrend/PSAR — K-Fold, cliff test, L7 correlation
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from backtest.runner import (DataBundle, run_variant, run_kfold,
                             LIVE_PARITY_KWARGS, load_h1_aligned, H1_CSV_PATH)

OUT_DIR = Path("results/round35_results")
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


def make_daily(trades):
    daily = {}
    for t in trades:
        exit_t = t.exit_time if hasattr(t, 'exit_time') else t['exit_time']
        pnl = t.pnl if hasattr(t, 'pnl') else t['pnl']
        d = pd.Timestamp(exit_t).date()
        daily.setdefault(d, 0); daily[d] += pnl
    return pd.Series(daily).sort_index()


def sharpe_from_daily(daily_pnl):
    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        return 0
    return daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)


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
        et = t.entry_time if hasattr(t, 'entry_time') else t['entry_time']
        td = t.direction if hasattr(t, 'direction') else getattr(t, 'dir', None)
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


def run_tsmom(df, label, weights, sl_atr=3.5, tp_atr=12.0,
              trail_act=0.28, trail_dist=0.06, max_hold=50,
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

    pnls = [t['pnl'] for t in trades]
    if not pnls: return {'label':label,'n':0,'sharpe':0,'total_pnl':0,'win_rate':0,'_trades':trades}
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date(); daily.setdefault(d,0); daily[d]+=t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
    return {'label':label,'n':len(trades),'sharpe':sh,'total_pnl':sum(pnls),
            'win_rate':sum(1 for p in pnls if p>0)/len(pnls)*100,'_trades':trades,'_daily_pnl':daily}


# ═══════════════════════════════════════════════════════════════
# Phase A: H1 KC Multi-TF Filter Deep Validation
# ═══════════════════════════════════════════════════════════════

def run_phase_A(data):
    print("\n" + "=" * 80)
    print("Phase A: H1 KC Multi-TF Filter — Deep Validation")
    print("=" * 80)

    h1_df = data.h1_df

    # A1: Parameter cliff test — EMA period and multiplier
    print(f"\n  --- A1: Parameter Cliff Test ---")
    print(f"  {'EMA':>5} {'Mult':>5} {'Sharpe':>7} {'N_kept':>7} {'PnL':>9}")
    base = run_variant(data, "A1_base", verbose=False, **L7_MH8)
    trades_base = base['_trades']
    sh_base = base['sharpe']

    for ema_p in [15, 18, 20, 22, 25, 30]:
        for mult in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            h1_kc = add_h1_kc(h1_df, ema_period=ema_p, mult=mult)
            kept = filter_by_h1_kc(trades_base, h1_kc)
            if not kept: continue
            daily = {}
            for t in kept:
                exit_t = t.exit_time if hasattr(t,'exit_time') else t['exit_time']
                pnl = t.pnl if hasattr(t,'pnl') else t['pnl']
                d = pd.Timestamp(exit_t).date(); daily.setdefault(d,0); daily[d]+=pnl
            da = np.array(list(daily.values()))
            sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
            print(f"  {ema_p:>5} {mult:>5.1f} {sh:>7.2f} {len(kept):>7} ${da.sum():>8.0f}")

    # A2: Spread robustness
    print(f"\n  --- A2: Spread Robustness ---")
    h1_kc20 = add_h1_kc(h1_df, ema_period=20, mult=2.0)
    print(f"  {'Spread':>7} {'Base_Sh':>8} {'Filter_Sh':>10} {'Delta':>7}")
    for sp in [0.30, 0.40, 0.50, 0.70, 1.00]:
        kwargs = {**L7_MH8, 'spread_model': 'fixed', 'fixed_spread': sp}
        try:
            b = run_variant(data, f"A2_sp{sp}", verbose=False, **kwargs)
        except:
            b = run_variant(data, f"A2_sp{sp}", verbose=False, **L7_MH8)
        tb = b['_trades']
        kept = filter_by_h1_kc(tb, h1_kc20)
        daily_b = {}; daily_f = {}
        for t in tb:
            exit_t = t.exit_time if hasattr(t,'exit_time') else t['exit_time']
            pnl = t.pnl if hasattr(t,'pnl') else t['pnl']
            d = pd.Timestamp(exit_t).date(); daily_b.setdefault(d,0); daily_b[d]+=pnl
        for t in kept:
            exit_t = t.exit_time if hasattr(t,'exit_time') else t['exit_time']
            pnl = t.pnl if hasattr(t,'pnl') else t['pnl']
            d = pd.Timestamp(exit_t).date(); daily_f.setdefault(d,0); daily_f[d]+=pnl
        da_b = np.array(list(daily_b.values())); da_f = np.array(list(daily_f.values()))
        sh_b = da_b.mean()/da_b.std()*np.sqrt(252) if len(da_b)>1 and da_b.std()>0 else 0
        sh_f = da_f.mean()/da_f.std()*np.sqrt(252) if len(da_f)>1 and da_f.std()>0 else 0
        print(f"  ${sp:>6.2f} {sh_b:>8.2f} {sh_f:>10.2f} {sh_f-sh_b:>+7.2f}")

    # A3: Yearly breakdown
    print(f"\n  --- A3: Yearly Breakdown ---")
    kept_full = filter_by_h1_kc(trades_base, h1_kc20)
    yearly_b = {}; yearly_f = {}
    for t in trades_base:
        yr = pd.Timestamp(t.exit_time if hasattr(t,'exit_time') else t['exit_time']).year
        pnl = t.pnl if hasattr(t,'pnl') else t['pnl']
        yearly_b.setdefault(yr, []).append(pnl)
    for t in kept_full:
        yr = pd.Timestamp(t.exit_time if hasattr(t,'exit_time') else t['exit_time']).year
        pnl = t.pnl if hasattr(t,'pnl') else t['pnl']
        yearly_f.setdefault(yr, []).append(pnl)
    print(f"  {'Year':>5} {'Base_PnL':>10} {'Filt_PnL':>10} {'Base_N':>7} {'Filt_N':>7} {'Delta':>10}")
    for yr in sorted(yearly_b.keys()):
        bp = sum(yearly_b.get(yr,[])); fp = sum(yearly_f.get(yr,[]))
        bn = len(yearly_b.get(yr,[])); fn = len(yearly_f.get(yr,[]))
        print(f"  {yr:>5} ${bp:>9.0f} ${fp:>9.0f} {bn:>7} {fn:>7} ${fp-bp:>+9.0f}")

    # A4: H1 filter + EqCurve LB=10 stacking
    print(f"\n  --- A4: H1 Filter + EqCurve LB=10 Stack ---")
    pnls_filt = [t.pnl if hasattr(t,'pnl') else t['pnl'] for t in kept_full]
    scaled, triggers = apply_eqcurve(pnls_filt, lb=10, cut=0, red=0.0)
    daily_eq = {}
    for i, t in enumerate(kept_full):
        d = pd.Timestamp(t.exit_time if hasattr(t,'exit_time') else t['exit_time']).date()
        daily_eq.setdefault(d, 0); daily_eq[d] += scaled[i]
    da_eq = np.array(list(daily_eq.values()))
    sh_eq = da_eq.mean()/da_eq.std()*np.sqrt(252) if len(da_eq)>1 and da_eq.std()>0 else 0
    eq_curve = np.cumsum(scaled)
    max_dd_eq = (np.maximum.accumulate(eq_curve+2000) - (eq_curve+2000)).max()
    print(f"  L7 base: Sharpe={sh_base:.2f}")
    print(f"  L7 + H1 filter: Sharpe={sharpe_from_daily(pd.Series(make_daily(kept_full).values)):.2f}, N={len(kept_full)}")
    print(f"  L7 + H1 filter + EqCurve: Sharpe={sh_eq:.2f}, PnL=${sum(scaled):.0f}, MaxDD=${max_dd_eq:.0f}, Triggers={triggers}")


# ═══════════════════════════════════════════════════════════════
# Phase B: TSMOM Deep Validation
# ═══════════════════════════════════════════════════════════════

def run_phase_B(h1_df):
    print("\n" + "=" * 80)
    print("Phase B: TSMOM Deep Validation")
    print("=" * 80)

    # B1: Exit parameter fine-tuning
    print(f"\n  --- B1: Exit Parameter Scan ---")
    print(f"  {'SL':>4} {'TP':>4} {'MH':>4} {'Trail_A':>8} {'Trail_D':>8} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'WR':>5}")
    best = None
    weights = [(480, 1.0), (1440, 0.5)]  # 20d=480h, 60d=1440h on H1

    for sl in [2.5, 3.0, 3.5, 4.0, 5.0]:
        for tp in [8, 10, 12, 15]:
            for mh in [30, 50, 80]:
                for ta, td in [(0.28, 0.06), (0.5, 0.1), (0.8, 0.2)]:
                    r = run_tsmom(h1_df, f"TS_{sl}_{tp}_{mh}_{ta}_{td}", weights,
                                  sl_atr=sl, tp_atr=tp, max_hold=mh, trail_act=ta, trail_dist=td)
                    if r['sharpe'] > 4.0:
                        print(f"  {sl:>4.1f} {tp:>4} {mh:>4} {ta:>8.2f} {td:>8.2f} "
                              f"{r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} {r['win_rate']:>4.1f}%")
                    if best is None or r['sharpe'] > best['sharpe']:
                        best = r

    if best:
        print(f"\n  Best TSMOM: {best['label']}, Sharpe={best['sharpe']:.2f}, N={best['n']}")

    # B2: Spread robustness for best TSMOM
    print(f"\n  --- B2: Spread Robustness ---")
    if best:
        parts = best['label'].split('_')
        sl, tp, mh, ta, td = float(parts[1]), float(parts[2]), int(parts[3]), float(parts[4]), float(parts[5])
        for sp in [0.30, 0.50, 0.70, 1.00, 1.50]:
            r = run_tsmom(h1_df, f"TS_sp{sp}", weights, sl_atr=sl, tp_atr=tp,
                          max_hold=mh, trail_act=ta, trail_dist=td, spread=sp)
            print(f"  Spread ${sp:.2f}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}")

    # B3: K-Fold on best TSMOM
    print(f"\n  --- B3: K-Fold Validation ---")
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"), ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"), ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"), ("Fold6", "2025-01-01", "2026-04-01"),
    ]
    pass_count = 0
    if best:
        parts = best['label'].split('_')
        sl, tp, mh, ta, td = float(parts[1]), float(parts[2]), int(parts[3]), float(parts[4]), float(parts[5])
        for fname, start, end in folds:
            fold_h1 = h1_df[(h1_df.index >= pd.Timestamp(start, tz='UTC')) &
                            (h1_df.index < pd.Timestamp(end, tz='UTC'))]
            if len(fold_h1) < 500: continue
            r = run_tsmom(fold_h1, f"TS_{fname}", weights, sl_atr=sl, tp_atr=tp,
                          max_hold=mh, trail_act=ta, trail_dist=td)
            if r['sharpe'] > 0: pass_count += 1
            print(f"  {fname}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}")
        print(f"\n  K-Fold pass: {pass_count}/6")

    # B4: TSMOM + EqCurve
    print(f"\n  --- B4: TSMOM + EqCurve ---")
    if best and best['_trades']:
        pnls_ts = [t['pnl'] for t in best['_trades']]
        for lb, cut, red in [(10, 0, 0.0), (10, 0, 0.5), (20, 0, 0.0), (5, 0, 0.0)]:
            scaled, trigs = apply_eqcurve(pnls_ts, lb=lb, cut=cut, red=red)
            daily = {}
            for i, t in enumerate(best['_trades']):
                d = pd.Timestamp(t['exit_time']).date()
                daily.setdefault(d, 0); daily[d] += scaled[i]
            da = np.array(list(daily.values()))
            sh = da.mean()/da.std()*np.sqrt(252) if len(da)>1 and da.std()>0 else 0
            print(f"  EqCurve LB={lb}/Cut={cut}/Red={red}: Sharpe={sh:.2f}, PnL=${sum(scaled):.0f}, Triggers={trigs}")

    return best


# ═══════════════════════════════════════════════════════════════
# Phase C: Full Portfolio Optimization
# ═══════════════════════════════════════════════════════════════

def run_phase_C(data, tsmom_best):
    print("\n" + "=" * 80)
    print("Phase C: Full Portfolio Optimization")
    print("=" * 80)

    h1_df = data.h1_df
    h1_kc20 = add_h1_kc(h1_df, ema_period=20, mult=2.0)

    # Run L7 baseline
    base = run_variant(data, "C_base", verbose=False, **L7_MH8)
    l7_trades = base['_trades']
    l7_filtered = filter_by_h1_kc(l7_trades, h1_kc20)

    # Apply EqCurve to filtered L7
    pnls_filt = [t.pnl if hasattr(t,'pnl') else t['pnl'] for t in l7_filtered]
    l7_eq_pnls, _ = apply_eqcurve(pnls_filt, lb=10, cut=0, red=0.0)

    # TSMOM trades
    tsmom_trades = tsmom_best['_trades'] if tsmom_best and tsmom_best['_trades'] else []
    ts_pnls = [t['pnl'] for t in tsmom_trades]

    # D1 KC trades (run with D1 params)
    d1_df = h1_df.resample('1D').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
    from experiments.run_round32b_fix_phaseBC import backtest_generic_v2, compute_supertrend
    # Simplified D1 KC using same framework
    tr_d1 = pd.DataFrame({'hl':d1_df['High']-d1_df['Low'],
                           'hc':(d1_df['High']-d1_df['Close'].shift(1)).abs(),
                           'lc':(d1_df['Low']-d1_df['Close'].shift(1)).abs()}).max(axis=1)
    d1_df['ATR'] = tr_d1.rolling(14).mean()

    # Build daily PnL series for each component
    daily_l7 = make_daily(l7_filtered)
    daily_l7_eq = {}
    for i, t in enumerate(l7_filtered):
        d = pd.Timestamp(t.exit_time if hasattr(t,'exit_time') else t['exit_time']).date()
        daily_l7_eq.setdefault(d, 0); daily_l7_eq[d] += l7_eq_pnls[i]
    daily_l7_eq = pd.Series(daily_l7_eq).sort_index()

    daily_ts = {}
    for t in tsmom_trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily_ts.setdefault(d, 0); daily_ts[d] += t['pnl']
    daily_ts = pd.Series(daily_ts).sort_index()

    # C1: Correlation matrix
    print(f"\n  --- C1: Daily PnL Correlations ---")
    all_idx = sorted(set(daily_l7.index) | set(daily_l7_eq.index) | set(daily_ts.index))
    corr_df = pd.DataFrame({
        'L7+H1filt': daily_l7.reindex(all_idx, fill_value=0),
        'L7+H1+EqC': daily_l7_eq.reindex(all_idx, fill_value=0),
        'TSMOM': daily_ts.reindex(all_idx, fill_value=0),
    })
    print(corr_df.corr().to_string())

    # C2: Portfolio optimization — scan lot allocations
    print(f"\n  --- C2: Portfolio Lot Optimization ---")
    print(f"  {'L7_lot':>7} {'TS_lot':>7} {'Sharpe':>7} {'PnL':>9} {'MaxDD':>7}")

    for l7_mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for ts_mult in [0.0, 0.25, 0.5, 0.75, 1.0]:
            combo = pd.Series(0, index=all_idx, dtype=float)
            combo += daily_l7_eq.reindex(all_idx, fill_value=0) * l7_mult
            combo += daily_ts.reindex(all_idx, fill_value=0) * ts_mult

            sh = combo.mean() / combo.std() * np.sqrt(252) if combo.std() > 0 else 0
            eq = combo.cumsum()
            dd = (eq.cummax() - eq).max()

            if sh > 8:
                print(f"  {l7_mult:>7.2f} {ts_mult:>7.2f} {sh:>7.2f} ${combo.sum():>8.0f} ${dd:>6.0f}")

    # C3: K-Fold on best portfolio
    print(f"\n  --- C3: Best Portfolio K-Fold ---")
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"), ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"), ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"), ("Fold6", "2025-01-01", "2026-04-01"),
    ]
    pass_count = 0
    for fname, start, end in folds:
        start_d = pd.Timestamp(start).date(); end_d = pd.Timestamp(end).date()
        fold_combo = pd.Series(0, dtype=float)
        l7_fold = daily_l7_eq[(daily_l7_eq.index >= start_d) & (daily_l7_eq.index < end_d)]
        ts_fold = daily_ts[(daily_ts.index >= start_d) & (daily_ts.index < end_d)]
        all_fold_idx = sorted(set(l7_fold.index) | set(ts_fold.index))
        if not all_fold_idx: continue
        fold_combo = l7_fold.reindex(all_fold_idx, fill_value=0) * 1.0 + \
                     ts_fold.reindex(all_fold_idx, fill_value=0) * 0.5
        sh = fold_combo.mean() / fold_combo.std() * np.sqrt(252) if len(fold_combo) > 1 and fold_combo.std() > 0 else 0
        if sh > 0: pass_count += 1
        print(f"  {fname}: Sharpe={sh:.2f}, PnL=${fold_combo.sum():.0f}, Days={len(fold_combo)}")
    print(f"\n  K-Fold pass: {pass_count}/6")


# ═══════════════════════════════════════════════════════════════
# Phase D: SuperTrend / PSAR Deep Validation
# ═══════════════════════════════════════════════════════════════

def run_phase_D(h1_df, data):
    print("\n" + "=" * 80)
    print("Phase D: SuperTrend / PSAR Deep Validation")
    print("=" * 80)

    close = h1_df['Close'].values; high = h1_df['High'].values; low = h1_df['Low'].values
    times = h1_df.index
    tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
    tr[0] = high[0]-low[0]
    atr14 = pd.Series(tr).rolling(14).mean().values

    from experiments.run_round32b_fix_phaseBC import compute_supertrend, compute_psar, backtest_generic_v2

    # D1: SuperTrend K-Fold (best: Period=20, Factor=3.0, SL=5.0/TP=12/MH=30)
    print(f"\n  --- D1: SuperTrend K-Fold (P20/F3.0, SL5/TP12/MH30) ---")
    atr_p = pd.Series(tr).rolling(20).mean().values
    st_dir = compute_supertrend(high, low, close, atr_p, period=20, factor=3.0)
    signals = [None] * len(close)
    for i in range(1, len(close)):
        if st_dir[i] == -1 and st_dir[i-1] == 1: signals[i] = 'BUY'
        elif st_dir[i] == 1 and st_dir[i-1] == -1: signals[i] = 'SELL'

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"), ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"), ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"), ("Fold6", "2025-01-01", "2026-04-01"),
    ]
    pass_count = 0
    for fname, start, end in folds:
        mask = (h1_df.index >= pd.Timestamp(start, tz='UTC')) & (h1_df.index < pd.Timestamp(end, tz='UTC'))
        idx = np.where(mask.values)[0]
        if len(idx) < 500: continue
        s, e = idx[0], idx[-1]+1
        r = backtest_generic_v2(close[s:e], high[s:e], low[s:e], atr14[s:e], times[s:e],
                                signals[s:e], f"ST_{fname}", sl_atr=5.0, tp_atr=12.0, max_hold=30)
        if r['sharpe'] > 0: pass_count += 1
        print(f"  {fname}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}")
    print(f"  K-Fold pass: {pass_count}/6")

    # D2: PSAR K-Fold (best: AF=0.01/Max=0.10)
    print(f"\n  --- D2: PSAR K-Fold (AF=0.01/Max=0.10) ---")
    _, psar_dir = compute_psar(high, low, close, af_start=0.01, af_max=0.10)
    signals_p = [None] * len(close)
    for i in range(1, len(close)):
        if psar_dir[i] == 1 and psar_dir[i-1] == -1: signals_p[i] = 'BUY'
        elif psar_dir[i] == -1 and psar_dir[i-1] == 1: signals_p[i] = 'SELL'

    pass_count_p = 0
    for fname, start, end in folds:
        mask = (h1_df.index >= pd.Timestamp(start, tz='UTC')) & (h1_df.index < pd.Timestamp(end, tz='UTC'))
        idx = np.where(mask.values)[0]
        if len(idx) < 500: continue
        s, e = idx[0], idx[-1]+1
        r = backtest_generic_v2(close[s:e], high[s:e], low[s:e], atr14[s:e], times[s:e],
                                signals_p[s:e], f"PSAR_{fname}", sl_atr=3.5, tp_atr=8.0, max_hold=50)
        if r['sharpe'] > 0: pass_count_p += 1
        print(f"  {fname}: Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['total_pnl']:.0f}")
    print(f"  K-Fold pass: {pass_count_p}/6")

    # D3: ST/PSAR correlation with L7
    print(f"\n  --- D3: Correlation with L7 ---")
    l7 = run_variant(data, "D3_l7", verbose=False, **L7_MH8)
    daily_l7 = make_daily(l7['_trades'])

    r_st = backtest_generic_v2(close, high, low, atr14, times, signals,
                               "ST_full", sl_atr=5.0, tp_atr=12.0, max_hold=30)
    r_psar = backtest_generic_v2(close, high, low, atr14, times, signals_p,
                                 "PSAR_full", sl_atr=3.5, tp_atr=8.0, max_hold=50)

    daily_st = {}; daily_psar = {}
    for t in r_st['_trades']:
        d = pd.Timestamp(t['exit_time']).date(); daily_st.setdefault(d,0); daily_st[d]+=t['pnl']
    for t in r_psar['_trades']:
        d = pd.Timestamp(t['exit_time']).date(); daily_psar.setdefault(d,0); daily_psar[d]+=t['pnl']

    daily_st = pd.Series(daily_st).sort_index()
    daily_psar = pd.Series(daily_psar).sort_index()
    all_idx = sorted(set(daily_l7.index) | set(daily_st.index) | set(daily_psar.index))

    corr_df = pd.DataFrame({
        'L7': daily_l7.reindex(all_idx, fill_value=0),
        'ST': daily_st.reindex(all_idx, fill_value=0),
        'PSAR': daily_psar.reindex(all_idx, fill_value=0),
    })
    print(corr_df.corr().to_string())

    # D4: ST cliff test
    print(f"\n  --- D4: SuperTrend Parameter Cliff ---")
    print(f"  {'Period':>7} {'Factor':>7} {'Sharpe':>7}")
    for period in [10, 14, 20, 25, 30]:
        for factor in [2.0, 2.5, 3.0, 3.5, 4.0]:
            atr_p2 = pd.Series(tr).rolling(period).mean().values
            st2 = compute_supertrend(high, low, close, atr_p2, period=period, factor=factor)
            sig2 = [None]*len(close)
            for i in range(1, len(close)):
                if st2[i]==-1 and st2[i-1]==1: sig2[i]='BUY'
                elif st2[i]==1 and st2[i-1]==-1: sig2[i]='SELL'
            r2 = backtest_generic_v2(close, high, low, atr14, times, sig2,
                                     f"ST_{period}_{factor}", sl_atr=5.0, tp_atr=12.0, max_hold=30)
            print(f"  {period:>7} {factor:>7.1f} {r2['sharpe']:>7.2f}")


def main():
    t0 = time.time()
    out_path = OUT_DIR / "R35_output.txt"
    out = open(out_path, 'w', encoding='utf-8', buffering=1)
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, out)

    print(f"# R35: Deep Validation of All Promising Strategies")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data = DataBundle.load_default()
    h1_df = data.h1_df.copy()

    tsmom_best = None
    for name, fn in [("A", lambda: run_phase_A(data)),
                     ("B", lambda: run_phase_B(h1_df)),
                     ("C", lambda: run_phase_C(data, tsmom_best)),
                     ("D", lambda: run_phase_D(h1_df, data))]:
        try:
            result = fn()
            if name == "B" and result is not None:
                tsmom_best = result
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
