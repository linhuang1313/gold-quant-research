#!/usr/bin/env python3
"""
R192b — Keltner V3 ATR Regime Trail vs Fixed 0.06/0.01
=======================================================
R192 tested fixed 0.14/0.025 (baseline) vs fixed 0.06/0.01 for Keltner.
But live Keltner uses V3 ATR Regime adaptive trail (3 tiers).
This test compares the ACTUAL live mechanism vs the proposed change.

Configs:
  A: V3 Regime (live) — high: 0.06/0.008, normal: 0.14/0.025, low: 0.22/0.04
  B: Fixed 0.06/0.01 (R192 proposal)
  C: Fixed 0.06/0.008 (V3 high-regime params for all)

Also tests Keltner SL 3.5 vs 6.0 (R192 GO, but needs live-trail-context confirmation).
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r192b_keltner")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30
import glob as _glob
t0 = time.time()

def compute_atr(df, period=14):
    tr = pd.DataFrame({'hl': df['High']-df['Low'],
        'hc': (df['High']-df['Close'].shift(1)).abs(),
        'lc': (df['Low']-df['Close'].shift(1)).abs()}).max(axis=1)
    return tr.rolling(period).mean()

def compute_adx(df, period=14):
    h,l,c = df['High'],df['Low'],df['Close']
    pdm=h.diff(); mdm=-l.diff()
    pdm=pdm.where((pdm>mdm)&(pdm>0),0.0); mdm=mdm.where((mdm>pdm)&(mdm>0),0.0)
    tr=pd.DataFrame({'hl':h-l,'hc':(h-c.shift(1)).abs(),'lc':(l-c.shift(1)).abs()}).max(axis=1)
    atr_s=tr.rolling(period).mean()
    pdi=100*(pdm.rolling(period).mean()/atr_s); mdi=100*(mdm.rolling(period).mean()/atr_s)
    dx=100*((pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan))
    return dx.rolling(period).mean()

def compute_atr_pctl_rolling(atr_series, lb=50):
    """Rolling ATR percentile matching live logic: (atr_series[-lb:] < current).mean()"""
    n = len(atr_series); p = np.full(n, 0.5); v = atr_series.values
    for i in range(lb, n):
        w = v[i-lb:i]; valid = w[~np.isnan(w)]
        if len(valid) >= 10:
            p[i] = float(np.sum(valid < v[i]) / len(valid))
    return pd.Series(p, index=atr_series.index)

def _mk(pos, ep, et, reason, bi, pnl):
    return {'dir':pos['dir'],'entry':pos['entry'],'exit':ep,'entry_time':pos['time'],'exit_time':et,
            'pnl':pnl,'reason':reason,'bars':bi-pos['bar'],'atr':pos['atr'],'strategy':'L8_MAX'}

def _daily(trades):
    if not trades: return pd.Series(dtype=float)
    d = {}
    for t in trades:
        k = pd.Timestamp(t['exit_time']).normalize(); d[k] = d.get(k, 0) + t['pnl']
    return pd.Series(d).sort_index()

def _sharpe(daily):
    if len(daily) < 10 or daily.std() == 0: return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))

def _stats(trades):
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'cap_pct':0,'sl_pct':0,'trail_pct':0,'timeout_pct':0}
    daily = _daily(trades); pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]
    reasons = [t['reason'] for t in trades]
    return {'n': n, 'sharpe': round(_sharpe(daily), 3), 'pnl': round(sum(pnls), 2),
            'wr': round(len(wins)/n*100, 1),
            'cap_pct': round(sum(1 for r in reasons if 'Cap' in r)/n*100, 1),
            'sl_pct': round(sum(1 for r in reasons if r == 'SL')/n*100, 1),
            'trail_pct': round(sum(1 for r in reasons if r == 'Trail')/n*100, 1),
            'timeout_pct': round(sum(1 for r in reasons if r == 'Timeout')/n*100, 1)}

ERA_SEGMENTS = {
    'full': None,
    'hike': [("2015-12-01","2019-01-01"),("2022-03-01","2023-08-01")],
    'cut':  [("2019-07-01","2022-03-01"),("2024-09-01","2026-06-01")],
    'recent_3y': [("2023-06-01","2026-06-01")],
}

def filter_era(trades, era):
    if era == 'full' or ERA_SEGMENTS[era] is None: return trades
    return [t for t in trades if any(pd.Timestamp(s) <= pd.Timestamp(t['entry_time']) < pd.Timestamp(e) for s, e in ERA_SEGMENTS[era])]

def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No H1 data")
    df = pd.read_csv(candidates[-1])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
    df = df[['Open','High','Low','Close']].copy()
    print(f"  Loaded {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df

def bt_keltner_v3(h1, sl_mult=3.5, cap=35, lot=0.02, max_hold=2, trail_mode='v3'):
    """
    Keltner backtest with V3 ATR Regime adaptive trailing.
    trail_mode:
      'v3' = live logic (3-tier regime)
      'fixed_06_01' = fixed 0.06/0.01
      'fixed_06_008' = fixed 0.06/0.008 (V3 high-regime for all)
    """
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR','ADX','EMA100','KC_upper'])

    # Compute rolling ATR percentile (matching live: 50-bar lookback)
    atr_pctl = compute_atr_pctl_rolling(df['ATR'], lb=50)

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; adx = df['ADX'].values; ema = df['EMA100'].values
    ku = df['KC_upper'].values; kl = df['KC_lower'].values
    pctl_v = atr_pctl.values
    times = df.index; n = len(df)
    tp_atr = 8.0

    trades = []; pos = None; le = -999

    for i in range(1, n):
        if pos:
            # Exit logic
            entry = pos['entry']; p_atr = pos['atr']; bar_i = pos['bar']
            if pos['dir'] == 'BUY':
                pnl_c = (c[i] - entry - SPREAD) * lot * PV
                pnl_h = (h[i] - entry - SPREAD) * lot * PV
                pnl_l = (lo[i] - entry - SPREAD) * lot * PV
            else:
                pnl_c = (entry - c[i] - SPREAD) * lot * PV
                pnl_h = (entry - lo[i] - SPREAD) * lot * PV
                pnl_l = (entry - h[i] - SPREAD) * lot * PV

            # TP
            tp_v = tp_atr * p_atr * lot * PV
            if pnl_h >= tp_v:
                trades.append(_mk(pos, c[i], times[i], "TP", i, tp_v)); pos = None; le = i; continue

            # SL
            sl_v = sl_mult * p_atr * lot * PV
            if pnl_l <= -sl_v:
                trades.append(_mk(pos, c[i], times[i], "SL", i, -sl_v)); pos = None; le = i; continue

            # Cap
            if cap > 0 and pnl_c < -cap:
                trades.append(_mk(pos, c[i], times[i], "Cap", i, -cap)); pos = None; le = i; continue

            # Trail — regime-dependent or fixed
            if trail_mode == 'v3':
                pctl_now = pctl_v[i] if not np.isnan(pctl_v[i]) else 0.5
                if pctl_now > 0.70:
                    ta = 0.06; td = 0.008
                elif pctl_now < 0.30:
                    ta = 0.22; td = 0.04
                else:
                    ta = 0.14; td = 0.025
            elif trail_mode == 'fixed_06_01':
                ta = 0.06; td = 0.01
            elif trail_mode == 'fixed_06_008':
                ta = 0.06; td = 0.008
            else:
                ta = 0.14; td = 0.025

            ad = ta * p_atr; tdd = td * p_atr
            if pos['dir'] == 'BUY' and h[i] - entry >= ad:
                ts = h[i] - tdd
                if lo[i] <= ts:
                    trades.append(_mk(pos, c[i], times[i], "Trail", i, (ts - entry - SPREAD) * lot * PV))
                    pos = None; le = i; continue
            elif pos['dir'] == 'SELL' and entry - lo[i] >= ad:
                ts = lo[i] + tdd
                if h[i] >= ts:
                    trades.append(_mk(pos, c[i], times[i], "Trail", i, (entry - ts - SPREAD) * lot * PV))
                    pos = None; le = i; continue

            # Timeout
            held = i - bar_i
            if held >= max_hold:
                trades.append(_mk(pos, c[i], times[i], "Timeout", i, pnl_c)); pos = None; le = i; continue
            continue

        # Entry
        if i - le < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(adx[i]) or adx[i] < 14: continue
        if c[i] > ku[i] and c[i] > ema[i]:
            pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < kl[i] and c[i] < ema[i]:
            pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}

    return trades


def kfold_test(h1, run_new, run_base, K=6):
    start = h1.index[0]; end = h1.index[-1]; total = (end - start).days; fd = total // K
    results = []
    for fold in range(K):
        fs = start + pd.Timedelta(days=fold * fd)
        fe = start + pd.Timedelta(days=(fold + 1) * fd) if fold < K - 1 else end + pd.Timedelta(days=1)
        h1f = h1[(h1.index >= fs) & (h1.index < fe)]
        if len(h1f) < 300: continue
        sh_new = run_new(h1f); sh_base = run_base(h1f)
        results.append({'fold':fold+1, 'new':round(sh_new,3), 'base':round(sh_base,3),
                        'win':'NEW' if sh_new > sh_base else 'BASE',
                        'period':f"{fs.date()}~{fe.date()}"})
    return results

def wf_test(h1, run_new, run_base, train_d=547, test_d=180):
    start = h1.index[0]; end = h1.index[-1]; cursor = start + pd.Timedelta(days=train_d)
    results = []; period = 0
    while cursor + pd.Timedelta(days=test_d) <= end + pd.Timedelta(days=1):
        period += 1; ts = cursor; te = cursor + pd.Timedelta(days=test_d)
        h1t = h1[(h1.index >= ts) & (h1.index < te)]
        if len(h1t) < 200: cursor += pd.Timedelta(days=test_d); continue
        sh_new = run_new(h1t); sh_base = run_base(h1t)
        results.append({'period':period, 'new':round(sh_new,3), 'base':round(sh_base,3),
                        'win':'NEW' if sh_new > sh_base else 'BASE'})
        cursor += pd.Timedelta(days=test_d)
    return results


def main():
    print("=" * 120)
    print("  R192b — Keltner V3 Regime Trail vs Fixed 0.06/0.01")
    print("=" * 120, flush=True)

    h1 = load_h1()
    results = {}

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Trail comparison — V3 vs Fixed 0.06/0.01 vs Fixed 0.06/0.008
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  === TEST 1: Trail Mode Comparison (SL=3.5, MH=2) ===")
    configs = [
        ('A_v3_regime',    'v3'),
        ('B_fixed_06_01',  'fixed_06_01'),
        ('C_fixed_06_008', 'fixed_06_008'),
        ('D_fixed_14_025', 'fixed_14_025'),
    ]
    print(f"\n  {'Config':<20} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Trail%':>7} {'Cap%':>6} {'SL%':>5} {'TO%':>5}")
    for label, mode in configs:
        t = bt_keltner_v3(h1, trail_mode=mode)
        s = _stats(t)
        print(f"  {label:<20} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['trail_pct']:>6.1f}% {s['cap_pct']:>5.1f}% {s['sl_pct']:>4.1f}% {s['timeout_pct']:>4.1f}%")
        results[label] = s

    # K-Fold: V3 vs Fixed 0.06/0.01
    print(f"\n  K-Fold: V3 (base) vs Fixed 0.06/0.01 (new):")
    def run_v3(h1f): return _stats(bt_keltner_v3(h1f, trail_mode='v3'))['sharpe']
    def run_fixed(h1f): return _stats(bt_keltner_v3(h1f, trail_mode='fixed_06_01'))['sharpe']
    kf = kfold_test(h1, run_fixed, run_v3)
    kf_wins = sum(1 for r in kf if r['win'] == 'NEW')
    for r in kf:
        print(f"    {r['period']}: V3={r['base']:.3f} Fixed={r['new']:.3f} -> {r['win']}")
    print(f"  K-Fold: Fixed wins {kf_wins}/{len(kf)}")

    # Walk-Forward
    wf = wf_test(h1, run_fixed, run_v3)
    wf_wins = sum(1 for r in wf if r['win'] == 'NEW')
    print(f"  Walk-Forward: Fixed wins {wf_wins}/{len(wf)}")

    # Era
    t_v3 = bt_keltner_v3(h1, trail_mode='v3')
    t_fixed = bt_keltner_v3(h1, trail_mode='fixed_06_01')
    print(f"\n  Era comparison:")
    for era in ['full','hike','cut','recent_3y']:
        ev3 = filter_era(t_v3, era); efx = filter_era(t_fixed, era)
        sv3 = _sharpe(_daily(ev3)); sfx = _sharpe(_daily(efx))
        print(f"    {era:<12}: V3={sv3:.3f} Fixed={sfx:.3f} (d={sfx-sv3:+.3f})")

    kf_pass = kf_wins >= 4; wf_pass = wf_wins >= 13
    v1 = 'GO' if kf_pass and wf_pass else 'NO-GO'
    print(f"\n  >>> Trail VERDICT (Fixed 0.06/0.01 vs V3): {v1}")
    results['trail_verdict'] = {'kf_wins': kf_wins, 'wf_wins': wf_wins, 'verdict': v1}

    # K-Fold: V3 vs Fixed 0.06/0.008
    print(f"\n  K-Fold: V3 (base) vs Fixed 0.06/0.008 (new):")
    def run_fixed_008(h1f): return _stats(bt_keltner_v3(h1f, trail_mode='fixed_06_008'))['sharpe']
    kf2 = kfold_test(h1, run_fixed_008, run_v3)
    kf2_wins = sum(1 for r in kf2 if r['win'] == 'NEW')
    for r in kf2:
        print(f"    {r['period']}: V3={r['base']:.3f} Fixed008={r['new']:.3f} -> {r['win']}")
    wf2 = wf_test(h1, run_fixed_008, run_v3)
    wf2_wins = sum(1 for r in wf2 if r['win'] == 'NEW')
    print(f"  K-Fold: Fixed008 wins {kf2_wins}/{len(kf2)}, WF: {wf2_wins}/{len(wf2)}")
    v2 = 'GO' if kf2_wins >= 4 and wf2_wins >= 13 else 'NO-GO'
    print(f"  >>> Trail VERDICT (Fixed 0.06/0.008 vs V3): {v2}")
    results['trail_008_verdict'] = {'kf_wins': kf2_wins, 'wf_wins': wf2_wins, 'verdict': v2}

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: SL 3.5 vs 6.0 with V3 trail (actual live context)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  === TEST 2: SL 3.5 vs 6.0 (with V3 trail, MH=2) ===")
    t_sl35 = bt_keltner_v3(h1, sl_mult=3.5, trail_mode='v3')
    t_sl60 = bt_keltner_v3(h1, sl_mult=6.0, trail_mode='v3')
    s35 = _stats(t_sl35); s60 = _stats(t_sl60)
    print(f"  SL=3.5: Sharpe={s35['sharpe']:.3f}, SL%={s35['sl_pct']:.1f}%")
    print(f"  SL=6.0: Sharpe={s60['sharpe']:.3f}, SL%={s60['sl_pct']:.1f}%")

    def run_sl35(h1f): return _stats(bt_keltner_v3(h1f, sl_mult=3.5, trail_mode='v3'))['sharpe']
    def run_sl60(h1f): return _stats(bt_keltner_v3(h1f, sl_mult=6.0, trail_mode='v3'))['sharpe']
    kf_sl = kfold_test(h1, run_sl60, run_sl35)
    kf_sl_w = sum(1 for r in kf_sl if r['win'] == 'NEW')
    print(f"\n  K-Fold (SL=6.0 vs 3.5 with V3 trail):")
    for r in kf_sl:
        print(f"    {r['period']}: SL35={r['base']:.3f} SL60={r['new']:.3f} -> {r['win']}")
    wf_sl = wf_test(h1, run_sl60, run_sl35)
    wf_sl_w = sum(1 for r in wf_sl if r['win'] == 'NEW')
    print(f"  K-Fold: {kf_sl_w}/{len(kf_sl)}, WF: {wf_sl_w}/{len(wf_sl)}")

    for era in ['full','hike','cut','recent_3y']:
        e35 = filter_era(t_sl35, era); e60 = filter_era(t_sl60, era)
        s35e = _sharpe(_daily(e35)); s60e = _sharpe(_daily(e60))
        print(f"  Era {era}: SL35={s35e:.3f} SL60={s60e:.3f} (d={s60e-s35e:+.3f})")

    v_sl = 'GO' if kf_sl_w >= 4 and wf_sl_w >= 13 else 'NO-GO'
    print(f"  >>> SL VERDICT (6.0 vs 3.5 with V3 trail): {v_sl}")
    results['sl_verdict'] = {'kf_wins': kf_sl_w, 'wf_wins': wf_sl_w, 'verdict': v_sl}

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Combined — best trail + SL=6.0 vs current V3+SL=3.5
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  === TEST 3: Combined Best vs Current ===")
    # Determine best trail from Test 1
    best_trail = 'v3'
    if v1 == 'GO': best_trail = 'fixed_06_01'
    elif v2 == 'GO': best_trail = 'fixed_06_008'

    best_sl = 3.5
    if v_sl == 'GO': best_sl = 6.0

    print(f"  Best trail: {best_trail}, Best SL: {best_sl}")
    t_cur = bt_keltner_v3(h1, sl_mult=3.5, trail_mode='v3')
    t_best = bt_keltner_v3(h1, sl_mult=best_sl, trail_mode=best_trail)
    s_cur = _stats(t_cur); s_best = _stats(t_best)
    print(f"  Current (V3+SL35): Sharpe={s_cur['sharpe']:.3f}")
    print(f"  Best ({best_trail}+SL{best_sl}): Sharpe={s_best['sharpe']:.3f} (d={s_best['sharpe']-s_cur['sharpe']:+.3f})")

    if best_trail != 'v3' or best_sl != 3.5:
        def run_best(h1f): return _stats(bt_keltner_v3(h1f, sl_mult=best_sl, trail_mode=best_trail))['sharpe']
        def run_cur(h1f): return _stats(bt_keltner_v3(h1f, sl_mult=3.5, trail_mode='v3'))['sharpe']
        kf_c = kfold_test(h1, run_best, run_cur)
        kf_c_w = sum(1 for r in kf_c if r['win'] == 'NEW')
        wf_c = wf_test(h1, run_best, run_cur)
        wf_c_w = sum(1 for r in wf_c if r['win'] == 'NEW')
        print(f"  Combined K-Fold: {kf_c_w}/{len(kf_c)}, WF: {wf_c_w}/{len(wf_c)}")
        v_c = 'GO' if kf_c_w >= 4 and wf_c_w >= 13 else 'NO-GO'
        print(f"  >>> COMBINED VERDICT: {v_c}")
        results['combined_verdict'] = {'kf_wins': kf_c_w, 'wf_wins': wf_c_w, 'verdict': v_c,
                                       'trail': best_trail, 'sl': best_sl}

    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*120}")

    with open(OUTPUT_DIR / "r192b_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {OUTPUT_DIR / 'r192b_results.json'}")

if __name__ == "__main__":
    main()
