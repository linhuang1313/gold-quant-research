#!/usr/bin/env python3
"""
R196d — Deep Validation of Trail ta=0.02/td=0.005 + Skip Hours
================================================================
Rigorous testing before deployment:
1. Combined effect (Trail + Skip Hours together)
2. Year-by-year Sharpe comparison
3. Parameter neighborhood sensitivity (no cliff)
4. Monte Carlo parameter perturbation (1000 trials, ±20%)
5. Exit reason distribution change
6. Max consecutive loss / drawdown duration comparison
7. Annual PnL worst-year analysis
8. Walk-Forward with expanded windows (25 windows)
9. Bootstrapped confidence interval for Sharpe improvement
10. Real-world scenario: spread=0.40 and spread=0.50 robustness
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r196d_deep_trail")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
CURRENT_CONFIG = {'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}
NEW_TRAIL = {'trail_act': 0.02, 'trail_dist': 0.005}
SKIP_HOURS = {1, 20, 22, 23}

import glob as _glob
t0 = time.time()

# ═══════════════ Core helpers ═══════════════
def compute_atr(df, period=14):
    tr = pd.DataFrame({'hl': df['High']-df['Low'],'hc': (df['High']-df['Close'].shift(1)).abs(),'lc': (df['Low']-df['Close'].shift(1)).abs()}).max(axis=1)
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

def compute_atr_pctl(atr_series, lb=300):
    n=len(atr_series); p=np.full(n,np.nan); v=atr_series.values
    for i in range(lb,n):
        w=v[i-lb:i]; valid=w[~np.isnan(w)]
        if len(valid)>=30: p[i]=np.sum(valid<=v[i])/len(valid)*100
    return pd.Series(p, index=atr_series.index)

def _mk(pos,ep,et,reason,bi,pnl):
    return {'dir':pos['dir'],'entry':pos['entry'],'exit':ep,'entry_time':pos['time'],'exit_time':et,'pnl':pnl,'reason':reason,'bars':bi-pos['bar'],'atr':pos['atr']}

def _run_exit(pos,i,h,lo,c,spread,lot,pv,times,sl_atr,tp_atr,ta,td,mh,cap):
    if pos['dir']=='BUY':
        pnl_c=(c-pos['entry']-spread)*lot*pv; pnl_h=(h-pos['entry']-spread)*lot*pv; pnl_l=(lo-pos['entry']-spread)*lot*pv
    else:
        pnl_c=(pos['entry']-c-spread)*lot*pv; pnl_h=(pos['entry']-lo-spread)*lot*pv; pnl_l=(pos['entry']-h-spread)*lot*pv
    tp_v=tp_atr*pos['atr']*lot*pv; sl_v=sl_atr*pos['atr']*lot*pv
    if pnl_h>=tp_v: return _mk(pos,c,times[i],"TP",i,tp_v)
    if pnl_l<=-sl_v: return _mk(pos,c,times[i],"SL",i,-sl_v)
    if cap>0 and pnl_c<-cap: return _mk(pos,c,times[i],"Cap",i,-cap)
    ad=ta*pos['atr']; tdd=td*pos['atr']
    if pos['dir']=='BUY' and h-pos['entry']>=ad:
        ts=h-tdd
        if lo<=ts: return _mk(pos,c,times[i],"Trail",i,(ts-pos['entry']-spread)*lot*pv)
    elif pos['dir']=='SELL' and pos['entry']-lo>=ad:
        ts=lo+tdd
        if h>=ts: return _mk(pos,c,times[i],"Trail",i,(pos['entry']-ts-spread)*lot*pv)
    held=i-pos['bar']
    if held>=mh: return _mk(pos,c,times[i],"Timeout",i,pnl_c)
    return None

def bt_keltner(h1, cfg, pctl_v, pctl_f=30, spread=0.30, skip_hours=None):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
    hrs = df.index.hour
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],spread,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<14: continue
        if skip_hours and hrs[i] in skip_hours: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def _daily(trades):
    if not trades: return pd.Series(dtype=float)
    d={}
    for t in trades:
        k=pd.Timestamp(t['exit_time']).normalize(); d[k]=d.get(k,0)+t['pnl']
    return pd.Series(d).sort_index()

def _sharpe(daily):
    if len(daily)<10 or daily.std()==0: return 0.0
    return float(daily.mean()/daily.std()*np.sqrt(252))

def _stats(trades):
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0}
    daily=_daily(trades); pnls=[t['pnl'] for t in trades]; n=len(trades)
    wins=[p for p in pnls if p>0]
    eq=daily.cumsum(); dd=float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2)}

def load_h1():
    candidates=sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No H1 data")
    df=pd.read_csv(candidates[-1])
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True)
    df=df.set_index('timestamp')
    df.index=df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'},inplace=True)
    df=df[['Open','High','Low','Close']].copy()
    print(f"  H1: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})",flush=True)
    return df

# ═══════════════ TEST FUNCTIONS ═══════════════
def test_1_combined(h1, pctl):
    """Test all 4 combinations: base, trail_only, skip_only, both."""
    print(f"\n{'='*80}\n  TEST 1: COMBINED EFFECTS (4 CONFIGURATIONS)\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new_trail = copy.deepcopy(CURRENT_CONFIG); cfg_new_trail.update(NEW_TRAIL)
    
    t_base = bt_keltner(h1, cfg_base, pctl)
    t_trail = bt_keltner(h1, cfg_new_trail, pctl)
    t_skip = bt_keltner(h1, cfg_base, pctl, skip_hours=SKIP_HOURS)
    t_both = bt_keltner(h1, cfg_new_trail, pctl, skip_hours=SKIP_HOURS)
    
    configs = [
        ("A: Baseline (current)", t_base),
        ("B: Trail 0.02/0.005 only", t_trail),
        ("C: Skip Hours only", t_skip),
        ("D: Trail + Skip (combined)", t_both),
    ]
    
    results = {}
    for label, trades in configs:
        s = _stats(trades)
        print(f"  {label}: Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:.0f}, WR={s['wr']:.1f}%, MaxDD=${s['max_dd']:.0f}, N={s['n']}", flush=True)
        results[label] = s
    
    return results

def test_2_yearly(h1, pctl):
    """Year-by-year comparison."""
    print(f"\n{'='*80}\n  TEST 2: YEAR-BY-YEAR COMPARISON\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    
    results = {}
    years = sorted(h1.index.year.unique())
    print(f"  {'Year':<6} {'Base_Sh':>8} {'New_Sh':>8} {'Delta':>8} {'Base_PnL':>10} {'New_PnL':>10} {'Base_N':>7} {'New_N':>7}", flush=True)
    print(f"  {'-'*70}", flush=True)
    
    wins = 0
    for yr in years:
        h1_yr = h1[h1.index.year == yr]
        if len(h1_yr) < 500: continue
        p_yr = compute_atr_pctl(compute_atr(h1_yr), lb=min(300, len(h1_yr)//3))
        t_base = bt_keltner(h1_yr, cfg_base, p_yr)
        t_new = bt_keltner(h1_yr, cfg_new, p_yr, skip_hours=SKIP_HOURS)
        sb = _stats(t_base); sn = _stats(t_new)
        delta = sn['sharpe'] - sb['sharpe']
        if delta > 0: wins += 1
        print(f"  {yr:<6} {sb['sharpe']:>8.3f} {sn['sharpe']:>8.3f} {delta:>+8.3f} {sb['pnl']:>10.0f} {sn['pnl']:>10.0f} {sb['n']:>7} {sn['n']:>7}", flush=True)
        results[str(yr)] = {'base': sb, 'new': sn, 'delta': round(delta, 3)}
    
    print(f"\n  Years with improvement: {wins}/{len(results)}", flush=True)
    results['wins'] = wins; results['total'] = len(results) - 1
    return results

def test_3_neighborhood(h1, pctl):
    """Parameter neighborhood - check for cliff/isolated peak."""
    print(f"\n{'='*80}\n  TEST 3: PARAMETER NEIGHBORHOOD (CLIFF CHECK)\n{'='*80}", flush=True)
    
    results = {}
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    base_sh = _stats(bt_keltner(h1, cfg_base, pctl))['sharpe']
    
    ta_range = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    td_range = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]
    
    print(f"  Baseline (ta=0.06, td=0.01): Sharpe={base_sh:.3f}", flush=True)
    header = "ta\\td"
    print(f"\n  {header:<8}", end='', flush=True)
    for td in td_range: print(f" {td:>7}", end='', flush=True)
    print(flush=True)
    
    for ta in ta_range:
        print(f"  {ta:<8}", end='', flush=True)
        for td in td_range:
            if td >= ta:
                print(f" {'---':>7}", end='', flush=True)
                continue
            cfg_t = copy.deepcopy(CURRENT_CONFIG); cfg_t['trail_act'] = ta; cfg_t['trail_dist'] = td
            t = bt_keltner(h1, cfg_t, pctl)
            s = _stats(t)
            print(f" {s['sharpe']:>7.3f}", end='', flush=True)
            results[f"ta{ta}_td{td}"] = s['sharpe']
        print(flush=True)
    
    # Check for cliff: is the improvement monotonic in the neighborhood?
    target_sh = results.get("ta0.02_td0.005", 0)
    neighbors = [results.get(f"ta{ta}_td{td}", 0) for ta in [0.015, 0.02, 0.025] for td in [0.003, 0.004, 0.005, 0.006] if td < ta]
    if neighbors:
        min_neighbor = min(neighbors)
        max_drop = target_sh - min_neighbor
        print(f"\n  Target (0.02/0.005): {target_sh:.3f}", flush=True)
        print(f"  Worst neighbor: {min_neighbor:.3f}", flush=True)
        print(f"  Max drop in neighborhood: {max_drop:.3f}", flush=True)
        print(f"  Cliff risk: {'LOW' if max_drop < 0.3 else 'HIGH'}", flush=True)
    
    return results

def test_4_monte_carlo(h1, pctl):
    """Monte Carlo: randomly perturb trail params ±20%, 1000 trials."""
    print(f"\n{'='*80}\n  TEST 4: MONTE CARLO PARAMETER PERTURBATION (1000 trials, +/-20%)\n{'='*80}", flush=True)
    
    np.random.seed(42)
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    base_sh = _stats(bt_keltner(h1, cfg_base, pctl))['sharpe']
    
    ta_center = 0.02; td_center = 0.005
    n_trials = 1000
    sharpes = []
    
    for trial in range(n_trials):
        ta_pert = ta_center * (1 + np.random.uniform(-0.2, 0.2))
        td_pert = td_center * (1 + np.random.uniform(-0.2, 0.2))
        if td_pert >= ta_pert: td_pert = ta_pert * 0.8
        cfg_t = copy.deepcopy(CURRENT_CONFIG); cfg_t['trail_act'] = ta_pert; cfg_t['trail_dist'] = td_pert
        t = bt_keltner(h1, cfg_t, pctl)
        sharpes.append(_stats(t)['sharpe'])
        if (trial+1) % 100 == 0:
            print(f"    Trial {trial+1}/1000: mean={np.mean(sharpes):.3f}, min={np.min(sharpes):.3f}", flush=True)
    
    sharpes = np.array(sharpes)
    pct_better = np.sum(sharpes > base_sh) / n_trials * 100
    
    print(f"\n  Results (1000 trials with +-20% perturbation):", flush=True)
    print(f"    Baseline: {base_sh:.3f}", flush=True)
    print(f"    MC Mean: {sharpes.mean():.3f}", flush=True)
    print(f"    MC Median: {np.median(sharpes):.3f}", flush=True)
    print(f"    MC Min: {sharpes.min():.3f}", flush=True)
    print(f"    MC Max: {sharpes.max():.3f}", flush=True)
    print(f"    MC Std: {sharpes.std():.3f}", flush=True)
    print(f"    % better than baseline: {pct_better:.1f}%", flush=True)
    print(f"    5th percentile: {np.percentile(sharpes, 5):.3f}", flush=True)
    print(f"    95th percentile: {np.percentile(sharpes, 95):.3f}", flush=True)
    
    return {
        'baseline': base_sh, 'mc_mean': round(float(sharpes.mean()),3),
        'mc_median': round(float(np.median(sharpes)),3), 'mc_min': round(float(sharpes.min()),3),
        'mc_max': round(float(sharpes.max()),3), 'mc_std': round(float(sharpes.std()),3),
        'pct_better': round(pct_better,1), 'p5': round(float(np.percentile(sharpes,5)),3),
        'p95': round(float(np.percentile(sharpes,95)),3)
    }

def test_5_exit_distribution(h1, pctl):
    """Compare exit reason distribution."""
    print(f"\n{'='*80}\n  TEST 5: EXIT REASON DISTRIBUTION\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    
    t_base = bt_keltner(h1, cfg_base, pctl)
    t_new = bt_keltner(h1, cfg_new, pctl, skip_hours=SKIP_HOURS)
    
    reasons_base = Counter(t['reason'] for t in t_base)
    reasons_new = Counter(t['reason'] for t in t_new)
    
    print(f"  {'Reason':<12} {'Base_N':>8} {'Base_%':>8} {'New_N':>8} {'New_%':>8} {'Base_AvgPnL':>12} {'New_AvgPnL':>12}", flush=True)
    print(f"  {'-'*74}", flush=True)
    
    results = {}
    for reason in ['TP', 'SL', 'Cap', 'Trail', 'Timeout']:
        nb = reasons_base.get(reason, 0)
        nn = reasons_new.get(reason, 0)
        pct_b = nb/len(t_base)*100 if t_base else 0
        pct_n = nn/len(t_new)*100 if t_new else 0
        avg_b = np.mean([t['pnl'] for t in t_base if t['reason']==reason]) if nb > 0 else 0
        avg_n = np.mean([t['pnl'] for t in t_new if t['reason']==reason]) if nn > 0 else 0
        print(f"  {reason:<12} {nb:>8} {pct_b:>7.1f}% {nn:>8} {pct_n:>7.1f}% {avg_b:>12.2f} {avg_n:>12.2f}", flush=True)
        results[reason] = {'base_n': nb, 'new_n': nn, 'base_pct': round(pct_b,1), 'new_pct': round(pct_n,1), 'base_avg': round(avg_b,2), 'new_avg': round(avg_n,2)}
    
    # Average trade PnL
    avg_base = np.mean([t['pnl'] for t in t_base]) if t_base else 0
    avg_new = np.mean([t['pnl'] for t in t_new]) if t_new else 0
    print(f"\n  Avg trade PnL: base=${avg_base:.2f}, new=${avg_new:.2f}", flush=True)
    
    # Trade duration
    bars_base = [t['bars'] for t in t_base]
    bars_new = [t['bars'] for t in t_new]
    print(f"  Avg hold bars: base={np.mean(bars_base):.2f}, new={np.mean(bars_new):.2f}", flush=True)
    print(f"  Median hold bars: base={np.median(bars_base):.0f}, new={np.median(bars_new):.0f}", flush=True)
    
    results['avg_pnl'] = {'base': round(avg_base,2), 'new': round(avg_new,2)}
    results['avg_bars'] = {'base': round(float(np.mean(bars_base)),2), 'new': round(float(np.mean(bars_new)),2)}
    return results

def test_6_drawdown(h1, pctl):
    """Drawdown and consecutive loss analysis."""
    print(f"\n{'='*80}\n  TEST 6: DRAWDOWN & CONSECUTIVE LOSS ANALYSIS\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    
    t_base = bt_keltner(h1, cfg_base, pctl)
    t_new = bt_keltner(h1, cfg_new, pctl, skip_hours=SKIP_HOURS)
    
    def analyze_dd(trades, label):
        pnls = [t['pnl'] for t in sorted(trades, key=lambda x: x['entry_time'])]
        # Max consecutive losses
        max_consec = 0; cur_consec = 0; max_consec_loss = 0; cur_loss = 0
        for p in pnls:
            if p < 0:
                cur_consec += 1; cur_loss += p
                max_consec = max(max_consec, cur_consec)
                max_consec_loss = min(max_consec_loss, cur_loss)
            else:
                cur_consec = 0; cur_loss = 0
        
        # Daily drawdown duration
        daily = _daily(trades)
        eq = daily.cumsum()
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        in_dd = dd > 0
        max_dd_duration = 0; cur_dur = 0
        for val in in_dd.values:
            if val: cur_dur += 1; max_dd_duration = max(max_dd_duration, cur_dur)
            else: cur_dur = 0
        
        print(f"  {label}:", flush=True)
        print(f"    Max consecutive losses: {max_consec}", flush=True)
        print(f"    Max consecutive loss $: ${max_consec_loss:.2f}", flush=True)
        print(f"    Max DD duration (days): {max_dd_duration}", flush=True)
        print(f"    Max DD $: ${dd.max():.2f}", flush=True)
        print(f"    Largest single loss: ${min(pnls):.2f}", flush=True)
        
        return {'max_consec': max_consec, 'max_consec_loss': round(max_consec_loss,2),
                'max_dd_duration': max_dd_duration, 'max_dd': round(float(dd.max()),2),
                'largest_loss': round(min(pnls),2)}
    
    results = {}
    results['base'] = analyze_dd(t_base, "Baseline (ta=0.06/td=0.01)")
    results['new'] = analyze_dd(t_new, "New (ta=0.02/td=0.005 + Skip Hours)")
    return results

def test_7_spread_robustness(h1, pctl):
    """Test under different spread assumptions."""
    print(f"\n{'='*80}\n  TEST 7: SPREAD ROBUSTNESS\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    
    results = {}
    print(f"  {'Spread':<10} {'Base_Sh':>8} {'New_Sh':>8} {'Delta':>8} {'Base_PnL':>10} {'New_PnL':>10}", flush=True)
    print(f"  {'-'*54}", flush=True)
    
    for spread in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00]:
        t_base = bt_keltner(h1, cfg_base, pctl, spread=spread)
        t_new = bt_keltner(h1, cfg_new, pctl, spread=spread, skip_hours=SKIP_HOURS)
        sb = _stats(t_base); sn = _stats(t_new)
        delta = sn['sharpe'] - sb['sharpe']
        print(f"  ${spread:<9.2f} {sb['sharpe']:>8.3f} {sn['sharpe']:>8.3f} {delta:>+8.3f} {sb['pnl']:>10.0f} {sn['pnl']:>10.0f}", flush=True)
        results[f"sp_{spread}"] = {'base': sb['sharpe'], 'new': sn['sharpe'], 'delta': round(delta,3)}
    
    return results

def test_8_bootstrap_ci(h1, pctl):
    """Bootstrap confidence interval for Sharpe improvement."""
    print(f"\n{'='*80}\n  TEST 8: BOOTSTRAP CONFIDENCE INTERVAL (5000 resamples)\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    
    t_base = bt_keltner(h1, cfg_base, pctl)
    t_new = bt_keltner(h1, cfg_new, pctl, skip_hours=SKIP_HOURS)
    
    daily_base = _daily(t_base); daily_new = _daily(t_new)
    
    np.random.seed(123)
    n_boot = 5000
    sharpe_diffs = []
    
    for _ in range(n_boot):
        idx_b = np.random.choice(len(daily_base), size=len(daily_base), replace=True)
        idx_n = np.random.choice(len(daily_new), size=len(daily_new), replace=True)
        sh_b = _sharpe(pd.Series(daily_base.values[idx_b]))
        sh_n = _sharpe(pd.Series(daily_new.values[idx_n]))
        sharpe_diffs.append(sh_n - sh_b)
    
    sharpe_diffs = np.array(sharpe_diffs)
    ci_low = np.percentile(sharpe_diffs, 2.5)
    ci_high = np.percentile(sharpe_diffs, 97.5)
    pct_positive = np.sum(sharpe_diffs > 0) / n_boot * 100
    
    print(f"  Observed Sharpe improvement: {_sharpe(daily_new) - _sharpe(daily_base):.3f}", flush=True)
    print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]", flush=True)
    print(f"  Bootstrap mean: {sharpe_diffs.mean():.3f}", flush=True)
    print(f"  P(improvement > 0): {pct_positive:.1f}%", flush=True)
    print(f"  P(improvement > 0.2): {np.sum(sharpe_diffs > 0.2)/n_boot*100:.1f}%", flush=True)
    
    return {
        'observed': round(float(_sharpe(daily_new) - _sharpe(daily_base)), 3),
        'ci_low': round(float(ci_low), 3), 'ci_high': round(float(ci_high), 3),
        'mean': round(float(sharpe_diffs.mean()), 3),
        'pct_positive': round(pct_positive, 1)
    }

def test_9_extended_wf(h1, pctl):
    """Extended walk-forward with 25 windows."""
    print(f"\n{'='*80}\n  TEST 9: EXTENDED WALK-FORWARD (25 windows)\n{'='*80}", flush=True)
    
    cfg_base = copy.deepcopy(CURRENT_CONFIG)
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    
    n = len(h1); wins = 0
    step = int(n * 0.4 / 25)
    
    print(f"  {'Window':<8} {'Base':>8} {'New':>8} {'Winner':>8}", flush=True)
    for w in range(25):
        oos_start = int(n * 0.6) + w * step
        oos_end = min(oos_start + step, n)
        if oos_end <= oos_start: continue
        h1f = h1.iloc[oos_start:oos_end]
        if len(h1f) < 200: continue
        p_f = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        sb = _stats(bt_keltner(h1f, cfg_base, p_f))['sharpe']
        sn = _stats(bt_keltner(h1f, cfg_new, p_f, skip_hours=SKIP_HOURS))['sharpe']
        winner = 'NEW' if sn > sb else 'BASE'
        if sn > sb: wins += 1
        print(f"  W{w+1:<6} {sb:>8.3f} {sn:>8.3f} {winner:>8}", flush=True)
    
    print(f"\n  Walk-Forward: {wins}/25 windows favor new params", flush=True)
    print(f"  PASS threshold (13/25 = 52%): {'PASS' if wins >= 13 else 'FAIL'}", flush=True)
    return {'wins': wins, 'total': 25, 'pass': wins >= 13}

def test_10_worst_case(h1, pctl):
    """Worst-case scenario analysis."""
    print(f"\n{'='*80}\n  TEST 10: WORST-CASE SCENARIO ANALYSIS\n{'='*80}", flush=True)
    
    cfg_new = copy.deepcopy(CURRENT_CONFIG); cfg_new.update(NEW_TRAIL)
    t_new = bt_keltner(h1, cfg_new, pctl, skip_hours=SKIP_HOURS)
    
    daily = _daily(t_new)
    weekly = daily.resample('W').sum()
    monthly = daily.resample('ME').sum()
    annual = daily.resample('YE').sum()
    
    print(f"  Worst day: ${daily.min():.2f} ({daily.idxmin().date()})", flush=True)
    print(f"  Worst week: ${weekly.min():.2f} ({weekly.idxmin().date()})", flush=True)
    print(f"  Worst month: ${monthly.min():.2f} ({monthly.idxmin().date()})", flush=True)
    print(f"  Worst year: ${annual.min():.2f} ({annual.idxmin().year})", flush=True)
    print(f"  Best year: ${annual.max():.2f} ({annual.idxmax().year})", flush=True)
    print(f"  Avg annual: ${annual.mean():.2f}", flush=True)
    print(f"  Negative months: {(monthly < 0).sum()}/{len(monthly)}", flush=True)
    
    # VaR
    var_95 = np.percentile(daily.values, 5)
    var_99 = np.percentile(daily.values, 1)
    print(f"  Daily VaR 95%: ${var_95:.2f}", flush=True)
    print(f"  Daily VaR 99%: ${var_99:.2f}", flush=True)
    
    return {
        'worst_day': round(float(daily.min()), 2),
        'worst_week': round(float(weekly.min()), 2),
        'worst_month': round(float(monthly.min()), 2),
        'worst_year': round(float(annual.min()), 2),
        'neg_months': int((monthly < 0).sum()),
        'total_months': len(monthly),
        'var_95': round(float(var_95), 2),
        'var_99': round(float(var_99), 2)
    }

# ═══════════════ MAIN ═══════════════
if __name__ == '__main__':
    print(f"{'='*80}")
    print(f"  R196d — DEEP VALIDATION: Trail ta=0.02/td=0.005 + Skip Hours")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    h1 = load_h1()
    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR pctl: {pctl.notna().sum()} valid\n", flush=True)

    all_results = {}
    all_results['test_1_combined'] = test_1_combined(h1, pctl)
    all_results['test_2_yearly'] = test_2_yearly(h1, pctl)
    all_results['test_3_neighborhood'] = test_3_neighborhood(h1, pctl)
    all_results['test_4_monte_carlo'] = test_4_monte_carlo(h1, pctl)
    all_results['test_5_exit_dist'] = test_5_exit_distribution(h1, pctl)
    all_results['test_6_drawdown'] = test_6_drawdown(h1, pctl)
    all_results['test_7_spread'] = test_7_spread_robustness(h1, pctl)
    all_results['test_8_bootstrap'] = test_8_bootstrap_ci(h1, pctl)
    all_results['test_9_ext_wf'] = test_9_extended_wf(h1, pctl)
    all_results['test_10_worst'] = test_10_worst_case(h1, pctl)

    # Final verdict
    print(f"\n{'='*80}")
    print(f"  R196d FINAL VERDICT")
    print(f"{'='*80}", flush=True)
    
    with open(OUTPUT_DIR / "r196d_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    total_m = (time.time()-t0)/60
    print(f"\n  Total time: {total_m:.1f} minutes")
    print(f"{'='*80}")
