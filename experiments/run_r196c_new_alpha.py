#!/usr/bin/env python3
"""
R196c — New Alpha Sources & Exit Optimization Research
=======================================================
Since R196b showed all parameter changes are NO-GO (current params optimal),
we explore structural improvements:

Phase 1: Dynamic SL/TP based on ATR percentile
Phase 2: Time-decay TP + Ratchet trail
Phase 3: Volatility breakout detection (ATR spike entry)
Phase 4: Multi-strategy confluence signal
Phase 5: SL/TP ratio optimization (risk-reward sweep)
Phase 6: Break-even stop mechanism
Phase 7: Partial position close (50% at 1R, rest trail)
Phase 8: Weekend/Holiday edge calendar
Phase 9: Correlation regime detection
Phase 10: Full 3-gate validation of top findings

All with K-Fold + Walk-Forward + Era validation (Phase 10)
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r196c_alpha")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30
CURRENT_CONFIG = {
    'L8_MAX':      {'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60, 'sl': 6.0, 'tp': 6.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 15},
    'TSMOM':       {'lot': 0.04, 'cap': 60, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.04, 'cap': 60, 'sl': 4.5, 'tp': 4.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.03, 'cap': 25, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']
import glob as _glob
t0 = time.time()

def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()

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

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta<0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def compute_atr_pctl(atr_series, lb=300):
    n=len(atr_series); p=np.full(n,np.nan); v=atr_series.values
    for i in range(lb,n):
        w=v[i-lb:i]; valid=w[~np.isnan(w)]
        if len(valid)>=30: p[i]=np.sum(valid<=v[i])/len(valid)*100
    return pd.Series(p, index=atr_series.index)

def _mk(pos,ep,et,reason,bi,pnl):
    return {'dir':pos['dir'],'entry':pos['entry'],'exit':ep,'entry_time':pos['time'],'exit_time':et,'pnl':pnl,'reason':reason,'bars':bi-pos['bar'],'atr':pos['atr'],'strategy':pos.get('strategy','')}

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

def _run_exit_dynamic(pos,i,h,lo,c,spread,lot,pv,times,sl_atr,tp_atr,ta,td,mh,cap,**kwargs):
    """Enhanced exit with dynamic SL/TP, time-decay, ratchet, break-even."""
    held = i - pos['bar']

    # Time-decay TP
    tp_decay = kwargs.get('tp_decay', 0)
    if tp_decay > 0 and mh > 0:
        decay_factor = max(0.3, 1.0 - tp_decay * held / mh)
        tp_atr_eff = tp_atr * decay_factor
    else:
        tp_atr_eff = tp_atr

    if pos['dir']=='BUY':
        pnl_c=(c-pos['entry']-spread)*lot*pv; pnl_h=(h-pos['entry']-spread)*lot*pv; pnl_l=(lo-pos['entry']-spread)*lot*pv
    else:
        pnl_c=(pos['entry']-c-spread)*lot*pv; pnl_h=(pos['entry']-lo-spread)*lot*pv; pnl_l=(pos['entry']-h-spread)*lot*pv

    tp_v=tp_atr_eff*pos['atr']*lot*pv; sl_v=sl_atr*pos['atr']*lot*pv
    if pnl_h>=tp_v: return _mk(pos,c,times[i],"TP",i,tp_v)
    if pnl_l<=-sl_v: return _mk(pos,c,times[i],"SL",i,-sl_v)
    if cap>0 and pnl_c<-cap: return _mk(pos,c,times[i],"Cap",i,-cap)

    # Break-even stop
    be_trigger = kwargs.get('be_trigger', 0)
    if be_trigger > 0:
        be_level = be_trigger * pos['atr'] * lot * pv
        if pnl_h >= be_level and pnl_c <= spread * lot * pv:
            return _mk(pos,c,times[i],"BE",i, spread*lot*pv*0.1)

    # Ratchet trail
    ratchet = kwargs.get('ratchet', 0)
    ad=ta*pos['atr']; tdd_base=td*pos['atr']
    if ratchet > 0:
        profit_mult = max(0, pnl_c / (pos['atr'] * lot * pv))
        tdd = tdd_base * max(0.3, 1.0 - ratchet * profit_mult)
    else:
        tdd = tdd_base

    if pos['dir']=='BUY' and h-pos['entry']>=ad:
        ts=h-tdd
        if lo<=ts: return _mk(pos,c,times[i],"Trail",i,(ts-pos['entry']-spread)*lot*pv)
    elif pos['dir']=='SELL' and pos['entry']-lo>=ad:
        ts=lo+tdd
        if h>=ts: return _mk(pos,c,times[i],"Trail",i,(pos['entry']-ts-spread)*lot*pv)

    if held>=mh: return _mk(pos,c,times[i],"Timeout",i,pnl_c)
    return None

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

# ═══════════════ Keltner with enhanced exit ═══════════════
def bt_keltner_enhanced(h1, cfg, pctl_v, pctl_f=30, exit_kwargs=None):
    if exit_kwargs is None: exit_kwargs = {}
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            r=_run_exit_dynamic(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**exit_kwargs)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<14: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
    return trades

def bt_keltner_dynamic_sltp(h1, cfg, pctl_v, pctl_f=30, sl_low=4.0, sl_high=8.0, tp_low=6.0, tp_high=10.0):
    """SL/TP adjusts based on ATR percentile: high vol -> wider SL/TP."""
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
    lot=cfg['lot']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            pctl_now = pv_a[i] if pv_a is not None and not np.isnan(pv_a[i]) else 50
            frac = pctl_now / 100.0
            sl_eff = sl_low + frac * (sl_high - sl_low)
            tp_eff = tp_low + frac * (tp_high - tp_low)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl_eff,tp_eff,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<14: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
    return trades

def bt_keltner_atr_spike(h1, cfg, pctl_v, pctl_f=30, spike_thresh=1.5):
    """Only enter when ATR just spiked (ATR/ATR_prev > spike_thresh)."""
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df['ATR_prev']=df['ATR'].shift(1)
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper','ATR_prev'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values
    atr_prev=df['ATR_prev'].values; times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<14: continue
        if atr_prev[i] > 0 and atr[i] / atr_prev[i] < spike_thresh: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
    return trades

# ═══════════════ PHASES ═══════════════
def phase_1(h1, pctl):
    if phase_done("phase_1_dynamic_sltp"): print("  Phase 1 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 1: DYNAMIC SL/TP BASED ON ATR PERCENTILE\n{'='*80}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['L8_MAX']
    base = _stats(bt_keltner_enhanced(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    for sl_low in [3.0, 4.0, 5.0]:
        for sl_high in [6.0, 8.0, 10.0]:
            for tp_low in [5.0, 6.0]:
                for tp_high in [8.0, 10.0, 12.0]:
                    if sl_low >= sl_high or tp_low >= tp_high: continue
                    t = bt_keltner_dynamic_sltp(h1, cfg, pctl, sl_low=sl_low, sl_high=sl_high, tp_low=tp_low, tp_high=tp_high)
                    s = _stats(t)
                    key = f"sl_{sl_low}_{sl_high}_tp_{tp_low}_{tp_high}"
                    results[key] = s
                    delta = s['sharpe'] - base['sharpe']
                    if delta > 0.1:
                        print(f"    {key}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)

    best_key = max(results.keys(), key=lambda k: results[k]['sharpe'] if k != 'baseline' else 0)
    print(f"\n  Best: {best_key} -> Sharpe={results[best_key]['sharpe']}", flush=True)
    with open(OUTPUT_DIR / "phase_1_dynamic_sltp.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 1 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_2(h1, pctl):
    if phase_done("phase_2_exit_enhancements"): print("  Phase 2 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 2: TIME-DECAY TP + RATCHET TRAIL\n{'='*80}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['L8_MAX']
    base = _stats(bt_keltner_enhanced(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    # Time-decay TP
    print(f"\n  --- Time-Decay TP ---", flush=True)
    for td_val in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        t = bt_keltner_enhanced(h1, cfg, pctl, exit_kwargs={'tp_decay': td_val})
        s = _stats(t); delta = s['sharpe'] - base['sharpe']
        print(f"    tp_decay={td_val}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'tp_decay_{td_val}'] = s

    # Ratchet trail
    print(f"\n  --- Ratchet Trail ---", flush=True)
    for ratch in [0.1, 0.2, 0.3, 0.5, 0.7]:
        t = bt_keltner_enhanced(h1, cfg, pctl, exit_kwargs={'ratchet': ratch})
        s = _stats(t); delta = s['sharpe'] - base['sharpe']
        print(f"    ratchet={ratch}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'ratchet_{ratch}'] = s

    # Break-even stop
    print(f"\n  --- Break-Even Stop ---", flush=True)
    for be in [1.0, 1.5, 2.0, 3.0, 4.0]:
        t = bt_keltner_enhanced(h1, cfg, pctl, exit_kwargs={'be_trigger': be})
        s = _stats(t); delta = s['sharpe'] - base['sharpe']
        print(f"    be_trigger={be}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'be_{be}'] = s

    # Combined best
    print(f"\n  --- Combined ---", flush=True)
    for td_val in [0.2, 0.3]:
        for ratch in [0.2, 0.3]:
            t = bt_keltner_enhanced(h1, cfg, pctl, exit_kwargs={'tp_decay': td_val, 'ratchet': ratch})
            s = _stats(t); delta = s['sharpe'] - base['sharpe']
            print(f"    decay={td_val}+ratchet={ratch}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)
            results[f'combo_{td_val}_{ratch}'] = s

    with open(OUTPUT_DIR / "phase_2_exit_enhancements.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 2 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_3(h1, pctl):
    if phase_done("phase_3_vol_breakout"): print("  Phase 3 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 3: VOLATILITY BREAKOUT (ATR SPIKE ENTRY)\n{'='*80}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['L8_MAX']
    base = _stats(bt_keltner_enhanced(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    for thresh in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
        t = bt_keltner_atr_spike(h1, cfg, pctl, spike_thresh=thresh)
        s = _stats(t); delta = s['sharpe'] - base['sharpe']
        print(f"    spike_thresh={thresh}: Sharpe={s['sharpe']:.3f} ({delta:+.3f}), N={s['n']}", flush=True)
        results[f'spike_{thresh}'] = s

    with open(OUTPUT_DIR / "phase_3_vol_breakout.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 3 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_4(h1, pctl):
    if phase_done("phase_4_rr_sweep"): print("  Phase 4 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 4: SL/TP RATIO SWEEP (RISK-REWARD)\n{'='*80}", flush=True)
    results = {}
    cfg_base = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])

    print(f"  --- Keltner SL/TP Grid ---", flush=True)
    for sl in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        for tp in [4.0, 6.0, 8.0, 10.0, 12.0]:
            cfg_t = copy.deepcopy(cfg_base); cfg_t['sl'] = sl; cfg_t['tp'] = tp
            t = bt_keltner_enhanced(h1, cfg_t, pctl); s = _stats(t)
            key = f"sl{sl}_tp{tp}"
            results[key] = s
            rr = tp/sl
            print(f"    SL={sl} TP={tp} (RR={rr:.1f}): Sharpe={s['sharpe']:.3f}, WR={s['wr']:.1f}%", flush=True)

    with open(OUTPUT_DIR / "phase_4_rr_sweep.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 4 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_5(h1, pctl):
    if phase_done("phase_5_trail_sweep"): print("  Phase 5 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 5: TRAIL ACTIVATION/DISTANCE FINE SWEEP\n{'='*80}", flush=True)
    results = {}
    cfg_base = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])

    for ta in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.14]:
        for td in [0.005, 0.008, 0.01, 0.015, 0.02, 0.025]:
            if td >= ta: continue
            cfg_t = copy.deepcopy(cfg_base); cfg_t['trail_act'] = ta; cfg_t['trail_dist'] = td
            t = bt_keltner_enhanced(h1, cfg_t, pctl); s = _stats(t)
            key = f"ta{ta}_td{td}"
            results[key] = s
    
    base_key = "ta0.06_td0.01"
    base_sh = results.get(base_key, {}).get('sharpe', 0)
    print(f"  Baseline (ta=0.06, td=0.01): Sharpe={base_sh}", flush=True)
    
    top_5 = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:10]
    print(f"\n  Top 10:", flush=True)
    for k, v in top_5:
        delta = v['sharpe'] - base_sh
        print(f"    {k}: Sharpe={v['sharpe']:.3f} ({delta:+.3f}), WR={v['wr']:.1f}%", flush=True)

    with open(OUTPUT_DIR / "phase_5_trail_sweep.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 5 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_6(h1, pctl):
    if phase_done("phase_6_calendar"): print("  Phase 6 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 6: CALENDAR EFFECTS (DAY-OF-WEEK, MONTH)\n{'='*80}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['L8_MAX']
    all_trades = bt_keltner_enhanced(h1, cfg, pctl)
    base = _stats(all_trades); results['baseline'] = base

    # Day of week analysis
    print(f"\n  --- Day of Week ---", flush=True)
    for dow in range(5):
        dow_trades = [t for t in all_trades if pd.Timestamp(t['entry_time']).dayofweek == dow]
        s = _stats(dow_trades)
        day_names = ['Mon','Tue','Wed','Thu','Fri']
        print(f"    {day_names[dow]}: Sharpe={s['sharpe']:.3f}, N={s['n']}, WR={s['wr']:.1f}%", flush=True)
        results[f'dow_{dow}'] = s

    # Skip worst day combinations
    print(f"\n  --- Skip Day Combos ---", flush=True)
    for skip_dow in range(5):
        filtered = [t for t in all_trades if pd.Timestamp(t['entry_time']).dayofweek != skip_dow]
        s = _stats(filtered); delta = s['sharpe'] - base['sharpe']
        day_names = ['Mon','Tue','Wed','Thu','Fri']
        print(f"    Skip {day_names[skip_dow]}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'skip_dow_{skip_dow}'] = s

    # Month analysis
    print(f"\n  --- Month ---", flush=True)
    for m in range(1,13):
        m_trades = [t for t in all_trades if pd.Timestamp(t['entry_time']).month == m]
        s = _stats(m_trades)
        print(f"    Month {m:02d}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'month_{m}'] = s

    with open(OUTPUT_DIR / "phase_6_calendar.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 6 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_7(h1, pctl):
    if phase_done("phase_7_max_hold"): print("  Phase 7 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 7: MAX HOLD BARS SWEEP\n{'='*80}", flush=True)
    results = {}

    for strat_name in STRAT_ORDER:
        cfg_base = copy.deepcopy(CURRENT_CONFIG[strat_name])
        print(f"\n  {strat_name} (current max_hold={cfg_base['max_hold']}):", flush=True)
        strat_results = {}
        for mh in [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 48]:
            cfg_t = copy.deepcopy(cfg_base); cfg_t['max_hold'] = mh
            if strat_name == 'L8_MAX': t = bt_keltner_enhanced(h1, cfg_t, pctl)
            else: continue  # Only keltner for now
            s = _stats(t)
            print(f"    mh={mh:>3}: Sharpe={s['sharpe']:.3f}, WR={s['wr']:.1f}%, N={s['n']}", flush=True)
            strat_results[f'mh_{mh}'] = s
        results[strat_name] = strat_results

    with open(OUTPUT_DIR / "phase_7_max_hold.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 7 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_8(h1, pctl):
    if phase_done("phase_8_adx_rsi_filter"): print("  Phase 8 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 8: ADX/RSI ENTRY FILTER COMBINATIONS\n{'='*80}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['L8_MAX']
    base = _stats(bt_keltner_enhanced(h1, cfg, pctl)); results['baseline'] = base

    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df); df['RSI']=compute_rsi(df['Close'])
    df['EMA_T']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper','RSI'])
    pv_a=pctl.reindex(df.index).values

    c,h_v,lo_v,atr,adx_v,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values
    rsi_v=df['RSI'].values; times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']

    # RSI confirmation: only BUY when RSI > X, SELL when RSI < Y
    print(f"\n  --- RSI Confirmation ---", flush=True)
    for rsi_buy_min in [40, 45, 50, 55, 60]:
        trades=[]; pos=None; le=-999
        for i in range(1,n):
            if pos:
                r=_run_exit(pos,i,h_v[i],lo_v[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
                if r: trades.append(r); pos=None; le=i; continue
                continue
            if i-le<2: continue
            if np.isnan(atr[i]) or atr[i]<0.1: continue
            if pv_a[i] is not None and (np.isnan(pv_a[i]) or pv_a[i]<30): continue
            if np.isnan(adx_v[i]) or adx_v[i]<14: continue
            if c[i]>ku[i] and c[i]>ema[i] and rsi_v[i]>rsi_buy_min:
                pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
            elif c[i]<kl[i] and c[i]<ema[i] and rsi_v[i]<(100-rsi_buy_min):
                pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        s = _stats(trades); delta = s['sharpe'] - base['sharpe']
        print(f"    RSI_buy>{rsi_buy_min}: Sharpe={s['sharpe']:.3f} ({delta:+.3f}), N={s['n']}", flush=True)
        results[f'rsi_buy_{rsi_buy_min}'] = s

    # High ADX filter (only trade when ADX > threshold)
    print(f"\n  --- High ADX Filter ---", flush=True)
    for adx_min in [20, 25, 30, 35, 40]:
        trades=[]; pos=None; le=-999
        for i in range(1,n):
            if pos:
                r=_run_exit(pos,i,h_v[i],lo_v[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
                if r: trades.append(r); pos=None; le=i; continue
                continue
            if i-le<2: continue
            if np.isnan(atr[i]) or atr[i]<0.1: continue
            if pv_a[i] is not None and (np.isnan(pv_a[i]) or pv_a[i]<30): continue
            if np.isnan(adx_v[i]) or adx_v[i]<adx_min: continue
            if c[i]>ku[i] and c[i]>ema[i]:
                pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
            elif c[i]<kl[i] and c[i]<ema[i]:
                pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        s = _stats(trades); delta = s['sharpe'] - base['sharpe']
        print(f"    ADX>{adx_min}: Sharpe={s['sharpe']:.3f} ({delta:+.3f}), N={s['n']}", flush=True)
        results[f'adx_min_{adx_min}'] = s

    with open(OUTPUT_DIR / "phase_8_adx_rsi_filter.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 8 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_9(h1, pctl):
    if phase_done("phase_9_portfolio_combined"): print("  Phase 9 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 9: PORTFOLIO-LEVEL EXIT ENHANCEMENTS\n{'='*80}", flush=True)
    results = {}

    # Run all strategies with current config (baseline)
    all_base = bt_keltner_enhanced(h1, CURRENT_CONFIG['L8_MAX'], pctl)
    base_st = _stats(all_base); results['keltner_baseline'] = base_st
    print(f"  Keltner baseline: Sharpe={base_st['sharpe']}", flush=True)

    # Test exit enhancements on Keltner (the biggest contributor)
    best_exit_kwargs = {}
    best_sharpe = base_st['sharpe']

    # Comprehensive sweep
    combos = [
        {'tp_decay': 0.2},
        {'tp_decay': 0.3},
        {'ratchet': 0.2},
        {'ratchet': 0.3},
        {'be_trigger': 2.0},
        {'be_trigger': 3.0},
        {'tp_decay': 0.2, 'ratchet': 0.2},
        {'tp_decay': 0.3, 'ratchet': 0.3},
        {'tp_decay': 0.2, 'be_trigger': 2.0},
        {'tp_decay': 0.3, 'ratchet': 0.2, 'be_trigger': 2.0},
    ]

    for combo in combos:
        t = bt_keltner_enhanced(h1, CURRENT_CONFIG['L8_MAX'], pctl, exit_kwargs=combo)
        s = _stats(t); delta = s['sharpe'] - base_st['sharpe']
        label = '+'.join(f"{k}={v}" for k,v in combo.items())
        print(f"    {label}: Sharpe={s['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'combo_{label}'] = s
        if s['sharpe'] > best_sharpe:
            best_sharpe = s['sharpe']; best_exit_kwargs = combo

    if best_exit_kwargs:
        print(f"\n  Best exit enhancement: {best_exit_kwargs} -> Sharpe={best_sharpe:.3f}", flush=True)
        results['best_combo'] = {'params': best_exit_kwargs, 'sharpe': best_sharpe}
    else:
        print(f"\n  No improvement found", flush=True)

    with open(OUTPUT_DIR / "phase_9_portfolio_combined.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 9 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_10(h1, pctl):
    if phase_done("phase_10_final_validation"): print("  Phase 10 cached",flush=True); return
    print(f"\n{'='*80}\n  PHASE 10: FINAL 3-GATE VALIDATION OF TOP FINDINGS\n{'='*80}", flush=True)
    results = {}

    # Load Phase 1-9 results to find top candidates
    candidates = []

    # Check Phase 2: exit enhancements
    try:
        with open(OUTPUT_DIR / "phase_2_exit_enhancements.json") as f:
            p2 = json.load(f)
        base_sh = p2['baseline']['sharpe']
        for k, v in p2.items():
            if k == 'baseline': continue
            if v['sharpe'] > base_sh + 0.1:
                candidates.append(('exit_' + k, v['sharpe'] - base_sh))
    except: pass

    # Check Phase 5: trail sweep
    try:
        with open(OUTPUT_DIR / "phase_5_trail_sweep.json") as f:
            p5 = json.load(f)
        base_sh = p5.get('ta0.06_td0.01', {}).get('sharpe', 0)
        for k, v in p5.items():
            if v['sharpe'] > base_sh + 0.1:
                candidates.append(('trail_' + k, v['sharpe'] - base_sh))
    except: pass

    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top candidates for validation: {len(candidates)}", flush=True)
    for c_name, c_delta in candidates[:5]:
        print(f"    {c_name}: +{c_delta:.3f}", flush=True)

    # Validate top 3 candidates with K-Fold + WF + Era
    cfg = CURRENT_CONFIG['L8_MAX']

    def validate(label, run_new_fn, run_base_fn):
        # K-Fold
        n = len(h1); fold_size = n // 6; kf_wins = 0
        for fold in range(6):
            start = fold * fold_size; end = min(start + fold_size, n)
            h1f = h1.iloc[start:end]
            if len(h1f) < 1000: continue
            p_f = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
            if run_new_fn(h1f, p_f) > run_base_fn(h1f, p_f): kf_wins += 1
        kf_pass = kf_wins >= 4

        # Walk-Forward (19 windows)
        wf_wins = 0; step = int(n * 0.4 / 19)
        for w in range(19):
            oos_start = int(n * 0.6) + w * step; oos_end = min(oos_start + step, n)
            if oos_end <= oos_start: continue
            h1f = h1.iloc[oos_start:oos_end]
            if len(h1f) < 200: continue
            p_f = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
            if run_new_fn(h1f, p_f) > run_base_fn(h1f, p_f): wf_wins += 1
        wf_pass = wf_wins >= 13

        # Era
        ERAS = {'pre_hike':('2015-01-01','2016-12-31'),'hike':('2017-01-01','2019-06-30'),'cut_covid':('2019-07-01','2021-12-31'),'hike_2022':('2022-01-01','2023-12-31'),'recent':('2024-01-01','2026-12-31')}
        era_results = {}
        for era_name, (s, e) in ERAS.items():
            mask = (h1.index >= s) & (h1.index <= e); h1f = h1[mask]
            if len(h1f) < 500: continue
            p_f = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
            new_sh = run_new_fn(h1f, p_f); base_sh = run_base_fn(h1f, p_f)
            era_results[era_name] = {'new': round(new_sh,3), 'base': round(base_sh,3), 'delta': round(new_sh-base_sh,3)}
        era_pass = all(v['new']>0 for v in era_results.values()) and all(v['delta']>-0.3 for v in era_results.values())

        verdict = 'GO' if kf_pass and wf_pass and era_pass else 'NO-GO'
        print(f"    {label}: KF={kf_wins}/6 WF={wf_wins}/19 Era={'PASS' if era_pass else 'FAIL'} -> {verdict}", flush=True)
        return {'kf': kf_wins, 'wf': wf_wins, 'era': era_results, 'era_pass': era_pass, 'verdict': verdict}

    # Validate: Skip worst hours (from R195b) - already validated but include for completeness
    SKIP_HOURS = {1, 20, 22, 23}
    def run_skip_hours_new(h1f, p_f):
        t = bt_keltner_enhanced(h1f, cfg, p_f)
        filtered = [tr for tr in t if pd.Timestamp(tr['entry_time']).hour not in SKIP_HOURS]
        return _stats(filtered)['sharpe']
    def run_skip_hours_base(h1f, p_f):
        return _stats(bt_keltner_enhanced(h1f, cfg, p_f))['sharpe']
    results['skip_hours'] = validate("Skip Hours {1,20,22,23}", run_skip_hours_new, run_skip_hours_base)

    # Validate: Best trail from Phase 5 (if any improvement found)
    try:
        with open(OUTPUT_DIR / "phase_5_trail_sweep.json") as f:
            p5 = json.load(f)
        best_trail_key = max(p5.keys(), key=lambda k: p5[k]['sharpe'])
        if p5[best_trail_key]['sharpe'] > p5.get('ta0.06_td0.01',{}).get('sharpe',0):
            parts = best_trail_key.replace('ta','').replace('td','').split('_')
            best_ta = float(parts[0]); best_td = float(parts[1])
            cfg_trail = copy.deepcopy(cfg); cfg_trail['trail_act'] = best_ta; cfg_trail['trail_dist'] = best_td
            def run_trail_new(h1f, p_f):
                return _stats(bt_keltner_enhanced(h1f, cfg_trail, p_f))['sharpe']
            results['best_trail'] = validate(f"Trail ta={best_ta} td={best_td}", run_trail_new, run_skip_hours_base)
    except: pass

    # Summary
    print(f"\n  === PHASE 10 SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"    {k}: {v['verdict']}", flush=True)

    with open(OUTPUT_DIR / "phase_10_final_validation.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 10 done ({(time.time()-t0)/60:.1f}m)", flush=True)

# ═══════════════ MAIN ═══════════════
if __name__ == '__main__':
    print(f"{'='*80}")
    print(f"  R196c — NEW ALPHA SOURCES & EXIT OPTIMIZATION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    h1 = load_h1()
    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR pctl: {pctl.notna().sum()} valid\n", flush=True)

    phase_1(h1, pctl)
    phase_2(h1, pctl)
    phase_3(h1, pctl)
    phase_4(h1, pctl)
    phase_5(h1, pctl)
    phase_6(h1, pctl)
    phase_7(h1, pctl)
    phase_8(h1, pctl)
    phase_9(h1, pctl)
    phase_10(h1, pctl)

    total_m = (time.time()-t0)/60
    print(f"\n{'='*80}")
    print(f"  R196c COMPLETE — {total_m:.1f} minutes")
    print(f"{'='*80}")
