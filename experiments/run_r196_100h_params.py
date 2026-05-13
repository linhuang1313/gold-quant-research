#!/usr/bin/env python3
"""
R196 — 100-Hour Deep Strategy Parameter Exploration
====================================================
Phase 1: Keltner Channel params (KC mult, EMA span, ADX)
Phase 2: PSAR (AF step/max)
Phase 3: TSMOM (lookback windows)
Phase 4: Dual Thrust (k-factor, lookback)
Phase 5: Chandelier (ATR period, mult, EMA)
Phase 6: Session BO (lookback, hours)
Phase 7: Cap Optimization (per-strategy)
Phase 8: Intraday Volatility Patterns
Phase 9: Trade Management (scale-in, re-entry)
Phase 10: Stress Testing (tail risk, black swan sim)
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

OUTPUT_DIR = Path("results/r196_params")
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
CAPITAL = 5000
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

# ═══════════════ Parametric Strategy Backtests ═══════════════
def bt_keltner_param(h1, cfg, pctl_v, pctl_f=30, kc_mult=1.2, ema_span=25, adx_min=14, ema_trend_span=100):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=ema_trend_span,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=ema_span,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+kc_mult*df['ATR']; df['KC_lower']=df['KC_mid']-kc_mult*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
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
        if np.isnan(adx[i]) or adx[i]<adx_min: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
    return trades

def bt_psar_param(h1, cfg, pctl_v, pctl_f=30, af_step=0.01, af_max=0.05):
    df=h1.copy()
    ha,la,ca=df['High'].values,df['Low'].values,df['Close'].values; n=len(df)
    psar=np.empty(n); psar[:]=np.nan; af=af_step; rising=True; ep=ha[0]; psar[0]=la[0]
    for i in range(1,n):
        p=psar[i-1]
        if rising:
            psar[i]=p+af*(ep-p); psar[i]=min(psar[i],la[i-1],la[max(0,i-2)])
            if la[i]<psar[i]: rising=False; psar[i]=ep; ep=la[i]; af=af_step
            else:
                if ha[i]>ep: ep=ha[i]; af=min(af+af_step,af_max)
        else:
            psar[i]=p+af*(ep-p); psar[i]=max(psar[i],ha[i-1],ha[max(0,i-2)])
            if ha[i]>psar[i]: rising=True; psar[i]=ep; ep=ha[i]; af=af_step
            else:
                if la[i]<ep: ep=la[i]; af=min(af+af_step,af_max)
    df['PSAR']=psar; df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR','PSAR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,ps=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['PSAR'].values
    times=df.index; n2=len(df); lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999; prev=c[0]>ps[0]
    for i in range(1,n2):
        cur=c[i]>ps[i]
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; prev=cur; continue
            prev=cur; continue
        if i-le<2: prev=cur; continue
        if np.isnan(atr[i]) or atr[i]<0.1: prev=cur; continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): prev=cur; continue
        if cur and not prev: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        elif not cur and prev: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        prev=cur
    return trades

def bt_tsmom_param(h1, cfg, pctl_v, pctl_f=30, fast_lb=480, slow_lb=720):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    times=df.index; n=len(df); mx=max(fast_lb, slow_lb)
    sc=np.full(n,np.nan)
    for i in range(mx,n):
        v=0.0
        if c[i-fast_lb]>0: v+=0.5*np.sign(c[i]/c[i-fast_lb]-1.0)
        if c[i-slow_lb]>0: v+=0.5*np.sign(c[i]/c[i-slow_lb]-1.0)
        sc[i]=v
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(mx+1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            if pos['dir']=='BUY' and sc[i]<0:
                trades.append(_mk(pos,c[i],times[i],"Rev",i,(c[i]-pos['entry']-SPREAD)*lot*PV)); pos=None; le=i; continue
            elif pos['dir']=='SELL' and sc[i]>0:
                trades.append(_mk(pos,c[i],times[i],"Rev",i,(pos['entry']-c[i]-SPREAD)*lot*PV)); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(sc[i]) or np.isnan(sc[i-1]): continue
        if sc[i]>0 and sc[i-1]<=0: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'TSMOM'}
        elif sc[i]<0 and sc[i-1]>=0: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'TSMOM'}
    return trades

def bt_dt_param(h1, cfg, pctl_v, pctl_f=30, nb=6, k_factor=0.5):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,o,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(nb,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        hh=np.max(h[i-nb:i]); lc=np.min(c[i-nb:i]); hc=np.max(c[i-nb:i]); ll=np.min(lo[i-nb:i])
        rng=max(hh-lc,hc-ll)
        if c[i]>o[i]+k_factor*rng: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'DUAL_THRUST'}
        elif c[i]<o[i]-k_factor*rng: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'DUAL_THRUST'}
    return trades

def bt_chand_param(h1, cfg, pctl_v, pctl_f=30, atr_period=22, chand_mult=3.0, ema_span=100):
    df=h1.copy(); df['ATR']=compute_atr(df,period=atr_period)
    df['EMA']=df['Close'].ewm(span=ema_span,adjust=False).mean()
    df['RSI14']=compute_rsi(df['Close'],14)
    df=df.dropna(subset=['ATR','EMA'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['EMA'].values
    rsi_v=df['RSI14'].values; times=df.index; n=len(df); p=atr_period
    cl=np.full(n,np.nan); cs=np.full(n,np.nan)
    for i in range(p,n): cl[i]=np.max(h[i-p+1:i+1])-chand_mult*atr[i]; cs[i]=np.min(lo[i-p+1:i+1])+chand_mult*atr[i]
    d=np.zeros(n)
    for i in range(p+1,n):
        if np.isnan(cl[i]) or np.isnan(cs[i]): d[i]=d[i-1]; continue
        if c[i]>cs[i-1]: d[i]=1
        elif c[i]<cl[i-1]: d[i]=-1
        else: d[i]=d[i-1]
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(p+2,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if d[i]==1 and d[i-1]!=1 and c[i]>ema[i]:
            if not np.isnan(rsi_v[i]) and rsi_v[i]>70: continue
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'CHANDELIER'}
        elif d[i]==-1 and d[i-1]!=-1 and c[i]<ema[i]:
            if not np.isnan(rsi_v[i]) and rsi_v[i]<30: continue
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'CHANDELIER'}
    return trades

def bt_sess_bo_param(h1, cfg, pctl_v, pctl_f=30, lb=4, entry_hour=12):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    hrs=df.index.hour; times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(lb,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if hrs[i]!=entry_hour: continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        hh=max(h[i-j] for j in range(1,lb+1)); ll=min(lo[i-j] for j in range(1,lb+1))
        if c[i]>hh: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
        elif c[i]<ll: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
    return trades

# ═══════════════ PHASES ═══════════════
def phase_1(h1, pctl):
    if phase_done("phase_1_keltner"): print("  Phase 1 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 1: KELTNER CHANNEL DEEP DIVE\n{'='*100}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['L8_MAX']
    base = _stats(bt_keltner_param(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    print(f"\n  --- KC Multiplier ---", flush=True)
    for mult in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        t = bt_keltner_param(h1, cfg, pctl, kc_mult=mult); s = _stats(t)
        print(f"    kc_mult={mult}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'kc_mult_{mult}'] = s

    print(f"\n  --- EMA Span ---", flush=True)
    for span in [10, 15, 20, 25, 30, 40, 50]:
        t = bt_keltner_param(h1, cfg, pctl, ema_span=span); s = _stats(t)
        print(f"    ema_span={span}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'ema_span_{span}'] = s

    print(f"\n  --- ADX Threshold ---", flush=True)
    for adx in [0, 10, 12, 14, 16, 18, 20, 25, 30]:
        t = bt_keltner_param(h1, cfg, pctl, adx_min=adx); s = _stats(t)
        print(f"    adx_min={adx}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'adx_min_{adx}'] = s

    print(f"\n  --- Trend EMA Span ---", flush=True)
    for tspan in [50, 75, 100, 150, 200]:
        t = bt_keltner_param(h1, cfg, pctl, ema_trend_span=tspan); s = _stats(t)
        print(f"    trend_ema={tspan}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'trend_ema_{tspan}'] = s

    print(f"\n  --- Best Combo Search ---", flush=True)
    best_sh = base['sharpe']; best_combo = None
    for mult in [1.0, 1.2, 1.5]:
        for span in [20, 25, 30]:
            for adx in [12, 14, 16]:
                for tspan in [75, 100, 150]:
                    t = bt_keltner_param(h1, cfg, pctl, kc_mult=mult, ema_span=span, adx_min=adx, ema_trend_span=tspan)
                    s = _stats(t)
                    if s['sharpe'] > best_sh:
                        best_sh = s['sharpe']; best_combo = {'kc_mult':mult,'ema_span':span,'adx_min':adx,'trend_ema':tspan,'stats':s}
    if best_combo:
        print(f"    Best: kc={best_combo['kc_mult']}, ema={best_combo['ema_span']}, adx={best_combo['adx_min']}, trend={best_combo['trend_ema']} -> Sharpe={best_sh:.3f}", flush=True)
        results['best_combo'] = best_combo
    else:
        print(f"    Current params are optimal", flush=True)
    with open(OUTPUT_DIR / "phase_1_keltner.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 1 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_2(h1, pctl):
    if phase_done("phase_2_psar"): print("  Phase 2 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 2: PSAR OPTIMIZATION\n{'='*100}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['PSAR']
    base = _stats(bt_psar_param(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    print(f"\n  --- AF Step/Max ---", flush=True)
    for step in [0.005, 0.01, 0.015, 0.02, 0.03]:
        for mx in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            if step >= mx: continue
            t = bt_psar_param(h1, cfg, pctl, af_step=step, af_max=mx); s = _stats(t)
            print(f"    step={step}/max={mx}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
            results[f'af_{step}_{mx}'] = s

    with open(OUTPUT_DIR / "phase_2_psar.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 2 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_3(h1, pctl):
    if phase_done("phase_3_tsmom"): print("  Phase 3 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 3: TSMOM LOOKBACK\n{'='*100}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['TSMOM']
    base = _stats(bt_tsmom_param(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    print(f"\n  --- Fast/Slow Lookback ---", flush=True)
    for fast in [120, 240, 360, 480, 600, 720]:
        for slow in [360, 480, 720, 960, 1200]:
            if fast >= slow: continue
            t = bt_tsmom_param(h1, cfg, pctl, fast_lb=fast, slow_lb=slow); s = _stats(t)
            if s['n'] > 50:
                print(f"    fast={fast}/slow={slow}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
                results[f'lb_{fast}_{slow}'] = s

    with open(OUTPUT_DIR / "phase_3_tsmom.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 3 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_4(h1, pctl):
    if phase_done("phase_4_dt"): print("  Phase 4 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 4: DUAL THRUST\n{'='*100}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['DUAL_THRUST']
    base = _stats(bt_dt_param(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    print(f"\n  --- K-Factor ---", flush=True)
    for k in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        t = bt_dt_param(h1, cfg, pctl, k_factor=k); s = _stats(t)
        print(f"    k={k}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'k_{k}'] = s

    print(f"\n  --- Lookback Bars ---", flush=True)
    for nb in [3, 4, 5, 6, 8, 10, 12, 15, 20]:
        t = bt_dt_param(h1, cfg, pctl, nb=nb); s = _stats(t)
        print(f"    nb={nb}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'nb_{nb}'] = s

    print(f"\n  --- Best Combo ---", flush=True)
    best_sh = base['sharpe']; best_combo = None
    for k in [0.4, 0.5, 0.6, 0.7]:
        for nb in [4, 5, 6, 8, 10]:
            t = bt_dt_param(h1, cfg, pctl, k_factor=k, nb=nb); s = _stats(t)
            if s['sharpe'] > best_sh:
                best_sh = s['sharpe']; best_combo = {'k':k,'nb':nb,'stats':s}
    if best_combo:
        print(f"    Best: k={best_combo['k']}, nb={best_combo['nb']} -> Sharpe={best_sh:.3f}", flush=True)
        results['best_combo'] = best_combo
    with open(OUTPUT_DIR / "phase_4_dt.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 4 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_5(h1, pctl):
    if phase_done("phase_5_chandelier"): print("  Phase 5 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 5: CHANDELIER\n{'='*100}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['CHANDELIER']
    base = _stats(bt_chand_param(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    print(f"\n  --- ATR Period ---", flush=True)
    for p in [10, 14, 18, 22, 28, 35]:
        t = bt_chand_param(h1, cfg, pctl, atr_period=p); s = _stats(t)
        print(f"    atr_period={p}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'atr_p_{p}'] = s

    print(f"\n  --- Chandelier Multiplier ---", flush=True)
    for m in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        t = bt_chand_param(h1, cfg, pctl, chand_mult=m); s = _stats(t)
        print(f"    mult={m}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'mult_{m}'] = s

    print(f"\n  --- EMA Span ---", flush=True)
    for ema in [50, 75, 100, 150, 200]:
        t = bt_chand_param(h1, cfg, pctl, ema_span=ema); s = _stats(t)
        print(f"    ema={ema}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'ema_{ema}'] = s

    with open(OUTPUT_DIR / "phase_5_chandelier.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 5 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_6(h1, pctl):
    if phase_done("phase_6_sessbo"): print("  Phase 6 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 6: SESSION BREAKOUT\n{'='*100}", flush=True)
    results = {}
    cfg = CURRENT_CONFIG['SESS_BO']
    base = _stats(bt_sess_bo_param(h1, cfg, pctl)); results['baseline'] = base
    print(f"  Baseline: Sharpe={base['sharpe']}, N={base['n']}", flush=True)

    print(f"\n  --- Entry Hour ---", flush=True)
    for hr in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
        t = bt_sess_bo_param(h1, cfg, pctl, entry_hour=hr); s = _stats(t)
        print(f"    hour={hr}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'hour_{hr}'] = s

    print(f"\n  --- Lookback ---", flush=True)
    for lb in [2, 3, 4, 5, 6, 8, 10, 12]:
        t = bt_sess_bo_param(h1, cfg, pctl, lb=lb); s = _stats(t)
        print(f"    lb={lb}: Sharpe={s['sharpe']:.3f}, N={s['n']}", flush=True)
        results[f'lb_{lb}'] = s

    print(f"\n  --- Best Hour+LB ---", flush=True)
    best_sh = base['sharpe']; best_combo = None
    for hr in [11, 12, 13, 14]:
        for lb in [3, 4, 5, 6]:
            t = bt_sess_bo_param(h1, cfg, pctl, entry_hour=hr, lb=lb); s = _stats(t)
            if s['sharpe'] > best_sh:
                best_sh = s['sharpe']; best_combo = {'hour':hr,'lb':lb,'stats':s}
    if best_combo:
        print(f"    Best: hour={best_combo['hour']}, lb={best_combo['lb']} -> Sharpe={best_sh:.3f}", flush=True)
        results['best_combo'] = best_combo
    with open(OUTPUT_DIR / "phase_6_sessbo.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 6 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_7(h1, pctl):
    if phase_done("phase_7_cap"): print("  Phase 7 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 7: CAP OPTIMIZATION\n{'='*100}", flush=True)
    results = {}

    # Per-strategy cap sweep
    for strat in STRAT_ORDER:
        cfg_base = CURRENT_CONFIG[strat]
        base_cap = cfg_base['cap']
        print(f"\n  {strat} (current cap={base_cap}):", flush=True)
        strat_results = []
        for cap_test in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 200, 0]:
            cfg_t = copy.deepcopy(cfg_base); cfg_t['cap'] = cap_test
            if strat == 'L8_MAX': t = bt_keltner_param(h1, cfg_t, pctl)
            elif strat == 'PSAR': t = bt_psar_param(h1, cfg_t, pctl)
            elif strat == 'TSMOM': t = bt_tsmom_param(h1, cfg_t, pctl)
            elif strat == 'SESS_BO': t = bt_sess_bo_param(h1, cfg_t, pctl)
            elif strat == 'DUAL_THRUST': t = bt_dt_param(h1, cfg_t, pctl)
            else: t = bt_chand_param(h1, cfg_t, pctl)
            s = _stats(t)
            strat_results.append({'cap': cap_test, 'sharpe': s['sharpe'], 'pnl': s['pnl'], 'max_dd': s['max_dd']})
            cap_label = 'OFF' if cap_test == 0 else str(cap_test)
            print(f"    cap={cap_label:>4}: Sharpe={s['sharpe']:.3f}, MaxDD=${s['max_dd']:.0f}", flush=True)
        results[strat] = strat_results

    with open(OUTPUT_DIR / "phase_7_cap.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 7 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_8(h1, pctl):
    if phase_done("phase_8_intraday"): print("  Phase 8 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 8: INTRADAY VOLATILITY PATTERNS\n{'='*100}", flush=True)
    results = {}

    df = h1.copy(); df['ATR'] = compute_atr(df)
    df['range'] = df['High'] - df['Low']
    df['hour'] = df.index.hour; df['dow'] = df.index.dayofweek

    # Hourly average range
    print(f"\n  --- Hourly Average Range ---", flush=True)
    hr_range = df.groupby('hour')['range'].agg(['mean','std','count'])
    for hr in range(24):
        if hr in hr_range.index:
            r = hr_range.loc[hr]
            print(f"    Hour {hr:02d}: mean_range={r['mean']:.3f}, std={r['std']:.3f}", flush=True)
            results[f'hr_range_{hr}'] = {'mean': round(float(r['mean']),4), 'std': round(float(r['std']),4)}

    # Volatility clustering: autocorrelation of range
    print(f"\n  --- Range Autocorrelation ---", flush=True)
    for lag in [1, 2, 3, 5, 10, 24]:
        ac = df['range'].autocorr(lag=lag)
        print(f"    Lag {lag}: autocorr={ac:.4f}", flush=True)
        results[f'autocorr_lag_{lag}'] = round(float(ac), 4)

    # Spread-to-range ratio by hour
    print(f"\n  --- Spread/Range Ratio by Hour ---", flush=True)
    df['spread_ratio'] = SPREAD / df['range'].replace(0, np.nan)
    sr_by_hour = df.groupby('hour')['spread_ratio'].mean()
    for hr in range(24):
        if hr in sr_by_hour.index:
            ratio = sr_by_hour[hr]
            print(f"    Hour {hr:02d}: spread/range={ratio:.4f}", flush=True)
            results[f'spread_ratio_{hr}'] = round(float(ratio), 4)

    # Momentum persistence
    print(f"\n  --- Momentum Persistence (same direction continuation) ---", flush=True)
    df['ret'] = df['Close'].pct_change()
    df['dir'] = np.sign(df['ret'])
    df['same_dir_next'] = (df['dir'] == df['dir'].shift(1)).astype(float)
    persistence = df.groupby('hour')['same_dir_next'].mean()
    for hr in range(24):
        if hr in persistence.index:
            p = persistence[hr]
            results[f'persistence_{hr}'] = round(float(p), 4)
    avg_persistence = persistence.mean()
    print(f"    Average persistence: {avg_persistence:.4f} (>0.5 = momentum, <0.5 = mean-revert)", flush=True)
    results['avg_persistence'] = round(float(avg_persistence), 4)

    with open(OUTPUT_DIR / "phase_8_intraday.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 8 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_9(h1, pctl):
    if phase_done("phase_9_management"): print("  Phase 9 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 9: TRADE MANAGEMENT\n{'='*100}", flush=True)
    results = {}

    # Re-entry after SL/Cap: does taking the next signal improve?
    cfg = CURRENT_CONFIG['L8_MAX']
    all_trades = bt_keltner_param(h1, cfg, pctl)
    base_st = _stats(all_trades)
    results['baseline'] = base_st

    # Cooldown analysis: performance of trades after losses
    print(f"\n  --- Post-Loss Cooldown ---", flush=True)
    sorted_trades = sorted(all_trades, key=lambda t: t['entry_time'])
    for cooldown in [0, 1, 2, 3, 5]:
        filtered = []; skip_until = -999
        for idx, t in enumerate(sorted_trades):
            if idx <= skip_until: continue
            filtered.append(t)
            if t['pnl'] < 0:
                skip_until = idx + cooldown
        s = _stats(filtered)
        delta = s['sharpe'] - base_st['sharpe']
        print(f"    cooldown={cooldown}: Sharpe={s['sharpe']:.3f} ({delta:+.3f}), N={s['n']}", flush=True)
        results[f'cooldown_{cooldown}'] = s

    # Consecutive win/loss streaks analysis
    print(f"\n  --- Streak Reversal ---", flush=True)
    for streak_thresh in [2, 3, 4, 5]:
        filtered = []; streak = 0
        for t in sorted_trades:
            if streak >= streak_thresh:
                filtered.append(t)
            elif streak <= -streak_thresh:
                continue
            else:
                filtered.append(t)
            streak = streak + 1 if t['pnl'] > 0 else -1 if t['pnl'] < 0 else 0
        s = _stats(filtered)
        delta = s['sharpe'] - base_st['sharpe']
        print(f"    skip_after_{streak_thresh}_losses: Sharpe={s['sharpe']:.3f} ({delta:+.3f}), N={s['n']}", flush=True)
        results[f'streak_{streak_thresh}'] = s

    # Win rate by trade duration
    print(f"\n  --- Win Rate by Hold Duration ---", flush=True)
    for max_bars in [1, 2, 3, 5, 10, 15, 20]:
        dur_trades = [t for t in all_trades if t['bars'] <= max_bars]
        if dur_trades:
            s = _stats(dur_trades)
            print(f"    bars<={max_bars}: WR={s['wr']:.1f}%, N={s['n']}, avg_pnl=${s['pnl']/s['n']:.2f}", flush=True)
            results[f'dur_{max_bars}'] = s

    with open(OUTPUT_DIR / "phase_9_management.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 9 done ({(time.time()-t0)/60:.1f}m)", flush=True)

def phase_10(h1, pctl):
    if phase_done("phase_10_stress"): print("  Phase 10 cached",flush=True); return
    print(f"\n{'='*100}\n  PHASE 10: STRESS TESTING\n{'='*100}", flush=True)
    results = {}

    # Run all strategies
    all_trades = []
    for strat in STRAT_ORDER:
        cfg = CURRENT_CONFIG[strat]
        if strat == 'L8_MAX': t = bt_keltner_param(h1, cfg, pctl)
        elif strat == 'PSAR': t = bt_psar_param(h1, cfg, pctl)
        elif strat == 'TSMOM': t = bt_tsmom_param(h1, cfg, pctl)
        elif strat == 'SESS_BO': t = bt_sess_bo_param(h1, cfg, pctl)
        elif strat == 'DUAL_THRUST': t = bt_dt_param(h1, cfg, pctl)
        else: t = bt_chand_param(h1, cfg, pctl)
        all_trades.extend(t)

    # Worst day/week/month
    daily = _daily(all_trades)
    weekly = daily.resample('W').sum()
    monthly = daily.resample('ME').sum()

    print(f"\n  --- Worst Periods ---", flush=True)
    worst_day = daily.min(); worst_day_date = daily.idxmin()
    worst_week = weekly.min(); worst_week_date = weekly.idxmin()
    worst_month = monthly.min(); worst_month_date = monthly.idxmin()
    print(f"    Worst day: ${worst_day:.2f} ({worst_day_date.date()})", flush=True)
    print(f"    Worst week: ${worst_week:.2f} ({worst_week_date.date()})", flush=True)
    print(f"    Worst month: ${worst_month:.2f} ({worst_month_date.date()})", flush=True)
    results['worst_day'] = {'pnl': round(float(worst_day),2), 'date': str(worst_day_date.date())}
    results['worst_week'] = {'pnl': round(float(worst_week),2), 'date': str(worst_week_date.date())}
    results['worst_month'] = {'pnl': round(float(worst_month),2), 'date': str(worst_month_date.date())}

    # VaR and CVaR
    print(f"\n  --- Risk Metrics ---", flush=True)
    var_95 = np.percentile(daily.values, 5)
    var_99 = np.percentile(daily.values, 1)
    cvar_95 = daily[daily <= var_95].mean()
    cvar_99 = daily[daily <= var_99].mean()
    print(f"    VaR 95%: ${var_95:.2f}", flush=True)
    print(f"    VaR 99%: ${var_99:.2f}", flush=True)
    print(f"    CVaR 95%: ${cvar_95:.2f}", flush=True)
    print(f"    CVaR 99%: ${cvar_99:.2f}", flush=True)
    results['var_95'] = round(float(var_95),2)
    results['var_99'] = round(float(var_99),2)
    results['cvar_95'] = round(float(cvar_95),2)
    results['cvar_99'] = round(float(cvar_99),2)

    # Monte Carlo: reshuffle daily returns, compute max DD distribution
    print(f"\n  --- Monte Carlo Max DD (10000 paths) ---", flush=True)
    np.random.seed(42)
    daily_returns = daily.values
    mc_max_dds = []
    for _ in range(10000):
        shuffled = np.random.permutation(daily_returns)
        eq = np.cumsum(shuffled)
        peak = np.maximum.accumulate(eq)
        mc_dd = (peak - eq).max()
        mc_max_dds.append(mc_dd)
    mc_max_dds = np.array(mc_max_dds)
    print(f"    Actual Max DD: ${_stats(all_trades)['max_dd']:.0f}", flush=True)
    print(f"    MC Mean Max DD: ${mc_max_dds.mean():.0f}", flush=True)
    print(f"    MC 95th pctl DD: ${np.percentile(mc_max_dds, 95):.0f}", flush=True)
    print(f"    MC 99th pctl DD: ${np.percentile(mc_max_dds, 99):.0f}", flush=True)
    results['mc_mean_dd'] = round(float(mc_max_dds.mean()),2)
    results['mc_95_dd'] = round(float(np.percentile(mc_max_dds, 95)),2)
    results['mc_99_dd'] = round(float(np.percentile(mc_max_dds, 99)),2)

    # Ruin probability (equity drops below X% of starting capital)
    print(f"\n  --- Ruin Probability ---", flush=True)
    eq = daily.cumsum()
    for ruin_pct in [20, 30, 50]:
        ruin_threshold = -CAPITAL * ruin_pct / 100  # e.g., -1000 for 20% of 5000
        # MC simulation
        ruin_count = 0
        for _ in range(10000):
            shuffled = np.random.permutation(daily_returns)
            cum = np.cumsum(shuffled)
            if cum.min() < ruin_threshold:
                ruin_count += 1
        ruin_prob = ruin_count / 10000 * 100
        print(f"    P(DD > {ruin_pct}% of $5000 = ${abs(ruin_threshold):.0f}): {ruin_prob:.2f}%", flush=True)
        results[f'ruin_{ruin_pct}pct'] = round(ruin_prob, 2)

    # Calmar ratio
    calmar = daily.sum() / _stats(all_trades)['max_dd'] if _stats(all_trades)['max_dd'] > 0 else 999
    print(f"\n    Calmar Ratio: {calmar:.2f}", flush=True)
    results['calmar'] = round(float(calmar), 2)

    with open(OUTPUT_DIR / "phase_10_stress.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Phase 10 done ({(time.time()-t0)/60:.1f}m)", flush=True)

# ═══════════════ MAIN ═══════════════
if __name__ == '__main__':
    print(f"{'='*100}")
    print(f"  R196 — 100-HOUR DEEP STRATEGY PARAMETER EXPLORATION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

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
    print(f"\n{'='*100}")
    print(f"  R196 COMPLETE — {total_m:.1f} minutes")
    print(f"{'='*100}")
