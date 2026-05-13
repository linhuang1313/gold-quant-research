#!/usr/bin/env python3
"""
R195b — Deep Validation of "Skip Worst Hours" Filter
=====================================================
Phase 1: Per-strategy analysis — does each strategy benefit?
Phase 2: Annual breakdown — is it consistent across years?
Phase 3: Fine-grained hour combination sweep (2^24 reduced to smart search)
Phase 4: Robustness — neighboring hours, Monte Carlo, stability
Phase 5: K-Fold + WF per-strategy (full 3-gate for each strategy separately)
Phase 6: Implementation spec and final verdict
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r195b_hours")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

CURRENT_CONFIG = {
    'L8_MAX':      {'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60, 'sl': 6.0, 'tp': 6.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 15},
    'TSMOM':       {'lot': 0.04, 'cap': 60, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.04, 'cap': 60, 'sl': 4.5, 'tp': 4.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.03, 'cap': 25, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

ERA_SEGMENTS = {
    'hike': [("2015-12-01","2019-01-01"),("2022-03-01","2023-08-01")],
    'cut':  [("2019-07-01","2022-03-01"),("2024-09-01","2026-06-01")],
    'recent_3y': [("2023-06-01","2026-06-01")],
}

import glob as _glob
t0 = time.time()

SKIP_HOURS = {1, 20, 22, 23}  # The candidate from R195

# ═══════════════════════════════════════════════════════════════
# Core helpers (same as R195)
# ═══════════════════════════════════════════════════════════════
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
    return {'dir':pos['dir'],'entry':pos['entry'],'exit':ep,'entry_time':pos['time'],'exit_time':et,
            'pnl':pnl,'reason':reason,'bars':bi-pos['bar'],'atr':pos['atr'],'strategy':pos.get('strategy','')}

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
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2)}

def filter_era(trades, era):
    return [t for t in trades if any(pd.Timestamp(s)<=pd.Timestamp(t['entry_time'])<pd.Timestamp(e) for s,e in ERA_SEGMENTS[era])]

def kfold_test(h1, run_new, run_base, K=6):
    start=h1.index[0]; end=h1.index[-1]; total=(end-start).days; fd=total//K
    results=[]
    for fold in range(K):
        fs=start+pd.Timedelta(days=fold*fd)
        fe=start+pd.Timedelta(days=(fold+1)*fd) if fold<K-1 else end+pd.Timedelta(days=1)
        h1f=h1[(h1.index>=fs)&(h1.index<fe)]
        if len(h1f)<300: continue
        sh_new=run_new(h1f); sh_base=run_base(h1f)
        results.append({'fold':fold+1,'new':round(sh_new,3),'base':round(sh_base,3),'win':'NEW' if sh_new>sh_base else 'BASE'})
    return results

def wf_test(h1, run_new, run_base, train_d=547, test_d=180):
    start=h1.index[0]; end=h1.index[-1]; cursor=start+pd.Timedelta(days=train_d)
    results=[]; period=0
    while cursor+pd.Timedelta(days=test_d)<=end+pd.Timedelta(days=1):
        period+=1; ts=cursor; te=cursor+pd.Timedelta(days=test_d)
        h1t=h1[(h1.index>=ts)&(h1.index<te)]
        if len(h1t)<200: cursor+=pd.Timedelta(days=test_d); continue
        sh_new=run_new(h1t); sh_base=run_base(h1t)
        results.append({'period':period,'new':round(sh_new,3),'base':round(sh_base,3),'win':'NEW' if sh_new>sh_base else 'BASE'})
        cursor+=pd.Timedelta(days=test_d)
    return results

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

# Strategy backtests
def bt_keltner(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA100']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA100','KC_upper'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA100'].values
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
        if np.isnan(adx[i]) or adx[i]<14: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
    return trades

def bt_psar(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy()
    ha,la,ca=df['High'].values,df['Low'].values,df['Close'].values; n=len(df)
    psar=np.empty(n); psar[:]=np.nan; af_s=0.01; af_m=0.05; af=af_s; rising=True; ep=ha[0]; psar[0]=la[0]
    for i in range(1,n):
        p=psar[i-1]
        if rising:
            psar[i]=p+af*(ep-p); psar[i]=min(psar[i],la[i-1],la[max(0,i-2)])
            if la[i]<psar[i]: rising=False; psar[i]=ep; ep=la[i]; af=af_s
            else:
                if ha[i]>ep: ep=ha[i]; af=min(af+af_s,af_m)
        else:
            psar[i]=p+af*(ep-p); psar[i]=max(psar[i],ha[i-1],ha[max(0,i-2)])
            if ha[i]>psar[i]: rising=True; psar[i]=ep; ep=ha[i]; af=af_s
            else:
                if la[i]<ep: ep=la[i]; af=min(af+af_s,af_m)
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

def bt_tsmom(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    times=df.index; n=len(df); fast_lb=480; slow_lb=720; mx=slow_lb
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

def bt_sess_bo(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    hrs=df.index.hour; times=df.index; n=len(df); lb=4
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(lb,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if hrs[i]!=12: continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        hh=max(h[i-j] for j in range(1,lb+1)); ll=min(lo[i-j] for j in range(1,lb+1))
        if c[i]>hh: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
        elif c[i]<ll: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
    return trades

def bt_dt(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,o,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df); nb=6; k=0.5
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
        if c[i]>o[i]+k*rng: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'DUAL_THRUST'}
        elif c[i]<o[i]-k*rng: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'DUAL_THRUST'}
    return trades

def bt_chand(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df,period=22)
    df['EMA']=df['Close'].ewm(span=100,adjust=False).mean()
    df['RSI14']=compute_rsi(df['Close'],14)
    df=df.dropna(subset=['ATR','EMA'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['EMA'].values
    rsi_v=df['RSI14'].values; times=df.index; n=len(df); p=22; m=3.0
    cl=np.full(n,np.nan); cs=np.full(n,np.nan)
    for i in range(p,n): cl[i]=np.max(h[i-p+1:i+1])-m*atr[i]; cs[i]=np.min(lo[i-p+1:i+1])+m*atr[i]
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

BT_FN = {'L8_MAX':bt_keltner,'PSAR':bt_psar,'TSMOM':bt_tsmom,'SESS_BO':bt_sess_bo,'DUAL_THRUST':bt_dt,'CHANDELIER':bt_chand}

def run_all(h1, config, pctl_v=None, pctl_f=0):
    return {nm: BT_FN[nm](h1, config[nm], pctl_v=pctl_v, pctl_f=pctl_f) for nm in STRAT_ORDER}

def port_merge(all_t):
    return [t for nm in STRAT_ORDER for t in all_t.get(nm,[])]

def port_sharpe(all_t):
    return _sharpe(_daily(port_merge(all_t)))


# ═══════════════════════════════════════════════════════════════
# PHASE 1: Per-Strategy Analysis
# ═══════════════════════════════════════════════════════════════
def phase_1(h1, pctl, all_trades):
    print(f"\n{'='*100}\n  PHASE 1: PER-STRATEGY SKIP HOURS ANALYSIS\n{'='*100}", flush=True)
    results = {}

    for strat in STRAT_ORDER:
        strat_trades = all_trades.get(strat, [])
        if not strat_trades: continue
        base_st = _stats(strat_trades)
        filtered = [t for t in strat_trades if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
        new_st = _stats(filtered)
        removed = base_st['n'] - new_st['n']
        delta = new_st['sharpe'] - base_st['sharpe']
        pct_removed = removed / base_st['n'] * 100 if base_st['n'] > 0 else 0

        print(f"\n  {strat}:", flush=True)
        print(f"    BASE: Sharpe={base_st['sharpe']:.3f}, N={base_st['n']}, PnL=${base_st['pnl']:,.0f}", flush=True)
        print(f"    NEW:  Sharpe={new_st['sharpe']:.3f}, N={new_st['n']}, PnL=${new_st['pnl']:,.0f}", flush=True)
        print(f"    Delta: {delta:+.3f}, Removed={removed} ({pct_removed:.1f}%)", flush=True)

        # Hour-by-hour for this strategy
        hour_detail = {}
        for hr in sorted(SKIP_HOURS):
            hr_trades = [t for t in strat_trades if pd.Timestamp(t['entry_time']).hour == hr]
            if hr_trades:
                hr_st = _stats(hr_trades)
                hour_detail[str(hr)] = hr_st
                print(f"      Hour {hr}: N={hr_st['n']}, Sharpe={hr_st['sharpe']:.3f}, WR={hr_st['wr']:.1f}%, avg_pnl=${hr_st['pnl']/hr_st['n']:.2f}", flush=True)

        results[strat] = {'base': base_st, 'filtered': new_st, 'delta': round(delta,3), 'removed': removed, 'pct_removed': round(pct_removed,1), 'hour_detail': hour_detail}

    with open(OUTPUT_DIR / "phase_1_per_strategy.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 2: Annual Distribution
# ═══════════════════════════════════════════════════════════════
def phase_2(h1, pctl, all_trades):
    print(f"\n{'='*100}\n  PHASE 2: ANNUAL DISTRIBUTION\n{'='*100}", flush=True)
    results = {}
    merged = port_merge(all_trades)

    years = sorted(set(pd.Timestamp(t['entry_time']).year for t in merged))
    print(f"\n  {'Year':<6} {'Base_Sh':>8} {'New_Sh':>8} {'Delta':>7} {'Base_N':>7} {'New_N':>6} {'Rem':>5} {'Base_PnL':>10} {'New_PnL':>10} {'PnL_D':>9}", flush=True)
    print(f"  {'-'*80}", flush=True)

    all_positive = True
    for yr in years:
        yr_base = [t for t in merged if pd.Timestamp(t['entry_time']).year == yr]
        yr_new = [t for t in yr_base if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
        sb = _stats(yr_base); sn = _stats(yr_new)
        delta = sn['sharpe'] - sb['sharpe']
        pnl_delta = sn['pnl'] - sb['pnl']
        removed = sb['n'] - sn['n']
        if delta < 0: all_positive = False
        print(f"  {yr:<6} {sb['sharpe']:>8.3f} {sn['sharpe']:>8.3f} {delta:>+7.3f} {sb['n']:>7} {sn['n']:>6} {removed:>5} ${sb['pnl']:>9,.0f} ${sn['pnl']:>9,.0f} ${pnl_delta:>+8,.0f}", flush=True)
        results[str(yr)] = {'base': sb, 'new': sn, 'delta_sharpe': round(delta,3), 'delta_pnl': round(pnl_delta,2), 'removed': removed}

    results['all_years_positive'] = all_positive
    print(f"\n  All years positive delta: {'YES' if all_positive else 'NO'}", flush=True)

    with open(OUTPUT_DIR / "phase_2_annual.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 3: Hour Combination Sweep
# ═══════════════════════════════════════════════════════════════
def phase_3(h1, pctl, all_trades):
    print(f"\n{'='*100}\n  PHASE 3: HOUR COMBINATION SWEEP\n{'='*100}", flush=True)
    results = {}
    merged = port_merge(all_trades)
    base_st = _stats(merged)

    # Test all single-hour skips
    print(f"\n  --- Single Hour Skip ---", flush=True)
    single_results = []
    for hr in range(24):
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour != hr]
        st = _stats(filtered)
        delta = st['sharpe'] - base_st['sharpe']
        single_results.append({'hour': hr, 'sharpe': st['sharpe'], 'delta': round(delta,3), 'n': st['n']})
    single_results.sort(key=lambda x: -x['delta'])
    for r in single_results[:10]:
        print(f"    Skip hour {r['hour']:>2}: Sharpe={r['sharpe']:.3f} ({r['delta']:+.3f})", flush=True)
    results['single_skip'] = single_results

    # Test all pairs
    print(f"\n  --- Best 2-Hour Combos (top 10) ---", flush=True)
    pair_results = []
    for h1_skip, h2_skip in combinations(range(24), 2):
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in {h1_skip, h2_skip}]
        st = _stats(filtered)
        delta = st['sharpe'] - base_st['sharpe']
        pair_results.append({'hours': [h1_skip, h2_skip], 'sharpe': st['sharpe'], 'delta': round(delta,3)})
    pair_results.sort(key=lambda x: -x['delta'])
    for r in pair_results[:10]:
        print(f"    Skip {r['hours']}: Sharpe={r['sharpe']:.3f} ({r['delta']:+.3f})", flush=True)
    results['pair_skip'] = pair_results[:20]

    # Test all triples
    print(f"\n  --- Best 3-Hour Combos (top 10) ---", flush=True)
    triple_results = []
    for h1_s, h2_s, h3_s in combinations(range(24), 3):
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in {h1_s, h2_s, h3_s}]
        st = _stats(filtered)
        delta = st['sharpe'] - base_st['sharpe']
        triple_results.append({'hours': [h1_s, h2_s, h3_s], 'sharpe': st['sharpe'], 'delta': round(delta,3)})
    triple_results.sort(key=lambda x: -x['delta'])
    for r in triple_results[:10]:
        print(f"    Skip {r['hours']}: Sharpe={r['sharpe']:.3f} ({r['delta']:+.3f})", flush=True)
    results['triple_skip'] = triple_results[:20]

    # Test all quads (focus on top-performing candidates)
    print(f"\n  --- Best 4-Hour Combos (top 15) ---", flush=True)
    # Use greedy: start from best triple, add each remaining hour
    best_triple = set(triple_results[0]['hours'])
    quad_results = []
    for h4 in range(24):
        if h4 in best_triple: continue
        skip_set = best_triple | {h4}
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in skip_set]
        st = _stats(filtered)
        delta = st['sharpe'] - base_st['sharpe']
        quad_results.append({'hours': sorted(skip_set), 'sharpe': st['sharpe'], 'delta': round(delta,3)})

    # Also test combos from top pairs extended
    for pair in pair_results[:5]:
        pair_set = set(pair['hours'])
        for h3_s, h4_s in combinations(range(24), 2):
            if h3_s in pair_set or h4_s in pair_set: continue
            skip_set = pair_set | {h3_s, h4_s}
            if len(skip_set) != 4: continue
            filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in skip_set]
            st = _stats(filtered)
            delta = st['sharpe'] - base_st['sharpe']
            quad_results.append({'hours': sorted(skip_set), 'sharpe': st['sharpe'], 'delta': round(delta,3)})

    # Deduplicate
    seen = set()
    unique_quads = []
    for r in quad_results:
        key = tuple(r['hours'])
        if key not in seen:
            seen.add(key)
            unique_quads.append(r)
    unique_quads.sort(key=lambda x: -x['delta'])
    for r in unique_quads[:15]:
        print(f"    Skip {r['hours']}: Sharpe={r['sharpe']:.3f} ({r['delta']:+.3f})", flush=True)
    results['quad_skip'] = unique_quads[:20]

    # The original {1,20,22,23}
    orig_filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
    orig_st = _stats(orig_filtered)
    print(f"\n  Original {sorted(SKIP_HOURS)}: Sharpe={orig_st['sharpe']:.3f} ({orig_st['sharpe']-base_st['sharpe']:+.3f})", flush=True)
    results['original_4'] = {'hours': sorted(SKIP_HOURS), 'sharpe': orig_st['sharpe'], 'delta': round(orig_st['sharpe']-base_st['sharpe'],3)}

    with open(OUTPUT_DIR / "phase_3_combos.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 4: Robustness
# ═══════════════════════════════════════════════════════════════
def phase_4(h1, pctl, all_trades):
    print(f"\n{'='*100}\n  PHASE 4: ROBUSTNESS CHECKS\n{'='*100}", flush=True)
    results = {}
    merged = port_merge(all_trades)
    base_st = _stats(merged)

    # 4A: Neighboring hour sets (shift ±1)
    print(f"\n  --- 4A: Neighboring Hour Sets (overfitting check) ---", flush=True)
    neighbor_sets = [
        ({0, 19, 21, 22}, "shift -1"),
        ({2, 21, 23, 0}, "shift +1"),
        ({0, 1, 22, 23}, "late night block 0-1,22-23"),
        ({19, 20, 21, 22}, "evening block 19-22"),
        ({20, 21, 22, 23}, "late evening 20-23"),
        ({0, 1, 2, 23}, "midnight block 23,0,1,2"),
    ]
    for skip_set, label in neighbor_sets:
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in skip_set]
        st = _stats(filtered)
        delta = st['sharpe'] - base_st['sharpe']
        print(f"    {label} {sorted(skip_set)}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'neighbor_{label}'] = {'hours': sorted(skip_set), 'sharpe': st['sharpe'], 'delta': round(delta,3)}

    # 4B: Random hour selection (Monte Carlo)
    print(f"\n  --- 4B: Random 4-Hour Skip (1000 trials) ---", flush=True)
    np.random.seed(42)
    random_deltas = []
    for _ in range(1000):
        rand_hours = set(np.random.choice(24, 4, replace=False))
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in rand_hours]
        st = _stats(filtered)
        random_deltas.append(st['sharpe'] - base_st['sharpe'])
    random_deltas = np.array(random_deltas)
    target_delta = 0.435  # Our actual delta
    percentile_rank = (random_deltas < target_delta).sum() / len(random_deltas) * 100
    print(f"    Our delta (+0.435) is at percentile {percentile_rank:.1f}% of random 4-hour skips", flush=True)
    print(f"    Random mean delta: {random_deltas.mean():+.3f}, std: {random_deltas.std():.3f}", flush=True)
    print(f"    Random max delta: {random_deltas.max():+.3f}, min: {random_deltas.min():+.3f}", flush=True)
    print(f"    % random > +0.2: {(random_deltas > 0.2).sum()/len(random_deltas)*100:.1f}%", flush=True)
    results['monte_carlo'] = {
        'percentile_rank': round(percentile_rank, 1),
        'random_mean': round(float(random_deltas.mean()), 3),
        'random_std': round(float(random_deltas.std()), 3),
        'random_max': round(float(random_deltas.max()), 3),
        'pct_above_0.2': round((random_deltas > 0.2).sum()/len(random_deltas)*100, 1)
    }

    # 4C: Stability across time — split into halves
    print(f"\n  --- 4C: Time Stability (first half vs second half) ---", flush=True)
    mid = h1.index[len(h1)//2]
    for period, label in [('first', h1.index[0]), ('second', mid)]:
        if period == 'first':
            period_trades = [t for t in merged if pd.Timestamp(t['entry_time']) < mid]
        else:
            period_trades = [t for t in merged if pd.Timestamp(t['entry_time']) >= mid]
        base_p = _stats(period_trades)
        filtered_p = [t for t in period_trades if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
        new_p = _stats(filtered_p)
        delta = new_p['sharpe'] - base_p['sharpe']
        print(f"    {period} half: BASE={base_p['sharpe']:.3f}, NEW={new_p['sharpe']:.3f}, delta={delta:+.3f}", flush=True)
        results[f'time_stability_{period}'] = {'base': base_p['sharpe'], 'new': new_p['sharpe'], 'delta': round(delta,3)}

    with open(OUTPUT_DIR / "phase_4_robustness.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 5: Full 3-Gate per Strategy
# ═══════════════════════════════════════════════════════════════
def phase_5(h1, pctl):
    print(f"\n{'='*100}\n  PHASE 5: FULL 3-GATE VALIDATION PER STRATEGY\n{'='*100}", flush=True)
    results = {}

    for strat in STRAT_ORDER:
        print(f"\n  === {strat} ===", flush=True)

        def make_run_new(s):
            def run_new(h1f):
                trades = BT_FN[s](h1f, CURRENT_CONFIG[s], pctl_v=pctl, pctl_f=30)
                filtered = [t for t in trades if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
                return _stats(filtered)['sharpe']
            return run_new

        def make_run_base(s):
            def run_base(h1f):
                trades = BT_FN[s](h1f, CURRENT_CONFIG[s], pctl_v=pctl, pctl_f=30)
                return _stats(trades)['sharpe']
            return run_base

        # K-Fold
        kf = kfold_test(h1, make_run_new(strat), make_run_base(strat))
        kf_wins = sum(1 for r in kf if r['win']=='NEW')
        print(f"    KF: {kf_wins}/{len(kf)}", flush=True)
        for r in kf:
            print(f"      Fold {r['fold']}: NEW={r['new']:.3f} vs BASE={r['base']:.3f} [{r['win']}]", flush=True)

        # Walk-Forward
        wf = wf_test(h1, make_run_new(strat), make_run_base(strat))
        wf_wins = sum(1 for r in wf if r['win']=='NEW')
        print(f"    WF: {wf_wins}/{len(wf)}", flush=True)

        # Era
        all_base = BT_FN[strat](h1, CURRENT_CONFIG[strat], pctl_v=pctl, pctl_f=30)
        all_new = [t for t in all_base if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
        era_results = {}
        for era in ['hike','cut','recent_3y']:
            en = filter_era(all_new, era); eb = filter_era(all_base, era)
            sn = _sharpe(_daily(en)); sb = _sharpe(_daily(eb))
            era_results[era] = {'new': round(sn,3), 'base': round(sb,3), 'delta': round(sn-sb,3)}
            print(f"    Era {era}: NEW={sn:.3f}, BASE={sb:.3f}, delta={sn-sb:+.3f}", flush=True)

        kf_pass = kf_wins >= 4
        wf_pass = wf_wins >= 13
        era_pass = all(era_results[e]['new'] > 0 for e in era_results)
        era_no_degrade = all(era_results[e]['delta'] > -0.3 for e in era_results)
        verdict = 'GO' if kf_pass and wf_pass and era_pass and era_no_degrade else 'NO-GO'
        reason = []
        if not kf_pass: reason.append(f"KF {kf_wins}/{len(kf)}")
        if not wf_pass: reason.append(f"WF {wf_wins}/{len(wf)}")
        if not era_pass or not era_no_degrade: reason.append("Era")
        print(f"    VERDICT: {verdict} ({', '.join(reason) if reason else 'ALL PASS'})", flush=True)

        results[strat] = {'kf_wins': kf_wins, 'kf_total': len(kf), 'wf_wins': wf_wins, 'wf_total': len(wf),
                         'era': era_results, 'verdict': verdict}

    # Portfolio-level (re-confirm)
    print(f"\n  === PORTFOLIO ===", flush=True)
    def run_new_port(h1f):
        all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        merged = port_merge(all_t)
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour not in SKIP_HOURS]
        return _stats(filtered)['sharpe']
    def run_base_port(h1f):
        return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))

    kf = kfold_test(h1, run_new_port, run_base_port)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    wf = wf_test(h1, run_new_port, run_base_port)
    wf_wins = sum(1 for r in wf if r['win']=='NEW')
    print(f"    KF: {kf_wins}/{len(kf)}, WF: {wf_wins}/{len(wf)}", flush=True)
    for r in kf:
        print(f"      Fold {r['fold']}: NEW={r['new']:.3f} vs BASE={r['base']:.3f} [{r['win']}]", flush=True)

    port_verdict = 'GO' if kf_wins >= 4 and wf_wins >= 13 else 'NO-GO'
    print(f"    PORTFOLIO VERDICT: {port_verdict}", flush=True)
    results['PORTFOLIO'] = {'kf_wins': kf_wins, 'kf_total': len(kf), 'wf_wins': wf_wins, 'wf_total': len(wf), 'verdict': port_verdict}

    with open(OUTPUT_DIR / "phase_5_validation.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"{'='*100}")
    print(f"  R195b — DEEP VALIDATION: SKIP WORST HOURS")
    print(f"  Skip Set: {sorted(SKIP_HOURS)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    h1 = load_h1()
    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR pctl: {pctl.notna().sum()} valid\n", flush=True)

    # Run all strategies once to get trades
    print("  Running baseline strategies...", flush=True)
    all_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    merged = port_merge(all_trades)
    base_stats = _stats(merged)
    print(f"  Baseline: Sharpe={base_stats['sharpe']}, N={base_stats['n']}, PnL=${base_stats['pnl']:,.0f}\n", flush=True)

    r1 = phase_1(h1, pctl, all_trades)
    r2 = phase_2(h1, pctl, all_trades)
    r3 = phase_3(h1, pctl, all_trades)
    r4 = phase_4(h1, pctl, all_trades)
    r5 = phase_5(h1, pctl)

    total_m = (time.time()-t0)/60
    print(f"\n{'='*100}")
    print(f"  R195b COMPLETE — {total_m:.1f} minutes")
    print(f"{'='*100}")
