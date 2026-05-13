#!/usr/bin/env python3
"""
R194b — Deep Validation of R194 Top Findings
=============================================
Phase 1: COT |Z| > threshold skip — K-Fold/WF/Era validation (multiple thresholds)
Phase 2: Bearish macro regime skip — K-Fold/WF/Era validation
Phase 3: VIX spike pause — K-Fold/WF/Era validation
Phase 4: Combined COT + Macro filter optimization
Phase 5: Monte Carlo robustness (parameter perturbation + bootstrap)
Phase 6: Implementation feasibility (lookback sensitivity, cold start, lag)

Validation standard: K-Fold >= 4/6, WF >= 13/19, Era all positive & no degrade > 0.3
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

OUTPUT_DIR = Path("results/r194b_validation")
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
    'full': None,
    'hike': [("2015-12-01","2019-01-01"),("2022-03-01","2023-08-01")],
    'cut':  [("2019-07-01","2022-03-01"),("2024-09-01","2026-06-01")],
    'recent_3y': [("2023-06-01","2026-06-01")],
}

import glob as _glob
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Core helpers (from R194)
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
    if era=='full' or ERA_SEGMENTS[era] is None: return trades
    return [t for t in trades if any(pd.Timestamp(s)<=pd.Timestamp(t['entry_time'])<pd.Timestamp(e) for s,e in ERA_SEGMENTS[era])]

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

def load_cot():
    fp = "data/cot_gold_weekly.csv"
    if not os.path.exists(fp): fp = "data/external/cot_gold_weekly.csv"
    df = pd.read_csv(fp, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    print(f"  COT: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df

def load_macro():
    fp = "data/external/aligned_daily.csv"
    df = pd.read_csv(fp, parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    print(f"  Macro: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df

# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════
def bt_keltner(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA100']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA100','KC_upper'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
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
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): continue
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
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
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
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): prev=cur; continue
        if cur and not prev: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        elif not cur and prev: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        prev=cur
    return trades

def bt_tsmom(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
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
        if pv_arr is not None and (np.isnan(pv_arr[i]) or pv_arr[i]<pctl_f): continue
        if np.isnan(sc[i]) or np.isnan(sc[i-1]): continue
        if sc[i]>0 and sc[i-1]<=0: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'TSMOM'}
        elif sc[i]<0 and sc[i-1]>=0: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'TSMOM'}
    return trades

def bt_sess_bo(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
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
        if pv_arr is not None and (np.isnan(pv_arr[i]) or pv_arr[i]<pctl_f): continue
        hh=max(h[i-j] for j in range(1,lb+1)); ll=min(lo[i-j] for j in range(1,lb+1))
        if c[i]>hh: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
        elif c[i]<ll: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
    return trades

def bt_dt(h1, cfg, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
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
        if pv_arr is not None and (np.isnan(pv_arr[i]) or pv_arr[i]<pctl_f): continue
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
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
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
        if pv_arr is not None and (np.isnan(pv_arr[i]) or pv_arr[i]<pctl_f): continue
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
# Validation framework
# ═══════════════════════════════════════════════════════════════
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

def full_validate(h1, pctl, run_new_fn, run_base_fn, trades_new, trades_base, label):
    """Run full 3-gate validation"""
    print(f"\n    --- {label}: K-Fold ---", flush=True)
    kf = kfold_test(h1, run_new_fn, run_base_fn)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    print(f"      K-Fold: {kf_wins}/{len(kf)}", flush=True)
    for r in kf:
        print(f"        Fold {r['fold']}: NEW={r['new']:.3f} vs BASE={r['base']:.3f} [{r['win']}]", flush=True)

    print(f"\n    --- {label}: Walk-Forward ---", flush=True)
    wf = wf_test(h1, run_new_fn, run_base_fn)
    wf_wins = sum(1 for r in wf if r['win']=='NEW')
    print(f"      WF: {wf_wins}/{len(wf)}", flush=True)

    print(f"\n    --- {label}: Era ---", flush=True)
    era_results = {}
    for era in ['hike','cut','recent_3y']:
        en = filter_era(trades_new, era); eb = filter_era(trades_base, era)
        sn = _sharpe(_daily(en)); sb = _sharpe(_daily(eb))
        delta = sn - sb
        era_results[era] = {'new': round(sn,3), 'base': round(sb,3), 'delta': round(delta,3)}
        print(f"      {era}: NEW={sn:.3f}, BASE={sb:.3f}, delta={delta:+.3f}", flush=True)

    kf_pass = kf_wins >= 4
    wf_pass = wf_wins >= 13
    era_pass = all(era_results[e]['new'] > 0 for e in era_results)
    era_no_degrade = all(era_results[e]['delta'] > -0.3 for e in era_results)
    all_pass = kf_pass and wf_pass and era_pass and era_no_degrade
    verdict_str = 'GO' if all_pass else 'NO-GO'

    print(f"\n    VERDICT: {verdict_str} (KF={kf_wins}/{len(kf)} {'PASS' if kf_pass else 'FAIL'}, WF={wf_wins}/{len(wf)} {'PASS' if wf_pass else 'FAIL'}, Era={'PASS' if era_pass and era_no_degrade else 'FAIL'})", flush=True)

    return {
        'kf_wins': kf_wins, 'kf_total': len(kf), 'kf_pass': kf_pass,
        'wf_wins': wf_wins, 'wf_total': len(wf), 'wf_pass': wf_pass,
        'era': era_results, 'era_pass': era_pass, 'era_no_degrade': era_no_degrade,
        'verdict': verdict_str
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 1: COT |Z| Skip — Rigorous Validation
# ═══════════════════════════════════════════════════════════════
def phase_1(h1, pctl, cot):
    print(f"\n{'='*120}\n  PHASE 1: COT |Z| SKIP — RIGOROUS VALIDATION\n{'='*120}", flush=True)
    results = {}

    # Prepare COT Z-score
    if 'net_spec' in cot.columns:
        cot['net_spec_z52'] = (cot['net_spec'] - cot['net_spec'].rolling(52).mean()) / cot['net_spec'].rolling(52).std()
    cot_z = cot['net_spec_z52'].reindex(h1.index, method='ffill') if 'net_spec_z52' in cot.columns else None

    if cot_z is None:
        print("  ERROR: Cannot compute COT Z-score", flush=True)
        results['error'] = 'no net_spec column'
        with open(OUTPUT_DIR / "phase_1_cot.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Baseline
    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    print(f"  Baseline: Sharpe={base_stats['sharpe']:.3f}, PnL=${base_stats['pnl']:,.0f}, N={base_stats['n']}", flush=True)
    results['baseline'] = base_stats

    # Test multiple thresholds
    for z_thresh in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        print(f"\n  === COT |Z| > {z_thresh} Skip ===", flush=True)
        filtered = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in cot_z.index and
                           not np.isnan(cot_z.loc[pd.Timestamp(t['entry_time'])]) and
                           abs(cot_z.loc[pd.Timestamp(t['entry_time'])]) > z_thresh)]
        stats_new = _stats(filtered)
        removed = base_stats['n'] - stats_new['n']
        delta = stats_new['sharpe'] - base_stats['sharpe']
        print(f"    Sharpe={stats_new['sharpe']:.3f} ({delta:+.3f}), N={stats_new['n']}, Removed={removed}", flush=True)

        # Full validation
        def make_run_new(thresh):
            def run_new(h1f):
                all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                merged = port_merge(all_t)
                filtered_f = [t for t in merged
                             if not (pd.Timestamp(t['entry_time']) in cot_z.index and
                                    not np.isnan(cot_z.loc[pd.Timestamp(t['entry_time'])]) and
                                    abs(cot_z.loc[pd.Timestamp(t['entry_time'])]) > thresh)]
                return _stats(filtered_f)['sharpe']
            return run_new

        def run_base(h1f):
            all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
            return port_sharpe(all_t)

        validation = full_validate(h1, pctl, make_run_new(z_thresh), run_base, filtered, base_merged, f"COT_Z>{z_thresh}")
        results[f'z_{z_thresh}'] = {'stats': stats_new, 'removed': removed, 'validation': validation}

    with open(OUTPUT_DIR / "phase_1_cot.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 1 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 2: Bearish Macro Regime Skip
# ═══════════════════════════════════════════════════════════════
def phase_2(h1, pctl, macro):
    print(f"\n{'='*120}\n  PHASE 2: BEARISH MACRO REGIME SKIP\n{'='*120}", flush=True)
    results = {}

    def detect_regimes(macro_df, vix_thresh=1.0):
        regimes = pd.Series('neutral', index=macro_df.index)
        dxy_mom = macro_df.get('DXY_Mom20', pd.Series(0, index=macro_df.index))
        vix_z = macro_df.get('VIX_Zscore', pd.Series(0, index=macro_df.index))
        curve = macro_df.get('YIELD_CURVE_10Y2Y', pd.Series(0, index=macro_df.index))
        credit = macro_df.get('CREDIT_STRESS', pd.Series(0, index=macro_df.index))
        bull_mask = (dxy_mom < 0) & (vix_z < 0) & (curve > 0)
        bear_mask = (dxy_mom > 0) & (vix_z > vix_thresh) & (credit > 0)
        regimes[bull_mask] = 'bullish'
        regimes[bear_mask] = 'bearish'
        return regimes

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    results['baseline'] = base_stats

    # Test different VIX thresholds for bear definition
    for vix_thresh in [0.5, 0.75, 1.0, 1.5, 2.0]:
        print(f"\n  === Bearish regime (VIX_thresh={vix_thresh}) Skip ===", flush=True)
        regimes = detect_regimes(macro, vix_thresh)
        regime_aligned = regimes.reindex(h1.index, method='ffill')

        filtered = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in regime_aligned.index and
                           regime_aligned.loc[pd.Timestamp(t['entry_time'])] == 'bearish')]
        stats_new = _stats(filtered)
        removed = base_stats['n'] - stats_new['n']
        delta = stats_new['sharpe'] - base_stats['sharpe']
        print(f"    Sharpe={stats_new['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)

        def make_run_new(vt):
            def run_new(h1f):
                all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                merged = port_merge(all_t)
                regs = detect_regimes(macro, vt).reindex(h1f.index, method='ffill')
                f = [t for t in merged
                     if not (pd.Timestamp(t['entry_time']) in regs.index and regs.loc[pd.Timestamp(t['entry_time'])] == 'bearish')]
                return _stats(f)['sharpe']
            return run_new

        def run_base(h1f):
            return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))

        validation = full_validate(h1, pctl, make_run_new(vix_thresh), run_base, filtered, base_merged, f"Bear_VIX>{vix_thresh}")
        results[f'vix_{vix_thresh}'] = {'stats': stats_new, 'removed': removed, 'validation': validation}

    with open(OUTPUT_DIR / "phase_2_macro.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 2 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 3: VIX Spike Pause
# ═══════════════════════════════════════════════════════════════
def phase_3(h1, pctl, macro):
    print(f"\n{'='*120}\n  PHASE 3: VIX SPIKE PAUSE\n{'='*120}", flush=True)
    results = {}

    vix_z = macro.get('VIX_Zscore', pd.Series(dtype=float)).reindex(h1.index, method='ffill')

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    results['baseline'] = base_stats

    for vix_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        print(f"\n  === VIX Z > {vix_thresh} Pause ===", flush=True)
        filtered = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in vix_z.index and
                           not np.isnan(vix_z.loc[pd.Timestamp(t['entry_time'])]) and
                           vix_z.loc[pd.Timestamp(t['entry_time'])] > vix_thresh)]
        stats_new = _stats(filtered)
        removed = base_stats['n'] - stats_new['n']
        delta = stats_new['sharpe'] - base_stats['sharpe']
        print(f"    Sharpe={stats_new['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)

        if delta > 0.05:
            def make_run_new(vt):
                def run_new(h1f):
                    all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                    merged = port_merge(all_t)
                    f = [t for t in merged
                         if not (pd.Timestamp(t['entry_time']) in vix_z.index and
                                not np.isnan(vix_z.loc[pd.Timestamp(t['entry_time'])]) and
                                vix_z.loc[pd.Timestamp(t['entry_time'])] > vt)]
                    return _stats(f)['sharpe']
                return run_new
            def run_base(h1f):
                return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))

            validation = full_validate(h1, pctl, make_run_new(vix_thresh), run_base, filtered, base_merged, f"VIX>{vix_thresh}")
            results[f'vix_{vix_thresh}'] = {'stats': stats_new, 'removed': removed, 'validation': validation}
        else:
            results[f'vix_{vix_thresh}'] = {'stats': stats_new, 'removed': removed, 'validation': {'verdict': 'SKIP (delta too small)'}}

    with open(OUTPUT_DIR / "phase_3_vix.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 3 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 4: Combined COT + Macro
# ═══════════════════════════════════════════════════════════════
def phase_4(h1, pctl, cot, macro):
    print(f"\n{'='*120}\n  PHASE 4: COMBINED COT + MACRO FILTER\n{'='*120}", flush=True)
    results = {}

    # Prepare signals
    if 'net_spec' in cot.columns:
        cot['net_spec_z52'] = (cot['net_spec'] - cot['net_spec'].rolling(52).mean()) / cot['net_spec'].rolling(52).std()
    cot_z = cot.get('net_spec_z52', pd.Series(dtype=float)).reindex(h1.index, method='ffill')
    vix_z = macro.get('VIX_Zscore', pd.Series(dtype=float)).reindex(h1.index, method='ffill')

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    results['baseline'] = base_stats

    # Combo: COT |Z| > X OR VIX Z > Y
    combos = [
        (1.5, 2.0, 'COT1.5_VIX2.0'),
        (1.5, 1.5, 'COT1.5_VIX1.5'),
        (2.0, 2.0, 'COT2.0_VIX2.0'),
        (1.25, 2.0, 'COT1.25_VIX2.0'),
        (1.5, 2.5, 'COT1.5_VIX2.5'),
    ]

    for cot_thresh, vix_thresh, label in combos:
        print(f"\n  === Combo: {label} (skip if |COT_Z|>{cot_thresh} OR VIX_Z>{vix_thresh}) ===", flush=True)
        filtered = []
        for t in base_merged:
            et = pd.Timestamp(t['entry_time'])
            skip = False
            if et in cot_z.index:
                cz = cot_z.loc[et]
                if not np.isnan(cz) and abs(cz) > cot_thresh: skip = True
            if not skip and et in vix_z.index:
                vz = vix_z.loc[et]
                if not np.isnan(vz) and vz > vix_thresh: skip = True
            if not skip:
                filtered.append(t)

        stats_new = _stats(filtered)
        removed = base_stats['n'] - stats_new['n']
        delta = stats_new['sharpe'] - base_stats['sharpe']
        print(f"    Sharpe={stats_new['sharpe']:.3f} ({delta:+.3f}), Removed={removed} ({removed/base_stats['n']*100:.1f}%)", flush=True)

        # Full validation for best combo
        def make_combo_run(ct, vt):
            def run_new(h1f):
                all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                merged = port_merge(all_t)
                f = []
                for t in merged:
                    et = pd.Timestamp(t['entry_time'])
                    skip = False
                    if et in cot_z.index:
                        cz = cot_z.loc[et]
                        if not np.isnan(cz) and abs(cz) > ct: skip = True
                    if not skip and et in vix_z.index:
                        vz = vix_z.loc[et]
                        if not np.isnan(vz) and vz > vt: skip = True
                    if not skip: f.append(t)
                return _stats(f)['sharpe']
            return run_new

        def run_base(h1f):
            return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))

        validation = full_validate(h1, pctl, make_combo_run(cot_thresh, vix_thresh), run_base, filtered, base_merged, label)
        results[label] = {'stats': stats_new, 'removed': removed, 'validation': validation}

    with open(OUTPUT_DIR / "phase_4_combo.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 4 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 5: Monte Carlo Robustness
# ═══════════════════════════════════════════════════════════════
def phase_5(h1, pctl, cot):
    print(f"\n{'='*120}\n  PHASE 5: MONTE CARLO ROBUSTNESS\n{'='*120}", flush=True)
    results = {}

    if 'net_spec' in cot.columns:
        cot['net_spec_z52'] = (cot['net_spec'] - cot['net_spec'].rolling(52).mean()) / cot['net_spec'].rolling(52).std()
    cot_z = cot.get('net_spec_z52', pd.Series(dtype=float)).reindex(h1.index, method='ffill')

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)

    # Parameter perturbation: vary Z threshold around 1.5
    print(f"\n  --- Parameter Perturbation (Z threshold) ---", flush=True)
    perturb_results = []
    for z_thresh in np.arange(1.0, 2.1, 0.1):
        filtered = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in cot_z.index and
                           not np.isnan(cot_z.loc[pd.Timestamp(t['entry_time'])]) and
                           abs(cot_z.loc[pd.Timestamp(t['entry_time'])]) > z_thresh)]
        stats = _stats(filtered)
        perturb_results.append({'threshold': round(z_thresh, 2), 'sharpe': stats['sharpe'], 'pnl': stats['pnl'], 'n': stats['n']})
        print(f"    Z>{z_thresh:.2f}: Sharpe={stats['sharpe']:.3f}, N={stats['n']}", flush=True)

    results['perturbation'] = perturb_results

    # Monotonicity check
    sharpes = [r['sharpe'] for r in perturb_results]
    monotonic_up = all(sharpes[i] >= sharpes[i+1] - 0.1 for i in range(len(sharpes)-1))
    sweet_spot = max(perturb_results, key=lambda x: x['sharpe'])
    print(f"\n  Sweet spot: Z>{sweet_spot['threshold']} (Sharpe={sweet_spot['sharpe']:.3f})", flush=True)
    print(f"  Monotonicity: {'OK' if monotonic_up else 'NON-MONOTONIC (potential overfitting)'}", flush=True)
    results['sweet_spot'] = sweet_spot
    results['monotonic'] = monotonic_up

    # Bootstrap: resample trades 5000 times, compute Sharpe distribution
    print(f"\n  --- Bootstrap (5000 paths) ---", flush=True)
    best_z = sweet_spot['threshold']
    filtered_trades = [t for t in base_merged
                      if not (pd.Timestamp(t['entry_time']) in cot_z.index and
                             not np.isnan(cot_z.loc[pd.Timestamp(t['entry_time'])]) and
                             abs(cot_z.loc[pd.Timestamp(t['entry_time'])]) > best_z)]

    pnls_filtered = np.array([t['pnl'] for t in filtered_trades])
    pnls_base = np.array([t['pnl'] for t in base_merged])
    n_boot = 5000; boot_sharpe_diff = []

    for _ in range(n_boot):
        idx_f = np.random.randint(0, len(pnls_filtered), len(pnls_filtered))
        idx_b = np.random.randint(0, len(pnls_base), len(pnls_base))
        daily_f = pd.Series(pnls_filtered[idx_f]).groupby(np.arange(len(idx_f)) // 10).sum()
        daily_b = pd.Series(pnls_base[idx_b]).groupby(np.arange(len(idx_b)) // 10).sum()
        sh_f = daily_f.mean() / daily_f.std() * np.sqrt(252) if daily_f.std() > 0 else 0
        sh_b = daily_b.mean() / daily_b.std() * np.sqrt(252) if daily_b.std() > 0 else 0
        boot_sharpe_diff.append(sh_f - sh_b)

    boot_sharpe_diff = np.array(boot_sharpe_diff)
    pct_positive = (boot_sharpe_diff > 0).mean() * 100
    ci_5 = np.percentile(boot_sharpe_diff, 5)
    ci_95 = np.percentile(boot_sharpe_diff, 95)
    print(f"    P(Sharpe_new > Sharpe_base) = {pct_positive:.1f}%", flush=True)
    print(f"    95% CI of Sharpe difference: [{ci_5:.3f}, {ci_95:.3f}]", flush=True)
    results['bootstrap'] = {'pct_positive': round(pct_positive, 1), 'ci_5': round(ci_5, 3), 'ci_95': round(ci_95, 3), 'mean_diff': round(boot_sharpe_diff.mean(), 3)}

    with open(OUTPUT_DIR / "phase_5_mc.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 5 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# PHASE 6: Implementation Feasibility
# ═══════════════════════════════════════════════════════════════
def phase_6(h1, pctl, cot):
    print(f"\n{'='*120}\n  PHASE 6: IMPLEMENTATION FEASIBILITY\n{'='*120}", flush=True)
    results = {}

    # Lookback sensitivity for Z-score calculation
    print(f"\n  --- Z-Score Lookback Sensitivity ---", flush=True)
    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)

    for lookback in [26, 39, 52, 78, 104]:
        cot_copy = cot.copy()
        cot_copy[f'z_{lookback}'] = (cot_copy['net_spec'] - cot_copy['net_spec'].rolling(lookback).mean()) / cot_copy['net_spec'].rolling(lookback).std()
        cot_z_lb = cot_copy[f'z_{lookback}'].reindex(h1.index, method='ffill')

        filtered = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in cot_z_lb.index and
                           not np.isnan(cot_z_lb.loc[pd.Timestamp(t['entry_time'])]) and
                           abs(cot_z_lb.loc[pd.Timestamp(t['entry_time'])]) > 1.5)]
        stats = _stats(filtered)
        delta = stats['sharpe'] - _stats(base_merged)['sharpe']
        print(f"    Lookback={lookback}w: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), N={stats['n']}", flush=True)
        results[f'lookback_{lookback}w'] = stats

    # Data lag: COT reports come out Friday for Tuesday data (3-day lag + weekend)
    print(f"\n  --- Publication Lag Sensitivity ---", flush=True)
    if 'net_spec' in cot.columns:
        cot_copy = cot.copy()
        cot_copy['z52'] = (cot_copy['net_spec'] - cot_copy['net_spec'].rolling(52).mean()) / cot_copy['net_spec'].rolling(52).std()

        for lag_days in [0, 3, 5, 7, 10]:
            cot_lagged = cot_copy['z52'].copy()
            cot_lagged.index = cot_lagged.index + pd.Timedelta(days=lag_days)
            cot_z_lag = cot_lagged.reindex(h1.index, method='ffill')

            filtered = [t for t in base_merged
                        if not (pd.Timestamp(t['entry_time']) in cot_z_lag.index and
                               not np.isnan(cot_z_lag.loc[pd.Timestamp(t['entry_time'])]) and
                               abs(cot_z_lag.loc[pd.Timestamp(t['entry_time'])]) > 1.5)]
            stats = _stats(filtered)
            delta = stats['sharpe'] - _stats(base_merged)['sharpe']
            print(f"    Lag={lag_days}d: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), N={stats['n']}", flush=True)
            results[f'lag_{lag_days}d'] = stats

    # Annual breakdown
    print(f"\n  --- Annual Performance (with COT filter) ---", flush=True)
    cot['z52'] = (cot['net_spec'] - cot['net_spec'].rolling(52).mean()) / cot['net_spec'].rolling(52).std()
    cot_z = cot['z52'].reindex(h1.index, method='ffill')

    filtered_all = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in cot_z.index and
                           not np.isnan(cot_z.loc[pd.Timestamp(t['entry_time'])]) and
                           abs(cot_z.loc[pd.Timestamp(t['entry_time'])]) > 1.5)]

    years = sorted(set(pd.Timestamp(t['entry_time']).year for t in filtered_all))
    annual_results = {}
    for yr in years:
        yr_trades_new = [t for t in filtered_all if pd.Timestamp(t['entry_time']).year == yr]
        yr_trades_base = [t for t in base_merged if pd.Timestamp(t['entry_time']).year == yr]
        sn = _stats(yr_trades_new); sb = _stats(yr_trades_base)
        delta_pnl = sn['pnl'] - sb['pnl']
        print(f"    {yr}: NEW Sharpe={sn['sharpe']:.3f} PnL=${sn['pnl']:>9,.0f} | BASE Sharpe={sb['sharpe']:.3f} PnL=${sb['pnl']:>9,.0f} | delta=${delta_pnl:>+8,.0f}", flush=True)
        annual_results[str(yr)] = {'new': sn, 'base': sb, 'delta_pnl': round(delta_pnl, 2)}

    results['annual'] = annual_results

    # Worst-case: what if COT data stops updating?
    print(f"\n  --- Fallback Behavior (COT unavailable) ---", flush=True)
    print(f"    If COT data unavailable: fall back to baseline (no filter) — Sharpe={_stats(base_merged)['sharpe']:.3f}", flush=True)
    results['fallback'] = 'no filter (safe degradation)'

    with open(OUTPUT_DIR / "phase_6_feasibility.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 6 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"{'='*120}")
    print(f"  R194b — DEEP VALIDATION OF R194 TOP FINDINGS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*120}\n")

    h1 = load_h1()
    cot = load_cot()
    macro = load_macro()

    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR pctl: {pctl.notna().sum()} valid\n", flush=True)

    r1 = phase_1(h1, pctl, cot)
    r2 = phase_2(h1, pctl, macro)
    r3 = phase_3(h1, pctl, macro)
    r4 = phase_4(h1, pctl, cot, macro)
    r5 = phase_5(h1, pctl, cot)
    r6 = phase_6(h1, pctl, cot)

    total_h = (time.time()-t0)/3600
    print(f"\n{'='*120}")
    print(f"  R194b COMPLETE — {total_h:.2f}h")
    print(f"{'='*120}")

    # Final verdict summary
    print(f"\n  VERDICTS:")
    for phase_key in ['z_1.0','z_1.25','z_1.5','z_1.75','z_2.0','z_2.5']:
        if phase_key in r1:
            v = r1[phase_key]['validation']['verdict']
            print(f"    COT {phase_key}: {v}")
