#!/usr/bin/env python3
"""
R193b — Brute Force 6-Strategy Lot Grid Search
================================================
Every strategy sweeps lot from 0.02 to 0.10 (step 0.02).
Total combinations: 5^6 = 15,625.
Cap scales proportionally with lot.
Constraints: total worst-case Cap <= $500, single Cap <= $75.

For top-10 portfolios: full K-Fold + WF + Era validation.
"""
import sys, os, time, json, warnings, copy, itertools
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r193b_brute")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

CURRENT_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap': 35,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60,  'sl': 4.0, 'tp': 6.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 15},
    'TSMOM':       {'lot': 0.04, 'cap': 60,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.04, 'cap': 60,  'sl': 4.5, 'tp': 4.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.03, 'cap': 25,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
}

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

# Cap per lot reference: at what lot was the original cap designed
CAP_REF = {
    'L8_MAX': (35, 0.02),
    'PSAR': (60, 0.09),
    'TSMOM': (60, 0.04),
    'SESS_BO': (60, 0.04),
    'DUAL_THRUST': (18, 0.04),
    'CHANDELIER': (25, 0.03),
}

ERA_SEGMENTS = {
    'full': None,
    'hike': [("2015-12-01","2019-01-01"),("2022-03-01","2023-08-01")],
    'cut':  [("2019-07-01","2022-03-01"),("2024-09-01","2026-06-01")],
    'recent_3y': [("2023-06-01","2026-06-01")],
}

import glob as _glob
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Core helpers
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
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0,'cap_pct':0,'sl_pct':0,'trail_pct':0,'timeout_pct':0}
    daily=_daily(trades); pnls=[t['pnl'] for t in trades]; n=len(trades)
    wins=[p for p in pnls if p>0]
    eq=daily.cumsum(); dd=float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    reasons=[t['reason'] for t in trades]
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2),
            'cap_pct':round(sum(1 for r in reasons if 'Cap' in r)/n*100,1),
            'sl_pct':round(sum(1 for r in reasons if r=='SL')/n*100,1),
            'trail_pct':round(sum(1 for r in reasons if r=='Trail')/n*100,1),
            'timeout_pct':round(sum(1 for r in reasons if r=='Timeout')/n*100,1)}

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
    print(f"  Loaded {len(df)} bars ({df.index[0]} ~ {df.index[-1]})",flush=True)
    return df

def kfold_test(h1, run_new, run_base, K=6):
    start=h1.index[0]; end=h1.index[-1]; total=(end-start).days; fd=total//K
    results=[]
    for fold in range(K):
        fs=start+pd.Timedelta(days=fold*fd)
        fe=start+pd.Timedelta(days=(fold+1)*fd) if fold<K-1 else end+pd.Timedelta(days=1)
        h1f=h1[(h1.index>=fs)&(h1.index<fe)]
        if len(h1f)<300: continue
        sh_new=run_new(h1f); sh_base=run_base(h1f)
        results.append({'fold':fold+1,'new':round(sh_new,3),'base':round(sh_base,3),
                        'win':'NEW' if sh_new>sh_base else 'BASE'})
    return results

def wf_test(h1, run_new, run_base, train_d=547, test_d=180):
    start=h1.index[0]; end=h1.index[-1]; cursor=start+pd.Timedelta(days=train_d)
    results=[]
    while cursor+pd.Timedelta(days=test_d)<=end+pd.Timedelta(days=1):
        ts=cursor; te=cursor+pd.Timedelta(days=test_d)
        h1t=h1[(h1.index>=ts)&(h1.index<te)]
        if len(h1t)<200: cursor+=pd.Timedelta(days=test_d); continue
        sh_new=run_new(h1t); sh_base=run_base(h1t)
        results.append({'new':round(sh_new,3),'base':round(sh_base,3),
                        'win':'NEW' if sh_new>sh_base else 'BASE'})
        cursor+=pd.Timedelta(days=test_d)
    return results

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
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    cap=cfg['cap']
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
    times=df.index; n2=len(df); lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    cap=cfg['cap']
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

def bt_tsmom(h1, cfg, pctl_v=None, pctl_f=0, fast_lb=480, slow_lb=720):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    times=df.index; n=len(df); mx=max(fast_lb,slow_lb)
    sc=np.full(n,np.nan)
    for i in range(mx,n):
        v=0.0
        if c[i-fast_lb]>0: v+=0.5*np.sign(c[i]/c[i-fast_lb]-1.0)
        if c[i-slow_lb]>0: v+=0.5*np.sign(c[i]/c[i-slow_lb]-1.0)
        sc[i]=v
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    cap=cfg['cap']
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
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    cap=cfg['cap']
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
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    cap=cfg['cap']
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
    rsi_v=df['RSI14'].values
    times=df.index; n=len(df); p=22; m=3.0
    cl=np.full(n,np.nan); cs=np.full(n,np.nan)
    for i in range(p,n): cl[i]=np.max(h[i-p+1:i+1])-m*atr[i]; cs[i]=np.min(lo[i-p+1:i+1])+m*atr[i]
    d=np.zeros(n)
    for i in range(p+1,n):
        if np.isnan(cl[i]) or np.isnan(cs[i]): d[i]=d[i-1]; continue
        if c[i]>cs[i-1]: d[i]=1
        elif c[i]<cl[i-1]: d[i]=-1
        else: d[i]=d[i-1]
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    cap=cfg['cap']
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

def run_strat(name, h1, cfg, pctl_v=None, pctl_f=0):
    return BT_FN[name](h1, cfg, pctl_v=pctl_v, pctl_f=pctl_f)

def run_all_cfg(h1, config, pctl_v=None, pctl_f=0):
    return {nm: run_strat(nm, h1, config[nm], pctl_v=pctl_v, pctl_f=pctl_f) for nm in STRAT_ORDER}

def port_sharpe_from_trades(all_t):
    merged = [t for nm in STRAT_ORDER for t in all_t.get(nm,[])]
    return _sharpe(_daily(merged))


def main():
    print("=" * 120)
    print("  R193b — Brute Force 6-Strategy Lot Grid (0.02~0.10, step 0.02)")
    print("=" * 120, flush=True)

    h1 = load_h1()
    atr_series = compute_atr(h1).dropna()
    pctl = compute_atr_pctl(atr_series)

    LOT_GRID = [0.02, 0.04, 0.06, 0.08, 0.10]

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Pre-compute all individual strategy trades at each lot
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Step 1: Pre-computing trades for {len(STRAT_ORDER)} strategies x {len(LOT_GRID)} lots = {len(STRAT_ORDER)*len(LOT_GRID)} backtests...")
    trade_cache = {}
    for nm in STRAT_ORDER:
        for lot in LOT_GRID:
            cfg = copy.deepcopy(CURRENT_CONFIG[nm])
            cfg['lot'] = lot
            ref_cap, ref_lot = CAP_REF[nm]
            cfg['cap'] = round(ref_cap * lot / ref_lot, 1)
            trade_cache[(nm, lot)] = run_strat(nm, h1, cfg, pctl_v=pctl, pctl_f=30)
            s = _stats(trade_cache[(nm, lot)])
            print(f"    {nm:<15} lot={lot:.2f} cap=${cfg['cap']:>6.1f}: N={s['n']:>6}, Sharpe={s['sharpe']:>7.3f}, PnL=${s['pnl']:>10,.0f}, Cap%={s['cap_pct']:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Brute force grid search (5^6 = 15,625 combos)
    # ═══════════════════════════════════════════════════════════════
    total_combos = len(LOT_GRID) ** 6
    print(f"\n  Step 2: Grid search over {total_combos} combinations...")
    print(f"  Constraints: total Cap <= $500, single Cap <= $75")

    valid_combos = []; checked = 0; skipped = 0

    for combo in itertools.product(LOT_GRID, repeat=6):
        checked += 1
        lots = dict(zip(STRAT_ORDER, combo))
        caps = {}
        for nm in STRAT_ORDER:
            ref_cap, ref_lot = CAP_REF[nm]
            caps[nm] = round(ref_cap * lots[nm] / ref_lot, 1)

        total_cap = sum(caps.values())
        if total_cap > 500 or any(v > 75 for v in caps.values()):
            skipped += 1; continue

        all_trades = {nm: trade_cache[(nm, lots[nm])] for nm in STRAT_ORDER}
        sh = port_sharpe_from_trades(all_trades)
        total_pnl = sum(_stats(all_trades[nm])['pnl'] for nm in STRAT_ORDER)

        valid_combos.append({
            'lots': lots, 'caps': caps, 'sharpe': round(sh, 3),
            'pnl': round(total_pnl, 0), 'total_cap': round(total_cap, 1)
        })

    valid_combos.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Checked: {checked}, Valid: {len(valid_combos)}, Skipped: {skipped}")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Display top-30 results
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Top 30 portfolios:")
    print(f"  {'#':>3} {'Sharpe':>7} {'PnL':>12} {'TotCap':>8} | {'Kelt':>5} {'PSAR':>5} {'TSMOM':>5} {'SESS':>5} {'DT':>5} {'CH':>5} | Cap details")
    for idx, r in enumerate(valid_combos[:30]):
        l = r['lots']; c = r['caps']
        cap_str = ' '.join(f"${c[nm]:.0f}" for nm in STRAT_ORDER)
        print(f"  {idx+1:>3} {r['sharpe']:>7.3f} ${r['pnl']:>11,.0f} ${r['total_cap']:>7.1f} | {l['L8_MAX']:>5.2f} {l['PSAR']:>5.02f} {l['TSMOM']:>5.02f} {l['SESS_BO']:>5.02f} {l['DUAL_THRUST']:>5.02f} {l['CHANDELIER']:>5.02f} | {cap_str}")

    # Current config
    cur_trades = run_all_cfg(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    cur_sh = port_sharpe_from_trades(cur_trades)
    cur_pnl = sum(_stats(cur_trades[nm])['pnl'] for nm in STRAT_ORDER)
    cur_tcap = sum(CURRENT_CONFIG[nm]['cap'] for nm in STRAT_ORDER)
    print(f"\n  CURRENT: Sharpe={cur_sh:.3f}, PnL=${cur_pnl:,.0f}, TotalCap=${cur_tcap}")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: K-Fold + WF validation for top-5
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Step 4: K-Fold + WF validation for top-5 combos vs current...")

    def run_current_port(h1f):
        t = run_all_cfg(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        return port_sharpe_from_trades(t)

    validated = []
    for idx, r in enumerate(valid_combos[:5]):
        lots = r['lots']; caps = r['caps']

        def run_new_port(h1f, _lots=lots, _caps=caps):
            cfg = copy.deepcopy(CURRENT_CONFIG)
            for nm in STRAT_ORDER:
                cfg[nm]['lot'] = _lots[nm]
                cfg[nm]['cap'] = _caps[nm]
            t = run_all_cfg(h1f, cfg, pctl_v=pctl, pctl_f=30)
            return port_sharpe_from_trades(t)

        kf = kfold_test(h1, run_new_port, run_current_port)
        kf_wins = sum(1 for x in kf if x['win']=='NEW')
        wf = wf_test(h1, run_new_port, run_current_port)
        wf_wins = sum(1 for x in wf if x['win']=='NEW')

        # Era test
        cfg_new = copy.deepcopy(CURRENT_CONFIG)
        for nm in STRAT_ORDER: cfg_new[nm]['lot'] = lots[nm]; cfg_new[nm]['cap'] = caps[nm]
        new_t = run_all_cfg(h1, cfg_new, pctl_v=pctl, pctl_f=30)
        new_merged = [t for nm in STRAT_ORDER for t in new_t.get(nm,[])]
        cur_merged = [t for nm in STRAT_ORDER for t in cur_trades.get(nm,[])]

        era_results = {}
        for era in ['full','hike','cut','recent_3y']:
            en = filter_era(new_merged, era); ec = filter_era(cur_merged, era)
            sn = _sharpe(_daily(en)); sc = _sharpe(_daily(ec))
            era_results[era] = {'new': round(sn,3), 'cur': round(sc,3), 'delta': round(sn-sc,3)}

        l = lots
        print(f"\n  #{idx+1} Kelt={l['L8_MAX']:.2f} PSAR={l['PSAR']:.2f} TSMOM={l['TSMOM']:.2f} SESS={l['SESS_BO']:.2f} DT={l['DUAL_THRUST']:.2f} CH={l['CHANDELIER']:.2f}")
        print(f"      Sharpe={r['sharpe']:.3f}, PnL=${r['pnl']:,.0f}, TotalCap=${r['total_cap']:.0f}")
        print(f"      K-Fold: {kf_wins}/{len(kf)}, WF: {wf_wins}/{len(wf)}")
        for e in ['hike','cut','recent_3y']:
            print(f"      Era {e}: cur={era_results[e]['cur']:.3f} new={era_results[e]['new']:.3f} (d={era_results[e]['delta']:+.3f})")

        go = kf_wins >= 4 and wf_wins >= 13 and all(era_results[e]['new'] > 0 for e in ['hike','cut','recent_3y'])
        print(f"      >>> {'GO' if go else 'NO-GO'}")
        validated.append({**r, 'kf_wins': kf_wins, 'wf_wins': wf_wins, 'wf_total': len(wf),
                          'era': era_results, 'verdict': 'GO' if go else 'NO-GO'})

    # Save all results
    output = {
        'top30': valid_combos[:30],
        'current': {'sharpe': round(cur_sh,3), 'pnl': round(cur_pnl,0), 'total_cap': cur_tcap},
        'validated_top5': validated,
        'total_valid_combos': len(valid_combos),
        'total_checked': checked,
    }
    with open(OUTPUT_DIR / "r193b_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"  R193b Complete! Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Valid combos: {len(valid_combos)}/{total_combos}")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()
