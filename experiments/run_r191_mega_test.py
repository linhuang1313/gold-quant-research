#!/usr/bin/env python3
"""
R191 — 10-Hour Mega Test Suite
================================
12 batches covering: live parity, session ADX, ML R173, cap_atr_mult,
max_hold, TSMOM deep dive, slot competition, trailing, SL/TP,
cooldown, Monte Carlo, and combined optimal config.

Results saved incrementally per batch.
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r191_mega")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

# Live config as of 2026-05-09 (from gold-quant-trading/config.py)
LIVE_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap': 35,  'cap_atr_mult': 4.0, 'sl': 3.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60,  'cap_atr_mult': 4.5, 'sl': 4.0, 'tp': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15},
    'TSMOM':       {'lot': 0.04, 'cap': 60,  'cap_atr_mult': 6.5, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.04, 'cap': 60,  'cap_atr_mult': 5.0, 'sl': 4.5, 'tp': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18,  'cap_atr_mult': 5.0, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.03, 'cap': 25,  'cap_atr_mult': 5.0, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
}

OLD_LOTS = {'L8_MAX':0.02,'PSAR':0.09,'TSMOM':0.15,'SESS_BO':0.13,'DUAL_THRUST':0.04,'CHANDELIER':0.08}

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
# Core helpers (shared across all batches)
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
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0,'cap_pct':0,'sl_pct':0,'tp_pct':0,'max_loss':0}
    daily=_daily(trades); pnls=[t['pnl'] for t in trades]; n=len(trades)
    wins=[p for p in pnls if p>0]
    eq=daily.cumsum(); dd=float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    reasons=[t['reason'] for t in trades]
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2),
            'cap_pct':round(sum(1 for r in reasons if 'Cap' in r)/n*100,1),
            'sl_pct':round(sum(1 for r in reasons if r=='SL')/n*100,1),
            'tp_pct':round(sum(1 for r in reasons if r=='TP')/n*100,1),
            'max_loss':round(min(pnls),2) if pnls else 0}

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

def port_merge(all_t):
    return [t for nm in STRAT_ORDER for t in all_t.get(nm,[])]

def port_stats(all_t):
    m=port_merge(all_t); return _stats(m), _daily(m)

def save_batch(num, data):
    out = OUTPUT_DIR / f"batch_{num}_results.json"
    with open(out,'w') as f: json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {out}", flush=True)

def kfold_test(h1, run_fn, K=6):
    start=h1.index[0]; end=h1.index[-1]; total=(end-start).days; fd=total//K
    results=[]
    for fold in range(K):
        fs=start+pd.Timedelta(days=fold*fd)
        fe=start+pd.Timedelta(days=(fold+1)*fd) if fold<K-1 else end+pd.Timedelta(days=1)
        h1f=h1[(h1.index>=fs)&(h1.index<fe)]
        if len(h1f)<300: continue
        sh = run_fn(h1f)
        results.append({'fold':fold+1,'sharpe':sh,'period':f"{fs.date()}~{fe.date()}"})
    return results

def wf_test(h1, run_fn, train_d=547, test_d=180):
    start=h1.index[0]; end=h1.index[-1]; cursor=start+pd.Timedelta(days=train_d)
    results=[]; period=0
    while cursor+pd.Timedelta(days=test_d)<=end+pd.Timedelta(days=1):
        period+=1; ts=cursor; te=cursor+pd.Timedelta(days=test_d)
        h1t=h1[(h1.index>=ts)&(h1.index<te)]
        if len(h1t)<200: cursor+=pd.Timedelta(days=test_d); continue
        sh=run_fn(h1t)
        results.append({'period':period,'sharpe':sh,'test':f"{ts.date()}~{te.date()}"})
        cursor+=pd.Timedelta(days=test_d)
    return results

# ═══════════════════════════════════════════════════════════════
# Strategy backtest functions
# ═══════════════════════════════════════════════════════════════
def _get_cap(cfg, atr, use_dynamic=False):
    if use_dynamic and cfg.get('cap_atr_mult',0) > 0:
        return cfg['cap_atr_mult'] * atr * cfg['lot'] * PV
    return cfg['cap']

def bt_keltner(h1, cfg, adx_by_session=None, pctl_v=None, pctl_f=0, use_dynamic_cap=False):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA100']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA100','KC_upper'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA100'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
    hrs=df.index.hour
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            cap=_get_cap(cfg, pos['atr'], use_dynamic_cap)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): continue
        adx_thresh = 14
        if adx_by_session:
            hr = hrs[i]
            if 0<=hr<=7: adx_thresh=adx_by_session.get('asia',14)
            elif 8<=hr<=12: adx_thresh=adx_by_session.get('london',14)
            elif 13<=hr<=17: adx_thresh=adx_by_session.get('ny',14)
            else: adx_thresh=adx_by_session.get('evening',14)
        if np.isnan(adx[i]) or adx[i]<adx_thresh: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'L8_MAX'}
    return trades

def bt_psar(h1, cfg, pctl_v=None, pctl_f=0, use_dynamic_cap=False):
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
    trades=[]; pos=None; le=-999; prev=c[0]>ps[0]
    for i in range(1,n2):
        cur=c[i]>ps[i]
        if pos:
            cap=_get_cap(cfg,pos['atr'],use_dynamic_cap)
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

def bt_tsmom(h1, cfg, pctl_v=None, pctl_f=0, use_dynamic_cap=False, fast_lb=480, slow_lb=720):
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
    trades=[]; pos=None; le=-999
    for i in range(mx+1,n):
        if pos:
            cap=_get_cap(cfg,pos['atr'],use_dynamic_cap)
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

def bt_sess_bo(h1, cfg, pctl_v=None, pctl_f=0, use_dynamic_cap=False):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    hrs=df.index.hour; times=df.index; n=len(df); lb=4
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    trades=[]; pos=None; le=-999
    for i in range(lb,n):
        if pos:
            cap=_get_cap(cfg,pos['atr'],use_dynamic_cap)
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

def bt_dt(h1, cfg, pctl_v=None, pctl_f=0, use_dynamic_cap=False):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_arr=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,o,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df); nb=6; k=0.5
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']
    trades=[]; pos=None; le=-999
    for i in range(nb,n):
        if pos:
            cap=_get_cap(cfg,pos['atr'],use_dynamic_cap)
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

def bt_chand(h1, cfg, pctl_v=None, pctl_f=0, use_dynamic_cap=False):
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
    trades=[]; pos=None; le=-999
    for i in range(p+2,n):
        if pos:
            cap=_get_cap(cfg,pos['atr'],use_dynamic_cap)
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

BT={'L8_MAX':bt_keltner,'PSAR':bt_psar,'TSMOM':bt_tsmom,'SESS_BO':bt_sess_bo,'DUAL_THRUST':bt_dt,'CHANDELIER':bt_chand}

def run_all(h1, config=None, pctl_v=None, pctl_f=0, use_dynamic_cap=False, adx_by_session=None, strats=None):
    cfg = config or LIVE_CONFIG
    strats_to_run = strats or STRAT_ORDER
    r={}
    for name in strats_to_run:
        c = cfg.get(name, LIVE_CONFIG[name])
        if name=='L8_MAX':
            r[name]=bt_keltner(h1,c,adx_by_session=adx_by_session,pctl_v=pctl_v,pctl_f=pctl_f,use_dynamic_cap=use_dynamic_cap)
        elif name=='PSAR':
            r[name]=bt_psar(h1,c,pctl_v=pctl_v,pctl_f=pctl_f,use_dynamic_cap=use_dynamic_cap)
        elif name=='TSMOM':
            r[name]=bt_tsmom(h1,c,pctl_v=pctl_v,pctl_f=pctl_f,use_dynamic_cap=use_dynamic_cap)
        elif name=='SESS_BO':
            r[name]=bt_sess_bo(h1,c,pctl_v=pctl_v,pctl_f=pctl_f,use_dynamic_cap=use_dynamic_cap)
        elif name=='DUAL_THRUST':
            r[name]=bt_dt(h1,c,pctl_v=pctl_v,pctl_f=pctl_f,use_dynamic_cap=use_dynamic_cap)
        elif name=='CHANDELIER':
            r[name]=bt_chand(h1,c,pctl_v=pctl_v,pctl_f=pctl_f,use_dynamic_cap=use_dynamic_cap)
    return r


# ═══════════════════════════════════════════════════════════════
# BATCH 1: Live Parity Check
# ═══════════════════════════════════════════════════════════════
def batch_1(h1):
    print(f"\n{'='*120}\n  BATCH 1: Live Parity Check\n{'='*120}",flush=True)
    configs = {
        'old_lots': {n:{**LIVE_CONFIG[n],'lot':OLD_LOTS[n]} for n in STRAT_ORDER},
        'live_config': LIVE_CONFIG,
    }
    print(f"\n  {'Label':<16} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Cap%':>6} {'MaxLoss':>8}")
    results={}
    for label, cfg in configs.items():
        all_t=run_all(h1,cfg); s,_=port_stats(all_t)
        print(f"  {label:<16} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['cap_pct']:>5.1f}% ${s['max_loss']:>7,.0f}")
        results[label]=s
    # Dynamic cap parity (what live actually does)
    all_t=run_all(h1,LIVE_CONFIG,use_dynamic_cap=True); s,_=port_stats(all_t)
    print(f"  {'live_dyn_cap':<16} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['cap_pct']:>5.1f}% ${s['max_loss']:>7,.0f}")
    results['live_dyn_cap']=s
    save_batch(1, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 2: Session ADX Repair
# ═══════════════════════════════════════════════════════════════
def batch_2(h1):
    print(f"\n{'='*120}\n  BATCH 2: Session ADX Repair Test\n{'='*120}",flush=True)
    sessions = [
        ('flat_14', None),
        ('R178_intended', {'asia':14,'london':10,'ny':16,'evening':14}),
        ('london8_ny18', {'asia':14,'london':8,'ny':18,'evening':14}),
        ('london12_ny14', {'asia':14,'london':12,'ny':14,'evening':14}),
        ('all_10', {'asia':10,'london':10,'ny':10,'evening':10}),
        ('all_16', {'asia':16,'london':16,'ny':16,'evening':16}),
        ('all_18', {'asia':18,'london':18,'ny':18,'evening':18}),
        ('asia10_lon8_ny20_eve16', {'asia':10,'london':8,'ny':20,'evening':16}),
    ]
    print(f"\n  {'Config':<30} {'N':>6} {'Sharpe':>7} {'PnL':>10}")
    results={}
    for label, adx_s in sessions:
        t=bt_keltner(h1,LIVE_CONFIG['L8_MAX'],adx_by_session=adx_s); s=_stats(t)
        print(f"  {label:<30} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f}")
        results[label]=s
    # K-Fold for top 2
    sorted_r = sorted([(k,v['sharpe']) for k,v in results.items()], key=lambda x:-x[1])
    best_label = sorted_r[0][0]
    best_adx = dict(sessions)[best_label]
    print(f"\n  Best: {best_label} (Sharpe={sorted_r[0][1]:.3f})")
    def run_kf(h1f):
        t=bt_keltner(h1f,LIVE_CONFIG['L8_MAX'],adx_by_session=best_adx); return _stats(t)['sharpe']
    def run_kf_flat(h1f):
        t=bt_keltner(h1f,LIVE_CONFIG['L8_MAX']); return _stats(t)['sharpe']
    kf_best=kfold_test(h1,run_kf); kf_flat=kfold_test(h1,run_kf_flat)
    wins=sum(1 for b,f in zip(kf_best,kf_flat) if b['sharpe']>f['sharpe'])
    print(f"  K-Fold: Best wins {wins}/{len(kf_best)}")
    wf_best=wf_test(h1,run_kf); wf_flat=wf_test(h1,run_kf_flat)
    wf_wins=sum(1 for b,f in zip(wf_best,wf_flat) if b['sharpe']>f['sharpe'])
    print(f"  WF: Best wins {wf_wins}/{len(wf_best)}")
    results['validation']={'best':best_label,'kf_wins':wins,'kf_total':len(kf_best),'wf_wins':wf_wins,'wf_total':len(wf_best)}
    save_batch(2, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 3: ML R173 Shadow-to-Live
# ═══════════════════════════════════════════════════════════════
def batch_3(h1):
    print(f"\n{'='*120}\n  BATCH 3: ML R173 Shadow-to-Live Test\n{'='*120}",flush=True)
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['RSI14']=compute_rsi(df['Close'],14)
    bb_mid=df['Close'].rolling(20).mean(); bb_std=df['Close'].rolling(20).std()
    bb_upper=bb_mid+2*bb_std; bb_lower=bb_mid-2*bb_std
    kc_mid=df['Close'].ewm(span=25,adjust=False).mean()
    kc_upper=kc_mid+1.2*df['ATR']; kc_lower=kc_mid-1.2*df['ATR']
    df['squeeze']=((bb_lower>kc_lower)&(bb_upper<kc_upper)).astype(float)
    features_df = df[['ATR','ADX','RSI14','squeeze']].copy()
    # Simulate ML filter: for each bar, compute a simple score
    # Score = weighted combo of features (proxy for XGBoost)
    # Higher ADX + moderate RSI + no squeeze + reasonable ATR = higher score
    atr_v=features_df['ATR'].values; adx_v=features_df['ADX'].values
    rsi_v=features_df['RSI14'].values; sq_v=features_df['squeeze'].values
    n=len(features_df)
    ml_score=np.full(n, 0.5)
    for i in range(50,n):
        if np.isnan(adx_v[i]) or np.isnan(rsi_v[i]) or np.isnan(atr_v[i]): continue
        s = 0.0
        s += 0.3 * min(adx_v[i]/30, 1.0)  # ADX contribution
        rsi_dev = abs(rsi_v[i]-50)/50; s += 0.3 * rsi_dev  # RSI extremity
        s += 0.2 * (1.0 - sq_v[i])  # No squeeze = good
        atr_pctl = np.nanpercentile(atr_v[max(0,i-300):i], 50)
        if atr_pctl > 0: s += 0.2 * min(atr_v[i]/atr_pctl/2, 1.0)
        ml_score[i] = min(s, 1.0)
    ml_s = pd.Series(ml_score, index=features_df.index)

    non_keltner = ['PSAR','TSMOM','SESS_BO','DUAL_THRUST','CHANDELIER']
    thresholds = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"\n  {'Threshold':>10} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Filtered%':>10}")
    results = {}
    for thr in thresholds:
        all_t = run_all(h1, LIVE_CONFIG)
        if thr > 0:
            for name in non_keltner:
                filtered = []
                for t in all_t[name]:
                    idx = ml_s.index.get_indexer([pd.Timestamp(t['entry_time'])], method='nearest')[0]
                    if idx >= 0 and ml_score[idx] >= thr:
                        filtered.append(t)
                all_t[name] = filtered
        s,_ = port_stats(all_t); n_base = sum(len(run_all(h1,LIVE_CONFIG)[nm]) for nm in non_keltner)
        n_filt = sum(len(all_t[nm]) for nm in non_keltner)
        filt_pct = (1 - n_filt/n_base)*100 if n_base>0 else 0
        label = f"{thr:.2f}" if thr>0 else "none"
        print(f"  {label:>10} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {filt_pct:>9.1f}%")
        results[label] = {**s, 'filtered_pct': round(filt_pct,1)}
    save_batch(3, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 4: cap_atr_mult Parity
# ═══════════════════════════════════════════════════════════════
def batch_4(h1):
    print(f"\n{'='*120}\n  BATCH 4: cap_atr_mult Parity (Fixed vs Dynamic Caps)\n{'='*120}",flush=True)
    print(f"\n  --- Portfolio level ---")
    print(f"  {'Mode':<16} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Cap%':>6}")
    results={}
    for mode,dyn in [('fixed_cap',False),('dynamic_cap',True)]:
        all_t=run_all(h1,LIVE_CONFIG,use_dynamic_cap=dyn); s,_=port_stats(all_t)
        print(f"  {mode:<16} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['cap_pct']:>5.1f}%")
        results[mode]=s
    print(f"\n  --- Per-strategy ---")
    print(f"  {'Strategy':<15} {'Sh_fix':>7} {'Sh_dyn':>7} {'Cap%_f':>7} {'Cap%_d':>7}")
    for name in STRAT_ORDER:
        tf=BT[name](h1,LIVE_CONFIG[name]) if name!='L8_MAX' else bt_keltner(h1,LIVE_CONFIG[name])
        td_t=BT[name](h1,LIVE_CONFIG[name],use_dynamic_cap=True) if name!='L8_MAX' else bt_keltner(h1,LIVE_CONFIG[name],use_dynamic_cap=True)
        sf=_stats(tf); sd=_stats(td_t)
        print(f"  {name:<15} {sf['sharpe']:>7.3f} {sd['sharpe']:>7.3f} {sf['cap_pct']:>6.1f}% {sd['cap_pct']:>6.1f}%")
        results[f'{name}_fixed']=sf; results[f'{name}_dynamic']=sd
    # Era
    for era in ['full','hike','cut','recent_3y']:
        tf=run_all(h1,LIVE_CONFIG); td_t=run_all(h1,LIVE_CONFIG,use_dynamic_cap=True)
        ef=[t for nm in STRAT_ORDER for t in filter_era(tf[nm],era)]
        ed=[t for nm in STRAT_ORDER for t in filter_era(td_t[nm],era)]
        sf=_sharpe(_daily(ef)); sd=_sharpe(_daily(ed))
        print(f"  Era {era}: fixed={sf:.3f}, dynamic={sd:.3f} (d={sd-sf:+.3f})")
        results[f'era_{era}']={'fixed':round(sf,3),'dynamic':round(sd,3)}
    save_batch(4, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 5: Max Hold Hours vs Bars
# ═══════════════════════════════════════════════════════════════
def batch_5(h1):
    print(f"\n{'='*120}\n  BATCH 5: Max Hold Sweep\n{'='*120}",flush=True)
    results={}
    for name in STRAT_ORDER:
        orig_mh = LIVE_CONFIG[name]['max_hold']
        if name=='L8_MAX': sweep = [1,2,3,4,5,6,8,10]
        elif name=='TSMOM': sweep = [4,6,8,10,12,16,20,24]
        else: sweep = [5,8,10,12,15,18,20,25,30]
        print(f"\n  {name} (current={orig_mh}):")
        print(f"  {'MaxHold':>8} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Timeout%':>9}")
        best_sh=-999; best_mh=orig_mh; strat_res=[]
        for mh in sweep:
            cfg=copy.deepcopy(LIVE_CONFIG[name]); cfg['max_hold']=mh
            if name=='L8_MAX': t=bt_keltner(h1,cfg)
            else: t=BT[name](h1,cfg)
            s=_stats(t)
            to_pct=round(sum(1 for tr in t if tr['reason']=='Timeout')/max(len(t),1)*100,1)
            print(f"  {mh:>8} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {to_pct:>8.1f}%")
            strat_res.append({'mh':mh,**s,'timeout_pct':to_pct})
            if s['sharpe']>best_sh: best_sh=s['sharpe']; best_mh=mh
        print(f"  Best: max_hold={best_mh} (Sharpe={best_sh:.3f})")
        results[name]={'sweep':strat_res,'best_mh':best_mh,'best_sharpe':round(best_sh,3)}
    save_batch(5, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 6: TSMOM Deep Dive
# ═══════════════════════════════════════════════════════════════
def batch_6(h1):
    print(f"\n{'='*120}\n  BATCH 6: TSMOM Deep Dive\n{'='*120}",flush=True)
    cfg=LIVE_CONFIG['TSMOM']
    results={}
    # Standalone
    t=bt_tsmom(h1,cfg); s=_stats(t)
    print(f"\n  Standalone: N={s['n']}, Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:,.0f}")
    if t:
        bars_held=[tr['bars'] for tr in t]
        print(f"  Avg bars held: {np.mean(bars_held):.1f}, Median: {np.median(bars_held):.0f}")
        reasons=[tr['reason'] for tr in t]
        for r in set(reasons): print(f"    {r}: {sum(1 for x in reasons if x==r)} ({sum(1 for x in reasons if x==r)/len(t)*100:.1f}%)")
    results['standalone']=s

    # Fast/Slow combos
    combos = [(240,480),(360,720),(480,720),(480,960),(600,720),(720,960),(240,720)]
    print(f"\n  Score combos:")
    print(f"  {'Fast':>5} {'Slow':>5} {'N':>6} {'Sharpe':>7} {'PnL':>10}")
    for f_lb,s_lb in combos:
        t2=bt_tsmom(h1,cfg,fast_lb=f_lb,slow_lb=s_lb); s2=_stats(t2)
        print(f"  {f_lb:>5} {s_lb:>5} {s2['n']:>6} {s2['sharpe']:>7.3f} ${s2['pnl']:>9,.0f}")
        results[f'{f_lb}_{s_lb}']=s2

    # Portfolio with/without TSMOM
    print(f"\n  Portfolio impact:")
    all_with = run_all(h1,LIVE_CONFIG); s_w,_=port_stats(all_with)
    all_without = run_all(h1,LIVE_CONFIG,strats=[s for s in STRAT_ORDER if s!='TSMOM']); s_wo,_=port_stats(all_without)
    print(f"  With TSMOM: Sharpe={s_w['sharpe']:.3f}, PnL=${s_w['pnl']:,.0f}")
    print(f"  Without:    Sharpe={s_wo['sharpe']:.3f}, PnL=${s_wo['pnl']:,.0f}")
    results['portfolio_with']=s_w; results['portfolio_without']=s_wo

    # Era
    for era in ['full','hike','cut','recent_3y']:
        era_t=filter_era(t,era); s_e=_stats(era_t)
        print(f"  Era {era}: N={s_e['n']}, Sharpe={s_e['sharpe']:.3f}")
        results[f'era_{era}']=s_e
    save_batch(6, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 7: Strategy Slot Competition
# ═══════════════════════════════════════════════════════════════
def batch_7(h1):
    print(f"\n{'='*120}\n  BATCH 7: Strategy Slot Competition\n{'='*120}",flush=True)
    all_t = run_all(h1, LIVE_CONFIG)
    # All C(6,4) combos
    combos_4 = list(combinations(STRAT_ORDER, 4))
    print(f"\n  All C(6,4)=15 combos:")
    print(f"  {'Combo':<45} {'Sharpe':>7} {'PnL':>10}")
    results_combos=[]
    for combo in combos_4:
        sub = {nm: all_t[nm] for nm in combo}
        s,_ = port_stats(sub)
        label = '+'.join(combo)
        print(f"  {label:<45} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f}")
        results_combos.append({'combo':list(combo),'sharpe':s['sharpe'],'pnl':s['pnl']})
    results_combos.sort(key=lambda x:-x['sharpe'])
    print(f"\n  Best 4-combo: {'+'.join(results_combos[0]['combo'])} (Sharpe={results_combos[0]['sharpe']:.3f})")

    # MAX_POSITIONS sweep (using all 6 strategies but measuring Sharpe)
    print(f"\n  Slot count (all 6 strats enabled):")
    for n_slots in [3,4,5,6]:
        # Select top-N by standalone Sharpe
        standalone = [(nm, _stats(all_t[nm])['sharpe']) for nm in STRAT_ORDER]
        standalone.sort(key=lambda x:-x[1])
        top_n = [nm for nm,_ in standalone[:n_slots]]
        sub = {nm: all_t[nm] for nm in top_n}
        s,_ = port_stats(sub)
        print(f"  Top-{n_slots}: {'+'.join(top_n)} → Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:,.0f}")

    # Marginal value
    print(f"\n  Marginal value (drop each from 6):")
    s_all,_=port_stats(all_t)
    results_marginal={}
    for drop in STRAT_ORDER:
        sub={nm:all_t[nm] for nm in STRAT_ORDER if nm!=drop}
        s,_=port_stats(sub)
        d=s['sharpe']-s_all['sharpe']
        print(f"  Drop {drop:<15}: Sharpe={s['sharpe']:.3f} (d={d:+.3f}), PnL=${s['pnl']:,.0f}")
        results_marginal[drop]={'sharpe_without':s['sharpe'],'delta':round(d,3)}

    # Correlation
    print(f"\n  Strategy pair correlations (daily PnL):")
    dailies={nm:_daily(all_t[nm]) for nm in STRAT_ORDER}
    corr_data={}
    for i,a in enumerate(STRAT_ORDER):
        for b in STRAT_ORDER[i+1:]:
            da=dailies[a]; db=dailies[b]
            common=da.index.intersection(db.index)
            if len(common)>20:
                r=da.reindex(common).corr(db.reindex(common))
                print(f"    {a} vs {b}: {r:.3f}")
                corr_data[f'{a}_vs_{b}']=round(r,3)
    save_batch(7, {'combos':results_combos,'marginal':results_marginal,'correlations':corr_data})
    return {'combos':results_combos,'marginal':results_marginal}

# ═══════════════════════════════════════════════════════════════
# BATCH 8: Trailing Stop Optimization
# ═══════════════════════════════════════════════════════════════
def batch_8(h1):
    print(f"\n{'='*120}\n  BATCH 8: Trailing Stop Optimization\n{'='*120}",flush=True)
    trail_acts = [0.06, 0.08, 0.10, 0.12, 0.14, 0.18, 0.22]
    trail_dists = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]
    results={}
    for name in STRAT_ORDER:
        print(f"\n  {name}:")
        best_sh=-999; best_ta=0; best_td=0; sweep=[]
        # No trail baseline
        cfg_nt=copy.deepcopy(LIVE_CONFIG[name]); cfg_nt['trail_act']=999; cfg_nt['trail_dist']=999
        if name=='L8_MAX': t_nt=bt_keltner(h1,cfg_nt)
        else: t_nt=BT[name](h1,cfg_nt)
        s_nt=_stats(t_nt)
        print(f"    No trail: Sharpe={s_nt['sharpe']:.3f}")
        # Grid
        for ta in trail_acts:
            for td in trail_dists:
                cfg_g=copy.deepcopy(LIVE_CONFIG[name]); cfg_g['trail_act']=ta; cfg_g['trail_dist']=td
                if name=='L8_MAX': t_g=bt_keltner(h1,cfg_g)
                else: t_g=BT[name](h1,cfg_g)
                s_g=_stats(t_g)
                sweep.append({'ta':ta,'td':td,'sharpe':s_g['sharpe'],'pnl':s_g['pnl']})
                if s_g['sharpe']>best_sh: best_sh=s_g['sharpe']; best_ta=ta; best_td=td
        cur_ta=LIVE_CONFIG[name]['trail_act']; cur_td=LIVE_CONFIG[name]['trail_dist']
        cur_idx = next((i for i,s in enumerate(sweep) if s['ta']==cur_ta and s['td']==cur_td), -1)
        cur_sh = sweep[cur_idx]['sharpe'] if cur_idx>=0 else 0
        print(f"    Current ({cur_ta}/{cur_td}): Sharpe={cur_sh:.3f}")
        print(f"    Best ({best_ta}/{best_td}): Sharpe={best_sh:.3f} (d={best_sh-cur_sh:+.3f})")
        results[name]={'no_trail':s_nt['sharpe'],'current_sharpe':round(cur_sh,3),
                       'best_ta':best_ta,'best_td':best_td,'best_sharpe':round(best_sh,3),
                       'improvement':round(best_sh-cur_sh,3)}
    save_batch(8, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 9: SL/TP Re-optimization
# ═══════════════════════════════════════════════════════════════
def batch_9(h1):
    print(f"\n{'='*120}\n  BATCH 9: SL/TP Re-optimization\n{'='*120}",flush=True)
    sl_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    tp_range = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    results={}
    for name in STRAT_ORDER:
        print(f"\n  {name} (current SL={LIVE_CONFIG[name]['sl']}, TP={LIVE_CONFIG[name]['tp']}):")
        best_sh=-999; best_sl=0; best_tp=0; sweep=[]
        for sl in sl_range:
            for tp in tp_range:
                if tp<=sl: continue
                cfg_g=copy.deepcopy(LIVE_CONFIG[name]); cfg_g['sl']=sl; cfg_g['tp']=tp
                if name=='L8_MAX': t_g=bt_keltner(h1,cfg_g)
                else: t_g=BT[name](h1,cfg_g)
                s_g=_stats(t_g)
                sweep.append({'sl':sl,'tp':tp,'sharpe':s_g['sharpe'],'pnl':s_g['pnl'],'n':s_g['n']})
                if s_g['sharpe']>best_sh: best_sh=s_g['sharpe']; best_sl=sl; best_tp=tp
        cur_sl=LIVE_CONFIG[name]['sl']; cur_tp=LIVE_CONFIG[name]['tp']
        cur_idx=next((i for i,s in enumerate(sweep) if s['sl']==cur_sl and s['tp']==cur_tp),-1)
        cur_sh=sweep[cur_idx]['sharpe'] if cur_idx>=0 else 0
        print(f"    Current ({cur_sl}/{cur_tp}): Sharpe={cur_sh:.3f}")
        print(f"    Best ({best_sl}/{best_tp}): Sharpe={best_sh:.3f} (d={best_sh-cur_sh:+.3f})")
        # Top 5
        sweep.sort(key=lambda x:-x['sharpe'])
        for s in sweep[:5]:
            print(f"      SL={s['sl']}, TP={s['tp']}: Sharpe={s['sharpe']:.3f}, N={s['n']}")
        results[name]={'current_sl':cur_sl,'current_tp':cur_tp,'current_sharpe':round(cur_sh,3),
                       'best_sl':best_sl,'best_tp':best_tp,'best_sharpe':round(best_sh,3),
                       'top5':sweep[:5]}
    save_batch(9, results)
    return results

# ═══════════════════════════════════════════════════════════════
# BATCH 10: Cooldown & Circuit Breaker
# ═══════════════════════════════════════════════════════════════
def batch_10(h1):
    print(f"\n{'='*120}\n  BATCH 10: Cooldown & Circuit Breaker\n{'='*120}",flush=True)
    # Simulate cooldown by filtering trades that are too close after a loss
    all_t = run_all(h1, LIVE_CONFIG)
    merged = port_merge(all_t)
    merged.sort(key=lambda t: t['entry_time'])

    cooldowns = [0, 15, 30, 45, 60, 90, 120, 180]
    print(f"\n  Cooldown sweep (minutes):")
    print(f"  {'CD_min':>7} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Filtered':>8}")
    results_cd=[]
    for cd in cooldowns:
        filtered=[]; last_loss_time=None
        for t in merged:
            if last_loss_time and cd>0:
                gap=(pd.Timestamp(t['entry_time'])-last_loss_time).total_seconds()/60
                if gap<cd: continue
            filtered.append(t)
            if t['pnl']<0: last_loss_time=pd.Timestamp(t['exit_time'])
        s=_stats(filtered)
        n_filt=len(merged)-len(filtered)
        print(f"  {cd:>7} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {n_filt:>8}")
        results_cd.append({'cd':cd,**s,'filtered_n':n_filt})

    # Circuit breaker
    cb_configs = [(2,1),(3,1),(3,2),(4,2),(5,2)]
    print(f"\n  Circuit breaker (consec_losses, pause_hours):")
    print(f"  {'Config':>10} {'N':>6} {'Sharpe':>7} {'PnL':>10}")
    results_cb=[]
    for max_consec, pause_h in cb_configs:
        filtered=[]; consec=0; skip_until=None
        for t in merged:
            if skip_until and pd.Timestamp(t['entry_time'])<skip_until: continue
            skip_until=None
            filtered.append(t)
            if t['pnl']<0:
                consec+=1
                if consec>=max_consec:
                    skip_until=pd.Timestamp(t['exit_time'])+pd.Timedelta(hours=pause_h)
                    consec=0
            else: consec=0
        s=_stats(filtered)
        label=f"{max_consec}L/{pause_h}h"
        print(f"  {label:>10} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f}")
        results_cb.append({'max_consec':max_consec,'pause_h':pause_h,**s})

    # Loss streak analysis
    streaks=[]; cur=0
    for t in merged:
        if t['pnl']<0: cur+=1
        else:
            if cur>0: streaks.append(cur)
            cur=0
    if cur>0: streaks.append(cur)
    if streaks:
        print(f"\n  Loss streaks: max={max(streaks)}, mean={np.mean(streaks):.1f}, "
              f"3+={sum(1 for s in streaks if s>=3)}, 5+={sum(1 for s in streaks if s>=5)}")
    save_batch(10, {'cooldown':results_cd,'circuit_breaker':results_cb,
                    'streak_max':max(streaks) if streaks else 0,
                    'streak_3plus':sum(1 for s in streaks if s>=3) if streaks else 0})
    return results_cd

# ═══════════════════════════════════════════════════════════════
# BATCH 11: Monte Carlo Stress Test
# ═══════════════════════════════════════════════════════════════
def batch_11(h1):
    print(f"\n{'='*120}\n  BATCH 11: Monte Carlo Stress Test\n{'='*120}",flush=True)
    all_t=run_all(h1,LIVE_CONFIG); _,daily=port_stats(all_t)
    daily_v=daily.values

    # PnL bootstrap
    N_BOOT=1000
    boot_sharpes=[]; boot_pnls=[]; boot_dds=[]
    for _ in range(N_BOOT):
        sample=np.random.choice(daily_v,size=len(daily_v),replace=True)
        s_d=pd.Series(sample)
        sh=float(s_d.mean()/s_d.std()*np.sqrt(252)) if s_d.std()>0 else 0
        boot_sharpes.append(sh)
        boot_pnls.append(float(s_d.sum()))
        eq=s_d.cumsum(); boot_dds.append(float((np.maximum.accumulate(eq)-eq).max()))
    print(f"\n  PnL Bootstrap ({N_BOOT} resamples):")
    print(f"    Sharpe: mean={np.mean(boot_sharpes):.3f}, 5%={np.percentile(boot_sharpes,5):.3f}, "
          f"95%={np.percentile(boot_sharpes,95):.3f}")
    print(f"    PnL: mean=${np.mean(boot_pnls):,.0f}, 5%=${np.percentile(boot_pnls,5):,.0f}, "
          f"95%=${np.percentile(boot_pnls,95):,.0f}")
    print(f"    MaxDD: mean=${np.mean(boot_dds):,.0f}, 95%=${np.percentile(boot_dds,95):,.0f}")

    # Parameter perturbation
    N_PERTURB=200
    perturb_sharpes=[]
    for _ in range(N_PERTURB):
        cfg_p=copy.deepcopy(LIVE_CONFIG)
        for name in STRAT_ORDER:
            for param in ['sl','tp','trail_act','trail_dist']:
                cfg_p[name][param]*=(1+np.random.uniform(-0.10,0.10))
        all_tp=run_all(h1,cfg_p); sp,_=port_stats(all_tp)
        perturb_sharpes.append(sp['sharpe'])
    print(f"\n  Parameter Perturbation (+/-10%, {N_PERTURB} runs):")
    print(f"    Sharpe: mean={np.mean(perturb_sharpes):.3f}, std={np.std(perturb_sharpes):.3f}, "
          f"min={min(perturb_sharpes):.3f}, 5%={np.percentile(perturb_sharpes,5):.3f}")
    base_sh=_stats(port_merge(run_all(h1,LIVE_CONFIG)))['sharpe']
    degradation=sum(1 for s in perturb_sharpes if s<base_sh*0.8)/N_PERTURB*100
    print(f"    Degradation >20%: {degradation:.1f}% of runs")

    # Drawdown analysis
    eq=daily.cumsum(); dd=np.maximum.accumulate(eq)-eq
    dd_periods=[]; in_dd=False; dd_start=None
    for idx,val in dd.items():
        if val>0 and not in_dd: in_dd=True; dd_start=idx
        elif val==0 and in_dd: in_dd=False; dd_periods.append((dd_start,idx,(idx-dd_start).days))
    if dd_periods:
        durations=[d[2] for d in dd_periods]
        print(f"\n  Drawdown durations: max={max(durations)} days, mean={np.mean(durations):.0f}, "
              f"median={np.median(durations):.0f}")

    save_batch(11, {'bootstrap':{'sharpe_mean':round(np.mean(boot_sharpes),3),
                                  'sharpe_5pct':round(np.percentile(boot_sharpes,5),3),
                                  'sharpe_95pct':round(np.percentile(boot_sharpes,95),3),
                                  'pnl_mean':round(np.mean(boot_pnls),0),
                                  'maxdd_95pct':round(np.percentile(boot_dds,95),0)},
                    'perturbation':{'mean':round(np.mean(perturb_sharpes),3),
                                    'std':round(np.std(perturb_sharpes),3),
                                    'min':round(min(perturb_sharpes),3),
                                    'degradation_pct':round(degradation,1)},
                    'dd_max_days':max(durations) if dd_periods else 0})
    return {'bootstrap_sharpe_5pct':round(np.percentile(boot_sharpes,5),3)}

# ═══════════════════════════════════════════════════════════════
# BATCH 12: Combined Optimal Config
# ═══════════════════════════════════════════════════════════════
def batch_12(h1, b2, b5, b8, b9):
    print(f"\n{'='*120}\n  BATCH 12: Combined Optimal Config\n{'='*120}",flush=True)
    pctl=compute_atr_pctl(compute_atr(h1),lb=300)
    # Build optimal config from batch winners
    opt=copy.deepcopy(LIVE_CONFIG)
    for name in STRAT_ORDER:
        if name in b5 and b5[name]['best_sharpe']>b5[name].get('current_sharpe',0):
            opt[name]['max_hold']=b5[name]['best_mh']
        if name in b8 and b8[name]['improvement']>0.05:
            opt[name]['trail_act']=b8[name]['best_ta']
            opt[name]['trail_dist']=b8[name]['best_td']
        if name in b9 and b9[name]['best_sharpe']-b9[name]['current_sharpe']>0.05:
            opt[name]['sl']=b9[name]['best_sl']
            opt[name]['tp']=b9[name]['best_tp']

    best_adx=None
    if b2.get('validation',{}).get('kf_wins',0)>=3:
        best_label=b2['validation']['best']
        sessions_map={
            'R178_intended':{'asia':14,'london':10,'ny':16,'evening':14},
            'london8_ny18':{'asia':14,'london':8,'ny':18,'evening':14},
            'london12_ny14':{'asia':14,'london':12,'ny':14,'evening':14},
            'asia10_lon8_ny20_eve16':{'asia':10,'london':8,'ny':20,'evening':16},
        }
        best_adx=sessions_map.get(best_label)

    configs={
        'A_current_live': (LIVE_CONFIG, None, None, 0),
        'B_live+r187_all': (LIVE_CONFIG, None, pctl, 30),
        'C_optimized': (opt, best_adx, None, 0),
        'D_optimized+r187': (opt, best_adx, pctl, 30),
    }

    print(f"\n  Optimized params (changes from live):")
    for name in STRAT_ORDER:
        changes=[]
        for k in ['sl','tp','trail_act','trail_dist','max_hold']:
            if opt[name][k]!=LIVE_CONFIG[name][k]:
                changes.append(f"{k}: {LIVE_CONFIG[name][k]}->{opt[name][k]}")
        if changes: print(f"    {name}: {', '.join(changes)}")
    if best_adx: print(f"    Session ADX: {best_adx}")

    print(f"\n  {'Config':<22} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'Cap%':>6}")
    results={}
    for label,(cfg,adx_s,pctl_s,pf) in configs.items():
        all_t=run_all(h1,cfg,adx_by_session=adx_s,pctl_v=pctl_s,pctl_f=pf)
        s,_=port_stats(all_t)
        print(f"  {label:<22} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['cap_pct']:>5.1f}%")
        results[label]=s

    # K-Fold for D (best combined)
    def run_d(h1f):
        all_t=run_all(h1f,opt,adx_by_session=best_adx,pctl_v=pctl.reindex(h1f.index) if pctl is not None else None,pctl_f=30)
        s,_=port_stats(all_t); return s['sharpe']
    def run_a(h1f):
        all_t=run_all(h1f,LIVE_CONFIG); s,_=port_stats(all_t); return s['sharpe']
    kf_d=kfold_test(h1,run_d); kf_a=kfold_test(h1,run_a)
    kf_wins=sum(1 for d,a in zip(kf_d,kf_a) if d['sharpe']>a['sharpe'])
    print(f"\n  K-Fold: D wins {kf_wins}/{len(kf_d)}")
    for d,a in zip(kf_d,kf_a):
        print(f"    {d['period']}: A={a['sharpe']:.3f}, D={d['sharpe']:.3f} ({'D' if d['sharpe']>a['sharpe'] else 'A'})")

    # Walk-Forward
    wf_d=wf_test(h1,run_d); wf_a=wf_test(h1,run_a)
    wf_wins=sum(1 for d,a in zip(wf_d,wf_a) if d['sharpe']>a['sharpe'])
    print(f"\n  Walk-Forward: D wins {wf_wins}/{len(wf_d)}")

    # Era
    print(f"\n  Era validation:")
    for era in ['full','hike','cut','recent_3y']:
        ta=run_all(h1,LIVE_CONFIG); td_t=run_all(h1,opt,adx_by_session=best_adx,pctl_v=pctl,pctl_f=30)
        ea=[t for nm in STRAT_ORDER for t in filter_era(ta[nm],era)]
        ed=[t for nm in STRAT_ORDER for t in filter_era(td_t[nm],era)]
        sa=_sharpe(_daily(ea)); sd=_sharpe(_daily(ed))
        print(f"    {era:<12}: A={sa:.3f}, D={sd:.3f} (d={sd-sa:+.3f})")
        results[f'era_{era}']={'current':round(sa,3),'optimized':round(sd,3)}

    # Yearly
    print(f"\n  Yearly stability:")
    _,da=port_stats(run_all(h1,LIVE_CONFIG))
    _,dd=port_stats(run_all(h1,opt,adx_by_session=best_adx,pctl_v=pctl,pctl_f=30))
    years=sorted(set(da.index.year)|set(dd.index.year))
    for yr in years:
        ya=da[da.index.year==yr]; yd=dd[dd.index.year==yr]
        print(f"    {yr}: A=${float(ya.sum()):>9,.0f} (Sh={_sharpe(ya):.2f}), D=${float(yd.sum()):>9,.0f} (Sh={_sharpe(yd):.2f})")

    results['validation']={'kf_wins':kf_wins,'kf_total':len(kf_d),'wf_wins':wf_wins,'wf_total':len(wf_d)}
    results['optimized_config']={name:{k:v for k,v in opt[name].items()} for name in STRAT_ORDER}
    if best_adx: results['session_adx']=best_adx
    save_batch(12, results)
    return results


# ═══════════════════════════════════════════════════════════════
def main():
    print("="*120)
    print("  R191 — 10-Hour Mega Test Suite (12 Batches)")
    print("="*120, flush=True)

    h1=load_h1()
    atr_s=compute_atr(h1).dropna()
    print(f"  ATR: mean=${atr_s.mean():.2f}, median=${atr_s.median():.2f}, current=${atr_s.iloc[-1]:.2f}")

    b1=batch_1(h1)
    b2=batch_2(h1)
    b3=batch_3(h1)
    b4=batch_4(h1)
    b5=batch_5(h1)
    b6=batch_6(h1)
    b7=batch_7(h1)
    b8=batch_8(h1)
    b9=batch_9(h1)
    b10=batch_10(h1)
    b11=batch_11(h1)
    b12=batch_12(h1, b2, b5, b8, b9)

    elapsed=time.time()-t0
    print(f"\n{'='*120}")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min, {elapsed/3600:.1f}hr)")
    print(f"{'='*120}")

    master={'runtime_s':round(elapsed,1),'batches_completed':12}
    with open(OUTPUT_DIR/"r191_results.json",'w') as f: json.dump(master,f,indent=2,default=str)
    print(f"  Master saved: {OUTPUT_DIR/'r191_results.json'}")

if __name__=="__main__":
    main()
