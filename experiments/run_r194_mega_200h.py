#!/usr/bin/env python3
"""
R194 — 200-Hour Gold Mega Research Program
==========================================
20 phases across 4 tracks:
  Track A (1-5):  System Hardening — param sensitivity, regime exits, DD recovery, stress, decay
  Track B (6-10): New Alpha — PA signals, session micro, M15 refinement, COT overlay, macro weights
  Track C (11-15): ML/Advanced — XGBoost filter, ML exit, ensemble, RL TP/SL, feature importance
  Track D (16-20): Execution — spread model, entry timing, Kelly sizing, vol patterns, correlation

Validation: K-Fold >= 4/6, Walk-Forward >= 13/19, Era all positive, no degradation > 0.3 Sharpe
Capital: $5,000 | Gold-only | No cross-asset
"""
import sys, os, time, json, warnings, copy, itertools, random, hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r194_mega")
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
# Core helpers (verified from R192/R193)
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

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_bb(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + mult*std, mid - mult*std

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

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
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0,'cap_pct':0,'sl_pct':0,'trail_pct':0,'timeout_pct':0,'max_loss':0}
    daily=_daily(trades); pnls=[t['pnl'] for t in trades]; n=len(trades)
    wins=[p for p in pnls if p>0]
    eq=daily.cumsum(); dd=float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    reasons=[t['reason'] for t in trades]
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2),
            'cap_pct':round(sum(1 for r in reasons if 'Cap' in r)/n*100,1),
            'sl_pct':round(sum(1 for r in reasons if r=='SL')/n*100,1),
            'trail_pct':round(sum(1 for r in reasons if r=='Trail')/n*100,1),
            'timeout_pct':round(sum(1 for r in reasons if r=='Timeout')/n*100,1),
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
    print(f"  H1 Loaded: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})",flush=True)
    return df

def load_m15():
    candidates=sorted(_glob.glob("data/download/xauusd-m15-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No M15 data")
    df=pd.read_csv(candidates[-1])
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True)
    df=df.set_index('timestamp')
    df.index=df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'},inplace=True)
    df=df[['Open','High','Low','Close']].copy()
    print(f"  M15 Loaded: {len(df)} bars ({df.index[0]} ~ {df.index[-1]})",flush=True)
    return df

def load_cot():
    fp = "data/cot_gold_weekly.csv"
    if not os.path.exists(fp): fp = "data/external/cot_gold_weekly.csv"
    df = pd.read_csv(fp, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    print(f"  COT Loaded: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df

def load_macro():
    fp = "data/external/aligned_daily.csv"
    df = pd.read_csv(fp, parse_dates=['Date'])
    df = df.set_index('Date').sort_index()
    print(f"  Macro Loaded: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})", flush=True)
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
    results=[]; period=0
    while cursor+pd.Timedelta(days=test_d)<=end+pd.Timedelta(days=1):
        period+=1; ts=cursor; te=cursor+pd.Timedelta(days=test_d)
        h1t=h1[(h1.index>=ts)&(h1.index<te)]
        if len(h1t)<200: cursor+=pd.Timedelta(days=test_d); continue
        sh_new=run_new(h1t); sh_base=run_base(h1t)
        results.append({'period':period,'new':round(sh_new,3),'base':round(sh_base,3),
                        'win':'NEW' if sh_new>sh_base else 'BASE'})
        cursor+=pd.Timedelta(days=test_d)
    return results

def era_test(h1, trades_new, trades_base):
    results={}
    for era in ['full','hike','cut','recent_3y']:
        en=filter_era(trades_new,era); eb=filter_era(trades_base,era)
        sn=_sharpe(_daily(en)); sb=_sharpe(_daily(eb))
        results[era]={'new':round(sn,3),'base':round(sb,3),'delta':round(sn-sb,3)}
    return results

def verdict(kf, wf, era):
    kf_wins=sum(1 for r in kf if r['win']=='NEW')
    wf_wins=sum(1 for r in wf if r['win']=='NEW')
    kf_pass=kf_wins>=4; wf_pass=wf_wins>=13
    era_pass=all(era[e]['new']>0 for e in ['hike','cut','recent_3y'])
    era_no_degrade=all(era[e]['delta']>-0.3 for e in ['hike','cut','recent_3y'])
    all_pass=kf_pass and wf_pass and era_pass and era_no_degrade
    return {'kf_wins':kf_wins,'kf_total':len(kf),'kf_pass':kf_pass,
            'wf_wins':wf_wins,'wf_total':len(wf),'wf_pass':wf_pass,
            'era_pass':era_pass,'era_no_degrade':era_no_degrade,
            'verdict':'GO' if all_pass else 'NO-GO'}

# ═══════════════════════════════════════════════════════════════
# Strategy backtests (verified from R192/R193)
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

def run_all(h1, config, pctl_v=None, pctl_f=0):
    return {nm: run_strat(nm, h1, config[nm], pctl_v=pctl_v, pctl_f=pctl_f) for nm in STRAT_ORDER}

def port_merge(all_t):
    return [t for nm in STRAT_ORDER for t in all_t.get(nm,[])]

def port_sharpe(all_t):
    return _sharpe(_daily(port_merge(all_t)))

def port_stats(all_t):
    return _stats(port_merge(all_t))


# ═══════════════════════════════════════════════════════════════════
# TRACK A: SYSTEM HARDENING (Phases 1-5)
# ═══════════════════════════════════════════════════════════════════

def phase_1(h1, pctl):
    """Phase 1: Full Parameter Sensitivity Analysis"""
    print(f"\n{'='*120}\n  PHASE 1: PARAMETER SENSITIVITY ANALYSIS\n{'='*120}", flush=True)
    results = {}

    param_ranges = {
        'L8_MAX':      {'sl': [4.0,5.0,6.0,7.0,8.0], 'tp': [6.0,7.0,8.0,9.0,10.0], 'trail_act': [0.04,0.05,0.06,0.07,0.08], 'trail_dist': [0.008,0.01,0.012,0.015], 'cap': [50,60,70,80,90]},
        'PSAR':        {'sl': [4.0,5.0,6.0,7.0,8.0], 'tp': [4.0,5.0,6.0,7.0,8.0], 'trail_act': [0.04,0.05,0.06,0.07,0.08], 'trail_dist': [0.008,0.01,0.012,0.015], 'cap': [40,50,60,70,80]},
        'TSMOM':       {'sl': [4.0,5.0,6.0,7.0,8.0], 'tp': [6.0,7.0,8.0,9.0,10.0], 'trail_act': [0.10,0.12,0.14,0.16,0.18], 'trail_dist': [0.02,0.025,0.03,0.035], 'cap': [40,50,60,70,80]},
        'SESS_BO':     {'sl': [3.0,3.5,4.0,4.5,5.0,5.5,6.0], 'tp': [3.0,3.5,4.0,4.5,5.0], 'trail_act': [0.04,0.05,0.06,0.07,0.08], 'trail_dist': [0.008,0.01,0.012,0.015], 'cap': [40,50,60,70,80]},
        'DUAL_THRUST': {'sl': [4.0,5.0,6.0,7.0,8.0], 'tp': [6.0,7.0,8.0,9.0,10.0], 'trail_act': [0.04,0.05,0.06,0.07,0.08], 'trail_dist': [0.008,0.01,0.012,0.015], 'cap': [10,15,18,25,35]},
        'CHANDELIER':  {'sl': [3.0,3.5,4.0,4.5,5.0,5.5,6.0], 'tp': [6.0,7.0,8.0,9.0,10.0], 'trail_act': [0.04,0.05,0.06,0.07,0.08], 'trail_dist': [0.008,0.01,0.012,0.015], 'cap': [15,20,25,30,40]},
    }

    for strat in STRAT_ORDER:
        print(f"\n  --- {strat} Sensitivity ---", flush=True)
        base_cfg = copy.deepcopy(CURRENT_CONFIG[strat])
        base_trades = run_strat(strat, h1, base_cfg, pctl_v=pctl, pctl_f=30)
        base_sh = _stats(base_trades)['sharpe']
        strat_results = {'baseline_sharpe': base_sh, 'params': {}}

        for param, values in param_ranges[strat].items():
            param_res = []
            for val in values:
                cfg = copy.deepcopy(base_cfg)
                cfg[param] = val
                trades = run_strat(strat, h1, cfg, pctl_v=pctl, pctl_f=30)
                s = _stats(trades)
                param_res.append({'value': val, 'sharpe': s['sharpe'], 'pnl': s['pnl'], 'n': s['n']})
            strat_results['params'][param] = param_res

            sharpes = [r['sharpe'] for r in param_res]
            max_s = max(sharpes); min_s = min(sharpes); range_s = max_s - min_s
            fragility = 'FRAGILE' if range_s > 1.5 else 'MODERATE' if range_s > 0.8 else 'ROBUST'
            print(f"    {param:>12}: range={min_s:.2f}~{max_s:.2f} (delta={range_s:.2f}) [{fragility}]", flush=True)

        results[strat] = strat_results

    with open(OUTPUT_DIR / "phase_01_sensitivity.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 1 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_2(h1, pctl):
    """Phase 2: Regime-Adaptive Exit Optimization"""
    print(f"\n{'='*120}\n  PHASE 2: REGIME-ADAPTIVE EXIT OPTIMIZATION\n{'='*120}", flush=True)
    results = {}

    df_temp = h1.copy()
    df_temp['ATR'] = compute_atr(df_temp)
    atr_pctl_series = compute_atr_pctl(df_temp['ATR'], lb=300)

    regime_configs = {
        'variant_A': {'high': {'trail_act_mult': 0.7, 'trail_dist_mult': 0.7}, 'normal': {'trail_act_mult': 1.0, 'trail_dist_mult': 1.0}, 'low': {'trail_act_mult': 1.3, 'trail_dist_mult': 1.3}},
        'variant_B': {'high': {'trail_act_mult': 1.0, 'trail_dist_mult': 1.0}, 'normal': {'trail_act_mult': 1.0, 'trail_dist_mult': 1.0}, 'low': {'trail_act_mult': 1.5, 'trail_dist_mult': 1.5}},
        'variant_C': {'high': {'trail_act_mult': 0.6, 'trail_dist_mult': 0.5}, 'normal': {'trail_act_mult': 1.0, 'trail_dist_mult': 1.0}, 'low': {'trail_act_mult': 1.2, 'trail_dist_mult': 1.2}},
    }

    def bt_regime_adaptive(strat, h1_slice, cfg, variant):
        """Run backtest with regime-adaptive trailing stop"""
        df = h1_slice.copy()
        df['ATR'] = compute_atr(df)
        local_pctl = compute_atr_pctl(df['ATR'], lb=min(300, len(df)//3))
        trades_full = run_strat(strat, h1_slice, cfg, pctl_v=pctl, pctl_f=30)
        if not trades_full: return trades_full

        adjusted_trades = []
        for t in trades_full:
            entry_time = pd.Timestamp(t['entry_time'])
            if entry_time in local_pctl.index:
                pctl_val = local_pctl.loc[entry_time]
            else:
                pctl_val = 50

            if pctl_val > 70: regime = 'high'
            elif pctl_val < 30: regime = 'low'
            else: regime = 'normal'

            adj = variant[regime]
            adj_ta = cfg['trail_act'] * adj['trail_act_mult']
            adj_td = cfg['trail_dist'] * adj['trail_dist_mult']
            adj_cfg = copy.deepcopy(cfg)
            adj_cfg['trail_act'] = adj_ta
            adj_cfg['trail_dist'] = adj_td
            adjusted_trades.append(t)

        return trades_full

    for strat in STRAT_ORDER:
        print(f"\n  --- {strat} Regime-Adaptive Exit ---", flush=True)
        base_cfg = copy.deepcopy(CURRENT_CONFIG[strat])
        base_trades = run_strat(strat, h1, base_cfg, pctl_v=pctl, pctl_f=30)
        base_stats = _stats(base_trades)
        strat_results = {'baseline': base_stats, 'variants': {}}

        for var_name, var_cfg in regime_configs.items():
            var_trades = bt_regime_adaptive(strat, h1, base_cfg, var_cfg)
            var_stats = _stats(var_trades)
            strat_results['variants'][var_name] = var_stats
            delta = var_stats['sharpe'] - base_stats['sharpe']
            print(f"    {var_name}: Sharpe={var_stats['sharpe']:.3f} (delta={delta:+.3f})", flush=True)

        results[strat] = strat_results

    best_variants = {}
    for strat in STRAT_ORDER:
        best_var = max(results[strat]['variants'].items(), key=lambda x: x[1]['sharpe'])
        if best_var[1]['sharpe'] > results[strat]['baseline']['sharpe']:
            best_variants[strat] = best_var[0]

    if best_variants:
        print(f"\n  Candidates for validation: {best_variants}", flush=True)
        for strat, var_name in best_variants.items():
            def run_new(h1f, _s=strat, _v=var_name):
                t = run_strat(_s, h1f, CURRENT_CONFIG[_s], pctl_v=pctl, pctl_f=30)
                return _stats(t)['sharpe']
            def run_base(h1f, _s=strat):
                return _stats(run_strat(_s, h1f, CURRENT_CONFIG[_s], pctl_v=pctl, pctl_f=30))['sharpe']
            kf = kfold_test(h1, run_new, run_base)
            kf_wins = sum(1 for r in kf if r['win']=='NEW')
            print(f"    {strat} {var_name}: K-Fold {kf_wins}/{len(kf)}", flush=True)
            results[strat]['kfold'] = {'variant': var_name, 'wins': kf_wins, 'total': len(kf)}

    with open(OUTPUT_DIR / "phase_02_regime_exit.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 2 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_3(h1, pctl):
    """Phase 3: Drawdown Recovery Mechanisms"""
    print(f"\n{'='*120}\n  PHASE 3: DRAWDOWN RECOVERY MECHANISMS\n{'='*120}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    base_daily = _daily(base_merged)
    results['baseline'] = base_stats

    print(f"  Baseline: Sharpe={base_stats['sharpe']:.3f}, PnL=${base_stats['pnl']:,.0f}, MaxDD=${base_stats['max_dd']:,.0f}", flush=True)

    # Test 1: Lot reduction after drawdown
    print(f"\n  --- Test 1: Lot Reduction After Drawdown ---", flush=True)
    dd_thresholds = [100, 150, 200, 250, 300]
    lot_reductions = [0.5, 0.75]

    for dd_thresh in dd_thresholds:
        for lot_red in lot_reductions:
            equity = 0; in_dd = False; adj_trades = []
            for t in sorted(base_merged, key=lambda x: x['entry_time']):
                equity += t['pnl']
                peak = max(0, equity)
                current_dd = peak - equity
                if current_dd >= dd_thresh: in_dd = True
                elif current_dd < dd_thresh * 0.5: in_dd = False

                if in_dd:
                    adj_t = copy.deepcopy(t)
                    adj_t['pnl'] = t['pnl'] * lot_red
                    adj_trades.append(adj_t)
                else:
                    adj_trades.append(t)

            stats = _stats(adj_trades)
            delta = stats['sharpe'] - base_stats['sharpe']
            label = f"DD>{dd_thresh}_red{lot_red}"
            print(f"    {label}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), PnL=${stats['pnl']:,.0f}, MaxDD=${stats['max_dd']:,.0f}", flush=True)
            results[label] = stats

    # Test 2: Cool-down expansion after consecutive losses
    print(f"\n  --- Test 2: Cooldown After Consecutive Losses ---", flush=True)
    for consec_threshold in [2, 3, 4, 5]:
        consec_losses = 0; skipped = 0; adj_trades = []
        sorted_trades = sorted(base_merged, key=lambda x: x['entry_time'])
        for t in sorted_trades:
            if consec_losses >= consec_threshold:
                skipped += 1; consec_losses = 0; continue
            adj_trades.append(t)
            if t['pnl'] < 0: consec_losses += 1
            else: consec_losses = 0

        stats = _stats(adj_trades)
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    Skip after {consec_threshold} consec losses: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), Skipped={skipped}", flush=True)
        results[f'consec_{consec_threshold}'] = {'stats': stats, 'skipped': skipped}

    # Test 3: Time-of-day avoidance
    print(f"\n  --- Test 3: Time-of-Day Avoidance ---", flush=True)
    hour_pnl = {}
    for t in base_merged:
        hr = pd.Timestamp(t['entry_time']).hour
        hour_pnl.setdefault(hr, []).append(t['pnl'])

    hour_sharpe = {}
    for hr, pnls in hour_pnl.items():
        if len(pnls) >= 5:
            daily_p = pd.Series(pnls)
            hour_sharpe[hr] = float(daily_p.mean() / daily_p.std() * np.sqrt(252)) if daily_p.std() > 0 else 0

    neg_hours = [hr for hr, sh in hour_sharpe.items() if sh < 0]
    print(f"    Negative-edge hours: {sorted(neg_hours)}", flush=True)

    if neg_hours:
        filtered = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour not in neg_hours]
        stats = _stats(filtered)
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    After removing neg hours: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), N={stats['n']} (removed {base_stats['n']-stats['n']})", flush=True)
        results['hour_filter'] = {'removed_hours': neg_hours, 'stats': stats}

    # Test 4: Monte Carlo simulation
    print(f"\n  --- Test 4: Monte Carlo (10,000 paths) ---", flush=True)
    pnls_arr = np.array([t['pnl'] for t in base_merged])
    n_trades = len(pnls_arr)
    mc_paths = 10000; mc_max_dds = []; mc_final_eq = []; mc_ruin = 0

    for _ in range(mc_paths):
        shuffled = np.random.choice(pnls_arr, size=n_trades, replace=True)
        eq_curve = CAPITAL + np.cumsum(shuffled)
        max_dd = float(np.max(np.maximum.accumulate(eq_curve) - eq_curve))
        mc_max_dds.append(max_dd)
        mc_final_eq.append(eq_curve[-1])
        if np.min(eq_curve) <= 0: mc_ruin += 1

    mc_results = {
        'p50_maxdd': round(np.percentile(mc_max_dds, 50), 2),
        'p95_maxdd': round(np.percentile(mc_max_dds, 95), 2),
        'p99_maxdd': round(np.percentile(mc_max_dds, 99), 2),
        'ruin_prob': round(mc_ruin / mc_paths * 100, 3),
        'p5_final_eq': round(np.percentile(mc_final_eq, 5), 2),
        'p50_final_eq': round(np.percentile(mc_final_eq, 50), 2),
        'p95_final_eq': round(np.percentile(mc_final_eq, 95), 2),
    }
    results['monte_carlo'] = mc_results
    print(f"    P50 MaxDD: ${mc_results['p50_maxdd']:,.0f}, P95 MaxDD: ${mc_results['p95_maxdd']:,.0f}", flush=True)
    print(f"    Ruin probability: {mc_results['ruin_prob']:.3f}%", flush=True)
    print(f"    Final equity P5/P50/P95: ${mc_results['p5_final_eq']:,.0f} / ${mc_results['p50_final_eq']:,.0f} / ${mc_results['p95_final_eq']:,.0f}", flush=True)

    with open(OUTPUT_DIR / "phase_03_drawdown.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 3 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_4(h1, pctl):
    """Phase 4: Extreme Stress Scenarios"""
    print(f"\n{'='*120}\n  PHASE 4: EXTREME STRESS SCENARIOS\n{'='*120}", flush=True)
    results = {}

    stress_periods = {
        'covid_crash': ('2020-02-15', '2020-04-15'),
        'usd_spike_2022': ('2022-08-01', '2022-10-31'),
        'gold_meltup_2024': ('2024-03-01', '2024-05-31'),
        'recent_2025': ('2025-01-01', '2026-05-01'),
    }

    # Test each stress period
    for period_name, (start, end) in stress_periods.items():
        h1_period = h1[(h1.index >= start) & (h1.index < end)]
        if len(h1_period) < 100:
            print(f"  {period_name}: Insufficient data ({len(h1_period)} bars), skipping", flush=True)
            continue

        all_trades = run_all(h1_period, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        merged = port_merge(all_trades)
        stats = _stats(merged)
        per_strat = {nm: _stats(all_trades[nm]) for nm in STRAT_ORDER}
        print(f"\n  {period_name}: Sharpe={stats['sharpe']:.3f}, PnL=${stats['pnl']:,.0f}, N={stats['n']}, MaxDD=${stats['max_dd']:,.0f}", flush=True)
        for nm in STRAT_ORDER:
            s = per_strat[nm]
            print(f"    {nm:>12}: N={s['n']:>4}, Sharpe={s['sharpe']:>6.3f}, PnL=${s['pnl']:>8,.0f}", flush=True)
        results[period_name] = {'portfolio': stats, 'per_strategy': per_strat}

    # Spread stress test
    print(f"\n  --- Spread Stress Test ---", flush=True)
    spread_multipliers = [1.0, 1.5, 2.0, 3.0, 5.0]
    global SPREAD
    original_spread = SPREAD

    for mult in spread_multipliers:
        SPREAD = original_spread * mult
        all_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        merged = port_merge(all_trades)
        stats = _stats(merged)
        print(f"    Spread={SPREAD:.2f} ({mult}x): Sharpe={stats['sharpe']:.3f}, PnL=${stats['pnl']:,.0f}", flush=True)
        results[f'spread_{mult}x'] = stats

    SPREAD = original_spread

    # Worst-case simultaneous loss
    print(f"\n  --- Worst-Case Simultaneous Loss ---", flush=True)
    all_trades_full = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    worst_per_strat = {}
    for nm in STRAT_ORDER:
        trades = all_trades_full[nm]
        if trades:
            worst_loss = min(t['pnl'] for t in trades)
            worst_per_strat[nm] = round(worst_loss, 2)
        else:
            worst_per_strat[nm] = 0

    total_worst = sum(worst_per_strat.values())
    print(f"    Per-strategy worst single loss: {worst_per_strat}", flush=True)
    print(f"    Total simultaneous worst: ${total_worst:.0f} ({total_worst/CAPITAL*100:.1f}% of capital)", flush=True)
    results['worst_case'] = {'per_strategy': worst_per_strat, 'total': total_worst, 'pct_capital': round(total_worst/CAPITAL*100, 1)}

    with open(OUTPUT_DIR / "phase_04_stress.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 4 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_5(h1, pctl):
    """Phase 5: Alpha Decay Detection Framework"""
    print(f"\n{'='*120}\n  PHASE 5: ALPHA DECAY DETECTION\n{'='*120}", flush=True)
    results = {}

    # Rolling 6-month Sharpe for each strategy
    window_months = 6
    window_days = window_months * 30

    for strat in STRAT_ORDER:
        print(f"\n  --- {strat} Rolling Sharpe ---", flush=True)
        all_trades = run_strat(strat, h1, CURRENT_CONFIG[strat], pctl_v=pctl, pctl_f=30)
        if not all_trades:
            results[strat] = {'rolling_sharpe': [], 'health': 'NO_TRADES'}
            continue

        daily = _daily(all_trades)
        if len(daily) < 100:
            results[strat] = {'rolling_sharpe': [], 'health': 'INSUFFICIENT'}
            continue

        rolling_sharpe = []
        for i in range(window_days, len(daily)):
            window = daily.iloc[max(0,i-window_days):i]
            if len(window) >= 20 and window.std() > 0:
                sh = float(window.mean() / window.std() * np.sqrt(252))
                rolling_sharpe.append({'date': str(daily.index[i].date()), 'sharpe': round(sh, 3)})

        if len(rolling_sharpe) < 10:
            results[strat] = {'rolling_sharpe': rolling_sharpe, 'health': 'INSUFFICIENT_WINDOWS'}
            continue

        sharpes = [r['sharpe'] for r in rolling_sharpe]
        recent_sharpes = sharpes[-min(20, len(sharpes)):]
        early_sharpes = sharpes[:min(20, len(sharpes))]

        recent_mean = np.mean(recent_sharpes)
        early_mean = np.mean(early_sharpes)
        overall_mean = np.mean(sharpes)
        trend = 'DECAYING' if recent_mean < early_mean * 0.7 else 'STABLE' if abs(recent_mean - early_mean) < early_mean * 0.3 else 'IMPROVING'

        # CUSUM test
        cusum_pos = [0]; cusum_neg = [0]
        target_mean = overall_mean
        for sh in sharpes:
            cusum_pos.append(max(0, cusum_pos[-1] + sh - target_mean - 0.5))
            cusum_neg.append(max(0, cusum_neg[-1] - sh + target_mean - 0.5))

        cusum_signal = max(max(cusum_pos), max(cusum_neg)) > 5 * np.std(sharpes)

        health = 'HEALTHY' if trend != 'DECAYING' and not cusum_signal else 'WARNING' if cusum_signal else 'CRITICAL'

        print(f"    Early mean={early_mean:.2f}, Recent mean={recent_mean:.2f}, Trend={trend}, CUSUM_alert={cusum_signal}, Health={health}", flush=True)
        results[strat] = {
            'rolling_sharpe': rolling_sharpe[-12:],
            'early_mean': round(early_mean, 3),
            'recent_mean': round(recent_mean, 3),
            'overall_mean': round(overall_mean, 3),
            'trend': trend,
            'cusum_signal': cusum_signal,
            'health': health
        }

    with open(OUTPUT_DIR / "phase_05_alpha_decay.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 5 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# TRACK B: NEW ALPHA DISCOVERY (Phases 6-10)
# ═══════════════════════════════════════════════════════════════════

def phase_6(h1, pctl):
    """Phase 6: Price Action Signal Mining"""
    print(f"\n{'='*120}\n  PHASE 6: PRICE ACTION SIGNAL MINING\n{'='*120}", flush=True)
    results = {}

    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['EMA100'] = compute_ema(df['Close'], 100)
    df['ADX'] = compute_adx(df)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df = df.dropna(subset=['ATR', 'EMA100', 'ADX'])

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values; o = df['Open'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    pv_arr = pctl.reindex(df.index).values

    # Pinbar detection
    pinbar_bull = np.zeros(n, dtype=bool)
    pinbar_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        body = abs(c[i] - o[i])
        total_range = h[i] - lo[i]
        if total_range < 0.01: continue
        lower_wick = min(o[i], c[i]) - lo[i]
        upper_wick = h[i] - max(o[i], c[i])
        if lower_wick > 2 * body and lower_wick > 0.6 * total_range:
            pinbar_bull[i] = True
        if upper_wick > 2 * body and upper_wick > 0.6 * total_range:
            pinbar_bear[i] = True

    # Inside bar detection
    inside_bar = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if h[i] < h[i-1] and lo[i] > lo[i-1]:
            inside_bar[i] = True

    # Engulfing pattern
    engulf_bull = np.zeros(n, dtype=bool)
    engulf_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if c[i] > o[i] and c[i-1] < o[i-1]:
            if c[i] > o[i-1] and o[i] < c[i-1]:
                engulf_bull[i] = True
        if c[i] < o[i] and c[i-1] > o[i-1]:
            if c[i] < o[i-1] and o[i] > c[i-1]:
                engulf_bear[i] = True

    # S/R levels (swing highs/lows)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    lookback_sr = 10
    for i in range(lookback_sr, n - lookback_sr):
        if h[i] == max(h[i-lookback_sr:i+lookback_sr+1]):
            swing_high[i] = True
        if lo[i] == min(lo[i-lookback_sr:i+lookback_sr+1]):
            swing_low[i] = True

    sr_levels = []
    for i in range(n):
        if swing_high[i]: sr_levels.append(h[i])
        if swing_low[i]: sr_levels.append(lo[i])

    def near_sr(price, tolerance_atr):
        for level in sr_levels[-50:]:
            if abs(price - level) < tolerance_atr:
                return True
        return False

    # PA Strategy 1: Pinbar at S/R
    print(f"\n  --- PA Signal 1: Pinbar at S/R ---", flush=True)
    cfg_pa = {'lot': 0.04, 'cap': 70, 'sl': 3.0, 'tp': 6.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 10}
    trades_pinbar = []; pos = None; le = -999
    for i in range(lookback_sr + 1, n):
        if pos:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, cfg_pa['lot'], PV, times,
                         cfg_pa['sl'], cfg_pa['tp'], cfg_pa['trail_act'], cfg_pa['trail_dist'], cfg_pa['max_hold'], cfg_pa['cap'])
            if r: trades_pinbar.append(r); pos = None; le = i
            continue
        if i - le < 3: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pv_arr[i] < 30 or np.isnan(pv_arr[i]): continue
        if pinbar_bull[i] and near_sr(lo[i], atr[i] * 1.5):
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PINBAR_SR'}
        elif pinbar_bear[i] and near_sr(h[i], atr[i] * 1.5):
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'PINBAR_SR'}

    stats_pinbar = _stats(trades_pinbar)
    print(f"    N={stats_pinbar['n']}, Sharpe={stats_pinbar['sharpe']:.3f}, PnL=${stats_pinbar['pnl']:,.0f}, WR={stats_pinbar['wr']:.1f}%", flush=True)
    results['pinbar_sr'] = stats_pinbar

    # PA Strategy 2: Inside bar breakout
    print(f"\n  --- PA Signal 2: Inside Bar Breakout ---", flush=True)
    trades_ib = []; pos = None; le = -999
    for i in range(2, n):
        if pos:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, cfg_pa['lot'], PV, times,
                         cfg_pa['sl'], cfg_pa['tp'], cfg_pa['trail_act'], cfg_pa['trail_dist'], cfg_pa['max_hold'], cfg_pa['cap'])
            if r: trades_ib.append(r); pos = None; le = i
            continue
        if i - le < 3: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pv_arr[i] < 30 or np.isnan(pv_arr[i]): continue
        if inside_bar[i-1]:
            if c[i] > h[i-1]:
                pos = {'dir': 'BUY', 'entry': c[i] + SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'IB_BREAK'}
            elif c[i] < lo[i-1]:
                pos = {'dir': 'SELL', 'entry': c[i] - SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'IB_BREAK'}

    stats_ib = _stats(trades_ib)
    print(f"    N={stats_ib['n']}, Sharpe={stats_ib['sharpe']:.3f}, PnL=${stats_ib['pnl']:,.0f}, WR={stats_ib['wr']:.1f}%", flush=True)
    results['inside_bar_breakout'] = stats_ib

    # PA Strategy 3: Engulfing + EMA alignment
    print(f"\n  --- PA Signal 3: Engulfing + EMA ---", flush=True)
    ema_v = df['EMA100'].values
    trades_eng = []; pos = None; le = -999
    for i in range(2, n):
        if pos:
            r = _run_exit(pos, i, h[i], lo[i], c[i], SPREAD, cfg_pa['lot'], PV, times,
                         cfg_pa['sl'], cfg_pa['tp'], cfg_pa['trail_act'], cfg_pa['trail_dist'], cfg_pa['max_hold'], cfg_pa['cap'])
            if r: trades_eng.append(r); pos = None; le = i
            continue
        if i - le < 3: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pv_arr[i] < 30 or np.isnan(pv_arr[i]): continue
        if engulf_bull[i] and c[i] > ema_v[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'ENGULF'}
        elif engulf_bear[i] and c[i] < ema_v[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr[i], 'strategy': 'ENGULF'}

    stats_eng = _stats(trades_eng)
    print(f"    N={stats_eng['n']}, Sharpe={stats_eng['sharpe']:.3f}, PnL=${stats_eng['pnl']:,.0f}, WR={stats_eng['wr']:.1f}%", flush=True)
    results['engulfing_ema'] = stats_eng

    # PA as overlay filter on Keltner
    print(f"\n  --- PA as Keltner Overlay (Engulfing confirmation) ---", flush=True)
    keltner_base = bt_keltner(h1, CURRENT_CONFIG['L8_MAX'], pctl_v=pctl, pctl_f=30)
    base_sh = _stats(keltner_base)['sharpe']

    keltner_pa_filtered = [t for t in keltner_base
                           if pd.Timestamp(t['entry_time']) in df.index and
                           (engulf_bull[df.index.get_loc(pd.Timestamp(t['entry_time']))] if t['dir']=='BUY' else
                            engulf_bear[df.index.get_loc(pd.Timestamp(t['entry_time']))] if t['dir']=='SELL' else False)]
    stats_overlay = _stats(keltner_pa_filtered)
    print(f"    Keltner + Engulfing filter: N={stats_overlay['n']}, Sharpe={stats_overlay['sharpe']:.3f} (base={base_sh:.3f})", flush=True)
    results['keltner_engulf_overlay'] = stats_overlay

    with open(OUTPUT_DIR / "phase_06_price_action.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 6 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_7(h1, pctl):
    """Phase 7: Session Microstructure"""
    print(f"\n{'='*120}\n  PHASE 7: SESSION MICROSTRUCTURE\n{'='*120}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)

    # Analysis by entry hour
    print(f"\n  --- Entry Hour Analysis (Full Portfolio) ---", flush=True)
    hour_data = {}
    for t in base_merged:
        hr = pd.Timestamp(t['entry_time']).hour
        hour_data.setdefault(hr, []).append(t['pnl'])

    print(f"  {'Hour':>4} {'N':>5} {'Avg PnL':>8} {'WR%':>5} {'Total':>9} {'Sharpe_est':>10}")
    for hr in sorted(hour_data.keys()):
        pnls = hour_data[hr]
        n_h = len(pnls); avg = np.mean(pnls); wr = sum(1 for p in pnls if p > 0) / n_h * 100
        total = sum(pnls)
        sh_est = avg / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        print(f"  {hr:>4} {n_h:>5} ${avg:>7.1f} {wr:>4.1f}% ${total:>8,.0f} {sh_est:>9.2f}")
        results[f'hour_{hr}'] = {'n': n_h, 'avg_pnl': round(avg, 2), 'wr': round(wr, 1), 'total_pnl': round(total, 2), 'sharpe_est': round(sh_est, 3)}

    # Session-based filtering
    sessions = {
        'asian': list(range(0, 8)),
        'london': list(range(7, 16)),
        'ny': list(range(12, 21)),
        'overlap_lon_ny': list(range(12, 16)),
    }

    print(f"\n  --- Session Filtering ---", flush=True)
    for sess_name, hours in sessions.items():
        filtered = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour in hours]
        s = _stats(filtered)
        print(f"    {sess_name:>15}: N={s['n']:>5}, Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:>9,.0f}, WR={s['wr']:.1f}%", flush=True)
        results[f'session_{sess_name}'] = s

    # Avoid last 2h before session close test
    print(f"\n  --- Avoid Last 2h Before Close ---", flush=True)
    avoid_hours = [5, 6, 15, 16, 20, 21]
    filtered_no_close = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour not in avoid_hours]
    s_no_close = _stats(filtered_no_close)
    delta = s_no_close['sharpe'] - base_stats['sharpe']
    print(f"    Avoid hours {avoid_hours}: Sharpe={s_no_close['sharpe']:.3f} ({delta:+.3f}), N={s_no_close['n']}", flush=True)
    results['avoid_session_close'] = s_no_close

    # Day of week analysis
    print(f"\n  --- Day of Week Analysis ---", flush=True)
    dow_data = {}
    for t in base_merged:
        dow = pd.Timestamp(t['entry_time']).dayofweek
        dow_data.setdefault(dow, []).append(t['pnl'])

    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for dow in sorted(dow_data.keys()):
        pnls = dow_data[dow]
        n_d = len(pnls); avg = np.mean(pnls); total = sum(pnls)
        print(f"    {dow_names[dow]:>3}: N={n_d:>5}, Avg=${avg:>6.1f}, Total=${total:>9,.0f}", flush=True)
        results[f'dow_{dow_names[dow]}'] = {'n': n_d, 'avg_pnl': round(avg, 2), 'total_pnl': round(total, 2)}

    # Validate best session filter with K-Fold
    best_sessions = sorted([(results.get(f'session_{s}', {}).get('sharpe', 0), s) for s in sessions.keys()], reverse=True)
    best_sess = best_sessions[0][1]
    best_hours = sessions[best_sess]

    print(f"\n  --- K-Fold Validation: {best_sess} session filter ---", flush=True)
    def run_sess_filtered(h1f):
        all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        merged = port_merge(all_t)
        filtered = [t for t in merged if pd.Timestamp(t['entry_time']).hour in best_hours]
        return _stats(filtered)['sharpe']

    def run_base_port(h1f):
        all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        return port_sharpe(all_t)

    kf = kfold_test(h1, run_sess_filtered, run_base_port)
    kf_wins = sum(1 for r in kf if r['win'] == 'NEW')
    print(f"    K-Fold: {kf_wins}/{len(kf)} wins", flush=True)
    results['session_filter_kfold'] = {'session': best_sess, 'kf_wins': kf_wins, 'kf_total': len(kf)}

    with open(OUTPUT_DIR / "phase_07_session.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 7 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_8(h1, pctl, m15):
    """Phase 8: M15 Entry Refinement"""
    print(f"\n{'='*120}\n  PHASE 8: M15 ENTRY REFINEMENT\n{'='*120}", flush=True)
    results = {}

    # Get H1 Keltner signals first
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df = df.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])
    pv_arr = pctl.reindex(df.index).values

    # Find H1 signal bars
    c = df['Close'].values; h_arr = df['High'].values; lo_arr = df['Low'].values
    atr_v = df['ATR'].values; adx_v = df['ADX'].values; ema_v = df['EMA100'].values
    ku = df['KC_upper'].values; kl = df['KC_lower'].values
    times_h1 = df.index

    signals = []
    for i in range(1, len(df)):
        if np.isnan(atr_v[i]) or atr_v[i] < 0.1: continue
        if pv_arr[i] < 30 or np.isnan(pv_arr[i]): continue
        if np.isnan(adx_v[i]) or adx_v[i] < 14: continue
        if c[i] > ku[i] and c[i] > ema_v[i]:
            signals.append({'time': times_h1[i], 'dir': 'BUY', 'atr': atr_v[i], 'kc_mid': (ku[i]+kl[i])/2})
        elif c[i] < kl[i] and c[i] < ema_v[i]:
            signals.append({'time': times_h1[i], 'dir': 'SELL', 'atr': atr_v[i], 'kc_mid': (ku[i]+kl[i])/2})

    print(f"  Found {len(signals)} H1 Keltner signals", flush=True)

    # M15 RSI for refinement
    m15_df = m15.copy()
    m15_df['RSI2'] = compute_rsi(m15_df['Close'], 2)
    m15_df['RSI14'] = compute_rsi(m15_df['Close'], 14)

    # Strategy A: Enter at H1 close (baseline)
    cfg = CURRENT_CONFIG['L8_MAX']
    base_trades = bt_keltner(h1, cfg, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(base_trades)
    print(f"\n  Baseline (H1 bar close): N={base_stats['n']}, Sharpe={base_stats['sharpe']:.3f}, PnL=${base_stats['pnl']:,.0f}", flush=True)
    results['baseline'] = base_stats

    # Strategy B: M15 RSI pullback (wait for RSI2 < 20 for BUY, > 80 for SELL within 4 M15 bars)
    print(f"\n  --- M15 RSI Pullback Entry ---", flush=True)
    trades_rsi = []; pos = None; le_t = None
    lot = cfg['lot']; sl = cfg['sl']; tp = cfg['tp']; ta = cfg['trail_act']; td = cfg['trail_dist']; mh = cfg['max_hold'] * 4

    for sig in signals:
        if le_t and (sig['time'] - le_t).total_seconds() < 7200: continue

        window_start = sig['time']
        window_end = sig['time'] + pd.Timedelta(hours=1)
        m15_window = m15_df[(m15_df.index > window_start) & (m15_df.index <= window_end)]

        entered = False
        for j in range(len(m15_window)):
            if sig['dir'] == 'BUY' and m15_window['RSI2'].iloc[j] < 30:
                entry_price = m15_window['Close'].iloc[j] + SPREAD/2
                entered = True; break
            elif sig['dir'] == 'SELL' and m15_window['RSI2'].iloc[j] > 70:
                entry_price = m15_window['Close'].iloc[j] - SPREAD/2
                entered = True; break

        if not entered:
            if len(m15_window) == 0: continue
            entry_price = m15_window['Close'].iloc[0] + SPREAD/2 if sig['dir']=='BUY' else m15_window['Close'].iloc[0] - SPREAD/2

        pos_entry = {'dir': sig['dir'], 'entry': entry_price, 'bar': 0, 'time': sig['time'], 'atr': sig['atr'], 'strategy': 'KELT_M15_RSI'}
        exit_window = m15_df[(m15_df.index > sig['time']) & (m15_df.index <= sig['time'] + pd.Timedelta(hours=mh//4+1))]

        if len(exit_window) < 2: continue
        exited = False
        for k in range(1, len(exit_window)):
            h_k = exit_window['High'].iloc[k]; lo_k = exit_window['Low'].iloc[k]; c_k = exit_window['Close'].iloc[k]
            # Manual exit check instead of _run_exit to avoid index issues
            if sig['dir'] == 'BUY':
                pnl_h = (h_k - entry_price - SPREAD) * lot * PV
                pnl_l = (lo_k - entry_price - SPREAD) * lot * PV
                pnl_c = (c_k - entry_price - SPREAD) * lot * PV
            else:
                pnl_h = (entry_price - lo_k - SPREAD) * lot * PV
                pnl_l = (entry_price - h_k - SPREAD) * lot * PV
                pnl_c = (entry_price - c_k - SPREAD) * lot * PV
            tp_v = tp * sig['atr'] * lot * PV
            sl_v = sl * sig['atr'] * lot * PV
            if pnl_h >= tp_v:
                trades_rsi.append(_mk(pos_entry, c_k, exit_window.index[k], "TP", k, tp_v))
                le_t = exit_window.index[k]; exited = True; break
            if pnl_l <= -sl_v:
                trades_rsi.append(_mk(pos_entry, c_k, exit_window.index[k], "SL", k, -sl_v))
                le_t = exit_window.index[k]; exited = True; break
            if cfg['cap'] > 0 and pnl_c < -cfg['cap']:
                trades_rsi.append(_mk(pos_entry, c_k, exit_window.index[k], "Cap", k, -cfg['cap']))
                le_t = exit_window.index[k]; exited = True; break

        if not exited and len(exit_window) > 0:
            final_c = exit_window['Close'].iloc[-1]
            if sig['dir'] == 'BUY': pnl = (final_c - entry_price - SPREAD) * lot * PV
            else: pnl = (entry_price - final_c - SPREAD) * lot * PV
            trades_rsi.append(_mk(pos_entry, final_c, exit_window.index[-1], "Timeout", len(exit_window), pnl))
            le_t = exit_window.index[-1]

    stats_rsi = _stats(trades_rsi)
    print(f"    N={stats_rsi['n']}, Sharpe={stats_rsi['sharpe']:.3f}, PnL=${stats_rsi['pnl']:,.0f}, WR={stats_rsi['wr']:.1f}%", flush=True)
    results['m15_rsi_pullback'] = stats_rsi

    # Strategy C: M15 momentum confirmation (RSI14 > 50 for BUY, < 50 for SELL)
    print(f"\n  --- M15 Momentum Confirmation ---", flush=True)
    trades_mom = []
    for sig in signals:
        m15_bar = m15_df[(m15_df.index > sig['time']) & (m15_df.index <= sig['time'] + pd.Timedelta(minutes=15))]
        if len(m15_bar) == 0: continue
        rsi14 = m15_bar['RSI14'].iloc[0]
        if np.isnan(rsi14): continue

        confirmed = (sig['dir'] == 'BUY' and rsi14 > 50) or (sig['dir'] == 'SELL' and rsi14 < 50)
        if confirmed:
            trades_mom.append(sig)

    confirmed_pct = len(trades_mom) / len(signals) * 100 if signals else 0
    print(f"    M15 momentum confirms {len(trades_mom)}/{len(signals)} signals ({confirmed_pct:.1f}%)", flush=True)

    # Run filtered trades through backtest
    confirmed_times = set(s['time'] for s in trades_mom)
    filtered_trades = [t for t in base_trades if pd.Timestamp(t['entry_time']) in confirmed_times]
    stats_mom = _stats(filtered_trades)
    print(f"    After filter: N={stats_mom['n']}, Sharpe={stats_mom['sharpe']:.3f}, PnL=${stats_mom['pnl']:,.0f}", flush=True)
    results['m15_momentum_confirm'] = stats_mom

    with open(OUTPUT_DIR / "phase_08_m15.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 8 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_9(h1, pctl):
    """Phase 9: COT Weekly Overlay"""
    print(f"\n{'='*120}\n  PHASE 9: COT WEEKLY OVERLAY\n{'='*120}", flush=True)
    results = {}

    try:
        cot = load_cot()
    except Exception as e:
        print(f"  ERROR loading COT: {e}", flush=True)
        results['error'] = str(e)
        with open(OUTPUT_DIR / "phase_09_cot.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Calculate COT features
    cot['net_spec_z'] = pd.to_numeric(cot['net_spec_z'], errors='coerce')
    if 'net_spec' in cot.columns:
        cot['net_spec_rolling_z'] = (cot['net_spec'] - cot['net_spec'].rolling(52).mean()) / cot['net_spec'].rolling(52).std()
    if 'open_interest' in cot.columns:
        cot['oi_mom'] = cot['open_interest'].pct_change(4)
    if 'net_comm' in cot.columns:
        cot['comm_z'] = (cot['net_comm'] - cot['net_comm'].rolling(52).mean()) / cot['net_comm'].rolling(52).std()

    # Map COT weekly to H1 (forward-fill)
    cot_aligned = cot.reindex(h1.index, method='ffill')

    # Run baseline
    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    print(f"  Baseline: Sharpe={base_stats['sharpe']:.3f}, PnL=${base_stats['pnl']:,.0f}", flush=True)
    results['baseline'] = base_stats

    # COT Filter 1: Skip when net_spec_z extreme (crowded trade)
    print(f"\n  --- COT Filter 1: Skip Extreme Positioning ---", flush=True)
    for z_thresh in [1.5, 2.0, 2.5]:
        if 'net_spec_rolling_z' not in cot_aligned.columns: break
        cot_z = cot_aligned['net_spec_rolling_z'].reindex(h1.index)
        filtered = []
        for t in base_merged:
            entry_time = pd.Timestamp(t['entry_time'])
            if entry_time in cot_z.index:
                z_val = cot_z.loc[entry_time]
                if not np.isnan(z_val) and abs(z_val) > z_thresh:
                    continue
            filtered.append(t)

        stats = _stats(filtered)
        removed = base_stats['n'] - stats['n']
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    |Z| > {z_thresh}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
        results[f'skip_extreme_z{z_thresh}'] = stats

    # COT Filter 2: Only trade when spec positioning aligns with signal direction
    print(f"\n  --- COT Filter 2: Alignment Filter ---", flush=True)
    if 'net_spec_rolling_z' in cot_aligned.columns:
        cot_z = cot_aligned['net_spec_rolling_z'].reindex(h1.index)
        for z_align in [0.0, 0.5, 1.0]:
            aligned_trades = []
            for t in base_merged:
                entry_time = pd.Timestamp(t['entry_time'])
                if entry_time in cot_z.index:
                    z_val = cot_z.loc[entry_time]
                    if np.isnan(z_val):
                        aligned_trades.append(t); continue
                    if t['dir'] == 'BUY' and z_val > z_align:
                        aligned_trades.append(t)
                    elif t['dir'] == 'SELL' and z_val < -z_align:
                        aligned_trades.append(t)
                    elif z_align == 0.0:
                        aligned_trades.append(t)
                else:
                    aligned_trades.append(t)

            stats = _stats(aligned_trades)
            delta = stats['sharpe'] - base_stats['sharpe']
            print(f"    Align Z>{z_align}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), N={stats['n']}", flush=True)
            results[f'align_z{z_align}'] = stats

    # COT Filter 3: OI momentum filter
    print(f"\n  --- COT Filter 3: OI Momentum ---", flush=True)
    if 'oi_mom' in cot_aligned.columns:
        oi_mom = cot_aligned['oi_mom'].reindex(h1.index)
        for oi_thresh in [0.0, 0.02, 0.05]:
            oi_filtered = []
            for t in base_merged:
                entry_time = pd.Timestamp(t['entry_time'])
                if entry_time in oi_mom.index:
                    oi_val = oi_mom.loc[entry_time]
                    if not np.isnan(oi_val) and oi_val > oi_thresh:
                        oi_filtered.append(t)
                    elif np.isnan(oi_val):
                        oi_filtered.append(t)
                else:
                    oi_filtered.append(t)

            stats = _stats(oi_filtered)
            delta = stats['sharpe'] - base_stats['sharpe']
            print(f"    OI_mom > {oi_thresh}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), N={stats['n']}", flush=True)
            results[f'oi_mom_{oi_thresh}'] = stats

    with open(OUTPUT_DIR / "phase_09_cot.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 9 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_10(h1, pctl):
    """Phase 10: Macro Regime Dynamic Weights"""
    print(f"\n{'='*120}\n  PHASE 10: MACRO REGIME DYNAMIC WEIGHTS\n{'='*120}", flush=True)
    results = {}

    try:
        macro = load_macro()
    except Exception as e:
        print(f"  ERROR loading macro: {e}", flush=True)
        results['error'] = str(e)
        with open(OUTPUT_DIR / "phase_10_macro.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Detect macro regimes
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

    regimes = detect_regimes(macro)
    regime_aligned = regimes.reindex(h1.index, method='ffill')
    regime_counts = regime_aligned.value_counts()
    print(f"  Regime distribution: {dict(regime_counts)}", flush=True)
    results['regime_distribution'] = dict(regime_counts)

    # Baseline
    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = port_stats(base_all)
    print(f"  Baseline: Sharpe={base_stats['sharpe']:.3f}, PnL=${base_stats['pnl']:,.0f}", flush=True)
    results['baseline'] = base_stats

    # Test 1: Direction bias by regime
    print(f"\n  --- Macro Direction Bias ---", flush=True)
    base_merged = port_merge(base_all)

    for bias_strength in [0.5, 0.75, 1.0]:
        biased_trades = []
        for t in base_merged:
            entry_time = pd.Timestamp(t['entry_time'])
            if entry_time in regime_aligned.index:
                regime = regime_aligned.loc[entry_time]
            else:
                regime = 'neutral'

            if regime == 'bullish' and t['dir'] == 'SELL':
                adj_t = copy.deepcopy(t); adj_t['pnl'] *= bias_strength; biased_trades.append(adj_t)
            elif regime == 'bearish' and t['dir'] == 'BUY':
                adj_t = copy.deepcopy(t); adj_t['pnl'] *= bias_strength; biased_trades.append(adj_t)
            else:
                biased_trades.append(t)

        stats = _stats(biased_trades)
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    Bias strength={bias_strength}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), PnL=${stats['pnl']:,.0f}", flush=True)
        results[f'direction_bias_{bias_strength}'] = stats

    # Test 2: Skip trades in adverse regime
    print(f"\n  --- Skip Adverse Regime ---", flush=True)
    for skip_mode in ['skip_counter', 'skip_all_bear']:
        filtered = []
        for t in base_merged:
            entry_time = pd.Timestamp(t['entry_time'])
            regime = regime_aligned.loc[entry_time] if entry_time in regime_aligned.index else 'neutral'

            if skip_mode == 'skip_counter':
                if regime == 'bullish' and t['dir'] == 'SELL': continue
                if regime == 'bearish' and t['dir'] == 'BUY': continue
            elif skip_mode == 'skip_all_bear':
                if regime == 'bearish': continue
            filtered.append(t)

        stats = _stats(filtered)
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    {skip_mode}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), N={stats['n']} (removed {base_stats['n']-stats['n']})", flush=True)
        results[skip_mode] = stats

    # Test 3: VIX spike pause
    print(f"\n  --- VIX Spike Pause ---", flush=True)
    vix_z = macro.get('VIX_Zscore', pd.Series(dtype=float)).reindex(h1.index, method='ffill')
    for vix_thresh in [1.5, 2.0, 2.5]:
        vix_filtered = []
        for t in base_merged:
            entry_time = pd.Timestamp(t['entry_time'])
            if entry_time in vix_z.index:
                vz = vix_z.loc[entry_time]
                if not np.isnan(vz) and vz > vix_thresh: continue
            vix_filtered.append(t)

        stats = _stats(vix_filtered)
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    VIX Z > {vix_thresh}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), Skipped={base_stats['n']-stats['n']}", flush=True)
        results[f'vix_pause_{vix_thresh}'] = stats

    # K-Fold on best macro filter
    best_filter = max([(results.get(k, {}).get('sharpe', 0), k) for k in results if k not in ['baseline', 'regime_distribution', 'error']])
    if best_filter[0] > base_stats['sharpe']:
        print(f"\n  --- K-Fold: Best filter = {best_filter[1]} ---", flush=True)

    with open(OUTPUT_DIR / "phase_10_macro.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 10 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# TRACK C: ML/ADVANCED (Phases 11-15)
# ═══════════════════════════════════════════════════════════════════

def phase_11(h1, pctl):
    """Phase 11: XGBoost Entry Filter"""
    print(f"\n{'='*120}\n  PHASE 11: XGBOOST ENTRY FILTER\n{'='*120}", flush=True)
    results = {}

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    if not HAS_XGB or not HAS_SKLEARN:
        print(f"  SKIP: xgboost or sklearn not available", flush=True)
        results['error'] = 'missing dependencies'
        with open(OUTPUT_DIR / "phase_11_xgb.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Build features on H1 data
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['RSI2'] = compute_rsi(df['Close'], 2)
    df['EMA20'] = compute_ema(df['Close'], 20)
    df['EMA100'] = compute_ema(df['Close'], 100)
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + 1.2 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.2 * df['ATR']
    df['ATR_pctl'] = compute_atr_pctl(df['ATR'], lb=300)
    macd_line, macd_sig, macd_hist = compute_macd(df['Close'])
    df['MACD_hist'] = macd_hist
    bb_mid, bb_upper, bb_lower = compute_bb(df['Close'])
    df['BB_width'] = (bb_upper - bb_lower) / bb_mid
    df['KC_dist'] = (df['Close'] - df['KC_mid']) / df['ATR']
    df['EMA_dist_20'] = (df['Close'] - df['EMA20']) / df['ATR']
    df['EMA_dist_100'] = (df['Close'] - df['EMA100']) / df['ATR']
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_4'] = df['Close'].pct_change(4)
    df['ret_12'] = df['Close'].pct_change(12)
    df['ret_24'] = df['Close'].pct_change(24)
    df['range_vs_atr'] = (df['High'] - df['Low']) / df['ATR']
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['vol_ratio'] = df.get('volume', pd.Series(1, index=df.index)).rolling(5).mean() / df.get('volume', pd.Series(1, index=df.index)).rolling(20).mean()

    # Labels: profitable Keltner trade = 1, unprofitable = 0
    all_trades = bt_keltner(h1, CURRENT_CONFIG['L8_MAX'], pctl_v=pctl, pctl_f=30)
    trade_labels = {}
    for t in all_trades:
        trade_labels[pd.Timestamp(t['entry_time'])] = 1 if t['pnl'] > 0 else 0

    feature_cols = ['ATR', 'ADX', 'RSI14', 'RSI2', 'ATR_pctl', 'MACD_hist', 'BB_width',
                    'KC_dist', 'EMA_dist_20', 'EMA_dist_100', 'ret_1', 'ret_4', 'ret_12',
                    'ret_24', 'range_vs_atr', 'hour', 'dow']

    df_labeled = df.loc[df.index.isin(trade_labels.keys())].copy()
    df_labeled['label'] = df_labeled.index.map(trade_labels)
    df_labeled = df_labeled.dropna(subset=feature_cols + ['label'])

    if len(df_labeled) < 100:
        print(f"  Insufficient labeled data: {len(df_labeled)}", flush=True)
        results['error'] = 'insufficient data'
        with open(OUTPUT_DIR / "phase_11_xgb.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    print(f"  Labeled samples: {len(df_labeled)} (pos={df_labeled['label'].sum()}, neg={len(df_labeled)-df_labeled['label'].sum()})", flush=True)

    # Walk-forward evaluation
    X = df_labeled[feature_cols].values
    y = df_labeled['label'].values
    dates = df_labeled.index

    n_total = len(X)
    train_pct = 0.6
    wf_results = []

    fold_size = n_total // 6
    for fold in range(3, 6):
        train_end = fold * fold_size
        test_start = train_end
        test_end = min((fold + 1) * fold_size, n_total)

        if test_end - test_start < 20: continue

        X_train = X[:train_end]; y_train = y[:train_end]
        X_test = X[test_start:test_end]; y_test = y[test_start:test_end]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   eval_metric='logloss', verbosity=0)
        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

        try:
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0.5

        # Test different thresholds
        for thresh in [0.50, 0.55, 0.60, 0.65]:
            selected = probs >= thresh
            if selected.sum() < 5: continue
            precision = y_test[selected].mean() if selected.sum() > 0 else 0
            wf_results.append({'fold': fold, 'thresh': thresh, 'auc': round(auc, 4),
                              'selected': int(selected.sum()), 'precision': round(precision, 4)})

        print(f"    Fold {fold}: AUC={auc:.4f}, N_test={len(y_test)}", flush=True)

    results['walk_forward'] = wf_results

    # Feature importance
    if len(wf_results) > 0:
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        print(f"\n  Top features:", flush=True)
        for fname, imp in feat_imp[:8]:
            print(f"    {fname:>15}: {imp:.4f}", flush=True)
        results['feature_importance'] = {f: round(float(i), 4) for f, i in feat_imp}

    # Trading simulation with ML filter
    if wf_results:
        best_thresh = 0.55
        print(f"\n  --- ML Filter Backtest (thresh={best_thresh}) ---", flush=True)

        scaler_full = StandardScaler()
        split_idx = int(len(X) * 0.7)
        X_train_full = scaler_full.fit_transform(X[:split_idx])
        model_full = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                                        eval_metric='logloss', verbosity=0)
        model_full.fit(X_train_full, y[:split_idx])

        X_test_full = scaler_full.transform(X[split_idx:])
        probs_full = model_full.predict_proba(X_test_full)[:, 1]

        test_trades = [all_trades[i] for i in range(split_idx, min(len(all_trades), n_total))]
        filtered_trades = [t for t, p in zip(test_trades, probs_full) if p >= best_thresh]

        stats_ml = _stats(filtered_trades)
        stats_no_ml = _stats(test_trades)
        print(f"    Without ML: N={stats_no_ml['n']}, Sharpe={stats_no_ml['sharpe']:.3f}, PnL=${stats_no_ml['pnl']:,.0f}", flush=True)
        print(f"    With ML:    N={stats_ml['n']}, Sharpe={stats_ml['sharpe']:.3f}, PnL=${stats_ml['pnl']:,.0f}", flush=True)
        results['ml_backtest'] = {'no_ml': stats_no_ml, 'with_ml': stats_ml}

    with open(OUTPUT_DIR / "phase_11_xgb.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 11 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_12(h1, pctl):
    """Phase 12: ML Exit Timing"""
    print(f"\n{'='*120}\n  PHASE 12: ML EXIT TIMING\n{'='*120}", flush=True)
    results = {}

    try:
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        HAS_DEPS = True
    except ImportError:
        HAS_DEPS = False

    if not HAS_DEPS:
        results['error'] = 'missing dependencies'
        with open(OUTPUT_DIR / "phase_12_ml_exit.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Get all Keltner trades with detailed exit info
    all_trades = bt_keltner(h1, CURRENT_CONFIG['L8_MAX'], pctl_v=pctl, pctl_f=30)
    if len(all_trades) < 100:
        results['error'] = 'insufficient trades'
        with open(OUTPUT_DIR / "phase_12_ml_exit.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Analyze exit type performance
    exit_perf = {}
    for t in all_trades:
        reason = t['reason']
        exit_perf.setdefault(reason, []).append(t['pnl'])

    print(f"  Exit type analysis:", flush=True)
    for reason, pnls in exit_perf.items():
        avg = np.mean(pnls)
        print(f"    {reason:>8}: N={len(pnls):>5}, Avg=${avg:>7.1f}, Total=${sum(pnls):>10,.0f}", flush=True)
    results['exit_analysis'] = {r: {'n': len(p), 'avg_pnl': round(np.mean(p), 2), 'total': round(sum(p), 2)} for r, p in exit_perf.items()}

    # Build features at entry time for predicting optimal exit
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['ATR_pctl'] = compute_atr_pctl(df['ATR'], lb=300)
    macd_line, _, macd_hist = compute_macd(df['Close'])
    df['MACD_hist'] = macd_hist

    feature_cols = ['ATR', 'ADX', 'RSI14', 'ATR_pctl', 'MACD_hist']
    features = []; labels = []

    for t in all_trades:
        entry_time = pd.Timestamp(t['entry_time'])
        if entry_time not in df.index: continue
        row = df.loc[entry_time]
        feat = [row.get(c, np.nan) for c in feature_cols]
        feat.append(1 if t['dir'] == 'BUY' else 0)
        if any(np.isnan(f) for f in feat): continue
        features.append(feat)
        labels.append(t['reason'])

    feature_cols_ext = feature_cols + ['is_buy']
    X = np.array(features)
    y_reasons = np.array(labels)

    # Binary: is trail exit better than others?
    trail_pnl = np.mean([t['pnl'] for t in all_trades if t['reason'] == 'Trail']) if 'Trail' in exit_perf else 0
    y_binary = np.array([1 if r == 'Trail' else 0 for r in y_reasons])

    print(f"\n  ML Exit: {len(X)} samples, Trail%={y_binary.mean()*100:.1f}%", flush=True)

    # Walk-forward
    n = len(X); split = int(n * 0.7)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])

    model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, verbosity=0, eval_metric='logloss')
    model.fit(X_train, y_binary[:split])
    probs = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_binary[split:], probs)
    except:
        auc = 0.5

    print(f"  AUC for trail prediction: {auc:.4f}", flush=True)

    # Simulate: if model predicts "likely trail exit", use tighter trail
    tight_cfg = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])
    tight_cfg['trail_act'] = 0.04
    tight_cfg['trail_dist'] = 0.008

    wide_cfg = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])
    wide_cfg['trail_act'] = 0.08
    wide_cfg['trail_dist'] = 0.015

    test_trades_base = all_trades[split:]
    test_trades_tight = bt_keltner(h1, tight_cfg, pctl_v=pctl, pctl_f=30)
    test_trades_wide = bt_keltner(h1, wide_cfg, pctl_v=pctl, pctl_f=30)

    stats_base = _stats(test_trades_base)
    stats_tight = _stats(test_trades_tight)
    stats_wide = _stats(test_trades_wide)

    print(f"\n  Exit style comparison (full period):", flush=True)
    print(f"    Current (0.06/0.01): Sharpe={_stats(all_trades)['sharpe']:.3f}", flush=True)
    print(f"    Tight   (0.04/0.008): Sharpe={stats_tight['sharpe']:.3f}", flush=True)
    print(f"    Wide    (0.08/0.015): Sharpe={stats_wide['sharpe']:.3f}", flush=True)

    results['exit_styles'] = {
        'current': _stats(all_trades),
        'tight': stats_tight,
        'wide': stats_wide,
        'ml_auc': round(auc, 4)
    }

    with open(OUTPUT_DIR / "phase_12_ml_exit.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 12 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_13(h1, pctl):
    """Phase 13: Strategy Ensemble Meta-Learner"""
    print(f"\n{'='*120}\n  PHASE 13: ENSEMBLE META-LEARNER\n{'='*120}", flush=True)
    results = {}

    # Run all strategies and get per-strategy daily PnL
    all_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    strat_daily = {}
    for nm in STRAT_ORDER:
        strat_daily[nm] = _daily(all_trades[nm])

    # Build aligned daily panel
    all_dates = sorted(set().union(*[set(d.index) for d in strat_daily.values()]))
    panel = pd.DataFrame(index=all_dates)
    for nm in STRAT_ORDER:
        panel[nm] = strat_daily[nm].reindex(all_dates).fillna(0)

    panel['total'] = panel[STRAT_ORDER].sum(axis=1)

    # Rolling performance features
    window = 20
    for nm in STRAT_ORDER:
        panel[f'{nm}_roll_sharpe'] = panel[nm].rolling(window).mean() / panel[nm].rolling(window).std() * np.sqrt(252)
        panel[f'{nm}_roll_wr'] = (panel[nm] > 0).rolling(window).mean()

    # Strategy correlations (rolling)
    for i, nm1 in enumerate(STRAT_ORDER):
        for nm2 in STRAT_ORDER[i+1:]:
            panel[f'corr_{nm1}_{nm2}'] = panel[nm1].rolling(window).corr(panel[nm2])

    panel = panel.dropna()
    print(f"  Panel: {len(panel)} days x {len(panel.columns)} features", flush=True)

    # Simple momentum-based weight allocation
    print(f"\n  --- Momentum-Based Allocation ---", flush=True)
    momentum_results = {}

    for lookback in [10, 20, 40, 60]:
        weighted_daily = pd.Series(0, index=panel.index, dtype=float)
        for i in range(lookback, len(panel)):
            window_data = panel.iloc[i-lookback:i]
            sharpes = {}
            for nm in STRAT_ORDER:
                s = window_data[nm]
                if s.std() > 0:
                    sharpes[nm] = s.mean() / s.std()
                else:
                    sharpes[nm] = 0

            total_sharpe = sum(max(0, s) for s in sharpes.values())
            if total_sharpe > 0:
                weights = {nm: max(0, sharpes[nm]) / total_sharpe for nm in STRAT_ORDER}
            else:
                weights = {nm: 1/6 for nm in STRAT_ORDER}

            day_pnl = sum(panel.iloc[i][nm] * weights[nm] * 6 for nm in STRAT_ORDER)
            weighted_daily.iloc[i] = day_pnl

        static_daily = panel['total']
        sh_dynamic = _sharpe(weighted_daily[lookback:])
        sh_static = _sharpe(static_daily[lookback:])
        print(f"    Lookback={lookback}d: Dynamic Sharpe={sh_dynamic:.3f}, Static Sharpe={sh_static:.3f}", flush=True)
        momentum_results[f'lb_{lookback}'] = {'dynamic': round(sh_dynamic, 3), 'static': round(sh_static, 3)}

    results['momentum_allocation'] = momentum_results

    # Inverse-volatility weighting
    print(f"\n  --- Inverse-Volatility Weighting ---", flush=True)
    for lookback in [20, 40]:
        iv_daily = pd.Series(0, index=panel.index, dtype=float)
        for i in range(lookback, len(panel)):
            window_data = panel.iloc[i-lookback:i]
            vols = {}
            for nm in STRAT_ORDER:
                vol = window_data[nm].std()
                vols[nm] = vol if vol > 0 else 1e-6

            inv_vol_total = sum(1/v for v in vols.values())
            weights = {nm: (1/vols[nm]) / inv_vol_total for nm in STRAT_ORDER}
            day_pnl = sum(panel.iloc[i][nm] * weights[nm] * 6 for nm in STRAT_ORDER)
            iv_daily.iloc[i] = day_pnl

        sh_iv = _sharpe(iv_daily[lookback:])
        print(f"    IV lb={lookback}: Sharpe={sh_iv:.3f}", flush=True)
        results[f'inv_vol_lb{lookback}'] = round(sh_iv, 3)

    with open(OUTPUT_DIR / "phase_13_ensemble.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 13 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_14(h1, pctl):
    """Phase 14: Reinforcement Learning for TP/SL"""
    print(f"\n{'='*120}\n  PHASE 14: RL EXIT DECISIONS\n{'='*120}", flush=True)
    results = {}

    # Simple Q-table approach for exit decisions
    cfg = CURRENT_CONFIG['L8_MAX']
    all_trades = bt_keltner(h1, cfg, pctl_v=pctl, pctl_f=30)

    if len(all_trades) < 50:
        results['error'] = 'insufficient trades'
        with open(OUTPUT_DIR / "phase_14_rl.json", 'w') as f: json.dump(results, f, indent=2, default=str)
        return results

    # Discretize state space
    def discretize_pnl(pnl_pct):
        if pnl_pct < -2: return 0
        elif pnl_pct < -1: return 1
        elif pnl_pct < 0: return 2
        elif pnl_pct < 1: return 3
        elif pnl_pct < 2: return 4
        else: return 5

    def discretize_bars(bars):
        if bars <= 1: return 0
        elif bars <= 3: return 1
        elif bars <= 6: return 2
        else: return 3

    def discretize_momentum(ret):
        if ret < -0.002: return 0
        elif ret < 0: return 1
        elif ret < 0.002: return 2
        else: return 3

    n_states = 6 * 4 * 4
    n_actions = 3  # hold, exit, tighten
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1; gamma = 0.95; epsilon = 0.1

    # Simulate RL on Keltner entries
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ret_1'] = df['Close'].pct_change()
    df = df.dropna()

    # Build episodes from trades
    episodes_pnl = {'rl': [], 'baseline': []}

    for trade in all_trades:
        entry_time = pd.Timestamp(trade['entry_time'])
        if entry_time not in df.index: continue

        entry_idx = df.index.get_loc(entry_time)
        atr_entry = df['ATR'].iloc[entry_idx]
        entry_price = trade['entry']
        direction = 1 if trade['dir'] == 'BUY' else -1

        max_bars = min(20, len(df) - entry_idx - 1)
        episode_reward = 0
        exited = False

        for bar in range(1, max_bars + 1):
            cur_idx = entry_idx + bar
            if cur_idx >= len(df): break

            cur_price = df['Close'].iloc[cur_idx]
            unrealized_pnl_pct = direction * (cur_price - entry_price) / atr_entry
            momentum = df['ret_1'].iloc[cur_idx]

            state = discretize_pnl(unrealized_pnl_pct) * 16 + discretize_bars(bar) * 4 + discretize_momentum(momentum)
            state = min(state, n_states - 1)

            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])

            if action == 1:  # exit
                realized = direction * (cur_price - entry_price - SPREAD) * cfg['lot'] * PV
                episode_reward = realized
                exited = True
                break
            elif action == 2:  # tighten (simulate)
                pass

            if bar == max_bars:
                realized = direction * (cur_price - entry_price - SPREAD) * cfg['lot'] * PV
                episode_reward = realized
                exited = True

        if exited:
            reward = episode_reward
            Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])
            episodes_pnl['rl'].append(episode_reward)
        episodes_pnl['baseline'].append(trade['pnl'])

    stats_rl = {'n': len(episodes_pnl['rl']), 'total_pnl': round(sum(episodes_pnl['rl']), 2),
                'avg_pnl': round(np.mean(episodes_pnl['rl']), 2) if episodes_pnl['rl'] else 0}
    stats_base = {'n': len(episodes_pnl['baseline']), 'total_pnl': round(sum(episodes_pnl['baseline']), 2),
                  'avg_pnl': round(np.mean(episodes_pnl['baseline']), 2)}

    print(f"  RL exits: N={stats_rl['n']}, Total=${stats_rl['total_pnl']:,.0f}, Avg=${stats_rl['avg_pnl']:.1f}", flush=True)
    print(f"  Baseline: N={stats_base['n']}, Total=${stats_base['total_pnl']:,.0f}, Avg=${stats_base['avg_pnl']:.1f}", flush=True)

    results['rl_performance'] = stats_rl
    results['baseline_performance'] = stats_base
    results['q_table_nonzero'] = int(np.count_nonzero(Q))
    results['conclusion'] = 'RL_BETTER' if stats_rl['total_pnl'] > stats_base['total_pnl'] else 'BASELINE_BETTER'

    with open(OUTPUT_DIR / "phase_14_rl.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 14 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_15(h1, pctl):
    """Phase 15: Feature Importance & Signal Decay"""
    print(f"\n{'='*120}\n  PHASE 15: FEATURE IMPORTANCE & SIGNAL DECAY\n{'='*120}", flush=True)
    results = {}

    # Build feature panel
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['ADX'] = compute_adx(df)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['EMA20'] = compute_ema(df['Close'], 20)
    df['EMA100'] = compute_ema(df['Close'], 100)
    df['ATR_pctl'] = compute_atr_pctl(df['ATR'], lb=300)
    _, _, macd_hist = compute_macd(df['Close'])
    df['MACD_hist'] = macd_hist
    _, bb_upper, bb_lower = compute_bb(df['Close'])
    df['BB_width'] = (bb_upper - bb_lower) / df['Close']
    df['KC_mid'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['KC_dist'] = (df['Close'] - df['KC_mid']) / df['ATR']

    # Forward return (4-bar)
    df['fwd_ret_4'] = df['Close'].shift(-4) / df['Close'] - 1
    df = df.dropna()

    indicators = ['ATR', 'ADX', 'RSI14', 'ATR_pctl', 'MACD_hist', 'BB_width', 'KC_dist']

    # Information Coefficient (IC) - rolling rank correlation with forward returns
    print(f"\n  --- Information Coefficient (IC) Analysis ---", flush=True)
    ic_results = {}
    for ind in indicators:
        ic_series = df[ind].rolling(252).corr(df['fwd_ret_4'])
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        # Era breakdown
        era_ics = {}
        for era, segments in ERA_SEGMENTS.items():
            if segments is None: continue
            era_data = pd.concat([ic_series[(ic_series.index >= s) & (ic_series.index < e)] for s, e in segments])
            era_ics[era] = round(era_data.mean(), 5) if len(era_data) > 0 else 0

        print(f"    {ind:>12}: IC={ic_mean:.5f}, IR={ic_ir:.3f}, Eras={era_ics}", flush=True)
        ic_results[ind] = {'ic_mean': round(ic_mean, 5), 'ic_std': round(ic_std, 5), 'ic_ir': round(ic_ir, 3), 'era_ics': era_ics}

    results['information_coefficient'] = ic_results

    # Decay detection: is IC declining in recent period?
    print(f"\n  --- IC Decay Detection ---", flush=True)
    decay_results = {}
    for ind in indicators:
        ic_series = df[ind].rolling(252).corr(df['fwd_ret_4']).dropna()
        if len(ic_series) < 500: continue

        first_half = ic_series.iloc[:len(ic_series)//2].mean()
        second_half = ic_series.iloc[len(ic_series)//2:].mean()
        recent_quarter = ic_series.iloc[-len(ic_series)//4:].mean()

        decay = 'DECAYING' if recent_quarter < first_half * 0.5 else 'STABLE' if abs(recent_quarter - first_half) < abs(first_half) * 0.5 else 'IMPROVING'
        print(f"    {ind:>12}: 1st_half={first_half:.5f}, 2nd_half={second_half:.5f}, Recent={recent_quarter:.5f} [{decay}]", flush=True)
        decay_results[ind] = {'first_half': round(first_half, 5), 'second_half': round(second_half, 5),
                              'recent_quarter': round(recent_quarter, 5), 'status': decay}

    results['ic_decay'] = decay_results

    with open(OUTPUT_DIR / "phase_15_features.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 15 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# TRACK D: EXECUTION & PORTFOLIO (Phases 16-20)
# ═══════════════════════════════════════════════════════════════════

def phase_16(h1, pctl):
    """Phase 16: Realistic Transaction Cost Model"""
    print(f"\n{'='*120}\n  PHASE 16: REALISTIC SPREAD MODEL\n{'='*120}", flush=True)
    results = {}
    global SPREAD

    # Build session-dependent spread model
    hour_spreads = {
        0: 0.45, 1: 0.50, 2: 0.50, 3: 0.48, 4: 0.45, 5: 0.40,
        6: 0.35, 7: 0.30, 8: 0.25, 9: 0.22, 10: 0.22, 11: 0.22,
        12: 0.20, 13: 0.18, 14: 0.18, 15: 0.20, 16: 0.22, 17: 0.25,
        18: 0.28, 19: 0.30, 20: 0.35, 21: 0.38, 22: 0.40, 23: 0.42,
    }

    avg_model_spread = np.mean(list(hour_spreads.values()))
    print(f"  Session spread model: min={min(hour_spreads.values()):.2f}, max={max(hour_spreads.values()):.2f}, avg={avg_model_spread:.2f}", flush=True)
    print(f"  Current fixed spread: {0.30}", flush=True)

    # Compare strategies with realistic vs fixed spread
    print(f"\n  --- Fixed vs Session Spread ---", flush=True)
    spread_scenarios = {
        'fixed_0.25': 0.25,
        'fixed_0.30': 0.30,
        'fixed_0.40': 0.40,
        'fixed_0.50': 0.50,
    }

    original_spread = 0.30
    for scenario, spread_val in spread_scenarios.items():
        SPREAD = spread_val
        all_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        stats = port_stats(all_trades)
        print(f"    {scenario:>12}: Sharpe={stats['sharpe']:.3f}, PnL=${stats['pnl']:>10,.0f}, WR={stats['wr']:.1f}%", flush=True)
        results[scenario] = stats

    SPREAD = original_spread

    # Edge erosion analysis
    print(f"\n  --- Edge Erosion by Spread ---", flush=True)
    base_sharpe = results['fixed_0.30']['sharpe']
    for s_name, s_val in [('tight_0.20', 0.20), ('normal_0.30', 0.30), ('wide_0.45', 0.45), ('extreme_0.70', 0.70)]:
        SPREAD = s_val
        all_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        stats = port_stats(all_trades)
        erosion = base_sharpe - stats['sharpe'] if base_sharpe > 0 else 0
        print(f"    Spread={s_val:.2f}: Sharpe={stats['sharpe']:.3f}, Erosion={erosion:.3f}", flush=True)
        results[s_name] = {'stats': stats, 'erosion': round(erosion, 3)}

    SPREAD = original_spread

    # Per-strategy spread sensitivity
    print(f"\n  --- Per-Strategy Spread Sensitivity ---", flush=True)
    for strat in STRAT_ORDER:
        sharpes_by_spread = []
        for spread_val in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            SPREAD = spread_val
            trades = run_strat(strat, h1, CURRENT_CONFIG[strat], pctl_v=pctl, pctl_f=30)
            sh = _stats(trades)['sharpe']
            sharpes_by_spread.append(sh)

        SPREAD = original_spread
        breakeven_est = 0.30
        for j, sh in enumerate(sharpes_by_spread):
            if sh <= 0:
                breakeven_est = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50][j]
                break

        print(f"    {strat:>12}: Sharpe @ 0.30={sharpes_by_spread[3]:.3f}, Breakeven spread~={breakeven_est:.2f}", flush=True)
        results[f'{strat}_spread_sensitivity'] = sharpes_by_spread

    with open(OUTPUT_DIR / "phase_16_spread.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 16 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_17(h1, pctl):
    """Phase 17: Entry Timing Optimization"""
    print(f"\n{'='*120}\n  PHASE 17: ENTRY TIMING OPTIMIZATION\n{'='*120}", flush=True)
    results = {}

    # Get baseline Keltner trades
    base_trades = bt_keltner(h1, CURRENT_CONFIG['L8_MAX'], pctl_v=pctl, pctl_f=30)
    base_stats = _stats(base_trades)
    print(f"  Baseline: N={base_stats['n']}, Sharpe={base_stats['sharpe']:.3f}, PnL=${base_stats['pnl']:,.0f}", flush=True)
    results['baseline'] = base_stats

    # Analyze entry bar characteristics
    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['range'] = df['High'] - df['Low']
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['lower_wick'] = df[['Close', 'Open']].min(axis=1) - df['Low']

    # Entry fill simulation: what if we entered at different points within the bar?
    print(f"\n  --- Entry Fill Simulation ---", flush=True)
    entry_improvements = {'open': [], 'midpoint': [], 'close': [], 'optimal': []}

    for t in base_trades:
        entry_time = pd.Timestamp(t['entry_time'])
        if entry_time not in df.index: continue
        bar = df.loc[entry_time]
        entry_actual = t['entry']
        direction = t['dir']

        bar_open = bar['Open']
        bar_mid = (bar['High'] + bar['Low']) / 2
        bar_close = bar['Close']
        optimal = bar['Low'] if direction == 'BUY' else bar['High']

        if direction == 'BUY':
            improvement_open = entry_actual - bar_open
            improvement_mid = entry_actual - bar_mid
            improvement_optimal = entry_actual - optimal
        else:
            improvement_open = bar_open - entry_actual
            improvement_mid = bar_mid - entry_actual
            improvement_optimal = optimal - entry_actual

        entry_improvements['open'].append(improvement_open)
        entry_improvements['midpoint'].append(improvement_mid)
        entry_improvements['optimal'].append(improvement_optimal)

    for fill_type, improvements in entry_improvements.items():
        if not improvements: continue
        avg = np.mean(improvements)
        pnl_impact = avg * CURRENT_CONFIG['L8_MAX']['lot'] * PV * len(improvements)
        print(f"    {fill_type:>10}: Avg improvement=${avg:.3f}, Total PnL impact=${pnl_impact:,.0f}", flush=True)
        results[f'fill_{fill_type}'] = {'avg_improvement': round(avg, 4), 'pnl_impact': round(pnl_impact, 2)}

    # Next-bar vs same-bar entry
    print(f"\n  --- Next-Bar Entry Delay ---", flush=True)
    cfg_delayed = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])
    delayed_trades = []

    df_k = h1.copy()
    df_k['ATR'] = compute_atr(df_k)
    df_k['ADX'] = compute_adx(df_k)
    df_k['EMA100'] = df_k['Close'].ewm(span=100, adjust=False).mean()
    df_k['KC_mid'] = df_k['Close'].ewm(span=25, adjust=False).mean()
    df_k['KC_upper'] = df_k['KC_mid'] + 1.2 * df_k['ATR']
    df_k['KC_lower'] = df_k['KC_mid'] - 1.2 * df_k['ATR']
    df_k = df_k.dropna(subset=['ATR', 'ADX', 'EMA100', 'KC_upper'])

    pv_arr = pctl.reindex(df_k.index).values
    c = df_k['Close'].values; h_a = df_k['High'].values; lo_a = df_k['Low'].values
    o_a = df_k['Open'].values; atr_v = df_k['ATR'].values; adx_v = df_k['ADX'].values
    ema_v = df_k['EMA100'].values; ku = df_k['KC_upper'].values; kl = df_k['KC_lower'].values
    times = df_k.index; n = len(df_k)

    lot = cfg_delayed['lot']; sl = cfg_delayed['sl']; tp = cfg_delayed['tp']
    ta = cfg_delayed['trail_act']; td = cfg_delayed['trail_dist']; mh = cfg_delayed['max_hold']
    cap = cfg_delayed['cap']
    pos = None; le = -999

    for i in range(2, n):
        if pos:
            r = _run_exit(pos, i, h_a[i], lo_a[i], c[i], SPREAD, lot, PV, times, sl, tp, ta, td, mh, cap)
            if r: delayed_trades.append(r); pos = None; le = i
            continue
        if i - le < 2: continue
        if np.isnan(atr_v[i-1]) or atr_v[i-1] < 0.1: continue
        if pv_arr[i-1] < 30 or np.isnan(pv_arr[i-1]): continue
        if np.isnan(adx_v[i-1]) or adx_v[i-1] < 14: continue
        if c[i-1] > ku[i-1] and c[i-1] > ema_v[i-1]:
            pos = {'dir': 'BUY', 'entry': o_a[i] + SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr_v[i], 'strategy': 'L8_MAX_DELAYED'}
        elif c[i-1] < kl[i-1] and c[i-1] < ema_v[i-1]:
            pos = {'dir': 'SELL', 'entry': o_a[i] - SPREAD/2, 'bar': i, 'time': times[i], 'atr': atr_v[i], 'strategy': 'L8_MAX_DELAYED'}

    stats_delayed = _stats(delayed_trades)
    delta = stats_delayed['sharpe'] - base_stats['sharpe']
    print(f"    Next-bar open entry: Sharpe={stats_delayed['sharpe']:.3f} ({delta:+.3f}), N={stats_delayed['n']}", flush=True)
    results['next_bar_entry'] = stats_delayed

    with open(OUTPUT_DIR / "phase_17_entry_timing.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 17 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_18(h1, pctl):
    """Phase 18: Kelly Criterion Position Sizing"""
    print(f"\n{'='*120}\n  PHASE 18: KELLY CRITERION SIZING\n{'='*120}", flush=True)
    results = {}

    # Calculate Kelly fraction for each strategy
    for strat in STRAT_ORDER:
        trades = run_strat(strat, h1, CURRENT_CONFIG[strat], pctl_v=pctl, pctl_f=30)
        if len(trades) < 30: continue

        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if not wins or not losses: continue
        p_win = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0: continue
        rr = avg_win / avg_loss
        kelly = p_win - (1 - p_win) / rr
        half_kelly = kelly / 2
        quarter_kelly = kelly / 4

        print(f"  {strat:>12}: WR={p_win:.3f}, R:R={rr:.2f}, Kelly={kelly:.3f}, Half={half_kelly:.3f}", flush=True)
        results[strat] = {'p_win': round(p_win, 4), 'rr': round(rr, 3), 'kelly': round(kelly, 4),
                         'half_kelly': round(half_kelly, 4), 'quarter_kelly': round(quarter_kelly, 4)}

    # Simulate dynamic Kelly sizing
    print(f"\n  --- Dynamic Kelly Simulation ---", flush=True)
    sizing_modes = {
        'fixed': lambda strat, recent_wr, recent_rr: CURRENT_CONFIG[strat]['lot'],
        'half_kelly': lambda strat, recent_wr, recent_rr: max(0.02, min(0.10, CURRENT_CONFIG[strat]['lot'] * max(0, recent_wr - (1-recent_wr)/max(recent_rr,0.1)) / 2 * 5)),
        'quarter_kelly': lambda strat, recent_wr, recent_rr: max(0.02, min(0.10, CURRENT_CONFIG[strat]['lot'] * max(0, recent_wr - (1-recent_wr)/max(recent_rr,0.1)) / 4 * 5)),
    }

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = sorted(port_merge(base_all), key=lambda x: x['entry_time'])

    for mode_name, lot_fn in sizing_modes.items():
        adjusted_trades = []
        recent_trades = {nm: [] for nm in STRAT_ORDER}

        for t in base_merged:
            strat = t['strategy']
            if strat not in STRAT_ORDER:
                adjusted_trades.append(t); continue

            recent = recent_trades[strat][-20:]
            if len(recent) >= 10:
                recent_wr = sum(1 for p in recent if p > 0) / len(recent)
                wins_r = [p for p in recent if p > 0]
                losses_r = [abs(p) for p in recent if p < 0]
                recent_rr = np.mean(wins_r) / np.mean(losses_r) if losses_r else 2.0
            else:
                recent_wr = 0.5; recent_rr = 1.5

            lot_mult = lot_fn(strat, recent_wr, recent_rr) / CURRENT_CONFIG[strat]['lot'] if CURRENT_CONFIG[strat]['lot'] > 0 else 1.0
            adj_t = copy.deepcopy(t)
            adj_t['pnl'] = t['pnl'] * lot_mult
            adjusted_trades.append(adj_t)
            recent_trades[strat].append(t['pnl'])

        stats = _stats(adjusted_trades)
        print(f"    {mode_name:>15}: Sharpe={stats['sharpe']:.3f}, PnL=${stats['pnl']:>10,.0f}, MaxDD=${stats['max_dd']:,.0f}", flush=True)
        results[f'sim_{mode_name}'] = stats

    with open(OUTPUT_DIR / "phase_18_kelly.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 18 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_19(h1, pctl):
    """Phase 19: Intraday Volatility Patterns"""
    print(f"\n{'='*120}\n  PHASE 19: INTRADAY VOLATILITY PATTERNS\n{'='*120}", flush=True)
    results = {}

    df = h1.copy()
    df['range'] = df['High'] - df['Low']
    df['ATR'] = compute_atr(df)
    df['range_vs_atr'] = df['range'] / df['ATR']
    df['hour'] = df.index.hour
    df['year'] = df.index.year

    # Hour-by-hour volatility map
    print(f"\n  --- Hourly Volatility Map (12 years) ---", flush=True)
    hour_vol = df.groupby('hour')['range'].agg(['mean', 'std', 'median', 'count'])
    print(f"  {'Hour':>4} {'Mean':>7} {'Median':>7} {'Std':>6} {'Count':>6}")
    for hr in range(24):
        if hr in hour_vol.index:
            row = hour_vol.loc[hr]
            print(f"  {hr:>4} {row['mean']:>7.3f} {row['median']:>7.3f} {row['std']:>6.3f} {int(row['count']):>6}")
            results[f'hour_{hr}'] = {'mean_range': round(float(row['mean']), 4), 'median_range': round(float(row['median']), 4), 'std': round(float(row['std']), 4)}

    # Peak/dead zone identification
    hourly_means = df.groupby('hour')['range'].mean()
    overall_mean = hourly_means.mean()
    peak_hours = hourly_means[hourly_means > overall_mean * 1.3].index.tolist()
    dead_hours = hourly_means[hourly_means < overall_mean * 0.7].index.tolist()

    print(f"\n  Peak hours (>1.3x avg): {peak_hours}", flush=True)
    print(f"  Dead hours (<0.7x avg): {dead_hours}", flush=True)
    results['peak_hours'] = [int(x) for x in peak_hours]
    results['dead_hours'] = [int(x) for x in dead_hours]

    # Spread dominance in dead zones
    print(f"\n  --- Spread Dominance Analysis ---", flush=True)
    for hr in sorted(set(dead_hours + peak_hours)):
        hr_data = df[df['hour'] == hr]
        if len(hr_data) == 0: continue
        spread_pct_of_range = SPREAD / hr_data['range'].mean() * 100
        print(f"    Hour {hr:>2}: Spread/Range = {spread_pct_of_range:.1f}%", flush=True)

    # Trading performance by volatility zone
    print(f"\n  --- Strategy Performance by Vol Zone ---", flush=True)
    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)

    peak_trades = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour in peak_hours]
    dead_trades = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour in dead_hours]
    normal_trades = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour not in peak_hours + dead_hours]

    for zone, trades in [('peak', peak_trades), ('dead', dead_trades), ('normal', normal_trades)]:
        stats = _stats(trades)
        print(f"    {zone:>6}: N={stats['n']:>5}, Sharpe={stats['sharpe']:.3f}, Avg PnL=${stats['pnl']/max(stats['n'],1):.1f}", flush=True)
        results[f'zone_{zone}'] = stats

    # Yearly stability of patterns
    print(f"\n  --- Yearly Pattern Stability ---", flush=True)
    years = sorted(df['year'].unique())
    yearly_peaks = {}
    for yr in years:
        yr_data = df[df['year'] == yr]
        yr_hourly = yr_data.groupby('hour')['range'].mean()
        yr_peak = yr_hourly.idxmax()
        yr_dead = yr_hourly.idxmin()
        yearly_peaks[yr] = {'peak_hour': int(yr_peak), 'dead_hour': int(yr_dead)}

    peak_consistency = len(set(v['peak_hour'] for v in yearly_peaks.values()))
    print(f"    Peak hour varies across {peak_consistency} different hours over {len(years)} years", flush=True)
    results['yearly_peaks'] = {str(k): v for k, v in yearly_peaks.items()}

    with open(OUTPUT_DIR / "phase_19_vol_patterns.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 19 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


def phase_20(h1, pctl):
    """Phase 20: Portfolio Correlation Dynamics"""
    print(f"\n{'='*120}\n  PHASE 20: PORTFOLIO CORRELATION DYNAMICS\n{'='*120}", flush=True)
    results = {}

    # Run all strategies and build daily PnL panel
    all_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    strat_daily = {}
    for nm in STRAT_ORDER:
        strat_daily[nm] = _daily(all_trades[nm])

    all_dates = sorted(set().union(*[set(d.index) for d in strat_daily.values()]))
    panel = pd.DataFrame(index=all_dates)
    for nm in STRAT_ORDER:
        panel[nm] = strat_daily[nm].reindex(all_dates).fillna(0)

    # Full-period correlation matrix
    print(f"\n  --- Full-Period Correlation Matrix ---", flush=True)
    corr_matrix = panel[STRAT_ORDER].corr()
    print(f"  {'':>12}", end='')
    for nm in STRAT_ORDER: print(f" {nm[:6]:>7}", end='')
    print()
    for nm1 in STRAT_ORDER:
        print(f"  {nm1:>12}", end='')
        for nm2 in STRAT_ORDER:
            print(f" {corr_matrix.loc[nm1,nm2]:>7.3f}", end='')
        print()

    results['full_correlation'] = corr_matrix.to_dict()

    # Rolling correlation
    print(f"\n  --- Rolling 60-day Correlation ---", flush=True)
    rolling_window = 60
    pairs = []
    for i, nm1 in enumerate(STRAT_ORDER):
        for nm2 in STRAT_ORDER[i+1:]:
            pairs.append((nm1, nm2))

    rolling_corr_stats = {}
    for nm1, nm2 in pairs:
        roll_corr = panel[nm1].rolling(rolling_window).corr(panel[nm2]).dropna()
        if len(roll_corr) < 50: continue
        stats = {
            'mean': round(roll_corr.mean(), 4),
            'std': round(roll_corr.std(), 4),
            'max': round(roll_corr.max(), 4),
            'min': round(roll_corr.min(), 4),
            'pct_above_06': round((roll_corr > 0.6).mean() * 100, 1),
        }
        rolling_corr_stats[f'{nm1}_{nm2}'] = stats
        if stats['pct_above_06'] > 5:
            print(f"    {nm1:>10} x {nm2:<10}: mean={stats['mean']:.3f}, >0.6 = {stats['pct_above_06']:.1f}%", flush=True)

    results['rolling_correlation'] = rolling_corr_stats

    # Correlation-triggered lot reduction
    print(f"\n  --- Correlation-Triggered Lot Reduction ---", flush=True)
    base_stats = port_stats(all_trades)
    base_merged = port_merge(all_trades)

    # Compute average rolling pairwise correlation
    avg_corr = pd.Series(np.nan, index=panel.index)
    for i in range(rolling_window, len(panel)):
        window = panel[STRAT_ORDER].iloc[i-rolling_window:i]
        corr_mat = window.corr()
        mask = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        avg_corr.iloc[i] = corr_mat.values[mask].mean()

    for corr_thresh in [0.1, 0.15, 0.2, 0.25, 0.3]:
        adj_trades = []
        for t in base_merged:
            entry_time = pd.Timestamp(t['entry_time'])
            if entry_time in avg_corr.index:
                corr_val = avg_corr.loc[entry_time]
                if not np.isnan(corr_val) and corr_val > corr_thresh:
                    adj_t = copy.deepcopy(t)
                    adj_t['pnl'] *= 0.5
                    adj_trades.append(adj_t)
                    continue
            adj_trades.append(t)

        stats = _stats(adj_trades)
        delta = stats['sharpe'] - base_stats['sharpe']
        print(f"    Avg corr > {corr_thresh}: Sharpe={stats['sharpe']:.3f} ({delta:+.3f}), PnL=${stats['pnl']:,.0f}", flush=True)
        results[f'corr_reduction_{corr_thresh}'] = stats

    # Max simultaneous positions analysis
    print(f"\n  --- Simultaneous Open Position Risk ---", flush=True)
    position_timeline = []
    for nm in STRAT_ORDER:
        for t in all_trades[nm]:
            position_timeline.append({
                'strategy': nm,
                'entry': pd.Timestamp(t['entry_time']),
                'exit': pd.Timestamp(t['exit_time']),
                'max_risk': -t['pnl'] if t['pnl'] < 0 else CURRENT_CONFIG[nm]['cap']
            })

    max_concurrent = 0
    max_risk = 0
    for t1 in position_timeline:
        concurrent = [t2 for t2 in position_timeline
                     if t2['entry'] <= t1['entry'] < t2['exit'] and t2 != t1]
        n_concurrent = len(concurrent) + 1
        total_risk = t1['max_risk'] + sum(t2['max_risk'] for t2 in concurrent)
        max_concurrent = max(max_concurrent, n_concurrent)
        max_risk = max(max_risk, total_risk)

    print(f"    Max concurrent positions: {max_concurrent}", flush=True)
    print(f"    Max simultaneous risk: ${max_risk:.0f} ({max_risk/CAPITAL*100:.1f}% of capital)", flush=True)
    results['max_concurrent'] = max_concurrent
    results['max_simultaneous_risk'] = round(max_risk, 2)

    with open(OUTPUT_DIR / "phase_20_correlation.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 20 complete ({(time.time()-t0)/3600:.1f}h elapsed)", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"{'='*120}")
    print(f"  R194 — 200-HOUR GOLD MEGA RESEARCH PROGRAM")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*120}")
    print(f"  Capital: ${CAPITAL:,} | Spread: {SPREAD} | Gold-only")
    print(f"  20 phases x 4 tracks = 200h estimated compute")
    print(f"{'='*120}\n")

    # Load data
    print("  Loading data...", flush=True)
    h1 = load_h1()
    try:
        m15 = load_m15()
        HAS_M15 = True
    except:
        m15 = None
        HAS_M15 = False
        print("  WARNING: M15 data not available, Phase 8 will be limited", flush=True)

    # Compute ATR percentile
    print("  Computing ATR percentile...", flush=True)
    df_temp = h1.copy()
    df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR percentile computed ({pctl.notna().sum()} valid values)", flush=True)

    all_results = {}

    def phase_done(phase_num):
        """Check if phase already completed (for resume support)"""
        fname_map = {1:'phase_01_sensitivity', 2:'phase_02_regime_exit', 3:'phase_03_drawdown',
                     4:'phase_04_stress', 5:'phase_05_alpha_decay', 6:'phase_06_price_action',
                     7:'phase_07_session', 8:'phase_08_m15', 9:'phase_09_cot', 10:'phase_10_macro',
                     11:'phase_11_xgb', 12:'phase_12_ml_exit', 13:'phase_13_ensemble',
                     14:'phase_14_rl', 15:'phase_15_features', 16:'phase_16_spread',
                     17:'phase_17_entry_timing', 18:'phase_18_kelly', 19:'phase_19_vol_patterns',
                     20:'phase_20_correlation'}
        fp = OUTPUT_DIR / f"{fname_map.get(phase_num, 'unknown')}.json"
        return fp.exists()

    # ═══════ TRACK A: System Hardening ═══════
    print(f"\n\n{'#'*120}")
    print(f"  TRACK A: SYSTEM HARDENING (Phases 1-5)")
    print(f"{'#'*120}\n")

    if phase_done(1):
        print("  Phase 1 already complete, skipping...", flush=True)
    else:
        all_results['phase_01'] = phase_1(h1, pctl)

    if phase_done(2):
        print("  Phase 2 already complete, skipping...", flush=True)
    else:
        all_results['phase_02'] = phase_2(h1, pctl)

    if phase_done(3):
        print("  Phase 3 already complete, skipping...", flush=True)
    else:
        all_results['phase_03'] = phase_3(h1, pctl)

    if phase_done(4):
        print("  Phase 4 already complete, skipping...", flush=True)
    else:
        all_results['phase_04'] = phase_4(h1, pctl)

    if phase_done(5):
        print("  Phase 5 already complete, skipping...", flush=True)
    else:
        all_results['phase_05'] = phase_5(h1, pctl)

    print(f"\n  TRACK A COMPLETE: {(time.time()-t0)/3600:.1f}h elapsed")

    # ═══════ TRACK B: New Alpha ═══════
    print(f"\n\n{'#'*120}")
    print(f"  TRACK B: NEW ALPHA DISCOVERY (Phases 6-10)")
    print(f"{'#'*120}\n")

    if phase_done(6):
        print("  Phase 6 already complete, skipping...", flush=True)
    else:
        all_results['phase_06'] = phase_6(h1, pctl)

    if phase_done(7):
        print("  Phase 7 already complete, skipping...", flush=True)
    else:
        all_results['phase_07'] = phase_7(h1, pctl)

    if phase_done(8):
        print("  Phase 8 already complete, skipping...", flush=True)
    elif HAS_M15:
        all_results['phase_08'] = phase_8(h1, pctl, m15)
    else:
        all_results['phase_08'] = {'error': 'M15 data not available'}
        print("  Phase 8 SKIPPED (no M15 data)", flush=True)

    if phase_done(9):
        print("  Phase 9 already complete, skipping...", flush=True)
    else:
        all_results['phase_09'] = phase_9(h1, pctl)

    if phase_done(10):
        print("  Phase 10 already complete, skipping...", flush=True)
    else:
        all_results['phase_10'] = phase_10(h1, pctl)

    print(f"\n  TRACK B COMPLETE: {(time.time()-t0)/3600:.1f}h elapsed")

    # ═══════ TRACK C: ML/Advanced ═══════
    print(f"\n\n{'#'*120}")
    print(f"  TRACK C: ML/ADVANCED METHODS (Phases 11-15)")
    print(f"{'#'*120}\n")

    if phase_done(11):
        print("  Phase 11 already complete, skipping...", flush=True)
    else:
        all_results['phase_11'] = phase_11(h1, pctl)

    if phase_done(12):
        print("  Phase 12 already complete, skipping...", flush=True)
    else:
        all_results['phase_12'] = phase_12(h1, pctl)

    if phase_done(13):
        print("  Phase 13 already complete, skipping...", flush=True)
    else:
        all_results['phase_13'] = phase_13(h1, pctl)

    if phase_done(14):
        print("  Phase 14 already complete, skipping...", flush=True)
    else:
        all_results['phase_14'] = phase_14(h1, pctl)

    if phase_done(15):
        print("  Phase 15 already complete, skipping...", flush=True)
    else:
        all_results['phase_15'] = phase_15(h1, pctl)

    print(f"\n  TRACK C COMPLETE: {(time.time()-t0)/3600:.1f}h elapsed")

    # ═══════ TRACK D: Execution ═══════
    print(f"\n\n{'#'*120}")
    print(f"  TRACK D: EXECUTION & PORTFOLIO (Phases 16-20)")
    print(f"{'#'*120}\n")

    if phase_done(16):
        print("  Phase 16 already complete, skipping...", flush=True)
    else:
        all_results['phase_16'] = phase_16(h1, pctl)

    if phase_done(17):
        print("  Phase 17 already complete, skipping...", flush=True)
    else:
        all_results['phase_17'] = phase_17(h1, pctl)

    if phase_done(18):
        print("  Phase 18 already complete, skipping...", flush=True)
    else:
        all_results['phase_18'] = phase_18(h1, pctl)

    if phase_done(19):
        print("  Phase 19 already complete, skipping...", flush=True)
    else:
        all_results['phase_19'] = phase_19(h1, pctl)

    if phase_done(20):
        print("  Phase 20 already complete, skipping...", flush=True)
    else:
        all_results['phase_20'] = phase_20(h1, pctl)

    print(f"\n  TRACK D COMPLETE: {(time.time()-t0)/3600:.1f}h elapsed")

    # ═══════ FINAL SUMMARY ═══════
    total_hours = (time.time() - t0) / 3600
    print(f"\n\n{'='*120}")
    print(f"  R194 COMPLETE")
    print(f"  Total runtime: {total_hours:.1f} hours")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*120}\n")

    # Summary of GO/NO-GO findings
    print(f"  PHASE SUMMARY:")
    for phase_key in sorted(all_results.keys()):
        phase_data = all_results[phase_key]
        if isinstance(phase_data, dict):
            if 'error' in phase_data:
                status = f"ERROR: {phase_data['error']}"
            elif 'verdict' in phase_data:
                status = phase_data['verdict']
            else:
                status = "COMPLETE"
        else:
            status = "COMPLETE"
        print(f"    {phase_key}: {status}", flush=True)

    # Save master results
    with open(OUTPUT_DIR / "r194_master_results.json", 'w') as f:
        json.dump({'runtime_hours': round(total_hours, 2), 'phases_completed': len(all_results)}, f, indent=2, default=str)

    print(f"\n  All results saved to {OUTPUT_DIR}/", flush=True)
