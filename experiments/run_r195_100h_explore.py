#!/usr/bin/env python3
"""
R195 — 100-Hour Gold Technical Exploration
===========================================
10 Phases covering exit optimization, entry timing, regime-adaptive params,
new alpha, portfolio dynamics, advanced exits, parameter stability, drawdown
patterns, multi-timeframe confluence, and execution edge.

All validation: K-Fold >= 4/6, WF >= 13/19, Era all positive
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r195_explore")
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

def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()

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

def _run_exit_custom(pos,i,h,lo,c,spread,lot,pv,times,sl_atr,tp_atr,ta,td,mh,cap,**kwargs):
    """Extended exit with optional time-decay TP, ratchet trail, etc."""
    held = i - pos['bar']
    # Time-decay TP: reduce TP target over time
    tp_decay = kwargs.get('tp_decay', 0)
    if tp_decay > 0 and mh > 0:
        decay_factor = max(0.3, 1.0 - tp_decay * held / mh)
        tp_atr_eff = tp_atr * decay_factor
    else:
        tp_atr_eff = tp_atr

    # Ratchet trail: tighten trail as profit grows
    ratchet = kwargs.get('ratchet', 0)

    if pos['dir']=='BUY':
        pnl_c=(c-pos['entry']-spread)*lot*pv; pnl_h=(h-pos['entry']-spread)*lot*pv; pnl_l=(lo-pos['entry']-spread)*lot*pv
    else:
        pnl_c=(pos['entry']-c-spread)*lot*pv; pnl_h=(pos['entry']-lo-spread)*lot*pv; pnl_l=(pos['entry']-h-spread)*lot*pv

    tp_v=tp_atr_eff*pos['atr']*lot*pv; sl_v=sl_atr*pos['atr']*lot*pv
    if pnl_h>=tp_v: return _mk(pos,c,times[i],"TP",i,tp_v)
    if pnl_l<=-sl_v: return _mk(pos,c,times[i],"SL",i,-sl_v)
    if cap>0 and pnl_c<-cap: return _mk(pos,c,times[i],"Cap",i,-cap)

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
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2)}

def filter_era(trades, era):
    if era=='full' or ERA_SEGMENTS[era] is None: return trades
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

def full_validate(h1, run_new_fn, run_base_fn, trades_new, trades_base, label):
    kf = kfold_test(h1, run_new_fn, run_base_fn)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    print(f"      KF: {kf_wins}/{len(kf)}", flush=True)

    wf = wf_test(h1, run_new_fn, run_base_fn)
    wf_wins = sum(1 for r in wf if r['win']=='NEW')
    print(f"      WF: {wf_wins}/{len(wf)}", flush=True)

    era_results = {}
    for era in ['hike','cut','recent_3y']:
        en = filter_era(trades_new, era); eb = filter_era(trades_base, era)
        sn = _sharpe(_daily(en)); sb = _sharpe(_daily(eb))
        era_results[era] = {'new': round(sn,3), 'base': round(sb,3), 'delta': round(sn-sb,3)}

    kf_pass = kf_wins >= 4; wf_pass = wf_wins >= 13
    era_pass = all(era_results[e]['new'] > 0 for e in era_results)
    era_no_degrade = all(era_results[e]['delta'] > -0.3 for e in era_results)
    verdict = 'GO' if kf_pass and wf_pass and era_pass and era_no_degrade else 'NO-GO'
    reason = []
    if not kf_pass: reason.append(f"KF {kf_wins}/{len(kf)}")
    if not wf_pass: reason.append(f"WF {wf_wins}/{len(wf)}")
    if not era_pass or not era_no_degrade: reason.append("Era")
    print(f"      Verdict: {verdict} ({', '.join(reason) if reason else 'ALL PASS'})", flush=True)
    return {'kf_wins':kf_wins,'kf_total':len(kf),'wf_wins':wf_wins,'wf_total':len(wf),'era':era_results,'verdict':verdict}

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

# ═══════════════════════════════════════════════════════════════
# Strategy backtests (parametric)
# ═══════════════════════════════════════════════════════════════
def bt_keltner(h1, cfg, pctl_v=None, pctl_f=0, exit_kwargs=None):
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
    ekw = exit_kwargs or {}
    for i in range(1,n):
        if pos:
            if ekw:
                r=_run_exit_custom(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**ekw)
            else:
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

def bt_psar(h1, cfg, pctl_v=None, pctl_f=0, exit_kwargs=None):
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
    ekw = exit_kwargs or {}
    for i in range(1,n2):
        cur=c[i]>ps[i]
        if pos:
            if ekw: r=_run_exit_custom(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**ekw)
            else: r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; prev=cur; continue
            prev=cur; continue
        if i-le<2: prev=cur; continue
        if np.isnan(atr[i]) or atr[i]<0.1: prev=cur; continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): prev=cur; continue
        if cur and not prev: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        elif not cur and prev: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        prev=cur
    return trades

def bt_tsmom(h1, cfg, pctl_v=None, pctl_f=0, exit_kwargs=None):
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
    ekw = exit_kwargs or {}
    for i in range(mx+1,n):
        if pos:
            if ekw: r=_run_exit_custom(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**ekw)
            else: r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
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

def bt_sess_bo(h1, cfg, pctl_v=None, pctl_f=0, exit_kwargs=None, entry_hour=12):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    hrs=df.index.hour; times=df.index; n=len(df); lb=4
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    ekw = exit_kwargs or {}
    for i in range(lb,n):
        if pos:
            if ekw: r=_run_exit_custom(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**ekw)
            else: r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
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

def bt_dt(h1, cfg, pctl_v=None, pctl_f=0, exit_kwargs=None):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,o,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df); nb=6; k=0.5
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    ekw = exit_kwargs or {}
    for i in range(nb,n):
        if pos:
            if ekw: r=_run_exit_custom(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**ekw)
            else: r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
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

def bt_chand(h1, cfg, pctl_v=None, pctl_f=0, exit_kwargs=None):
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
    ekw = exit_kwargs or {}
    for i in range(p+2,n):
        if pos:
            if ekw: r=_run_exit_custom(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap,**ekw)
            else: r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
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

def run_all(h1, config, pctl_v=None, pctl_f=0, exit_kwargs=None):
    return {nm: BT_FN[nm](h1, config[nm], pctl_v=pctl_v, pctl_f=pctl_f, exit_kwargs=exit_kwargs) for nm in STRAT_ORDER}

def port_merge(all_t):
    return [t for nm in STRAT_ORDER for t in all_t.get(nm,[])]

def port_sharpe(all_t):
    return _sharpe(_daily(port_merge(all_t)))


# ═══════════════════════════════════════════════════════════════
# PHASE 1: Exit Logic Optimization
# ═══════════════════════════════════════════════════════════════
def phase_1(h1, pctl):
    if phase_done("phase_1_exit"): print("  Phase 1 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 1: EXIT LOGIC OPTIMIZATION\n{'='*100}", flush=True)
    results = {}

    # Baseline
    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(port_merge(base_all))
    results['baseline'] = base_stats
    print(f"  Baseline: Sharpe={base_stats['sharpe']}, PnL=${base_stats['pnl']:,.0f}", flush=True)

    # 1A: Time-decay TP (reduce TP target as trade ages)
    print(f"\n  --- 1A: Time-Decay TP ---", flush=True)
    for td_factor in [0.0, 0.2, 0.4, 0.6, 0.8]:
        all_t = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30, exit_kwargs={'tp_decay': td_factor})
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    tp_decay={td_factor}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), PnL=${st['pnl']:,.0f}", flush=True)
        results[f'tp_decay_{td_factor}'] = st

    # 1B: Ratchet trail (tighten trail as profit grows)
    print(f"\n  --- 1B: Ratchet Trail ---", flush=True)
    for ratchet in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        all_t = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30, exit_kwargs={'ratchet': ratchet})
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    ratchet={ratchet}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), PnL=${st['pnl']:,.0f}", flush=True)
        results[f'ratchet_{ratchet}'] = st

    # 1C: Max hold sweep (per strategy optimal)
    print(f"\n  --- 1C: Max Hold Sweep (per strategy) ---", flush=True)
    for strat in STRAT_ORDER:
        mh_results = []
        for mh_test in [1, 2, 3, 5, 8, 12, 15, 20, 30, 48]:
            cfg_test = copy.deepcopy(CURRENT_CONFIG)
            cfg_test[strat]['max_hold'] = mh_test
            trades = BT_FN[strat](h1, cfg_test[strat], pctl_v=pctl, pctl_f=30)
            st = _stats(trades)
            mh_results.append({'mh': mh_test, 'sharpe': st['sharpe'], 'pnl': st['pnl'], 'n': st['n']})
        best = max(mh_results, key=lambda x: x['sharpe'])
        cur_mh = CURRENT_CONFIG[strat]['max_hold']
        print(f"    {strat}: best_mh={best['mh']} (Sharpe={best['sharpe']:.3f}) vs current_mh={cur_mh}", flush=True)
        results[f'mh_{strat}'] = {'sweep': mh_results, 'best': best, 'current': cur_mh}

    # 1D: Trail activation sweep
    print(f"\n  --- 1D: Trail Activation/Distance Sweep ---", flush=True)
    trail_combos = [(0.03,0.005),(0.04,0.008),(0.05,0.01),(0.06,0.01),(0.08,0.015),(0.10,0.02),(0.14,0.025)]
    for ta_v, td_v in trail_combos:
        cfg_test = copy.deepcopy(CURRENT_CONFIG)
        for strat in STRAT_ORDER:
            if strat != 'TSMOM':
                cfg_test[strat]['trail_act'] = ta_v
                cfg_test[strat]['trail_dist'] = td_v
        all_t = run_all(h1, cfg_test, pctl_v=pctl, pctl_f=30)
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    ta={ta_v}/td={td_v}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'trail_{ta_v}_{td_v}'] = st

    # Validate best exit improvement
    best_exit = None; best_delta = 0
    for key, val in results.items():
        if key.startswith('tp_decay') or key.startswith('ratchet') or key.startswith('trail_'):
            if isinstance(val, dict) and 'sharpe' in val:
                d = val['sharpe'] - base_stats['sharpe']
                if d > best_delta: best_delta = d; best_exit = key

    if best_exit and best_delta > 0.1:
        print(f"\n  Best exit: {best_exit} (delta={best_delta:+.3f}) — running validation...", flush=True)
        if 'tp_decay' in best_exit:
            decay_val = float(best_exit.split('_')[-1])
            def run_new(h1f): return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30, exit_kwargs={'tp_decay': decay_val}))
        elif 'ratchet' in best_exit:
            ratchet_val = float(best_exit.split('_')[-1])
            def run_new(h1f): return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30, exit_kwargs={'ratchet': ratchet_val}))
        else:
            def run_new(h1f): return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))

        def run_base(h1f): return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))
        trades_new = port_merge(run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30, exit_kwargs={'tp_decay': 0.4} if 'tp_decay' in best_exit else {}))
        validation = full_validate(h1, run_new, run_base, trades_new, port_merge(base_all), best_exit)
        results['best_validation'] = validation
    else:
        print(f"\n  No exit improvement > 0.1 Sharpe found", flush=True)
        results['best_validation'] = {'verdict': 'NO CANDIDATE'}

    with open(OUTPUT_DIR / "phase_1_exit.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 1 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 2: Entry Timing
# ═══════════════════════════════════════════════════════════════
def phase_2(h1, pctl):
    if phase_done("phase_2_timing"): print("  Phase 2 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 2: ENTRY TIMING ANALYSIS\n{'='*100}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    results['baseline'] = base_stats

    # 2A: Hour-of-day performance
    print(f"\n  --- 2A: Performance by Hour-of-Day ---", flush=True)
    hour_perf = {}
    for hr in range(24):
        hr_trades = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour == hr]
        if hr_trades:
            st = _stats(hr_trades)
            hour_perf[str(hr)] = st
            print(f"    Hour {hr:02d}: N={st['n']:>5}, Sharpe={st['sharpe']:>7.3f}, WR={st['wr']:.1f}%, PnL=${st['pnl']:>10,.0f}", flush=True)
    results['hour_perf'] = hour_perf

    # 2B: Day-of-week performance
    print(f"\n  --- 2B: Performance by Day-of-Week ---", flush=True)
    dow_names = ['Mon','Tue','Wed','Thu','Fri']
    dow_perf = {}
    for dow in range(5):
        dow_trades = [t for t in base_merged if pd.Timestamp(t['entry_time']).dayofweek == dow]
        if dow_trades:
            st = _stats(dow_trades)
            dow_perf[dow_names[dow]] = st
            print(f"    {dow_names[dow]}: N={st['n']:>5}, Sharpe={st['sharpe']:>7.3f}, WR={st['wr']:.1f}%, PnL=${st['pnl']:>10,.0f}", flush=True)
    results['dow_perf'] = dow_perf

    # 2C: Skip worst hours
    print(f"\n  --- 2C: Skip Worst Hours ---", flush=True)
    sorted_hours = sorted(hour_perf.items(), key=lambda x: x[1]['sharpe'])
    for n_skip in [1, 2, 3, 4, 5]:
        skip_hrs = set(int(h) for h, _ in sorted_hours[:n_skip])
        filtered = [t for t in base_merged if pd.Timestamp(t['entry_time']).hour not in skip_hrs]
        st = _stats(filtered)
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    Skip {n_skip} worst hours {skip_hrs}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), N={st['n']}", flush=True)
        results[f'skip_{n_skip}_hours'] = {'stats': st, 'hours_skipped': list(skip_hrs)}

    # 2D: Skip worst day
    print(f"\n  --- 2D: Skip Worst Day ---", flush=True)
    sorted_dow = sorted(dow_perf.items(), key=lambda x: x[1]['sharpe'])
    worst_day = sorted_dow[0][0]
    worst_dow_idx = dow_names.index(worst_day)
    filtered = [t for t in base_merged if pd.Timestamp(t['entry_time']).dayofweek != worst_dow_idx]
    st = _stats(filtered)
    delta = st['sharpe'] - base_stats['sharpe']
    print(f"    Skip {worst_day}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), N={st['n']}", flush=True)
    results['skip_worst_day'] = {'stats': st, 'day': worst_day}

    # 2E: SESS_BO entry hour optimization
    print(f"\n  --- 2E: SESS_BO Entry Hour Sweep ---", flush=True)
    for entry_hr in [8, 10, 11, 12, 13, 14, 15, 16]:
        trades = bt_sess_bo(h1, CURRENT_CONFIG['SESS_BO'], pctl_v=pctl, pctl_f=30, entry_hour=entry_hr)
        st = _stats(trades)
        print(f"    SESS_BO @hour={entry_hr}: Sharpe={st['sharpe']:.3f}, N={st['n']}, PnL=${st['pnl']:,.0f}", flush=True)
        results[f'sessbo_hour_{entry_hr}'] = st

    # Validate best timing filter
    best_skip = None; best_delta_t = 0
    for key, val in results.items():
        if key.startswith('skip_') and 'stats' in val:
            d = val['stats']['sharpe'] - base_stats['sharpe']
            if d > best_delta_t: best_delta_t = d; best_skip = key

    if best_skip and best_delta_t > 0.1:
        skip_info = results[best_skip]
        if 'hours_skipped' in skip_info:
            skip_set = set(skip_info['hours_skipped'])
            def run_new(h1f):
                all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                m = port_merge(all_t)
                f = [t for t in m if pd.Timestamp(t['entry_time']).hour not in skip_set]
                return _stats(f)['sharpe']
        else:
            skip_dow = dow_names.index(skip_info['day'])
            def run_new(h1f):
                all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                m = port_merge(all_t)
                f = [t for t in m if pd.Timestamp(t['entry_time']).dayofweek != skip_dow]
                return _stats(f)['sharpe']
        def run_base(h1f): return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))
        print(f"\n  Best timing: {best_skip} (delta={best_delta_t:+.3f}) — validating...", flush=True)
        validation = full_validate(h1, run_new, run_base, port_merge(base_all), port_merge(base_all), best_skip)
        results['best_validation'] = validation
    else:
        results['best_validation'] = {'verdict': 'NO CANDIDATE'}

    with open(OUTPUT_DIR / "phase_2_timing.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 2 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 3: Regime-Adaptive Parameters
# ═══════════════════════════════════════════════════════════════
def phase_3(h1, pctl):
    if phase_done("phase_3_regime"): print("  Phase 3 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 3: REGIME-ADAPTIVE PARAMETERS\n{'='*100}", flush=True)
    results = {}

    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    atr_pctl = compute_atr_pctl(df_temp['ATR'], lb=300)

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(port_merge(base_all))
    results['baseline'] = base_stats

    # 3A: ATR regime → dynamic TP multiplier
    print(f"\n  --- 3A: ATR Regime Dynamic TP ---", flush=True)
    base_merged = port_merge(base_all)
    atr_at_entry = {}
    for t in base_merged:
        et = pd.Timestamp(t['entry_time'])
        if et in atr_pctl.index:
            pct_val = atr_pctl.loc[et]
            if not np.isnan(pct_val):
                if pct_val >= 70: atr_at_entry[id(t)] = 'high'
                elif pct_val >= 40: atr_at_entry[id(t)] = 'medium'
                else: atr_at_entry[id(t)] = 'low'

    for regime in ['high', 'medium', 'low']:
        regime_trades = [t for t in base_merged if atr_at_entry.get(id(t)) == regime]
        st = _stats(regime_trades)
        print(f"    ATR {regime}: N={st['n']}, Sharpe={st['sharpe']:.3f}, WR={st['wr']:.1f}%", flush=True)
        results[f'regime_{regime}'] = st

    # 3B: Dynamic TP based on ATR regime
    print(f"\n  --- 3B: ATR-Scaled TP (high ATR → larger TP) ---", flush=True)
    for tp_mult_high, tp_mult_low in [(1.5, 0.7), (1.3, 0.8), (1.2, 0.9), (1.0, 1.0)]:
        cfg_high = copy.deepcopy(CURRENT_CONFIG)
        cfg_low = copy.deepcopy(CURRENT_CONFIG)
        for s in STRAT_ORDER:
            cfg_high[s]['tp'] = CURRENT_CONFIG[s]['tp'] * tp_mult_high
            cfg_low[s]['tp'] = CURRENT_CONFIG[s]['tp'] * tp_mult_low

        # Split data by ATR regime and run separately
        h1_high = h1[atr_pctl >= 70].copy()
        h1_low = h1[(atr_pctl >= 30) & (atr_pctl < 70)].copy()

        if len(h1_high) > 500 and len(h1_low) > 500:
            t_high = port_merge(run_all(h1_high, cfg_high, pctl_v=pctl, pctl_f=30))
            t_low = port_merge(run_all(h1_low, cfg_low, pctl_v=pctl, pctl_f=30))
            combined = t_high + t_low
            st = _stats(combined)
            delta = st['sharpe'] - base_stats['sharpe']
            print(f"    TP_high_mult={tp_mult_high}, TP_low_mult={tp_mult_low}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
            results[f'dynamic_tp_{tp_mult_high}_{tp_mult_low}'] = st

    # 3C: Dynamic max_hold based on ATR
    print(f"\n  --- 3C: ATR-Scaled Max Hold ---", flush=True)
    for mh_mult in [0.5, 0.75, 1.0, 1.5, 2.0]:
        cfg_test = copy.deepcopy(CURRENT_CONFIG)
        for s in STRAT_ORDER:
            cfg_test[s]['max_hold'] = max(1, int(CURRENT_CONFIG[s]['max_hold'] * mh_mult))
        all_t = run_all(h1, cfg_test, pctl_v=pctl, pctl_f=30)
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    mh_mult={mh_mult}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'mh_mult_{mh_mult}'] = st

    # 3D: Dynamic SL based on ATR
    print(f"\n  --- 3D: SL Multiplier Sweep ---", flush=True)
    for sl_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        cfg_test = copy.deepcopy(CURRENT_CONFIG)
        for s in STRAT_ORDER:
            cfg_test[s]['sl'] = CURRENT_CONFIG[s]['sl'] * sl_mult
        all_t = run_all(h1, cfg_test, pctl_v=pctl, pctl_f=30)
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    sl_mult={sl_mult}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'sl_mult_{sl_mult}'] = st

    with open(OUTPUT_DIR / "phase_3_regime.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 3 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 4: New Alpha Sources
# ═══════════════════════════════════════════════════════════════
def phase_4(h1, pctl):
    if phase_done("phase_4_alpha"): print("  Phase 4 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 4: NEW ALPHA SOURCES\n{'='*100}", flush=True)
    results = {}

    df = h1.copy()
    df['ATR'] = compute_atr(df)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['RSI7'] = compute_rsi(df['Close'], 7)
    df['ADX'] = compute_adx(df)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2*df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
    df['MOM20'] = df['Close'] / df['Close'].shift(20) - 1
    df['MOM50'] = df['Close'] / df['Close'].shift(50) - 1

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    results['baseline'] = base_stats

    # 4A: RSI Divergence filter
    print(f"\n  --- 4A: RSI Divergence Filter ---", flush=True)
    df['price_mom5'] = df['Close'].diff(5)
    df['rsi_mom5'] = df['RSI14'].diff(5)
    # Bearish divergence: price up but RSI down
    df['bear_div'] = (df['price_mom5'] > 0) & (df['rsi_mom5'] < -5)
    # Bullish divergence: price down but RSI up
    df['bull_div'] = (df['price_mom5'] < 0) & (df['rsi_mom5'] > 5)
    bear_div_aligned = df['bear_div'].reindex(h1.index, method='ffill')
    bull_div_aligned = df['bull_div'].reindex(h1.index, method='ffill')

    # Skip BUY if bearish divergence, skip SELL if bullish divergence
    filtered = [t for t in base_merged
                if not (t['dir']=='BUY' and pd.Timestamp(t['entry_time']) in bear_div_aligned.index and bear_div_aligned.loc[pd.Timestamp(t['entry_time'])])
                and not (t['dir']=='SELL' and pd.Timestamp(t['entry_time']) in bull_div_aligned.index and bull_div_aligned.loc[pd.Timestamp(t['entry_time'])])]
    st = _stats(filtered)
    delta = st['sharpe'] - base_stats['sharpe']
    removed = base_stats['n'] - st['n']
    print(f"    RSI Divergence: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
    results['rsi_divergence'] = {'stats': st, 'removed': removed}

    # 4B: Bollinger Band Width filter (skip when BB is compressed — expecting breakout but uncertain direction)
    print(f"\n  --- 4B: BB Squeeze Filter ---", flush=True)
    bb_width = df['BB_width'].reindex(h1.index, method='ffill')
    bb_pctl = bb_width.rolling(300).rank(pct=True) * 100
    for bb_min in [10, 20, 30]:
        filtered = [t for t in base_merged
                    if not (pd.Timestamp(t['entry_time']) in bb_pctl.index and
                           not np.isnan(bb_pctl.loc[pd.Timestamp(t['entry_time'])]) and
                           bb_pctl.loc[pd.Timestamp(t['entry_time'])] < bb_min)]
        st = _stats(filtered)
        delta = st['sharpe'] - base_stats['sharpe']
        removed = base_stats['n'] - st['n']
        print(f"    BB_pctl>={bb_min}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
        results[f'bb_squeeze_{bb_min}'] = {'stats': st, 'removed': removed}

    # 4C: Multi-EMA alignment
    print(f"\n  --- 4C: Multi-EMA Alignment Filter ---", flush=True)
    ema20 = df['EMA20'].reindex(h1.index, method='ffill')
    ema50 = df['EMA50'].reindex(h1.index, method='ffill')
    ema200 = df['EMA200'].reindex(h1.index, method='ffill')
    # Strong uptrend: EMA20 > EMA50 > EMA200
    # Strong downtrend: EMA20 < EMA50 < EMA200
    filtered = [t for t in base_merged
                if (t['dir']=='BUY' and pd.Timestamp(t['entry_time']) in ema20.index and
                    ema20.loc[pd.Timestamp(t['entry_time'])] > ema50.loc[pd.Timestamp(t['entry_time'])])
                or (t['dir']=='SELL' and pd.Timestamp(t['entry_time']) in ema20.index and
                    ema20.loc[pd.Timestamp(t['entry_time'])] < ema50.loc[pd.Timestamp(t['entry_time'])])]
    st = _stats(filtered)
    delta = st['sharpe'] - base_stats['sharpe']
    removed = base_stats['n'] - st['n']
    print(f"    EMA20>50 alignment: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
    results['ema_alignment'] = {'stats': st, 'removed': removed}

    # 4D: Momentum confirmation
    print(f"\n  --- 4D: Momentum Confirmation ---", flush=True)
    mom20 = df['MOM20'].reindex(h1.index, method='ffill')
    filtered = [t for t in base_merged
                if (t['dir']=='BUY' and pd.Timestamp(t['entry_time']) in mom20.index and
                    not np.isnan(mom20.loc[pd.Timestamp(t['entry_time'])]) and
                    mom20.loc[pd.Timestamp(t['entry_time'])] > 0)
                or (t['dir']=='SELL' and pd.Timestamp(t['entry_time']) in mom20.index and
                    not np.isnan(mom20.loc[pd.Timestamp(t['entry_time'])]) and
                    mom20.loc[pd.Timestamp(t['entry_time'])] < 0)]
    st = _stats(filtered)
    delta = st['sharpe'] - base_stats['sharpe']
    removed = base_stats['n'] - st['n']
    print(f"    MOM20 confirmation: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
    results['mom_confirm'] = {'stats': st, 'removed': removed}

    # 4E: High ADX filter (trend strength)
    print(f"\n  --- 4E: ADX Threshold Sweep ---", flush=True)
    adx_vals = df['ADX'].reindex(h1.index, method='ffill')
    for adx_min in [14, 18, 22, 25, 30]:
        filtered = [t for t in base_merged
                    if pd.Timestamp(t['entry_time']) in adx_vals.index and
                       not np.isnan(adx_vals.loc[pd.Timestamp(t['entry_time'])]) and
                       adx_vals.loc[pd.Timestamp(t['entry_time'])] >= adx_min]
        st = _stats(filtered)
        delta = st['sharpe'] - base_stats['sharpe']
        removed = base_stats['n'] - st['n']
        print(f"    ADX>={adx_min}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), N={st['n']}", flush=True)
        results[f'adx_{adx_min}'] = {'stats': st, 'removed': removed}

    with open(OUTPUT_DIR / "phase_4_alpha.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 4 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 5: Portfolio Dynamics
# ═══════════════════════════════════════════════════════════════
def phase_5(h1, pctl):
    if phase_done("phase_5_portfolio"): print("  Phase 5 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 5: PORTFOLIO DYNAMICS\n{'='*100}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(port_merge(base_all))
    results['baseline'] = base_stats

    # 5A: Signal clustering — skip trades if too many signals in recent window
    print(f"\n  --- 5A: Anti-Clustering (max trades/window) ---", flush=True)
    base_merged = port_merge(base_all)
    base_merged_sorted = sorted(base_merged, key=lambda t: t['entry_time'])

    for max_per_window, window_h in [(3, 24), (5, 24), (3, 12), (5, 12), (2, 6)]:
        filtered = []
        recent_times = []
        for t in base_merged_sorted:
            et = pd.Timestamp(t['entry_time'])
            cutoff = et - pd.Timedelta(hours=window_h)
            recent_times = [rt for rt in recent_times if rt > cutoff]
            if len(recent_times) < max_per_window:
                filtered.append(t)
                recent_times.append(et)
        st = _stats(filtered)
        delta = st['sharpe'] - base_stats['sharpe']
        removed = base_stats['n'] - st['n']
        print(f"    max={max_per_window}/window={window_h}h: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
        results[f'cluster_{max_per_window}_{window_h}h'] = {'stats': st, 'removed': removed}

    # 5B: Strategy rotation — weight by recent performance
    print(f"\n  --- 5B: Strategy Contribution Analysis ---", flush=True)
    for strat in STRAT_ORDER:
        st = _stats(base_all.get(strat, []))
        print(f"    {strat}: Sharpe={st['sharpe']:.3f}, PnL=${st['pnl']:>10,.0f}, N={st['n']}", flush=True)
        results[f'contrib_{strat}'] = st

    # 5C: Leave-one-out analysis
    print(f"\n  --- 5C: Leave-One-Out ---", flush=True)
    for exclude in STRAT_ORDER:
        subset = {nm: trades for nm, trades in base_all.items() if nm != exclude}
        st = _stats(port_merge(subset))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    Without {exclude}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'loo_{exclude}'] = st

    # 5D: Pairwise correlation
    print(f"\n  --- 5D: Pairwise Strategy Correlation ---", flush=True)
    daily_series = {}
    for strat in STRAT_ORDER:
        d = _daily(base_all.get(strat, []))
        if len(d) > 0:
            daily_series[strat] = d
    if len(daily_series) >= 2:
        panel = pd.DataFrame(daily_series).fillna(0)
        corr = panel.corr()
        corr_results = {}
        for i, s1 in enumerate(STRAT_ORDER):
            for j, s2 in enumerate(STRAT_ORDER):
                if j > i and s1 in corr.columns and s2 in corr.columns:
                    c_val = corr.loc[s1, s2]
                    corr_results[f'{s1}_vs_{s2}'] = round(float(c_val), 3)
                    print(f"    {s1} vs {s2}: {c_val:.3f}", flush=True)
        results['correlations'] = corr_results

    with open(OUTPUT_DIR / "phase_5_portfolio.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 5 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 6: Advanced Exit Mechanisms
# ═══════════════════════════════════════════════════════════════
def phase_6(h1, pctl):
    if phase_done("phase_6_adv_exit"): print("  Phase 6 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 6: ADVANCED EXIT MECHANISMS\n{'='*100}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(port_merge(base_all))
    results['baseline'] = base_stats

    # 6A: Combined TP decay + Ratchet trail
    print(f"\n  --- 6A: Combined TP Decay + Ratchet ---", flush=True)
    for td_f, ratch in [(0.2, 0.1), (0.4, 0.1), (0.4, 0.15), (0.6, 0.1), (0.6, 0.2)]:
        all_t = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30, exit_kwargs={'tp_decay': td_f, 'ratchet': ratch})
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    decay={td_f}+ratchet={ratch}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'combo_{td_f}_{ratch}'] = st

    # 6B: Asymmetric TP/SL (different ratio per strategy)
    print(f"\n  --- 6B: TP/SL Ratio Sweep ---", flush=True)
    for tp_sl_ratio in [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        cfg_test = copy.deepcopy(CURRENT_CONFIG)
        for s in STRAT_ORDER:
            cfg_test[s]['tp'] = cfg_test[s]['sl'] * tp_sl_ratio
        all_t = run_all(h1, cfg_test, pctl_v=pctl, pctl_f=30)
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    TP/SL={tp_sl_ratio}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), WR={st['wr']:.1f}%", flush=True)
        results[f'tp_sl_ratio_{tp_sl_ratio}'] = st

    # 6C: Break-even stop (move SL to entry after X ATR profit)
    print(f"\n  --- 6C: Break-Even Stop Profit ---", flush=True)
    # This tests a conceptual "move to breakeven" strategy
    for be_atr in [1.0, 1.5, 2.0, 3.0]:
        cfg_test = copy.deepcopy(CURRENT_CONFIG)
        for s in STRAT_ORDER:
            cfg_test[s]['trail_act'] = be_atr / cfg_test[s]['sl']
            cfg_test[s]['trail_dist'] = be_atr / cfg_test[s]['sl']
        all_t = run_all(h1, cfg_test, pctl_v=pctl, pctl_f=30)
        st = _stats(port_merge(all_t))
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    BE@{be_atr}ATR: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'be_{be_atr}'] = st

    with open(OUTPUT_DIR / "phase_6_adv_exit.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 6 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 7: Parameter Stability
# ═══════════════════════════════════════════════════════════════
def phase_7(h1, pctl):
    if phase_done("phase_7_stability"): print("  Phase 7 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 7: PARAMETER STABILITY\n{'='*100}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(port_merge(base_all))
    results['baseline'] = base_stats

    # 7A: Perturbation analysis — randomly perturb all params by +/-10%, 20%, 30%
    print(f"\n  --- 7A: Random Perturbation (100 trials) ---", flush=True)
    np.random.seed(42)
    perturb_sharpes = {'10pct': [], '20pct': [], '30pct': []}

    for pct_label, pct in [('10pct', 0.10), ('20pct', 0.20), ('30pct', 0.30)]:
        for trial in range(100):
            cfg_test = copy.deepcopy(CURRENT_CONFIG)
            for s in STRAT_ORDER:
                for param in ['sl', 'tp', 'trail_act', 'trail_dist']:
                    orig = cfg_test[s][param]
                    cfg_test[s][param] = orig * (1 + np.random.uniform(-pct, pct))
                cfg_test[s]['max_hold'] = max(1, int(cfg_test[s]['max_hold'] * (1 + np.random.uniform(-pct, pct))))
            all_t = run_all(h1, cfg_test, pctl_v=pctl, pctl_f=30)
            sh = _sharpe(_daily(port_merge(all_t)))
            perturb_sharpes[pct_label].append(sh)

        arr = np.array(perturb_sharpes[pct_label])
        print(f"    {pct_label}: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}, pct_positive={((arr>0).sum()/len(arr)*100):.0f}%", flush=True)
        results[f'perturb_{pct_label}'] = {'mean': round(float(arr.mean()),3), 'std': round(float(arr.std()),3),
                                           'min': round(float(arr.min()),3), 'max': round(float(arr.max()),3),
                                           'pct_positive': round(float((arr>0).sum()/len(arr)*100),1)}

    # 7B: Single parameter sensitivity (one-at-a-time)
    print(f"\n  --- 7B: Single Parameter Sensitivity ---", flush=True)
    for strat in ['L8_MAX', 'PSAR', 'CHANDELIER']:
        for param in ['sl', 'tp', 'trail_act', 'trail_dist']:
            sensitivities = []
            base_val = CURRENT_CONFIG[strat][param]
            for mult in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
                cfg_test = copy.deepcopy(CURRENT_CONFIG)
                cfg_test[strat][param] = base_val * mult
                trades = BT_FN[strat](h1, cfg_test[strat], pctl_v=pctl, pctl_f=30)
                st = _stats(trades)
                sensitivities.append({'mult': mult, 'sharpe': st['sharpe'], 'pnl': st['pnl']})
            best = max(sensitivities, key=lambda x: x['sharpe'])
            print(f"    {strat}.{param}: best_mult={best['mult']} (Sharpe={best['sharpe']:.3f})", flush=True)
            results[f'sens_{strat}_{param}'] = sensitivities

    with open(OUTPUT_DIR / "phase_7_stability.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 7 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 8: Drawdown Analysis
# ═══════════════════════════════════════════════════════════════
def phase_8(h1, pctl):
    if phase_done("phase_8_drawdown"): print("  Phase 8 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 8: DRAWDOWN ANALYSIS\n{'='*100}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_merged_sorted = sorted(base_merged, key=lambda t: t['entry_time'])

    # 8A: Consecutive loss analysis
    print(f"\n  --- 8A: Consecutive Losses ---", flush=True)
    streaks = []; current_streak = 0; max_streak = 0
    for t in base_merged_sorted:
        if t['pnl'] < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            if current_streak > 0: streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0: streaks.append(current_streak)
    print(f"    Max consecutive losses: {max_streak}", flush=True)
    print(f"    Average loss streak: {np.mean(streaks):.2f}" if streaks else "    No streaks", flush=True)
    print(f"    Streaks > 5: {sum(1 for s in streaks if s > 5)}", flush=True)
    results['max_consec_loss'] = max_streak
    results['avg_streak'] = round(float(np.mean(streaks)), 2) if streaks else 0

    # 8B: Recovery time after drawdowns
    print(f"\n  --- 8B: Drawdown Recovery ---", flush=True)
    daily = _daily(base_merged)
    eq = daily.cumsum()
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    dd_periods = []
    in_dd = False; dd_start = None
    for i, (idx, val) in enumerate(dd.items()):
        if val > 0 and not in_dd:
            in_dd = True; dd_start = idx
        elif val == 0 and in_dd:
            in_dd = False
            if dd_start is not None:
                duration = (idx - dd_start).days
                max_dd_in_period = dd[dd_start:idx].max()
                dd_periods.append({'start': str(dd_start.date()), 'end': str(idx.date()),
                                  'duration_days': duration, 'max_dd': round(float(max_dd_in_period), 2)})
    dd_periods_sorted = sorted(dd_periods, key=lambda x: -x['max_dd'])[:10]
    for dp in dd_periods_sorted[:5]:
        print(f"    DD ${dp['max_dd']:.0f}, {dp['duration_days']}d ({dp['start']} ~ {dp['end']})", flush=True)
    results['top_drawdowns'] = dd_periods_sorted

    # 8C: Post-drawdown performance
    print(f"\n  --- 8C: Post-Drawdown Performance ---", flush=True)
    dd_threshold = 200
    post_dd_trades = []; normal_trades = []
    eq_values = eq.values; dd_values = dd.values
    for t in base_merged_sorted:
        et = pd.Timestamp(t['exit_time']).normalize()
        if et in dd.index:
            idx_pos = dd.index.get_loc(et)
            if dd_values[idx_pos] > dd_threshold:
                post_dd_trades.append(t)
            else:
                normal_trades.append(t)
        else:
            normal_trades.append(t)
    st_dd = _stats(post_dd_trades); st_normal = _stats(normal_trades)
    print(f"    During DD>${dd_threshold}: Sharpe={st_dd['sharpe']:.3f}, N={st_dd['n']}, WR={st_dd['wr']:.1f}%", flush=True)
    print(f"    Normal: Sharpe={st_normal['sharpe']:.3f}, N={st_normal['n']}, WR={st_normal['wr']:.1f}%", flush=True)
    results['during_dd'] = st_dd
    results['normal'] = st_normal

    # 8D: Rolling Sharpe
    print(f"\n  --- 8D: Rolling Sharpe (6-month windows) ---", flush=True)
    rolling_sharpes = []
    for yr in range(2015, 2027):
        for half in [1, 2]:
            start = pd.Timestamp(f"{yr}-{1 if half==1 else 7}-01")
            end = pd.Timestamp(f"{yr}-{7 if half==1 else 1}-01") if half==1 else pd.Timestamp(f"{yr+1}-01-01")
            period_trades = [t for t in base_merged_sorted if start <= pd.Timestamp(t['entry_time']) < end]
            if period_trades:
                st = _stats(period_trades)
                rolling_sharpes.append({'period': f"{yr}H{half}", 'sharpe': st['sharpe'], 'n': st['n'], 'pnl': st['pnl']})
    for rs in rolling_sharpes:
        print(f"    {rs['period']}: Sharpe={rs['sharpe']:.3f}, N={rs['n']}, PnL=${rs['pnl']:>8,.0f}", flush=True)
    results['rolling_sharpe'] = rolling_sharpes
    min_rs = min(rolling_sharpes, key=lambda x: x['sharpe']) if rolling_sharpes else {}
    max_rs = max(rolling_sharpes, key=lambda x: x['sharpe']) if rolling_sharpes else {}
    print(f"\n    Worst: {min_rs.get('period','')} Sharpe={min_rs.get('sharpe',0):.3f}", flush=True)
    print(f"    Best: {max_rs.get('period','')} Sharpe={max_rs.get('sharpe',0):.3f}", flush=True)

    with open(OUTPUT_DIR / "phase_8_drawdown.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 8 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 9: Multi-Timeframe Confluence
# ═══════════════════════════════════════════════════════════════
def phase_9(h1, pctl):
    if phase_done("phase_9_mtf"): print("  Phase 9 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 9: MULTI-TIMEFRAME CONFLUENCE\n{'='*100}", flush=True)
    results = {}

    # Create D1 data from H1
    d1 = h1.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    d1['ATR'] = compute_atr(d1, period=14)
    d1['EMA20'] = d1['Close'].ewm(span=20, adjust=False).mean()
    d1['EMA50'] = d1['Close'].ewm(span=50, adjust=False).mean()
    d1['RSI14'] = compute_rsi(d1['Close'], 14)
    d1['ADX'] = compute_adx(d1, period=14)

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_merged = port_merge(base_all)
    base_stats = _stats(base_merged)
    results['baseline'] = base_stats

    # 9A: D1 EMA trend alignment
    print(f"\n  --- 9A: D1 EMA20 Trend Filter ---", flush=True)
    d1_trend = (d1['Close'] > d1['EMA20']).astype(int) - (d1['Close'] < d1['EMA20']).astype(int)
    d1_trend_aligned = d1_trend.reindex(h1.index, method='ffill')

    filtered = [t for t in base_merged
                if (t['dir']=='BUY' and pd.Timestamp(t['entry_time']) in d1_trend_aligned.index and
                    d1_trend_aligned.loc[pd.Timestamp(t['entry_time'])] >= 0)
                or (t['dir']=='SELL' and pd.Timestamp(t['entry_time']) in d1_trend_aligned.index and
                    d1_trend_aligned.loc[pd.Timestamp(t['entry_time'])] <= 0)]
    st = _stats(filtered)
    delta = st['sharpe'] - base_stats['sharpe']
    removed = base_stats['n'] - st['n']
    print(f"    D1 EMA20 trend: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
    results['d1_ema20'] = {'stats': st, 'removed': removed}

    # 9B: D1 EMA50 trend filter
    print(f"\n  --- 9B: D1 EMA50 Trend Filter ---", flush=True)
    d1_trend50 = (d1['Close'] > d1['EMA50']).astype(int) - (d1['Close'] < d1['EMA50']).astype(int)
    d1_trend50_aligned = d1_trend50.reindex(h1.index, method='ffill')

    filtered = [t for t in base_merged
                if (t['dir']=='BUY' and pd.Timestamp(t['entry_time']) in d1_trend50_aligned.index and
                    d1_trend50_aligned.loc[pd.Timestamp(t['entry_time'])] >= 0)
                or (t['dir']=='SELL' and pd.Timestamp(t['entry_time']) in d1_trend50_aligned.index and
                    d1_trend50_aligned.loc[pd.Timestamp(t['entry_time'])] <= 0)]
    st = _stats(filtered)
    delta = st['sharpe'] - base_stats['sharpe']
    removed = base_stats['n'] - st['n']
    print(f"    D1 EMA50 trend: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
    results['d1_ema50'] = {'stats': st, 'removed': removed}

    # 9C: D1 ADX filter (only trade when D1 trend is strong)
    print(f"\n  --- 9C: D1 ADX Filter ---", flush=True)
    d1_adx = d1['ADX'].reindex(h1.index, method='ffill')
    for adx_thresh in [15, 20, 25, 30]:
        filtered = [t for t in base_merged
                    if pd.Timestamp(t['entry_time']) in d1_adx.index and
                       not np.isnan(d1_adx.loc[pd.Timestamp(t['entry_time'])]) and
                       d1_adx.loc[pd.Timestamp(t['entry_time'])] >= adx_thresh]
        st = _stats(filtered)
        delta = st['sharpe'] - base_stats['sharpe']
        removed = base_stats['n'] - st['n']
        print(f"    D1 ADX>={adx_thresh}: Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
        results[f'd1_adx_{adx_thresh}'] = {'stats': st, 'removed': removed}

    # 9D: D1 RSI overbought/oversold filter
    print(f"\n  --- 9D: D1 RSI Extremes Filter ---", flush=True)
    d1_rsi = d1['RSI14'].reindex(h1.index, method='ffill')
    # Skip BUY if D1 RSI > 75 (overbought), skip SELL if D1 RSI < 25 (oversold)
    for rsi_extreme in [70, 75, 80]:
        filtered = [t for t in base_merged
                    if not (t['dir']=='BUY' and pd.Timestamp(t['entry_time']) in d1_rsi.index and
                           not np.isnan(d1_rsi.loc[pd.Timestamp(t['entry_time'])]) and
                           d1_rsi.loc[pd.Timestamp(t['entry_time'])] > rsi_extreme)
                    and not (t['dir']=='SELL' and pd.Timestamp(t['entry_time']) in d1_rsi.index and
                            not np.isnan(d1_rsi.loc[pd.Timestamp(t['entry_time'])]) and
                            d1_rsi.loc[pd.Timestamp(t['entry_time'])] < (100-rsi_extreme))]
        st = _stats(filtered)
        delta = st['sharpe'] - base_stats['sharpe']
        removed = base_stats['n'] - st['n']
        print(f"    D1 RSI filter ({100-rsi_extreme}<RSI<{rsi_extreme}): Sharpe={st['sharpe']:.3f} ({delta:+.3f}), Removed={removed}", flush=True)
        results[f'd1_rsi_{rsi_extreme}'] = {'stats': st, 'removed': removed}

    # Validate best MTF filter
    best_mtf = None; best_delta_m = 0
    for key, val in results.items():
        if isinstance(val, dict) and 'stats' in val:
            d = val['stats']['sharpe'] - base_stats['sharpe']
            if d > best_delta_m: best_delta_m = d; best_mtf = key

    if best_mtf and best_delta_m > 0.1:
        print(f"\n  Best MTF: {best_mtf} ({best_delta_m:+.3f}) — validating...", flush=True)
        # Build run_new based on best filter type
        def run_base(h1f): return port_sharpe(run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30))

        if 'd1_adx' in best_mtf:
            thresh = int(best_mtf.split('_')[-1])
            def run_new(h1f):
                d1f = h1f.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
                d1f_adx = compute_adx(d1f).reindex(h1f.index, method='ffill')
                all_t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
                merged = port_merge(all_t)
                f = [t for t in merged if pd.Timestamp(t['entry_time']) in d1f_adx.index and not np.isnan(d1f_adx.loc[pd.Timestamp(t['entry_time'])]) and d1f_adx.loc[pd.Timestamp(t['entry_time'])] >= thresh]
                return _stats(f)['sharpe']
        else:
            def run_new(h1f): return run_base(h1f)

        validation = full_validate(h1, run_new, run_base, port_merge(base_all), port_merge(base_all), best_mtf)
        results['best_validation'] = validation
    else:
        results['best_validation'] = {'verdict': 'NO CANDIDATE'}

    with open(OUTPUT_DIR / "phase_9_mtf.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 9 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 10: Execution Edge
# ═══════════════════════════════════════════════════════════════
def phase_10(h1, pctl):
    if phase_done("phase_10_execution"): print("  Phase 10 cached", flush=True); return
    print(f"\n{'='*100}\n  PHASE 10: EXECUTION EDGE\n{'='*100}", flush=True)
    results = {}

    base_all = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    base_stats = _stats(port_merge(base_all))
    results['baseline'] = base_stats

    # 10A: Spread sensitivity analysis
    print(f"\n  --- 10A: Spread Sensitivity ---", flush=True)
    global SPREAD
    orig_spread = SPREAD
    for test_spread in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00]:
        SPREAD = test_spread
        all_t = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        st = _stats(port_merge(all_t))
        print(f"    Spread={test_spread:.2f}: Sharpe={st['sharpe']:.3f}, PnL=${st['pnl']:>10,.0f}", flush=True)
        results[f'spread_{test_spread}'] = st
    SPREAD = orig_spread

    # 10B: Entry improvement (limit order simulation — enter at slight pullback)
    print(f"\n  --- 10B: Entry Improvement (limit pullback) ---", flush=True)
    base_merged = port_merge(base_all)
    for improvement_pips in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        adjusted_trades = []
        for t in base_merged:
            t_adj = t.copy()
            t_adj['pnl'] = t['pnl'] + improvement_pips * CURRENT_CONFIG[t['strategy']]['lot'] * PV
            adjusted_trades.append(t_adj)
        st = _stats(adjusted_trades)
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    Entry improvement {improvement_pips:.2f}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'entry_improve_{improvement_pips}'] = st

    # 10C: Slippage impact
    print(f"\n  --- 10C: Slippage Impact ---", flush=True)
    for slip_pips in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        adjusted_trades = []
        for t in base_merged:
            t_adj = t.copy()
            t_adj['pnl'] = t['pnl'] - slip_pips * CURRENT_CONFIG[t['strategy']]['lot'] * PV
            adjusted_trades.append(t_adj)
        st = _stats(adjusted_trades)
        delta = st['sharpe'] - base_stats['sharpe']
        print(f"    Slippage {slip_pips:.2f}: Sharpe={st['sharpe']:.3f} ({delta:+.3f})", flush=True)
        results[f'slippage_{slip_pips}'] = st

    # 10D: Time-in-market efficiency
    print(f"\n  --- 10D: Time-in-Market ---", flush=True)
    total_bars = len(h1)
    bars_in_market = sum(t['bars'] for t in base_merged)
    utilization = bars_in_market / total_bars * 100
    avg_hold = np.mean([t['bars'] for t in base_merged])
    print(f"    Total bars: {total_bars}, In-market: {bars_in_market}, Utilization: {utilization:.1f}%", flush=True)
    print(f"    Avg hold: {avg_hold:.1f} bars", flush=True)

    # Per-strategy hold time
    for strat in STRAT_ORDER:
        strat_trades = [t for t in base_merged if t['strategy'] == strat]
        if strat_trades:
            avg_h = np.mean([t['bars'] for t in strat_trades])
            print(f"    {strat}: avg_hold={avg_h:.1f} bars, N={len(strat_trades)}", flush=True)
            results[f'hold_{strat}'] = {'avg_bars': round(avg_h, 1), 'n': len(strat_trades)}

    results['utilization'] = round(utilization, 1)
    results['avg_hold'] = round(avg_hold, 1)

    # 10E: Exit reason breakdown
    print(f"\n  --- 10E: Exit Reason Breakdown ---", flush=True)
    reasons = {}
    for t in base_merged:
        r = t['reason']
        if r not in reasons: reasons[r] = {'n': 0, 'pnl': 0}
        reasons[r]['n'] += 1
        reasons[r]['pnl'] += t['pnl']
    for reason, data in sorted(reasons.items(), key=lambda x: -x[1]['n']):
        pct = data['n'] / len(base_merged) * 100
        avg_pnl = data['pnl'] / data['n']
        print(f"    {reason}: N={data['n']} ({pct:.1f}%), Total PnL=${data['pnl']:,.0f}, Avg=${avg_pnl:.2f}", flush=True)
        results[f'exit_{reason}'] = {'n': data['n'], 'pct': round(pct, 1), 'total_pnl': round(data['pnl'], 2), 'avg_pnl': round(avg_pnl, 2)}

    with open(OUTPUT_DIR / "phase_10_execution.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 10 done ({(time.time()-t0)/60:.1f}m)", flush=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"{'='*100}")
    print(f"  R195 — 100-HOUR GOLD TECHNICAL EXPLORATION")
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
    print(f"  R195 COMPLETE — {total_m:.1f} minutes")
    print(f"{'='*100}")
