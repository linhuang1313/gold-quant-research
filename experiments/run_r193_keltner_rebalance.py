#!/usr/bin/env python3
"""
R193 — Keltner 0.04 Lot Rebalance + 6-Strategy Portfolio Optimization
======================================================================
Phase 1: Keltner 0.04 Cap sweep ($35~$100)
Phase 2: Keltner 0.04 SL/TP sweep (best Cap from Phase 1)
Phase 3: 6-strategy portfolio lot rebalance (Keltner=0.04 fixed)
Phase 4: Final validation — best portfolio vs current

Constraints:
  - Capital = $5,000
  - Single trade max loss <= $75 (1.5%)
  - 6-strategy worst-case simultaneous loss <= $500 (10%)
  - Keltner trail = fixed 0.06/0.01 (R192b GO)
  - Keltner SL = 6.0 (R192b GO)
"""
import sys, os, time, json, warnings, copy, itertools
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r193_rebalance")
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
# Strategy backtests (from R192, verified)
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


# ═══════════════════════════════════════════════════════════════
# PHASE 1: Keltner 0.04 Cap Sweep
# ═══════════════════════════════════════════════════════════════
def phase_1(h1, pctl):
    print(f"\n{'='*120}\n  PHASE 1: Keltner 0.04 Cap Sweep\n{'='*120}", flush=True)
    results = {}

    cap_values = [35, 50, 60, 70, 80, 100, 0]
    base_cfg = {'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}

    print(f"\n  {'Cap':>6} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'Cap%':>6} {'SL%':>5} {'Trail%':>7} {'TO%':>5} {'MaxLoss':>8}")
    best_sharpe = -999; best_cap = 70

    for cap_v in cap_values:
        cfg = copy.deepcopy(base_cfg); cfg['cap'] = cap_v
        t = bt_keltner(h1, cfg, pctl_v=pctl, pctl_f=30)
        s = _stats(t)
        label = f"${cap_v}" if cap_v > 0 else "None"
        print(f"  {label:>6} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['wr']:>5.1f}% {s['cap_pct']:>5.1f}% {s['sl_pct']:>4.1f}% {s['trail_pct']:>6.1f}% {s['timeout_pct']:>4.1f}% ${s['max_loss']:>7.0f}")
        results[f'cap_{cap_v}'] = s
        if s['sharpe'] > best_sharpe:
            best_sharpe = s['sharpe']; best_cap = cap_v

    print(f"\n  Best Cap: ${best_cap} (Sharpe={best_sharpe:.3f})")

    # K-Fold validation for top-3 Caps
    sorted_caps = sorted([(results[f'cap_{c}']['sharpe'], c) for c in cap_values], reverse=True)
    top_caps = [c for _, c in sorted_caps[:3]]

    # Use current config as baseline for K-Fold
    base_current = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])  # lot=0.02, cap=35
    def run_current(h1f): return _stats(bt_keltner(h1f, base_current, pctl_v=pctl, pctl_f=30))['sharpe']

    for cap_v in top_caps:
        cfg_new = copy.deepcopy(base_cfg); cfg_new['cap'] = cap_v
        def run_new(h1f, _cfg=cfg_new): return _stats(bt_keltner(h1f, _cfg, pctl_v=pctl, pctl_f=30))['sharpe']
        kf = kfold_test(h1, run_new, run_current)
        kf_wins = sum(1 for r in kf if r['win']=='NEW')
        wf = wf_test(h1, run_new, run_current)
        wf_wins = sum(1 for r in wf if r['win']=='NEW')
        print(f"  Cap=${cap_v}: K-Fold {kf_wins}/{len(kf)}, WF {wf_wins}/{len(wf)}")
        results[f'cap_{cap_v}_kf'] = kf_wins
        results[f'cap_{cap_v}_wf'] = wf_wins

    results['best_cap'] = best_cap
    with open(OUTPUT_DIR / "phase_1.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Saved phase_1.json", flush=True)
    return best_cap, results


# ═══════════════════════════════════════════════════════════════
# PHASE 2: Keltner 0.04 SL/TP Sweep (with best Cap)
# ═══════════════════════════════════════════════════════════════
def phase_2(h1, pctl, best_cap):
    print(f"\n{'='*120}\n  PHASE 2: Keltner 0.04 SL/TP Sweep (Cap=${best_cap})\n{'='*120}", flush=True)
    results = {}

    sl_values = [4.0, 5.0, 6.0, 7.0, 8.0]
    tp_values = [6.0, 8.0, 10.0, 12.0]

    print(f"\n  {'SL':>4} {'TP':>4} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Cap%':>6} {'SL%':>5} {'Trail%':>7}")
    best_sharpe = -999; best_sl = 6.0; best_tp = 8.0

    for sl_v in sl_values:
        for tp_v in tp_values:
            cfg = {'lot': 0.04, 'cap': best_cap, 'sl': sl_v, 'tp': tp_v, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}
            t = bt_keltner(h1, cfg, pctl_v=pctl, pctl_f=30)
            s = _stats(t)
            print(f"  {sl_v:>4.1f} {tp_v:>4.1f} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['cap_pct']:>5.1f}% {s['sl_pct']:>4.1f}% {s['trail_pct']:>6.1f}%")
            results[f'sl{sl_v}_tp{tp_v}'] = s
            if s['sharpe'] > best_sharpe:
                best_sharpe = s['sharpe']; best_sl = sl_v; best_tp = tp_v

    print(f"\n  Best: SL={best_sl}, TP={best_tp} (Sharpe={best_sharpe:.3f})")

    # K-Fold/WF for best config vs current
    best_cfg = {'lot': 0.04, 'cap': best_cap, 'sl': best_sl, 'tp': best_tp, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}
    base_current = copy.deepcopy(CURRENT_CONFIG['L8_MAX'])

    def run_new(h1f): return _stats(bt_keltner(h1f, best_cfg, pctl_v=pctl, pctl_f=30))['sharpe']
    def run_cur(h1f): return _stats(bt_keltner(h1f, base_current, pctl_v=pctl, pctl_f=30))['sharpe']

    kf = kfold_test(h1, run_new, run_cur)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    wf = wf_test(h1, run_new, run_cur)
    wf_wins = sum(1 for r in wf if r['win']=='NEW')

    t_new = bt_keltner(h1, best_cfg, pctl_v=pctl, pctl_f=30)
    t_cur = bt_keltner(h1, base_current, pctl_v=pctl, pctl_f=30)
    era = era_test(h1, t_new, t_cur)

    print(f"\n  Best config K-Fold: {kf_wins}/{len(kf)}, WF: {wf_wins}/{len(wf)}")
    for e in ['full','hike','cut','recent_3y']:
        print(f"  Era {e}: current={era[e]['base']:.3f} new={era[e]['new']:.3f} (d={era[e]['delta']:+.3f})")
    for r in kf:
        print(f"    Fold {r['fold']}: cur={r['base']:.3f} new={r['new']:.3f} -> {r['win']}")

    v = verdict(kf, wf, era)
    print(f"  >>> VERDICT: {v['verdict']}")

    results['best_sl'] = best_sl; results['best_tp'] = best_tp
    results['kf'] = kf; results['wf_wins'] = wf_wins; results['era'] = era; results['verdict'] = v
    with open(OUTPUT_DIR / "phase_2.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Saved phase_2.json", flush=True)
    return best_sl, best_tp, results


# ═══════════════════════════════════════════════════════════════
# PHASE 3: 6-Strategy Portfolio Lot Rebalance
# ═══════════════════════════════════════════════════════════════
def phase_3(h1, pctl, keltner_cfg):
    print(f"\n{'='*120}\n  PHASE 3: Portfolio Lot Rebalance (Keltner fixed at 0.04)\n{'='*120}", flush=True)
    results = {}

    # Pre-compute all strategy trades at various lot sizes
    # Lot grid for each strategy (around current values)
    lot_grid = {
        'PSAR':        [0.04, 0.06, 0.09, 0.12],
        'TSMOM':       [0.02, 0.04, 0.06],
        'SESS_BO':     [0.02, 0.04, 0.06],
        'DUAL_THRUST': [0.02, 0.04, 0.06],
        'CHANDELIER':  [0.02, 0.03, 0.04],
    }

    # Cap scales with lot to maintain same price-distance
    # Current cap/lot ratios: PSAR=$60/0.09=$667/lot, TSMOM=$60/0.04=$1500/lot, etc.
    # We use a cap_per_lot_unit approach: cap = lot * PV * cap_atr_mult * median_atr
    # But simpler: scale cap proportionally with lot
    cap_base = {
        'PSAR': (60, 0.09), 'TSMOM': (60, 0.04), 'SESS_BO': (60, 0.04),
        'DUAL_THRUST': (18, 0.04), 'CHANDELIER': (25, 0.03),
    }

    # Pre-compute trades for each strategy at each lot (entry logic is lot-independent, only exit changes)
    print(f"\n  Pre-computing trade tables...")
    trade_cache = {}
    for nm in STRAT_ORDER[1:]:  # skip Keltner
        for lot in lot_grid[nm]:
            cfg = copy.deepcopy(CURRENT_CONFIG[nm])
            cfg['lot'] = lot
            base_cap, base_lot = cap_base[nm]
            cfg['cap'] = round(base_cap * lot / base_lot, 0)
            trade_cache[(nm, lot)] = run_strat(nm, h1, cfg, pctl_v=pctl, pctl_f=30)
            s = _stats(trade_cache[(nm, lot)])
            print(f"    {nm} lot={lot:.2f} cap=${cfg['cap']:.0f}: N={s['n']}, Sharpe={s['sharpe']:.3f}, PnL=${s['pnl']:,.0f}")

    # Keltner fixed
    kelt_trades = bt_keltner(h1, keltner_cfg, pctl_v=pctl, pctl_f=30)
    kelt_stats = _stats(kelt_trades)
    print(f"    L8_MAX lot=0.04 cap=${keltner_cfg['cap']}: N={kelt_stats['n']}, Sharpe={kelt_stats['sharpe']:.3f}, PnL=${kelt_stats['pnl']:,.0f}")

    # Grid search over lot combinations
    print(f"\n  Grid search over {len(lot_grid['PSAR'])*len(lot_grid['TSMOM'])*len(lot_grid['SESS_BO'])*len(lot_grid['DUAL_THRUST'])*len(lot_grid['CHANDELIER'])} combinations...")

    best_sharpe = -999; best_combo = None; combo_results = []

    for psar_lot in lot_grid['PSAR']:
        for tsmom_lot in lot_grid['TSMOM']:
            for sess_lot in lot_grid['SESS_BO']:
                for dt_lot in lot_grid['DUAL_THRUST']:
                    for ch_lot in lot_grid['CHANDELIER']:
                        # Check constraint: sum of all caps <= $500
                        caps = {
                            'L8_MAX': keltner_cfg['cap'],
                            'PSAR': round(cap_base['PSAR'][0] * psar_lot / cap_base['PSAR'][1], 0),
                            'TSMOM': round(cap_base['TSMOM'][0] * tsmom_lot / cap_base['TSMOM'][1], 0),
                            'SESS_BO': round(cap_base['SESS_BO'][0] * sess_lot / cap_base['SESS_BO'][1], 0),
                            'DUAL_THRUST': round(cap_base['DUAL_THRUST'][0] * dt_lot / cap_base['DUAL_THRUST'][1], 0),
                            'CHANDELIER': round(cap_base['CHANDELIER'][0] * ch_lot / cap_base['CHANDELIER'][1], 0),
                        }
                        total_cap = sum(caps.values())
                        if total_cap > 500: continue

                        # Check individual cap <= $75
                        if any(v > 75 for v in caps.values()): continue

                        all_trades = {
                            'L8_MAX': kelt_trades,
                            'PSAR': trade_cache[('PSAR', psar_lot)],
                            'TSMOM': trade_cache[('TSMOM', tsmom_lot)],
                            'SESS_BO': trade_cache[('SESS_BO', sess_lot)],
                            'DUAL_THRUST': trade_cache[('DUAL_THRUST', dt_lot)],
                            'CHANDELIER': trade_cache[('CHANDELIER', ch_lot)],
                        }
                        sh = port_sharpe(all_trades)
                        total_pnl = sum(_stats(all_trades[nm])['pnl'] for nm in STRAT_ORDER)
                        lots = {'L8_MAX': 0.04, 'PSAR': psar_lot, 'TSMOM': tsmom_lot,
                                'SESS_BO': sess_lot, 'DUAL_THRUST': dt_lot, 'CHANDELIER': ch_lot}

                        combo_results.append({
                            'lots': lots, 'caps': caps, 'sharpe': round(sh, 3),
                            'pnl': round(total_pnl, 0), 'total_cap': total_cap
                        })

                        if sh > best_sharpe:
                            best_sharpe = sh; best_combo = combo_results[-1]

    # Sort and display top-10
    combo_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10 portfolios (of {len(combo_results)} valid combos):")
    print(f"  {'#':>3} {'Sharpe':>7} {'PnL':>10} {'TotalCap':>9} | {'Kelt':>5} {'PSAR':>5} {'TSMOM':>5} {'SESS':>5} {'DT':>5} {'CH':>5}")
    for idx, r in enumerate(combo_results[:10]):
        l = r['lots']
        print(f"  {idx+1:>3} {r['sharpe']:>7.3f} ${r['pnl']:>9,.0f} ${r['total_cap']:>8,.0f} | {l['L8_MAX']:>5.2f} {l['PSAR']:>5.2f} {l['TSMOM']:>5.2f} {l['SESS_BO']:>5.2f} {l['DUAL_THRUST']:>5.2f} {l['CHANDELIER']:>5.2f}")

    # Also show current config for comparison
    current_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
    current_sh = port_sharpe(current_trades)
    current_pnl = sum(_stats(current_trades[nm])['pnl'] for nm in STRAT_ORDER)
    current_caps = sum(CURRENT_CONFIG[nm]['cap'] for nm in STRAT_ORDER)
    print(f"\n  CURRENT: Sharpe={current_sh:.3f}, PnL=${current_pnl:,.0f}, TotalCap=${current_caps}")
    print(f"  BEST:    Sharpe={best_combo['sharpe']:.3f}, PnL=${best_combo['pnl']:,.0f}, TotalCap=${best_combo['total_cap']}")

    results['top10'] = combo_results[:10]
    results['best'] = best_combo
    results['current_sharpe'] = round(current_sh, 3)
    results['current_pnl'] = round(current_pnl, 0)

    with open(OUTPUT_DIR / "phase_3.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Saved phase_3.json", flush=True)
    return best_combo, results


# ═══════════════════════════════════════════════════════════════
# PHASE 4: Final Validation — Best Portfolio vs Current
# ═══════════════════════════════════════════════════════════════
def phase_4(h1, pctl, best_combo, keltner_cfg):
    print(f"\n{'='*120}\n  PHASE 4: Final Validation\n{'='*120}", flush=True)
    results = {}

    # Build new config from best combo
    new_config = copy.deepcopy(CURRENT_CONFIG)
    new_config['L8_MAX'] = copy.deepcopy(keltner_cfg)
    cap_base = {
        'PSAR': (60, 0.09), 'TSMOM': (60, 0.04), 'SESS_BO': (60, 0.04),
        'DUAL_THRUST': (18, 0.04), 'CHANDELIER': (25, 0.03),
    }
    for nm in STRAT_ORDER[1:]:
        new_config[nm]['lot'] = best_combo['lots'][nm]
        base_cap, base_lot = cap_base[nm]
        new_config[nm]['cap'] = round(base_cap * best_combo['lots'][nm] / base_lot, 0)

    print(f"\n  New config:")
    print(f"  {'Strategy':<15} {'Lot':>5} {'Cap':>6} {'SL':>5} {'TP':>5}")
    for nm in STRAT_ORDER:
        c = new_config[nm]
        print(f"  {nm:<15} {c['lot']:>5.2f} ${c['cap']:>5.0f} {c['sl']:>5.1f} {c['tp']:>5.1f}")

    total_cap = sum(new_config[nm]['cap'] for nm in STRAT_ORDER)
    print(f"  Total worst-case Cap: ${total_cap} ({total_cap/CAPITAL*100:.1f}% of capital)")

    # Full-period comparison
    new_trades = run_all(h1, new_config, pctl_v=pctl, pctl_f=30)
    cur_trades = run_all(h1, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)

    new_merged = port_merge(new_trades); cur_merged = port_merge(cur_trades)
    s_new = _stats(new_merged); s_cur = _stats(cur_merged)

    print(f"\n  Full period:")
    print(f"  Current: Sharpe={s_cur['sharpe']:.3f}, PnL=${s_cur['pnl']:,.0f}, MaxDD=${s_cur['max_dd']:,.0f}")
    print(f"  New:     Sharpe={s_new['sharpe']:.3f}, PnL=${s_new['pnl']:,.0f}, MaxDD=${s_new['max_dd']:,.0f}")

    # Per-strategy comparison
    print(f"\n  Per-strategy:")
    print(f"  {'Strategy':<15} {'Cur_Sh':>7} {'New_Sh':>7} {'Delta':>7} {'Cur_PnL':>10} {'New_PnL':>10}")
    for nm in STRAT_ORDER:
        sc = _stats(cur_trades[nm]); sn = _stats(new_trades[nm])
        print(f"  {nm:<15} {sc['sharpe']:>7.3f} {sn['sharpe']:>7.3f} {sn['sharpe']-sc['sharpe']:>+7.3f} ${sc['pnl']:>9,.0f} ${sn['pnl']:>9,.0f}")

    # K-Fold
    def run_new_port(h1f):
        t = run_all(h1f, new_config, pctl_v=pctl, pctl_f=30)
        return _sharpe(_daily(port_merge(t)))
    def run_cur_port(h1f):
        t = run_all(h1f, CURRENT_CONFIG, pctl_v=pctl, pctl_f=30)
        return _sharpe(_daily(port_merge(t)))

    print(f"\n  K-Fold validation:")
    kf = kfold_test(h1, run_new_port, run_cur_port)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    for r in kf:
        print(f"    Fold {r['fold']}: cur={r['base']:.3f} new={r['new']:.3f} -> {r['win']}")

    # Walk-Forward
    wf = wf_test(h1, run_new_port, run_cur_port)
    wf_wins = sum(1 for r in wf if r['win']=='NEW')
    print(f"  K-Fold: {kf_wins}/{len(kf)}, WF: {wf_wins}/{len(wf)}")

    # Era
    era = era_test(h1, new_merged, cur_merged)
    for e in ['full','hike','cut','recent_3y']:
        print(f"  Era {e}: cur={era[e]['base']:.3f} new={era[e]['new']:.3f} (d={era[e]['delta']:+.3f})")

    v = verdict(kf, wf, era)
    print(f"\n  >>> FINAL VERDICT: {v['verdict']} (KF={v['kf_wins']}/{v['kf_total']}, WF={v['wf_wins']}/{v['wf_total']})")

    results['new_config'] = {nm: new_config[nm] for nm in STRAT_ORDER}
    results['full_new'] = s_new; results['full_cur'] = s_cur
    results['kf'] = kf; results['wf_wins'] = wf_wins; results['wf_total'] = len(wf)
    results['era'] = era; results['verdict'] = v
    results['total_cap'] = total_cap

    with open(OUTPUT_DIR / "phase_4.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"  Saved phase_4.json", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 120)
    print("  R193 — Keltner 0.04 Lot Rebalance + Portfolio Optimization")
    print("=" * 120, flush=True)

    h1 = load_h1()

    # Compute ATR percentile for R187 filter
    atr_series = compute_atr(h1).dropna()
    pctl = compute_atr_pctl(atr_series)

    # Phase 1: Cap sweep
    best_cap, p1 = phase_1(h1, pctl)

    # Phase 2: SL/TP sweep
    best_sl, best_tp, p2 = phase_2(h1, pctl, best_cap)

    # Build optimal Keltner config
    keltner_cfg = {'lot': 0.04, 'cap': best_cap, 'sl': best_sl, 'tp': best_tp,
                   'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}
    print(f"\n  Optimal Keltner: lot=0.04, cap=${best_cap}, SL={best_sl}, TP={best_tp}")

    # Phase 3: Portfolio rebalance
    best_combo, p3 = phase_3(h1, pctl, keltner_cfg)

    # Phase 4: Final validation
    p4 = phase_4(h1, pctl, best_combo, keltner_cfg)

    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"  R193 Complete! Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()
