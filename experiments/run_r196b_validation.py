#!/usr/bin/env python3
"""
R196b — K-Fold + Walk-Forward + Era Validation of R196 Top Findings
====================================================================
Validates the best parameter improvements from R196:
1. Keltner: kc=1.5, ema=30  (Sharpe 7.19 -> 7.54)
2. PSAR: step=0.03, max=0.10  (Sharpe 7.16 -> 7.55)
3. TSMOM: fast=480, slow=1200  (Sharpe 6.39 -> 6.66)
4. Dual Thrust: k=0.4, nb=10  (Sharpe 7.00 -> 7.72)
5. Chandelier: atr_p=14  (Sharpe 6.32 -> 6.70)
6. Session BO: hour=13, lb=3  (Sharpe 7.57 -> 7.64)

Each finding is tested with:
- 6-Fold Cross-Validation (PASS = 4/6 folds win)
- 19-Window Walk-Forward (PASS = 13/19 windows win)
- Era Segmentation (PASS = all eras positive, no era degrades > 0.3 Sharpe)
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

OUTPUT_DIR = Path("results/r196b_validation")
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
import glob as _glob
t0 = time.time()

def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()

# ═══════════════ Core helpers (same as R196) ═══════════════
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

# ═══════════════ Strategy BT functions (same as R196) ═══════════════
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

# ═══════════════ Validation framework ═══════════════
def kfold_test(h1, run_new_fn, run_base_fn, n_folds=6):
    n = len(h1); fold_size = n // n_folds; wins = 0
    for fold in range(n_folds):
        start = fold * fold_size
        end = min(start + fold_size, n)
        h1_fold = h1.iloc[start:end]
        if len(h1_fold) < 1000: continue
        new_sh = run_new_fn(h1_fold)
        base_sh = run_base_fn(h1_fold)
        if new_sh > base_sh: wins += 1
    return wins, n_folds

def wf_test(h1, run_new_fn, run_base_fn, train_pct=0.6, n_windows=19):
    n = len(h1); wins = 0
    step = int(n * (1 - train_pct) / n_windows)
    for w in range(n_windows):
        oos_start = int(n * train_pct) + w * step
        oos_end = min(oos_start + step, n)
        if oos_end <= oos_start: continue
        h1_oos = h1.iloc[oos_start:oos_end]
        if len(h1_oos) < 200: continue
        new_sh = run_new_fn(h1_oos)
        base_sh = run_base_fn(h1_oos)
        if new_sh > base_sh: wins += 1
    return wins, n_windows

ERAS = {
    'pre_hike': ('2015-01-01', '2016-12-31'),
    'hike': ('2017-01-01', '2019-06-30'),
    'cut_covid': ('2019-07-01', '2021-12-31'),
    'hike_2022': ('2022-01-01', '2023-12-31'),
    'recent': ('2024-01-01', '2026-12-31'),
}

def era_test(h1, run_new_fn, run_base_fn):
    results = {}
    for era_name, (start, end) in ERAS.items():
        mask = (h1.index >= start) & (h1.index <= end)
        h1_era = h1[mask]
        if len(h1_era) < 500: continue
        new_sh = run_new_fn(h1_era)
        base_sh = run_base_fn(h1_era)
        results[era_name] = {'new': round(new_sh, 3), 'base': round(base_sh, 3), 'delta': round(new_sh - base_sh, 3)}
    return results

def full_validate(h1, pctl, label, run_new_fn, run_base_fn):
    print(f"\n  === VALIDATING: {label} ===", flush=True)

    # K-Fold
    kf_wins, kf_total = kfold_test(h1, run_new_fn, run_base_fn)
    kf_pass = kf_wins >= 4
    print(f"    K-Fold: {kf_wins}/{kf_total} ({'PASS' if kf_pass else 'FAIL'})", flush=True)

    # Walk-Forward
    wf_wins, wf_total = wf_test(h1, run_new_fn, run_base_fn)
    wf_pass = wf_wins >= 13
    print(f"    Walk-Forward: {wf_wins}/{wf_total} ({'PASS' if wf_pass else 'FAIL'})", flush=True)

    # Era
    era_results = era_test(h1, run_new_fn, run_base_fn)
    era_all_positive = all(v['new'] > 0 for v in era_results.values())
    era_no_degrade = all(v['delta'] > -0.3 for v in era_results.values())
    era_pass = era_all_positive and era_no_degrade
    print(f"    Era: all_positive={era_all_positive}, no_degrade={era_no_degrade} ({'PASS' if era_pass else 'FAIL'})", flush=True)
    for era, v in era_results.items():
        print(f"      {era}: new={v['new']}, base={v['base']}, delta={v['delta']:+.3f}", flush=True)

    all_pass = kf_pass and wf_pass and era_pass
    verdict = 'GO' if all_pass else 'NO-GO'
    print(f"    >>> VERDICT: {verdict} <<<", flush=True)

    return {
        'label': label,
        'kfold': {'wins': kf_wins, 'total': kf_total, 'pass': kf_pass},
        'wf': {'wins': wf_wins, 'total': wf_total, 'pass': wf_pass},
        'era': {'results': era_results, 'all_positive': era_all_positive, 'no_degrade': era_no_degrade, 'pass': era_pass},
        'verdict': verdict
    }

# ═══════════════ VALIDATION TARGETS ═══════════════
def validate_all(h1, pctl):
    if phase_done("r196b_results"): print("  Already done",flush=True); return
    results = {}
    cfg = copy.deepcopy(CURRENT_CONFIG)

    # 1. Keltner: kc=1.5, ema=30 vs baseline (kc=1.2, ema=25)
    def keltner_new(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_keltner_param(h1f, cfg['L8_MAX'], p, kc_mult=1.5, ema_span=30))['sharpe']
    def keltner_base(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_keltner_param(h1f, cfg['L8_MAX'], p, kc_mult=1.2, ema_span=25))['sharpe']
    results['keltner_kc15_ema30'] = full_validate(h1, pctl, "Keltner kc=1.5 ema=30", keltner_new, keltner_base)

    # 2. PSAR: step=0.03, max=0.10 vs baseline (step=0.01, max=0.05)
    def psar_new(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_psar_param(h1f, cfg['PSAR'], p, af_step=0.03, af_max=0.10))['sharpe']
    def psar_base(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_psar_param(h1f, cfg['PSAR'], p, af_step=0.01, af_max=0.05))['sharpe']
    results['psar_af03_max10'] = full_validate(h1, pctl, "PSAR step=0.03 max=0.10", psar_new, psar_base)

    # 3. TSMOM: fast=480, slow=1200 vs baseline (fast=480, slow=720)
    def tsmom_new(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_tsmom_param(h1f, cfg['TSMOM'], p, fast_lb=480, slow_lb=1200))['sharpe']
    def tsmom_base(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_tsmom_param(h1f, cfg['TSMOM'], p, fast_lb=480, slow_lb=720))['sharpe']
    results['tsmom_480_1200'] = full_validate(h1, pctl, "TSMOM fast=480 slow=1200", tsmom_new, tsmom_base)

    # 4. Dual Thrust: k=0.4, nb=10 vs baseline (k=0.5, nb=6)
    def dt_new(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_dt_param(h1f, cfg['DUAL_THRUST'], p, k_factor=0.4, nb=10))['sharpe']
    def dt_base(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_dt_param(h1f, cfg['DUAL_THRUST'], p, k_factor=0.5, nb=6))['sharpe']
    results['dt_k04_nb10'] = full_validate(h1, pctl, "DualThrust k=0.4 nb=10", dt_new, dt_base)

    # 5. Chandelier: atr_period=14 vs baseline (atr_period=22)
    def chand_new(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_chand_param(h1f, cfg['CHANDELIER'], p, atr_period=14))['sharpe']
    def chand_base(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_chand_param(h1f, cfg['CHANDELIER'], p, atr_period=22))['sharpe']
    results['chandelier_atrp14'] = full_validate(h1, pctl, "Chandelier atr_period=14", chand_new, chand_base)

    # 6. Session BO: hour=13, lb=3 vs baseline (hour=12, lb=4)
    def sess_new(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_sess_bo_param(h1f, cfg['SESS_BO'], p, lb=3, entry_hour=13))['sharpe']
    def sess_base(h1f):
        p = compute_atr_pctl(compute_atr(h1f), lb=min(300, len(h1f)//3))
        return _stats(bt_sess_bo_param(h1f, cfg['SESS_BO'], p, lb=4, entry_hour=12))['sharpe']
    results['sessbo_hr13_lb3'] = full_validate(h1, pctl, "SessionBO hour=13 lb=3", sess_new, sess_base)

    # Summary
    print(f"\n{'='*100}", flush=True)
    print(f"  R196b VALIDATION SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  {'Strategy':<30} {'K-Fold':<12} {'WF':<12} {'Era':<12} {'Verdict':<10}", flush=True)
    print(f"  {'-'*76}", flush=True)
    for key, r in results.items():
        kf_str = f"{r['kfold']['wins']}/{r['kfold']['total']}"
        wf_str = f"{r['wf']['wins']}/{r['wf']['total']}"
        era_str = 'PASS' if r['era']['pass'] else 'FAIL'
        print(f"  {r['label']:<30} {kf_str:<12} {wf_str:<12} {era_str:<12} {r['verdict']:<10}", flush=True)

    with open(OUTPUT_DIR / "r196b_results.json", 'w') as f: json.dump(results, f, indent=2, default=str)
    print(f"\n  Total time: {(time.time()-t0)/60:.1f}m", flush=True)

# ═══════════════ MAIN ═══════════════
if __name__ == '__main__':
    print(f"{'='*100}")
    print(f"  R196b — DEEP VALIDATION OF R196 TOP FINDINGS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    h1 = load_h1()
    df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
    pctl = compute_atr_pctl(df_temp['ATR'], lb=300)
    print(f"  ATR pctl: {pctl.notna().sum()} valid\n", flush=True)

    validate_all(h1, pctl)

    print(f"\n{'='*100}")
    print(f"  R196b COMPLETE — {(time.time()-t0)/60:.1f} minutes")
    print(f"{'='*100}")
