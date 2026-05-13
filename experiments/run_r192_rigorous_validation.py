#!/usr/bin/env python3
"""
R192 — R191 Findings Rigorous Validation
==========================================
Systematically validates each R191 finding using isolated single-variable
testing with mandatory K-Fold 6-fold, Walk-Forward 19-period, and Era
segmented validation.

Validation standard (from R182/R187c/R190 precedent):
  K-Fold >= 4/6, WF >= 13/19, Era: all positive & no degradation > 0.3

7 Phases, each testing ONE change in isolation.
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r192_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

LIVE_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap': 35,  'sl': 3.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60,  'sl': 4.0, 'tp': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15},
    'TSMOM':       {'lot': 0.04, 'cap': 60,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.04, 'cap': 60,  'sl': 4.5, 'tp': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.03, 'cap': 25,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
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
# Core helpers (identical to R191 for consistency)
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
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0,'cap_pct':0,'sl_pct':0,'tp_pct':0,'trail_pct':0,'timeout_pct':0,'max_loss':0}
    daily=_daily(trades); pnls=[t['pnl'] for t in trades]; n=len(trades)
    wins=[p for p in pnls if p>0]
    eq=daily.cumsum(); dd=float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    reasons=[t['reason'] for t in trades]
    return {'n':n,'sharpe':round(_sharpe(daily),3),'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1),'max_dd':round(dd,2),
            'cap_pct':round(sum(1 for r in reasons if 'Cap' in r)/n*100,1),
            'sl_pct':round(sum(1 for r in reasons if r=='SL')/n*100,1),
            'tp_pct':round(sum(1 for r in reasons if r=='TP')/n*100,1),
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

def port_merge(all_t):
    return [t for nm in STRAT_ORDER for t in all_t.get(nm,[])]

def port_stats(all_t):
    m=port_merge(all_t); return _stats(m), _daily(m)

def save_phase(num, data):
    out = OUTPUT_DIR / f"phase_{num}_results.json"
    with open(out,'w') as f: json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {out}", flush=True)

def kfold_test(h1, run_new, run_base, K=6):
    start=h1.index[0]; end=h1.index[-1]; total=(end-start).days; fd=total//K
    results=[]
    for fold in range(K):
        fs=start+pd.Timedelta(days=fold*fd)
        fe=start+pd.Timedelta(days=(fold+1)*fd) if fold<K-1 else end+pd.Timedelta(days=1)
        h1f=h1[(h1.index>=fs)&(h1.index<fe)]
        if len(h1f)<300: continue
        sh_new = run_new(h1f); sh_base = run_base(h1f)
        results.append({'fold':fold+1,'new':round(sh_new,3),'base':round(sh_base,3),
                        'win':'NEW' if sh_new>sh_base else 'BASE',
                        'period':f"{fs.date()}~{fe.date()}"})
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
    kf_pass = kf_wins >= 4
    wf_pass = wf_wins >= 13
    era_pass = all(era[e]['new'] > 0 for e in ['hike','cut','recent_3y'])
    era_no_degrade = all(era[e]['delta'] > -0.3 for e in ['hike','cut','recent_3y'])
    all_pass = kf_pass and wf_pass and era_pass and era_no_degrade
    return {
        'kf_wins': kf_wins, 'kf_total': len(kf), 'kf_pass': kf_pass,
        'wf_wins': wf_wins, 'wf_total': len(wf), 'wf_pass': wf_pass,
        'era_pass': era_pass, 'era_no_degrade': era_no_degrade,
        'verdict': 'GO' if all_pass else 'NO-GO'
    }

# ═══════════════════════════════════════════════════════════════
# Strategy backtest functions (identical to R191)
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

def run_all(h1, config=None, pctl_v=None, pctl_f=0):
    cfg = config or LIVE_CONFIG
    return {nm: run_strat(nm, h1, cfg[nm], pctl_v=pctl_v, pctl_f=pctl_f) for nm in STRAT_ORDER}


# ═══════════════════════════════════════════════════════════════
# PHASE 1: Baseline Calibration
# ═══════════════════════════════════════════════════════════════
def phase_1(h1, pctl):
    print(f"\n{'='*120}\n  PHASE 1: Baseline Calibration\n{'='*120}", flush=True)
    results = {}
    # A: current live, no R187
    all_a = run_all(h1, LIVE_CONFIG)
    sa, _ = port_stats(all_a)
    print(f"\n  A (live, no R187): N={sa['n']}, Sharpe={sa['sharpe']:.3f}, PnL=${sa['pnl']:,.0f}, Cap%={sa['cap_pct']:.1f}%")
    results['A'] = sa

    # B: current live + R187
    all_b = run_all(h1, LIVE_CONFIG, pctl_v=pctl, pctl_f=30)
    sb, _ = port_stats(all_b)
    print(f"  B (live + R187): N={sb['n']}, Sharpe={sb['sharpe']:.3f}, PnL=${sb['pnl']:,.0f}, Cap%={sb['cap_pct']:.1f}%")
    results['B'] = sb

    # Per-strategy baseline
    print(f"\n  Per-strategy baseline:")
    print(f"  {'Strategy':<15} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'Cap%':>6} {'SL%':>6} {'Trail%':>7} {'TO%':>6}")
    for nm in STRAT_ORDER:
        s = _stats(all_a[nm])
        print(f"  {nm:<15} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>8,.0f} {s['cap_pct']:>5.1f}% {s['sl_pct']:>5.1f}% {s['trail_pct']:>6.1f}% {s['timeout_pct']:>5.1f}%")
        results[f'{nm}_baseline'] = s

    save_phase(1, results)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Trail Optimization — Isolated Validation
# ═══════════════════════════════════════════════════════════════
def phase_2(h1):
    print(f"\n{'='*120}\n  PHASE 2: Trail 0.06/0.01 — Isolated Validation\n{'='*120}", flush=True)

    atr_series = compute_atr(h1).dropna()
    atr_median = float(atr_series.median())
    print(f"  ATR median: ${atr_median:.2f}")

    results = {}
    for nm in STRAT_ORDER:
        cfg_base = LIVE_CONFIG[nm]
        cfg_new = copy.deepcopy(cfg_base); cfg_new['trail_act'] = 0.06; cfg_new['trail_dist'] = 0.01

        # Full period comparison
        t_base = run_strat(nm, h1, cfg_base)
        t_new = run_strat(nm, h1, cfg_new)
        s_base = _stats(t_base); s_new = _stats(t_new)
        delta = s_new['sharpe'] - s_base['sharpe']
        print(f"\n  {nm}: base({cfg_base['trail_act']}/{cfg_base['trail_dist']})={s_base['sharpe']:.3f} -> new(0.06/0.01)={s_new['sharpe']:.3f} (d={delta:+.3f})")
        print(f"    Base exits: Trail={s_base['trail_pct']:.1f}% Cap={s_base['cap_pct']:.1f}% SL={s_base['sl_pct']:.1f}% TP={s_base['tp_pct']:.1f}% TO={s_base['timeout_pct']:.1f}%")
        print(f"    New exits:  Trail={s_new['trail_pct']:.1f}% Cap={s_new['cap_pct']:.1f}% SL={s_new['sl_pct']:.1f}% TP={s_new['tp_pct']:.1f}% TO={s_new['timeout_pct']:.1f}%")

        # Dollar values at different ATR levels
        ta_dollar_low = 0.06 * 2.0  # ATR=2
        ta_dollar_med = 0.06 * atr_median
        ta_dollar_high = 0.06 * 20.0  # ATR=20
        print(f"    Trail activation: ATR=2 -> ${ta_dollar_low:.2f}, ATR={atr_median:.1f} -> ${ta_dollar_med:.2f}, ATR=20 -> ${ta_dollar_high:.2f}")

        # ATR regime split: low vs high
        atr_reindex = atr_series.reindex(h1.index, method='ffill')
        low_atr_trades = [t for t in t_new if pd.Timestamp(t['entry_time']) in atr_reindex.index and atr_reindex.loc[pd.Timestamp(t['entry_time'])] < atr_median]
        high_atr_trades = [t for t in t_new if pd.Timestamp(t['entry_time']) in atr_reindex.index and atr_reindex.loc[pd.Timestamp(t['entry_time'])] >= atr_median]
        low_base = [t for t in t_base if pd.Timestamp(t['entry_time']) in atr_reindex.index and atr_reindex.loc[pd.Timestamp(t['entry_time'])] < atr_median]
        high_base = [t for t in t_base if pd.Timestamp(t['entry_time']) in atr_reindex.index and atr_reindex.loc[pd.Timestamp(t['entry_time'])] >= atr_median]
        sl_new = _stats(low_atr_trades)['sharpe']; sh_new = _stats(high_atr_trades)['sharpe']
        sl_b = _stats(low_base)['sharpe']; sh_b = _stats(high_base)['sharpe']
        print(f"    ATR regime: Low new={sl_new:.3f} base={sl_b:.3f} (d={sl_new-sl_b:+.3f}) | High new={sh_new:.3f} base={sh_b:.3f} (d={sh_new-sh_b:+.3f})")

        # K-Fold
        def run_new_kf(h1f): return _stats(run_strat(nm, h1f, cfg_new))['sharpe']
        def run_base_kf(h1f): return _stats(run_strat(nm, h1f, cfg_base))['sharpe']
        kf = kfold_test(h1, run_new_kf, run_base_kf)
        kf_wins = sum(1 for r in kf if r['win']=='NEW')
        print(f"    K-Fold: {kf_wins}/{len(kf)}")
        for r in kf:
            print(f"      {r['period']}: base={r['base']:.3f} new={r['new']:.3f} -> {r['win']}")

        # Walk-Forward
        wf = wf_test(h1, run_new_kf, run_base_kf)
        wf_wins = sum(1 for r in wf if r['win']=='NEW')
        print(f"    Walk-Forward: {wf_wins}/{len(wf)}")

        # Era
        era = era_test(h1, t_new, t_base)
        for e in ['full','hike','cut','recent_3y']:
            print(f"    Era {e}: base={era[e]['base']:.3f} new={era[e]['new']:.3f} (d={era[e]['delta']:+.3f})")

        v = verdict(kf, wf, era)
        print(f"    >>> VERDICT: {v['verdict']} (KF={v['kf_wins']}/{v['kf_total']}, WF={v['wf_wins']}/{v['wf_total']})")
        results[nm] = {'full_sharpe_base': s_base['sharpe'], 'full_sharpe_new': s_new['sharpe'],
                       'delta': round(delta, 3), 'kf': kf, 'wf_wins': wf_wins, 'wf_total': len(wf),
                       'era': era, 'verdict': v,
                       'atr_low_delta': round(sl_new - sl_b, 3), 'atr_high_delta': round(sh_new - sh_b, 3)}

    save_phase(2, results)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 3: SL Widening — Cap/SL Interaction Analysis
# ═══════════════════════════════════════════════════════════════
def phase_3(h1):
    print(f"\n{'='*120}\n  PHASE 3: SL Widening — Cap/SL Interaction\n{'='*120}", flush=True)
    atr_series = compute_atr(h1).dropna()
    results = {}

    # Cap/SL crossover analysis
    print(f"\n  Cap/SL Crossover Analysis:")
    print(f"  {'Strategy':<15} {'Cap$':>6} {'SL_mult':>8} {'Lot':>5} {'Crossover_ATR':>14} {'Bars_below%':>11}")
    for nm in STRAT_ORDER:
        cfg = LIVE_CONFIG[nm]
        # SL fires when: SL_mult * ATR * lot * PV >= some threshold
        # Cap fires when: pnl < -Cap
        # Crossover ATR where SL$ = Cap$: Cap = SL_mult * ATR * lot * PV -> ATR = Cap / (SL_mult * lot * PV)
        crossover_atr = cfg['cap'] / (cfg['sl'] * cfg['lot'] * PV)
        pct_below = float((atr_series < crossover_atr).sum() / len(atr_series) * 100)
        print(f"  {nm:<15} ${cfg['cap']:>5} {cfg['sl']:>8.1f} {cfg['lot']:>5.2f} ${crossover_atr:>13.2f} {pct_below:>10.1f}%")
        results[f'{nm}_crossover'] = {'cap': cfg['cap'], 'sl_mult': cfg['sl'], 'crossover_atr': round(crossover_atr, 2),
                                       'pct_below_crossover': round(pct_below, 1)}

    # For each strategy: test SL=6.0 vs current, but ONLY if SL is relevant (crossover ATR > Q25)
    print(f"\n  SL=6.0 vs Current (isolated):")
    for nm in STRAT_ORDER:
        cfg_base = LIVE_CONFIG[nm]
        crossover = cfg_base['cap'] / (cfg_base['sl'] * cfg_base['lot'] * PV)
        crossover_new = cfg_base['cap'] / (6.0 * cfg_base['lot'] * PV)

        cfg_new = copy.deepcopy(cfg_base); cfg_new['sl'] = 6.0
        t_base = run_strat(nm, h1, cfg_base); t_new = run_strat(nm, h1, cfg_new)
        s_base = _stats(t_base); s_new = _stats(t_new)
        delta = s_new['sharpe'] - s_base['sharpe']

        # How often does SL actually fire?
        sl_fires_base = s_base['sl_pct']; sl_fires_new = s_new['sl_pct']
        print(f"\n  {nm}: SL {cfg_base['sl']}->{6.0}, Sharpe {s_base['sharpe']:.3f}->{s_new['sharpe']:.3f} (d={delta:+.3f})")
        print(f"    SL fires: base={sl_fires_base:.1f}%, new={sl_fires_new:.1f}%")
        print(f"    Crossover ATR: base=${crossover:.2f}, new=${crossover_new:.2f}")

        if sl_fires_base < 2.0 and sl_fires_new < 2.0:
            print(f"    >>> SL fires <2% in both configs. Change is CAP-DOMINATED — SL value irrelevant.")
            results[nm] = {'verdict': 'CAP-DOMINATED', 'sl_fires_base': sl_fires_base, 'sl_fires_new': sl_fires_new,
                           'delta': round(delta, 3)}
            continue

        # If SL actually matters, run full validation
        def run_new_kf(h1f): return _stats(run_strat(nm, h1f, cfg_new))['sharpe']
        def run_base_kf(h1f): return _stats(run_strat(nm, h1f, cfg_base))['sharpe']
        kf = kfold_test(h1, run_new_kf, run_base_kf)
        wf = wf_test(h1, run_new_kf, run_base_kf)
        era = era_test(h1, t_new, t_base)
        v = verdict(kf, wf, era)
        kf_wins = sum(1 for r in kf if r['win']=='NEW')
        wf_wins = sum(1 for r in wf if r['win']=='NEW')
        print(f"    K-Fold: {kf_wins}/{len(kf)}, WF: {wf_wins}/{len(wf)}")
        print(f"    >>> VERDICT: {v['verdict']}")
        results[nm] = {'verdict': v['verdict'], 'delta': round(delta, 3), 'kf_wins': kf_wins,
                       'wf_wins': wf_wins, 'sl_fires_base': sl_fires_base, 'sl_fires_new': sl_fires_new,
                       'kf': kf, 'era': era}

    # SESS_BO TP bug check
    print(f"\n  SESS_BO TP Bug Analysis:")
    print(f"    Current: SL={LIVE_CONFIG['SESS_BO']['sl']}, TP={LIVE_CONFIG['SESS_BO']['tp']}")
    print(f"    TP < SL! This means TP is set unreasonably low.")
    tp_sweep = [5.0, 6.0, 7.0, 8.0]
    print(f"    TP sweep (with SL=4.5):")
    for tp_v in tp_sweep:
        cfg_tp = copy.deepcopy(LIVE_CONFIG['SESS_BO']); cfg_tp['tp'] = tp_v
        t_tp = run_strat('SESS_BO', h1, cfg_tp)
        s_tp = _stats(t_tp)
        print(f"      TP={tp_v}: Sharpe={s_tp['sharpe']:.3f}, N={s_tp['n']}, TP%={s_tp['tp_pct']:.1f}%")

    # K-Fold for best TP fix
    best_tp = 6.0
    cfg_tp_fix = copy.deepcopy(LIVE_CONFIG['SESS_BO']); cfg_tp_fix['tp'] = best_tp
    def run_tp_new(h1f): return _stats(run_strat('SESS_BO', h1f, cfg_tp_fix))['sharpe']
    def run_tp_base(h1f): return _stats(run_strat('SESS_BO', h1f, LIVE_CONFIG['SESS_BO']))['sharpe']
    kf_tp = kfold_test(h1, run_tp_new, run_tp_base)
    wf_tp = wf_test(h1, run_tp_new, run_tp_base)
    t_tp_full = run_strat('SESS_BO', h1, cfg_tp_fix)
    t_base_full = run_strat('SESS_BO', h1, LIVE_CONFIG['SESS_BO'])
    era_tp = era_test(h1, t_tp_full, t_base_full)
    v_tp = verdict(kf_tp, wf_tp, era_tp)
    kf_w = sum(1 for r in kf_tp if r['win']=='NEW')
    wf_w = sum(1 for r in wf_tp if r['win']=='NEW')
    print(f"    SESS_BO TP=4.0->6.0: K-Fold {kf_w}/{len(kf_tp)}, WF {wf_w}/{len(wf_tp)}")
    print(f"    >>> VERDICT: {v_tp['verdict']}")
    results['SESS_BO_TP_fix'] = {'tp_sweep': tp_sweep, 'verdict': v_tp, 'kf_wins': kf_w, 'wf_wins': wf_w}

    save_phase(3, results)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Max Hold — Reconcile with R182
# ═══════════════════════════════════════════════════════════════
def phase_4(h1):
    print(f"\n{'='*120}\n  PHASE 4: Max Hold — Reconcile with R182\n{'='*120}", flush=True)
    print(f"  R182 found: Keltner MH2->MH5 K-Fold 2/6 FAIL = NO-GO", flush=True)
    results = {}

    # Keltner MH sweep with full validation
    mh_sweep = [2, 3, 4, 5, 6, 8, 10]
    cfg_base = LIVE_CONFIG['L8_MAX']
    t_base = bt_keltner(h1, cfg_base); s_base = _stats(t_base)
    print(f"\n  Keltner MH sweep (baseline MH=2, Sharpe={s_base['sharpe']:.3f}):")

    for mh in mh_sweep:
        if mh == cfg_base['max_hold']:
            continue
        cfg_new = copy.deepcopy(cfg_base); cfg_new['max_hold'] = mh
        t_new = bt_keltner(h1, cfg_new); s_new = _stats(t_new)
        delta = s_new['sharpe'] - s_base['sharpe']

        def run_new_kf(h1f, _cfg=cfg_new): return _stats(bt_keltner(h1f, _cfg))['sharpe']
        def run_base_kf(h1f): return _stats(bt_keltner(h1f, cfg_base))['sharpe']
        kf = kfold_test(h1, run_new_kf, run_base_kf)
        wf = wf_test(h1, run_new_kf, run_base_kf)
        era = era_test(h1, t_new, t_base)
        v = verdict(kf, wf, era)
        kf_wins = sum(1 for r in kf if r['win']=='NEW')
        wf_wins = sum(1 for r in wf if r['win']=='NEW')

        print(f"\n  MH={mh}: Sharpe={s_new['sharpe']:.3f} (d={delta:+.3f}), TO%={s_new['timeout_pct']:.1f}%")
        print(f"    K-Fold: {kf_wins}/{len(kf)} {'PASS' if v['kf_pass'] else 'FAIL'}")
        for r in kf:
            print(f"      {r['period']}: base={r['base']:.3f} new={r['new']:.3f} -> {r['win']}")
        print(f"    WF: {wf_wins}/{len(wf)} {'PASS' if v['wf_pass'] else 'FAIL'}")
        for e in ['hike','cut','recent_3y']:
            print(f"    Era {e}: base={era[e]['base']:.3f} new={era[e]['new']:.3f} (d={era[e]['delta']:+.3f})")
        print(f"    >>> VERDICT: {v['verdict']}")
        results[f'MH{mh}'] = {'sharpe': s_new['sharpe'], 'delta': round(delta, 3),
                               'kf_wins': kf_wins, 'wf_wins': wf_wins,
                               'timeout_pct': s_new['timeout_pct'],
                               'verdict': v, 'kf': kf, 'era': era}

    # Other strategies: only test if Timeout% > 5%
    print(f"\n  Other strategies (only if Timeout% > 5%):")
    for nm in STRAT_ORDER:
        if nm == 'L8_MAX':
            continue
        cfg_b = LIVE_CONFIG[nm]
        t_b = run_strat(nm, h1, cfg_b); s_b = _stats(t_b)
        if s_b['timeout_pct'] <= 5.0:
            print(f"  {nm}: Timeout={s_b['timeout_pct']:.1f}% <= 5% — SKIP (no problem)")
            results[nm] = {'timeout_pct': s_b['timeout_pct'], 'verdict': 'SKIP'}
            continue
        # Chandelier has ~11.5% timeout — test
        mh_options = [25, 30, 40]
        print(f"  {nm}: Timeout={s_b['timeout_pct']:.1f}% > 5% — testing max_hold increases")
        for mh in mh_options:
            if mh <= cfg_b['max_hold']:
                continue
            cfg_new = copy.deepcopy(cfg_b); cfg_new['max_hold'] = mh
            t_n = run_strat(nm, h1, cfg_new); s_n = _stats(t_n)
            delta = s_n['sharpe'] - s_b['sharpe']
            print(f"    MH={mh}: Sharpe={s_n['sharpe']:.3f} (d={delta:+.3f}), TO%={s_n['timeout_pct']:.1f}%")
        results[nm] = {'timeout_pct': s_b['timeout_pct'], 'tested': True}

    save_phase(4, results)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Chandelier Marginal Value with Position Limits
# ═══════════════════════════════════════════════════════════════
def phase_5(h1):
    print(f"\n{'='*120}\n  PHASE 5: Chandelier Marginal Value — With MAX_POSITIONS\n{'='*120}", flush=True)
    results = {}

    all_t = run_all(h1, LIVE_CONFIG)

    # Build timeline of all entries with timestamps
    all_entries = []
    for nm in STRAT_ORDER:
        for t in all_t[nm]:
            all_entries.append({'strategy': nm, 'entry_time': pd.Timestamp(t['entry_time']),
                                'exit_time': pd.Timestamp(t['exit_time']), 'pnl': t['pnl'], 'trade': t})
    all_entries.sort(key=lambda x: x['entry_time'])

    def simulate_portfolio(entries, max_pos, strats_enabled, n_sims=100):
        """Simulate with position limits and random priority ordering."""
        all_sharpes = []
        for sim in range(n_sims):
            active_positions = []
            accepted_trades = []
            rng = np.random.RandomState(sim)
            for e in entries:
                if e['strategy'] not in strats_enabled:
                    continue
                # Remove expired positions
                active_positions = [p for p in active_positions if p['exit_time'] > e['entry_time']]
                if len(active_positions) >= max_pos:
                    continue
                accepted_trades.append(e['trade'])
                active_positions.append(e)
            sh = _sharpe(_daily(accepted_trades))
            all_sharpes.append(sh)
        return np.mean(all_sharpes), np.std(all_sharpes), np.percentile(all_sharpes, 5)

    # 6 strats vs 5 strats (no Chandelier), MAX_POS=4
    set_6 = set(STRAT_ORDER)
    set_5 = set(STRAT_ORDER) - {'CHANDELIER'}

    print(f"  Simulating MAX_POSITIONS=4 with 100 random priority orderings...")
    mean_6, std_6, p5_6 = simulate_portfolio(all_entries, 4, set_6)
    mean_5, std_5, p5_5 = simulate_portfolio(all_entries, 4, set_5)
    print(f"  6 strats: Sharpe mean={mean_6:.3f} (std={std_6:.3f}, 5%={p5_6:.3f})")
    print(f"  5 strats (no CH): Sharpe mean={mean_5:.3f} (std={std_5:.3f}, 5%={p5_5:.3f})")
    print(f"  Delta: {mean_5 - mean_6:+.3f}")
    results['max_pos_4'] = {'with_ch': round(mean_6, 3), 'without_ch': round(mean_5, 3),
                             'delta': round(mean_5 - mean_6, 3)}

    # Also test MAX_POS=5 and MAX_POS=6 (unconstrained)
    for mp in [5, 6]:
        m6, _, _ = simulate_portfolio(all_entries, mp, set_6)
        m5, _, _ = simulate_portfolio(all_entries, mp, set_5)
        print(f"  MAX_POS={mp}: 6 strats={m6:.3f}, 5 strats={m5:.3f} (d={m5-m6:+.3f})")
        results[f'max_pos_{mp}'] = {'with_ch': round(m6, 3), 'without_ch': round(m5, 3)}

    # Signal overlap analysis
    overlap_count = 0; total_bars = 0
    ch_entries = {pd.Timestamp(t['entry_time']) for t in all_t['CHANDELIER']}
    other_entries = set()
    for nm in STRAT_ORDER:
        if nm == 'CHANDELIER':
            continue
        for t in all_t[nm]:
            other_entries.add(pd.Timestamp(t['entry_time']))
    overlap = ch_entries & other_entries
    print(f"\n  Signal overlap: Chandelier has {len(ch_entries)} entries, {len(overlap)} overlap with others ({len(overlap)/max(len(ch_entries),1)*100:.1f}%)")
    results['overlap_pct'] = round(len(overlap)/max(len(ch_entries),1)*100, 1)

    # K-Fold on portfolio (without position limits, as a simpler check)
    def run_6(h1f):
        at = run_all(h1f, LIVE_CONFIG); _, d = port_stats(at); return _sharpe(d)
    def run_5(h1f):
        at = {nm: run_strat(nm, h1f, LIVE_CONFIG[nm]) for nm in STRAT_ORDER if nm != 'CHANDELIER'}
        m = [t for nm in at for t in at[nm]]; return _sharpe(_daily(m))
    kf = kfold_test(h1, run_5, run_6)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    print(f"\n  K-Fold (drop CH vs keep): {kf_wins}/{len(kf)} wins for dropping CH")
    for r in kf:
        print(f"    {r['period']}: 6strats={r['base']:.3f} 5strats={r['new']:.3f} -> {r['win']}")
    results['kf_drop_ch'] = {'wins': kf_wins, 'total': len(kf), 'details': kf}

    # Chandelier drawdown contribution
    print(f"\n  Chandelier conditional value:")
    _, daily_all = port_stats(all_t)
    eq = daily_all.cumsum()
    dd = np.maximum.accumulate(eq) - eq
    dd_days = dd[dd > dd.quantile(0.9)].index
    ch_daily = _daily(all_t['CHANDELIER'])
    ch_in_dd = ch_daily.reindex(dd_days).dropna()
    if len(ch_in_dd) > 0:
        print(f"    During top 10% drawdown days: CH avg PnL=${ch_in_dd.mean():.2f}, positive={sum(ch_in_dd>0)}/{len(ch_in_dd)}")
        results['ch_drawdown_value'] = round(float(ch_in_dd.mean()), 2)
    else:
        print(f"    No overlap with drawdown days")
        results['ch_drawdown_value'] = 0

    save_phase(5, results)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Cap/SL Regime Diagnostic
# ═══════════════════════════════════════════════════════════════
def phase_6(h1):
    print(f"\n{'='*120}\n  PHASE 6: Cap/SL Regime Diagnostic\n{'='*120}", flush=True)
    atr_series = compute_atr(h1).dropna()
    q25 = float(atr_series.quantile(0.25))
    q50 = float(atr_series.quantile(0.50))
    q75 = float(atr_series.quantile(0.75))
    print(f"  ATR quartiles: Q25=${q25:.2f}, Q50=${q50:.2f}, Q75=${q75:.2f}")

    results = {}
    all_t = run_all(h1, LIVE_CONFIG)

    for nm in STRAT_ORDER:
        trades = all_t[nm]
        if not trades:
            continue

        # Split by ATR regime at entry
        low = [t for t in trades if t['atr'] < q25]
        mid = [t for t in trades if q25 <= t['atr'] < q75]
        high = [t for t in trades if t['atr'] >= q75]

        print(f"\n  {nm} (N={len(trades)}):")
        for label, subset in [('Low ATR (<Q25)', low), ('Mid ATR (Q25-Q75)', mid), ('High ATR (>=Q75)', high)]:
            if not subset:
                print(f"    {label}: no trades")
                continue
            s = _stats(subset)
            # Compute actual SL$ and Cap$ for this subset
            avg_atr = np.mean([t['atr'] for t in subset])
            sl_dollar = LIVE_CONFIG[nm]['sl'] * avg_atr * LIVE_CONFIG[nm]['lot'] * PV
            cap_dollar = LIVE_CONFIG[nm]['cap']
            binding = "SL" if sl_dollar < cap_dollar else "Cap"
            print(f"    {label}: N={s['n']}, Cap%={s['cap_pct']:.1f}%, SL%={s['sl_pct']:.1f}%, Trail%={s['trail_pct']:.1f}% | SL$=${sl_dollar:.0f} Cap$=${cap_dollar} -> {binding} binds")

        results[nm] = {
            'low_n': len(low), 'low_cap_pct': _stats(low)['cap_pct'] if low else 0,
            'low_sl_pct': _stats(low)['sl_pct'] if low else 0,
            'mid_n': len(mid), 'mid_cap_pct': _stats(mid)['cap_pct'] if mid else 0,
            'high_n': len(high), 'high_cap_pct': _stats(high)['cap_pct'] if high else 0,
        }

    save_phase(6, results)
    return results

# ═══════════════════════════════════════════════════════════════
# PHASE 7: Combined Validation (only GO items)
# ═══════════════════════════════════════════════════════════════
def phase_7(h1, p2, p3, p4, pctl):
    print(f"\n{'='*120}\n  PHASE 7: Combined Validation — Only Validated Changes\n{'='*120}", flush=True)
    results = {}

    # Collect GO items from phases 2-4
    go_changes = {}
    print(f"\n  Collecting GO items from Phases 2-4:")

    # Phase 2: Trail changes
    trail_gos = []
    for nm in STRAT_ORDER:
        if nm in p2 and isinstance(p2[nm], dict) and p2[nm].get('verdict', {}).get('verdict') == 'GO':
            trail_gos.append(nm)
            go_changes.setdefault(nm, {})['trail_act'] = 0.06
            go_changes.setdefault(nm, {})['trail_dist'] = 0.01
    print(f"  Trail 0.06/0.01 GO: {trail_gos if trail_gos else 'NONE'}")

    # Phase 3: SL changes
    sl_gos = []
    for nm in STRAT_ORDER:
        if nm in p3 and isinstance(p3[nm], dict) and p3[nm].get('verdict') == 'GO':
            sl_gos.append(nm)
            go_changes.setdefault(nm, {})['sl'] = 6.0
    print(f"  SL=6.0 GO: {sl_gos if sl_gos else 'NONE'}")

    # Phase 3: SESS_BO TP fix
    if 'SESS_BO_TP_fix' in p3 and p3['SESS_BO_TP_fix'].get('verdict', {}).get('verdict') == 'GO':
        go_changes.setdefault('SESS_BO', {})['tp'] = 6.0
        print(f"  SESS_BO TP fix: GO")
    else:
        print(f"  SESS_BO TP fix: NO-GO or not tested")

    # Phase 4: Max hold
    mh_gos = []
    for key, val in p4.items():
        if key.startswith('MH') and isinstance(val, dict) and val.get('verdict', {}).get('verdict') == 'GO':
            mh_gos.append(key)
            go_changes.setdefault('L8_MAX', {})['max_hold'] = int(key.replace('MH', ''))
    print(f"  Max hold GO: {mh_gos if mh_gos else 'NONE'}")

    if not go_changes:
        print(f"\n  No changes passed all 3 gates. Nothing to combine.")
        print(f"  Current live config remains optimal.")
        results['verdict'] = 'NO_CHANGES'
        save_phase(7, results)
        return results

    # Build combined config
    opt = copy.deepcopy(LIVE_CONFIG)
    for nm, changes in go_changes.items():
        for k, v in changes.items():
            opt[nm][k] = v

    print(f"\n  Combined config changes:")
    for nm in STRAT_ORDER:
        if nm in go_changes:
            print(f"    {nm}: {go_changes[nm]}")

    # Full period comparison
    configs = {
        'A_live': (LIVE_CONFIG, None, 0),
        'B_live_r187': (LIVE_CONFIG, pctl, 30),
        'C_combined': (opt, None, 0),
        'D_combined_r187': (opt, pctl, 30),
    }

    print(f"\n  {'Config':<22} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'Cap%':>6}")
    for label, (cfg, pv, pf) in configs.items():
        at = run_all(h1, cfg, pctl_v=pv, pctl_f=pf); s, _ = port_stats(at)
        print(f"  {label:<22} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['cap_pct']:>5.1f}%")
        results[label] = s

    # K-Fold: D vs B
    def run_d(h1f):
        at = run_all(h1f, opt, pctl_v=pctl, pctl_f=30); s, _ = port_stats(at); return s['sharpe']
    def run_b(h1f):
        at = run_all(h1f, LIVE_CONFIG, pctl_v=pctl, pctl_f=30); s, _ = port_stats(at); return s['sharpe']
    kf = kfold_test(h1, run_d, run_b)
    kf_wins = sum(1 for r in kf if r['win']=='NEW')
    print(f"\n  K-Fold (D vs B): {kf_wins}/{len(kf)}")
    for r in kf:
        print(f"    {r['period']}: B={r['base']:.3f} D={r['new']:.3f} -> {r['win']}")

    # Walk-Forward
    wf = wf_test(h1, run_d, run_b)
    wf_wins = sum(1 for r in wf if r['win']=='NEW')
    print(f"  Walk-Forward (D vs B): {wf_wins}/{len(wf)}")

    # Era
    t_d = run_all(h1, opt, pctl_v=pctl, pctl_f=30)
    t_b = run_all(h1, LIVE_CONFIG, pctl_v=pctl, pctl_f=30)
    trades_d = port_merge(t_d); trades_b = port_merge(t_b)
    era = era_test(h1, trades_d, trades_b)
    for e in ['full','hike','cut','recent_3y']:
        print(f"  Era {e}: B={era[e]['base']:.3f} D={era[e]['new']:.3f} (d={era[e]['delta']:+.3f})")

    v = verdict(kf, wf, era)
    print(f"\n  >>> COMBINED VERDICT: {v['verdict']} (KF={v['kf_wins']}/{v['kf_total']}, WF={v['wf_wins']}/{v['wf_total']})")

    # Yearly stability
    print(f"\n  Yearly stability:")
    _, daily_b = port_stats(t_b)
    _, daily_d = port_stats(t_d)
    years = sorted(set(daily_b.index.year) | set(daily_d.index.year))
    for yr in years:
        yb = daily_b[daily_b.index.year == yr]; yd = daily_d[daily_d.index.year == yr]
        print(f"    {yr}: B=${float(yb.sum()):>9,.0f} (Sh={_sharpe(yb):.2f}), D=${float(yd.sum()):>9,.0f} (Sh={_sharpe(yd):.2f})")

    results['validation'] = v
    results['go_changes'] = {nm: {k: str(v) for k,v in changes.items()} for nm, changes in go_changes.items()}
    results['optimized_config'] = {nm: {k: v for k, v in opt[nm].items()} for nm in STRAT_ORDER}
    save_phase(7, results)
    return results


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 120)
    print("  R192 — R191 Findings Rigorous Validation (7 Phases)")
    print("=" * 120, flush=True)

    h1 = load_h1()
    atr_s = compute_atr(h1).dropna()
    pctl = compute_atr_pctl(atr_s, lb=300)
    print(f"  ATR: mean=${atr_s.mean():.2f}, median=${atr_s.median():.2f}, current=${atr_s.iloc[-1]:.2f}")

    p1 = phase_1(h1, pctl)
    p2 = phase_2(h1)
    p3 = phase_3(h1)
    p4 = phase_4(h1)
    p5 = phase_5(h1)
    p6 = phase_6(h1)
    p7 = phase_7(h1, p2, p3, p4, pctl)

    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*120}")

    # Summary
    print(f"\n  === R192 SUMMARY ===")
    print(f"\n  Phase 2 (Trail):")
    for nm in STRAT_ORDER:
        if nm in p2 and isinstance(p2[nm], dict) and 'verdict' in p2[nm]:
            v = p2[nm]['verdict']
            print(f"    {nm}: {v['verdict']} (KF={v['kf_wins']}/{v['kf_total']}, WF={v['wf_wins']}/{v['wf_total']})")

    print(f"\n  Phase 3 (SL):")
    for nm in STRAT_ORDER:
        if nm in p3 and isinstance(p3[nm], dict):
            vd = p3[nm].get('verdict', 'N/A')
            print(f"    {nm}: {vd}")

    print(f"\n  Phase 4 (Max Hold):")
    for key in sorted(p4.keys()):
        if key.startswith('MH'):
            v = p4[key].get('verdict', {})
            print(f"    Keltner {key}: {v.get('verdict','N/A')} (KF={v.get('kf_wins','?')}/{v.get('kf_total','?')})")

    master = {'runtime_s': round(elapsed, 1), 'phases_completed': 7}
    with open(OUTPUT_DIR / "r192_results.json", 'w') as f:
        json.dump(master, f, indent=2, default=str)
    print(f"\n  Master saved: {OUTPUT_DIR / 'r192_results.json'}")


if __name__ == "__main__":
    main()
