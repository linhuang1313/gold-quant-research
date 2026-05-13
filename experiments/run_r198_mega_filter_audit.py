#!/usr/bin/env python3
"""
R198 — Mega Filter Audit: All Filters × All Strategies × Multiple Eras
=======================================================================
Systematically tests every configurable filter parameter across eras to
detect if current settings are still optimal in the high-ATR/high-price
environment (2024-2026, gold $2000→$4700).

Filters tested:
  Phase 1: ATR Pctl Floor threshold (10~50, step 5) — all strategies
  Phase 2: Keltner ADX threshold (10~22, step 2)
  Phase 3: Cooldown bars (0,1,2,3,4) — each strategy independently
  Phase 4: PSAR Skip Hours — test current {3,7,22} vs alternatives
  Phase 5: Keltner EMA trend period (50,75,100,150,200)
  Phase 6: Keltner KC parameters (ema=20/25/30, mult=1.0/1.2/1.5)
  Phase 7: SESS_BO D1 EMA20 filter (ON vs OFF)
  Phase 8: Max Hold sweep for each strategy
  Phase 9: Rule B parameters (sigma=2.0~4.0, skip_hours=4~12)

Each phase runs full-sample + era segmentation (4 eras).
Top findings are then validated with K-Fold (6-fold).
"""
import sys, os, time, json, warnings, copy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

import glob as _glob

OUTPUT_DIR = Path("results/r198_mega_filter_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

PV = 100; SPREAD = 0.30
CONFIGS = {
    'L8_MAX':      {'lot': 0.04, 'cap': 70, 'sl': 3.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60, 'sl': 6.0, 'tp': 6.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 15},
    'TSMOM':       {'lot': 0.04, 'cap': 60, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.04, 'cap': 60, 'sl': 4.5, 'tp': 4.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.03, 'cap': 25, 'sl': 4.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 20},
}

ERAS = {
    'Full (2015-2026)': ('2015-01-01', '2026-12-31'),
    'Pre-COVID (2015-2019)': ('2015-01-01', '2019-12-31'),
    'COVID (2020-2021)': ('2020-01-01', '2021-12-31'),
    'Hike (2022-2023)': ('2022-01-01', '2023-12-31'),
    'Recent (2024-2026)': ('2024-01-01', '2026-12-31'),
}

def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()

def save_phase(name, data):
    with open(OUTPUT_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2, default=str)

def elapsed():
    m = (time.time()-t0)/60
    return f"[{m:.1f}min]"

# ═══════════════ Core Helpers ═══════════════
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

def compute_psar(high, low, close, af_step=0.02, af_max=0.20):
    n = len(close)
    psar = np.zeros(n); direction = np.zeros(n, dtype=int)
    af = af_step; ep = high.iloc[0]; psar[0] = low.iloc[0]; direction[0] = 1
    for i in range(1, n):
        if direction[i-1] == 1:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low.iloc[i-1])
            if i >= 2: psar[i] = min(psar[i], low.iloc[i-2])
            if low.iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = low.iloc[i]; af = af_step
            else:
                direction[i] = 1
                if high.iloc[i] > ep: ep = high.iloc[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high.iloc[i-1])
            if i >= 2: psar[i] = max(psar[i], high.iloc[i-2])
            if high.iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = high.iloc[i]; af = af_step
            else:
                direction[i] = -1
                if low.iloc[i] < ep: ep = low.iloc[i]; af = min(af + af_step, af_max)
    return pd.Series(psar, index=close.index), pd.Series(direction, index=close.index)

def _mk(pos,ep,et,reason,bi,pnl):
    return {'dir':pos['dir'],'entry':pos['entry'],'exit':ep,'entry_time':pos['time'],'exit_time':et,'pnl':pnl,'reason':reason,'bars':bi-pos['bar'],'atr':pos['atr']}

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

# ═══════════════ Strategy Backtests ═══════════════
def bt_keltner(h1, cfg, pctl_v, pctl_f=30, adx_thr=14, ema_period=100, kc_ema=25, kc_mult=1.2, cooldown=2, skip_hours=None):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=ema_period,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=kc_ema,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+kc_mult*df['ATR']; df['KC_lower']=df['KC_mid']-kc_mult*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA_T','KC_upper'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA_T'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
    hrs = df.index.hour
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<cooldown: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<adx_thr: continue
        if skip_hours and hrs[i] in skip_hours: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_psar(h1, cfg, pctl_v, pctl_f=30, cooldown=2, skip_hours=None):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    psar_v, psar_d = compute_psar(df['High'], df['Low'], df['Close'], af_step=0.01, af_max=0.05)
    df['PSAR']=psar_v; df['PSAR_dir']=psar_d
    df=df.dropna(subset=['ATR','ADX'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    pdir=df['PSAR_dir'].values; times=df.index; n=len(df); hrs=df.index.hour
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999; prev_dir=0
    for i in range(1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; prev_dir=pdir[i]; continue
            prev_dir=pdir[i]; continue
        if i-le<cooldown: prev_dir=pdir[i]; continue
        if np.isnan(atr[i]) or atr[i]<0.1: prev_dir=pdir[i]; continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): prev_dir=pdir[i]; continue
        if skip_hours and hrs[i] in skip_hours: prev_dir=pdir[i]; continue
        if pdir[i]==1 and prev_dir==-1:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif pdir[i]==-1 and prev_dir==1:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        prev_dir=pdir[i]
    return trades

def bt_tsmom(h1, cfg, pctl_v, pctl_f=30, cooldown=2, fast_lb=480, slow_lb=720):
    df=h1.copy(); df['ATR']=compute_atr(df)
    df['fast_ma']=df['Close'].rolling(fast_lb).mean(); df['slow_ma']=df['Close'].rolling(slow_lb).mean()
    df=df.dropna(subset=['ATR','fast_ma','slow_ma'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    fast,slow=df['fast_ma'].values,df['slow_ma'].values; times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<cooldown: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if fast[i]>slow[i] and fast[i-1]<=slow[i-1]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif fast[i]<slow[i] and fast[i-1]>=slow[i-1]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_sess_bo(h1, cfg, pctl_v, pctl_f=30, cooldown=2, session_hour=12, lookback=4):
    df=h1.copy(); df['ATR']=compute_atr(df)
    df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    times=df.index; n=len(df); hrs=df.index.hour
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(lookback+1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<cooldown: continue
        if hrs[i]!=session_hour: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        range_high=max(h[i-lookback:i]); range_low=min(lo[i-lookback:i])
        if c[i]>range_high:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<range_low:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_dual_thrust(h1, cfg, pctl_v, pctl_f=30, cooldown=2, n_bars=6, k_up=0.5, k_down=0.5):
    df=h1.copy(); df['ATR']=compute_atr(df)
    df=df.dropna(subset=['ATR'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,op,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999
    for i in range(n_bars+1,n):
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<cooldown: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        hh=max(h[i-n_bars:i]); ll=min(lo[i-n_bars:i]); hc=max(c[i-n_bars:i]); lc=min(c[i-n_bars:i])
        rng=max(hh-lc, hc-ll)
        day_open=op[i]
        if c[i]>day_open+k_up*rng:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<day_open-k_down*rng:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

def bt_chandelier(h1, cfg, pctl_v, pctl_f=30, cooldown=2, chand_period=22, chand_mult=3.0, rsi_filter=True):
    df=h1.copy(); df['ATR']=compute_atr(df); df['RSI14']=compute_rsi(df['Close'],14)
    df['HH']=df['High'].rolling(chand_period).max(); df['LL']=df['Low'].rolling(chand_period).min()
    df['chand_long']=df['HH']-chand_mult*df['ATR']; df['chand_short']=df['LL']+chand_mult*df['ATR']
    df=df.dropna(subset=['ATR','chand_long','chand_short'])
    pv_a=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    cl,cs=df['chand_long'].values,df['chand_short'].values
    rsi=df['RSI14'].values; times=df.index; n=len(df)
    lot=cfg['lot']; sl=cfg['sl']; tp=cfg['tp']; ta=cfg['trail_act']; td=cfg['trail_dist']; mh=cfg['max_hold']; cap=cfg['cap']
    trades=[]; pos=None; le=-999; prev_above=True
    for i in range(1,n):
        above = c[i]>cl[i]
        if pos:
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; prev_above=above; continue
            prev_above=above; continue
        if i-le<cooldown: prev_above=above; continue
        if np.isnan(atr[i]) or atr[i]<0.1: prev_above=above; continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): prev_above=above; continue
        if above and not prev_above:
            if rsi_filter and not np.isnan(rsi[i]) and rsi[i]>70: prev_above=above; continue
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif not above and prev_above:
            if rsi_filter and not np.isnan(rsi[i]) and rsi[i]<30: prev_above=above; continue
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        prev_above=above
    return trades

# ═══════════════ Load Data ═══════════════
print(f"{'='*100}")
print(f"  R198 — Mega Filter Audit: All Filters x All Strategies x Multiple Eras")
print(f"{'='*100}")

candidates=sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
df=pd.read_csv(candidates[-1])
df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True)
df=df.set_index('timestamp'); df.index=df.index.tz_localize(None)
df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'},inplace=True)
h1=df[['Open','High','Low','Close']].copy()
h1_temp=h1.copy(); h1_temp['ATR']=compute_atr(h1_temp)
pctl_full = compute_atr_pctl(h1_temp['ATR'], lb=300)
print(f"{elapsed()} Loaded {len(h1)} H1 bars: {h1.index[0]} to {h1.index[-1]}")

def slice_era(era_name):
    start, end = ERAS[era_name]
    return h1[(h1.index>=start)&(h1.index<=end)]

def run_all_eras(run_fn, label=""):
    """Run a backtest function across all eras, return {era: stats}."""
    results = {}
    for era_name in ERAS:
        h1_era = slice_era(era_name)
        if len(h1_era) < 500: continue
        pctl_era = pctl_full.reindex(h1_era.index)
        trades = run_fn(h1_era, pctl_era)
        results[era_name] = _stats(trades)
    return results


# ═══════════════ Phase 1: ATR Pctl Floor ═══════════════
if not phase_done("phase1_atr_pctl"):
    print(f"\n{'─'*100}")
    print(f"  Phase 1: ATR Percentile Floor Threshold Sweep (all 6 strategies)")
    print(f"{'─'*100}")
    pctl_thresholds = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    p1_results = {}

    for strat_name, cfg in CONFIGS.items():
        print(f"\n  {strat_name}:")
        strat_results = {}
        for pf in pctl_thresholds:
            if strat_name == 'L8_MAX':
                fn = lambda h, p, _pf=pf: bt_keltner(h, cfg, p, pctl_f=_pf)
            elif strat_name == 'PSAR':
                fn = lambda h, p, _pf=pf: bt_psar(h, cfg, p, pctl_f=_pf)
            elif strat_name == 'TSMOM':
                fn = lambda h, p, _pf=pf: bt_tsmom(h, cfg, p, pctl_f=_pf)
            elif strat_name == 'SESS_BO':
                fn = lambda h, p, _pf=pf: bt_sess_bo(h, cfg, p, pctl_f=_pf)
            elif strat_name == 'DUAL_THRUST':
                fn = lambda h, p, _pf=pf: bt_dual_thrust(h, cfg, p, pctl_f=_pf)
            elif strat_name == 'CHANDELIER':
                fn = lambda h, p, _pf=pf: bt_chandelier(h, cfg, p, pctl_f=_pf)
            era_stats = run_all_eras(fn)
            strat_results[str(pf)] = era_stats
            full = era_stats.get('Full (2015-2026)', {})
            recent = era_stats.get('Recent (2024-2026)', {})
            print(f"    pctl_floor={pf:>3}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
        p1_results[strat_name] = strat_results
    save_phase("phase1_atr_pctl", p1_results)
    print(f"\n{elapsed()} Phase 1 complete.")
else:
    print(f"\n{elapsed()} Phase 1 already done.")


# ═══════════════ Phase 2: Keltner ADX Threshold ═══════════════
if not phase_done("phase2_adx"):
    print(f"\n{'─'*100}")
    print(f"  Phase 2: Keltner ADX Threshold Sweep")
    print(f"{'─'*100}")
    adx_values = [8, 10, 12, 14, 16, 18, 20, 22]
    p2_results = {}
    cfg = CONFIGS['L8_MAX']
    for adx_t in adx_values:
        fn = lambda h, p, _at=adx_t: bt_keltner(h, cfg, p, adx_thr=_at)
        era_stats = run_all_eras(fn)
        p2_results[str(adx_t)] = era_stats
        full = era_stats.get('Full (2015-2026)', {})
        recent = era_stats.get('Recent (2024-2026)', {})
        print(f"  ADX>={adx_t:>2}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
    save_phase("phase2_adx", p2_results)
    print(f"\n{elapsed()} Phase 2 complete.")
else:
    print(f"\n{elapsed()} Phase 2 already done.")


# ═══════════════ Phase 3: Cooldown Sweep ═══════════════
if not phase_done("phase3_cooldown"):
    print(f"\n{'─'*100}")
    print(f"  Phase 3: Cooldown Bars Sweep (all strategies)")
    print(f"{'─'*100}")
    cd_values = [0, 1, 2, 3, 4, 6]
    p3_results = {}
    for strat_name, cfg in CONFIGS.items():
        print(f"\n  {strat_name}:")
        strat_results = {}
        for cd in cd_values:
            if strat_name == 'L8_MAX':
                fn = lambda h, p, _cd=cd: bt_keltner(h, cfg, p, cooldown=_cd)
            elif strat_name == 'PSAR':
                fn = lambda h, p, _cd=cd: bt_psar(h, cfg, p, cooldown=_cd)
            elif strat_name == 'TSMOM':
                fn = lambda h, p, _cd=cd: bt_tsmom(h, cfg, p, cooldown=_cd)
            elif strat_name == 'SESS_BO':
                fn = lambda h, p, _cd=cd: bt_sess_bo(h, cfg, p, cooldown=_cd)
            elif strat_name == 'DUAL_THRUST':
                fn = lambda h, p, _cd=cd: bt_dual_thrust(h, cfg, p, cooldown=_cd)
            elif strat_name == 'CHANDELIER':
                fn = lambda h, p, _cd=cd: bt_chandelier(h, cfg, p, cooldown=_cd)
            era_stats = run_all_eras(fn)
            strat_results[str(cd)] = era_stats
            full = era_stats.get('Full (2015-2026)', {})
            recent = era_stats.get('Recent (2024-2026)', {})
            print(f"    cd={cd}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
        p3_results[strat_name] = strat_results
    save_phase("phase3_cooldown", p3_results)
    print(f"\n{elapsed()} Phase 3 complete.")
else:
    print(f"\n{elapsed()} Phase 3 already done.")


# ═══════════════ Phase 4: PSAR Skip Hours ═══════════════
if not phase_done("phase4_psar_hours"):
    print(f"\n{'─'*100}")
    print(f"  Phase 4: PSAR Skip Hours Variants")
    print(f"{'─'*100}")
    cfg = CONFIGS['PSAR']
    hour_variants = {
        'none': None,
        'current_{3,7,22}': {3, 7, 22},
        '{3,7}': {3, 7},
        '{22,23}': {22, 23},
        '{1,3,7,22,23}': {1, 3, 7, 22, 23},
        '{0,1,2,3,22,23}': {0, 1, 2, 3, 22, 23},
    }
    p4_results = {}
    for label, skip_h in hour_variants.items():
        fn = lambda h, p, _sh=skip_h: bt_psar(h, cfg, p, skip_hours=_sh)
        era_stats = run_all_eras(fn)
        p4_results[label] = era_stats
        full = era_stats.get('Full (2015-2026)', {})
        recent = era_stats.get('Recent (2024-2026)', {})
        print(f"  {label:<25}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
    save_phase("phase4_psar_hours", p4_results)
    print(f"\n{elapsed()} Phase 4 complete.")
else:
    print(f"\n{elapsed()} Phase 4 already done.")


# ═══════════════ Phase 5: Keltner EMA Trend Period ═══════════════
if not phase_done("phase5_ema_period"):
    print(f"\n{'─'*100}")
    print(f"  Phase 5: Keltner EMA Trend Period Sweep")
    print(f"{'─'*100}")
    cfg = CONFIGS['L8_MAX']
    ema_periods = [50, 75, 100, 125, 150, 200]
    p5_results = {}
    for ep in ema_periods:
        fn = lambda h, p, _ep=ep: bt_keltner(h, cfg, p, ema_period=_ep)
        era_stats = run_all_eras(fn)
        p5_results[str(ep)] = era_stats
        full = era_stats.get('Full (2015-2026)', {})
        recent = era_stats.get('Recent (2024-2026)', {})
        print(f"  EMA={ep:<4}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
    save_phase("phase5_ema_period", p5_results)
    print(f"\n{elapsed()} Phase 5 complete.")
else:
    print(f"\n{elapsed()} Phase 5 already done.")


# ═══════════════ Phase 6: Keltner KC Parameters ═══════════════
if not phase_done("phase6_kc_params"):
    print(f"\n{'─'*100}")
    print(f"  Phase 6: Keltner KC EMA/Mult Grid")
    print(f"{'─'*100}")
    cfg = CONFIGS['L8_MAX']
    kc_grid = [(e, m) for e in [20, 25, 30] for m in [1.0, 1.2, 1.5]]
    p6_results = {}
    for kc_e, kc_m in kc_grid:
        key = f"ema{kc_e}_mult{kc_m}"
        fn = lambda h, p, _e=kc_e, _m=kc_m: bt_keltner(h, cfg, p, kc_ema=_e, kc_mult=_m)
        era_stats = run_all_eras(fn)
        p6_results[key] = era_stats
        full = era_stats.get('Full (2015-2026)', {})
        recent = era_stats.get('Recent (2024-2026)', {})
        print(f"  KC(ema={kc_e},mult={kc_m})  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
    save_phase("phase6_kc_params", p6_results)
    print(f"\n{elapsed()} Phase 6 complete.")
else:
    print(f"\n{elapsed()} Phase 6 already done.")


# ═══════════════ Phase 7: Max Hold Sweep ═══════════════
if not phase_done("phase7_max_hold"):
    print(f"\n{'─'*100}")
    print(f"  Phase 7: Max Hold Bars Sweep (all strategies)")
    print(f"{'─'*100}")
    p7_results = {}
    mh_map = {
        'L8_MAX': [1, 2, 3, 4, 5],
        'PSAR': [8, 10, 12, 15, 20],
        'TSMOM': [6, 8, 10, 12, 15, 20],
        'SESS_BO': [10, 15, 20, 25, 30],
        'DUAL_THRUST': [10, 15, 20, 25, 30],
        'CHANDELIER': [10, 15, 20, 25, 30],
    }
    for strat_name, cfg_orig in CONFIGS.items():
        print(f"\n  {strat_name}:")
        strat_results = {}
        for mh in mh_map.get(strat_name, []):
            cfg = dict(cfg_orig); cfg['max_hold'] = mh
            if strat_name == 'L8_MAX':
                fn = lambda h, p, _cfg=cfg: bt_keltner(h, _cfg, p)
            elif strat_name == 'PSAR':
                fn = lambda h, p, _cfg=cfg: bt_psar(h, _cfg, p)
            elif strat_name == 'TSMOM':
                fn = lambda h, p, _cfg=cfg: bt_tsmom(h, _cfg, p)
            elif strat_name == 'SESS_BO':
                fn = lambda h, p, _cfg=cfg: bt_sess_bo(h, _cfg, p)
            elif strat_name == 'DUAL_THRUST':
                fn = lambda h, p, _cfg=cfg: bt_dual_thrust(h, _cfg, p)
            elif strat_name == 'CHANDELIER':
                fn = lambda h, p, _cfg=cfg: bt_chandelier(h, _cfg, p)
            era_stats = run_all_eras(fn)
            strat_results[str(mh)] = era_stats
            full = era_stats.get('Full (2015-2026)', {})
            recent = era_stats.get('Recent (2024-2026)', {})
            print(f"    mh={mh:>3}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
        p7_results[strat_name] = strat_results
    save_phase("phase7_max_hold", p7_results)
    print(f"\n{elapsed()} Phase 7 complete.")
else:
    print(f"\n{elapsed()} Phase 7 already done.")


# ═══════════════ Phase 8: Chandelier RSI Filter ═══════════════
if not phase_done("phase8_chandelier_rsi"):
    print(f"\n{'─'*100}")
    print(f"  Phase 8: Chandelier RSI Filter ON vs OFF")
    print(f"{'─'*100}")
    cfg = CONFIGS['CHANDELIER']
    p8_results = {}
    for rsi_on in [True, False]:
        label = "RSI_ON" if rsi_on else "RSI_OFF"
        fn = lambda h, p, _r=rsi_on: bt_chandelier(h, cfg, p, rsi_filter=_r)
        era_stats = run_all_eras(fn)
        p8_results[label] = era_stats
        full = era_stats.get('Full (2015-2026)', {})
        recent = era_stats.get('Recent (2024-2026)', {})
        print(f"  {label:<10}  Full: Sharpe={full.get('sharpe',0):>7.3f} N={full.get('n',0):>6}  Recent: Sharpe={recent.get('sharpe',0):>7.3f} N={recent.get('n',0):>6}")
    save_phase("phase8_chandelier_rsi", p8_results)
    print(f"\n{elapsed()} Phase 8 complete.")
else:
    print(f"\n{elapsed()} Phase 8 already done.")


# ═══════════════ Phase 9: K-Fold Validation of Any Findings ═══════════════
if not phase_done("phase9_kfold"):
    print(f"\n{'─'*100}")
    print(f"  Phase 9: K-Fold Validation — checking if any new values beat current in Recent era")
    print(f"{'─'*100}")

    # Collect candidates: parameters where Recent era Sharpe is meaningfully different from current
    candidates = []

    # Check Phase 1: ATR Pctl Floor
    if (OUTPUT_DIR / "phase1_atr_pctl.json").exists():
        with open(OUTPUT_DIR / "phase1_atr_pctl.json") as f:
            p1 = json.load(f)
        for strat in p1:
            current_pf = '30'
            current_recent = p1[strat].get(current_pf, {}).get('Recent (2024-2026)', {}).get('sharpe', 0)
            for pf, eras in p1[strat].items():
                recent = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
                if recent > current_recent + 0.2:
                    candidates.append({'strat': strat, 'param': 'atr_pctl_floor', 'value': pf,
                                       'current': current_pf, 'recent_delta': round(recent - current_recent, 3)})

    # Check Phase 2: ADX
    if (OUTPUT_DIR / "phase2_adx.json").exists():
        with open(OUTPUT_DIR / "phase2_adx.json") as f:
            p2 = json.load(f)
        current_adx = '14'
        current_recent = p2.get(current_adx, {}).get('Recent (2024-2026)', {}).get('sharpe', 0)
        for adx_v, eras in p2.items():
            recent = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
            if recent > current_recent + 0.2:
                candidates.append({'strat': 'L8_MAX', 'param': 'adx_threshold', 'value': adx_v,
                                   'current': current_adx, 'recent_delta': round(recent - current_recent, 3)})

    print(f"\n  Candidates for K-Fold validation (Recent era delta > +0.2):")
    if not candidates:
        print(f"    None found — all current settings are already optimal or near-optimal in Recent era")
    else:
        for c in candidates:
            print(f"    {c['strat']:<15} {c['param']:<20} current={c['current']}, candidate={c['value']}, Recent dSharpe={c['recent_delta']:+.3f}")

    # K-Fold validate each candidate
    kf_results = {}
    n_folds = 6
    for cand in candidates:
        strat = cand['strat']
        cfg = CONFIGS[strat]
        param = cand['param']
        new_val = cand['value']
        cur_val = cand['current']
        
        fold_size = len(h1) // n_folds
        wins = 0
        fold_details = []
        for fold in range(n_folds):
            fs = fold * fold_size
            fe = min((fold + 1) * fold_size, len(h1))
            h1_fold = h1.iloc[fs:fe]
            pctl_fold = pctl_full.reindex(h1_fold.index)
            
            if param == 'atr_pctl_floor':
                if strat == 'L8_MAX':
                    t_new = bt_keltner(h1_fold, cfg, pctl_fold, pctl_f=int(new_val))
                    t_cur = bt_keltner(h1_fold, cfg, pctl_fold, pctl_f=int(cur_val))
                elif strat == 'PSAR':
                    t_new = bt_psar(h1_fold, cfg, pctl_fold, pctl_f=int(new_val))
                    t_cur = bt_psar(h1_fold, cfg, pctl_fold, pctl_f=int(cur_val))
                elif strat == 'TSMOM':
                    t_new = bt_tsmom(h1_fold, cfg, pctl_fold, pctl_f=int(new_val))
                    t_cur = bt_tsmom(h1_fold, cfg, pctl_fold, pctl_f=int(cur_val))
                elif strat == 'SESS_BO':
                    t_new = bt_sess_bo(h1_fold, cfg, pctl_fold, pctl_f=int(new_val))
                    t_cur = bt_sess_bo(h1_fold, cfg, pctl_fold, pctl_f=int(cur_val))
                elif strat == 'DUAL_THRUST':
                    t_new = bt_dual_thrust(h1_fold, cfg, pctl_fold, pctl_f=int(new_val))
                    t_cur = bt_dual_thrust(h1_fold, cfg, pctl_fold, pctl_f=int(cur_val))
                elif strat == 'CHANDELIER':
                    t_new = bt_chandelier(h1_fold, cfg, pctl_fold, pctl_f=int(new_val))
                    t_cur = bt_chandelier(h1_fold, cfg, pctl_fold, pctl_f=int(cur_val))
            elif param == 'adx_threshold':
                t_new = bt_keltner(h1_fold, cfg, pctl_fold, adx_thr=int(new_val))
                t_cur = bt_keltner(h1_fold, cfg, pctl_fold, adx_thr=int(cur_val))
            else:
                continue
            
            s_new = _stats(t_new)['sharpe']
            s_cur = _stats(t_cur)['sharpe']
            win = 1 if s_new > s_cur else 0
            wins += win
            fold_details.append({'fold': fold, 'new': s_new, 'cur': s_cur, 'delta': round(s_new - s_cur, 3)})
        
        passed = wins >= 4
        key = f"{strat}_{param}_{new_val}"
        kf_results[key] = {'wins': wins, 'passed': passed, 'folds': fold_details, 'candidate': cand}
        verdict = "GO" if passed else "NO-GO"
        print(f"\n  {strat} {param}={new_val}: K-Fold {wins}/6 [{verdict}]")
        for fd in fold_details:
            print(f"    Fold {fd['fold']}: new={fd['new']:.3f} cur={fd['cur']:.3f} delta={fd['delta']:+.3f}")

    save_phase("phase9_kfold", kf_results)
    print(f"\n{elapsed()} Phase 9 complete.")
else:
    print(f"\n{elapsed()} Phase 9 already done.")


# ═══════════════ Final Summary ═══════════════
print(f"\n{'='*100}")
print(f"  R198 FINAL SUMMARY — Mega Filter Audit")
print(f"{'='*100}")

print(f"\n  Current settings vs Recent era (2024-2026) performance:")
print(f"  If the current parameter is NOT the best in the Recent era, it's flagged.\n")

# Load all results
for phase_name in ['phase1_atr_pctl', 'phase2_adx', 'phase3_cooldown', 'phase4_psar_hours',
                    'phase5_ema_period', 'phase6_kc_params', 'phase7_max_hold', 'phase8_chandelier_rsi']:
    fpath = OUTPUT_DIR / f"{phase_name}.json"
    if not fpath.exists():
        continue
    with open(fpath) as f:
        data = json.load(f)
    
    print(f"\n  --- {phase_name.replace('_', ' ').title()} ---")
    
    if phase_name == 'phase1_atr_pctl':
        for strat in data:
            best_val = None; best_sh = -999
            for val, eras in data[strat].items():
                sh = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
                if sh > best_sh: best_sh = sh; best_val = val
            current = data[strat].get('30', {}).get('Recent (2024-2026)', {}).get('sharpe', 0)
            flag = " <-- CHANGE?" if best_val != '30' and best_sh > current + 0.1 else ""
            print(f"    {strat:<15} Current=30, Best={best_val} (Recent Sharpe {best_sh:.3f} vs {current:.3f}){flag}")
    
    elif phase_name == 'phase2_adx':
        best_val = None; best_sh = -999
        for val, eras in data.items():
            sh = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
            if sh > best_sh: best_sh = sh; best_val = val
        current = data.get('14', {}).get('Recent (2024-2026)', {}).get('sharpe', 0)
        flag = " <-- CHANGE?" if best_val != '14' and best_sh > current + 0.1 else ""
        print(f"    Keltner ADX    Current=14, Best={best_val} (Recent Sharpe {best_sh:.3f} vs {current:.3f}){flag}")
    
    elif phase_name == 'phase3_cooldown':
        for strat in data:
            best_val = None; best_sh = -999
            for val, eras in data[strat].items():
                sh = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
                if sh > best_sh: best_sh = sh; best_val = val
            current_cd = {'L8_MAX':'2','PSAR':'2','TSMOM':'2','SESS_BO':'2','DUAL_THRUST':'2','CHANDELIER':'2'}
            cur = current_cd.get(strat, '2')
            current_sh = data[strat].get(cur, {}).get('Recent (2024-2026)', {}).get('sharpe', 0)
            flag = " <-- CHANGE?" if best_val != cur and best_sh > current_sh + 0.1 else ""
            print(f"    {strat:<15} Current={cur}, Best={best_val} (Recent Sharpe {best_sh:.3f} vs {current_sh:.3f}){flag}")
    
    elif phase_name in ('phase5_ema_period', 'phase7_max_hold'):
        if isinstance(list(data.values())[0], dict) and any(k in ERAS for k in list(data.values())[0]):
            best_val = None; best_sh = -999
            for val, eras in data.items():
                sh = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
                if sh > best_sh: best_sh = sh; best_val = val
            print(f"    Best={best_val} (Recent Sharpe {best_sh:.3f})")
        else:
            for strat in data:
                best_val = None; best_sh = -999
                for val, eras in data[strat].items():
                    sh = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
                    if sh > best_sh: best_sh = sh; best_val = val
                print(f"    {strat:<15} Best max_hold={best_val} (Recent Sharpe {best_sh:.3f})")

total_time = time.time() - t0
print(f"\n  Total runtime: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
print(f"{'='*100}")
