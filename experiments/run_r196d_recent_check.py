#!/usr/bin/env python3
"""Quick check: old vs new trail params on recent 1-year data."""
import sys, os, warnings, copy
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')
import glob as _glob

PV = 100; SPREAD = 0.30
CFG_OLD = {'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}
CFG_NEW = {'lot': 0.04, 'cap': 70, 'sl': 6.0, 'tp': 8.0, 'trail_act': 0.02, 'trail_dist': 0.005, 'max_hold': 2}
SKIP_HOURS = {1, 20, 22, 23}

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

def compute_atr_pctl(atr_series, lb=300):
    n=len(atr_series); p=np.full(n,np.nan); v=atr_series.values
    for i in range(lb,n):
        w=v[i-lb:i]; valid=w[~np.isnan(w)]
        if len(valid)>=30: p[i]=np.sum(valid<=v[i])/len(valid)*100
    return pd.Series(p, index=atr_series.index)

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

def bt_keltner(h1, cfg, pctl_v, pctl_f=30, skip_hours=None):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
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
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<14: continue
        if skip_hours and hrs[i] in skip_hours: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

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

# Load data
candidates=sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
df=pd.read_csv(candidates[-1])
df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True)
df=df.set_index('timestamp')
df.index=df.index.tz_localize(None)
df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'},inplace=True)
h1=df[['Open','High','Low','Close']].copy()

# Full pctl on all data
df_temp = h1.copy(); df_temp['ATR'] = compute_atr(df_temp)
pctl_full = compute_atr_pctl(df_temp['ATR'], lb=300)

# Recent periods
from collections import Counter
periods = {
    'Last 12 months (2025-05 ~ 2026-05)': ('2025-05-01', '2026-12-31'),
    'Last 6 months (2025-11 ~ 2026-05)': ('2025-11-01', '2026-12-31'),
    'Last 3 months (2026-02 ~ 2026-05)': ('2026-02-01', '2026-12-31'),
    '2025 full year': ('2025-01-01', '2025-12-31'),
    '2024 full year': ('2024-01-01', '2024-12-31'),
    'High ATR era (2024-10 ~ 2026-05)': ('2024-10-01', '2026-12-31'),
}

print("=" * 100)
print("  Keltner Trail Parameter: Old (0.06/0.01) vs New (0.02/0.005) + Skip Hours")
print("=" * 100)

for label, (start, end) in periods.items():
    h1_p = h1[(h1.index >= start) & (h1.index <= end)]
    if len(h1_p) < 200:
        continue
    pctl_p = pctl_full.reindex(h1_p.index)
    
    t_old = bt_keltner(h1_p, CFG_OLD, pctl_p)
    t_new_trail = bt_keltner(h1_p, CFG_NEW, pctl_p)
    t_new_both = bt_keltner(h1_p, CFG_NEW, pctl_p, skip_hours=SKIP_HOURS)
    
    s_old = _stats(t_old)
    s_trail = _stats(t_new_trail)
    s_both = _stats(t_new_both)
    
    # Exit reason breakdown for old and new
    reasons_old = Counter(t['reason'] for t in t_old)
    reasons_new = Counter(t['reason'] for t in t_new_both)
    
    print(f"\n{'─'*80}")
    print(f"  {label} ({len(h1_p)} bars)")
    print(f"{'─'*80}")
    print(f"  {'Config':<30} {'Sharpe':>8} {'PnL':>10} {'WR':>7} {'MaxDD':>8} {'N':>6} {'Avg$/trade':>11}")
    print(f"  {'-'*82}")
    
    avg_old = s_old['pnl']/s_old['n'] if s_old['n']>0 else 0
    avg_trail = s_trail['pnl']/s_trail['n'] if s_trail['n']>0 else 0
    avg_both = s_both['pnl']/s_both['n'] if s_both['n']>0 else 0
    
    print(f"  {'Old (0.06/0.01)':<30} {s_old['sharpe']:>8.3f} {s_old['pnl']:>10.0f} {s_old['wr']:>6.1f}% {s_old['max_dd']:>8.0f} {s_old['n']:>6} {avg_old:>11.2f}")
    print(f"  {'New Trail only':<30} {s_trail['sharpe']:>8.3f} {s_trail['pnl']:>10.0f} {s_trail['wr']:>6.1f}% {s_trail['max_dd']:>8.0f} {s_trail['n']:>6} {avg_trail:>11.2f}")
    print(f"  {'New Trail + Skip Hours':<30} {s_both['sharpe']:>8.3f} {s_both['pnl']:>10.0f} {s_both['wr']:>6.1f}% {s_both['max_dd']:>8.0f} {s_both['n']:>6} {avg_both:>11.2f}")
    
    # Exit reasons
    print(f"\n  Exit reasons (Old -> New Trail+Skip):")
    for reason in ['Trail', 'Timeout', 'Cap', 'SL', 'TP']:
        n_old = reasons_old.get(reason, 0)
        n_new = reasons_new.get(reason, 0)
        pct_old = n_old/len(t_old)*100 if t_old else 0
        pct_new = n_new/len(t_new_both)*100 if t_new_both else 0
        avg_pnl_old = np.mean([t['pnl'] for t in t_old if t['reason']==reason]) if n_old>0 else 0
        avg_pnl_new = np.mean([t['pnl'] for t in t_new_both if t['reason']==reason]) if n_new>0 else 0
        print(f"    {reason:<8}: {n_old:>5} ({pct_old:>5.1f}%) avg ${avg_pnl_old:>7.2f}  ->  {n_new:>5} ({pct_new:>5.1f}%) avg ${avg_pnl_new:>7.2f}")

    # Monthly PnL for this period
    daily_old = _daily(t_old)
    daily_new = _daily(t_new_both)
    monthly_old = daily_old.resample('ME').sum()
    monthly_new = daily_new.resample('ME').sum()
    
    neg_old = (monthly_old < 0).sum()
    neg_new = (monthly_new < 0).sum()
    print(f"\n  Monthly: Old {neg_old}/{len(monthly_old)} negative months, New {neg_new}/{len(monthly_new)} negative months")

print(f"\n{'='*100}")
