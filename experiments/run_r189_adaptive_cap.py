#!/usr/bin/env python3
"""
R189 — Adaptive Cap Validation + Lot Recalibration
====================================================
R188 discovered ALL 6 strategies have Cap << SL at current ATR≈20.
Cap was calibrated in ATR=2-5 era; now ATR is 4x higher but Cap unchanged.

Solution: Adaptive Cap = cap_atr_mult × ATR × lot × PV
This makes Cap scale with ATR, maintaining consistent Cap/SL ratio.

Phases:
  1:  Baseline with current fixed Caps
  2:  Adaptive Cap sweep: cap_atr_mult = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
      (a) with current lots
      (b) with proposed new lots
  3:  Best adaptive Cap — per-strategy detail
  4:  K-Fold 6-fold validation (best config)
  5:  Walk-Forward OOS validation (best config)
  6:  Era segmented validation
  7:  Yearly stability
  8:  Risk metrics: max single-trade loss, max concurrent exposure, drawdown
  9:  Combined with R187 ATR Pctl Floor (all-strategy, lb=300, pctl=30)
  10: Final scorecard
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r189_adaptive_cap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

CURRENT_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap': 35,  'sl': 3.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60,  'sl': 4.0, 'tp': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15},
    'TSMOM':       {'lot': 0.15, 'cap': 60,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.13, 'cap': 60,  'sl': 4.5, 'tp': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.08, 'cap': 25,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
}

NEW_LOTS = {
    'L8_MAX':      0.02,
    'PSAR':        0.04,
    'TSMOM':       0.04,
    'SESS_BO':     0.04,
    'DUAL_THRUST': 0.02,
    'CHANDELIER':  0.03,
}

STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

ERA_SEGMENTS = {
    'full': None,
    'hike': [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':  [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

CAP_ATR_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

import glob as _glob
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def compute_atr(df, period=14):
    tr = pd.DataFrame({'hl': df['High']-df['Low'],
        'hc': (df['High']-df['Close'].shift(1)).abs(),
        'lc': (df['Low']-df['Close'].shift(1)).abs()}).max(axis=1)
    return tr.rolling(period).mean()

def compute_adx(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pdm = h.diff(); mdm = -l.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0.0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0.0)
    tr = pd.DataFrame({'hl': h-l, 'hc': (h-c.shift(1)).abs(), 'lc': (l-c.shift(1)).abs()}).max(axis=1)
    atr_s = tr.rolling(period).mean()
    pdi = 100*(pdm.rolling(period).mean()/atr_s)
    mdi = 100*(mdm.rolling(period).mean()/atr_s)
    dx = 100*((pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan))
    return dx.rolling(period).mean()

def compute_atr_pctl(atr_series, lb=300):
    n = len(atr_series); p = np.full(n, np.nan); v = atr_series.values
    for i in range(lb, n):
        w = v[i-lb:i]; valid = w[~np.isnan(w)]
        if len(valid) >= 30: p[i] = np.sum(valid <= v[i]) / len(valid) * 100
    return pd.Series(p, index=atr_series.index)

def _mk(pos, ep, et, reason, bi, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': ep,
            'entry_time': pos['time'], 'exit_time': et,
            'pnl': pnl, 'reason': reason, 'bars': bi - pos['bar'],
            'atr': pos['atr'], 'strategy': pos.get('strategy', '')}

def _run_exit(pos, i, h, lo, c, spread, lot, pv, times,
              sl_atr, tp_atr, ta, td, mh, cap):
    """cap can be fixed $ or will be computed adaptively by caller."""
    if pos['dir'] == 'BUY':
        pnl_c = (c-pos['entry']-spread)*lot*pv
        pnl_h = (h-pos['entry']-spread)*lot*pv
        pnl_l = (lo-pos['entry']-spread)*lot*pv
    else:
        pnl_c = (pos['entry']-c-spread)*lot*pv
        pnl_h = (pos['entry']-lo-spread)*lot*pv
        pnl_l = (pos['entry']-h-spread)*lot*pv
    tp_v = tp_atr*pos['atr']*lot*pv; sl_v = sl_atr*pos['atr']*lot*pv
    if pnl_h >= tp_v: return _mk(pos, c, times[i], "TP", i, tp_v)
    if pnl_l <= -sl_v: return _mk(pos, c, times[i], "SL", i, -sl_v)
    if cap > 0 and pnl_c < -cap: return _mk(pos, c, times[i], "Cap", i, -cap)
    ad = ta*pos['atr']; tdd = td*pos['atr']
    if pos['dir'] == 'BUY' and h-pos['entry'] >= ad:
        ts = h - tdd
        if lo <= ts: return _mk(pos, c, times[i], "Trail", i, (ts-pos['entry']-spread)*lot*pv)
    elif pos['dir'] == 'SELL' and pos['entry']-lo >= ad:
        ts = lo + tdd
        if h >= ts: return _mk(pos, c, times[i], "Trail", i, (pos['entry']-ts-spread)*lot*pv)
    if held := (i - pos['bar']):
        if held >= mh: return _mk(pos, c, times[i], "Timeout", i, pnl_c)
    return None

def _daily(trades):
    if not trades: return pd.Series(dtype=float)
    d = {}
    for t in trades:
        k = pd.Timestamp(t['exit_time']).normalize()
        d[k] = d.get(k, 0) + t['pnl']
    return pd.Series(d).sort_index()

def _sharpe(daily):
    if len(daily) < 10 or daily.std() == 0: return 0.0
    return float(daily.mean()/daily.std()*np.sqrt(252))

def _stats(trades):
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0,'cap_pct':0,'sl_pct':0,'tp_pct':0,'max_loss':0}
    daily = _daily(trades); pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]
    eq = daily.cumsum(); dd = float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    reasons = [t['reason'] for t in trades]
    cap_exits = sum(1 for r in reasons if 'Cap' in r)
    return {'n':n, 'sharpe':round(_sharpe(daily),3), 'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1), 'max_dd':round(dd,2),
            'cap_pct': round(cap_exits/n*100,1),
            'sl_pct': round(sum(1 for r in reasons if r=='SL')/n*100,1),
            'tp_pct': round(sum(1 for r in reasons if r=='TP')/n*100,1),
            'max_loss': round(min(pnls),2) if pnls else 0}

def filter_era(trades, era):
    if era == 'full' or ERA_SEGMENTS[era] is None: return trades
    return [t for t in trades if any(pd.Timestamp(s) <= pd.Timestamp(t['entry_time']) < pd.Timestamp(e) for s,e in ERA_SEGMENTS[era])]

def load_h1():
    candidates = sorted(_glob.glob("data/download/xauusd-h1-bid-2015-*.csv"))
    if not candidates: raise FileNotFoundError("No H1 data")
    df = pd.read_csv(candidates[-1])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
    df = df[['Open','High','Low','Close']].copy()
    print(f"  {len(df)} bars ({df.index[0]} ~ {df.index[-1]})", flush=True)
    return df

# ═══════════════════════════════════════════════════════════════
# Strategy backtests — adaptive cap version
# cap_atr_mult=0 → use fixed cap from config; >0 → cap = cap_atr_mult × ATR × lot × PV
# ═══════════════════════════════════════════════════════════════
def _get_cap(fixed_cap, cap_atr_mult, atr, lot):
    if cap_atr_mult > 0:
        return cap_atr_mult * atr * lot * PV
    return fixed_cap

def bt_keltner(h1, lot, fixed_cap, sl, tp, ta, td, mh, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    df = h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA100']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA100','KC_upper'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA100'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
    trades=[]; pos=None; le=-999
    for i in range(1,n):
        if pos:
            cap = _get_cap(fixed_cap, cap_atr_mult, pos['atr'], lot)
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

def bt_psar(h1, lot, fixed_cap, sl, tp, ta, td, mh, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    df = h1.copy()
    ha,la,ca = df['High'].values,df['Low'].values,df['Close'].values; n=len(df)
    psar=np.empty(n); psar[:]=np.nan; af_s=0.01; af_m=0.05
    af=af_s; rising=True; ep=ha[0]; psar[0]=la[0]
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
    times=df.index; n2=len(df); trades=[]; pos=None; le=-999; prev=c[0]>ps[0]
    for i in range(1,n2):
        cur=c[i]>ps[i]
        if pos:
            cap = _get_cap(fixed_cap, cap_atr_mult, pos['atr'], lot)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; prev=cur; continue
            prev=cur; continue
        if i-le<2: prev=cur; continue
        if np.isnan(atr[i]) or atr[i]<0.1: prev=cur; continue
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): prev=cur; continue
        if cur and not prev:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        elif not cur and prev:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'PSAR'}
        prev=cur
    return trades

def bt_tsmom(h1, lot, fixed_cap, sl, tp, ta, td, mh, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    times=df.index; n=len(df); f,s=480,720; mx=max(f,s)
    sc=np.full(n,np.nan)
    for i in range(mx,n):
        v=0.0
        if c[i-f]>0: v+=0.5*np.sign(c[i]/c[i-f]-1.0)
        if c[i-s]>0: v+=0.5*np.sign(c[i]/c[i-s]-1.0)
        sc[i]=v
    trades=[]; pos=None; le=-999
    for i in range(mx+1,n):
        if pos:
            cap = _get_cap(fixed_cap, cap_atr_mult, pos['atr'], lot)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            if pos['dir']=='BUY' and sc[i]<0:
                trades.append(_mk(pos,c[i],times[i],"Rev",i,(c[i]-pos['entry']-SPREAD)*lot*PV)); pos=None; le=i; continue
            elif pos['dir']=='SELL' and sc[i]>0:
                trades.append(_mk(pos,c[i],times[i],"Rev",i,(pos['entry']-c[i]-SPREAD)*lot*PV)); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): continue
        if np.isnan(sc[i]) or np.isnan(sc[i-1]): continue
        if sc[i]>0 and sc[i-1]<=0:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'TSMOM'}
        elif sc[i]<0 and sc[i-1]>=0:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'TSMOM'}
    return trades

def bt_sess_bo(h1, lot, fixed_cap, sl, tp, ta, td, mh, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    hrs=df.index.hour; times=df.index; n=len(df); lb=4
    trades=[]; pos=None; le=-999
    for i in range(lb,n):
        if pos:
            cap = _get_cap(fixed_cap, cap_atr_mult, pos['atr'], lot)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if hrs[i]!=12: continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): continue
        hh=max(h[i-j] for j in range(1,lb+1)); ll=min(lo[i-j] for j in range(1,lb+1))
        if c[i]>hh: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
        elif c[i]<ll: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'SESS_BO'}
    return trades

def bt_dt(h1, lot, fixed_cap, sl, tp, ta, td, mh, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,o,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df); nb=6; k=0.5
    trades=[]; pos=None; le=-999
    for i in range(nb,n):
        if pos:
            cap = _get_cap(fixed_cap, cap_atr_mult, pos['atr'], lot)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): continue
        hh=np.max(h[i-nb:i]); lc=np.min(c[i-nb:i]); hc=np.max(c[i-nb:i]); ll=np.min(lo[i-nb:i])
        rng=max(hh-lc,hc-ll)
        if c[i]>o[i]+k*rng: pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'DUAL_THRUST'}
        elif c[i]<o[i]-k*rng: pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'DUAL_THRUST'}
    return trades

def bt_chand(h1, lot, fixed_cap, sl, tp, ta, td, mh, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df,period=22)
    df['EMA']=df['Close'].ewm(span=100,adjust=False).mean()
    df=df.dropna(subset=['ATR','EMA'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['EMA'].values
    times=df.index; n=len(df); p=22; m=3.0
    cl=np.full(n,np.nan); cs=np.full(n,np.nan)
    for i in range(p,n): cl[i]=np.max(h[i-p+1:i+1])-m*atr[i]; cs[i]=np.min(lo[i-p+1:i+1])+m*atr[i]
    d=np.zeros(n)
    for i in range(p+1,n):
        if np.isnan(cl[i]) or np.isnan(cs[i]): d[i]=d[i-1]; continue
        if c[i]>cs[i-1]: d[i]=1
        elif c[i]<cl[i-1]: d[i]=-1
        else: d[i]=d[i-1]
    trades=[]; pos=None; le=-999
    for i in range(p+2,n):
        if pos:
            cap = _get_cap(fixed_cap, cap_atr_mult, pos['atr'], lot)
            r=_run_exit(pos,i,h[i],lo[i],c[i],SPREAD,lot,PV,times,sl,tp,ta,td,mh,cap)
            if r: trades.append(r); pos=None; le=i; continue
            continue
        if i-le<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv is not None and (np.isnan(pv[i]) or pv[i]<pctl_f): continue
        if d[i]==1 and d[i-1]!=1 and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'CHANDELIER'}
        elif d[i]==-1 and d[i-1]!=-1 and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i],'strategy':'CHANDELIER'}
    return trades

BT = {'L8_MAX':bt_keltner,'PSAR':bt_psar,'TSMOM':bt_tsmom,
      'SESS_BO':bt_sess_bo,'DUAL_THRUST':bt_dt,'CHANDELIER':bt_chand}

def run_all(h1, lots=None, cap_atr_mult=0, pctl_v=None, pctl_f=0):
    """Run all strategies. lots=dict overrides lot per strategy. cap_atr_mult=0 uses fixed cap."""
    r = {}
    for name in STRAT_ORDER:
        cfg = CURRENT_CONFIG[name]
        lot = (lots or {}). get(name, cfg['lot'])
        r[name] = BT[name](h1, lot, cfg['cap'], cfg['sl'], cfg['tp'],
                           cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'],
                           cap_atr_mult=cap_atr_mult, pctl_v=pctl_v, pctl_f=pctl_f)
    return r

def port_stats(all_t):
    merged = [t for nm in STRAT_ORDER for t in all_t[nm]]
    return _stats(merged), _daily(merged)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Baseline
# ═══════════════════════════════════════════════════════════════
def phase1(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 1: Baseline — Current Fixed Caps + Current Lots")
    print(f"{'='*120}")

    all_t = run_all(h1)
    print(f"\n  {'Strategy':<15} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'Cap%':>6} {'SL%':>6} "
          f"{'MaxLoss':>8} {'MaxDD':>8}")
    for name in STRAT_ORDER:
        s = _stats(all_t[name])
        print(f"  {name:<15} {s['n']:>6} {s['sharpe']:>7.3f} ${s['pnl']:>9,.0f} {s['wr']:>5.1f}% "
              f"{s['cap_pct']:>5.1f}% {s['sl_pct']:>5.1f}% ${s['max_loss']:>7,.0f} ${s['max_dd']:>7,.0f}")
    ps, _ = port_stats(all_t)
    print(f"\n  PORTFOLIO: Sharpe={ps['sharpe']:.3f}, PnL=${ps['pnl']:,.0f}, MaxDD=${ps['max_dd']:,.0f}, Cap%={ps['cap_pct']:.1f}%")
    return ps


# ═══════════════════════════════════════════════════════════════
# Phase 2: Adaptive Cap Sweep
# ═══════════════════════════════════════════════════════════════
def phase2(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 2: Adaptive Cap Sweep — cap_atr_mult = {CAP_ATR_MULTS}")
    print(f"{'='*120}")

    results = {}

    # 2a: Current lots + adaptive cap
    print(f"\n  --- 2a: Current Lots + Adaptive Cap ---")
    print(f"  {'cap_mult':>8} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Cap%':>6} {'SL%':>6} "
          f"{'MaxLoss':>8} {'MaxDD':>8}")

    print(f"  {'FIXED':>8}", end="")
    all_t = run_all(h1); ps, _ = port_stats(all_t)
    print(f" {ps['n']:>6} {ps['sharpe']:>7.3f} ${ps['pnl']:>9,.0f} {ps['cap_pct']:>5.1f}% {ps['sl_pct']:>5.1f}% "
          f"${ps['max_loss']:>7,.0f} ${ps['max_dd']:>7,.0f}")
    results['2a'] = [{'cap_mult': 'fixed', **ps}]

    for cm in CAP_ATR_MULTS:
        all_t = run_all(h1, cap_atr_mult=cm)
        ps, _ = port_stats(all_t)
        print(f"  {cm:>8.1f} {ps['n']:>6} {ps['sharpe']:>7.3f} ${ps['pnl']:>9,.0f} {ps['cap_pct']:>5.1f}% {ps['sl_pct']:>5.1f}% "
              f"${ps['max_loss']:>7,.0f} ${ps['max_dd']:>7,.0f}")
        results['2a'].append({'cap_mult': cm, **ps})

    # 2b: New lots + adaptive cap
    print(f"\n  --- 2b: New Lots ({', '.join(f'{k}={v}' for k,v in NEW_LOTS.items())}) + Adaptive Cap ---")
    print(f"  {'cap_mult':>8} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'Cap%':>6} {'SL%':>6} "
          f"{'MaxLoss':>8} {'MaxDD':>8}")

    print(f"  {'FIXED':>8}", end="")
    all_t = run_all(h1, lots=NEW_LOTS); ps, _ = port_stats(all_t)
    print(f" {ps['n']:>6} {ps['sharpe']:>7.3f} ${ps['pnl']:>9,.0f} {ps['cap_pct']:>5.1f}% {ps['sl_pct']:>5.1f}% "
          f"${ps['max_loss']:>7,.0f} ${ps['max_dd']:>7,.0f}")
    results['2b'] = [{'cap_mult': 'fixed', **ps}]

    for cm in CAP_ATR_MULTS:
        all_t = run_all(h1, lots=NEW_LOTS, cap_atr_mult=cm)
        ps, _ = port_stats(all_t)
        print(f"  {cm:>8.1f} {ps['n']:>6} {ps['sharpe']:>7.3f} ${ps['pnl']:>9,.0f} {ps['cap_pct']:>5.1f}% {ps['sl_pct']:>5.1f}% "
              f"${ps['max_loss']:>7,.0f} ${ps['max_dd']:>7,.0f}")
        results['2b'].append({'cap_mult': cm, **ps})

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: Best Config — Per-Strategy Detail
# ═══════════════════════════════════════════════════════════════
def phase3(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 3: Per-Strategy Detail — New Lots + cap_atr_mult={best_cm}")
    print(f"{'='*120}")

    all_t_fixed = run_all(h1, lots=NEW_LOTS)
    all_t_adap = run_all(h1, lots=NEW_LOTS, cap_atr_mult=best_cm)

    print(f"\n  {'Strategy':<15} {'Sh_fix':>7} {'Sh_adap':>8} {'dSh':>7} {'Cap%_fix':>8} {'Cap%_adap':>9} "
          f"{'MaxLoss_f':>10} {'MaxLoss_a':>10}")

    results = {}
    for name in STRAT_ORDER:
        sf = _stats(all_t_fixed[name]); sa = _stats(all_t_adap[name])
        d = sa['sharpe'] - sf['sharpe']
        print(f"  {name:<15} {sf['sharpe']:>7.3f} {sa['sharpe']:>8.3f} {d:>+7.3f} "
              f"{sf['cap_pct']:>7.1f}% {sa['cap_pct']:>8.1f}% ${sf['max_loss']:>9,.0f} ${sa['max_loss']:>9,.0f}")
        results[name] = {'sh_fixed': sf['sharpe'], 'sh_adap': sa['sharpe'], 'd': round(d,3),
                         'cap_fixed': sf['cap_pct'], 'cap_adap': sa['cap_pct'],
                         'maxloss_f': sf['max_loss'], 'maxloss_a': sa['max_loss']}

    # Cap$ at various ATR levels for reference
    print(f"\n  Adaptive Cap $ at different ATR levels (cap_atr_mult={best_cm}):")
    print(f"  {'Strategy':<15} {'Lot':>5} {'ATR=2.5':>8} {'ATR=5':>8} {'ATR=10':>8} {'ATR=20':>8} {'Fixed':>8}")
    for name in STRAT_ORDER:
        lot = NEW_LOTS[name]
        for_label = [2.5, 5, 10, 20]
        caps = [round(best_cm * a * lot * PV, 0) for a in for_label]
        fc = CURRENT_CONFIG[name]['cap']
        print(f"  {name:<15} {lot:>5.2f} " + " ".join(f"${c:>7,.0f}" for c in caps) + f" ${fc:>7}")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 4: K-Fold
# ═══════════════════════════════════════════════════════════════
def phase4(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 4: K-Fold 6-Fold — New Lots + cap_atr_mult={best_cm} vs Fixed Cap")
    print(f"{'='*120}")

    K = 6; start = h1.index[0]; end = h1.index[-1]
    total = (end - start).days; fd = total // K

    print(f"\n  {'Fold':>5} {'Period':>25} {'Sh_fixed':>9} {'Sh_adap':>9} {'Winner':>8}")

    adap_wins = 0; results = []
    for fold in range(K):
        fs = start + pd.Timedelta(days=fold*fd)
        fe = start + pd.Timedelta(days=(fold+1)*fd) if fold < K-1 else end + pd.Timedelta(days=1)
        h1f = h1[(h1.index >= fs) & (h1.index < fe)]
        if len(h1f) < 300: continue

        tf = run_all(h1f, lots=NEW_LOTS); sf, _ = port_stats(tf)
        ta = run_all(h1f, lots=NEW_LOTS, cap_atr_mult=best_cm); sa, _ = port_stats(ta)

        w = "ADAP" if sa['sharpe'] > sf['sharpe'] else "FIXED"
        if w == "ADAP": adap_wins += 1
        per = f"{fs.date()} ~ {fe.date()}"
        print(f"  {fold+1:>5} {per:>25} {sf['sharpe']:>9.3f} {sa['sharpe']:>9.3f} {w:>8}")
        results.append({'fold': fold+1, 'sh_fixed': sf['sharpe'], 'sh_adap': sa['sharpe'], 'winner': w})

    print(f"\n  K-Fold: ADAP wins {adap_wins}/{K}")
    return {'folds': results, 'adap_wins': adap_wins, 'total': K}


# ═══════════════════════════════════════════════════════════════
# Phase 5: Walk-Forward
# ═══════════════════════════════════════════════════════════════
def phase5(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 5: Walk-Forward OOS — New Lots + cap_atr_mult={best_cm} vs Fixed Cap")
    print(f"{'='*120}")

    start = h1.index[0]; end = h1.index[-1]
    train_d, test_d = int(1.5*365), 180
    cursor = start + pd.Timedelta(days=train_d)

    print(f"\n  {'#':>3} {'Test':>25} {'Sh_fixed':>9} {'Sh_adap':>9} {'Winner':>8}")

    aw = 0; tot = 0; period = 0; results = []
    while cursor + pd.Timedelta(days=test_d) <= end + pd.Timedelta(days=1):
        period += 1; ts = cursor; te = cursor + pd.Timedelta(days=test_d)
        h1t = h1[(h1.index >= ts) & (h1.index < te)]
        if len(h1t) < 200: cursor += pd.Timedelta(days=test_d); continue

        tf = run_all(h1t, lots=NEW_LOTS); sf, _ = port_stats(tf)
        ta = run_all(h1t, lots=NEW_LOTS, cap_atr_mult=best_cm); sa, _ = port_stats(ta)

        w = "ADAP" if sa['sharpe'] > sf['sharpe'] else "FIXED"
        if w == "ADAP": aw += 1
        tot += 1
        print(f"  {period:>3} {ts.date()} ~ {te.date()} {sf['sharpe']:>9.3f} {sa['sharpe']:>9.3f} {w:>8}")
        results.append({'period': period, 'sh_fixed': sf['sharpe'], 'sh_adap': sa['sharpe'], 'winner': w})
        cursor += pd.Timedelta(days=test_d)

    print(f"\n  Walk-Forward: ADAP wins {aw}/{tot}")
    return {'periods': results, 'adap_wins': aw, 'total': tot}


# ═══════════════════════════════════════════════════════════════
# Phase 6: Era Validation
# ═══════════════════════════════════════════════════════════════
def phase6(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 6: Era Validation — New Lots + cap_atr_mult={best_cm}")
    print(f"{'='*120}")

    tf = run_all(h1, lots=NEW_LOTS)
    ta = run_all(h1, lots=NEW_LOTS, cap_atr_mult=best_cm)

    print(f"\n  {'Era':<12} {'Sh_fixed':>9} {'Sh_adap':>9} {'dSh':>7} {'PnL_f':>10} {'PnL_a':>10}")
    results = {}
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        fe = [t for nm in STRAT_ORDER for t in filter_era(tf[nm], era)]
        ae = [t for nm in STRAT_ORDER for t in filter_era(ta[nm], era)]
        sf = _sharpe(_daily(fe)); sa = _sharpe(_daily(ae)); d = sa - sf
        pf = sum(t['pnl'] for t in fe); pa = sum(t['pnl'] for t in ae)
        print(f"  {era:<12} {sf:>9.3f} {sa:>9.3f} {d:>+7.3f} ${pf:>9,.0f} ${pa:>9,.0f}")
        results[era] = {'sh_fixed': round(sf,3), 'sh_adap': round(sa,3), 'd': round(d,3)}
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 7: Yearly Stability
# ═══════════════════════════════════════════════════════════════
def phase7(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 7: Yearly Stability — New Lots + cap_atr_mult={best_cm}")
    print(f"{'='*120}")

    tf = run_all(h1, lots=NEW_LOTS); _, df = port_stats(tf)
    ta = run_all(h1, lots=NEW_LOTS, cap_atr_mult=best_cm); _, da = port_stats(ta)

    print(f"\n  {'Year':>6} {'PnL_f':>10} {'PnL_a':>10} {'Sh_f':>7} {'Sh_a':>7} {'dSh':>7}")
    years = sorted(set(df.index.year) | set(da.index.year))
    results = {}
    for yr in years:
        yf = df[df.index.year==yr]; ya = da[da.index.year==yr]
        sf = _sharpe(yf); sa = _sharpe(ya)
        print(f"  {yr:>6} ${float(yf.sum()):>9,.0f} ${float(ya.sum()):>9,.0f} {sf:>7.2f} {sa:>7.2f} {sa-sf:>+7.2f}")
        results[yr] = {'sh_fixed': round(sf,2), 'sh_adap': round(sa,2)}

    neg_years = [yr for yr, v in results.items() if v['sh_adap'] < 0]
    print(f"\n  Negative Sharpe years (adaptive): {neg_years if neg_years else 'NONE'}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 8: Risk Metrics
# ═══════════════════════════════════════════════════════════════
def phase8(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 8: Risk Metrics — Single Trade Risk Analysis")
    print(f"{'='*120}")

    ta = run_all(h1, lots=NEW_LOTS, cap_atr_mult=best_cm)
    all_trades = [t for nm in STRAT_ORDER for t in ta[nm]]

    losses = [t for t in all_trades if t['pnl'] < 0]
    loss_amts = [abs(t['pnl']) for t in losses]

    print(f"\n  Total trades: {len(all_trades)}")
    print(f"  Loss trades: {len(losses)} ({len(losses)/len(all_trades)*100:.1f}%)")
    if loss_amts:
        print(f"  Max single loss: ${max(loss_amts):,.2f} ({max(loss_amts)/CAPITAL*100:.1f}% of capital)")
        print(f"  Avg loss: ${np.mean(loss_amts):,.2f}")
        print(f"  P95 loss: ${np.percentile(loss_amts, 95):,.2f}")
        print(f"  P99 loss: ${np.percentile(loss_amts, 99):,.2f}")

    # Per-strategy max loss
    print(f"\n  {'Strategy':<15} {'MaxLoss$':>9} {'%Capital':>9} {'AvgLoss$':>9} {'P99Loss$':>9}")
    results = {}
    for name in STRAT_ORDER:
        t = ta[name]
        ls = [abs(tr['pnl']) for tr in t if tr['pnl'] < 0]
        if not ls: ls = [0]
        ml = max(ls); al = np.mean(ls); p99 = np.percentile(ls, 99) if len(ls) > 1 else ml
        print(f"  {name:<15} ${ml:>8,.2f} {ml/CAPITAL*100:>8.1f}% ${al:>8,.2f} ${p99:>8,.2f}")
        results[name] = {'max_loss': round(ml,2), 'pct_cap': round(ml/CAPITAL*100,1),
                         'avg_loss': round(al,2), 'p99_loss': round(p99,2)}

    # Worst day
    _, da = port_stats(ta)
    worst_day = da.min()
    print(f"\n  Worst single day: ${worst_day:,.2f} ({worst_day/CAPITAL*100:.1f}% of capital)")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 9: Combined with R187 ATR Pctl Floor
# ═══════════════════════════════════════════════════════════════
def phase9(h1, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 9: Combined — New Lots + Adaptive Cap + R187 ATR Pctl Floor (all-strategy)")
    print(f"{'='*120}")

    pctl = compute_atr_pctl(compute_atr(h1), lb=300)
    PF = 30

    configs = {
        'A_current_live':          {'lots': None, 'cm': 0, 'pctl': None, 'pf': 0},
        'B_newlots_fixedcap':      {'lots': NEW_LOTS, 'cm': 0, 'pctl': None, 'pf': 0},
        'C_newlots_adapcap':       {'lots': NEW_LOTS, 'cm': best_cm, 'pctl': None, 'pf': 0},
        'D_newlots_adapcap+r187':  {'lots': NEW_LOTS, 'cm': best_cm, 'pctl': pctl, 'pf': PF},
        'E_curlots_r187_all':      {'lots': None, 'cm': 0, 'pctl': pctl, 'pf': PF},
    }

    print(f"\n  {'Config':<30} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'Cap%':>6} {'MaxLoss':>8}")
    results = {}
    for label, cfg in configs.items():
        all_t = run_all(h1, lots=cfg['lots'], cap_atr_mult=cfg['cm'],
                        pctl_v=cfg['pctl'], pctl_f=cfg['pf'])
        s, _ = port_stats(all_t)
        pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {label:<30} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} ${s['max_dd']:>7,.0f} "
              f"{s['cap_pct']:>5.1f}% ${s['max_loss']:>7,.0f}")
        results[label] = s

    # Era check for best combined
    print(f"\n  Era check for D (newlots+adapcap+r187):")
    ta = run_all(h1, lots=NEW_LOTS, cap_atr_mult=best_cm, pctl_v=pctl, pctl_f=PF)
    tc = run_all(h1)  # current live baseline
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        ae = [t for nm in STRAT_ORDER for t in filter_era(ta[nm], era)]
        ce = [t for nm in STRAT_ORDER for t in filter_era(tc[nm], era)]
        sa = _sharpe(_daily(ae)); sc = _sharpe(_daily(ce))
        print(f"    {era:<12}: Current={sc:.3f}, Combined={sa:.3f} (d={sa-sc:+.3f})")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 10: Final Scorecard
# ═══════════════════════════════════════════════════════════════
def phase10(p1, p2, p3, p4, p5, p6, p7, p8, p9, best_cm):
    print(f"\n{'='*120}")
    print(f"  PHASE 10: Final Scorecard")
    print(f"{'='*120}")

    print(f"\n  ┌─ ADAPTIVE CAP ─────────────────────────────────────────────────┐")
    print(f"  │ Best cap_atr_mult: {best_cm}")
    print(f"  │ K-Fold: ADAP wins {p4['adap_wins']}/{p4['total']}")
    print(f"  │ Walk-Forward: ADAP wins {p5['adap_wins']}/{p5['total']}")
    kf_pass = p4['adap_wins'] >= p4['total'] * 0.5
    wf_pass = p5['adap_wins'] >= p5['total'] * 0.5
    print(f"  │ K-Fold: {'PASS' if kf_pass else 'FAIL'}")
    print(f"  │ Walk-Forward: {'PASS' if wf_pass else 'FAIL'}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ ERA STABILITY ─────────────────────────────────────────────────┐")
    era_ok = all(p6[e]['d'] >= -0.2 for e in p6)
    for e in p6:
        d = p6[e]['d']
        status = "OK" if d >= 0 else ("MARGINAL" if d >= -0.2 else "DEGRADED")
        print(f"  │ {e:<12}: dSharpe={d:+.3f} [{status}]")
    print(f"  │ Overall: {'PASS' if era_ok else 'CONCERNS'}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ RISK ────────────────────────────────────────────────────────┐")
    max_loss = max(v['max_loss'] for v in p8.values())
    print(f"  │ Max single-trade loss: ${max_loss:,.2f} ({max_loss/CAPITAL*100:.1f}% of capital)")
    risk_ok = max_loss / CAPITAL <= 0.10
    print(f"  │ Single-trade risk < 10%: {'PASS' if risk_ok else 'FAIL'}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ COMBINED RESULT ──────────────────────────────────────────────┐")
    if 'D_newlots_adapcap+r187' in p9 and 'A_current_live' in p9:
        d = p9['D_newlots_adapcap+r187']; a = p9['A_current_live']
        print(f"  │ Current live: Sharpe={a['sharpe']:.3f}")
        print(f"  │ Best combined (newlots+adap+r187): Sharpe={d['sharpe']:.3f}")
        print(f"  │ Delta: {d['sharpe']-a['sharpe']:+.3f}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    verdict = "GO" if kf_pass and wf_pass and era_ok and risk_ok else "CONDITIONAL"
    print(f"\n  VERDICT: {verdict}")

    return {'best_cm': best_cm, 'kf_pass': kf_pass, 'wf_pass': wf_pass,
            'era_ok': era_ok, 'risk_ok': risk_ok, 'verdict': verdict}


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 120)
    print("  R189 — Adaptive Cap Validation + Lot Recalibration")
    print("=" * 120, flush=True)

    h1 = load_h1()
    atr_s = compute_atr(h1).dropna()
    print(f"  ATR: mean=${atr_s.mean():.2f}, median=${atr_s.median():.2f}, "
          f"current=${atr_s.iloc[-1]:.2f}, Q90=${np.percentile(atr_s,90):.2f}")

    p1 = phase1(h1)
    p2 = phase2(h1)

    # Select best cap_atr_mult from Phase 2b (new lots)
    # Criteria: highest Sharpe among cap_atr_mult values
    best_row = max([r for r in p2['2b'] if r['cap_mult'] != 'fixed'], key=lambda x: x['sharpe'])
    best_cm = best_row['cap_mult']
    print(f"\n  >>> Best cap_atr_mult from sweep: {best_cm} (Sharpe={best_row['sharpe']:.3f})")

    p3 = phase3(h1, best_cm)
    p4 = phase4(h1, best_cm)
    p5 = phase5(h1, best_cm)
    p6 = phase6(h1, best_cm)
    p7 = phase7(h1, best_cm)
    p8 = phase8(h1, best_cm)
    p9 = phase9(h1, best_cm)
    p10 = phase10(p1, p2, p3, p4, p5, p6, p7, p8, p9, best_cm)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {'phase1': p1, 'phase2': p2, 'best_cm': best_cm,
            'phase3': p3, 'phase4': p4, 'phase5': p5, 'phase6': p6,
            'phase7': {str(k):v for k,v in p7.items()}, 'phase8': p8,
            'phase9': p9, 'phase10': p10,
            'new_lots': NEW_LOTS, 'runtime_s': round(elapsed,1)}
    out = OUTPUT_DIR / "r189_results.json"
    with open(out, 'w') as f: json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out}")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()
