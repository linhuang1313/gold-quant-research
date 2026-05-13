#!/usr/bin/env python3
"""
R188 — Lot Recalibration, Cap/SL Mismatch Audit & R187 Full-Strategy Expansion
================================================================================
Addresses 3 critical issues from live system review:

P0: TSMOM 0.15 lot × ATR20 = $1,800 SL but Cap=$60 → Cap always hits first
    Need ATR-adaptive lot recalibration for ALL strategies
P1: R187 ATR Pctl Floor deployed Keltner-only, but PSAR/DT/Chandelier benefit more
    Test full-strategy expansion with K-Fold + WF validation
P2: Cap vs SL mismatch across all strategies in current high-ATR environment
    Audit which strategies have Cap << SL (structural R/R distortion)

Phases:
  1: Cap vs SL mismatch audit — for each strategy at various ATR levels
  2: TSMOM lot recalibration sweep (0.15 → dynamic lots)
  3: All-strategy lot recalibration by target risk per trade
  4: Portfolio comparison: current lots vs recalibrated lots
  5: R187 ATR Pctl Floor: Keltner-only vs all-strategy (K-Fold + WF)
  6: Combined scenario: recalibrated lots + full-strategy R187 filter
  7: Era segmented validation of best config
  8: Final scorecard + deployment recommendation
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r188_lot_risk_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

LIVE_CONFIG = {
    'L8_MAX':      {'lot': 0.02, 'cap': 35,  'sl': 3.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 2},
    'PSAR':        {'lot': 0.09, 'cap': 60,  'sl': 4.0, 'tp': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15},
    'TSMOM':       {'lot': 0.15, 'cap': 60,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'lot': 0.13, 'cap': 60,  'sl': 4.5, 'tp': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'DUAL_THRUST': {'lot': 0.04, 'cap': 18,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'CHANDELIER':  {'lot': 0.08, 'cap': 25,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO', 'DUAL_THRUST', 'CHANDELIER']

ERA_SEGMENTS = {
    'full': None,
    'hike': [("2015-12-01", "2019-01-01"), ("2022-03-01", "2023-08-01")],
    'cut':  [("2019-07-01", "2022-03-01"), ("2024-09-01", "2026-06-01")],
    'recent_3y': [("2023-06-01", "2026-06-01")],
}

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
              sl_atr, tp_atr, ta, td, mh, cap=0):
    held = i - pos['bar']
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
    if not trades: return {'n':0,'sharpe':0,'pnl':0,'wr':0,'max_dd':0,'cap_pct':0}
    daily = _daily(trades); pnls = [t['pnl'] for t in trades]; n = len(trades)
    wins = [p for p in pnls if p > 0]
    eq = daily.cumsum(); dd = float((np.maximum.accumulate(eq)-eq).max()) if len(eq)>1 else 0
    reasons = [t['reason'] for t in trades]
    cap_exits = sum(1 for r in reasons if 'Cap' in r)
    return {'n':n, 'sharpe':round(_sharpe(daily),3), 'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1), 'max_dd':round(dd,2),
            'cap_pct': round(cap_exits/n*100,1) if n > 0 else 0,
            'sl_pct': round(sum(1 for r in reasons if r=='SL')/n*100,1) if n > 0 else 0,
            'tp_pct': round(sum(1 for r in reasons if r=='TP')/n*100,1) if n > 0 else 0}

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
# Strategy backtests (with lot/cap/atr_pctl override)
# ═══════════════════════════════════════════════════════════════
def bt_keltner(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
    df = h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA100']=df['Close'].ewm(span=100,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=25,adjust=False).mean()
    df['KC_upper']=df['KC_mid']+1.2*df['ATR']; df['KC_lower']=df['KC_mid']-1.2*df['ATR']
    df=df.dropna(subset=['ATR','ADX','EMA100','KC_upper'])
    pv = pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr,adx,ema=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values,df['ADX'].values,df['EMA100'].values
    ku,kl=df['KC_upper'].values,df['KC_lower'].values; times=df.index; n=len(df)
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

def bt_psar(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
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

def bt_tsmom(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
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

def bt_sess_bo(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,atr=df['Close'].values,df['High'].values,df['Low'].values,df['ATR'].values
    hrs=df.index.hour; times=df.index; n=len(df); lb=4
    trades=[]; pos=None; le=-999
    for i in range(lb,n):
        if pos:
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

def bt_dt(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
    df=h1.copy(); df['ATR']=compute_atr(df); df=df.dropna(subset=['ATR'])
    pv=pctl_v.reindex(df.index).values if pctl_v is not None else None
    c,h,lo,o,atr=df['Close'].values,df['High'].values,df['Low'].values,df['Open'].values,df['ATR'].values
    times=df.index; n=len(df); nb=6; k=0.5
    trades=[]; pos=None; le=-999
    for i in range(nb,n):
        if pos:
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

def bt_chand(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
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

def run_all(h1, config_override=None, pctl_v=None, pctl_f=0):
    r = {}
    for name in STRAT_ORDER:
        cfg = (config_override or LIVE_CONFIG)[name]
        r[name] = BT[name](h1, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                           cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'],
                           pctl_v=pctl_v, pctl_f=pctl_f)
    return r

def port_stats(all_t):
    merged = [t for nm in STRAT_ORDER for t in all_t[nm]]
    return _stats(merged), _daily(merged)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Cap vs SL Mismatch Audit
# ═══════════════════════════════════════════════════════════════
def phase1_cap_sl_audit(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 1: Cap vs SL Mismatch Audit")
    print(f"  Which strategies have Cap << SL (meaning Cap always hits first)?")
    print(f"{'='*120}")

    atr_s = compute_atr(h1).dropna()
    atr_levels = {'Q25': np.percentile(atr_s, 25), 'Q50': np.percentile(atr_s, 50),
                  'Q75': np.percentile(atr_s, 75), 'Q90': np.percentile(atr_s, 90),
                  'Current(~20)': 20.0}

    print(f"\n  ATR levels: " + ", ".join(f"{k}=${v:.2f}" for k,v in atr_levels.items()))

    print(f"\n  {'Strategy':<15} {'ATR_level':<14} {'Lot':>5} {'SL_$(lot*ATR*sl*PV)':>20} {'Cap$':>6} "
          f"{'Cap/SL%':>8} {'Verdict':>10}")

    results = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        strat_results = {}
        for atr_name, atr_val in atr_levels.items():
            sl_usd = cfg['lot'] * PV * cfg['sl'] * atr_val
            cap_usd = cfg['cap']
            ratio = cap_usd / sl_usd * 100 if sl_usd > 0 else 999
            verdict = "OK" if ratio >= 80 else ("TIGHT" if ratio >= 40 else "MISMATCH")
            print(f"  {name:<15} {atr_name:<14} {cfg['lot']:>5.2f} ${sl_usd:>19.0f} ${cap_usd:>5} "
                  f"{ratio:>7.0f}% {verdict:>10}")
            strat_results[atr_name] = {'sl_usd': round(sl_usd,0), 'cap_usd': cap_usd,
                                        'ratio': round(ratio,1), 'verdict': verdict}
        results[name] = strat_results
        print()

    # Actual exit reason distribution from backtest
    print(f"  Actual exit reason distribution (full backtest):")
    print(f"  {'Strategy':<15} {'N':>6} {'TP%':>6} {'SL%':>6} {'Cap%':>6} {'Trail%':>7} {'Timeout%':>9} {'Rev%':>6}")
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        trades = BT[name](h1, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                          cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])
        n = len(trades)
        if n == 0: print(f"  {name:<15} {0:>6}"); continue
        reasons = [t['reason'] for t in trades]
        tp = sum(1 for r in reasons if r=='TP')/n*100
        sl = sum(1 for r in reasons if r=='SL')/n*100
        cap = sum(1 for r in reasons if 'Cap' in r)/n*100
        trail = sum(1 for r in reasons if 'Trail' in r)/n*100
        timeout = sum(1 for r in reasons if 'Timeout' in r)/n*100
        rev = sum(1 for r in reasons if 'Rev' in r)/n*100
        print(f"  {name:<15} {n:>6} {tp:>5.1f}% {sl:>5.1f}% {cap:>5.1f}% {trail:>6.1f}% {timeout:>8.1f}% {rev:>5.1f}%")
        results[name]['exit_dist'] = {'n':n, 'tp':round(tp,1), 'sl':round(sl,1),
                                       'cap':round(cap,1), 'trail':round(trail,1)}
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: TSMOM Lot Recalibration
# ═══════════════════════════════════════════════════════════════
def phase2_tsmom_recal(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 2: TSMOM Lot Recalibration Sweep")
    print(f"  Current: 0.15 lot → test 0.01~0.15 to find ATR-appropriate size")
    print(f"{'='*120}")

    LOTS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    cfg = LIVE_CONFIG['TSMOM']

    print(f"\n  {'Lot':>5} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'Cap%':>6} {'SL%':>6} "
          f"{'MaxDD':>8} {'SL$@ATR20':>10} {'Cap/SL':>7}")

    results = []
    for lot in LOTS:
        trades = bt_tsmom(h1, lot, cfg['cap'], cfg['sl'], cfg['tp'],
                          cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])
        s = _stats(trades)
        sl_at_20 = lot * PV * cfg['sl'] * 20
        cap_sl = cfg['cap'] / sl_at_20 * 100 if sl_at_20 > 0 else 999
        pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {lot:>5.2f} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} {s['wr']:>5.1f}% {s['cap_pct']:>5.1f}% "
              f"{s['sl_pct']:>5.1f}% ${s['max_dd']:>7,.0f} ${sl_at_20:>9,.0f} {cap_sl:>6.0f}%")
        results.append({'lot':lot, **s, 'sl_at_atr20': round(sl_at_20,0), 'cap_sl_ratio': round(cap_sl,1)})

    # Recommend lot where Cap/SL ratio >= 80% at ATR=20
    good = [r for r in results if r['cap_sl_ratio'] >= 80 and r['sharpe'] > 0]
    if good:
        best = max(good, key=lambda x: x['sharpe'])
        print(f"\n  RECOMMENDED TSMOM lot: {best['lot']:.2f} (Sharpe={best['sharpe']:.3f}, "
              f"Cap/SL={best['cap_sl_ratio']:.0f}% at ATR=20)")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: All-Strategy Risk-Targeted Recalibration
# ═══════════════════════════════════════════════════════════════
def phase3_risk_target(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 3: Risk-Targeted Lot Recalibration")
    print(f"  Target: SL loss at ATR=20 should not exceed Cap")
    print(f"  Formula: max_lot = Cap / (SL_mult × ATR_target × PV)")
    print(f"{'='*120}")

    ATR_TARGET = 20  # current environment
    print(f"\n  {'Strategy':<15} {'Current':>8} {'Cap':>5} {'SL_mult':>8} "
          f"{'SL$_now':>8} {'Max_lot':>8} {'Proposed':>8} {'Change':>10}")

    recal = {}
    for name in STRAT_ORDER:
        cfg = LIVE_CONFIG[name]
        cur_lot = cfg['lot']
        sl_now = cur_lot * PV * cfg['sl'] * ATR_TARGET
        max_lot = cfg['cap'] / (cfg['sl'] * ATR_TARGET * PV)
        # Round down to 0.01
        proposed = max(0.01, round(int(max_lot * 100) / 100, 2))
        change = "KEEP" if proposed >= cur_lot else f"{cur_lot:.2f}→{proposed:.2f}"
        if proposed >= cur_lot: proposed = cur_lot
        print(f"  {name:<15} {cur_lot:>8.2f} ${cfg['cap']:>4} {cfg['sl']:>8.1f} "
              f"${sl_now:>7.0f} {max_lot:>8.3f} {proposed:>8.2f} {change:>10}")
        recal[name] = {**cfg, 'lot': proposed}

    return recal


# ═══════════════════════════════════════════════════════════════
# Phase 4: Portfolio Comparison (current vs recalibrated)
# ═══════════════════════════════════════════════════════════════
def phase4_portfolio_compare(h1, recal_config):
    print(f"\n{'='*120}")
    print(f"  PHASE 4: Portfolio — Current Lots vs Recalibrated Lots")
    print(f"{'='*120}")

    cur = run_all(h1, LIVE_CONFIG)
    rec = run_all(h1, recal_config)

    print(f"\n  --- Per-Strategy ---")
    print(f"  {'Strategy':<15} {'Sh_cur':>7} {'Sh_rec':>7} {'dSh':>7} {'PnL_cur':>10} {'PnL_rec':>10} "
          f"{'Cap%_cur':>8} {'Cap%_rec':>8}")

    for name in STRAT_ORDER:
        sc = _stats(cur[name]); sr = _stats(rec[name])
        d = sr['sharpe'] - sc['sharpe']
        print(f"  {name:<15} {sc['sharpe']:>7.3f} {sr['sharpe']:>7.3f} {d:>+7.3f} "
              f"${sc['pnl']:>9,.0f} ${sr['pnl']:>9,.0f} {sc['cap_pct']:>7.1f}% {sr['cap_pct']:>7.1f}%")

    sc_p, dc_p = port_stats(cur); sr_p, dr_p = port_stats(rec)
    print(f"\n  PORTFOLIO: Current Sharpe={sc_p['sharpe']:.3f}, Recal Sharpe={sr_p['sharpe']:.3f} "
          f"(d={sr_p['sharpe']-sc_p['sharpe']:+.3f})")
    print(f"  PORTFOLIO: Current PnL=${sc_p['pnl']:,.0f}, Recal PnL=${sr_p['pnl']:,.0f}")

    # Yearly comparison
    print(f"\n  --- Yearly ---")
    print(f"  {'Year':>6} {'PnL_cur':>10} {'PnL_rec':>10} {'Sh_cur':>7} {'Sh_rec':>7}")
    years = sorted(set(dc_p.index.year) | set(dr_p.index.year))
    for yr in years:
        yc = dc_p[dc_p.index.year==yr]; yr_r = dr_p[dr_p.index.year==yr]
        print(f"  {yr:>6} ${float(yc.sum()):>9,.0f} ${float(yr_r.sum()):>9,.0f} "
              f"{_sharpe(yc):>7.2f} {_sharpe(yr_r):>7.2f}")

    return {'current': sc_p, 'recal': sr_p}


# ═══════════════════════════════════════════════════════════════
# Phase 5: R187 ATR Pctl: Keltner-only vs All-Strategy
# ═══════════════════════════════════════════════════════════════
def phase5_r187_scope(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 5: R187 ATR Pctl Floor — Keltner-only vs All-Strategy")
    print(f"  K-Fold + Walk-Forward validation for full-strategy expansion")
    print(f"{'='*120}")

    pctl = compute_atr_pctl(compute_atr(h1), lb=300)
    PF = 30

    # Full period comparison
    configs = {
        'A_no_filter': (None, 0),
        'B_keltner_only': ('keltner', PF),
        'C_all_strategy': ('all', PF),
    }

    print(f"\n  Full period comparison:")
    print(f"  {'Config':<20} {'N':>6} {'Sharpe':>7} {'PnL':>10}")

    for label, (scope, pf) in configs.items():
        if scope is None:
            all_t = run_all(h1)
        elif scope == 'keltner':
            all_t = {}
            for name in STRAT_ORDER:
                cfg = LIVE_CONFIG[name]
                if name == 'L8_MAX':
                    all_t[name] = BT[name](h1, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                           cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'],
                                           pctl_v=pctl, pctl_f=pf)
                else:
                    all_t[name] = BT[name](h1, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                           cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])
        else:
            all_t = run_all(h1, pctl_v=pctl, pctl_f=pf)
        s, _ = port_stats(all_t)
        pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {label:<20} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s}")

    # K-Fold: keltner-only vs all-strategy
    K = 6; start = h1.index[0]; end = h1.index[-1]
    total = (end - start).days; fd = total // K

    print(f"\n  K-Fold {K}-fold: Keltner-only vs All-strategy (which scope wins more folds?)")
    print(f"  {'Fold':>5} {'Period':>25} {'Sh_kelt':>8} {'Sh_all':>8} {'Winner':>10}")

    kelt_wins = 0; all_wins = 0
    for fold in range(K):
        fs = start + pd.Timedelta(days=fold*fd)
        fe = start + pd.Timedelta(days=(fold+1)*fd) if fold < K-1 else end + pd.Timedelta(days=1)
        h1f = h1[(h1.index >= fs) & (h1.index < fe)]
        if len(h1f) < 300: continue
        pf = pctl.reindex(h1f.index)

        # Keltner-only
        kt = {}
        for name in STRAT_ORDER:
            cfg = LIVE_CONFIG[name]
            if name == 'L8_MAX':
                kt[name] = BT[name](h1f, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                    cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'], pctl_v=pf, pctl_f=PF)
            else:
                kt[name] = BT[name](h1f, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                    cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])
        sk, _ = port_stats(kt)

        # All-strategy
        at = run_all(h1f, pctl_v=pf, pctl_f=PF)
        sa, _ = port_stats(at)

        w = "ALL" if sa['sharpe'] > sk['sharpe'] else "KELT"
        if w == "ALL": all_wins += 1
        else: kelt_wins += 1
        per = f"{fs.date()} ~ {fe.date()}"
        print(f"  {fold+1:>5} {per:>25} {sk['sharpe']:>8.3f} {sa['sharpe']:>8.3f} {w:>10}")

    print(f"\n  K-Fold: ALL wins {all_wins}/{K}, KELT wins {kelt_wins}/{K}")

    # Walk-Forward
    train_d, test_d = int(1.5*365), 180
    cursor = start + pd.Timedelta(days=train_d)
    aw = 0; kw = 0; tot = 0

    print(f"\n  Walk-Forward (18mo/6mo): All vs Keltner-only")
    print(f"  {'#':>3} {'Test':>25} {'Sh_kelt':>8} {'Sh_all':>8} {'Winner':>8}")

    period = 0
    while cursor + pd.Timedelta(days=test_d) <= end + pd.Timedelta(days=1):
        period += 1; ts = cursor; te = cursor + pd.Timedelta(days=test_d)
        h1t = h1[(h1.index >= ts) & (h1.index < te)]
        if len(h1t) < 200: cursor += pd.Timedelta(days=test_d); continue
        pf = pctl.reindex(h1t.index)

        kt = {}
        for name in STRAT_ORDER:
            cfg = LIVE_CONFIG[name]
            if name == 'L8_MAX':
                kt[name] = BT[name](h1t, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                    cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'], pctl_v=pf, pctl_f=PF)
            else:
                kt[name] = BT[name](h1t, cfg['lot'], cfg['cap'], cfg['sl'], cfg['tp'],
                                    cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'])
        sk, _ = port_stats(kt)

        at = run_all(h1t, pctl_v=pf, pctl_f=PF)
        sa, _ = port_stats(at)

        w = "ALL" if sa['sharpe'] > sk['sharpe'] else "KELT"
        if w == "ALL": aw += 1
        else: kw += 1
        tot += 1
        print(f"  {period:>3} {ts.date()} ~ {te.date()} {sk['sharpe']:>8.3f} {sa['sharpe']:>8.3f} {w:>8}")
        cursor += pd.Timedelta(days=test_d)

    print(f"\n  Walk-Forward: ALL wins {aw}/{tot}, KELT wins {kw}/{tot}")
    verdict = "EXPAND_TO_ALL" if aw >= tot * 0.6 else "KEEP_KELTNER_ONLY"
    print(f"  Verdict: {verdict}")

    return {'kfold_all_wins': all_wins, 'kfold_total': K,
            'wf_all_wins': aw, 'wf_total': tot, 'verdict': verdict}


# ═══════════════════════════════════════════════════════════════
# Phase 6: Combined Best Config
# ═══════════════════════════════════════════════════════════════
def phase6_combined(h1, recal_config, r187_scope):
    print(f"\n{'='*120}")
    print(f"  PHASE 6: Combined — Recalibrated Lots + R187 Scope")
    print(f"{'='*120}")

    pctl = compute_atr_pctl(compute_atr(h1), lb=300)
    use_all = (r187_scope == 'EXPAND_TO_ALL')
    pf = 30

    configs = {
        'A_current_live': (LIVE_CONFIG, None, 0),
        'B_recal_only': (recal_config, None, 0),
        'C_current+r187_kelt': (LIVE_CONFIG, 'keltner', pf),
        'D_recal+r187_all': (recal_config, 'all' if use_all else 'keltner', pf),
    }

    print(f"\n  {'Config':<25} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'Cap%':>6}")
    results = {}
    for label, (cfg, scope, pf_val) in configs.items():
        if scope is None:
            all_t = run_all(h1, cfg)
        elif scope == 'keltner':
            all_t = {}
            for name in STRAT_ORDER:
                c = cfg[name]
                if name == 'L8_MAX':
                    all_t[name] = BT[name](h1, c['lot'], c['cap'], c['sl'], c['tp'],
                                           c['trail_act'], c['trail_dist'], c['max_hold'],
                                           pctl_v=pctl, pctl_f=pf_val)
                else:
                    all_t[name] = BT[name](h1, c['lot'], c['cap'], c['sl'], c['tp'],
                                           c['trail_act'], c['trail_dist'], c['max_hold'])
        else:
            all_t = run_all(h1, cfg, pctl_v=pctl, pctl_f=pf_val)
        s, _ = port_stats(all_t)
        pnl_s = f"${s['pnl']:>9,.0f}" if s['pnl'] >= 0 else f"-${abs(s['pnl']):>8,.0f}"
        print(f"  {label:<25} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} ${s['max_dd']:>7,.0f} {s['cap_pct']:>5.1f}%")
        results[label] = s

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 7: Era Validation
# ═══════════════════════════════════════════════════════════════
def phase7_era(h1, recal_config, r187_scope):
    print(f"\n{'='*120}")
    print(f"  PHASE 7: Era Validation — Best Combined Config")
    print(f"{'='*120}")

    pctl = compute_atr_pctl(compute_atr(h1), lb=300)
    use_all = (r187_scope == 'EXPAND_TO_ALL')
    pf = 30

    # Current live
    cur = run_all(h1, LIVE_CONFIG)
    # Best combined
    if use_all:
        best = run_all(h1, recal_config, pctl_v=pctl, pctl_f=pf)
    else:
        best = {}
        for name in STRAT_ORDER:
            c = recal_config[name]
            if name == 'L8_MAX':
                best[name] = BT[name](h1, c['lot'], c['cap'], c['sl'], c['tp'],
                                      c['trail_act'], c['trail_dist'], c['max_hold'],
                                      pctl_v=pctl, pctl_f=pf)
            else:
                best[name] = BT[name](h1, c['lot'], c['cap'], c['sl'], c['tp'],
                                      c['trail_act'], c['trail_dist'], c['max_hold'])

    print(f"\n  {'Era':<12} {'Sh_cur':>7} {'Sh_best':>8} {'dSh':>7} {'PnL_cur':>10} {'PnL_best':>10}")
    results = {}
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        cur_era = [t for name in STRAT_ORDER for t in filter_era(cur[name], era)]
        best_era = [t for name in STRAT_ORDER for t in filter_era(best[name], era)]
        sc = _sharpe(_daily(cur_era)); sb = _sharpe(_daily(best_era))
        pc = sum(t['pnl'] for t in cur_era); pb = sum(t['pnl'] for t in best_era)
        d = sb - sc
        print(f"  {era:<12} {sc:>7.3f} {sb:>8.3f} {d:>+7.3f} ${pc:>9,.0f} ${pb:>9,.0f}")
        results[era] = {'sh_cur': round(sc,3), 'sh_best': round(sb,3), 'd': round(d,3)}

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 8: Final Scorecard
# ═══════════════════════════════════════════════════════════════
def phase8_scorecard(p1, p2, p3, p4, p5, p6, p7):
    print(f"\n{'='*120}")
    print(f"  PHASE 8: Final Scorecard & Recommendations")
    print(f"{'='*120}")

    print(f"\n  ┌─ ISSUE 1: Cap vs SL Mismatch ──────────────────────────────────┐")
    mismatch_strats = []
    for name in STRAT_ORDER:
        if 'Current(~20)' in p1[name]:
            v = p1[name]['Current(~20)']['verdict']
            if v != 'OK':
                mismatch_strats.append(f"{name}({v})")
    if mismatch_strats:
        print(f"  │ MISMATCHED at ATR=20: {', '.join(mismatch_strats)}")
        print(f"  │ ACTION: Recalibrate lots per Phase 3 formula")
    else:
        print(f"  │ All strategies OK at ATR=20")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ ISSUE 2: TSMOM Lot ───────────────────────────────────────────┐")
    tsmom_good = [r for r in p2 if r['cap_sl_ratio'] >= 80 and r['sharpe'] > 0]
    if tsmom_good:
        best = max(tsmom_good, key=lambda x: x['sharpe'])
        print(f"  │ Recommended: {best['lot']:.2f} lot (Sharpe={best['sharpe']:.3f})")
        print(f"  │ Current 0.15 lot at ATR=20: Cap/SL = {p2[-1]['cap_sl_ratio']:.0f}%")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ ISSUE 3: R187 Scope ──────────────────────────────────────────┐")
    print(f"  │ K-Fold: ALL wins {p5['kfold_all_wins']}/{p5['kfold_total']}")
    print(f"  │ WF: ALL wins {p5['wf_all_wins']}/{p5['wf_total']}")
    print(f"  │ Verdict: {p5['verdict']}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ COMBINED RESULT ──────────────────────────────────────────────┐")
    if 'D_recal+r187_all' in p6:
        d = p6['D_recal+r187_all']
        a = p6['A_current_live']
        print(f"  │ Current live: Sharpe={a['sharpe']:.3f}, PnL=${a['pnl']:,.0f}")
        print(f"  │ Best combined: Sharpe={d['sharpe']:.3f}, PnL=${d['pnl']:,.0f}")
        print(f"  │ Delta: dSharpe={d['sharpe']-a['sharpe']:+.3f}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    era_ok = all(p7[e]['d'] >= -0.1 for e in p7)
    print(f"\n  Era validation: {'ALL PASS' if era_ok else 'SOME DEGRADED'}")

    return {'mismatch_strats': mismatch_strats, 'r187_verdict': p5['verdict'], 'era_ok': era_ok}


# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 120)
    print("  R188 — Lot Recalibration, Cap/SL Audit & R187 Full-Strategy Expansion")
    print("=" * 120, flush=True)

    h1 = load_h1()
    print(f"  Mean ATR: ${compute_atr(h1).dropna().mean():.2f}")

    p1 = phase1_cap_sl_audit(h1)
    p2 = phase2_tsmom_recal(h1)
    p3 = phase3_risk_target(h1)
    p4 = phase4_portfolio_compare(h1, p3)
    p5 = phase5_r187_scope(h1)
    p6 = phase6_combined(h1, p3, p5['verdict'])
    p7 = phase7_era(h1, p3, p5['verdict'])
    p8 = phase8_scorecard(p1, p2, p3, p4, p5, p6, p7)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {'phase1':p1, 'phase2':p2, 'phase3':{k:v for k,v in p3.items()},
            'phase4':p4, 'phase5':p5, 'phase6':p6, 'phase7':p7, 'phase8':p8,
            'runtime_s': round(elapsed,1)}
    out = OUTPUT_DIR / "r188_results.json"
    with open(out, 'w') as f: json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out}")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()
