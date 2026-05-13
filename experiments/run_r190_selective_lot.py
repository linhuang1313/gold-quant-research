#!/usr/bin/env python3
"""
R190 — Selective Lot Reduction
================================
R189 showed full lot reduction (all 6 → 0.01-0.04) Sharpe is better but PnL halved.
Key insight: only TSMOM (3%) and SESS_BO (5%) have extreme Cap/SL mismatch.
Others are moderate (L8 25%, PSAR 8%, DT 5%, CH 3%).

Test selective approaches:
  Config A: Current live (baseline)
  Config B: Only reduce TSMOM + SESS_BO (worst offenders)
  Config C: Reduce TSMOM + SESS_BO + CHANDELIER (3 worst)
  Config D: Reduce TSMOM + SESS_BO + PSAR + CHANDELIER (4 worst, keep L8 + DT)
  Config E: Full reduction (R189 new lots — all 6)
  Config F: ATR-adaptive lots (lot scales inversely with ATR regime)

Each config tested with: full period + K-Fold + WF + Era + yearly + risk + R187 combo
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r190_selective_lot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; CAPITAL = 5000

CURRENT_LOTS = {
    'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.15,
    'SESS_BO': 0.13, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08,
}

CONFIGS = {
    'A_current': {
        'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.15,
        'SESS_BO': 0.13, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08,
    },
    'B_tsmom_sess': {
        'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.04,
        'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.08,
    },
    'C_3worst': {
        'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.04,
        'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.03,
    },
    'D_4worst': {
        'L8_MAX': 0.02, 'PSAR': 0.04, 'TSMOM': 0.04,
        'SESS_BO': 0.04, 'DUAL_THRUST': 0.04, 'CHANDELIER': 0.03,
    },
    'E_full_reduce': {
        'L8_MAX': 0.02, 'PSAR': 0.04, 'TSMOM': 0.04,
        'SESS_BO': 0.04, 'DUAL_THRUST': 0.02, 'CHANDELIER': 0.03,
    },
}

STRAT_CFG = {
    'L8_MAX':      {'cap': 35,  'sl': 3.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 2},
    'PSAR':        {'cap': 60,  'sl': 4.0, 'tp': 6.0, 'trail_act': 0.08, 'trail_dist': 0.015, 'max_hold': 15},
    'TSMOM':       {'cap': 60,  'sl': 6.0, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 12},
    'SESS_BO':     {'cap': 60,  'sl': 4.5, 'tp': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'DUAL_THRUST': {'cap': 18,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
    'CHANDELIER':  {'cap': 25,  'sl': 4.5, 'tp': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20},
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
              sl_atr, tp_atr, ta, td, mh, cap):
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
    held = i - pos['bar']
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
    return {'n':n, 'sharpe':round(_sharpe(daily),3), 'pnl':round(sum(pnls),2),
            'wr':round(len(wins)/n*100,1), 'max_dd':round(dd,2),
            'cap_pct': round(sum(1 for r in reasons if 'Cap' in r)/n*100,1),
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
# Strategy backtests
# ═══════════════════════════════════════════════════════════════
def bt_keltner(h1, lot, cap, sl, tp, ta, td, mh, pctl_v=None, pctl_f=0):
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

def run_all(h1, lots, pctl_v=None, pctl_f=0):
    r = {}
    for name in STRAT_ORDER:
        cfg = STRAT_CFG[name]
        r[name] = BT[name](h1, lots[name], cfg['cap'], cfg['sl'], cfg['tp'],
                           cfg['trail_act'], cfg['trail_dist'], cfg['max_hold'],
                           pctl_v=pctl_v, pctl_f=pctl_f)
    return r

def port_stats(all_t):
    merged = [t for nm in STRAT_ORDER for t in all_t[nm]]
    return _stats(merged), _daily(merged)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Full Period Comparison
# ═══════════════════════════════════════════════════════════════
def phase1(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 1: Full Period — All Configs")
    print(f"{'='*120}")

    print(f"\n  Lot assignments:")
    print(f"  {'Config':<16} " + " ".join(f"{'  '+s:>8}" for s in STRAT_ORDER))
    for label, lots in CONFIGS.items():
        print(f"  {label:<16} " + " ".join(f"{lots[s]:>8.2f}" for s in STRAT_ORDER))

    print(f"\n  {'Config':<16} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'Cap%':>6} "
          f"{'MaxLoss':>8} {'MaxDD':>8}")

    results = {}
    for label, lots in CONFIGS.items():
        all_t = run_all(h1, lots)
        s, _ = port_stats(all_t)
        pnl_s = f"${s['pnl']:>9,.0f}"
        print(f"  {label:<16} {s['n']:>6} {s['sharpe']:>7.3f} {pnl_s} {s['wr']:>5.1f}% "
              f"{s['cap_pct']:>5.1f}% ${s['max_loss']:>7,.0f} ${s['max_dd']:>7,.0f}")
        results[label] = s

    # Per-strategy detail for each config
    print(f"\n  Per-strategy Sharpe:")
    print(f"  {'Config':<16} " + " ".join(f"{s:>10}" for s in STRAT_ORDER))
    for label, lots in CONFIGS.items():
        all_t = run_all(h1, lots)
        shs = [_stats(all_t[name])['sharpe'] for name in STRAT_ORDER]
        print(f"  {label:<16} " + " ".join(f"{sh:>10.3f}" for sh in shs))

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: Yearly Breakdown
# ═══════════════════════════════════════════════════════════════
def phase2(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 2: Yearly PnL — All Configs")
    print(f"{'='*120}")

    dailies = {}
    for label, lots in CONFIGS.items():
        all_t = run_all(h1, lots)
        _, d = port_stats(all_t)
        dailies[label] = d

    years = sorted(set().union(*[set(d.index.year) for d in dailies.values()]))
    print(f"\n  {'Year':>6} " + " ".join(f"{label:>16}" for label in CONFIGS))
    for yr in years:
        vals = []
        for label in CONFIGS:
            yv = dailies[label][dailies[label].index.year==yr]
            vals.append(f"${float(yv.sum()):>9,.0f}" if len(yv) > 0 else f"{'N/A':>10}")
        print(f"  {yr:>6} " + " ".join(f"{v:>16}" for v in vals))

    # Recent high-ATR years focus
    print(f"\n  Recent high-ATR years (2024-2026) PnL retention vs A_current:")
    a_d = dailies['A_current']
    for yr in [2024, 2025, 2026]:
        a_pnl = float(a_d[a_d.index.year==yr].sum())
        if a_pnl == 0: continue
        for label in list(CONFIGS.keys())[1:]:
            d = dailies[label]
            pnl = float(d[d.index.year==yr].sum())
            print(f"    {yr} {label}: ${pnl:>9,.0f} ({pnl/a_pnl*100:.0f}% of A)")

    results = {}
    for label in CONFIGS:
        results[label] = {}
        for yr in years:
            yv = dailies[label][dailies[label].index.year==yr]
            results[label][str(yr)] = round(float(yv.sum()), 0)
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: K-Fold — Best selective vs Current
# ═══════════════════════════════════════════════════════════════
def phase3(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 3: K-Fold 6-Fold — Each Config vs A_current")
    print(f"{'='*120}")

    K = 6; start = h1.index[0]; end = h1.index[-1]
    total = (end - start).days; fd = total // K

    results = {label: {'wins': 0, 'total': K} for label in CONFIGS if label != 'A_current'}

    print(f"\n  {'Fold':>5} {'Period':>25} " + " ".join(f"{label:>12}" for label in CONFIGS))
    for fold in range(K):
        fs = start + pd.Timedelta(days=fold*fd)
        fe = start + pd.Timedelta(days=(fold+1)*fd) if fold < K-1 else end + pd.Timedelta(days=1)
        h1f = h1[(h1.index >= fs) & (h1.index < fe)]
        if len(h1f) < 300: continue

        shs = {}
        for label, lots in CONFIGS.items():
            all_t = run_all(h1f, lots)
            s, _ = port_stats(all_t)
            shs[label] = s['sharpe']

        per = f"{fs.date()} ~ {fe.date()}"
        vals = " ".join(f"{shs[label]:>12.3f}" for label in CONFIGS)
        print(f"  {fold+1:>5} {per:>25} {vals}")

        for label in results:
            if shs[label] > shs['A_current']:
                results[label]['wins'] += 1

    print(f"\n  Wins vs A_current:")
    for label, r in results.items():
        print(f"    {label}: {r['wins']}/{r['total']} {'PASS' if r['wins'] >= r['total']//2 else 'FAIL'}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 4: Walk-Forward
# ═══════════════════════════════════════════════════════════════
def phase4(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 4: Walk-Forward OOS — Each Config vs A_current")
    print(f"{'='*120}")

    start = h1.index[0]; end = h1.index[-1]
    train_d, test_d = int(1.5*365), 180
    cursor = start + pd.Timedelta(days=train_d)

    results = {label: {'wins': 0, 'total': 0} for label in CONFIGS if label != 'A_current'}

    print(f"\n  {'#':>3} {'Test':>25} " + " ".join(f"{label:>12}" for label in CONFIGS))

    period = 0
    while cursor + pd.Timedelta(days=test_d) <= end + pd.Timedelta(days=1):
        period += 1; ts = cursor; te = cursor + pd.Timedelta(days=test_d)
        h1t = h1[(h1.index >= ts) & (h1.index < te)]
        if len(h1t) < 200: cursor += pd.Timedelta(days=test_d); continue

        shs = {}
        for label, lots in CONFIGS.items():
            all_t = run_all(h1t, lots)
            s, _ = port_stats(all_t)
            shs[label] = s['sharpe']

        vals = " ".join(f"{shs[label]:>12.3f}" for label in CONFIGS)
        print(f"  {period:>3} {ts.date()} ~ {te.date()} {vals}")

        for label in results:
            results[label]['total'] += 1
            if shs[label] > shs['A_current']:
                results[label]['wins'] += 1

        cursor += pd.Timedelta(days=test_d)

    print(f"\n  Wins vs A_current:")
    for label, r in results.items():
        pct = r['wins']/r['total']*100 if r['total'] > 0 else 0
        print(f"    {label}: {r['wins']}/{r['total']} ({pct:.0f}%) {'PASS' if pct >= 50 else 'FAIL'}")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 5: Era Validation
# ═══════════════════════════════════════════════════════════════
def phase5(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 5: Era Validation — All Configs")
    print(f"{'='*120}")

    results = {}
    for era in ['full', 'hike', 'cut', 'recent_3y']:
        print(f"\n  {era}:")
        print(f"  {'Config':<16} {'Sharpe':>7} {'PnL':>10}")
        for label, lots in CONFIGS.items():
            all_t = run_all(h1, lots)
            era_t = [t for nm in STRAT_ORDER for t in filter_era(all_t[nm], era)]
            sh = _sharpe(_daily(era_t)); pnl = sum(t['pnl'] for t in era_t)
            print(f"  {label:<16} {sh:>7.3f} ${pnl:>9,.0f}")
            if label not in results: results[label] = {}
            results[label][era] = {'sharpe': round(sh,3), 'pnl': round(pnl,0)}
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 6: Combined with R187 ATR Pctl Floor
# ═══════════════════════════════════════════════════════════════
def phase6(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 6: Combined with R187 ATR Pctl Floor (all-strategy, lb=300, pctl=30)")
    print(f"{'='*120}")

    pctl = compute_atr_pctl(compute_atr(h1), lb=300)
    PF = 30

    print(f"\n  {'Config':<16} {'N_base':>7} {'Sh_base':>8} {'N_r187':>7} {'Sh_r187':>8} {'dSh':>7} "
          f"{'PnL_base':>10} {'PnL_r187':>10}")

    results = {}
    for label, lots in CONFIGS.items():
        t_base = run_all(h1, lots)
        t_r187 = run_all(h1, lots, pctl_v=pctl, pctl_f=PF)
        sb, _ = port_stats(t_base); sr, _ = port_stats(t_r187)
        d = sr['sharpe'] - sb['sharpe']
        print(f"  {label:<16} {sb['n']:>7} {sb['sharpe']:>8.3f} {sr['n']:>7} {sr['sharpe']:>8.3f} {d:>+7.3f} "
              f"${sb['pnl']:>9,.0f} ${sr['pnl']:>9,.0f}")
        results[label] = {'base': sb, 'r187': sr, 'delta': round(d,3)}

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 7: Risk Analysis for Best Config
# ═══════════════════════════════════════════════════════════════
def phase7(h1):
    print(f"\n{'='*120}")
    print(f"  PHASE 7: Risk Analysis — All Configs")
    print(f"{'='*120}")

    print(f"\n  {'Config':<16} {'MaxLoss$':>9} {'%Capital':>8} {'P99$':>8} {'WorstDay$':>10} {'%Cap':>6}")
    results = {}
    for label, lots in CONFIGS.items():
        all_t = run_all(h1, lots)
        all_trades = [t for nm in STRAT_ORDER for t in all_t[nm]]
        losses = [abs(t['pnl']) for t in all_trades if t['pnl'] < 0]
        _, d = port_stats(all_t)
        ml = max(losses) if losses else 0
        p99 = np.percentile(losses, 99) if len(losses) > 1 else ml
        wd = float(d.min()) if len(d) > 0 else 0
        print(f"  {label:<16} ${ml:>8,.0f} {ml/CAPITAL*100:>7.1f}% ${p99:>7,.0f} ${wd:>9,.0f} {wd/CAPITAL*100:>5.1f}%")
        results[label] = {'max_loss': round(ml,0), 'p99': round(p99,0),
                          'worst_day': round(wd,0), 'max_loss_pct': round(ml/CAPITAL*100,1)}
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 8: Final Scorecard
# ═══════════════════════════════════════════════════════════════
def phase8(p1, p2, p3, p4, p5, p6, p7):
    print(f"\n{'='*120}")
    print(f"  PHASE 8: Final Scorecard")
    print(f"{'='*120}")

    print(f"\n  ┌─ SUMMARY TABLE ────────────────────────────────────────────────┐")
    print(f"  │ {'Config':<14} {'Sharpe':>7} {'PnL':>10} {'KF':>5} {'WF':>5} {'Risk':>7} {'Sh+R187':>8} │")
    for label in CONFIGS:
        sh = p1[label]['sharpe']
        pnl = p1[label]['pnl']
        kf = f"{p3[label]['wins']}/{p3[label]['total']}" if label in p3 else "BASE"
        wf_data = p4.get(label, {})
        wf = f"{wf_data.get('wins','')}/{wf_data.get('total','')}" if label in p4 else "BASE"
        risk = f"${p7[label]['max_loss']:,.0f}"
        sh_r = p6[label]['r187']['sharpe']
        print(f"  │ {label:<14} {sh:>7.3f} ${pnl:>9,.0f} {kf:>5} {wf:>5} {risk:>7} {sh_r:>8.3f} │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # Find best config
    # Criteria: must beat A_current Sharpe, K-Fold >= 3/6, WF >= 50%, max risk < 10%
    candidates = []
    for label in list(CONFIGS.keys())[1:]:
        sh = p1[label]['sharpe']
        pnl = p1[label]['pnl']
        kf_pass = p3[label]['wins'] >= 3
        wf_pass = p4[label]['wins'] >= p4[label]['total'] * 0.5 if p4[label]['total'] > 0 else False
        risk_ok = p7[label]['max_loss_pct'] <= 10
        sh_r187 = p6[label]['r187']['sharpe']
        if sh > p1['A_current']['sharpe'] and kf_pass and risk_ok:
            candidates.append({'label': label, 'sharpe': sh, 'pnl': pnl,
                               'kf': p3[label]['wins'], 'wf_pct': p4[label]['wins']/p4[label]['total']*100,
                               'sh_r187': sh_r187})

    if candidates:
        # Sort by Sharpe with R187
        best = max(candidates, key=lambda x: x['sh_r187'])
        print(f"\n  RECOMMENDED: {best['label']}")
        print(f"    Sharpe: {best['sharpe']:.3f} (with R187: {best['sh_r187']:.3f})")
        print(f"    PnL: ${best['pnl']:,.0f}")
        print(f"    K-Fold: {best['kf']}/6, WF: {best['wf_pct']:.0f}%")
    else:
        print(f"\n  No config passes all criteria. Closest options:")
        for label in list(CONFIGS.keys())[1:]:
            sh = p1[label]['sharpe']
            kf = p3[label]['wins']
            print(f"    {label}: Sharpe={sh:.3f}, KF={kf}/6")

    return {'candidates': candidates}


def main():
    print("=" * 120)
    print("  R190 — Selective Lot Reduction")
    print("=" * 120, flush=True)

    h1 = load_h1()
    atr_s = compute_atr(h1).dropna()
    print(f"  ATR: mean=${atr_s.mean():.2f}, median=${atr_s.median():.2f}, current=${atr_s.iloc[-1]:.2f}")

    p1 = phase1(h1)
    p2 = phase2(h1)
    p3 = phase3(h1)
    p4 = phase4(h1)
    p5 = phase5(h1)
    p6 = phase6(h1)
    p7 = phase7(h1)
    p8 = phase8(p1, p2, p3, p4, p5, p6, p7)

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    save = {'phase1':p1, 'phase2':p2, 'phase3':p3, 'phase4':p4,
            'phase5':p5, 'phase6':p6, 'phase7':p7, 'phase8':p8,
            'configs': {k: {s: v for s, v in lots.items()} for k, lots in CONFIGS.items()},
            'runtime_s': round(elapsed,1)}
    out = OUTPUT_DIR / "r190_results.json"
    with open(out, 'w') as f: json.dump(save, f, indent=2, default=str)
    print(f"  Saved: {out}")
    print(f"{'='*120}")

if __name__ == "__main__":
    main()
