#!/usr/bin/env python3
"""
R198b — K-Fold + Walk-Forward + Era Validation for two R198 candidates:
  1. Keltner EMA Trend Period: 100 (current) vs 125 (candidate)
  2. Keltner Cooldown: 2 (current) vs 3 (candidate)

Full validation suite: 6-Fold, 19-Window Walk-Forward, 4-Era segmentation.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import glob as _glob

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r198b_kfold_validate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

PV = 100; SPREAD = 0.30
CFG = {'lot': 0.04, 'cap': 70, 'sl': 3.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01, 'max_hold': 2}

def elapsed():
    return f"[{(time.time()-t0)/60:.1f}min]"

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

def bt_keltner(h1, cfg, pctl_v, pctl_f=30, adx_thr=14, ema_period=100, kc_ema=25, kc_mult=1.2, cooldown=2):
    df=h1.copy(); df['ATR']=compute_atr(df); df['ADX']=compute_adx(df)
    df['EMA_T']=df['Close'].ewm(span=ema_period,adjust=False).mean()
    df['KC_mid']=df['Close'].ewm(span=kc_ema,adjust=False).mean()
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
        if i-le<cooldown: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pv_a is not None and (np.isnan(pv_a[i]) or pv_a[i]<pctl_f): continue
        if np.isnan(adx[i]) or adx[i]<adx_thr: continue
        if c[i]>ku[i] and c[i]>ema[i]:
            pos={'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<kl[i] and c[i]<ema[i]:
            pos={'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades

# ═══════════════ Load Data ═══════════════
print(f"{'='*100}")
print(f"  R198b — K-Fold + Walk-Forward + Era Validation")
print(f"  Candidate 1: EMA Period 100 → 125")
print(f"  Candidate 2: Cooldown 2 → 3")
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

ERAS = {
    'Pre-COVID (2015-2019)': ('2015-01-01', '2019-12-31'),
    'COVID (2020-2021)': ('2020-01-01', '2021-12-31'),
    'Hike (2022-2023)': ('2022-01-01', '2023-12-31'),
    'Recent (2024-2026)': ('2024-01-01', '2026-12-31'),
}

# ═══════════════ Test both candidates ═══════════════
test_configs = [
    {
        'name': 'EMA Period 100→125',
        'current_kwargs': {'ema_period': 100, 'cooldown': 2},
        'new_kwargs': {'ema_period': 125, 'cooldown': 2},
    },
    {
        'name': 'Cooldown 2→3',
        'current_kwargs': {'ema_period': 100, 'cooldown': 2},
        'new_kwargs': {'ema_period': 100, 'cooldown': 3},
    },
    {
        'name': 'EMA 125 + Cooldown 3 (combined)',
        'current_kwargs': {'ema_period': 100, 'cooldown': 2},
        'new_kwargs': {'ema_period': 125, 'cooldown': 3},
    },
]

all_results = {}

for tc in test_configs:
    name = tc['name']
    cur_kw = tc['current_kwargs']
    new_kw = tc['new_kwargs']
    print(f"\n{'─'*100}")
    print(f"  Testing: {name}")
    print(f"  Current: {cur_kw}  →  New: {new_kw}")
    print(f"{'─'*100}")

    result = {'name': name, 'current': cur_kw, 'new': new_kw}

    # --- Full Sample ---
    t_cur = bt_keltner(h1, CFG, pctl_full, **cur_kw)
    t_new = bt_keltner(h1, CFG, pctl_full, **new_kw)
    s_cur = _stats(t_cur); s_new = _stats(t_new)
    print(f"\n  Full Sample:")
    print(f"    Current: Sharpe={s_cur['sharpe']:.3f}  PnL=${s_cur['pnl']:,.0f}  N={s_cur['n']}  WR={s_cur['wr']}%  MaxDD=${s_cur['max_dd']:,.0f}")
    print(f"    New:     Sharpe={s_new['sharpe']:.3f}  PnL=${s_new['pnl']:,.0f}  N={s_new['n']}  WR={s_new['wr']}%  MaxDD=${s_new['max_dd']:,.0f}")
    print(f"    Delta:   Sharpe={s_new['sharpe']-s_cur['sharpe']:+.3f}  PnL=${s_new['pnl']-s_cur['pnl']:+,.0f}")
    result['full_sample'] = {'current': s_cur, 'new': s_new}

    # --- 6-Fold K-Fold ---
    print(f"\n  6-Fold Cross-Validation:")
    n_folds = 6; fold_size = len(h1) // n_folds; wins = 0; fold_details = []
    for fold in range(n_folds):
        fs = fold * fold_size; fe = min((fold+1)*fold_size, len(h1))
        h1_f = h1.iloc[fs:fe]; pctl_f = pctl_full.reindex(h1_f.index)
        tc_f = bt_keltner(h1_f, CFG, pctl_f, **cur_kw)
        tn_f = bt_keltner(h1_f, CFG, pctl_f, **new_kw)
        sc = _stats(tc_f)['sharpe']; sn = _stats(tn_f)['sharpe']
        delta = round(sn - sc, 3)
        win = 1 if sn > sc else 0; wins += win
        fold_details.append({'fold': fold, 'cur': sc, 'new': sn, 'delta': delta, 'win': bool(win)})
        marker = "WIN" if win else "LOSE"
        print(f"    Fold {fold}: cur={sc:.3f} new={sn:.3f} delta={delta:+.3f} [{marker}]")
    kf_pass = wins >= 4
    verdict = "PASS" if kf_pass else "FAIL"
    print(f"    K-Fold: {wins}/6 [{verdict}]")
    result['kfold'] = {'wins': wins, 'passed': kf_pass, 'folds': fold_details}

    # --- 19-Window Walk-Forward ---
    print(f"\n  19-Window Walk-Forward:")
    n_windows = 19; train_pct = 0.7; step = len(h1) // (n_windows + 5)
    wf_wins = 0; wf_details = []
    for w in range(n_windows):
        ws = w * step; we = min(ws + int(step * (1/0.3)), len(h1))
        if we > len(h1): we = len(h1)
        total = we - ws; train_end = ws + int(total * train_pct)
        h1_oos = h1.iloc[train_end:we]; pctl_oos = pctl_full.reindex(h1_oos.index)
        if len(h1_oos) < 200: continue
        tc_w = bt_keltner(h1_oos, CFG, pctl_oos, **cur_kw)
        tn_w = bt_keltner(h1_oos, CFG, pctl_oos, **new_kw)
        sc = _stats(tc_w)['sharpe']; sn = _stats(tn_w)['sharpe']
        win = 1 if sn > sc else 0; wf_wins += win
        wf_details.append({'window': w, 'cur': sc, 'new': sn, 'delta': round(sn-sc, 3), 'win': bool(win)})
    wf_total = len(wf_details)
    wf_pass = wf_wins >= int(wf_total * 0.65)
    verdict = "PASS" if wf_pass else "FAIL"
    print(f"    Walk-Forward: {wf_wins}/{wf_total} OOS wins [{verdict}]")
    for wd in wf_details:
        marker = "WIN" if wd['win'] else "LOSE"
        print(f"      W{wd['window']:>2}: cur={wd['cur']:.3f} new={wd['new']:.3f} delta={wd['delta']:+.3f} [{marker}]")
    result['walk_forward'] = {'wins': wf_wins, 'total': wf_total, 'passed': wf_pass, 'windows': wf_details}

    # --- Era Segmentation ---
    print(f"\n  Era Segmentation:")
    era_details = {}
    all_eras_positive = True
    for era_name, (start, end) in ERAS.items():
        h1_e = h1[(h1.index >= start) & (h1.index <= end)]
        if len(h1_e) < 500: continue
        pctl_e = pctl_full.reindex(h1_e.index)
        tc_e = bt_keltner(h1_e, CFG, pctl_e, **cur_kw)
        tn_e = bt_keltner(h1_e, CFG, pctl_e, **new_kw)
        sc = _stats(tc_e); sn = _stats(tn_e)
        delta_sh = round(sn['sharpe'] - sc['sharpe'], 3)
        era_details[era_name] = {'current': sc, 'new': sn, 'delta_sharpe': delta_sh}
        if delta_sh < -0.3: all_eras_positive = False
        marker = "OK" if delta_sh > 0 else ("~" if delta_sh > -0.3 else "BAD")
        print(f"    {era_name:<25} cur={sc['sharpe']:.3f} new={sn['sharpe']:.3f} delta={delta_sh:+.3f} [{marker}]")
    era_pass = all_eras_positive
    verdict = "PASS" if era_pass else "FAIL"
    print(f"    Era Check: {'All eras OK' if era_pass else 'Degradation >0.3 detected'} [{verdict}]")
    result['era'] = {'passed': era_pass, 'eras': era_details}

    # --- Overall Verdict ---
    overall = kf_pass and wf_pass and era_pass
    result['overall'] = 'GO' if overall else 'NO-GO'
    print(f"\n  {'='*60}")
    print(f"  VERDICT: {result['overall']}")
    print(f"    K-Fold:       {wins}/6 {'PASS' if kf_pass else 'FAIL'}")
    print(f"    Walk-Forward: {wf_wins}/{wf_total} {'PASS' if wf_pass else 'FAIL'}")
    print(f"    Era:          {'PASS' if era_pass else 'FAIL'}")
    print(f"  {'='*60}")

    all_results[name] = result

# ═══════════════ Save & Summary ═══════════════
with open(OUTPUT_DIR / "r198b_results.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*100}")
print(f"  R198b FINAL SUMMARY")
print(f"{'='*100}")
for name, r in all_results.items():
    kf = r['kfold']
    wf = r['walk_forward']
    era = r['era']
    fs = r['full_sample']
    print(f"\n  {name}:")
    print(f"    Full: Sharpe {fs['current']['sharpe']:.3f} → {fs['new']['sharpe']:.3f} ({fs['new']['sharpe']-fs['current']['sharpe']:+.3f})")
    print(f"    K-Fold: {kf['wins']}/6  WF: {wf['wins']}/{wf['total']}  Era: {'PASS' if era['passed'] else 'FAIL'}")
    print(f"    VERDICT: {r['overall']}")

print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} min")
print(f"{'='*100}")
