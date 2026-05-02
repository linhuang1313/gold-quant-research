#!/usr/bin/env python3
"""
R86 — Risk Parity Portfolio Optimization
==========================================
Compare portfolio allocation methods for PSAR + TSMOM + SESS_BO:

  1. Equal Weight (current: each strategy runs independently with 0.03 lot)
  2. Inverse Volatility (lot_i = base / vol_i, normalized)
  3. Risk Parity / Equal Risk Contribution (ERC)
  4. Max Sharpe (mean-variance optimal)

Also includes:
  - Correlation analysis between strategies
  - Combined equity curves
  - Drawdown comparison
  - Crisis Alpha analysis (does portfolio perform better in crises?)
  - Yearly breakdown

Estimated runtime: ~3-5 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r86_risk_parity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
BASE_LOT = 0.03
PV = 100
ACCOUNT = 5000


def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_p, exit_time, reason, bar_idx, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_p,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_idx - pos['bar']}


def _trades_to_daily(trades):
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return pd.Series(dtype=float)
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def _sharpe(arr):
    if len(arr) < 10 or arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _calmar(arr):
    if len(arr) == 0:
        return 0.0
    dd = _max_dd(arr)
    ann = float(arr.mean()) * 252
    return ann / dd if dd > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (PSAR, TSMOM, SESS_BO)
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0,i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def backtest_psar(h1_df, spread=SPREAD, lot=BASE_LOT,
                  sl_atr=4.5, tp_atr=16.0, trail_act_atr=0.20,
                  trail_dist_atr=0.04, max_hold=20):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    pdir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir']=='BUY':
                ph=(h[i]-pos['entry']-spread)*lot*PV; pl=(lo[i]-pos['entry']-spread)*lot*PV; pc=(c[i]-pos['entry']-spread)*lot*PV
            else:
                ph=(pos['entry']-lo[i]-spread)*lot*PV; pl=(pos['entry']-h[i]-spread)*lot*PV; pc=(pos['entry']-c[i]-spread)*lot*PV
            tp_v=tp_atr*pos['atr']*lot*PV; sl_v=sl_atr*pos['atr']*lot*PV
            ex=False
            if ph>=tp_v: trades.append(_mk(pos,c[i],times[i],"TP",i,tp_v)); ex=True
            elif pl<=-sl_v: trades.append(_mk(pos,c[i],times[i],"SL",i,-sl_v)); ex=True
            else:
                ad=trail_act_atr*pos['atr']; td=trail_dist_atr*pos['atr']
                if pos['dir']=='BUY' and h[i]-pos['entry']>=ad:
                    ts=h[i]-td
                    if lo[i]<=ts: trades.append(_mk(pos,c[i],times[i],"Trail",i,(ts-pos['entry']-spread)*lot*PV)); ex=True
                elif pos['dir']=='SELL' and pos['entry']-lo[i]>=ad:
                    ts=lo[i]+td
                    if h[i]>=ts: trades.append(_mk(pos,c[i],times[i],"Trail",i,(pos['entry']-ts-spread)*lot*PV)); ex=True
                if not ex and held>=max_hold: trades.append(_mk(pos,c[i],times[i],"Timeout",i,pc)); ex=True
            if ex: pos=None; last_exit=i; continue
            continue
        if i-last_exit<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if pdir[i-1]==-1 and pdir[i]==1:
            pos={'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif pdir[i-1]==1 and pdir[i]==-1:
            pos={'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


def backtest_tsmom(h1_df, spread=SPREAD, lot=BASE_LOT,
                   fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fma'] = df['Close'].rolling(fast).mean()
    df['sma'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR','fma','sma'])
    c=df['Close'].values; h=df['High'].values; lo=df['Low'].values
    atr=df['ATR'].values; fm=df['fma'].values; sm=df['sma'].values
    times=df.index; n=len(df)
    trades=[]; pos=None; last_exit=-999
    for i in range(1,n):
        if pos is not None:
            held=i-pos['bar']
            if pos['dir']=='BUY':
                ph=(h[i]-pos['entry']-spread)*lot*PV; pl=(lo[i]-pos['entry']-spread)*lot*PV; pc=(c[i]-pos['entry']-spread)*lot*PV
            else:
                ph=(pos['entry']-lo[i]-spread)*lot*PV; pl=(pos['entry']-h[i]-spread)*lot*PV; pc=(pos['entry']-c[i]-spread)*lot*PV
            tp_v=tp_atr*pos['atr']*lot*PV; sl_v=sl_atr*pos['atr']*lot*PV
            ex=False
            if ph>=tp_v: trades.append(_mk(pos,c[i],times[i],"TP",i,tp_v)); ex=True
            elif pl<=-sl_v: trades.append(_mk(pos,c[i],times[i],"SL",i,-sl_v)); ex=True
            else:
                ad=trail_act*pos['atr']; td_v=trail_dist*pos['atr']
                if pos['dir']=='BUY' and h[i]-pos['entry']>=ad:
                    ts=h[i]-td_v
                    if lo[i]<=ts: trades.append(_mk(pos,c[i],times[i],"Trail",i,(ts-pos['entry']-spread)*lot*PV)); ex=True
                elif pos['dir']=='SELL' and pos['entry']-lo[i]>=ad:
                    ts=lo[i]+td_v
                    if h[i]>=ts: trades.append(_mk(pos,c[i],times[i],"Trail",i,(pos['entry']-ts-spread)*lot*PV)); ex=True
                if not ex and held>=max_hold: trades.append(_mk(pos,c[i],times[i],"Timeout",i,pc)); ex=True
            if ex: pos=None; last_exit=i; continue
            if (pos['dir']=='BUY' and fm[i]<sm[i]) or (pos['dir']=='SELL' and fm[i]>sm[i]):
                if pos['dir']=='BUY': pnl=(c[i]-pos['entry']-spread)*lot*PV
                else: pnl=(pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos,c[i],times[i],"Reversal",i,pnl)); pos=None; last_exit=i; continue
            continue
        if i-last_exit<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        if fm[i]>sm[i] and fm[i-1]<=sm[i-1]:
            pos={'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif fm[i]<sm[i] and fm[i-1]>=sm[i-1]:
            pos={'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


def backtest_sess_bo(h1_df, spread=SPREAD, lot=BASE_LOT,
                     session_hour=12, lookback=4, atr_period=14,
                     sl_atr=4.5, tp_atr=4.0, max_hold=20,
                     trail_act=0.14, trail_dist=0.025):
    df = h1_df.copy(); df['ATR'] = compute_atr(df, atr_period)
    df = df.dropna(subset=['ATR'])
    c=df['Close'].values; h=df['High'].values; lo=df['Low'].values
    atr=df['ATR'].values; hours=df.index.hour; times=df.index; n=len(df)
    trades=[]; pos=None; last_exit=-999
    for i in range(lookback, n):
        if pos is not None:
            held=i-pos['bar']; lt=lot
            if pos['dir']=='BUY':
                pc=(c[i]-pos['entry']-spread)*lt*PV
            else:
                pc=(pos['entry']-c[i]-spread)*lt*PV
            tp_v=tp_atr*pos['atr']*lt*PV; sl_v=sl_atr*pos['atr']*lt*PV
            ex=False
            if pc>=tp_v: trades.append(_mk(pos,c[i],times[i],"TP",i,tp_v)); ex=True
            elif pc<=-sl_v: trades.append(_mk(pos,c[i],times[i],"SL",i,-sl_v)); ex=True
            else:
                ad=trail_act*pos['atr']; td_v=trail_dist*pos['atr']
                if pos['dir']=='BUY' and h[i]-pos['entry']>=ad:
                    ts=h[i]-td_v
                    if lo[i]<=ts: trades.append(_mk(pos,c[i],times[i],"Trail",i,(ts-pos['entry']-spread)*lt*PV)); ex=True
                elif pos['dir']=='SELL' and pos['entry']-lo[i]>=ad:
                    ts=lo[i]+td_v
                    if h[i]>=ts: trades.append(_mk(pos,c[i],times[i],"Trail",i,(pos['entry']-ts-spread)*lt*PV)); ex=True
                if not ex and held>=max_hold: trades.append(_mk(pos,c[i],times[i],"Timeout",i,pc)); ex=True
            if ex: pos=None; last_exit=i; continue
            continue
        if hours[i]!=session_hour: continue
        if i-last_exit<2: continue
        if np.isnan(atr[i]) or atr[i]<0.1: continue
        hh=max(h[i-j] for j in range(1,lookback+1))
        ll=min(lo[i-j] for j in range(1,lookback+1))
        if c[i]>hh:
            pos={'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i]<ll:
            pos={'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Portfolio allocation methods
# ═══════════════════════════════════════════════════════════════

def equal_weight(n):
    return np.ones(n) / n


def inverse_volatility(daily_returns_list):
    """Weight inversely proportional to volatility."""
    vols = np.array([s.std() for s in daily_returns_list])
    vols = np.where(vols == 0, 1e-10, vols)
    inv = 1.0 / vols
    return inv / inv.sum()


def risk_parity_weights(daily_returns_list):
    """Equal Risk Contribution (ERC) portfolio.

    Each strategy contributes equally to total portfolio risk.
    Uses scipy optimization.
    """
    n = len(daily_returns_list)
    # Align all series to common dates
    all_dates = set()
    for s in daily_returns_list:
        all_dates.update(s.index)
    all_dates = sorted(all_dates)
    aligned = np.zeros((len(all_dates), n))
    for j, s in enumerate(daily_returns_list):
        for k, d in enumerate(all_dates):
            if d in s.index:
                aligned[k, j] = s.loc[d]

    cov = np.cov(aligned.T)
    if np.any(np.isnan(cov)):
        return equal_weight(n)

    def risk_contrib_obj(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 1e10
        marginal = cov @ w
        rc = w * marginal / port_vol
        target = port_vol / n
        return np.sum((rc - target) ** 2)

    w0 = equal_weight(n)
    bounds = [(0.05, 0.80)] * n
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    result = minimize(risk_contrib_obj, w0, method='SLSQP',
                      bounds=bounds, constraints=cons,
                      options={'maxiter': 1000, 'ftol': 1e-12})
    if result.success:
        return result.x / result.x.sum()
    return equal_weight(n)


def max_sharpe_weights(daily_returns_list):
    """Mean-variance optimal (max Sharpe) weights."""
    n = len(daily_returns_list)
    all_dates = set()
    for s in daily_returns_list:
        all_dates.update(s.index)
    all_dates = sorted(all_dates)
    aligned = np.zeros((len(all_dates), n))
    for j, s in enumerate(daily_returns_list):
        for k, d in enumerate(all_dates):
            if d in s.index:
                aligned[k, j] = s.loc[d]

    mu = aligned.mean(axis=0)
    cov = np.cov(aligned.T)
    if np.any(np.isnan(cov)):
        return equal_weight(n)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 0
        return -(ret / vol)

    w0 = equal_weight(n)
    bounds = [(0.05, 0.80)] * n
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    result = minimize(neg_sharpe, w0, method='SLSQP',
                      bounds=bounds, constraints=cons)
    if result.success:
        return result.x / result.x.sum()
    return equal_weight(n)


# ═══════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  R86 — Risk Parity Portfolio Optimization")
    print("  Strategies: PSAR + TSMOM + SESS_BO")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    strat_names = ['PSAR', 'TSMOM', 'SESS_BO']

    # Run each strategy at both spread levels
    print("  Running individual strategy backtests...", flush=True)
    all_trades = {}
    all_trades_real = {}
    for sp_label, sp_val in [('ideal', SPREAD), ('real', REALISTIC_SPREAD)]:
        for name, bt_fn in [
            ('PSAR', lambda df, sp, lt: backtest_psar(df, sp, lt)),
            ('TSMOM', lambda df, sp, lt: backtest_tsmom(df, sp, lt)),
            ('SESS_BO', lambda df, sp, lt: backtest_sess_bo(df, sp, lt)),
        ]:
            trades = bt_fn(h1_df, sp_val, BASE_LOT)
            key = f"{name}_{sp_label}"
            if sp_label == 'ideal':
                all_trades[name] = trades
            else:
                all_trades_real[name] = trades
            daily = _trades_to_daily(trades)
            print(f"    {key}: {len(trades)} trades, Sharpe={_sharpe(daily.values):.2f}, "
                  f"PnL=${daily.sum():,.0f}", flush=True)

    # Build daily return series (aligned to common date range)
    daily_series = {}
    daily_series_real = {}
    for name in strat_names:
        daily_series[name] = _trades_to_daily(all_trades[name])
        daily_series_real[name] = _trades_to_daily(all_trades_real[name])

    ds_list = [daily_series[n] for n in strat_names]
    ds_list_real = [daily_series_real[n] for n in strat_names]

    # ── Correlation analysis ──
    print(f"\n  === Correlation Analysis ===", flush=True)
    all_dates = sorted(set().union(*[set(s.index) for s in ds_list]))
    corr_matrix = np.zeros((3, 3))
    aligned = np.zeros((len(all_dates), 3))
    for j, s in enumerate(ds_list):
        for k, d in enumerate(all_dates):
            if d in s.index:
                aligned[k, j] = s.loc[d]
    corr_matrix = np.corrcoef(aligned.T)
    print(f"    Correlation matrix (ideal spread):")
    for i, ni in enumerate(strat_names):
        row = "    " + f"  {ni:>8}: " + "  ".join(f"{corr_matrix[i,j]:+.3f}" for j in range(3))
        print(row, flush=True)

    # ── Compute weights for each method ──
    print(f"\n  === Portfolio Weights ===", flush=True)
    methods = {
        'Equal_Weight': equal_weight(3),
        'Inv_Volatility': inverse_volatility(ds_list),
        'Risk_Parity': risk_parity_weights(ds_list),
        'Max_Sharpe': max_sharpe_weights(ds_list),
    }

    for method_name, weights in methods.items():
        w_str = " / ".join(f"{strat_names[i]}={weights[i]:.1%}" for i in range(3))
        print(f"    {method_name:>20}: {w_str}", flush=True)

    # ── Build portfolio returns for each method ──
    print(f"\n  === Portfolio Performance ===", flush=True)
    results = {}
    total_base_lot = BASE_LOT * 3  # total lot budget across 3 strategies

    for method_name, weights in methods.items():
        for sp_label, ds_l in [('ideal', ds_list), ('real', ds_list_real)]:
            # Weighted sum: scale each strategy's daily PnL by weight
            # weight=1/3 means same as current (equal), weight>1/3 means more allocation
            port_daily = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
            for j, name in enumerate(strat_names):
                scale = weights[j] / (1.0 / 3.0)  # relative to equal weight
                s = ds_l[j]
                for d in s.index:
                    if d in port_daily.index:
                        port_daily.loc[d] += s.loc[d] * scale

            arr = port_daily.values
            sh = _sharpe(arr)
            pnl = float(arr.sum())
            dd = _max_dd(arr)
            cal = _calmar(arr)
            n_days = int((arr != 0).sum())

            # Yearly breakdown
            yearly = {}
            for d, v in zip(port_daily.index, arr):
                yr = d.year
                yearly.setdefault(yr, []).append(v)
            yearly_stats = {}
            for yr in sorted(yearly.keys()):
                ya = np.array(yearly[yr])
                yearly_stats[yr] = {
                    'pnl': round(float(ya.sum()), 2),
                    'sharpe': round(_sharpe(ya), 2),
                }

            key = f"{method_name}_{sp_label}"
            results[key] = {
                'method': method_name,
                'spread': sp_label,
                'weights': {strat_names[i]: round(float(weights[i]), 4) for i in range(3)},
                'sharpe': round(sh, 2),
                'pnl': round(pnl, 2),
                'max_dd': round(dd, 2),
                'calmar': round(cal, 3),
                'active_days': n_days,
                'dd_pct_account': round(dd / ACCOUNT * 100, 1),
                'yearly': yearly_stats,
            }
            if sp_label == 'ideal':
                print(f"    {method_name:>20} ({sp_label}): Sharpe={sh:.2f} "
                      f"PnL=${pnl:,.0f} DD=${dd:,.0f} ({dd/ACCOUNT*100:.1f}%) "
                      f"Calmar={cal:.3f}", flush=True)
            else:
                print(f"    {method_name:>20} (real  ): Sharpe={sh:.2f} "
                      f"PnL=${pnl:,.0f} DD=${dd:,.0f} ({dd/ACCOUNT*100:.1f}%) "
                      f"Calmar={cal:.3f}", flush=True)

    # ── Individual strategy stats for reference ──
    print(f"\n  === Individual Strategy Stats ===", flush=True)
    indiv = {}
    for sp_label, ds_l in [('ideal', ds_list), ('real', ds_list_real)]:
        for j, name in enumerate(strat_names):
            arr = ds_l[j].values
            sh = _sharpe(arr)
            pnl = float(arr.sum())
            dd = _max_dd(arr)
            cal = _calmar(arr)
            key = f"{name}_{sp_label}"
            indiv[key] = {
                'sharpe': round(sh, 2), 'pnl': round(pnl, 2),
                'max_dd': round(dd, 2), 'calmar': round(cal, 3),
                'n_trades': len(all_trades[name] if sp_label == 'ideal' else all_trades_real[name]),
            }
            print(f"    {key:>20}: Sharpe={sh:.2f} PnL=${pnl:,.0f} "
                  f"DD=${dd:,.0f} Calmar={cal:.3f}", flush=True)

    # ── Crisis Alpha: performance during high-vol periods ──
    print(f"\n  === Crisis Alpha Analysis ===", flush=True)
    # Identify high-volatility months (top 20% by gold price range)
    monthly_range = {}
    for i in range(len(h1_df)):
        ym = f"{h1_df.index[i].year}-{h1_df.index[i].month:02d}"
        monthly_range.setdefault(ym, []).append(h1_df['High'].iloc[i] - h1_df['Low'].iloc[i])
    monthly_avg_range = {ym: np.mean(ranges) for ym, ranges in monthly_range.items()}
    threshold = np.percentile(list(monthly_avg_range.values()), 80)
    crisis_months = {ym for ym, r in monthly_avg_range.items() if r >= threshold}
    print(f"    High-vol months (top 20%): {len(crisis_months)} months, "
          f"threshold avg range > {threshold:.1f}", flush=True)

    crisis_results = {}
    for method_name, weights in methods.items():
        port_daily = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
        for j, name in enumerate(strat_names):
            scale = weights[j] / (1.0 / 3.0)
            s = ds_list[j]
            for d in s.index:
                if d in port_daily.index:
                    port_daily.loc[d] += s.loc[d] * scale

        crisis_pnl = 0; normal_pnl = 0; crisis_days = 0; normal_days = 0
        for d, v in zip(port_daily.index, port_daily.values):
            ym = f"{d.year}-{d.month:02d}"
            if ym in crisis_months:
                crisis_pnl += v; crisis_days += 1
            else:
                normal_pnl += v; normal_days += 1

        crisis_results[method_name] = {
            'crisis_pnl': round(crisis_pnl, 2),
            'normal_pnl': round(normal_pnl, 2),
            'crisis_pnl_per_day': round(crisis_pnl / max(crisis_days, 1), 2),
            'normal_pnl_per_day': round(normal_pnl / max(normal_days, 1), 2),
        }
        print(f"    {method_name:>20}: Crisis PnL/day=${crisis_pnl/max(crisis_days,1):.2f} "
              f"Normal PnL/day=${normal_pnl/max(normal_days,1):.2f} "
              f"Ratio={crisis_pnl/max(crisis_days,1)/(normal_pnl/max(normal_days,1)+1e-10):.2f}x",
              flush=True)

    # ── Lot allocation recommendation ──
    print(f"\n  === Lot Allocation for ${ACCOUNT:,} Account ===", flush=True)
    rp_w = methods['Risk_Parity']
    print(f"    Risk Parity weights: {' / '.join(f'{strat_names[i]}={rp_w[i]:.1%}' for i in range(3))}")
    total_lot = BASE_LOT * 3  # 0.09 total
    for i, name in enumerate(strat_names):
        lot_i = round(total_lot * rp_w[i], 2)
        lot_i = max(0.01, lot_i)
        print(f"    {name:>8}: {rp_w[i]:.1%} -> {lot_i:.2f} lot", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  R86 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 72}")

    combined = {
        'correlation_matrix': {
            f"{strat_names[i]}_{strat_names[j]}": round(float(corr_matrix[i, j]), 4)
            for i in range(3) for j in range(3)
        },
        'weights': {m: {strat_names[i]: round(float(w[i]), 4) for i in range(3)}
                    for m, w in methods.items()},
        'portfolio_results': results,
        'individual': indiv,
        'crisis_alpha': crisis_results,
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "r86_results.json", 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
