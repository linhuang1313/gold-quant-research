#!/usr/bin/env python3
"""
R87 — Five Advanced Risk Research Directions
=============================================
R87-A: CVaR integration (uses stats.py compute_risk_metrics)
R87-B: Strategy health monitor (rolling Sharpe/WR decay, CUSUM change-point)
R87-C: Crowding proxy (volume surge, signal clustering, correlation, spread)
R87-D: Dynamic Half-Kelly (60-trade rolling window, per-strategy)
R87-E: Factor decomposition / return attribution (OLS, rolling alpha)

Estimated runtime: ~10-15 minutes on server.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r87_advanced_risk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
BASE_LOT = 0.03
PV = 100
ACCOUNT = 5000


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

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
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return pd.Series(dtype=float)
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def _sharpe(arr, ann=252):
    if len(arr) < 10:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr, ddof=1)
    if s == 0:
        return 0.0
    return float(m / s * np.sqrt(ann))


def _max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def _calmar(arr):
    if len(arr) == 0:
        return 0.0
    dd = _max_dd(arr)
    ann = float(np.mean(arr)) * 252
    return ann / dd if dd > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (PSAR, TSMOM, SESS_BO) — same as R86
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_start=0.01, af_max=0.05):
    df = df.copy(); n = len(df)
    psar = np.zeros(n); direction = np.ones(n)
    af = af_start; ep = df['High'].iloc[0]; psar[0] = df['Low'].iloc[0]
    for i in range(1, n):
        prev = psar[i-1]
        if direction[i-1] == 1:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], df['Low'].iloc[i-1], df['Low'].iloc[max(0, i-2)])
            if df['Low'].iloc[i] < psar[i]:
                direction[i] = -1; psar[i] = ep; ep = df['Low'].iloc[i]; af = af_start
            else:
                direction[i] = 1
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0, i-2)])
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
            if pos['dir'] == 'BUY':
                ph = (h[i]-pos['entry']-spread)*lot*PV; pl = (lo[i]-pos['entry']-spread)*lot*PV; pc = (c[i]-pos['entry']-spread)*lot*PV
            else:
                ph = (pos['entry']-lo[i]-spread)*lot*PV; pl = (pos['entry']-h[i]-spread)*lot*PV; pc = (pos['entry']-c[i]-spread)*lot*PV
            tp_v = tp_atr*pos['atr']*lot*PV; sl_v = sl_atr*pos['atr']*lot*PV
            ex = False
            if ph >= tp_v: trades.append(_mk(pos, c[i], times[i], "TP", i, tp_v)); ex = True
            elif pl <= -sl_v: trades.append(_mk(pos, c[i], times[i], "SL", i, -sl_v)); ex = True
            else:
                ad = trail_act_atr*pos['atr']; td = trail_dist_atr*pos['atr']
                if pos['dir'] == 'BUY' and h[i]-pos['entry'] >= ad:
                    ts = h[i]-td
                    if lo[i] <= ts: trades.append(_mk(pos, c[i], times[i], "Trail", i, (ts-pos['entry']-spread)*lot*PV)); ex = True
                elif pos['dir'] == 'SELL' and pos['entry']-lo[i] >= ad:
                    ts = lo[i]+td
                    if h[i] >= ts: trades.append(_mk(pos, c[i], times[i], "Trail", i, (pos['entry']-ts-spread)*lot*PV)); ex = True
                if not ex and held >= max_hold: trades.append(_mk(pos, c[i], times[i], "Timeout", i, pc)); ex = True
            if ex: pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if pdir[i-1] == -1 and pdir[i] == 1:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif pdir[i-1] == 1 and pdir[i] == -1:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def backtest_tsmom(h1_df, spread=SPREAD, lot=BASE_LOT,
                   fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df['fma'] = df['Close'].rolling(fast).mean()
    df['sma'] = df['Close'].rolling(slow).mean()
    df = df.dropna(subset=['ATR', 'fma', 'sma'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; fm = df['fma'].values; sm = df['sma'].values
    times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                ph = (h[i]-pos['entry']-spread)*lot*PV; pl = (lo[i]-pos['entry']-spread)*lot*PV; pc = (c[i]-pos['entry']-spread)*lot*PV
            else:
                ph = (pos['entry']-lo[i]-spread)*lot*PV; pl = (pos['entry']-h[i]-spread)*lot*PV; pc = (pos['entry']-c[i]-spread)*lot*PV
            tp_v = tp_atr*pos['atr']*lot*PV; sl_v = sl_atr*pos['atr']*lot*PV
            ex = False
            if ph >= tp_v: trades.append(_mk(pos, c[i], times[i], "TP", i, tp_v)); ex = True
            elif pl <= -sl_v: trades.append(_mk(pos, c[i], times[i], "SL", i, -sl_v)); ex = True
            else:
                ad = trail_act*pos['atr']; td_v = trail_dist*pos['atr']
                if pos['dir'] == 'BUY' and h[i]-pos['entry'] >= ad:
                    ts = h[i]-td_v
                    if lo[i] <= ts: trades.append(_mk(pos, c[i], times[i], "Trail", i, (ts-pos['entry']-spread)*lot*PV)); ex = True
                elif pos['dir'] == 'SELL' and pos['entry']-lo[i] >= ad:
                    ts = lo[i]+td_v
                    if h[i] >= ts: trades.append(_mk(pos, c[i], times[i], "Trail", i, (pos['entry']-ts-spread)*lot*PV)); ex = True
                if not ex and held >= max_hold: trades.append(_mk(pos, c[i], times[i], "Timeout", i, pc)); ex = True
            if ex: pos = None; last_exit = i; continue
            if (pos['dir'] == 'BUY' and fm[i] < sm[i]) or (pos['dir'] == 'SELL' and fm[i] > sm[i]):
                if pos['dir'] == 'BUY': pnl = (c[i]-pos['entry']-spread)*lot*PV
                else: pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if fm[i] > sm[i] and fm[i-1] <= sm[i-1]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif fm[i] < sm[i] and fm[i-1] >= sm[i-1]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def backtest_sess_bo(h1_df, spread=SPREAD, lot=BASE_LOT,
                     session_hour=12, lookback=4, atr_period=14,
                     sl_atr=4.5, tp_atr=4.0, max_hold=20,
                     trail_act=0.14, trail_dist=0.025):
    df = h1_df.copy(); df['ATR'] = compute_atr(df, atr_period)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            held = i - pos['bar']; lt = lot
            if pos['dir'] == 'BUY': pc = (c[i]-pos['entry']-spread)*lt*PV
            else: pc = (pos['entry']-c[i]-spread)*lt*PV
            tp_v = tp_atr*pos['atr']*lt*PV; sl_v = sl_atr*pos['atr']*lt*PV
            ex = False
            if pc >= tp_v: trades.append(_mk(pos, c[i], times[i], "TP", i, tp_v)); ex = True
            elif pc <= -sl_v: trades.append(_mk(pos, c[i], times[i], "SL", i, -sl_v)); ex = True
            else:
                ad = trail_act*pos['atr']; td_v = trail_dist*pos['atr']
                if pos['dir'] == 'BUY' and h[i]-pos['entry'] >= ad:
                    ts = h[i]-td_v
                    if lo[i] <= ts: trades.append(_mk(pos, c[i], times[i], "Trail", i, (ts-pos['entry']-spread)*lt*PV)); ex = True
                elif pos['dir'] == 'SELL' and pos['entry']-lo[i] >= ad:
                    ts = lo[i]+td_v
                    if h[i] >= ts: trades.append(_mk(pos, c[i], times[i], "Trail", i, (pos['entry']-ts-spread)*lt*PV)); ex = True
                if not ex and held >= max_hold: trades.append(_mk(pos, c[i], times[i], "Timeout", i, pc)); ex = True
            if ex: pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# R87-A: CVaR Integration (uses stats.py compute_risk_metrics)
# ═══════════════════════════════════════════════════════════════

def run_r87a(strat_dailies, portfolio_daily):
    """Compute CVaR and risk metrics for each strategy and portfolio."""
    print(f"\n{'#'*72}")
    print(f"  R87-A — CVaR / Tail Risk Metrics")
    print(f"{'#'*72}\n", flush=True)

    from backtest.stats import compute_risk_metrics

    results = {}
    for name, ds in strat_dailies.items():
        rm = compute_risk_metrics(ds.values.tolist())
        results[name] = rm
        print(f"  {name:>10}: VaR95=${rm['var_95']:>8.2f}  CVaR95=${rm['cvar_95']:>8.2f}  "
              f"VaR99=${rm['var_99']:>8.2f}  CVaR99=${rm['cvar_99']:>8.2f}  "
              f"Ulcer={rm['ulcer_index']:.4f}  MaxConsecLoss={rm['max_consec_loss_days']}d",
              flush=True)

    # Portfolio-level
    port_rm = compute_risk_metrics(portfolio_daily.values.tolist())
    results['PORTFOLIO'] = port_rm
    print(f"\n  {'PORTFOLIO':>10}: VaR95=${port_rm['var_95']:>8.2f}  CVaR95=${port_rm['cvar_95']:>8.2f}  "
          f"VaR99=${port_rm['var_99']:>8.2f}  CVaR99=${port_rm['cvar_99']:>8.2f}  "
          f"Ulcer={port_rm['ulcer_index']:.4f}  MaxConsecLoss={port_rm['max_consec_loss_days']}d",
          flush=True)

    # CVaR-based lot recommendation
    print(f"\n  CVaR-based lot sizing (target max daily loss = $100):")
    for name, rm in results.items():
        cvar99 = abs(rm['cvar_99'])
        if cvar99 > 0:
            max_lot = round(100 / cvar99 * BASE_LOT, 2)
            print(f"    {name:>10}: |CVaR99|=${cvar99:.2f} -> max_lot={max_lot:.2f}", flush=True)

    with open(OUTPUT_DIR / "r87a_cvar.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: r87a_cvar.json", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# R87-B: Strategy Health Monitor / Decay Detection
# ═══════════════════════════════════════════════════════════════

def run_r87b(strat_trades, strat_dailies, h1_df):
    """Rolling health checks + CUSUM change-point detection."""
    print(f"\n{'#'*72}")
    print(f"  R87-B — Strategy Health Monitor & Decay Detection")
    print(f"{'#'*72}\n", flush=True)

    ROLL_DAYS = 60
    results = {}

    for name, ds in strat_dailies.items():
        arr = ds.values
        dates = ds.index
        n = len(arr)
        full_sharpe = _sharpe(arr)
        full_wr_pnl = (arr > 0).mean() * 100

        trades = strat_trades[name]
        full_wr_trade = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100 if trades else 0

        # Rolling 60-day Sharpe
        roll_sharpes = []
        roll_dates = []
        for i in range(ROLL_DAYS, n):
            window = arr[i-ROLL_DAYS:i]
            sh = _sharpe(window)
            roll_sharpes.append(sh)
            roll_dates.append(str(dates[i].date()))

        # Rolling 60-day win rate (on daily PnL)
        roll_wr = []
        for i in range(ROLL_DAYS, n):
            window = arr[i-ROLL_DAYS:i]
            wr = (window > 0).mean() * 100
            roll_wr.append(wr)

        # Decay alerts: periods where rolling Sharpe < 50% of full-sample
        threshold_sharpe = full_sharpe * 0.5
        decay_periods = []
        in_decay = False
        decay_start = None
        for i, (sh, dt) in enumerate(zip(roll_sharpes, roll_dates)):
            if sh < threshold_sharpe and not in_decay:
                in_decay = True
                decay_start = dt
            elif sh >= threshold_sharpe and in_decay:
                in_decay = False
                decay_periods.append({'start': decay_start, 'end': dt,
                                      'min_sharpe': round(min(roll_sharpes[max(0,i-30):i+1]), 2)})
        if in_decay:
            decay_periods.append({'start': decay_start, 'end': roll_dates[-1],
                                  'min_sharpe': round(min(roll_sharpes[-30:]), 2)})

        # CUSUM change-point detection on daily returns
        cusum_results = _cusum_detect(arr)

        # Consecutive loss days tracking
        max_consec = 0; streak = 0
        worst_streak_end = None
        for i, v in enumerate(arr):
            if v < 0:
                streak += 1
                if streak > max_consec:
                    max_consec = streak
                    worst_streak_end = str(dates[i].date())
            else:
                streak = 0

        # Rolling 20-day PnL sign test
        sign_test_fails = 0
        for i in range(20, n):
            window = arr[i-20:i]
            if window.sum() < 0:
                sign_test_fails += 1

        # Pause-test: simulate pausing during decay periods
        pause_pnl = 0.0; trade_pnl = 0.0
        for i, v in enumerate(arr):
            dt_str = str(dates[i].date())
            paused = any(p['start'] <= dt_str <= p['end'] for p in decay_periods)
            if paused:
                pause_pnl += 0  # skip
            else:
                pause_pnl += v
            trade_pnl += v

        results[name] = {
            'full_sharpe': round(full_sharpe, 2),
            'full_wr_daily': round(full_wr_pnl, 1),
            'full_wr_trade': round(full_wr_trade, 1),
            'n_decay_periods': len(decay_periods),
            'decay_periods': decay_periods[:10],  # top 10
            'decay_threshold_sharpe': round(threshold_sharpe, 2),
            'cusum': cusum_results,
            'max_consecutive_loss_days': max_consec,
            'worst_streak_end': worst_streak_end,
            'sign_test_20d_fail_days': sign_test_fails,
            'sign_test_20d_fail_pct': round(sign_test_fails / max(n-20, 1) * 100, 1),
            'pause_test': {
                'total_pnl_no_pause': round(trade_pnl, 2),
                'total_pnl_with_pause': round(pause_pnl, 2),
                'delta': round(pause_pnl - trade_pnl, 2),
                'pause_helps': pause_pnl > trade_pnl,
            },
            'health_summary': _health_score(full_sharpe, roll_sharpes, max_consec, sign_test_fails, n),
        }

        hs = results[name]['health_summary']
        print(f"  {name:>10}: FullSharpe={full_sharpe:.2f}  "
              f"Decays={len(decay_periods)}  CUSUM_breaks={cusum_results['n_changepoints']}  "
              f"MaxConsecLoss={max_consec}d  HealthScore={hs['score']}/{hs['max_score']}  "
              f"Pause {'HELPS' if results[name]['pause_test']['pause_helps'] else 'HURTS'} "
              f"(delta=${results[name]['pause_test']['delta']:+.0f})", flush=True)

    with open(OUTPUT_DIR / "r87b_health.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: r87b_health.json", flush=True)
    return results


def _cusum_detect(arr, drift=0.5):
    """CUSUM change-point detection on daily PnL."""
    n = len(arr)
    if n < 20:
        return {'n_changepoints': 0, 'changepoints': []}

    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return {'n_changepoints': 0, 'changepoints': []}

    z = (arr - mu) / sigma
    threshold = 4.0  # standard CUSUM threshold

    s_pos = np.zeros(n); s_neg = np.zeros(n)
    changepoints = []
    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i-1] + z[i] - drift)
        s_neg[i] = max(0, s_neg[i-1] - z[i] - drift)
        if s_pos[i] > threshold:
            changepoints.append({'index': int(i), 'direction': 'positive_shift', 'cusum': round(float(s_pos[i]), 2)})
            s_pos[i] = 0
        if s_neg[i] > threshold:
            changepoints.append({'index': int(i), 'direction': 'negative_shift', 'cusum': round(float(s_neg[i]), 2)})
            s_neg[i] = 0

    return {
        'n_changepoints': len(changepoints),
        'changepoints': changepoints[:20],  # limit output
    }


def _health_score(full_sharpe, roll_sharpes, max_consec, sign_fails, n):
    """Composite health score (0-5)."""
    score = 0; max_score = 5
    if full_sharpe >= 2.0: score += 1
    if len(roll_sharpes) > 0 and min(roll_sharpes) > 0: score += 1
    if max_consec <= 5: score += 1
    if sign_fails / max(n-20, 1) < 0.15: score += 1
    recent = roll_sharpes[-60:] if len(roll_sharpes) >= 60 else roll_sharpes
    if recent and np.mean(recent) > full_sharpe * 0.7: score += 1
    labels = {0: 'CRITICAL', 1: 'POOR', 2: 'FAIR', 3: 'GOOD', 4: 'STRONG', 5: 'EXCELLENT'}
    return {'score': score, 'max_score': max_score, 'label': labels.get(score, 'UNKNOWN')}


# ═══════════════════════════════════════════════════════════════
# R87-C: Crowding Proxy Indicators
# ═══════════════════════════════════════════════════════════════

def run_r87c(strat_trades, strat_dailies, h1_df):
    """Crowding proxy analysis: volume, signal clustering, correlation, spread."""
    print(f"\n{'#'*72}")
    print(f"  R87-C — Strategy Crowding Proxy Indicators")
    print(f"{'#'*72}\n", flush=True)

    results = {}

    # 1. Volume surge analysis
    print("  [1/4] Volume surge vs strategy returns...", flush=True)
    vol_result = _volume_surge_analysis(strat_trades, strat_dailies, h1_df)
    results['volume_surge'] = vol_result

    # 2. Signal clustering
    print("  [2/4] Signal clustering analysis...", flush=True)
    cluster_result = _signal_clustering(strat_trades)
    results['signal_clustering'] = cluster_result

    # 3. Rolling correlation between strategies
    print("  [3/4] Rolling inter-strategy correlation...", flush=True)
    corr_result = _rolling_correlation(strat_dailies)
    results['rolling_correlation'] = corr_result

    # 4. Spread sensitivity
    print("  [4/4] Spread sensitivity analysis...", flush=True)
    spread_result = _spread_sensitivity(h1_df)
    results['spread_sensitivity'] = spread_result

    with open(OUTPUT_DIR / "r87c_crowding.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: r87c_crowding.json", flush=True)
    return results


def _volume_surge_analysis(strat_trades, strat_dailies, h1_df):
    """When gold volume surges, does strategy alpha degrade?"""
    if 'Volume' not in h1_df.columns or h1_df['Volume'].sum() == 0:
        return {'available': False, 'reason': 'no volume data'}

    daily_vol = h1_df.resample('D')['Volume'].sum().dropna()
    if len(daily_vol) < 100:
        return {'available': False, 'reason': 'insufficient volume data'}

    vol_pctile = daily_vol.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False).dropna()

    results = {}
    for name, ds in strat_dailies.items():
        high_vol_pnl = []; normal_pnl = []
        for d, pnl in ds.items():
            if d in vol_pctile.index:
                if vol_pctile.loc[d] >= 0.80:
                    high_vol_pnl.append(pnl)
                else:
                    normal_pnl.append(pnl)

        if high_vol_pnl and normal_pnl:
            results[name] = {
                'high_vol_avg_pnl': round(float(np.mean(high_vol_pnl)), 2),
                'normal_avg_pnl': round(float(np.mean(normal_pnl)), 2),
                'high_vol_days': len(high_vol_pnl),
                'normal_days': len(normal_pnl),
                'alpha_degrades_in_high_vol': float(np.mean(high_vol_pnl)) < float(np.mean(normal_pnl)),
            }
            print(f"    {name}: HighVol avg=${np.mean(high_vol_pnl):.2f} vs "
                  f"Normal avg=${np.mean(normal_pnl):.2f}  "
                  f"{'DEGRADES' if results[name]['alpha_degrades_in_high_vol'] else 'OK'}", flush=True)
        else:
            results[name] = {'available': False, 'reason': 'no matching days'}

    return {'available': True, 'strategies': results}


def _signal_clustering(strat_trades):
    """How often do multiple strategies trigger within the same 4-hour window?"""
    # Build entry time buckets (4-hour windows)
    buckets = defaultdict(list)
    for name, trades in strat_trades.items():
        for t in trades:
            ts = pd.Timestamp(t['entry_time'])
            bucket = ts.floor('4h')
            buckets[bucket].append({'strategy': name, 'pnl': t['pnl']})

    # Count concurrent signals
    concurrent_counts = defaultdict(int)
    pnl_by_concurrency = defaultdict(list)
    for bucket, entries in buckets.items():
        n_strats = len(set(e['strategy'] for e in entries))
        concurrent_counts[n_strats] += 1
        total_pnl = sum(e['pnl'] for e in entries)
        pnl_by_concurrency[n_strats].append(total_pnl)

    result = {'n_windows': len(buckets)}
    for k in sorted(concurrent_counts.keys()):
        pnls = pnl_by_concurrency[k]
        result[f'{k}_strats_concurrent'] = {
            'count': concurrent_counts[k],
            'avg_pnl': round(float(np.mean(pnls)), 2),
            'total_pnl': round(float(np.sum(pnls)), 2),
        }
        print(f"    {k} strategies concurrent: {concurrent_counts[k]} windows, "
              f"avg_pnl=${np.mean(pnls):.2f}", flush=True)

    return result


def _rolling_correlation(strat_dailies, window=60):
    """Rolling 60-day pairwise correlation between strategy daily PnL."""
    names = list(strat_dailies.keys())
    all_dates = sorted(set().union(*[set(s.index) for s in strat_dailies.values()]))
    aligned = pd.DataFrame(0.0, index=pd.DatetimeIndex(all_dates), columns=names)
    for name, ds in strat_dailies.items():
        for d in ds.index:
            if d in aligned.index:
                aligned.loc[d, name] = ds.loc[d]

    result = {}
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]
    for a, b in pairs:
        rolling_corr = aligned[a].rolling(window).corr(aligned[b]).dropna()
        if len(rolling_corr) > 0:
            pair_key = f"{a}_vs_{b}"
            full_corr = float(aligned[a].corr(aligned[b]))
            recent_corr = float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else full_corr
            mean_corr = float(rolling_corr.mean())
            max_corr = float(rolling_corr.max())
            min_corr = float(rolling_corr.min())

            # Trend: is correlation increasing over time?
            if len(rolling_corr) >= 252:
                first_half = rolling_corr.iloc[:len(rolling_corr)//2].mean()
                second_half = rolling_corr.iloc[len(rolling_corr)//2:].mean()
                trend = 'INCREASING' if second_half > first_half + 0.05 else \
                        'DECREASING' if second_half < first_half - 0.05 else 'STABLE'
            else:
                trend = 'INSUFFICIENT_DATA'

            result[pair_key] = {
                'full_sample': round(full_corr, 3),
                'recent_60d': round(recent_corr, 3),
                'mean_rolling': round(mean_corr, 3),
                'max_rolling': round(max_corr, 3),
                'min_rolling': round(min_corr, 3),
                'trend': trend,
            }
            print(f"    {pair_key}: full={full_corr:.3f}  recent={recent_corr:.3f}  "
                  f"trend={trend}", flush=True)

    return result


def _spread_sensitivity(h1_df):
    """How fast does alpha decay as spread increases?"""
    spreads = [0.20, 0.30, 0.50, 0.88, 1.00, 1.30, 1.50, 2.00, 3.00]
    strat_fns = {
        'PSAR': backtest_psar,
        'TSMOM': backtest_tsmom,
        'SESS_BO': backtest_sess_bo,
    }

    results = {}
    for name, fn in strat_fns.items():
        sharpes = []
        for sp in spreads:
            trades = fn(h1_df, spread=sp, lot=BASE_LOT)
            daily = _trades_to_daily(trades)
            sh = _sharpe(daily.values)
            sharpes.append(round(sh, 2))

        # Break-even spread (where Sharpe drops below 1.0)
        break_even = None
        for sp, sh in zip(spreads, sharpes):
            if sh < 1.0:
                break_even = sp
                break

        # Sensitivity: Sharpe drop per $0.10 spread increase
        if len(sharpes) >= 2:
            dsh = sharpes[0] - sharpes[-1]
            d_spread = spreads[-1] - spreads[0]
            sensitivity = round(dsh / (d_spread / 0.10), 3)
        else:
            sensitivity = 0

        results[name] = {
            'sharpe_by_spread': dict(zip([str(s) for s in spreads], sharpes)),
            'break_even_spread': break_even,
            'sensitivity_per_10c': sensitivity,
        }
        print(f"    {name}: baseline_sharpe={sharpes[1]:.2f}  "
              f"break_even={'$'+str(break_even) if break_even else '>$3.00'}  "
              f"sensitivity={sensitivity:.3f}/10c", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# R87-D: Dynamic Half-Kelly Position Sizing
# ═══════════════════════════════════════════════════════════════

def run_r87d(strat_trades, h1_df):
    """Dynamic Half-Kelly vs fixed lot, per strategy + K-Fold."""
    print(f"\n{'#'*72}")
    print(f"  R87-D — Dynamic Half-Kelly Position Sizing")
    print(f"{'#'*72}\n", flush=True)

    KELLY_WINDOW = 60
    KELLY_FRAC = 0.5
    MIN_LOT = 0.01
    MAX_LOT = BASE_LOT * 2  # 0.06

    results = {}

    for name, trades in strat_trades.items():
        if len(trades) < KELLY_WINDOW + 10:
            results[name] = {'error': f'insufficient trades ({len(trades)})'}
            continue

        # Fixed lot baseline
        fixed_pnls = [t['pnl'] for t in trades]
        fixed_daily = _trades_to_daily(trades)

        # Dynamic Kelly: use rolling 60-trade window to compute optimal lot
        kelly_trades = []
        for i in range(len(trades)):
            if i < KELLY_WINDOW:
                # Use fixed lot for warmup
                kelly_trades.append({**trades[i]})
                continue

            window = [trades[j]['pnl'] for j in range(i-KELLY_WINDOW, i)]
            wins = [p for p in window if p > 0]
            losses = [p for p in window if p <= 0]
            win_rate = len(wins) / len(window)
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1

            if avg_loss > 0 and win_rate > 0:
                rr = avg_win / avg_loss
                kelly_raw = win_rate - (1 - win_rate) / rr
                kelly_adj = kelly_raw * KELLY_FRAC
                lot_scale = max(MIN_LOT / BASE_LOT, min(MAX_LOT / BASE_LOT, 1.0 + kelly_adj))
            else:
                lot_scale = 1.0

            scaled_pnl = trades[i]['pnl'] * lot_scale
            kelly_trades.append({**trades[i], 'pnl': scaled_pnl, 'lot_scale': lot_scale})

        kelly_daily = _trades_to_daily(kelly_trades)

        # Compare
        fixed_sh = _sharpe(fixed_daily.values)
        kelly_sh = _sharpe(kelly_daily.values)
        fixed_pnl = float(fixed_daily.sum())
        kelly_pnl = float(kelly_daily.sum())
        fixed_dd = _max_dd(fixed_daily.values)
        kelly_dd = _max_dd(kelly_daily.values)
        fixed_cal = _calmar(fixed_daily.values)
        kelly_cal = _calmar(kelly_daily.values)

        # Yearly breakdown
        yearly_compare = {}
        for daily, label in [(fixed_daily, 'fixed'), (kelly_daily, 'kelly')]:
            for d, v in daily.items():
                yr = d.year
                yearly_compare.setdefault(yr, {})[f'{label}_pnl'] = \
                    yearly_compare.get(yr, {}).get(f'{label}_pnl', 0) + v

        results[name] = {
            'n_trades': len(trades),
            'fixed': {
                'sharpe': round(fixed_sh, 2), 'pnl': round(fixed_pnl, 2),
                'max_dd': round(fixed_dd, 2), 'calmar': round(fixed_cal, 3),
            },
            'half_kelly': {
                'sharpe': round(kelly_sh, 2), 'pnl': round(kelly_pnl, 2),
                'max_dd': round(kelly_dd, 2), 'calmar': round(kelly_cal, 3),
            },
            'kelly_better': kelly_sh > fixed_sh,
            'yearly': {str(yr): {k: round(v, 2) for k, v in vals.items()}
                       for yr, vals in sorted(yearly_compare.items())},
        }

        print(f"  {name:>10}:  Fixed Sharpe={fixed_sh:.2f} PnL=${fixed_pnl:,.0f} DD=${fixed_dd:,.0f}  |  "
              f"Kelly Sharpe={kelly_sh:.2f} PnL=${kelly_pnl:,.0f} DD=${kelly_dd:,.0f}  "
              f"{'KELLY WINS' if kelly_sh > fixed_sh else 'FIXED WINS'}", flush=True)

    # K-Fold validation for Kelly
    print(f"\n  K-Fold validation (6 folds)...", flush=True)
    folds = [
        ("2015-01-01", "2017-01-01"), ("2017-01-01", "2019-01-01"),
        ("2019-01-01", "2021-01-01"), ("2021-01-01", "2023-01-01"),
        ("2023-01-01", "2025-01-01"), ("2025-01-01", "2026-05-01"),
    ]

    strat_fns = {'PSAR': backtest_psar, 'TSMOM': backtest_tsmom, 'SESS_BO': backtest_sess_bo}
    kfold_results = {}
    for name, fn in strat_fns.items():
        fold_results = []
        for fold_i, (start, end) in enumerate(folds):
            h1_slice = h1_df[start:end]
            if len(h1_slice) < 100:
                continue
            trades = fn(h1_slice, spread=SPREAD, lot=BASE_LOT)
            if len(trades) < KELLY_WINDOW + 5:
                fold_results.append({'fold': fold_i+1, 'fixed_sharpe': 0, 'kelly_sharpe': 0})
                continue

            fixed_daily = _trades_to_daily(trades)

            kelly_trades = []
            for i in range(len(trades)):
                if i < KELLY_WINDOW:
                    kelly_trades.append({**trades[i]})
                    continue
                window = [trades[j]['pnl'] for j in range(i-KELLY_WINDOW, i)]
                wins = [p for p in window if p > 0]
                losses = [p for p in window if p <= 0]
                wr = len(wins) / len(window)
                aw = np.mean(wins) if wins else 0
                al = abs(np.mean(losses)) if losses else 1
                if al > 0 and wr > 0:
                    rr = aw / al
                    k_raw = wr - (1 - wr) / rr
                    k_adj = k_raw * KELLY_FRAC
                    ls = max(MIN_LOT/BASE_LOT, min(MAX_LOT/BASE_LOT, 1.0 + k_adj))
                else:
                    ls = 1.0
                kelly_trades.append({**trades[i], 'pnl': trades[i]['pnl'] * ls})

            kelly_daily = _trades_to_daily(kelly_trades)
            fold_results.append({
                'fold': fold_i+1, 'period': f"{start}~{end}",
                'fixed_sharpe': round(_sharpe(fixed_daily.values), 2),
                'kelly_sharpe': round(_sharpe(kelly_daily.values), 2),
                'kelly_wins': _sharpe(kelly_daily.values) > _sharpe(fixed_daily.values),
            })

        kelly_wins_count = sum(1 for f in fold_results if f.get('kelly_wins', False))
        kfold_results[name] = {
            'folds': fold_results,
            'kelly_wins_folds': kelly_wins_count,
            'total_folds': len(fold_results),
        }
        print(f"    {name}: Kelly wins {kelly_wins_count}/{len(fold_results)} folds", flush=True)

    results['kfold'] = kfold_results

    with open(OUTPUT_DIR / "r87d_kelly.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: r87d_kelly.json", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# R87-E: Factor Decomposition / Return Attribution
# ═══════════════════════════════════════════════════════════════

def run_r87e(strat_dailies, portfolio_daily, h1_df):
    """OLS factor decomposition of P6 portfolio returns."""
    print(f"\n{'#'*72}")
    print(f"  R87-E — Factor Decomposition / Return Attribution")
    print(f"{'#'*72}\n", flush=True)

    # Build daily factor data from H1
    daily_df = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    # Remove timezone to align with tz-naive trade dates
    if daily_df.index.tz is not None:
        daily_df.index = daily_df.index.tz_localize(None)
    daily_df['Return'] = daily_df['Close'].pct_change()
    daily_df['ATR14'] = compute_atr(daily_df, 14)
    daily_df['ATR_change'] = daily_df['ATR14'].pct_change()
    daily_df['Momentum_20'] = daily_df['Close'] / daily_df['Close'].shift(20) - 1
    daily_df['DayOfWeek'] = daily_df.index.dayofweek
    daily_df['Month'] = daily_df.index.month
    daily_df = daily_df.dropna()

    targets = {**strat_dailies, 'PORTFOLIO': portfolio_daily}
    results = {}

    for name, ds in targets.items():
        # Align dates — normalize both to date-only for matching
        ds_dates = pd.DatetimeIndex([d.date() if hasattr(d, 'date') else d for d in ds.index])
        df_dates = pd.DatetimeIndex([d.date() if hasattr(d, 'date') else d for d in daily_df.index])
        ds_reindexed = pd.Series(ds.values, index=ds_dates)
        df_reindexed = daily_df.set_index(df_dates)
        common = ds_reindexed.index.intersection(df_reindexed.index)
        if len(common) < 100:
            results[name] = {'error': f'insufficient aligned days ({len(common)})'}
            continue

        y = ds_reindexed.reindex(common).fillna(0).values
        factors_df = df_reindexed.loc[common].copy()

        # Build factor matrix
        X_parts = []
        factor_names = []

        # Market factor: gold daily return
        X_parts.append(factors_df['Return'].values)
        factor_names.append('Market_Return')

        # Trend factor: momentum direction * return (captures trend-following alpha)
        trend_signal = np.sign(factors_df['Momentum_20'].values)
        X_parts.append(trend_signal * factors_df['Return'].values)
        factor_names.append('Trend_Return')

        # Volatility factor: ATR change
        X_parts.append(factors_df['ATR_change'].values)
        factor_names.append('Vol_Change')

        # Calendar factors: day-of-week dummies (Mon-Thu, Fri is reference)
        for dow in range(4):
            X_parts.append((factors_df['DayOfWeek'].values == dow).astype(float))
            factor_names.append(f'DOW_{["Mon","Tue","Wed","Thu"][dow]}')

        X = np.column_stack(X_parts)
        X = np.column_stack([np.ones(len(y)), X])  # add intercept
        factor_names = ['Alpha'] + factor_names

        # Handle NaN/Inf
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X_clean = X[valid]
        y_clean = y[valid]

        if len(y_clean) < 50:
            results[name] = {'error': 'insufficient clean data after filtering'}
            continue

        # OLS regression
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_clean, y_clean, rcond=None)
            y_hat = X_clean @ beta
            ss_res = np.sum((y_clean - y_hat) ** 2)
            ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # T-statistics
            n_obs = len(y_clean)
            n_params = X_clean.shape[1]
            mse = ss_res / max(n_obs - n_params, 1)
            try:
                cov_beta = mse * np.linalg.inv(X_clean.T @ X_clean)
                se = np.sqrt(np.diag(cov_beta))
                t_stats = beta / np.where(se > 0, se, 1e-10)
            except np.linalg.LinAlgError:
                t_stats = np.zeros_like(beta)

            # Factor loadings
            loadings = {}
            for i, fname in enumerate(factor_names):
                loadings[fname] = {
                    'coefficient': round(float(beta[i]), 6),
                    't_stat': round(float(t_stats[i]), 2),
                    'significant': abs(float(t_stats[i])) > 1.96,
                }

            # Residual alpha (annualized)
            alpha_daily = float(beta[0])
            alpha_annual = alpha_daily * 252

            # What-if: hedge out market beta
            market_beta = float(beta[1])
            hedged_pnl = y_clean - market_beta * X_clean[:, 1]
            hedged_sharpe = _sharpe(hedged_pnl)

            results[name] = {
                'r_squared': round(float(r_squared), 4),
                'alpha_daily': round(alpha_daily, 4),
                'alpha_annual': round(alpha_annual, 2),
                'market_beta': round(market_beta, 4),
                'hedged_sharpe': round(hedged_sharpe, 2),
                'factor_loadings': loadings,
                'n_observations': n_obs,
            }

            print(f"  {name:>10}: R²={r_squared:.4f}  Alpha=${alpha_annual:.2f}/yr  "
                  f"Market_β={market_beta:.4f}  Hedged_Sharpe={hedged_sharpe:.2f}", flush=True)
            for fname, ld in loadings.items():
                sig = '*' if ld['significant'] else ' '
                print(f"    {fname:<16} β={ld['coefficient']:>10.6f}  t={ld['t_stat']:>6.2f} {sig}", flush=True)

        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"  {name:>10}: OLS failed — {e}", flush=True)

    # Rolling alpha (252-day window)
    print(f"\n  Rolling 252-day alpha analysis...", flush=True)
    for name in ['PORTFOLIO']:
        ds = targets.get(name)
        if ds is None:
            continue
        ds_dates = pd.DatetimeIndex([d.date() if hasattr(d, 'date') else d for d in ds.index])
        df_dates = pd.DatetimeIndex([d.date() if hasattr(d, 'date') else d for d in daily_df.index])
        ds_ri = pd.Series(ds.values, index=ds_dates)
        df_ri = daily_df.set_index(df_dates)
        common = ds_ri.index.intersection(df_ri.index)
        if len(common) < 300:
            continue

        y_full = ds_ri.reindex(common).fillna(0).values
        ret_full = df_ri.loc[common, 'Return'].values
        trend_full = np.sign(df_ri.loc[common, 'Momentum_20'].values) * ret_full
        vol_full = df_ri.loc[common, 'ATR_change'].values

        roll_alpha = []
        roll_r2 = []
        roll_dates = []
        window = 252

        for i in range(window, len(y_full)):
            y_w = y_full[i-window:i]
            X_w = np.column_stack([
                np.ones(window), ret_full[i-window:i],
                trend_full[i-window:i], vol_full[i-window:i]
            ])
            valid_w = np.all(np.isfinite(X_w), axis=1) & np.isfinite(y_w)
            if valid_w.sum() < 50:
                continue
            try:
                beta_w, _, _, _ = np.linalg.lstsq(X_w[valid_w], y_w[valid_w], rcond=None)
                y_hat_w = X_w[valid_w] @ beta_w
                ss_res_w = np.sum((y_w[valid_w] - y_hat_w) ** 2)
                ss_tot_w = np.sum((y_w[valid_w] - y_w[valid_w].mean()) ** 2)
                r2_w = 1 - ss_res_w / ss_tot_w if ss_tot_w > 0 else 0
                roll_alpha.append(float(beta_w[0]) * 252)
                roll_r2.append(float(r2_w))
                roll_dates.append(str(common[i]))
            except Exception:
                continue

        if roll_alpha:
            results[f'{name}_rolling'] = {
                'mean_alpha': round(float(np.mean(roll_alpha)), 2),
                'recent_alpha': round(float(np.mean(roll_alpha[-60:])), 2) if len(roll_alpha) >= 60 else None,
                'mean_r2': round(float(np.mean(roll_r2)), 4),
                'alpha_trend': 'INCREASING' if len(roll_alpha) >= 120 and
                    np.mean(roll_alpha[-60:]) > np.mean(roll_alpha[:60]) else
                    'DECREASING' if len(roll_alpha) >= 120 and
                    np.mean(roll_alpha[-60:]) < np.mean(roll_alpha[:60]) * 0.7 else 'STABLE',
                'n_windows': len(roll_alpha),
            }
            print(f"    {name} rolling alpha: mean=${np.mean(roll_alpha):.2f}/yr  "
                  f"recent=${np.mean(roll_alpha[-60:]):.2f}/yr  "
                  f"R²={np.mean(roll_r2):.4f}", flush=True)

    with open(OUTPUT_DIR / "r87e_factor.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: r87e_factor.json", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  R87 — Five Advanced Risk Research Directions")
    print("  A: CVaR  B: Health  C: Crowding  D: Kelly  E: Factor")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    # Run strategies
    print("  Running strategy backtests...", flush=True)
    strat_fns = {
        'PSAR': backtest_psar,
        'TSMOM': backtest_tsmom,
        'SESS_BO': backtest_sess_bo,
    }
    strat_trades = {}
    strat_dailies = {}
    for name, fn in strat_fns.items():
        trades = fn(h1_df, spread=SPREAD, lot=BASE_LOT)
        strat_trades[name] = trades
        ds = _trades_to_daily(trades)
        strat_dailies[name] = ds
        print(f"    {name}: {len(trades)} trades, Sharpe={_sharpe(ds.values):.2f}, "
              f"PnL=${ds.sum():,.0f}", flush=True)

    # Portfolio daily PnL (equal weight)
    all_dates = sorted(set().union(*[set(s.index) for s in strat_dailies.values()]))
    portfolio_daily = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
    for name, ds in strat_dailies.items():
        for d in ds.index:
            if d in portfolio_daily.index:
                portfolio_daily.loc[d] += ds.loc[d]
    print(f"    PORTFOLIO: Sharpe={_sharpe(portfolio_daily.values):.2f}, "
          f"PnL=${portfolio_daily.sum():,.0f}\n", flush=True)

    # Run all 5 modules
    r87a = run_r87a(strat_dailies, portfolio_daily)

    r87b = run_r87b(strat_trades, strat_dailies, h1_df)

    r87c = run_r87c(strat_trades, strat_dailies, h1_df)

    r87d = run_r87d(strat_trades, h1_df)

    r87e = run_r87e(strat_dailies, portfolio_daily, h1_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  R87 ALL MODULES COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'=' * 72}")

    summary = {
        'elapsed_s': round(elapsed, 1),
        'modules': {
            'r87a_cvar': 'DONE',
            'r87b_health': 'DONE',
            'r87c_crowding': 'DONE',
            'r87d_kelly': 'DONE',
            'r87e_factor': 'DONE',
        },
    }
    with open(OUTPUT_DIR / "r87_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
