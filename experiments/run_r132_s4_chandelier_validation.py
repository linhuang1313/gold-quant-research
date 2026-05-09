#!/usr/bin/env python3
"""
R132 — S4 Chandelier Exit Flip Complete Validation + Portfolio Optimization
============================================================================
Validates the S4 Chandelier Exit Flip strategy as a standalone signal, then
integrates it into the existing 4-strategy portfolio for a 5-strategy combo.

Phases:
  1. S4 standalone backtest (period=22, mult=3.0, EMA100 filter)
  2. 8-stage validator pipeline on S4
  3. Run all 4 existing strategies (L8_MAX, PSAR, TSMOM, SESS_BO)
  4. 5-strategy portfolio with S4 lot grid search (0.01–0.15)
  5. K-Fold 5-fold validation of portfolio
  6. Monte Carlo (1000 paths) of portfolio
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS, load_csv
from backtest.validator import StrategyValidator, ValidatorConfig
from indicators import calc_chandelier

OUTPUT_DIR = Path("results/r132_s4_chandelier_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
CAPS = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.08, 'SESS_BO': 0.08}
STRAT_ORDER = ['L8_MAX', 'PSAR', 'TSMOM', 'SESS_BO']

PSAR_DEFAULT = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04, 'max_hold': 20}

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


def _trades_to_daily(trades):
    if not trades:
        return np.array([0.0])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    if not daily:
        return np.array([0.0])
    return np.array([daily[k] for k in sorted(daily.keys())])


def _sharpe(arr):
    if len(arr) < 10 or np.std(arr, ddof=1) == 0:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))


def _max_dd(arr):
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 1 else 0.0


def _compute_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'wr': 0, 'max_dd': 0, 'calmar': 0}
    daily = _trades_to_daily(trades)
    pnls = [t['pnl'] for t in trades]
    n = len(trades)
    sh = _sharpe(daily)
    dd = _max_dd(daily)
    pnl = sum(pnls)
    return {
        'n': n, 'sharpe': round(sh, 3), 'pnl': round(pnl, 2),
        'wr': round(sum(1 for p in pnls if p > 0) / n * 100, 1),
        'max_dd': round(dd, 2),
        'calmar': round(pnl / dd, 2) if dd > 0 else 9999,
    }


# ═══════════════════════════════════════════════════════════════
# Strategy backtests
# ═══════════════════════════════════════════════════════════════

def bt_s4_chandelier(h1_df, spread, lot, maxloss_cap=0,
                     period=22, mult=3.0, ema_filter=True,
                     sl_atr=3.0, tp_atr=8.0, trail_act=0.28,
                     trail_dist=0.06, max_hold=20):
    """S4 Chandelier Exit Flip strategy."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    chand = calc_chandelier(df, period=period, mult=mult)
    df['Chand_long'] = chand['Chand_long']
    df['Chand_short'] = chand['Chand_short']
    if ema_filter:
        df['EMA100'] = df['Close'].ewm(span=100).mean()
    df = df.dropna(subset=['ATR', 'Chand_long', 'Chand_short'])

    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    cl_long = df['Chand_long'].values; cl_short = df['Chand_short'].values
    ema = df['EMA100'].values if ema_filter else None

    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue

        was_below_long = c[i-1] < cl_long[i-1]
        now_above_long = c[i] >= cl_long[i]
        was_above_short = c[i-1] > cl_short[i-1]
        now_below_short = c[i] <= cl_short[i]

        if was_below_long and now_above_long:
            if ema is not None and (np.isnan(ema[i]) or c[i] < ema[i]):
                continue
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif was_above_short and now_below_short:
            if ema is not None and (np.isnan(ema[i]) or c[i] > ema[i]):
                continue
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_psar(h1_df, spread, lot, maxloss_cap=0, params=None):
    if params is None:
        params = PSAR_DEFAULT
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               params['sl_atr'], params['tp_atr'], params['trail_act'],
                               params['trail_dist'], params['max_hold'], maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_tsmom(h1_df, spread, lot, maxloss_cap=0,
             fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i]/c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i]/c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb+1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i]-pos['entry']-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry']-c[i]-spread)*lot*PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i-last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


def bt_sess_bo(h1_df, spread, lot, maxloss_cap=0,
               session_hour=12, lookback=4, sl_atr=4.5, tp_atr=4.0,
               trail_act=0.14, trail_dist=0.025, max_hold=20):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i]+spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i]-spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_h1():
    candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    ]
    h1_path = next((p for p in candidates if p.exists()), candidates[-1])
    return load_csv(str(h1_path))


# ═══════════════════════════════════════════════════════════════
# Portfolio helpers
# ═══════════════════════════════════════════════════════════════

def run_all_strategies(h1_df, bundle, spread=SPREAD):
    """Run all 4 existing strategies + return trade lists."""
    psar_trades = bt_psar(h1_df, spread, UNIT_LOT, CAPS['PSAR'])
    tsmom_trades = bt_tsmom(h1_df, spread, UNIT_LOT, CAPS['TSMOM'])
    sessbo_trades = bt_sess_bo(h1_df, spread, UNIT_LOT, CAPS['SESS_BO'])

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': CAPS['L8_MAX'],
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
    l8_result = run_variant(bundle, "L8_MAX", verbose=False, **kw)
    l8_trades = [{'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                  'entry_time': t.entry_time, 'exit_time': t.exit_time,
                  'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
                 for t in l8_result.get('_trades', [])]

    return {'L8_MAX': l8_trades, 'PSAR': psar_trades,
            'TSMOM': tsmom_trades, 'SESS_BO': sessbo_trades}


def build_portfolio_daily(strat_trades, lots, spread=SPREAD):
    """Combine strategy trades into portfolio daily PnL array."""
    all_daily = {}
    for sn, trades in strat_trades.items():
        lot = lots.get(sn, UNIT_LOT)
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            scaled_pnl = t['pnl'] * (lot / UNIT_LOT)
            all_daily[d] = all_daily.get(d, 0) + scaled_pnl
    dates = sorted(all_daily.keys())
    daily_arr = np.array([all_daily[d] for d in dates]) if dates else np.array([0.0])
    return daily_arr, dates


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R132 S4 Chandelier Validation + 5-Strat Portfolio'}

    print("=" * 80, flush=True)
    print("  R132 — S4 Chandelier Exit Flip Validation + Portfolio Optimization", flush=True)
    print("=" * 80, flush=True)

    print("\n  Loading data...", flush=True)
    h1_df = load_h1()
    bundle = DataBundle.load_default()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    # ═══════════════════════════════════════════════════════════
    # Phase 1: S4 Standalone Backtest
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 1: S4 Chandelier Standalone Backtest", flush=True)
    print("=" * 70, flush=True)

    s4_trades = bt_s4_chandelier(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0)
    s4_stats = _compute_stats(s4_trades)

    print(f"\n  S4 Chandelier (period=22, mult=3.0, EMA100 filter):", flush=True)
    print(f"    Trades:  {s4_stats['n']}", flush=True)
    print(f"    Sharpe:  {s4_stats['sharpe']:.3f}", flush=True)
    print(f"    PnL:     ${s4_stats['pnl']:,.0f}", flush=True)
    print(f"    WinRate: {s4_stats['wr']:.1f}%", flush=True)
    print(f"    MaxDD:   ${s4_stats['max_dd']:,.0f}", flush=True)
    print(f"    Calmar:  {s4_stats['calmar']:.1f}", flush=True)

    results['phase1_s4_standalone'] = s4_stats

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 8-Stage Validator on S4
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 2: 8-Stage Validator Pipeline on S4", flush=True)
    print("=" * 70, flush=True)

    def s4_bt_fn(h1, spread, lot):
        return bt_s4_chandelier(h1, spread, lot, maxloss_cap=0)

    validator = StrategyValidator(
        name="S4_CHANDELIER",
        backtest_fn=s4_bt_fn,
        spread=SPREAD, lot=0.03,
        config=ValidatorConfig(n_trials_tested=50),
        output_dir=str(OUTPUT_DIR / "validator_s4"),
        h1_df=h1_df,
    )
    val_report = validator.run_all(stop_on_fail=False)

    phase2 = {}
    for stage_num, sr in sorted(val_report.items()):
        phase2[f"stage{stage_num}"] = {
            'name': sr.name, 'passed': sr.passed,
            'sharpe': sr.sharpe, 'verdict': sr.verdict,
        }
    results['phase2_validator'] = phase2

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Run All 4 Existing Strategies
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 3: Run All 4 Existing Strategies", flush=True)
    print("=" * 70, flush=True)

    strat_trades = run_all_strategies(h1_df, bundle)

    print(f"\n  {'Strategy':<12s} {'N':>6} {'Sharpe':>7} {'PnL':>12} {'WR':>6} {'MaxDD':>10}", flush=True)
    print(f"  {'─'*55}", flush=True)
    phase3 = {}
    for sn in STRAT_ORDER:
        st = _compute_stats(strat_trades[sn])
        phase3[sn] = st
        print(f"  {sn:<12s} {st['n']:>6} {st['sharpe']:>7.3f} ${st['pnl']:>11,.0f} {st['wr']:>5.1f}% ${st['max_dd']:>9,.0f}", flush=True)

    results['phase3_existing_strategies'] = phase3

    # ═══════════════════════════════════════════════════════════
    # Phase 4: 5-Strategy Portfolio + S4 Lot Grid Search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 4: 5-Strategy Portfolio — S4 Lot Grid Search", flush=True)
    print("=" * 70, flush=True)

    s4_lot_grid = np.arange(0.01, 0.16, 0.01)
    all_strat_trades = {**strat_trades, 'S4_CHAND': s4_trades}

    print(f"\n  {'S4 Lot':>8s} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>10} {'Calmar':>8}", flush=True)
    print(f"  {'─'*50}", flush=True)

    best_sharpe = -999; best_lot = 0.01; best_daily = None
    grid_results = []
    for s4_lot in s4_lot_grid:
        lots = {**R89_LOTS, 'S4_CHAND': round(s4_lot, 2)}
        daily, dates = build_portfolio_daily(all_strat_trades, lots)
        sh = _sharpe(daily); dd = _max_dd(daily); pnl = float(np.sum(daily))
        cal = pnl / dd if dd > 0 else 9999
        grid_results.append({
            's4_lot': round(s4_lot, 2), 'sharpe': round(sh, 3),
            'pnl': round(pnl, 2), 'max_dd': round(dd, 2), 'calmar': round(cal, 1),
        })
        marker = " <<" if sh > best_sharpe else ""
        print(f"  {s4_lot:>8.2f} {sh:>7.3f} ${pnl:>11,.0f} ${dd:>9,.0f} {cal:>8.1f}{marker}", flush=True)
        if sh > best_sharpe:
            best_sharpe = sh; best_lot = round(s4_lot, 2); best_daily = daily

    base_lots = {**R89_LOTS}
    base_daily, _ = build_portfolio_daily(strat_trades, base_lots)
    base_sh = _sharpe(base_daily)

    print(f"\n  4-strat baseline Sharpe: {base_sh:.3f}", flush=True)
    print(f"  Best 5-strat:  S4_lot={best_lot}, Sharpe={best_sharpe:.3f} (delta={best_sharpe - base_sh:+.3f})", flush=True)

    results['phase4_lot_grid'] = {
        'grid': grid_results, 'best_s4_lot': best_lot,
        'best_sharpe': round(best_sharpe, 3),
        'baseline_sharpe': round(base_sh, 3),
        'improvement': round(best_sharpe - base_sh, 3),
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold 5-Fold Validation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 5: K-Fold 5-Fold Validation of 5-Strategy Portfolio", flush=True)
    print("=" * 70, flush=True)

    optimal_lots = {**R89_LOTS, 'S4_CHAND': best_lot}
    fold_edges = pd.date_range("2015-01-01", "2026-05-01", periods=6)
    kfold_sharpes = []

    print(f"\n  {'Fold':>6s} {'Period':<28s} {'Sharpe':>7} {'PnL':>12}", flush=True)
    print(f"  {'─'*55}", flush=True)

    for fi in range(5):
        fs = str(fold_edges[fi].date()); fe = str(fold_edges[fi+1].date())
        h1_fold = h1_df[(h1_df.index >= fs) & (h1_df.index < fe)]
        if len(h1_fold) < 500:
            continue

        fold_strat = {}
        fold_strat['PSAR'] = bt_psar(h1_fold, SPREAD, UNIT_LOT, CAPS['PSAR'])
        fold_strat['TSMOM'] = bt_tsmom(h1_fold, SPREAD, UNIT_LOT, CAPS['TSMOM'])
        fold_strat['SESS_BO'] = bt_sess_bo(h1_fold, SPREAD, UNIT_LOT, CAPS['SESS_BO'])
        fold_strat['S4_CHAND'] = bt_s4_chandelier(h1_fold, SPREAD, UNIT_LOT, maxloss_cap=0)

        kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': CAPS['L8_MAX'],
              'spread_cost': SPREAD, 'initial_capital': 2000,
              'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
        try:
            fold_bundle = bundle.slice(fs, fe)
            l8r = run_variant(fold_bundle, f"L8_fold{fi+1}", verbose=False, **kw)
            fold_strat['L8_MAX'] = [{'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
                                     'entry_time': t.entry_time, 'exit_time': t.exit_time,
                                     'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0}
                                    for t in l8r.get('_trades', [])]
        except Exception:
            fold_strat['L8_MAX'] = []

        daily, _ = build_portfolio_daily(fold_strat, optimal_lots)
        sh = _sharpe(daily); pnl = float(np.sum(daily))
        kfold_sharpes.append(round(sh, 3))
        print(f"  {fi+1:>6d} {fs} ~ {fe:<16s} {sh:>7.3f} ${pnl:>11,.0f}", flush=True)

    mean_kf = np.mean(kfold_sharpes) if kfold_sharpes else 0
    pos_folds = sum(1 for s in kfold_sharpes if s > 0)
    print(f"\n  K-Fold mean Sharpe: {mean_kf:.3f}, positive: {pos_folds}/{len(kfold_sharpes)}", flush=True)

    results['phase5_kfold'] = {
        'fold_sharpes': kfold_sharpes, 'mean': round(mean_kf, 3),
        'positive_folds': pos_folds, 'total_folds': len(kfold_sharpes),
        'pass': pos_folds >= len(kfold_sharpes) - 1,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Monte Carlo (1000 paths)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70, flush=True)
    print("  Phase 6: Monte Carlo Robustness (1000 paths)", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.RandomState(42)
    mc_sharpes = []
    daily_ref = best_daily if best_daily is not None else base_daily

    for _ in range(1000):
        perturbed = daily_ref * rng.uniform(0.92, 1.08, size=len(daily_ref))
        mask = rng.random(len(daily_ref)) > 0.02
        perturbed = perturbed * mask
        mc_sharpes.append(_sharpe(perturbed))

    mc_arr = np.array(mc_sharpes)
    mc_result = {
        'p5': round(float(np.percentile(mc_arr, 5)), 3),
        'p25': round(float(np.percentile(mc_arr, 25)), 3),
        'p50': round(float(np.percentile(mc_arr, 50)), 3),
        'p75': round(float(np.percentile(mc_arr, 75)), 3),
        'p95': round(float(np.percentile(mc_arr, 95)), 3),
        'pct_positive': round(float(np.mean(mc_arr > 0) * 100), 1),
        'pct_gt1': round(float(np.mean(mc_arr > 1) * 100), 1),
    }

    print(f"  P5={mc_result['p5']:.3f}  P25={mc_result['p25']:.3f}  "
          f"P50={mc_result['p50']:.3f}  P75={mc_result['p75']:.3f}  P95={mc_result['p95']:.3f}", flush=True)
    print(f"  Positive: {mc_result['pct_positive']:.0f}%, Sharpe>1: {mc_result['pct_gt1']:.0f}%", flush=True)

    results['phase6_monte_carlo'] = mc_result

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    print("\n" + "=" * 80, flush=True)
    print("  R132 FINAL SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  S4 Standalone:    Sharpe={s4_stats['sharpe']:.3f}, PnL=${s4_stats['pnl']:,.0f}, "
          f"N={s4_stats['n']}, WR={s4_stats['wr']:.1f}%", flush=True)

    val_stages = sum(1 for s in val_report.values() if s.passed)
    print(f"  Validator:        {val_stages}/8 stages passed", flush=True)

    print(f"  4-Strat Baseline: Sharpe={base_sh:.3f}", flush=True)
    print(f"  5-Strat Optimal:  Sharpe={best_sharpe:.3f} (S4 lot={best_lot}, delta={best_sharpe - base_sh:+.3f})", flush=True)

    kf_pass = "PASS" if results['phase5_kfold']['pass'] else "FAIL"
    print(f"  K-Fold:           mean={mean_kf:.3f}, {pos_folds}/{len(kfold_sharpes)} positive ({kf_pass})", flush=True)
    print(f"  MC P5 Sharpe:     {mc_result['p5']:.3f}, positive={mc_result['pct_positive']:.0f}%", flush=True)

    s4_worth = (s4_stats['sharpe'] > 0.5 and best_sharpe > base_sh
                and results['phase5_kfold']['pass'] and mc_result['p5'] > 0)
    verdict = "S4 VALIDATED — add to portfolio" if s4_worth else "S4 MARGINAL — needs more evidence"
    print(f"\n  VERDICT: {verdict}", flush=True)
    results['verdict'] = verdict

    out_file = OUTPUT_DIR / "r132_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)
    print(f"  Saved: {out_file}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
