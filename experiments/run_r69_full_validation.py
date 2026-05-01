#!/usr/bin/env python3
"""
R69 — Full 8-Stage Validation for All 4 Strategies
=====================================================
Runs the new professional validator pipeline on each strategy independently:
  1. PSAR (H1)
  2. TSMOM (H1)
  3. SESS_BO (H1)
  4. L8_MAX (M15 engine)

Each strategy gets a complete 8-stage report saved to results/r69_validation/.
Estimated runtime: ~30-60 minutes on server.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r69_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88

# ═══════════════════════════════════════════════════════════════
# Shared indicators
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


# ═══════════════════════════════════════════════════════════════
# Strategy 1: PSAR (H1) — R51 optimized: AF 0.01/0.05, SL 2.0, TP 16, MH 80
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
    df['PSAR_dir'] = direction
    df['ATR'] = compute_atr(df)
    return df


def backtest_psar(h1_df, spread=SPREAD, lot=0.03,
                  sl_atr=2.0, tp_atr=16.0, trail_act_atr=0.20,
                  trail_dist_atr=0.04, max_hold=80):
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999
    for i in range(1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        prev_d = psar_dir[i-1]; cur_d = psar_dir[i]
        if prev_d == -1 and cur_d == 1:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif prev_d == 1 and cur_d == -1:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def psar_backtest_fn(h1_df, spread, lot):
    return backtest_psar(h1_df, spread, lot)

def psar_base_fn(h1_df, spread, lot):
    return backtest_psar(h1_df, spread, lot, sl_atr=3.0, tp_atr=6.0,
                         trail_act_atr=99.0, trail_dist_atr=99.0, max_hold=200)

def psar_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20):
        return base * (1 + rng.uniform(-pct, pct))
    return backtest_psar(h1_df, spread, lot, sl_atr=p(2.0), tp_atr=p(16.0),
                         trail_act_atr=p(0.20), trail_dist_atr=p(0.04),
                         max_hold=max(10, int(p(80))))

def psar_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 4.5]:
        for tp in [8.0, 12.0, 16.0, 20.0]:
            for mh in [40, 80, 120]:
                trades = backtest_psar(h1_df, spread, lot, sl_atr=sl, tp_atr=tp, max_hold=mh)
                daily = _trades_to_daily(trades)
                sh = _sharpe(daily)
                results[f"SL={sl}_TP={tp}_MH={mh}"] = sh
    return results


# ═══════════════════════════════════════════════════════════════
# Strategy 2: TSMOM (H1) — R53 optimized: W480/720, SL 4.5, TP 6, MH 20
# ═══════════════════════════════════════════════════════════════

def backtest_tsmom(h1_df, spread=SPREAD, lot=0.03,
                   fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
                   trail_act_atr=0.14, trail_dist_atr=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(close)
    weights = [(fast, 0.5), (slow, 0.5)]
    max_lb = max(lb for lb, _ in weights)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        for lb, w in weights:
            if i >= lb:
                s += w * np.sign(close[i] / close[i - lb] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= ad:
                    ts_p = lo_v + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if not exited and not np.isnan(score[i]):
                if pos['dir'] == 'BUY' and score[i] < 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
                elif pos['dir'] == 'SELL' and score[i] > 0:
                    trades.append(_mk(pos, c, times[i], "Reversal", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None or i - last_exit < 2: continue
        if np.isnan(score[i]) or np.isnan(cur_atr) or cur_atr < 0.1: continue
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def tsmom_backtest_fn(h1_df, spread, lot):
    return backtest_tsmom(h1_df, spread, lot)

def tsmom_base_fn(h1_df, spread, lot):
    return backtest_tsmom(h1_df, spread, lot, sl_atr=3.0, tp_atr=3.0,
                          trail_act_atr=99.0, trail_dist_atr=99.0, max_hold=50)

def tsmom_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20):
        return base * (1 + rng.uniform(-pct, pct))
    return backtest_tsmom(h1_df, spread, lot, sl_atr=p(4.5), tp_atr=p(6.0),
                          trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                          max_hold=max(5, int(p(20))))

def tsmom_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [3.0, 4.0, 4.5, 5.0, 6.0]:
        for tp in [4.0, 5.0, 6.0, 8.0]:
            for mh in [15, 20, 30]:
                trades = backtest_tsmom(h1_df, spread, lot, sl_atr=sl, tp_atr=tp, max_hold=mh)
                daily = _trades_to_daily(trades)
                sh = _sharpe(daily)
                results[f"SL={sl}_TP={tp}_MH={mh}"] = sh
    return results


# ═══════════════════════════════════════════════════════════════
# Strategy 3: SESS_BO (H1) — R55a: peak_12_14, LB 3, SL 3.0, TP 6, MH 20
# ═══════════════════════════════════════════════════════════════

def backtest_sess_bo(h1_df, spread=SPREAD, lot=0.03,
                     session="peak_12_14", lookback_bars=3,
                     sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.14,
                     trail_dist_atr=0.025, max_hold=20):
    SESSION_DEFS = {"asian": (0,7), "london": (8,11), "ny_peak": (12,16),
                    "late": (17,23), "peak_12_14": (12,14)}
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour
    times = df.index; n = len(df); last_exit = -999
    sess_start, sess_end = SESSION_DEFS[session]
    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo_v - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * 100)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * 100)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if cur_hour != sess_start: continue
        if i > 0 and hours[i-1] == sess_start: continue
        range_high = max(high[i - lookback_bars:i])
        range_low  = min(low[i - lookback_bars:i])
        if c > range_high:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif c < range_low:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


def sess_bo_backtest_fn(h1_df, spread, lot):
    return backtest_sess_bo(h1_df, spread, lot)

def sess_bo_base_fn(h1_df, spread, lot):
    return backtest_sess_bo(h1_df, spread, lot, sl_atr=3.0, tp_atr=3.0,
                            trail_act_atr=99.0, trail_dist_atr=99.0, max_hold=50)

def sess_bo_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20):
        return base * (1 + rng.uniform(-pct, pct))
    return backtest_sess_bo(h1_df, spread, lot, sl_atr=p(3.0), tp_atr=p(6.0),
                            trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                            max_hold=max(5, int(p(20))))

def sess_bo_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [2.0, 3.0, 4.0, 4.5]:
        for tp in [4.0, 5.0, 6.0, 8.0]:
            for lb in [2, 3, 4, 5]:
                trades = backtest_sess_bo(h1_df, spread, lot, sl_atr=sl, tp_atr=tp,
                                          lookback_bars=lb)
                daily = _trades_to_daily(trades)
                sh = _sharpe(daily)
                results[f"SL={sl}_TP={tp}_LB={lb}"] = sh
    return results


# ═══════════════════════════════════════════════════════════════
# Strategy 4: L8_MAX (M15 engine) — LIVE_PARITY_KWARGS
# ═══════════════════════════════════════════════════════════════

_L8_DATA = None  # DataBundle with indicators prepared

def _load_l8_data():
    global _L8_DATA
    if _L8_DATA is None:
        from backtest.runner import DataBundle
        _L8_DATA = DataBundle.load_default()
    return _L8_DATA


def _run_l8_engine(data_bundle, spread, lot, maxloss_cap=37):
    from backtest.runner import run_variant, LIVE_PARITY_KWARGS
    import backtest.runner as runner_mod
    try:
        from indicators import signals as signals_mod
    except ImportError:
        signals_mod = None

    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
          'spread_cost': spread, 'initial_capital': 2000,
          'min_lot_size': lot, 'max_lot_size': lot}

    if hasattr(runner_mod, 'get_orb_strategy'):
        runner_mod.get_orb_strategy().reset_daily()
    if signals_mod and hasattr(signals_mod, '_friday_close_price'):
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False

    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw = result.get('_trades', [])
    trades = []
    for t in raw:
        pnl = t.pnl if hasattr(t, 'pnl') else t.get('pnl', 0)
        ext = t.exit_time if hasattr(t, 'exit_time') else t.get('exit_time', '')
        ent = t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', ext)
        d = t.direction if hasattr(t, 'direction') else t.get('dir', 'BUY')
        trades.append({'pnl': pnl, 'exit_time': ext, 'entry_time': ent, 'dir': d})
    return trades


def l8_backtest_fn(h1_df, spread, lot):
    """L8_MAX always uses pre-loaded DataBundle with indicators.
    The h1_df arg is used for date-range slicing only."""
    full_data = _load_l8_data()
    if len(h1_df) < len(full_data.h1_df):
        start = h1_df.index[0]
        end = h1_df.index[-1]
        data = full_data.slice(str(start), str(end + pd.Timedelta(hours=1)))
    else:
        data = full_data
    return _run_l8_engine(data, spread, lot)


def l8_perturb_fn(h1_df, spread, lot, rng):
    """L8_MAX engine is slow — perturb maxloss_cap only."""
    full_data = _load_l8_data()
    if len(h1_df) < len(full_data.h1_df):
        start = h1_df.index[0]
        end = h1_df.index[-1]
        data = full_data.slice(str(start), str(end + pd.Timedelta(hours=1)))
    else:
        data = full_data
    cap = int(37 * (1 + rng.uniform(-0.30, 0.30)))
    return _run_l8_engine(data, spread, lot, maxloss_cap=max(10, cap))


# ═══════════════════════════════════════════════════════════════
# Run all validations
# ═══════════════════════════════════════════════════════════════

def main():
    from backtest.validator import StrategyValidator, ValidatorConfig

    t0_total = time.time()
    print("=" * 70)
    print("  R69 — Full 8-Stage Validation Pipeline")
    print("  4 Strategies x 8 Stages = Comprehensive Gold Strategy Audit")
    print("=" * 70, flush=True)

    # Load H1 data once for H1 strategies
    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    strategies = [
        {
            'name': 'PSAR_H1',
            'backtest_fn': psar_backtest_fn,
            'base_backtest_fn': psar_base_fn,
            'param_perturb_fn': psar_perturb_fn,
            'param_grid_fn': psar_grid_fn,
            'lot': 0.03,
            'n_trials': 72,  # 6 SL x 4 TP x 3 MH combos tested
        },
        {
            'name': 'TSMOM_H1',
            'backtest_fn': tsmom_backtest_fn,
            'base_backtest_fn': tsmom_base_fn,
            'param_perturb_fn': tsmom_perturb_fn,
            'param_grid_fn': tsmom_grid_fn,
            'lot': 0.03,
            'n_trials': 60,  # 5 SL x 4 TP x 3 MH
        },
        {
            'name': 'SESS_BO_H1',
            'backtest_fn': sess_bo_backtest_fn,
            'base_backtest_fn': sess_bo_base_fn,
            'param_perturb_fn': sess_bo_perturb_fn,
            'param_grid_fn': sess_bo_grid_fn,
            'lot': 0.03,
            'n_trials': 64,  # 4 SL x 4 TP x 4 LB
        },
        {
            'name': 'L8_MAX',
            'backtest_fn': l8_backtest_fn,
            'base_backtest_fn': None,  # L8 IS the base engine — no "default" version
            'param_perturb_fn': l8_perturb_fn,
            'param_grid_fn': None,  # engine too slow for full grid
            'lot': 0.01,
            'n_trials': 1,  # L8 uses preset config, not grid-optimized
        },
    ]

    all_summaries = {}
    for strat in strategies:
        print(f"\n\n{'#' * 70}")
        print(f"  VALIDATING: {strat['name']}")
        print(f"{'#' * 70}\n", flush=True)

        t0 = time.time()
        config = ValidatorConfig(
            n_trials_tested=strat['n_trials'],
            realistic_spread=REALISTIC_SPREAD,
            purge_bars=30,
            n_param_perturb=50 if strat['name'] == 'L8_MAX' else 200,
            n_bootstrap=5000,
            n_trade_removal=500,
        )

        validator = StrategyValidator(
            name=strat['name'],
            backtest_fn=strat['backtest_fn'],
            spread=SPREAD,
            lot=strat['lot'],
            h1_df=h1_df,
            base_backtest_fn=strat.get('base_backtest_fn'),
            param_perturb_fn=strat.get('param_perturb_fn'),
            param_grid_fn=strat.get('param_grid_fn'),
            config=config,
            output_dir=str(OUTPUT_DIR / strat['name']),
        )

        results = validator.run_all(stop_on_fail=False)

        elapsed = time.time() - t0
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        all_summaries[strat['name']] = {
            'passed': passed, 'total': total,
            'elapsed_s': round(elapsed, 1),
            'stages': {f"stage{s}": {'passed': r.passed, 'sharpe': r.sharpe, 'verdict': r.verdict}
                       for s, r in sorted(results.items())},
        }
        print(f"\n  {strat['name']}: {passed}/{total} stages passed ({elapsed:.0f}s)", flush=True)

    # Final cross-strategy summary
    total_elapsed = time.time() - t0_total
    print(f"\n\n{'=' * 70}")
    print(f"  R69 FINAL SUMMARY")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'=' * 70}")
    for name, s in all_summaries.items():
        icon = "ALL PASS" if s['passed'] == s['total'] else f"{s['passed']}/{s['total']}"
        print(f"  {name:>12}: [{icon}] ({s['elapsed_s']:.0f}s)")
        for stage_key, stage_info in sorted(s['stages'].items()):
            status = "PASS" if stage_info['passed'] else "FAIL"
            print(f"    {stage_key}: [{status}] {stage_info['verdict']}")
    print(f"{'=' * 70}", flush=True)

    # Save combined summary
    summary_path = OUTPUT_DIR / "r69_combined_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
