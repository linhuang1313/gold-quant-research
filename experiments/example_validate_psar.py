#!/usr/bin/env python3
"""
Example: Validate PSAR strategy using the 8-stage pipeline.

Usage:
    python experiments/example_validate_psar.py            # all 8 stages
    python experiments/example_validate_psar.py --stage 1   # stage 1 only
"""
import sys, os, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from backtest.validator import StrategyValidator, ValidatorConfig


# ═══════════════════════════════════════════════════════════════
# Step 1: Define your strategy as a function
#         Signature: (h1_df, spread, lot) -> list[dict]
#         Each dict needs: pnl, exit_time, entry_time, dir
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


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
                if df['High'].iloc[i] > ep: ep = df['High'].iloc[i]; af = min(af+af_start, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], df['High'].iloc[i-1], df['High'].iloc[max(0,i-2)])
            if df['High'].iloc[i] > psar[i]:
                direction[i] = 1; psar[i] = ep; ep = df['High'].iloc[i]; af = af_start
            else:
                direction[i] = -1
                if df['Low'].iloc[i] < ep: ep = df['Low'].iloc[i]; af = min(af+af_start, af_max)
    df['PSAR_dir'] = direction; df['ATR'] = compute_atr(df)
    return df


def backtest_psar(h1_df, spread=0.30, lot=0.03,
                  sl_atr=2.0, tp_atr=16.0, trail_act=0.20, trail_dist=0.04, max_hold=80):
    """PSAR H1 strategy — returns list of trade dicts."""
    df = add_psar(h1_df).dropna(subset=['PSAR_dir', 'ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    psar_dir = df['PSAR_dir'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999

    for i in range(1, n):
        c = close[i]; h = high[i]; lo = low[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * 100
                pnl_l = (lo - pos['entry'] - spread) * lot * 100
                pnl_c = (c - pos['entry'] - spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - lo - spread) * lot * 100
                pnl_l = (pos['entry'] - h - spread) * lot * 100
                pnl_c = (pos['entry'] - c - spread) * lot * 100
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                trades.append({**pos, 'exit_time': times[i], 'pnl': tp_val, 'bars': held}); exited = True
            elif pnl_l <= -sl_val:
                trades.append({**pos, 'exit_time': times[i], 'pnl': -sl_val, 'bars': held}); exited = True
            else:
                ad = trail_act * pos['atr']; td = trail_dist * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo <= ts_p:
                        trades.append({**pos, 'exit_time': times[i],
                                       'pnl': (ts_p - pos['entry'] - spread) * lot * 100, 'bars': held}); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append({**pos, 'exit_time': times[i],
                                       'pnl': (pos['entry'] - ts_p - spread) * lot * 100, 'bars': held}); exited = True
                if not exited and held >= max_hold:
                    trades.append({**pos, 'exit_time': times[i], 'pnl': pnl_c, 'bars': held}); exited = True
            if exited: pos = None; last_exit = i; continue

        if pos is None and i - last_exit >= 2 and not np.isnan(cur_atr) and cur_atr >= 0.1:
            if psar_dir[i-1] == -1 and psar_dir[i] == 1:
                pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i,
                       'entry_time': times[i], 'atr': cur_atr}
            elif psar_dir[i-1] == 1 and psar_dir[i] == -1:
                pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i,
                       'entry_time': times[i], 'atr': cur_atr}
    return trades


# ═══════════════════════════════════════════════════════════════
# Step 2: (Optional) Base backtest for Stage 0
#         Uses "textbook" defaults — NOT optimized params
# ═══════════════════════════════════════════════════════════════

def psar_base(h1_df, spread, lot):
    """PSAR with conservative textbook defaults (SL=3, TP=6, no trailing)."""
    return backtest_psar(h1_df, spread, lot,
                         sl_atr=3.0, tp_atr=6.0,
                         trail_act=99.0, trail_dist=99.0,
                         max_hold=200)


# ═══════════════════════════════════════════════════════════════
# Step 3: Parameter perturbation for Stage 4 (Monte Carlo + PBO)
# ═══════════════════════════════════════════════════════════════

def psar_perturbed(h1_df, spread, lot, rng):
    """Run PSAR with +/-20% random parameter perturbation."""
    def p(base, pct=0.20):
        return base * (1 + rng.uniform(-pct, pct))
    return backtest_psar(h1_df, spread, lot,
                         sl_atr=p(2.0), tp_atr=p(16.0),
                         trail_act=p(0.20), trail_dist=p(0.04),
                         max_hold=max(10, int(p(80))))


# ═══════════════════════════════════════════════════════════════
# Step 4: (Optional) Parameter grid for Stage 6 stability zone
# ═══════════════════════════════════════════════════════════════

def psar_param_grid(h1_df, spread, lot):
    """Scan key parameter combos, return {label: sharpe}."""
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [1.5, 2.0, 2.5, 3.0]:
        for tp in [8.0, 12.0, 16.0, 20.0]:
            trades = backtest_psar(h1_df, spread, lot, sl_atr=sl, tp_atr=tp)
            daily = _trades_to_daily(trades)
            sh = _sharpe(daily)
            results[f"SL={sl}_TP={tp}"] = sh
    return results


# ═══════════════════════════════════════════════════════════════
# Step 5: Run the validator
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    stage_arg = None
    if len(sys.argv) > 2 and sys.argv[1] == '--stage':
        stage_arg = int(sys.argv[2])

    validator = StrategyValidator(
        name="PSAR_H1",
        backtest_fn=backtest_psar,
        spread=0.30,
        lot=0.03,
        base_backtest_fn=psar_base,
        param_perturb_fn=psar_perturbed,
        param_grid_fn=psar_param_grid,
        config=ValidatorConfig(
            n_trials_tested=50,
            realistic_spread=0.88,
            purge_bars=30,
        ),
    )

    if stage_arg is not None:
        validator.run_stage(stage_arg)
    else:
        validator.run_all(stop_on_fail=False)
