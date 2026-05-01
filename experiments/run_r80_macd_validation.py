#!/usr/bin/env python3
"""
R80 — Full 8-Stage Validation for MACD_H1 on XAUUSD
======================================================
MACD passed R78 screening (Sharpe=5.13, 6/6 K-Fold, low correlation).
Now run the complete StrategyValidator pipeline.

Parameters: MACD(12,26,9) + EMA100 trend + ADX>20 gate
Exit: SL=4.5x ATR, TP=6.0x ATR, Trail 0.14/0.025, MaxHold=20

Estimated runtime: ~10-15 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r80_macd_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
REALISTIC_SPREAD = 0.88
LOT = 0.03
PV = 100


# ═══════════════════════════════════════════════════════════════
# Helpers
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


def _compute_adx(df):
    """Compute ADX from OHLC data."""
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs(),
    }).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.rolling(14).mean()


# ═══════════════════════════════════════════════════════════════
# MACD backtest
# ═══════════════════════════════════════════════════════════════

def backtest_macd(h1_df, spread=SPREAD, lot=LOT,
                  fast_period=12, slow_period=26, signal_period=9,
                  ema_trend=100, adx_threshold=20,
                  sl_atr=4.5, tp_atr=6.0, trail_act_atr=0.14,
                  trail_dist_atr=0.025, max_hold=20):
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    df['EMA_trend'] = df['Close'].ewm(span=ema_trend, adjust=False).mean()
    df['ADX'] = _compute_adx(df)
    df = df.dropna(subset=['ATR', 'Hist', 'EMA_trend', 'ADX'])

    c_arr = df['Close'].values; h_arr = df['High'].values; l_arr = df['Low'].values
    atr = df['ATR'].values; hist = df['Hist'].values; ema_t = df['EMA_trend'].values
    adx = df['ADX'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999

    for i in range(1, n):
        c = c_arr[i]; h = h_arr[i]; lo = l_arr[i]; cur_atr = atr[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * PV
                pnl_l = (lo - pos['entry'] - spread) * lot * PV
                pnl_c = (c - pos['entry'] - spread) * lot * PV
            else:
                pnl_h = (pos['entry'] - lo - spread) * lot * PV
                pnl_l = (pos['entry'] - h - spread) * lot * PV
                pnl_c = (pos['entry'] - c - spread) * lot * PV
            tp_val = tp_atr * pos['atr'] * lot * PV
            sl_val = sl_atr * pos['atr'] * lot * PV
            exited = False
            if pnl_h >= tp_val:
                trades.append(_mk(pos, c, times[i], "TP", i, tp_val)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_mk(pos, c, times[i], "SL", i, -sl_val)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= ad:
                    ts_p = h - td
                    if lo <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo >= ad:
                    ts_p = lo + td
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * PV)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_mk(pos, c, times[i], "Timeout", i, pnl_c)); exited = True
            if exited: pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(cur_atr) or cur_atr < 0.1: continue
        if adx[i] < adx_threshold: continue
        if hist[i] > 0 and hist[i-1] <= 0 and c > ema_t[i]:
            pos = {'dir': 'BUY', 'entry': c + spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
        elif hist[i] < 0 and hist[i-1] >= 0 and c < ema_t[i]:
            pos = {'dir': 'SELL', 'entry': c - spread/2, 'bar': i, 'time': times[i], 'atr': cur_atr}
    return trades


# ═══════════════════════════════════════════════════════════════
# Perturbation & grid functions for Stage 4 & 6
# ═══════════════════════════════════════════════════════════════

def macd_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20):
        return base * (1 + rng.uniform(-pct, pct))
    return backtest_macd(h1_df, spread, lot,
                         fast_period=max(5, int(p(12))),
                         slow_period=max(15, int(p(26))),
                         signal_period=max(3, int(p(9))),
                         ema_trend=max(50, int(p(100))),
                         adx_threshold=p(20),
                         sl_atr=p(4.5), tp_atr=p(6.0),
                         trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                         max_hold=max(5, int(p(20))))


def macd_grid_fn(h1_df, spread, lot):
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        for tp in [4.0, 5.0, 6.0, 8.0]:
            for adx in [15, 20, 25]:
                trades = backtest_macd(h1_df, spread, lot,
                                       sl_atr=sl, tp_atr=tp, adx_threshold=adx)
                daily = _trades_to_daily(trades)
                sh = _sharpe(daily)
                results[f"SL={sl}_TP={tp}_ADX={adx}"] = sh
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    from backtest.validator import StrategyValidator, ValidatorConfig

    t0 = time.time()
    print("=" * 72)
    print("  R80 — Full 8-Stage Validation: MACD_H1 on XAUUSD")
    print("  MACD(12,26,9) + EMA100 trend + ADX>20")
    print("  Exit: SL=4.5x, TP=6.0x, Trail 0.14/0.025, MH=20")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    config = ValidatorConfig(
        n_trials_tested=72,
        realistic_spread=REALISTIC_SPREAD,
        purge_bars=30,
        n_param_perturb=200,
        n_bootstrap=5000,
        n_trade_removal=500,
    )

    validator = StrategyValidator(
        name='MACD_H1',
        backtest_fn=backtest_macd,
        spread=SPREAD,
        lot=LOT,
        h1_df=h1_df,
        base_backtest_fn=backtest_macd,
        param_perturb_fn=macd_perturb_fn,
        param_grid_fn=macd_grid_fn,
        config=config,
        output_dir=str(OUTPUT_DIR),
    )

    results = validator.run_all(stop_on_fail=False)

    elapsed = time.time() - t0
    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)

    print(f"\n\n{'=' * 72}")
    print(f"  R80 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  MACD_H1: {passed}/{total} stages passed")
    print(f"{'=' * 72}")

    summary = {
        'strategy': 'MACD_H1',
        'passed': passed, 'total': total,
        'elapsed_s': round(elapsed, 1),
        'stages': {f"stage{s}": {'passed': r.passed, 'sharpe': r.sharpe, 'verdict': r.verdict}
                   for s, r in sorted(results.items())},
    }
    with open(OUTPUT_DIR / "r80_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
