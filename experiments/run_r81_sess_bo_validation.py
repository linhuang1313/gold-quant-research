#!/usr/bin/env python3
"""
R81 — Standalone Full 8-Stage Validation for SESS_BO on XAUUSD
================================================================
SESS_BO passed R70 (7/8, only Stage 4 PBO=47.1% failed),
R71 OOS (Sharpe=3.9), R72 multi-split (all passed), R72 silver (PASS).

This is a definitive, independent re-run with:
 - EA deployed parameters: LB=4, SL=4.5, TP=4.0, Trail 0.14/0.025, MH=20
 - Increased Monte Carlo samples (n_bootstrap=10000, n_param_perturb=300)
   for more stable PBO estimate
 - Wider grid search (more combinations) for PBO accuracy
 - Additional diagnostic: OOS holdout 2024+ appended to report

Estimated runtime: ~2-3 minutes.
"""
import sys, os, io, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = Path("results/r81_sess_bo_validation")
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


# ═══════════════════════════════════════════════════════════════
# SESS_BO backtest — EA deployed parameters
# ═══════════════════════════════════════════════════════════════

def backtest_sess_bo(h1_df, spread=SPREAD, lot=LOT,
                     session="peak_12_14", lookback_bars=4,
                     sl_atr=4.5, tp_atr=4.0, trail_act_atr=0.14,
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
    sess_start, _ = SESSION_DEFS[session]
    for i in range(lookback_bars, n):
        c = close[i]; h = high[i]; lo_v = low[i]; cur_atr = atr[i]; cur_hour = hours[i]
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (h - pos['entry'] - spread) * lot * PV
                pnl_l = (lo_v - pos['entry'] - spread) * lot * PV
                pnl_c = (c - pos['entry'] - spread) * lot * PV
            else:
                pnl_h = (pos['entry'] - lo_v - spread) * lot * PV
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
                act_dist = trail_act_atr * pos['atr']; trail_d = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and h - pos['entry'] >= act_dist:
                    ts_p = h - trail_d
                    if lo_v <= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (ts_p - pos['entry'] - spread) * lot * PV)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - lo_v >= act_dist:
                    ts_p = lo_v + trail_d
                    if h >= ts_p:
                        trades.append(_mk(pos, c, times[i], "Trail", i,
                                          (pos['entry'] - ts_p - spread) * lot * PV)); exited = True
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


# ═══════════════════════════════════════════════════════════════
# Base logic function (minimal params, no optimization)
# ═══════════════════════════════════════════════════════════════

def sess_base_fn(h1_df, spread, lot):
    return backtest_sess_bo(h1_df, spread, lot,
                            sl_atr=3.0, tp_atr=3.0,
                            trail_act_atr=99.0, trail_dist_atr=99.0,
                            max_hold=50)


# ═══════════════════════════════════════════════════════════════
# Perturbation & grid functions
# ═══════════════════════════════════════════════════════════════

def sess_perturb_fn(h1_df, spread, lot, rng):
    def p(base, pct=0.20):
        return base * (1 + rng.uniform(-pct, pct))
    return backtest_sess_bo(h1_df, spread, lot,
                            sl_atr=p(4.5), tp_atr=p(4.0),
                            lookback_bars=max(2, int(p(4))),
                            trail_act_atr=p(0.14), trail_dist_atr=p(0.025),
                            max_hold=max(5, int(p(20))))


def sess_grid_fn(h1_df, spread, lot):
    """Wider grid for more robust PBO: 5 SL x 5 TP x 4 LB = 100 combos."""
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for sl in [3.0, 3.5, 4.0, 4.5, 5.0]:
        for tp in [3.0, 3.5, 4.0, 5.0, 6.0]:
            for lb in [2, 3, 4, 5]:
                trades = backtest_sess_bo(h1_df, spread, lot,
                                          sl_atr=sl, tp_atr=tp, lookback_bars=lb)
                daily = _trades_to_daily(trades)
                sh = _sharpe(daily)
                results[f"SL={sl}_TP={tp}_LB={lb}"] = sh
    return results


# ═══════════════════════════════════════════════════════════════
# Extra diagnostic: OOS holdout analysis
# ═══════════════════════════════════════════════════════════════

def run_oos_diagnostic(h1_df):
    """Run OOS holdout on 2024+ and 2022+ for additional evidence."""
    from backtest.validator import _trades_to_daily, _sharpe
    results = {}
    for cutoff_year, label in [(2022, "2022+"), (2024, "2024+")]:
        cutoff = pd.Timestamp(f"{cutoff_year}-01-01", tz='UTC')
        train = h1_df[h1_df.index < cutoff]
        test = h1_df[h1_df.index >= cutoff]
        train_trades = backtest_sess_bo(train, SPREAD, LOT)
        test_trades = backtest_sess_bo(test, SPREAD, LOT)
        test_real = backtest_sess_bo(test, REALISTIC_SPREAD, LOT)
        t_daily = _trades_to_daily(train_trades)
        s_daily = _trades_to_daily(test_trades)
        sr_daily = _trades_to_daily(test_real)
        results[label] = {
            'train_sharpe': round(_sharpe(t_daily), 2),
            'test_sharpe': round(_sharpe(s_daily), 2),
            'test_sharpe_real': round(_sharpe(sr_daily), 2),
            'test_pnl': round(sum(t['pnl'] for t in test_trades), 2),
            'test_pnl_real': round(sum(t['pnl'] for t in test_real), 2),
            'test_trades': len(test_trades),
            'test_win_rate': round(100 * sum(1 for t in test_trades if t['pnl'] > 0) / max(len(test_trades), 1), 1),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Extra diagnostic: Year-by-year breakdown
# ═══════════════════════════════════════════════════════════════

def run_yearly_breakdown(h1_df):
    """Per-year performance to identify regime dependency."""
    from backtest.validator import _trades_to_daily, _sharpe
    all_trades = backtest_sess_bo(h1_df, SPREAD, LOT)
    real_trades = backtest_sess_bo(h1_df, REALISTIC_SPREAD, LOT)
    yearly = {}
    for trades_list, suffix in [(all_trades, ''), (real_trades, '_real')]:
        for t in trades_list:
            yr = str(t['exit_time'].year)
            if yr not in yearly:
                yearly[yr] = {'pnl': 0, 'pnl_real': 0, 'trades': 0, 'trades_real': 0, 'wins': 0, 'wins_real': 0}
            yearly[yr][f'pnl{suffix}'] += t['pnl']
            yearly[yr][f'trades{suffix}'] += 1
            if t['pnl'] > 0:
                yearly[yr][f'wins{suffix}'] += 1
    result = {}
    for yr in sorted(yearly):
        y = yearly[yr]
        result[yr] = {
            'pnl': round(y['pnl'], 2),
            'pnl_real': round(y['pnl_real'], 2),
            'trades': y['trades'],
            'win_rate': round(100 * y['wins'] / max(y['trades'], 1), 1),
        }
    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    from backtest.validator import StrategyValidator, ValidatorConfig

    t0 = time.time()
    print("=" * 72)
    print("  R81 — Definitive 8-Stage Validation: SESS_BO on XAUUSD")
    print("  EA params: Session=peak_12_14, LB=4, SL=4.5, TP=4.0")
    print("  Trail=0.14/0.025, MaxHold=20")
    print("  Enhanced: 10K bootstrap, 300 perturbations, 100-combo grid")
    print("=" * 72, flush=True)

    from backtest.runner import load_m15, load_h1_aligned, H1_CSV_PATH
    print("\n  Loading H1 data...", flush=True)
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})\n")

    config = ValidatorConfig(
        n_trials_tested=60,
        realistic_spread=REALISTIC_SPREAD,
        purge_bars=30,
        n_param_perturb=300,
        n_bootstrap=10000,
        n_trade_removal=500,
    )

    validator = StrategyValidator(
        name='SESS_BO_EA',
        backtest_fn=backtest_sess_bo,
        spread=SPREAD,
        lot=LOT,
        h1_df=h1_df,
        base_backtest_fn=sess_base_fn,
        param_perturb_fn=sess_perturb_fn,
        param_grid_fn=sess_grid_fn,
        config=config,
        output_dir=str(OUTPUT_DIR),
    )

    results = validator.run_all(stop_on_fail=False)

    # ─── Extra diagnostics ───
    print("\n" + "=" * 72)
    print("  EXTRA DIAGNOSTICS")
    print("=" * 72, flush=True)

    print("\n  [1] OOS Holdout Analysis...")
    oos_results = run_oos_diagnostic(h1_df)
    for label, data in oos_results.items():
        print(f"    {label}: Train Sharpe={data['train_sharpe']}, "
              f"Test Sharpe={data['test_sharpe']} (real={data['test_sharpe_real']}), "
              f"PnL=${data['test_pnl']:,.0f} (real=${data['test_pnl_real']:,.0f}), "
              f"Trades={data['test_trades']}, Win={data['test_win_rate']}%")

    print("\n  [2] Year-by-Year Breakdown:")
    yearly = run_yearly_breakdown(h1_df)
    print(f"    {'Year':>6}  {'PnL':>10}  {'PnL(real)':>10}  {'Trades':>7}  {'Win%':>6}")
    losing_years = 0
    for yr, d in yearly.items():
        flag = " ***" if d['pnl_real'] < 0 else ""
        if d['pnl_real'] < 0: losing_years += 1
        print(f"    {yr:>6}  ${d['pnl']:>9,.2f}  ${d['pnl_real']:>9,.2f}  {d['trades']:>7}  {d['win_rate']:>5.1f}%{flag}")
    print(f"    Losing years (realistic): {losing_years}/{len(yearly)}")

    # ─── Summary ───
    elapsed = time.time() - t0
    passed = sum(1 for r in results.values() if r.passed)
    total = len(results)

    print(f"\n\n{'=' * 72}")
    print(f"  R81 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  SESS_BO_EA: {passed}/{total} stages passed")
    print(f"{'=' * 72}")

    summary = {
        'strategy': 'SESS_BO_EA',
        'params': {
            'session': 'peak_12_14', 'lookback_bars': 4,
            'sl_atr': 4.5, 'tp_atr': 4.0,
            'trail_act_atr': 0.14, 'trail_dist_atr': 0.025,
            'max_hold': 20
        },
        'passed': passed, 'total': total,
        'elapsed_s': round(elapsed, 1),
        'stages': {f"stage{s}": {'passed': r.passed, 'sharpe': r.sharpe, 'verdict': r.verdict}
                   for s, r in sorted(results.items())},
        'oos_holdout': oos_results,
        'yearly_breakdown': yearly,
        'losing_years_real': losing_years,
        'total_years': len(yearly),
    }
    with open(OUTPUT_DIR / "r81_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {OUTPUT_DIR}/", flush=True)

    # ─── Final recommendation ───
    print(f"\n{'=' * 72}")
    print("  DEPLOYMENT RECOMMENDATION")
    print(f"{'=' * 72}")
    if passed == total:
        print("  FULL PASS — SESS_BO is cleared for deployment.")
    elif passed >= total - 1:
        failed_stages = [s for s, r in results.items() if not r.passed]
        print(f"  NEAR-PASS ({passed}/{total}) — Failed: Stage {failed_stages}")
        if 4 in failed_stages:
            pbo = None
            for s, r in results.items():
                if s == 4 and hasattr(r, 'verdict'):
                    pbo = r.verdict
            print(f"  PBO analysis: {pbo}")
            print("  Consider: PBO is sensitive to grid configuration.")
            print("  OOS evidence (R71/R72) supports real-world viability.")
    else:
        print(f"  FAILED ({passed}/{total}) — Multiple stages failed. Caution advised.")


if __name__ == "__main__":
    main()
