#!/usr/bin/env python3
"""R231: Cap-After-Fuse & Breakout-Distance Filter Validation
================================================================
Validates two proposed Keltner entry filters based on 5/12 Cap-trigger analysis.

Hypothesis P1: "Cap Fuse" — after a MaxLossCap exit, block same-direction
    re-entry for N hours (preventing the BUY→Cap→BUY→Cap→BUY→Cap loop).
    Grid: fuse_hours ∈ {1, 2, 4, 8, 12, 24}

Hypothesis P2: "Breakout Distance" — block entry when price is already
    too far beyond KC_upper/KC_lower (prevents chasing).
    Grid: max_distance_atr ∈ {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}

Validation for each config:
  1. Full-sample stats (Sharpe, N, WinRate, PnL, PF, MaxDD)
  2. 6-Fold K-Fold (require ≥4/6 positive Sharpe)
  3. Era breakdown (4 eras)
  4. Comparison vs baseline (no filter)

Uses LIVE_PARITY_KWARGS + maxloss_cap=80 to match live system.
"""
from __future__ import annotations
import sys, json, time, copy
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord
from backtest.stats import calc_stats
import indicators as signals_mod

OUTPUT_DIR = Path("results/r231_cap_breakout_filters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

# ── P1: Cap Fuse hours to test ──
CAP_FUSE_HOURS = [0, 1, 2, 4, 8, 12, 24]

# ── P2: Max breakout distance (in ATR multiples) ──
MAX_DISTANCE_ATR = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def pf(msg):
    print(msg, flush=True)

def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    pf(f'  -> saved {p}')


def filter_period(trades, start, end):
    ts_s = pd.Timestamp(start, tz='UTC')
    ts_e = pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def trade_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0, 'profit_factor': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    pf_val = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.9
    return {'n': n, 'pnl': round(float(pnls.sum()), 2), 'sharpe': round(sharpe, 3),
            'win_rate': round(100 * (pnls > 0).sum() / n, 2),
            'max_dd': round(float(dd.max()), 2), 'profit_factor': round(pf_val, 3)}


def kfold_6(trades):
    """6-Fold chronological K-Fold validation."""
    if len(trades) < 60:
        return {'skip': True, 'verdict': 'SKIP', 'reason': f'n={len(trades)}<60'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds, kf_pass = [], 0
    for fold in range(6):
        s = fold * fold_size
        e = s + fold_size if fold < 5 else len(pnls)
        fp = pnls[s:e]
        if len(fp) < 5:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0:
            kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.67 else 'FAIL'}


# ═══════════════════════════════════════════════════════════════
# P1: Cap Fuse — post-hoc filter on trade list
# ═══════════════════════════════════════════════════════════════

def apply_cap_fuse(trades: list, fuse_hours: float) -> list:
    """Remove trades that entered within fuse_hours of a same-direction Cap exit.
    
    Logic: After a MaxLossCap exit at time T in direction D,
    block all entries in direction D until T + fuse_hours.
    """
    if fuse_hours <= 0:
        return trades

    blocked_until_buy = pd.Timestamp.min.tz_localize('UTC')
    blocked_until_sell = pd.Timestamp.min.tz_localize('UTC')
    result = []

    for t in trades:
        entry_ts = pd.Timestamp(t.entry_time)
        exit_ts = pd.Timestamp(t.exit_time)
        direction = t.direction

        if direction == 'BUY' and entry_ts <= blocked_until_buy:
            continue
        if direction == 'SELL' and entry_ts <= blocked_until_sell:
            continue

        result.append(t)

        if t.exit_reason == 'MaxLossCap':
            fuse_end = exit_ts + pd.Timedelta(hours=fuse_hours)
            if direction == 'BUY':
                blocked_until_buy = max(blocked_until_buy, fuse_end)
            else:
                blocked_until_sell = max(blocked_until_sell, fuse_end)

    return result


# ═══════════════════════════════════════════════════════════════
# P2: Breakout Distance Filter — signal wrapper
# ═══════════════════════════════════════════════════════════════

def make_keltner_distance_filter(max_dist_atr: float):
    """Create a patched check_keltner_signal that rejects entries too far from KC band.
    
    If max_dist_atr == 0, returns original function (no filter).
    Otherwise: reject BUY when (close - KC_upper) > max_dist_atr * ATR
               reject SELL when (KC_lower - close) > max_dist_atr * ATR
    """
    original_fn = signals_mod._original_check_keltner_signal

    if max_dist_atr <= 0:
        return original_fn

    def filtered_keltner(df):
        sig = original_fn(df)
        if sig is None:
            return None

        latest = df.iloc[-1]
        atr = float(latest.get('ATR', 0))
        if atr <= 0:
            return sig

        close = float(latest['Close'])
        kc_upper = float(latest['KC_upper'])
        kc_lower = float(latest['KC_lower'])
        threshold = max_dist_atr * atr

        if sig['signal'] == 'BUY' and (close - kc_upper) > threshold:
            return None
        if sig['signal'] == 'SELL' and (kc_lower - close) > threshold:
            return None
        return sig

    return filtered_keltner


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_engine(data: DataBundle, kw_override: dict = None) -> list:
    """Run BacktestEngine with LIVE_PARITY + Cap80 + optional overrides."""
    kw = dict(LIVE_PARITY_KWARGS)
    kw['maxloss_cap'] = 80
    if kw_override:
        kw.update(kw_override)
    engine = BacktestEngine(data.m15_df, data.h1_df, **kw)
    engine.run()
    return [t for t in engine.trades if t.strategy == 'keltner']


def full_eval(trades, label=""):
    """Run full evaluation: stats + eras + K-Fold."""
    s = trade_stats(trades)
    eras = {}
    for en, (es, ee) in ERA_SEGMENTS.items():
        eras[en] = trade_stats(filter_period(trades, es, ee))
    kf = kfold_6(trades)
    return {'label': label, 'stats': s, 'eras': eras, 'kfold': kf}


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R231: Cap-Fuse & Breakout-Distance Filter Validation')
    pf('=' * 80)

    data = DataBundle.load_default()

    # Save original keltner signal function for P2 patching
    signals_mod._original_check_keltner_signal = signals_mod.check_keltner_signal

    # ── Step 1: Baseline (no filters) ──
    pf(f'\n{"="*80}\nStep 1: Baseline (LIVE_PARITY + Cap80, no filters)\n{"="*80}')
    baseline_trades = run_engine(data)
    baseline = full_eval(baseline_trades, 'baseline')
    bs = baseline['stats']
    pf(f'  Baseline: n={bs["n"]}  Sh={bs["sharpe"]:.3f}  PnL=${bs["pnl"]:.0f}  '
       f'WR={bs["win_rate"]:.1f}%  PF={bs["profit_factor"]:.2f}  MaxDD=${bs["max_dd"]:.0f}')
    pf(f'  K-Fold: {baseline["kfold"]["verdict"]} ({baseline["kfold"].get("pass_count",0)}/6)')
    for en, es in baseline['eras'].items():
        pf(f'    {en:<30} n={es["n"]:>4} Sh={es["sharpe"]:.3f}')

    cap_count = sum(1 for t in baseline_trades if t.exit_reason == 'MaxLossCap')
    pf(f'  Cap triggers in baseline: {cap_count}')

    # Count same-direction re-entries after Cap
    cap_reentries = 0
    for i, t in enumerate(baseline_trades):
        if t.exit_reason == 'MaxLossCap' and i + 1 < len(baseline_trades):
            nxt = baseline_trades[i + 1]
            if nxt.direction == t.direction:
                gap_h = (pd.Timestamp(nxt.entry_time) - pd.Timestamp(t.exit_time)).total_seconds() / 3600
                if gap_h < 24:
                    cap_reentries += 1
    pf(f'  Same-dir re-entries within 24h of Cap: {cap_reentries}')

    save('step1_baseline', baseline)

    # ── Step 2: P1 — Cap Fuse sweep ──
    pf(f'\n{"="*80}\nStep 2: P1 — Cap Fuse (same-direction block after Cap)\n{"="*80}')
    p1_results = {}
    for fh in CAP_FUSE_HOURS:
        label = f'cap_fuse_{fh}h'
        filtered = apply_cap_fuse(baseline_trades, fh)
        ev = full_eval(filtered, label)
        s = ev['stats']
        removed = bs['n'] - s['n']
        pf(f'  Fuse={fh:>2}h: n={s["n"]:>4} ({removed:>3} removed)  Sh={s["sharpe"]:.3f}  '
           f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  KF={ev["kfold"]["verdict"]}')
        p1_results[label] = {**ev, 'fuse_hours': fh, 'trades_removed': removed}
    save('step2_p1_cap_fuse', p1_results)

    # ── Step 3: P2 — Breakout Distance sweep ──
    pf(f'\n{"="*80}\nStep 3: P2 — Breakout Distance Filter\n{"="*80}')
    p2_results = {}
    for md in MAX_DISTANCE_ATR:
        label = f'dist_{md:.1f}atr'
        patched_fn = make_keltner_distance_filter(md)
        signals_mod.check_keltner_signal = patched_fn
        trades = run_engine(data)
        ev = full_eval(trades, label)
        s = ev['stats']
        removed = bs['n'] - s['n']

        cap_ct = sum(1 for t in trades if t.exit_reason == 'MaxLossCap')
        pf(f'  MaxDist={md:.1f}×ATR: n={s["n"]:>4} ({removed:>3} removed)  Sh={s["sharpe"]:.3f}  '
           f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  Caps={cap_ct}  KF={ev["kfold"]["verdict"]}')
        p2_results[label] = {**ev, 'max_dist_atr': md, 'trades_removed': removed, 'cap_count': cap_ct}

    signals_mod.check_keltner_signal = signals_mod._original_check_keltner_signal
    save('step3_p2_breakout_dist', p2_results)

    # ── Step 4: Combined — best P1 × best P2 ──
    pf(f'\n{"="*80}\nStep 4: Combined (best P1 + best P2)\n{"="*80}')

    best_p1 = max(
        [(fh, p1_results[f'cap_fuse_{fh}h']['stats']['sharpe'])
         for fh in CAP_FUSE_HOURS if p1_results[f'cap_fuse_{fh}h']['kfold']['verdict'] == 'PASS'],
        key=lambda x: x[1], default=(0, 0)
    )
    best_p2 = max(
        [(md, p2_results[f'dist_{md:.1f}atr']['stats']['sharpe'])
         for md in MAX_DISTANCE_ATR if p2_results[f'dist_{md:.1f}atr']['kfold']['verdict'] == 'PASS'],
        key=lambda x: x[1], default=(0, 0)
    )

    pf(f'  Best P1: fuse={best_p1[0]}h (Sh={best_p1[1]:.3f})')
    pf(f'  Best P2: dist={best_p2[0]:.1f}×ATR (Sh={best_p2[1]:.3f})')

    if best_p1[0] > 0 or best_p2[0] > 0:
        patched_fn = make_keltner_distance_filter(best_p2[0])
        signals_mod.check_keltner_signal = patched_fn
        combined_trades = run_engine(data)
        signals_mod.check_keltner_signal = signals_mod._original_check_keltner_signal

        if best_p1[0] > 0:
            combined_trades = apply_cap_fuse(combined_trades, best_p1[0])

        combined = full_eval(combined_trades, f'combined_fuse{best_p1[0]}h_dist{best_p2[0]:.1f}')
        cs = combined['stats']
        pf(f'\n  Combined: n={cs["n"]}  Sh={cs["sharpe"]:.3f}  PnL=${cs["pnl"]:.0f}  '
           f'WR={cs["win_rate"]:.1f}%  PF={cs["profit_factor"]:.2f}  MaxDD=${cs["max_dd"]:.0f}')
        pf(f'  K-Fold: {combined["kfold"]["verdict"]} ({combined["kfold"].get("pass_count",0)}/6)')
        for en, es in combined['eras'].items():
            pf(f'    {en:<30} n={es["n"]:>4} Sh={es["sharpe"]:.3f}')
        save('step4_combined', combined)

    # ── Step 5: P2 deep dive — analyze what gets filtered ──
    pf(f'\n{"="*80}\nStep 5: Breakout Distance Analysis (what gets filtered)\n{"="*80}')

    patched_fn = make_keltner_distance_filter(0.0)
    signals_mod.check_keltner_signal = patched_fn
    all_trades = run_engine(data)
    signals_mod.check_keltner_signal = signals_mod._original_check_keltner_signal

    dist_analysis = {'buy': [], 'sell': []}
    for t in all_trades:
        entry_ts = pd.Timestamp(t.entry_time)
        h1_idx = data.h1_df.index.get_indexer([entry_ts], method='ffill')[0]
        if h1_idx < 0 or h1_idx >= len(data.h1_df):
            continue
        row = data.h1_df.iloc[h1_idx]
        atr = float(row.get('ATR', 0))
        kc_u = float(row.get('KC_upper', 0))
        kc_l = float(row.get('KC_lower', 0))
        if atr <= 0 or kc_u == 0:
            continue

        hold_h = (pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600

        if t.direction == 'BUY':
            dist = (t.entry_price - kc_u) / atr
            dist_analysis['buy'].append({'dist_atr': round(dist, 3), 'pnl': t.pnl,
                                          'exit_reason': t.exit_reason, 'hold_h': round(hold_h, 2)})
        else:
            dist = (kc_l - t.entry_price) / atr
            dist_analysis['sell'].append({'dist_atr': round(dist, 3), 'pnl': t.pnl,
                                           'exit_reason': t.exit_reason, 'hold_h': round(hold_h, 2)})

    for side in ['buy', 'sell']:
        entries = dist_analysis[side]
        if not entries:
            continue
        dists = [e['dist_atr'] for e in entries]
        pnls = [e['pnl'] for e in entries]
        pf(f'\n  {side.upper()} entries: {len(entries)}')
        for bucket_label, lo, hi in [('≤0.3', -99, 0.3), ('0.3-0.5', 0.3, 0.5),
                                      ('0.5-1.0', 0.5, 1.0), ('1.0-1.5', 1.0, 1.5), ('>1.5', 1.5, 99)]:
            bucket = [(d, p) for d, p in zip(dists, pnls) if lo < d <= hi]
            if not bucket:
                pf(f'    dist {bucket_label:<8}: n=0')
                continue
            bp = [p for _, p in bucket]
            wr = 100 * sum(1 for p in bp if p > 0) / len(bp)
            cap_n = sum(1 for e in entries if lo < e['dist_atr'] <= hi and e['exit_reason'] == 'MaxLossCap')
            pf(f'    dist {bucket_label:<8}: n={len(bucket):>4}  WR={wr:.1f}%  '
               f'avg_pnl=${np.mean(bp):.2f}  caps={cap_n}')

    save('step5_distance_analysis', dist_analysis)

    # ── Final Summary ──
    pf(f'\n{"="*80}\nFINAL SUMMARY\n{"="*80}')
    pf(f'\n  Baseline:     n={bs["n"]}  Sh={bs["sharpe"]:.3f}  PnL=${bs["pnl"]:.0f}  KF={baseline["kfold"]["verdict"]}')
    pf(f'\n  P1 Cap Fuse results:')
    for fh in CAP_FUSE_HOURS:
        r = p1_results[f'cap_fuse_{fh}h']
        s = r['stats']
        delta_sh = s['sharpe'] - bs['sharpe']
        pf(f'    {fh:>2}h: Sh={s["sharpe"]:.3f} ({"+" if delta_sh>=0 else ""}{delta_sh:.3f})  '
           f'n={s["n"]:>4}  KF={r["kfold"]["verdict"]}')

    pf(f'\n  P2 Breakout Distance results:')
    for md in MAX_DISTANCE_ATR:
        r = p2_results[f'dist_{md:.1f}atr']
        s = r['stats']
        delta_sh = s['sharpe'] - bs['sharpe']
        pf(f'    {md:.1f}×ATR: Sh={s["sharpe"]:.3f} ({"+" if delta_sh>=0 else ""}{delta_sh:.3f})  '
           f'n={s["n"]:>4}  Caps={r["cap_count"]:>3}  KF={r["kfold"]["verdict"]}')

    # Deployment recommendation
    pf(f'\n  ═══ Deployment Recommendation ═══')
    if best_p1[0] > 0:
        pf(f'  P1: Deploy cap_fuse={best_p1[0]}h if Sharpe improved and K-Fold PASS')
    else:
        pf(f'  P1: No improvement found — cap fuse not recommended')
    if best_p2[0] > 0:
        pf(f'  P2: Deploy max_breakout_dist={best_p2[0]:.1f}×ATR if Sharpe improved and K-Fold PASS')
    else:
        pf(f'  P2: No improvement found — distance filter not recommended')

    elapsed = time.time() - t0
    pf(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/3600:.1f}h)')
    pf(f'  Finished: {pd.Timestamp.now()}')


if __name__ == '__main__':
    main()
