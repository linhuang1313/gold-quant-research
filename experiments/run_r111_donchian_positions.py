#!/usr/bin/env python3
"""
R111 — Donchian max_positions × MaxLoss Cap 交叉测试
=====================================================
R102 回测默认隐含 max_positions=1 且只在最优参数上做了 Cap 扫描。
本实验补充:

  Phase 1: max_positions 1/2/3 单独对比 (R102最优参数 ch=60)
  Phase 2: max_positions × Cap 交叉网格 (2维扫描)
  Phase 3: K-Fold 验证 Top-5 组合
  Phase 4: 最优 Donchian 配置融入 5 策略组合 + Lot 优化
  Phase 5: 5 策略组合 K-Fold 对比
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, load_m15, load_h1_aligned, H1_CSV_PATH
from backtest.runner import run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r111_donchian_positions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

CAPS_OTHER = {'L8_MAX': 35, 'PSAR': 5, 'TSMOM': 0, 'SESS_BO': 35}
R89_LOTS = {'L8_MAX': 0.02, 'PSAR': 0.09, 'TSMOM': 0.09, 'SESS_BO': 0.09}

DONCH_BASE = {'channel': 60, 'sl_atr': 4.0, 'tp_atr': 3.0, 'max_hold': 20,
              'trail_act': 0.14, 'trail_dist': 0.025}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr
    tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            pnl = -sl * lot * pv
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, pnl)
        if hi >= pos['entry'] + tp:
            pnl = tp * lot * pv
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, pnl)
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if extreme - pos['entry'] >= act_dist:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                pnl = (trail_price - pos['entry'] - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (cl - pos['entry'] - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            pnl = -sl * lot * pv
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, pnl)
        if lo <= pos['entry'] - tp:
            pnl = tp * lot * pv
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, pnl)
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if pos['entry'] - extreme >= act_dist:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                pnl = (pos['entry'] - trail_price - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (pos['entry'] - cl - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    return None


# ═══════════════════════════════════════════════════════════════
# Donchian backtest — supports multi-position
# ═══════════════════════════════════════════════════════════════

def bt_donchian_mp(h1_df, spread, lot, maxloss_cap=0, max_positions=1,
                   channel=60, sl_atr=4.0, tp_atr=3.0,
                   trail_act=0.14, trail_dist=0.025, max_hold=20):
    """Donchian backtest with max_positions support."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['HH'] = df['High'].rolling(channel).max()
    df['LL'] = df['Low'].rolling(channel).min()
    df = df.dropna(subset=['ATR', 'HH', 'LL'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    atr = df['ATR'].values
    hh = df['HH'].values
    ll = df['LL'].values
    times = df.index
    n = len(df)
    trades = []
    positions = []
    last_exit = -999

    for i in range(1, n):
        closed_this_bar = []
        for pos in positions:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result)
                closed_this_bar.append(pos)
                last_exit = i

        for pos in closed_this_bar:
            positions.remove(pos)

        if len(positions) >= max_positions:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue

        if c[i] > hh[i - 1]:
            same_dir = sum(1 for p in positions if p['dir'] == 'BUY')
            if same_dir < max_positions:
                positions.append({'dir': 'BUY', 'entry': c[i] + spread / 2,
                                  'bar': i, 'time': times[i], 'atr': atr[i]})
        elif c[i] < ll[i - 1]:
            same_dir = sum(1 for p in positions if p['dir'] == 'SELL')
            if same_dir < max_positions:
                positions.append({'dir': 'SELL', 'entry': c[i] - spread / 2,
                                  'bar': i, 'time': times[i], 'atr': atr[i]})

    return trades


# ═══════════════════════════════════════════════════════════════
# Other strategies (from R102, for portfolio phase)
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True
    ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0,i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0,i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar


def bt_psar(h1_df, spread, lot, maxloss_cap=0,
            sl_atr=4.5, tp_atr=16.0, trail_act=0.20, trail_dist=0.04,
            max_hold=20, af_step=0.01, af_max=0.05, min_atr=0.1):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    add_psar(df, af_step, af_max)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < min_atr: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
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
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
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
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback+1))
        ll_val = min(lo[i-j] for j in range(1, lookback+1))
        if c[i] > hh:
            pos = {'dir':'BUY','entry':c[i]+spread/2,'bar':i,'time':times[i],'atr':atr[i]}
        elif c[i] < ll_val:
            pos = {'dir':'SELL','entry':c[i]-spread/2,'bar':i,'time':times[i],'atr':atr[i]}
    return trades


def bt_l8_max(data_bundle, spread, lot, maxloss_cap=35):
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': maxloss_cap,
           'spread_cost': spread, 'initial_capital': 2000,
           'min_lot_size': lot, 'max_lot_size': lot}
    result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
    raw_trades = result.get('_trades', [])
    trades = []
    for t in raw_trades:
        trades.append({
            'dir': t.direction, 'entry': t.entry_price, 'exit': t.exit_price,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'pnl': t.pnl, 'reason': t.exit_reason, 'bars': 0,
        })
    return trades


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_series(trades):
    if not trades: return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def win_rate(trades):
    if not trades: return 0.0
    return sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100


def avg_pnl(trades):
    if not trades: return 0.0
    return sum(t['pnl'] for t in trades) / len(trades)


def metrics(trades):
    daily = trades_to_daily_series(trades)
    return {
        'n_trades': len(trades),
        'sharpe': round(sharpe(daily.values), 3) if len(daily) > 0 else 0,
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'max_dd': round(max_dd(daily.values), 2) if len(daily) > 0 else 0,
        'wr': round(win_rate(trades), 1),
        'avg_pnl': round(avg_pnl(trades), 3),
    }


def build_portfolio(trade_dict, lots):
    all_daily = {}
    for name, trades in trade_dict.items():
        lot = lots.get(name, UNIT_LOT)
        scale = lot / UNIT_LOT
        for t in trades:
            d = pd.Timestamp(t['exit_time']).date()
            all_daily[d] = all_daily.get(d, 0) + t['pnl'] * scale
    dates = sorted(all_daily.keys())
    return pd.Series([all_daily[d] for d in dates], index=pd.DatetimeIndex(dates))


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R111 — Donchian max_positions × MaxLoss Cap 交叉测试")
    print("=" * 80)

    print("\n  Loading data...")
    m15_raw = load_m15()
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    bundle = DataBundle.load_custom()
    print(f"    H1: {len(h1_df)} bars ({h1_df.index[0].date()} ~ {h1_df.index[-1].date()})")

    results = {'experiment': 'R111 Donchian max_positions x Cap'}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: max_positions isolated comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: max_positions = 1 / 2 / 3 (无Cap, R102最优参数)")
    print("=" * 60)

    phase1 = {}
    for mp in [1, 2, 3]:
        trades = bt_donchian_mp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0,
                                max_positions=mp, **DONCH_BASE)
        m = metrics(trades)
        phase1[f"mp={mp}"] = m
        reasons = {}
        for t in trades:
            r = t.get('reason', 'Unknown')
            reasons[r] = reasons.get(r, 0) + 1
        print(f"\n  max_positions={mp}:")
        print(f"    {m['n_trades']} trades, Sharpe={m['sharpe']}, PnL=${m['pnl']}, "
              f"MaxDD=${m['max_dd']}, WR={m['wr']}%")
        print(f"    Exit reasons: {reasons}")

        # Year-by-year
        for year in range(2015, 2027):
            yt = [t for t in trades if pd.Timestamp(t['exit_time']).year == year]
            if yt:
                ym = metrics(yt)
                print(f"      {year}: {ym['n_trades']:4d} trades, Sharpe={ym['sharpe']:6.2f}, "
                      f"PnL=${ym['pnl']:8.2f}")

    results['phase1_positions'] = phase1

    # ═══════════════════════════════════════════════════════════
    # Phase 2: max_positions × Cap cross grid
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: max_positions × Cap 交叉网格")
    print("=" * 60)

    mp_range = [1, 2, 3]
    cap_range = [0, 5, 10, 15, 20, 25, 30, 35, 50]
    grid = []

    for mp in mp_range:
        for cap in cap_range:
            trades = bt_donchian_mp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=cap,
                                    max_positions=mp, **DONCH_BASE)
            m = metrics(trades)
            row = {'max_positions': mp, 'cap': cap, **m}
            grid.append(row)
            print(f"    mp={mp} cap=${cap:2d}: Sharpe={m['sharpe']:6.3f} PnL=${m['pnl']:8.0f} "
                  f"MaxDD=${m['max_dd']:6.0f} Trades={m['n_trades']} WR={m['wr']:.1f}%")

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    results['phase2_grid'] = grid

    print(f"\n  Top 5 configurations:")
    for i, g in enumerate(grid[:5]):
        print(f"    #{i+1}: mp={g['max_positions']} cap=${g['cap']} -> "
              f"Sharpe={g['sharpe']}, PnL=${g['pnl']}, MaxDD=${g['max_dd']}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: K-Fold on Top-5
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: K-Fold 验证 Top-5 配置")
    print("=" * 60)

    kfold_results = {}
    for rank, g in enumerate(grid[:5]):
        mp = g['max_positions']
        cap = g['cap']
        label = f"mp={mp}/cap=${cap}"
        fold_sharpes = []
        for fname, start, end in FOLDS:
            fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
            if len(fold_h1) < 100:
                fold_sharpes.append(0.0)
                continue
            trades = bt_donchian_mp(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=cap,
                                    max_positions=mp, **DONCH_BASE)
            m = metrics(trades)
            fold_sharpes.append(m['sharpe'])

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = np.mean(fold_sharpes)
        passed = positive >= 4
        kfold_results[label] = {
            'max_positions': mp, 'cap': cap,
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'positive_folds': positive,
            'mean_sharpe': round(mean_sh, 3),
            'pass_4of6': passed,
        }
        status = "PASS" if passed else "FAIL"
        print(f"  #{rank+1} {label}: folds={[f'{s:.2f}' for s in fold_sharpes]} -> "
              f"{positive}/6 [{status}] mean={mean_sh:.3f}")

    results['phase3_kfold'] = kfold_results

    # Pick best validated config
    best_validated = None
    for label, info in kfold_results.items():
        if info['pass_4of6']:
            if best_validated is None or info['mean_sharpe'] > best_validated[1]['mean_sharpe']:
                best_validated = (label, info)

    if best_validated:
        best_mp = best_validated[1]['max_positions']
        best_cap = best_validated[1]['cap']
        print(f"\n  Best validated: {best_validated[0]} (mean Sharpe={best_validated[1]['mean_sharpe']})")
    else:
        best_mp = 1
        best_cap = 0
        print(f"\n  No config passed K-Fold. Falling back to mp=1/cap=0")

    results['best_config'] = {'max_positions': best_mp, 'cap': best_cap}

    # ═══════════════════════════════════════════════════════════
    # Phase 4: 5-strategy portfolio lot optimization
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  Phase 4: 5策略组合 Lot优化 (Donchian mp={best_mp}, cap=${best_cap})")
    print("=" * 60)

    print("  Running 4 base strategies...")
    psar_trades = bt_psar(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['PSAR'])
    tsmom_trades = bt_tsmom(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['TSMOM'])
    sess_trades = bt_sess_bo(h1_df, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['SESS_BO'])
    l8_trades = bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['L8_MAX'])
    donch_trades = bt_donchian_mp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=best_cap,
                                  max_positions=best_mp, **DONCH_BASE)

    strat_trades = {
        'PSAR': psar_trades, 'TSMOM': tsmom_trades,
        'SESS_BO': sess_trades, 'L8_MAX': l8_trades,
        'DONCH': donch_trades,
    }

    print("\n  Individual (unit lot):")
    for name, trades in strat_trades.items():
        m = metrics(trades)
        print(f"    {name:10s}: {m['n_trades']:5d} trades, Sharpe={m['sharpe']:6.3f}, "
              f"PnL=${m['pnl']:10.2f}, MaxDD=${m['max_dd']:7.2f}")

    # 4-strat baseline
    base4 = {k: v for k, v in strat_trades.items() if k != 'DONCH'}
    port4 = build_portfolio(base4, R89_LOTS)
    sh4 = sharpe(port4.values)
    dd4 = max_dd(port4.values)
    pnl4 = port4.sum()
    print(f"\n  Current 4-strat: Sharpe={sh4:.3f}, PnL=${pnl4:.0f}, MaxDD=${dd4:.0f}")

    donch_lot_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    lot_results = []
    for dl in donch_lot_range:
        lots5 = {**R89_LOTS, 'DONCH': dl}
        p5 = build_portfolio(strat_trades, lots5)
        sh = sharpe(p5.values)
        dd = max_dd(p5.values)
        pnl = p5.sum()
        lot_results.append({'donch_lot': dl, 'sharpe': round(sh, 3),
                            'max_dd': round(dd, 2), 'pnl': round(pnl, 2)})
        marker = " <-- best" if sh == max(r['sharpe'] for r in lot_results) else ""
        print(f"    DONCH={dl:.2f}: Sharpe={sh:.3f}, PnL=${pnl:.0f}, MaxDD=${dd:.0f}{marker}")

    best_lot = max(lot_results, key=lambda x: x['sharpe'])
    results['phase4_lots'] = lot_results
    results['best_lot'] = best_lot
    print(f"\n  Best lot: {best_lot['donch_lot']} (Sharpe={best_lot['sharpe']} vs 4-strat {sh4:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Phase 5: 5-strategy portfolio K-Fold comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 5: 5策略组合 K-Fold 对比")
    print("=" * 60)

    best_dl = best_lot['donch_lot']
    lots5_best = {**R89_LOTS, 'DONCH': best_dl}

    fold_sh4 = []
    fold_sh5 = []
    for fname, start, end in FOLDS:
        fold_h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(fold_h1) < 100:
            fold_sh4.append(0.0)
            fold_sh5.append(0.0)
            continue
        ft = {
            'PSAR': bt_psar(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['PSAR']),
            'TSMOM': bt_tsmom(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['TSMOM']),
            'SESS_BO': bt_sess_bo(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['SESS_BO']),
            'L8_MAX': bt_l8_max(bundle, SPREAD, UNIT_LOT, maxloss_cap=CAPS_OTHER['L8_MAX']),
        }
        p4 = build_portfolio(ft, R89_LOTS)
        fold_sh4.append(sharpe(p4.values))

        ft['DONCH'] = bt_donchian_mp(fold_h1, SPREAD, UNIT_LOT, maxloss_cap=best_cap,
                                     max_positions=best_mp, **DONCH_BASE)
        p5 = build_portfolio(ft, lots5_best)
        fold_sh5.append(sharpe(p5.values))
        print(f"    {fname}: 4-strat={fold_sh4[-1]:.3f}, 5-strat={fold_sh5[-1]:.3f} "
              f"{'✓' if fold_sh5[-1] > fold_sh4[-1] else '✗'}")

    wins5 = sum(1 for a, b in zip(fold_sh5, fold_sh4) if a > b)
    print(f"\n  4-strat folds: {[f'{s:.3f}' for s in fold_sh4]} mean={np.mean(fold_sh4):.3f}")
    print(f"  5-strat folds: {[f'{s:.3f}' for s in fold_sh5]} mean={np.mean(fold_sh5):.3f}")
    print(f"  5-strat wins: {wins5}/6 folds")

    results['phase5_portfolio_kfold'] = {
        '4strat_folds': [round(s, 3) for s in fold_sh4],
        '5strat_folds': [round(s, 3) for s in fold_sh5],
        '4strat_mean': round(np.mean(fold_sh4), 3),
        '5strat_mean': round(np.mean(fold_sh5), 3),
        '5strat_wins': wins5,
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 6: R102 vs R111 comparison (mp=1 vs best mp)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 6: R102配置 vs R111最优配置 对比")
    print("=" * 60)

    r102_trades = bt_donchian_mp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=0,
                                 max_positions=1, **DONCH_BASE)
    r111_trades = bt_donchian_mp(h1_df, SPREAD, UNIT_LOT, maxloss_cap=best_cap,
                                 max_positions=best_mp, **DONCH_BASE)
    r102_m = metrics(r102_trades)
    r111_m = metrics(r111_trades)

    print(f"  R102 (mp=1, cap=0):   Sharpe={r102_m['sharpe']}, Trades={r102_m['n_trades']}, "
          f"PnL=${r102_m['pnl']}, MaxDD=${r102_m['max_dd']}")
    print(f"  R111 (mp={best_mp}, cap=${best_cap}): Sharpe={r111_m['sharpe']}, Trades={r111_m['n_trades']}, "
          f"PnL=${r111_m['pnl']}, MaxDD=${r111_m['max_dd']}")

    results['phase6_comparison'] = {
        'r102': {'mp': 1, 'cap': 0, **r102_m},
        'r111': {'mp': best_mp, 'cap': best_cap, **r111_m},
    }

    # ═══════════════════════════════════════════════════════════
    # Final recommendation
    # ═══════════════════════════════════════════════════════════
    if best_validated:
        rec = (f"Donchian: mp={best_mp}, cap=${best_cap}, lot={best_dl}, "
               f"5-strat wins {wins5}/6")
    else:
        rec = "Keep R102 config: mp=1, cap=0, lot=0.03"

    results['recommendation'] = rec
    results['elapsed_s'] = round(time.time() - t0, 1)

    out_file = OUTPUT_DIR / "r111_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  R111 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  RECOMMENDATION: {rec}")
    print(f"{'='*80}")
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()
