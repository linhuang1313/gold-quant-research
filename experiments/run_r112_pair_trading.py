#!/usr/bin/env python3
"""
R112 — Gold/Silver Pair Trading (均值回归策略)
==============================================
Edward Thorp 核心思想: 不赌方向, 只赚定价偏离的钱.

Gold/Silver Ratio 长期均值约 69, 但经常偏离到 50-100+.
当 ratio 极端偏离时做均值回归:
  - Ratio 过高(Z>阈值): 做空Gold + 做多Silver (等ratio回落)
  - Ratio 过低(Z<-阈值): 做多Gold + 做空Silver (等ratio回升)

本实验:
  Phase 1: Daily ratio 基础统计与可视化
  Phase 2: Z-Score 均值回归策略回测 (参数扫描)
  Phase 3: Bollinger Band ratio 策略
  Phase 4: Half-Life 最优回看期测试
  Phase 5: K-Fold 验证 Top-5 配置
  Phase 6: 与现有4策略组合的相关性分析
  Phase 7: Gold-only 实现 (仅交易黄金端, 用ratio做信号)
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r112_pair_trading")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
PV_GOLD = 100
SPREAD_GOLD = 0.30
UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2002-01-01", "2006-01-01"),
    ("Fold2", "2006-01-01", "2010-01-01"),
    ("Fold3", "2010-01-01", "2014-01-01"),
    ("Fold4", "2014-01-01", "2018-01-01"),
    ("Fold5", "2018-01-01", "2022-01-01"),
    ("Fold6", "2022-01-01", "2026-06-01"),
]

t0 = time.time()


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_daily_data():
    gd = pd.read_csv(DATA_DIR / "xauusd_daily_yf.csv", index_col=0, parse_dates=True)
    sd = pd.read_csv(DATA_DIR / "xagusd_daily_yf.csv", index_col=0, parse_dates=True)
    if gd.index.tz is not None:
        gd.index = gd.index.tz_localize(None)
    if sd.index.tz is not None:
        sd.index = sd.index.tz_localize(None)
    merged = pd.DataFrame({
        'gold_close': gd['Close'], 'gold_high': gd['High'], 'gold_low': gd['Low'],
        'silver_close': sd['Close'], 'silver_high': sd['High'], 'silver_low': sd['Low'],
    }).dropna()
    merged['ratio'] = merged['gold_close'] / merged['silver_close']
    return merged


def load_h1_data():
    gh = pd.read_csv(DATA_DIR / "xauusd_h1_yf.csv", index_col=0, parse_dates=True)
    sh = pd.read_csv(DATA_DIR / "xagusd_h1_yf.csv", index_col=0, parse_dates=True)
    if gh.index.tz is not None:
        gh.index = gh.index.tz_localize(None)
    if sh.index.tz is not None:
        sh.index = sh.index.tz_localize(None)
    merged = pd.DataFrame({
        'gold_close': gh['Close'], 'gold_high': gh['High'], 'gold_low': gh['Low'],
        'silver_close': sh['Close'], 'silver_high': sh['High'], 'silver_low': sh['Low'],
    }).dropna()
    merged['ratio'] = merged['gold_close'] / merged['silver_close']
    return merged


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def sharpe(arr):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def metrics_from_trades(trades, annualize=252):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0,
                'avg_hold': 0, 'profit_factor': 0}
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    arr = np.array([daily[d] for d in dates])
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    total_win = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0.01
    return {
        'n': len(trades),
        'sharpe': round(sharpe(arr), 3),
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'max_dd': round(max_dd(arr), 2),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'avg_pnl': round(sum(t['pnl'] for t in trades) / len(trades), 3),
        'avg_hold': round(np.mean([t.get('hold_days', 0) for t in trades]), 1),
        'profit_factor': round(total_win / total_loss, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Strategy 1: Z-Score Mean Reversion (Daily)
# ═══════════════════════════════════════════════════════════════

def bt_zscore_ratio(df, lookback=20, entry_z=2.0, exit_z=0.0,
                    stop_z=4.0, max_hold=30, lot=UNIT_LOT):
    """
    Z-Score ratio 均值回归:
    - Z > entry_z: ratio偏高, 做空gold (ratio会回落 = gold跌或silver涨)
    - Z < -entry_z: ratio偏低, 做多gold
    - Z回到exit_z附近平仓
    - Z超过stop_z止损
    - 最长持仓max_hold天
    
    Gold-only实现: 只交易黄金端, 因为我们MT4只有黄金
    """
    ratio = df['ratio'].values
    gold = df['gold_close'].values
    times = df.index
    n = len(df)

    rm = pd.Series(ratio).rolling(lookback).mean().values
    rs = pd.Series(ratio).rolling(lookback).std().values

    trades = []
    pos = None

    for i in range(lookback + 1, n):
        if np.isnan(rm[i]) or np.isnan(rs[i]) or rs[i] < 0.01:
            continue

        z = (ratio[i] - rm[i]) / rs[i]

        if pos is not None:
            hold = i - pos['bar']
            pnl_gold = 0
            if pos['dir'] == 'SELL':
                pnl_gold = (pos['gold_entry'] - gold[i] - SPREAD_GOLD) * lot * PV_GOLD
            else:
                pnl_gold = (gold[i] - pos['gold_entry'] - SPREAD_GOLD) * lot * PV_GOLD

            should_exit = False
            reason = ""

            if pos['dir'] == 'SELL' and z <= exit_z:
                should_exit = True
                reason = f"Z回归({z:.1f})"
            elif pos['dir'] == 'BUY' and z >= -exit_z:
                should_exit = True
                reason = f"Z回归({z:.1f})"
            elif pos['dir'] == 'SELL' and z > stop_z:
                should_exit = True
                reason = f"Z止损({z:.1f})"
            elif pos['dir'] == 'BUY' and z < -stop_z:
                should_exit = True
                reason = f"Z止损({z:.1f})"
            elif hold >= max_hold:
                should_exit = True
                reason = f"超时({hold}d)"

            if should_exit:
                trades.append({
                    'dir': pos['dir'], 'entry_time': pos['time'], 'exit_time': times[i],
                    'gold_entry': pos['gold_entry'], 'gold_exit': gold[i],
                    'pnl': pnl_gold, 'reason': reason,
                    'hold_days': hold, 'entry_z': pos['entry_z'], 'exit_z': z,
                })
                pos = None
            continue

        if z > entry_z:
            pos = {'dir': 'SELL', 'bar': i, 'time': times[i],
                   'gold_entry': gold[i], 'entry_z': z}
        elif z < -entry_z:
            pos = {'dir': 'BUY', 'bar': i, 'time': times[i],
                   'gold_entry': gold[i], 'entry_z': z}

    return trades


# ═══════════════════════════════════════════════════════════════
# Strategy 2: Bollinger Band Ratio
# ═══════════════════════════════════════════════════════════════

def bt_bband_ratio(df, lookback=20, num_std=2.0, exit_std=0.5,
                   max_hold=30, lot=UNIT_LOT):
    """Bollinger Band on ratio: touch upper band = sell gold, lower = buy gold."""
    ratio = df['ratio'].values
    gold = df['gold_close'].values
    times = df.index
    n = len(df)

    rm = pd.Series(ratio).rolling(lookback).mean().values
    rs = pd.Series(ratio).rolling(lookback).std().values

    trades = []
    pos = None

    for i in range(lookback + 1, n):
        if np.isnan(rm[i]) or np.isnan(rs[i]) or rs[i] < 0.01:
            continue

        upper = rm[i] + num_std * rs[i]
        lower = rm[i] - num_std * rs[i]
        mid = rm[i]
        exit_upper = rm[i] + exit_std * rs[i]
        exit_lower = rm[i] - exit_std * rs[i]

        if pos is not None:
            hold = i - pos['bar']
            if pos['dir'] == 'SELL':
                pnl = (pos['gold_entry'] - gold[i] - SPREAD_GOLD) * lot * PV_GOLD
            else:
                pnl = (gold[i] - pos['gold_entry'] - SPREAD_GOLD) * lot * PV_GOLD

            should_exit = False
            reason = ""
            if pos['dir'] == 'SELL' and ratio[i] <= exit_upper:
                should_exit = True
                reason = f"BB回归(r={ratio[i]:.1f})"
            elif pos['dir'] == 'BUY' and ratio[i] >= exit_lower:
                should_exit = True
                reason = f"BB回归(r={ratio[i]:.1f})"
            elif hold >= max_hold:
                should_exit = True
                reason = f"超时({hold}d)"

            if should_exit:
                trades.append({
                    'dir': pos['dir'], 'entry_time': pos['time'], 'exit_time': times[i],
                    'gold_entry': pos['gold_entry'], 'gold_exit': gold[i],
                    'pnl': pnl, 'reason': reason, 'hold_days': hold,
                    'entry_z': pos.get('entry_z', 0), 'exit_z': (ratio[i] - rm[i]) / rs[i],
                })
                pos = None
            continue

        if ratio[i] > upper:
            z = (ratio[i] - rm[i]) / rs[i]
            pos = {'dir': 'SELL', 'bar': i, 'time': times[i],
                   'gold_entry': gold[i], 'entry_z': z}
        elif ratio[i] < lower:
            z = (ratio[i] - rm[i]) / rs[i]
            pos = {'dir': 'BUY', 'bar': i, 'time': times[i],
                   'gold_entry': gold[i], 'entry_z': z}

    return trades


# ═══════════════════════════════════════════════════════════════
# Strategy 3: Adaptive lookback (Half-Life based)
# ═══════════════════════════════════════════════════════════════

def calc_half_life(series, max_lag=120):
    """OLS regression: delta_ratio = alpha + beta * ratio_lag -> HL = -ln(2)/beta"""
    s = series.dropna().values
    if len(s) < 30:
        return 20
    y = np.diff(s)
    x = s[:-1]
    x_mean = x.mean()
    beta = np.sum((x - x_mean) * y) / np.sum((x - x_mean) ** 2)
    if beta >= 0:
        return max_lag
    hl = -np.log(2) / beta
    return max(5, min(max_lag, int(round(hl))))


def bt_adaptive_zscore(df, hl_window=252, entry_z=2.0, exit_z=0.0,
                       stop_z=4.0, max_hold=30, lot=UNIT_LOT):
    """Z-Score with rolling Half-Life adaptive lookback."""
    ratio = df['ratio']
    gold = df['gold_close'].values
    times = df.index
    n = len(df)

    trades = []
    pos = None

    for i in range(hl_window + 60, n):
        hl = calc_half_life(ratio.iloc[i - hl_window:i])
        lb = max(5, min(60, hl))
        window = ratio.iloc[i - lb:i + 1].values
        if len(window) < lb:
            continue
        rm = np.mean(window[:-1])
        rs = np.std(window[:-1], ddof=1)
        if rs < 0.01:
            continue
        z = (ratio.iloc[i] - rm) / rs

        if pos is not None:
            hold = i - pos['bar']
            if pos['dir'] == 'SELL':
                pnl = (pos['gold_entry'] - gold[i] - SPREAD_GOLD) * lot * PV_GOLD
            else:
                pnl = (gold[i] - pos['gold_entry'] - SPREAD_GOLD) * lot * PV_GOLD

            should_exit = False
            reason = ""
            if pos['dir'] == 'SELL' and z <= exit_z:
                should_exit = True; reason = f"Z回归({z:.1f},HL={lb})"
            elif pos['dir'] == 'BUY' and z >= -exit_z:
                should_exit = True; reason = f"Z回归({z:.1f},HL={lb})"
            elif pos['dir'] == 'SELL' and z > stop_z:
                should_exit = True; reason = f"Z止损({z:.1f})"
            elif pos['dir'] == 'BUY' and z < -stop_z:
                should_exit = True; reason = f"Z止损({z:.1f})"
            elif hold >= max_hold:
                should_exit = True; reason = f"超时({hold}d)"

            if should_exit:
                trades.append({
                    'dir': pos['dir'], 'entry_time': pos['time'], 'exit_time': times[i],
                    'gold_entry': pos['gold_entry'], 'gold_exit': gold[i],
                    'pnl': pnl, 'reason': reason, 'hold_days': hold,
                    'entry_z': pos['entry_z'], 'exit_z': z,
                })
                pos = None
            continue

        if z > entry_z:
            pos = {'dir': 'SELL', 'bar': i, 'time': times[i],
                   'gold_entry': gold[i], 'entry_z': z}
        elif z < -entry_z:
            pos = {'dir': 'BUY', 'bar': i, 'time': times[i],
                   'gold_entry': gold[i], 'entry_z': z}

    return trades


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R112 -- Gold/Silver Pair Trading (Thorp-Inspired)")
    print("=" * 80)

    daily = load_daily_data()
    print(f"  Daily data: {len(daily)} bars ({daily.index[0].date()} ~ {daily.index[-1].date()})")

    results = {'experiment': 'R112 Gold/Silver Pair Trading'}

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Basic ratio statistics
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 1: Gold/Silver Ratio Statistics")
    print("=" * 60)

    r = daily['ratio']
    stats = {
        'mean': round(r.mean(), 2), 'std': round(r.std(), 2),
        'min': round(r.min(), 2), 'max': round(r.max(), 2),
        'current': round(r.iloc[-1], 2),
        'q25': round(r.quantile(0.25), 2), 'q75': round(r.quantile(0.75), 2),
    }
    print(f"  Mean={stats['mean']}, Std={stats['std']}")
    print(f"  Range: [{stats['min']} ~ {stats['max']}], Current={stats['current']}")

    hl = calc_half_life(r.iloc[-252:])
    print(f"  Half-life (last 252d): {hl} days")
    hl_full = calc_half_life(r)
    print(f"  Half-life (full): {hl_full} days")

    results['phase1_stats'] = stats
    results['half_life_1y'] = hl
    results['half_life_full'] = hl_full

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Z-Score parameter grid search
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 2: Z-Score Mean Reversion Grid Search")
    print("=" * 60)

    lookbacks = [10, 15, 20, 30, 40, 60]
    entry_zs = [1.5, 2.0, 2.5, 3.0]
    exit_zs = [0.0, 0.3, 0.5]
    stop_zs = [3.5, 4.0, 5.0]
    max_holds = [15, 20, 30, 40, 60]

    grid = []
    total = len(lookbacks) * len(entry_zs) * len(exit_zs) * len(stop_zs) * len(max_holds)
    tested = 0
    for lb in lookbacks:
        for ez in entry_zs:
            for xz in exit_zs:
                for sz in stop_zs:
                    if sz <= ez:
                        continue
                    for mh in max_holds:
                        trades = bt_zscore_ratio(daily, lookback=lb, entry_z=ez,
                                                 exit_z=xz, stop_z=sz, max_hold=mh)
                        m = metrics_from_trades(trades)
                        if m['n'] >= 20:
                            grid.append({
                                'params': {'lb': lb, 'entry_z': ez, 'exit_z': xz,
                                           'stop_z': sz, 'max_hold': mh},
                                **m
                            })
                        tested += 1
                        if tested % 100 == 0:
                            print(f"    {tested}/{total}...")

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Tested: {tested}, with >=20 trades: {len(grid)}")
    print(f"\n  Top 10:")
    for i, g in enumerate(grid[:10]):
        p = g['params']
        print(f"    #{i+1}: lb={p['lb']} ez={p['entry_z']} xz={p['exit_z']} "
              f"sz={p['stop_z']} mh={p['max_hold']} -> "
              f"Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%, "
              f"PF={g['profit_factor']}, AvgHold={g['avg_hold']}d")

    results['phase2_zscore_grid_top10'] = grid[:10]
    results['phase2_total_combos'] = tested

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Bollinger Band strategy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 3: Bollinger Band Ratio Strategy")
    print("=" * 60)

    bb_grid = []
    for lb in [15, 20, 30, 40, 60]:
        for ns in [1.5, 2.0, 2.5, 3.0]:
            for xs in [0.0, 0.3, 0.5, 1.0]:
                for mh in [15, 20, 30, 40, 60]:
                    trades = bt_bband_ratio(daily, lookback=lb, num_std=ns,
                                            exit_std=xs, max_hold=mh)
                    m = metrics_from_trades(trades)
                    if m['n'] >= 20:
                        bb_grid.append({
                            'params': {'lb': lb, 'num_std': ns, 'exit_std': xs,
                                       'max_hold': mh},
                            **m
                        })

    bb_grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Configs with >=20 trades: {len(bb_grid)}")
    print(f"\n  Top 10:")
    for i, g in enumerate(bb_grid[:10]):
        p = g['params']
        print(f"    #{i+1}: lb={p['lb']} std={p['num_std']} exit={p['exit_std']} "
              f"mh={p['max_hold']} -> "
              f"Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%, "
              f"PF={g['profit_factor']}, AvgHold={g['avg_hold']}d")

    results['phase3_bband_top10'] = bb_grid[:10]

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Adaptive Half-Life strategy
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 4: Adaptive Half-Life Z-Score")
    print("=" * 60)

    adapt_grid = []
    for hw in [120, 252, 504]:
        for ez in [1.5, 2.0, 2.5]:
            for xz in [0.0, 0.3, 0.5]:
                for sz in [3.5, 4.0, 5.0]:
                    if sz <= ez:
                        continue
                    for mh in [15, 20, 30, 40]:
                        trades = bt_adaptive_zscore(daily, hl_window=hw, entry_z=ez,
                                                    exit_z=xz, stop_z=sz, max_hold=mh)
                        m = metrics_from_trades(trades)
                        if m['n'] >= 10:
                            adapt_grid.append({
                                'params': {'hl_window': hw, 'entry_z': ez, 'exit_z': xz,
                                           'stop_z': sz, 'max_hold': mh},
                                **m
                            })

    adapt_grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Configs with >=10 trades: {len(adapt_grid)}")
    print(f"\n  Top 10:")
    for i, g in enumerate(adapt_grid[:10]):
        p = g['params']
        print(f"    #{i+1}: hw={p['hl_window']} ez={p['entry_z']} xz={p['exit_z']} "
              f"sz={p['stop_z']} mh={p['max_hold']} -> "
              f"Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%, "
              f"PF={g['profit_factor']}, AvgHold={g['avg_hold']}d")

    results['phase4_adaptive_top10'] = adapt_grid[:10]

    # ═══════════════════════════════════════════════════════════
    # Phase 5: K-Fold validation on best configs
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 5: K-Fold Validation")
    print("=" * 60)

    candidates = []
    if grid:
        candidates.append(('ZScore', grid[0]['params'], 'zscore'))
        if len(grid) > 1 and grid[1]['params'] != grid[0]['params']:
            candidates.append(('ZScore-2', grid[1]['params'], 'zscore'))
    if bb_grid:
        candidates.append(('BBand', bb_grid[0]['params'], 'bband'))
    if adapt_grid:
        candidates.append(('Adaptive', adapt_grid[0]['params'], 'adaptive'))

    kfold_results = {}
    for label, params, stype in candidates:
        fold_sharpes = []
        fold_trades = []
        for fname, start, end in FOLDS:
            fdata = daily[(daily.index >= start) & (daily.index < end)]
            if len(fdata) < 60:
                fold_sharpes.append(0.0)
                fold_trades.append(0)
                continue
            if stype == 'zscore':
                trades = bt_zscore_ratio(fdata, lookback=params['lb'],
                                         entry_z=params['entry_z'], exit_z=params['exit_z'],
                                         stop_z=params['stop_z'], max_hold=params['max_hold'])
            elif stype == 'bband':
                trades = bt_bband_ratio(fdata, lookback=params['lb'],
                                        num_std=params['num_std'], exit_std=params['exit_std'],
                                        max_hold=params['max_hold'])
            else:
                trades = bt_adaptive_zscore(fdata, hl_window=params['hl_window'],
                                            entry_z=params['entry_z'], exit_z=params['exit_z'],
                                            stop_z=params['stop_z'], max_hold=params['max_hold'])
            m = metrics_from_trades(trades)
            fold_sharpes.append(m['sharpe'])
            fold_trades.append(m['n'])

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = np.mean(fold_sharpes)
        passed = positive >= 4
        status = "PASS" if passed else "FAIL"
        kfold_results[label] = {
            'params': params, 'strategy_type': stype,
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'fold_trades': fold_trades,
            'positive_folds': positive, 'mean_sharpe': round(mean_sh, 3),
            'pass_4of6': passed,
        }
        print(f"  {label}: sharpes={[f'{s:.2f}' for s in fold_sharpes]} "
              f"trades={fold_trades} -> {positive}/6 [{status}] mean={mean_sh:.3f}")

    results['phase5_kfold'] = kfold_results

    # ═══════════════════════════════════════════════════════════
    # Phase 6: Correlation with existing strategies
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 6: Correlation with Existing Strategies")
    print("=" * 60)

    best_config = None
    for label, info in kfold_results.items():
        if info['pass_4of6']:
            if best_config is None or info['mean_sharpe'] > best_config[1]['mean_sharpe']:
                best_config = (label, info)

    if best_config:
        bp = best_config[1]['params']
        btype = best_config[1]['strategy_type']
        print(f"  Best validated: {best_config[0]} ({bp})")

        if btype == 'zscore':
            pair_trades = bt_zscore_ratio(daily, lookback=bp['lb'],
                                          entry_z=bp['entry_z'], exit_z=bp['exit_z'],
                                          stop_z=bp['stop_z'], max_hold=bp['max_hold'])
        elif btype == 'bband':
            pair_trades = bt_bband_ratio(daily, lookback=bp['lb'],
                                         num_std=bp['num_std'], exit_std=bp['exit_std'],
                                         max_hold=bp['max_hold'])
        else:
            pair_trades = bt_adaptive_zscore(daily, hl_window=bp['hl_window'],
                                             entry_z=bp['entry_z'], exit_z=bp['exit_z'],
                                             stop_z=bp['stop_z'], max_hold=bp['max_hold'])

        pair_daily = {}
        for t in pair_trades:
            d = pd.Timestamp(t['exit_time']).date()
            pair_daily[d] = pair_daily.get(d, 0) + t['pnl']
        pair_series = pd.Series(pair_daily)

        gold_daily_ret = daily['gold_close'].pct_change().dropna()
        common_idx = pair_series.index.intersection(gold_daily_ret.index)
        if len(common_idx) > 10:
            corr_gold = pair_series.loc[common_idx].corr(gold_daily_ret.loc[common_idx])
            print(f"  Corr with Gold returns: {corr_gold:.3f}")
            results['correlation_with_gold'] = round(corr_gold, 3)

        ratio_ret = daily['ratio'].pct_change().dropna()
        common_idx2 = pair_series.index.intersection(ratio_ret.index)
        if len(common_idx2) > 10:
            corr_ratio = pair_series.loc[common_idx2].corr(ratio_ret.loc[common_idx2])
            print(f"  Corr with Ratio changes: {corr_ratio:.3f}")
            results['correlation_with_ratio'] = round(corr_ratio, 3)

        # Direction distribution
        buy_count = sum(1 for t in pair_trades if t['dir'] == 'BUY')
        sell_count = sum(1 for t in pair_trades if t['dir'] == 'SELL')
        buy_pnl = sum(t['pnl'] for t in pair_trades if t['dir'] == 'BUY')
        sell_pnl = sum(t['pnl'] for t in pair_trades if t['dir'] == 'SELL')
        print(f"  BUY gold (ratio low): {buy_count} trades, PnL=${buy_pnl:.0f}")
        print(f"  SELL gold (ratio high): {sell_count} trades, PnL=${sell_pnl:.0f}")
        results['direction_split'] = {
            'buy_n': buy_count, 'buy_pnl': round(buy_pnl, 2),
            'sell_n': sell_count, 'sell_pnl': round(sell_pnl, 2),
        }

        # Year-by-year
        print(f"\n  Year-by-year:")
        for year in range(2001, 2027):
            yt = [t for t in pair_trades if pd.Timestamp(t['exit_time']).year == year]
            if yt:
                ym = metrics_from_trades(yt)
                print(f"    {year}: {ym['n']:3d} trades, Sharpe={ym['sharpe']:6.2f}, "
                      f"PnL=${ym['pnl']:8.2f}, WR={ym['wr']:.0f}%")
    else:
        print("  No config passed K-Fold validation.")
        results['correlation_with_gold'] = None

    # ═══════════════════════════════════════════════════════════
    # Phase 7: Summary comparison of all strategies
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Phase 7: Strategy Comparison Summary")
    print("=" * 60)

    all_best = []
    if grid:
        p = grid[0]['params']
        trades = bt_zscore_ratio(daily, lookback=p['lb'], entry_z=p['entry_z'],
                                 exit_z=p['exit_z'], stop_z=p['stop_z'],
                                 max_hold=p['max_hold'])
        m = metrics_from_trades(trades)
        all_best.append(('ZScore', p, m))
    if bb_grid:
        p = bb_grid[0]['params']
        trades = bt_bband_ratio(daily, lookback=p['lb'], num_std=p['num_std'],
                                exit_std=p['exit_std'], max_hold=p['max_hold'])
        m = metrics_from_trades(trades)
        all_best.append(('BBand', p, m))
    if adapt_grid:
        p = adapt_grid[0]['params']
        trades = bt_adaptive_zscore(daily, hl_window=p['hl_window'],
                                    entry_z=p['entry_z'], exit_z=p['exit_z'],
                                    stop_z=p['stop_z'], max_hold=p['max_hold'])
        m = metrics_from_trades(trades)
        all_best.append(('Adaptive', p, m))

    print(f"  {'Strategy':<12} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'Trades':>7} "
          f"{'WR':>6} {'PF':>6} {'AvgHold':>8}")
    for name, params, m in all_best:
        print(f"  {name:<12} {m['sharpe']:>7.3f} ${m['pnl']:>9.0f} ${m['max_dd']:>7.0f} "
              f"{m['n']:>7} {m['wr']:>5.1f}% {m['profit_factor']:>5.2f} {m['avg_hold']:>7.1f}d")

    results['phase7_comparison'] = [
        {'name': name, 'params': params, **m} for name, params, m in all_best
    ]

    # Save
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)

    if best_config:
        results['recommendation'] = (
            f"Best: {best_config[0]} with {best_config[1]['params']}, "
            f"K-Fold {best_config[1]['positive_folds']}/6, "
            f"mean Sharpe={best_config[1]['mean_sharpe']}"
        )
    else:
        results['recommendation'] = "No pair trading config passed K-Fold. Gold/Silver ratio may not be mean-reverting enough at current params."

    out_file = OUTPUT_DIR / "r112_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"  R112 COMPLETE -- {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  {results['recommendation']}")
    print(f"{'='*80}")
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()
