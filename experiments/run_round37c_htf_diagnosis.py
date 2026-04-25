"""
R37C: HTF Strategy Deep Diagnosis
===================================
诊断 R37B 结果中的异常（D1 Donchian 100% WR / MaxDD=$0）并用更合理的参数重测。

Phase A: 回测引擎验证 — 检查每笔交易明细（持仓时间、出场原因、盈利分布）
Phase B: D1 级别参数适配 — 用 D1-ATR 量级的 SL/TP/Trail 重新回测
Phase C: 隔夜利息(Swap)成本 — 模拟持仓 N 天的累计 swap 成本
Phase D: 大滑点测试 — $1~$5 滑点（D1 跳空场景）
Phase E: L7 交叉相关性修复 — 修复 Phase D 报错
Phase F: 合理化参数全套重测 — MaxHold/Cooldown/SL/TP 适配 D1
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

OUTPUT_FILE = "R37C_output.txt"

class Tee:
    def __init__(self, fname):
        self.file = open(fname, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee(OUTPUT_FILE)
print(f"R37C started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════
from backtest.runner import load_csv, load_m15, load_h1_aligned, DataBundle, run_variant

M15_PATH = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
H1_PATH = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

print("\nLoading data...")
m15_df_raw = load_m15(M15_PATH)
h1_df = load_h1_aligned(H1_PATH, m15_df_raw.index[0])

def resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

h4_df = resample_ohlcv(h1_df, '4h')
d1_df = resample_ohlcv(h1_df, '1D')
print(f"  H4: {len(h4_df)} bars, D1: {len(d1_df)} bars")

def _calc_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _calc_adx(df, period):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr = pd.DataFrame({
        'hl': high - low, 'hc': (high - close.shift(1)).abs(), 'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': (df['High'] - df['Close'].shift(1)).abs(),
        'lc': (df['Low'] - df['Close'].shift(1)).abs()
    }).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['RSI14'] = _calc_rsi(df['Close'], 14)
    df['ADX'] = _calc_adx(df, 14)
    for span in [9, 20, 25, 50, 100, 200]:
        df[f'EMA{span}'] = df['Close'].ewm(span=span).mean()
    for n in [5, 10, 15, 20, 30, 40, 50, 60, 80]:
        df[f'Donchian_high_{n}'] = df['High'].rolling(n).max()
        df[f'Donchian_low_{n}'] = df['Low'].rolling(n).min()
    df['ROC5'] = df['Close'].pct_change(5) * 100
    return df

print("Adding indicators...")
h4_df = add_indicators(h4_df)
d1_df = add_indicators(d1_df)
print("Indicators ready.\n")

# Average D1 ATR for reference
d1_atr_mean = d1_df['ATR'].dropna().mean()
h4_atr_mean = h4_df['ATR'].dropna().mean()
print(f"  D1 ATR(14) mean: ${d1_atr_mean:.2f}")
print(f"  H4 ATR(14) mean: ${h4_atr_mean:.2f}")
print(f"  (For reference: M15 ATR is typically ~$3-5, so D1 ATR is ~{d1_atr_mean/4:.0f}x larger)")

# ═══════════════════════════════════════════════════════════════
# Backtester with detailed trade logging
# ═══════════════════════════════════════════════════════════════

def backtest(df, label, signal_fn, sl_atr=3.5, tp_atr=8.0,
             trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
             spread=0.30, lot=0.03, cooldown=2,
             swap_per_day=0.0, slippage=0.0):
    """Enhanced backtester with swap costs and slippage."""
    df = df.dropna(subset=['ATR'])
    trades = []; pos = None
    close = df['Close'].values; high = df['High'].values; low = df['Low'].values
    open_ = df['Open'].values; atr = df['ATR'].values
    times = df.index; n = len(df); last_exit = -999
    total_spread = spread + slippage

    for i in range(1, n):
        if pos is not None:
            held = i - pos['bar']
            if pos['dir'] == 'BUY':
                pnl_h = (high[i] - pos['entry'] - total_spread) * lot * 100
                pnl_l = (low[i] - pos['entry'] - total_spread) * lot * 100
                pnl_c = (close[i] - pos['entry'] - total_spread) * lot * 100
            else:
                pnl_h = (pos['entry'] - low[i] - total_spread) * lot * 100
                pnl_l = (pos['entry'] - high[i] - total_spread) * lot * 100
                pnl_c = (pos['entry'] - close[i] - total_spread) * lot * 100
            
            swap_cost = swap_per_day * held * lot
            
            tp_val = tp_atr * pos['atr'] * lot * 100
            sl_val = sl_atr * pos['atr'] * lot * 100
            exited = False
            if pnl_h >= tp_val:
                _rec(trades, pos, close[i], times[i], "TP", i, tp_val - swap_cost); exited = True
            elif pnl_l <= -sl_val:
                _rec(trades, pos, close[i], times[i], "SL", i, -sl_val - swap_cost); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        raw_pnl = (ts - pos['entry'] - total_spread) * lot * 100
                        _rec(trades, pos, close[i], times[i], "Trail", i, raw_pnl - swap_cost)
                        exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        raw_pnl = (pos['entry'] - ts - total_spread) * lot * 100
                        _rec(trades, pos, close[i], times[i], "Trail", i, raw_pnl - swap_cost)
                        exited = True
                if not exited and held >= max_hold:
                    _rec(trades, pos, close[i], times[i], "Timeout", i, pnl_c - swap_cost)
                    exited = True
            if exited: pos = None; last_exit = i; continue
        if pos is not None: continue
        if i - last_exit < cooldown: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        sig = signal_fn(df, i)
        if sig == 'BUY':
            pos = {'dir': 'BUY', 'entry': open_[i] + total_spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif sig == 'SELL':
            pos = {'dir': 'SELL', 'entry': open_[i] - total_spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return _stats(trades, label)


def _rec(trades, pos, exit_price, exit_time, reason, bar_i, pnl):
    trades.append({
        'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
        'entry_time': pos['time'], 'exit_time': exit_time,
        'pnl': pnl, 'exit_reason': reason, 'bars_held': bar_i - pos['bar'],
        'entry_atr': pos['atr']
    })


def _stats(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0, 'total_pnl': 0, 'win_rate': 0,
                'max_dd': 0, '_trades': [], '_daily_pnl': {}}
    pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(pnls); dd = (np.maximum.accumulate(eq + 2000) - (eq + 2000)).max()
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily.setdefault(d, 0); daily[d] += t['pnl']
    da = np.array(list(daily.values()))
    sh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
    avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0
    return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
            'win_rate': wins / len(trades) * 100, 'max_dd': dd,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            '_trades': trades, '_daily_pnl': daily}


def pr(r):
    print(f"  {r['label']:>50}: N={r['n']:>5}, Sharpe={r['sharpe']:>7.2f}, "
          f"PnL=${r['total_pnl']:>8.0f}, WR={r['win_rate']:>5.1f}%, MaxDD=${r['max_dd']:>6.0f}")


def pr_detail(r):
    """Print detailed stats including avg win/loss and trade breakdown."""
    pr(r)
    trades = r['_trades']
    if not trades:
        return
    bars = [t['bars_held'] for t in trades]
    reasons = Counter(t['exit_reason'] for t in trades)
    pnls = [t['pnl'] for t in trades]
    print(f"    AvgWin=${r['avg_win']:>7.2f}, AvgLoss=${r['avg_loss']:>7.2f}, "
          f"Ratio={abs(r['avg_win']/(r['avg_loss'] or -1)):>.2f}")
    print(f"    BarsHeld: min={min(bars)}, median={int(np.median(bars))}, "
          f"max={max(bars)}, mean={np.mean(bars):.1f}")
    print(f"    ExitReasons: {dict(reasons)}")
    print(f"    PnL dist: min=${min(pnls):>.2f}, p25=${np.percentile(pnls,25):>.2f}, "
          f"median=${np.median(pnls):>.2f}, p75=${np.percentile(pnls,75):>.2f}, "
          f"max=${max(pnls):>.2f}")


def sharpe_from_daily(daily_dict):
    if not daily_dict: return 0
    da = np.array(list(daily_dict.values()))
    return da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0


# ═══════════════════════════════════════════════════════════════
# Signal functions
# ═══════════════════════════════════════════════════════════════

def don_sig(d, i, period=50):
    if i < 2: return None
    hc = f'Donchian_high_{period}'; lc = f'Donchian_low_{period}'
    ch = d[hc].iloc[i-1]; cl = d[lc].iloc[i-1]; c = d['Close'].iloc[i]
    if np.isnan(ch): return None
    if c > ch: return 'BUY'
    if c < cl: return 'SELL'
    return None

def ema_cross_sig(d, i, fast_col='EMA9', slow_col='EMA25', adx_gate=25):
    if i < 2: return None
    fn = d[fast_col].iloc[i]; fp = d[fast_col].iloc[i-1]
    sn = d[slow_col].iloc[i]; sp = d[slow_col].iloc[i-1]
    if np.isnan(fn) or np.isnan(sn): return None
    if adx_gate > 0:
        adx = d['ADX'].iloc[i]
        if np.isnan(adx) or adx < adx_gate: return None
    if fp <= sp and fn > sn: return 'BUY'
    if fp >= sp and fn < sn: return 'SELL'
    return None

d1_trend_ref = d1_df[['Close', 'EMA50', 'EMA20']].copy()

def topdown_sig(d, i):
    if i < 2: return None
    c = d['Close'].iloc[i]; t = d.index[i]
    mask = d1_trend_ref.index <= t
    if not mask.any(): return None
    d1r = d1_trend_ref.loc[d1_trend_ref.index[mask][-1]]
    trend = 'BULL' if d1r['Close'] > d1r['EMA50'] else 'BEAR'
    ema20 = d['EMA20'].iloc[i]
    if np.isnan(ema20): return None
    prev_c = d['Close'].iloc[i-1]
    if trend == 'BULL' and prev_c < ema20 and c > ema20: return 'BUY'
    if trend == 'BEAR' and prev_c > ema20 and c < ema20: return 'SELL'
    return None

def roc_sig(d, i, thresh=1.0):
    if i < 2: return None
    roc = d['ROC5'].iloc[i]; c = d['Close'].iloc[i]; ema = d['EMA200'].iloc[i]
    if np.isnan(roc) or np.isnan(ema): return None
    if roc > thresh and c > ema: return 'BUY'
    if roc < -thresh and c < ema: return 'SELL'
    return None


# ═══════════════════════════════════════════════════════════════
# Phase A: Backtest Engine Validation — Trade-by-Trade Audit
# ═══════════════════════════════════════════════════════════════

def run_phase_A():
    print("=" * 80)
    print("Phase A: Backtest Engine Validation (Trade-Level Audit)")
    print("=" * 80)
    print("\nUsing R37B original params: SL=3.5xATR, TP=8.0xATR, Trail_act=0.28, Trail_dist=0.06")
    print(f"D1 ATR mean = ${d1_atr_mean:.2f}")
    print(f"  => SL = 3.5 x ${d1_atr_mean:.2f} = ${3.5*d1_atr_mean:.2f}")
    print(f"  => TP = 8.0 x ${d1_atr_mean:.2f} = ${8.0*d1_atr_mean:.2f}")
    print(f"  => Trail activate = 0.28 x ${d1_atr_mean:.2f} = ${0.28*d1_atr_mean:.2f}")
    print(f"  => Trail distance = 0.06 x ${d1_atr_mean:.2f} = ${0.06*d1_atr_mean:.2f}")
    print(f"  PROBLEM: Trail activates at ${0.28*d1_atr_mean:.2f} (very easy on D1) "
          f"then trails at ${0.06*d1_atr_mean:.2f} — almost guaranteed instant trail-stop!")

    strats = [
        ('D1_Don50_R37B_params', d1_df, lambda d,i: don_sig(d,i,50),
         dict(sl_atr=3.5, tp_atr=8.0, trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=30)),
        ('D1+H4_TopDown_R37B', h4_df, lambda d,i: topdown_sig(d,i),
         dict(sl_atr=4.0, tp_atr=10.0, trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=30)),
        ('H4_ROC5_EMA200_R37B', h4_df, lambda d,i: roc_sig(d,i,1.0),
         dict(sl_atr=3.0, tp_atr=6.0, trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=20)),
    ]

    for name, df, sig_fn, params in strats:
        print(f"\n  --- {name} ---")
        r = backtest(df, name, sig_fn, **params)
        pr_detail(r)
        trades = r['_trades']
        if trades:
            print(f"\n    First 5 trades sample:")
            for t in trades[:5]:
                print(f"      {t['entry_time']} -> {t['exit_time']} | "
                      f"{t['dir']} | entry={t['entry']:.2f} exit={t['exit']:.2f} | "
                      f"bars={t['bars_held']} | {t['exit_reason']} | pnl=${t['pnl']:.2f} | "
                      f"ATR={t['entry_atr']:.2f}")
            print(f"\n    Last 5 trades sample:")
            for t in trades[-5:]:
                print(f"      {t['entry_time']} -> {t['exit_time']} | "
                      f"{t['dir']} | entry={t['entry']:.2f} exit={t['exit']:.2f} | "
                      f"bars={t['bars_held']} | {t['exit_reason']} | pnl=${t['pnl']:.2f} | "
                      f"ATR={t['entry_atr']:.2f}")


# ═══════════════════════════════════════════════════════════════
# Phase B: D1-Appropriate Parameter Sweep
# ═══════════════════════════════════════════════════════════════

def run_phase_B():
    print("\n" + "=" * 80)
    print("Phase B: D1-Appropriate Parameters (SL/TP/Trail scaled to D1 ATR)")
    print("=" * 80)
    
    print("\n  --- B1: D1 Don50 — SL/TP sweep with NO trailing stop ---")
    for sl in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for tp in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
            r = backtest(d1_df, f"D1_Don50_SL{sl}_TP{tp}_noTrail",
                         lambda d,i: don_sig(d,i,50),
                         sl_atr=sl, tp_atr=tp,
                         trail_act_atr=999, trail_dist_atr=999,
                         max_hold=60, cooldown=3)
            if r['n'] > 10:
                pr_detail(r)

    print("\n  --- B2: D1 Don50 — Realistic trailing stops (wide) ---")
    for ta in [1.0, 1.5, 2.0, 3.0]:
        for td in [0.3, 0.5, 0.8, 1.0, 1.5]:
            if td >= ta: continue
            r = backtest(d1_df, f"D1_Don50_trail({ta}/{td})",
                         lambda d,i: don_sig(d,i,50),
                         sl_atr=2.0, tp_atr=5.0,
                         trail_act_atr=ta, trail_dist_atr=td,
                         max_hold=60, cooldown=3)
            if r['n'] > 10:
                pr(r)

    print("\n  --- B3: D1+H4 TopDown — D1-appropriate params ---")
    for sl in [1.0, 1.5, 2.0]:
        for tp in [2.0, 3.0, 4.0, 5.0]:
            r = backtest(h4_df, f"TopDown_SL{sl}_TP{tp}_noTrail",
                         lambda d,i: topdown_sig(d,i),
                         sl_atr=sl, tp_atr=tp,
                         trail_act_atr=999, trail_dist_atr=999,
                         max_hold=40, cooldown=3)
            if r['n'] > 10:
                pr_detail(r)

    print("\n  --- B4: H4 ROC5 — Wider trail ---")
    for ta in [0.5, 1.0, 1.5, 2.0]:
        for td in [0.2, 0.3, 0.5, 0.8]:
            if td >= ta: continue
            r = backtest(h4_df, f"H4_ROC5_trail({ta}/{td})",
                         lambda d,i: roc_sig(d,i,1.0),
                         sl_atr=2.0, tp_atr=4.0,
                         trail_act_atr=ta, trail_dist_atr=td,
                         max_hold=30, cooldown=3)
            if r['n'] > 10:
                pr(r)


# ═══════════════════════════════════════════════════════════════
# Phase C: Swap (Overnight Interest) Cost Impact
# ═══════════════════════════════════════════════════════════════

def run_phase_C():
    print("\n" + "=" * 80)
    print("Phase C: Swap (Overnight Interest) Cost Impact")
    print("=" * 80)
    print("  XAUUSD swap costs: ~$3-8 per lot per night (varies by broker)")
    print("  At 0.03 lot: ~$0.09-0.24 per night")
    print("  Testing swap = $0, $3, $5, $8, $12 per standard lot per night\n")

    test_configs = [
        ('D1_Don50', d1_df, lambda d,i: don_sig(d,i,50),
         dict(sl_atr=2.0, tp_atr=5.0, trail_act_atr=1.5, trail_dist_atr=0.5, max_hold=60, cooldown=3)),
        ('D1+H4_TopDown', h4_df, lambda d,i: topdown_sig(d,i),
         dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=40, cooldown=3)),
        ('H4_ROC5_EMA200', h4_df, lambda d,i: roc_sig(d,i,1.0),
         dict(sl_atr=2.0, tp_atr=4.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=30, cooldown=3)),
    ]

    for swap_lot in [0, 3, 5, 8, 12]:
        print(f"  --- Swap = ${swap_lot}/lot/night ---")
        for name, df, sig_fn, params in test_configs:
            r = backtest(df, f"{name}_swap{swap_lot}", sig_fn, swap_per_day=swap_lot, **params)
            if r['n'] > 0:
                pr(r)
        print()


# ═══════════════════════════════════════════════════════════════
# Phase D: Large Slippage Test (D1 gap scenarios)
# ═══════════════════════════════════════════════════════════════

def run_phase_D():
    print("\n" + "=" * 80)
    print("Phase D: Large Slippage Test (D1 gap-open scenarios)")
    print("=" * 80)
    print("  D1 breakouts often occur at gap opens.")
    print("  Testing slippage = $0, $0.50, $1.00, $2.00, $3.00, $5.00\n")

    test_configs = [
        ('D1_Don50', d1_df, lambda d,i: don_sig(d,i,50),
         dict(sl_atr=2.0, tp_atr=5.0, trail_act_atr=1.5, trail_dist_atr=0.5, max_hold=60, cooldown=3)),
        ('D1+H4_TopDown', h4_df, lambda d,i: topdown_sig(d,i),
         dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=40, cooldown=3)),
        ('H4_ROC5_EMA200', h4_df, lambda d,i: roc_sig(d,i,1.0),
         dict(sl_atr=2.0, tp_atr=4.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=30, cooldown=3)),
    ]

    for slip in [0, 0.50, 1.00, 2.00, 3.00, 5.00]:
        print(f"  --- Slippage = ${slip:.2f} ---")
        for name, df, sig_fn, params in test_configs:
            r = backtest(df, f"{name}_slip{slip}", sig_fn, slippage=slip, **params)
            if r['n'] > 0:
                pr(r)
        print()


# ═══════════════════════════════════════════════════════════════
# Phase E: L7 Cross-Correlation (fixed)
# ═══════════════════════════════════════════════════════════════

def run_phase_E():
    print("\n" + "=" * 80)
    print("Phase E: Cross-Correlation with L7 (fixed)")
    print("=" * 80)

    print("  Loading L7 data & running baseline...")
    data = DataBundle.load_default()
    L7_KWARGS = {
        'trailing_activate_atr': 0.28, 'trailing_distance_atr': 0.06,
        'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0,
        'keltner_adx_threshold': 18, 'max_positions': 1,
        'intraday_adaptive': True, 'choppy_threshold': 0.50,
        'kc_only_threshold': 0.60, 'min_entry_gap_hours': 1.0,
        'regime_config': {
            'low_pct': 25, 'high_pct': 75,
            'low_trail': (0.10, 0.025), 'mid_trail': (0.20, 0.04),
            'high_trail': (0.28, 0.06),
        },
        'time_adaptive_trail': True, 'time_adaptive_trail_start': 2,
        'time_adaptive_trail_decay': 0.75, 'time_adaptive_trail_floor': 0.003,
        'keltner_max_hold_m15': 8,
        'spread_cost': 0.30,
    }
    l7_result = run_variant(data, "L7_MH8", verbose=False, **L7_KWARGS)

    l7_daily = {}
    trade_list = l7_result.get('trades', l7_result.get('_trades', []))
    if not trade_list:
        for key in l7_result:
            val = l7_result[key]
            if isinstance(val, list) and len(val) > 0:
                print(f"    Key '{key}': type={type(val[0])}, len={len(val)}")
        print("  WARNING: Could not find L7 trades. Dumping keys:", list(l7_result.keys()))
        return

    for t in trade_list:
        exit_t = getattr(t, 'exit_time', None) or t.get('exit_time', None)
        pnl = getattr(t, 'pnl', None) or t.get('pnl', 0)
        if exit_t is None: continue
        d = pd.Timestamp(exit_t).date()
        l7_daily.setdefault(d, 0); l7_daily[d] += pnl

    l7_daily_s = pd.Series(l7_daily).sort_index()
    l7_sh = sharpe_from_daily(l7_daily)
    l7_pnl = l7_daily_s.sum()
    print(f"  L7(MH=8): Sharpe={l7_sh:.2f}, PnL=${l7_pnl:.0f}, TradingDays={len(l7_daily_s)}")

    new_strats = [
        ('D1_Don50', d1_df, lambda d,i: don_sig(d,i,50),
         dict(sl_atr=2.0, tp_atr=5.0, trail_act_atr=1.5, trail_dist_atr=0.5, max_hold=60, cooldown=3)),
        ('D1+H4_TopDown', h4_df, lambda d,i: topdown_sig(d,i),
         dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=40, cooldown=3)),
        ('H4_ROC5_EMA200', h4_df, lambda d,i: roc_sig(d,i,1.0),
         dict(sl_atr=2.0, tp_atr=4.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=30, cooldown=3)),
    ]

    print(f"\n  {'Strategy':>25} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'Corr':>7} | "
          f"{'Combo_Sh':>8} {'Combo_PnL':>10}")
    print("  " + "-" * 85)

    for sname, df, sig_fn, params in new_strats:
        r = backtest(df, sname, sig_fn, **params)
        if not r['_daily_pnl']: continue
        new_daily = pd.Series(r['_daily_pnl']).sort_index()
        all_idx = sorted(set(l7_daily_s.index) | set(new_daily.index))
        corr_df = pd.DataFrame({
            'L7': l7_daily_s.reindex(all_idx, fill_value=0),
            sname: new_daily.reindex(all_idx, fill_value=0),
        })
        corr = corr_df.corr().iloc[0, 1]
        combo = corr_df['L7'] + corr_df[sname]
        combo_sh = combo.mean() / combo.std() * np.sqrt(252) if combo.std() > 0 else 0
        combo_pnl = combo.sum()
        print(f"  {sname:>25} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} {corr:>7.3f} | "
              f"{combo_sh:>8.2f} ${combo_pnl:>9.0f}")

        for w in [0.25, 0.5, 1.0, 1.5, 2.0]:
            port = corr_df['L7'] + w * corr_df[sname]
            p_sh = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else 0
            print(f"  {'':>25}   w={w:.2f}: Sharpe={p_sh:.2f}, PnL=${port.sum():.0f}")


# ═══════════════════════════════════════════════════════════════
# Phase F: Full Realistic D1 Backtest with K-Fold
# ═══════════════════════════════════════════════════════════════

def run_phase_F():
    print("\n" + "=" * 80)
    print("Phase F: Best Realistic Config — K-Fold + Yearly + Walk-Forward")
    print("=" * 80)

    best_configs = [
        ('D1_Don50_realistic', d1_df, lambda d,i: don_sig(d,i,50),
         dict(sl_atr=2.0, tp_atr=5.0, trail_act_atr=1.5, trail_dist_atr=0.5,
              max_hold=60, cooldown=3, swap_per_day=5, slippage=1.0)),
        ('D1_Don50_conservative', d1_df, lambda d,i: don_sig(d,i,50),
         dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=999, trail_dist_atr=999,
              max_hold=40, cooldown=5, swap_per_day=5, slippage=1.0)),
        ('TopDown_realistic', h4_df, lambda d,i: topdown_sig(d,i),
         dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=1.0, trail_dist_atr=0.3,
              max_hold=40, cooldown=3, swap_per_day=3, slippage=0.5)),
        ('H4_ROC5_realistic', h4_df, lambda d,i: roc_sig(d,i,1.0),
         dict(sl_atr=2.0, tp_atr=4.0, trail_act_atr=1.0, trail_dist_atr=0.3,
              max_hold=30, cooldown=3, swap_per_day=3, slippage=0.5)),
    ]

    for name, df, sig_fn, params in best_configs:
        print(f"\n  === {name} ===")
        r = backtest(df, name, sig_fn, **params)
        pr_detail(r)
        trades = r['_trades']
        if not trades:
            continue

        # Yearly breakdown
        print(f"\n    Yearly:")
        print(f"    {'Year':>6} {'N':>5} {'PnL':>9} {'WR':>6} {'Sharpe':>7}")
        for year in range(2015, 2027):
            yt = [t for t in trades if pd.Timestamp(t['exit_time']).year == year]
            if not yt: continue
            yp = sum(t['pnl'] for t in yt)
            yw = sum(1 for t in yt if t['pnl'] > 0) / len(yt) * 100
            yd = {}
            for t in yt:
                d = pd.Timestamp(t['exit_time']).date()
                yd.setdefault(d, 0); yd[d] += t['pnl']
            da = np.array(list(yd.values()))
            ysh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            print(f"    {year:>6} {len(yt):>5} ${yp:>8.0f} {yw:>5.1f}% {ysh:>7.2f}")

        # 6-fold K-Fold via time windows
        print(f"\n    6-Fold K-Fold:")
        folds = [
            ("F1", "2015-01-01", "2017-01-01"),
            ("F2", "2017-01-01", "2019-01-01"),
            ("F3", "2019-01-01", "2021-01-01"),
            ("F4", "2021-01-01", "2023-01-01"),
            ("F5", "2023-01-01", "2025-01-01"),
            ("F6", "2025-01-01", "2026-04-10"),
        ]
        sharpes = []
        for fname, start, end in folds:
            sd = pd.Timestamp(start).date(); ed = pd.Timestamp(end).date()
            ft = [t for t in trades
                  if sd <= pd.Timestamp(t['exit_time']).date() < ed]
            if not ft:
                print(f"    {fname}: no trades")
                sharpes.append(0)
                continue
            fp = sum(t['pnl'] for t in ft)
            fw = sum(1 for t in ft if t['pnl'] > 0) / len(ft) * 100
            fd = {}
            for t in ft:
                d = pd.Timestamp(t['exit_time']).date()
                fd.setdefault(d, 0); fd[d] += t['pnl']
            da = np.array(list(fd.values()))
            fsh = da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0
            sharpes.append(fsh)
            status = "PASS" if fsh > 0 else "FAIL"
            print(f"    {fname} ({start}~{end}): N={len(ft):>4}, Sharpe={fsh:>7.2f}, "
                  f"PnL=${fp:>8.0f}, WR={fw:>5.1f}% {status}")
        pos = sum(1 for s in sharpes if s > 0)
        print(f"    K-Fold: {pos}/{len(sharpes)} pos, "
              f"mean={np.mean(sharpes):.2f}, std={np.std(sharpes):.2f}, min={min(sharpes):.2f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    run_phase_A()
    print(f"\n  Phase A done: {time.time()-t0:.0f}s")

    run_phase_B()
    print(f"\n  Phase B done: {time.time()-t0:.0f}s")

    run_phase_C()
    print(f"\n  Phase C done: {time.time()-t0:.0f}s")

    run_phase_D()
    print(f"\n  Phase D done: {time.time()-t0:.0f}s")

    try:
        run_phase_E()
        print(f"\n  Phase E done: {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"\n  Phase E error: {e}")
        import traceback; traceback.print_exc()

    run_phase_F()
    print(f"\n  Phase F done: {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"R37C completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")
