"""
R37C Phase E Fix: Cross-Correlation with L7
Fix: TradeRecord uses attributes (.exit_time, .pnl), not dict .get()
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_FILE = "R37C_phaseE_output.txt"

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
print(f"R37C Phase E Fix started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

from backtest.runner import load_m15, load_h1_aligned, DataBundle, run_variant

M15_PATH = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
H1_PATH = Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv")

print("\nLoading HTF data...")
m15_df_raw = load_m15(M15_PATH)
h1_df = load_h1_aligned(H1_PATH, m15_df_raw.index[0])

def resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

h4_df = resample_ohlcv(h1_df, '4h')
d1_df = resample_ohlcv(h1_df, '1D')

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
    df['ADX'] = _calc_adx(df, 14)
    for span in [9, 20, 25, 50, 100, 200]:
        df[f'EMA{span}'] = df['Close'].ewm(span=span).mean()
    for n in [50]:
        df[f'Donchian_high_{n}'] = df['High'].rolling(n).max()
        df[f'Donchian_low_{n}'] = df['Low'].rolling(n).min()
    df['ROC5'] = df['Close'].pct_change(5) * 100
    return df

h4_df = add_indicators(h4_df)
d1_df = add_indicators(d1_df)

# ── Backtester (same as R37C) ──

def backtest(df, label, signal_fn, sl_atr=3.5, tp_atr=8.0,
             trail_act_atr=0.28, trail_dist_atr=0.06, max_hold=50,
             spread=0.30, lot=0.03, cooldown=2,
             swap_per_day=0.0, slippage=0.0):
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
                trades.append(_t(pos, close[i], times[i], "TP", i, tp_val - swap_cost)); exited = True
            elif pnl_l <= -sl_val:
                trades.append(_t(pos, close[i], times[i], "SL", i, -sl_val - swap_cost)); exited = True
            else:
                ad = trail_act_atr * pos['atr']; td = trail_dist_atr * pos['atr']
                if pos['dir'] == 'BUY' and high[i] - pos['entry'] >= ad:
                    ts = high[i] - td
                    if low[i] <= ts:
                        raw_pnl = (ts - pos['entry'] - total_spread) * lot * 100
                        trades.append(_t(pos, close[i], times[i], "Trail", i, raw_pnl - swap_cost)); exited = True
                elif pos['dir'] == 'SELL' and pos['entry'] - low[i] >= ad:
                    ts = low[i] + td
                    if high[i] >= ts:
                        raw_pnl = (pos['entry'] - ts - total_spread) * lot * 100
                        trades.append(_t(pos, close[i], times[i], "Trail", i, raw_pnl - swap_cost)); exited = True
                if not exited and held >= max_hold:
                    trades.append(_t(pos, close[i], times[i], "Timeout", i, pnl_c - swap_cost)); exited = True
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

def _t(pos, exit_price, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_price,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'exit_reason': reason, 'bars_held': bar_i - pos['bar']}

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
    return {'label': label, 'n': len(trades), 'sharpe': sh, 'total_pnl': sum(pnls),
            'win_rate': wins / len(trades) * 100, 'max_dd': dd,
            '_trades': trades, '_daily_pnl': daily}

def sharpe_from_daily(daily_dict):
    if not daily_dict: return 0
    da = np.array(list(daily_dict.values()))
    return da.mean() / da.std() * np.sqrt(252) if len(da) > 1 and da.std() > 0 else 0

# ── Signal functions ──

def don_sig(d, i, period=50):
    if i < 2: return None
    hc = f'Donchian_high_{period}'; lc = f'Donchian_low_{period}'
    ch = d[hc].iloc[i-1]; cl = d[lc].iloc[i-1]; c = d['Close'].iloc[i]
    if np.isnan(ch): return None
    if c > ch: return 'BUY'
    if c < cl: return 'SELL'
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
# Phase E: Cross-Correlation with L7 (FIXED)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Phase E: Cross-Correlation with L7 (FIXED)")
print("=" * 80)

print("  Loading L7 data & running baseline...")
t0 = time.time()
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

# FIX: TradeRecord uses attributes, not dict access
l7_daily = {}
trade_list = l7_result.get('trades', [])
if not trade_list:
    for key in sorted(l7_result.keys()):
        val = l7_result[key]
        if isinstance(val, (list, tuple)) and len(val) > 0:
            print(f"    Key '{key}': len={len(val)}, type[0]={type(val[0]).__name__}")

    possible_keys = [k for k in l7_result if isinstance(l7_result[k], list) and len(l7_result[k]) > 5]
    if possible_keys:
        key = possible_keys[0]
        trade_list = l7_result[key]
        print(f"  Using key '{key}' with {len(trade_list)} items")

if trade_list:
    sample = trade_list[0]
    print(f"  TradeRecord type: {type(sample).__name__}")
    print(f"  TradeRecord attrs: {[a for a in dir(sample) if not a.startswith('_')]}")

    for t in trade_list:
        exit_t = getattr(t, 'exit_time', None)
        pnl = getattr(t, 'pnl', None)
        if exit_t is None or pnl is None:
            continue
        d = pd.Timestamp(exit_t).date()
        l7_daily.setdefault(d, 0)
        l7_daily[d] += pnl

l7_daily_s = pd.Series(l7_daily).sort_index()
l7_sh = sharpe_from_daily(l7_daily)
l7_pnl = l7_daily_s.sum()
l7_n = len(trade_list) if trade_list else 0
print(f"  L7(MH=8): N={l7_n}, Sharpe={l7_sh:.2f}, PnL=${l7_pnl:.0f}, TradingDays={len(l7_daily_s)}")

# HTF strategies with realistic params (from R37C Phase F)
new_strats = [
    ('D1_Don50_realistic', d1_df, lambda d,i: don_sig(d,i,50),
     dict(sl_atr=2.0, tp_atr=5.0, trail_act_atr=1.5, trail_dist_atr=0.5, max_hold=60, cooldown=3)),
    ('D1_Don50_conservative', d1_df, lambda d,i: don_sig(d,i,50),
     dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=999, trail_dist_atr=999, max_hold=40, cooldown=5)),
    ('TopDown_realistic', h4_df, lambda d,i: topdown_sig(d,i),
     dict(sl_atr=1.5, tp_atr=3.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=40, cooldown=3)),
    ('H4_ROC5_realistic', h4_df, lambda d,i: roc_sig(d,i,1.0),
     dict(sl_atr=2.0, tp_atr=4.0, trail_act_atr=1.0, trail_dist_atr=0.3, max_hold=30, cooldown=3)),
]

print(f"\n  {'Strategy':>25} {'N':>5} {'Sharpe':>7} {'PnL':>9} {'Corr':>7} | "
      f"{'Combo_Sh':>8} {'Combo_PnL':>10}")
print("  " + "-" * 85)

for sname, df, sig_fn, params in new_strats:
    r = backtest(df, sname, sig_fn, **params)
    if not r['_daily_pnl']:
        print(f"  {sname:>25}: no trades")
        continue
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

    for w in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        port = corr_df['L7'] + w * corr_df[sname]
        p_sh = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else 0
        print(f"  {'':>25}   w={w:.2f}: Sharpe={p_sh:.2f}, PnL=${port.sum():.0f}")
    print()

# Combined portfolio: L7 + all HTF
print("\n  --- Combined Portfolio: L7 + All HTF ---")
all_daily = l7_daily_s.copy()
for sname, df, sig_fn, params in new_strats:
    r = backtest(df, sname, sig_fn, **params)
    if not r['_daily_pnl']: continue
    nd = pd.Series(r['_daily_pnl'])
    for d, pnl in nd.items():
        all_daily[d] = all_daily.get(d, 0) + pnl

all_daily = all_daily.sort_index()
all_da = all_daily.values
all_sh = all_da.mean() / all_da.std() * np.sqrt(252) if len(all_da) > 1 and all_da.std() > 0 else 0
all_pnl = all_da.sum()
print(f"  L7 alone:          Sharpe={l7_sh:.2f}, PnL=${l7_pnl:.0f}")
print(f"  L7 + ALL 4 HTF:    Sharpe={all_sh:.2f}, PnL=${all_pnl:.0f}")
print(f"  Improvement:       Sharpe +{(all_sh/l7_sh - 1)*100:.1f}%, PnL +${all_pnl - l7_pnl:.0f}")

elapsed = time.time() - t0
print(f"\n{'='*80}")
print(f"Phase E fix completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*80}")
