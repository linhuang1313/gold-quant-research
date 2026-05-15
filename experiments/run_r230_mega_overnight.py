#!/usr/bin/env python3
"""R230: 200-Hour Mega Overnight Experiment
=============================================
Comprehensive strategy research across M30, H1, H4 timeframes.

Architecture:
  Part A: M30 Strategy Universe (12 strategies × deep validation)
  Part B: H4 Strategy Universe (15 strategies × deep validation)
  Part C: H1 Strategy Universe (via BacktestEngine monkey-patch, 9 strategies)
  Part D: Cross-timeframe portfolio construction
  Part E: Slippage stress testing on all winners

Each timeframe pipeline:
  Phase 1: Screening (baseline + era breakdown)
  Phase 2: 6-Fold K-Fold validation
  Phase 3: Extended parameter sweep (very large grid)
  Phase 4: Walk-Forward optimization (6 expanding windows)
  Phase 5: Era stability (4 eras, require 3/4 positive)
  Phase 6: Parameter sensitivity (±20%, ±40% perturbation)
  Phase 7: Monte Carlo bootstrap (5000 resamples)
  Phase 8: Drawdown stress test
  Phase 9: Slippage testing (none / fixed / empirical / realistic)
  Phase 10: Final multi-gate verdict

Estimated runtime: ~200 hours on single-core CPU.
"""
from __future__ import annotations
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.h4_engine import H4BacktestEngine, load_h4_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r230_mega_overnight")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
N_BOOTSTRAP = 5000

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

WF_CUTOFFS = [
    ("2015-01-01", "2017-01-01", "2017-01-01", "2018-10-01"),
    ("2015-01-01", "2018-10-01", "2018-10-01", "2020-07-01"),
    ("2015-01-01", "2020-07-01", "2020-07-01", "2022-04-01"),
    ("2015-01-01", "2022-04-01", "2022-04-01", "2024-01-01"),
    ("2015-01-01", "2024-01-01", "2024-01-01", "2025-07-01"),
    ("2015-01-01", "2025-07-01", "2025-07-01", "2026-06-01"),
]

# Extended parameter grid for deep sweep
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
TP_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
TRAIL_GRID = [
    (0.0, 0.0),
    (0.15, 0.04), (0.2, 0.06), (0.25, 0.07),
    (0.3, 0.08), (0.4, 0.10), (0.5, 0.12),
    (0.5, 0.15), (0.8, 0.20), (1.0, 0.25),
]

# Slippage test configs
SLIPPAGE_CONFIGS = [
    {"name": "no_slippage", "slippage_model": "none"},
    {"name": "fixed_slippage", "slippage_model": "fixed"},
    {"name": "empirical_slippage", "slippage_model": "empirical"},
    {"name": "realistic_slippage", "slippage_model": "realistic"},
]

PROGRESS_FILE = OUTPUT_DIR / "_progress.json"


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')
    sys.stdout.flush()


def save_progress(part, phase, detail=""):
    prog = {}
    if PROGRESS_FILE.exists():
        try:
            prog = json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    prog[f'{part}_{phase}'] = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'detail': detail,
    }
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2, default=str))


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_dd': 0, 'profit_factor': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252)) if n > 1 else 0
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.9
    return {
        'n': n, 'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 4),
        'max_dd': round(float(dd.max()), 2),
        'profit_factor': round(pf, 3),
    }


def filter_period(trades, start, end):
    ts_s, ts_e = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts_s <= pd.Timestamp(t.entry_time) < ts_e]


def kfold_6(trades):
    if len(trades) < 30:
        return {'skip': True, 'reason': f'n={len(trades)} < 30', 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 6
    folds, kf_pass = [], 0
    for fold in range(6):
        s_idx = fold * fold_size
        e_idx = s_idx + fold_size if fold < 5 else len(pnls)
        fp = pnls[s_idx:e_idx]
        if len(fp) < 5:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0:
            kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.67 else 'FAIL'}


def kfold_10(trades):
    """More rigorous 10-fold validation."""
    if len(trades) < 50:
        return {'skip': True, 'verdict': 'SKIP'}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // 10
    folds, kf_pass = [], 0
    for fold in range(10):
        s_idx = fold * fold_size
        e_idx = s_idx + fold_size if fold < 9 else len(pnls)
        fp = pnls[s_idx:e_idx]
        if len(fp) < 3:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        folds.append({'fold': fold + 1, 'n': len(fp), 'sharpe': round(sh, 3)})
        if sh > 0:
            kf_pass += 1
    rate = kf_pass / max(len(folds), 1)
    return {'folds': folds, 'pass_count': kf_pass, 'total_folds': len(folds),
            'pass_rate': round(rate, 3), 'verdict': 'PASS' if rate >= 0.70 else 'FAIL'}


def monte_carlo_bootstrap(trades, n_boot=N_BOOTSTRAP, seed=42):
    if len(trades) < 20:
        return {'skip': True, 'n': len(trades)}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    rng = np.random.default_rng(seed)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(pnls, size=n, replace=True)
        sh = float(sample.mean() / max(sample.std(ddof=1), 1e-9) * np.sqrt(252))
        boot_sharpes.append(sh)
    boot_arr = np.array(boot_sharpes)
    p_value = (boot_arr <= 0).sum() / n_boot
    return {
        'n': n, 'n_bootstrap': n_boot,
        'p_value': round(float(p_value), 5),
        'median_sharpe': round(float(np.median(boot_arr)), 3),
        'ci_5': round(float(np.percentile(boot_arr, 5)), 3),
        'ci_95': round(float(np.percentile(boot_arr, 95)), 3),
        'ci_1': round(float(np.percentile(boot_arr, 1)), 3),
        'mc_verdict': 'PASS' if p_value < 0.05 else 'FAIL',
    }


def drawdown_analysis(trades):
    if len(trades) < 10:
        return {'skip': True}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    peaks = np.maximum.accumulate(cum)
    drawdowns = peaks - cum
    max_dd = float(drawdowns.max())
    max_dd_idx = int(drawdowns.argmax())
    streak, max_streak = 0, 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    recovery_bars = 0
    if max_dd_idx < n - 1:
        for ri in range(max_dd_idx + 1, n):
            if cum[ri] >= peaks[max_dd_idx]:
                recovery_bars = ri - max_dd_idx
                break
    return {
        'max_dd': round(max_dd, 2),
        'worst_losing_streak': max_streak,
        'recovery_trades': recovery_bars,
        'total_pnl': round(float(cum[-1]), 2),
    }


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
# M30 Signal Functions (12 strategies)
# ═══════════════════════════════════════════════════════════════

def m30_sig_kc_breakout(df):
    if len(df) < 30: return None
    row = df.iloc[-1]
    c, kc_u, kc_l = float(row['Close']), float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    if pd.isna(kc_u) or kc_u == 0 or float(row.get('ATR', 0)) <= 0: return None
    if c > kc_u: return {'strategy': 'm30_kc', 'signal': 'BUY'}
    if c < kc_l: return {'strategy': 'm30_kc', 'signal': 'SELL'}
    return None

def m30_sig_ema_fast_cross(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e9, e20 = float(curr['EMA9']), float(curr['EMA20'])
    e9p, e20p = float(prev['EMA9']), float(prev['EMA20'])
    if pd.isna(e9) or pd.isna(e20) or float(curr.get('ATR', 0)) <= 0: return None
    if e9 > e20 and e9p <= e20p: return {'strategy': 'm30_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p: return {'strategy': 'm30_ema_fast', 'signal': 'SELL'}
    return None

def m30_sig_ema_cross(df):
    if len(df) < 55: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20, e50 = float(curr['EMA20']), float(curr['EMA50'])
    e20p, e50p = float(prev['EMA20']), float(prev['EMA50'])
    if pd.isna(e20) or pd.isna(e50) or float(curr.get('ATR', 0)) <= 0: return None
    if e20 > e50 and e20p <= e50p: return {'strategy': 'm30_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p: return {'strategy': 'm30_ema_cross', 'signal': 'SELL'}
    return None

def m30_sig_macd_cross(df):
    if len(df) < 30: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd, sig = float(curr['MACD']), float(curr['MACD_signal'])
    macd_p, sig_p = float(prev['MACD']), float(prev['MACD_signal'])
    if pd.isna(macd) or pd.isna(sig) or float(curr.get('ATR', 0)) <= 0: return None
    if macd > sig and macd_p <= sig_p: return {'strategy': 'm30_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p: return {'strategy': 'm30_macd', 'signal': 'SELL'}
    return None

def m30_sig_rsi6_extreme(df):
    if len(df) < 20: return None
    row = df.iloc[-1]
    rsi6 = float(row.get('RSI6', 50))
    c, ema200 = float(row['Close']), float(row.get('EMA200', float(row['Close'])))
    if pd.isna(rsi6) or float(row.get('ATR', 0)) <= 0 or pd.isna(ema200): return None
    if rsi6 < 15 and c > ema200: return {'strategy': 'm30_rsi6', 'signal': 'BUY'}
    if rsi6 > 85 and c < ema200: return {'strategy': 'm30_rsi6', 'signal': 'SELL'}
    return None

def m30_sig_rsi14_trend(df):
    if len(df) < 55: return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    c, ema50 = float(row['Close']), float(row.get('EMA50', float(row['Close'])))
    slope = float(row.get('EMA50_slope', 0))
    if pd.isna(rsi) or float(row.get('ATR', 0)) <= 0: return None
    if rsi < 30 and c > ema50 and slope > 0: return {'strategy': 'm30_rsi14', 'signal': 'BUY'}
    if rsi > 70 and c < ema50 and slope < 0: return {'strategy': 'm30_rsi14', 'signal': 'SELL'}
    return None

def m30_sig_cci_momentum(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci, cci_p = float(curr.get('CCI', 0)), float(prev.get('CCI', 0))
    slope = float(curr.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_p) or float(curr.get('ATR', 0)) <= 0: return None
    if cci > 0 and cci_p <= 0 and slope > 0: return {'strategy': 'm30_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0: return {'strategy': 'm30_cci', 'signal': 'SELL'}
    return None

def m30_sig_stochastic(df):
    if len(df) < 20: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    k, d = float(curr.get('STOCH_K', 50)), float(curr.get('STOCH_D', 50))
    kp, dp = float(prev.get('STOCH_K', 50)), float(prev.get('STOCH_D', 50))
    if pd.isna(k) or pd.isna(d) or float(curr.get('ATR', 0)) <= 0: return None
    if k > d and kp <= dp and k < 30: return {'strategy': 'm30_stoch', 'signal': 'BUY'}
    if k < d and kp >= dp and k > 70: return {'strategy': 'm30_stoch', 'signal': 'SELL'}
    return None

def m30_sig_bb_squeeze(df):
    if len(df) < 15: return None
    row = df.iloc[-1]
    bb_u, bb_l = float(row.get('BB_upper', 0)), float(row.get('BB_lower', 0))
    kc_u, kc_l = float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    c = float(row['Close'])
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or float(row.get('ATR', 0)) <= 0: return None
    if (bb_u < kc_u) and (bb_l > kc_l): return None
    squeeze_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        if (float(r.get('BB_upper', 0)) < float(r.get('KC_upper', 0))
            and float(r.get('BB_lower', 0)) > float(r.get('KC_lower', 0))):
            squeeze_count += 1
        else:
            squeeze_count = 0
    if squeeze_count < 5: return None
    kc_mid = float(row.get('KC_mid', 0))
    if c > kc_mid: return {'strategy': 'm30_squeeze', 'signal': 'BUY'}
    return {'strategy': 'm30_squeeze', 'signal': 'SELL'}

def m30_sig_mean_revert(df):
    if len(df) < 25: return None
    row = df.iloc[-1]
    c, sma20 = float(row['Close']), float(row.get('SMA20', float(row['Close'])))
    atr = float(row.get('ATR', 0))
    slope = float(row.get('EMA50_slope', 0))
    if pd.isna(sma20) or atr <= 0: return None
    dev = (c - sma20) / atr
    if dev < -2.0 and slope > -0.5: return {'strategy': 'm30_mean_rev', 'signal': 'BUY'}
    if dev > 2.0 and slope < 0.5: return {'strategy': 'm30_mean_rev', 'signal': 'SELL'}
    return None

def m30_sig_inside_bar(df):
    if len(df) < 5: return None
    curr = df.iloc[-1]
    c_c = float(curr['Close'])
    if float(curr.get('ATR', 0)) <= 0: return None
    was_inside = (float(df.iloc[-3]['High']) <= float(df.iloc[-4]['High']) and
                  float(df.iloc[-3]['Low']) >= float(df.iloc[-4]['Low'])) if len(df) >= 5 else False
    if not was_inside: return None
    mother_high, mother_low = float(df.iloc[-4]['High']), float(df.iloc[-4]['Low'])
    if c_c > mother_high: return {'strategy': 'm30_inside_bar', 'signal': 'BUY'}
    if c_c < mother_low: return {'strategy': 'm30_inside_bar', 'signal': 'SELL'}
    return None

def m30_sig_engulfing(df):
    if len(df) < 10: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    c_o = float(curr.get('Open', curr.get('open', 0)))
    c_c = float(curr['Close'])
    p_o = float(prev.get('Open', prev.get('open', 0)))
    p_c = float(prev['Close'])
    ema50 = float(curr.get('EMA50', c_c))
    if float(curr.get('ATR', 0)) <= 0 or pd.isna(ema50): return None
    curr_body, prev_body = c_c - c_o, p_c - p_o
    if curr_body > 0 and prev_body < 0 and c_c > p_o and c_o < p_c and c_c > ema50:
        return {'strategy': 'm30_engulf', 'signal': 'BUY'}
    if curr_body < 0 and prev_body > 0 and c_c < p_o and c_o > p_c and c_c < ema50:
        return {'strategy': 'm30_engulf', 'signal': 'SELL'}
    return None

M30_STRATEGIES = [
    ('m30_kc', m30_sig_kc_breakout),
    ('m30_ema_fast', m30_sig_ema_fast_cross),
    ('m30_ema_cross', m30_sig_ema_cross),
    ('m30_macd', m30_sig_macd_cross),
    ('m30_rsi6', m30_sig_rsi6_extreme),
    ('m30_rsi14', m30_sig_rsi14_trend),
    ('m30_cci', m30_sig_cci_momentum),
    ('m30_stoch', m30_sig_stochastic),
    ('m30_squeeze', m30_sig_bb_squeeze),
    ('m30_mean_rev', m30_sig_mean_revert),
    ('m30_inside_bar', m30_sig_inside_bar),
    ('m30_engulf', m30_sig_engulfing),
]


# ═══════════════════════════════════════════════════════════════
# H4 Signal Functions (15 strategies = R220 7 + R224 8)
# ═══════════════════════════════════════════════════════════════

def h4_sig_kc_breakout(df):
    if len(df) < 30: return None
    row = df.iloc[-1]
    c, kc_u, kc_l = float(row['Close']), float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    if pd.isna(kc_u) or kc_u == 0 or float(row.get('ATR', 0)) <= 0: return None
    if c > kc_u: return {'strategy': 'h4_kc', 'signal': 'BUY'}
    if c < kc_l: return {'strategy': 'h4_kc', 'signal': 'SELL'}
    return None

def h4_sig_ema_cross(df):
    if len(df) < 55: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20, e50 = float(curr['EMA20']), float(curr['EMA50'])
    e20p, e50p = float(prev['EMA20']), float(prev['EMA50'])
    if pd.isna(e20) or pd.isna(e50) or float(curr.get('ATR', 0)) <= 0: return None
    if e20 > e50 and e20p <= e50p: return {'strategy': 'h4_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p: return {'strategy': 'h4_ema_cross', 'signal': 'SELL'}
    return None

def h4_sig_macd_cross(df):
    if len(df) < 30: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    macd, sig = float(curr['MACD']), float(curr['MACD_signal'])
    macd_p, sig_p = float(prev['MACD']), float(prev['MACD_signal'])
    if pd.isna(macd) or pd.isna(sig) or float(curr.get('ATR', 0)) <= 0: return None
    if macd > sig and macd_p <= sig_p: return {'strategy': 'h4_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p: return {'strategy': 'h4_macd', 'signal': 'SELL'}
    return None

def h4_sig_rsi(df):
    if len(df) < 30: return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    c = float(row['Close'])
    ema100 = float(row.get('EMA100', c))
    if pd.isna(rsi) or float(row.get('ATR', 0)) <= 0: return None
    if rsi < 25 and c > ema100: return {'strategy': 'h4_rsi', 'signal': 'BUY'}
    if rsi > 75 and c < ema100: return {'strategy': 'h4_rsi', 'signal': 'SELL'}
    return None

def h4_sig_cci_momentum(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    cci, cci_p = float(curr.get('CCI', 0)), float(prev.get('CCI', 0))
    slope = float(curr.get('EMA50_slope', 0))
    if pd.isna(cci) or pd.isna(cci_p) or float(curr.get('ATR', 0)) <= 0: return None
    if cci > 0 and cci_p <= 0 and slope > 0: return {'strategy': 'h4_cci', 'signal': 'BUY'}
    if cci < 0 and cci_p >= 0 and slope < 0: return {'strategy': 'h4_cci', 'signal': 'SELL'}
    return None

def h4_sig_squeeze(df):
    if len(df) < 15: return None
    row = df.iloc[-1]
    bb_u, bb_l = float(row.get('BB_upper', 0)), float(row.get('BB_lower', 0))
    kc_u, kc_l = float(row.get('KC_upper', 0)), float(row.get('KC_lower', 0))
    c = float(row['Close'])
    if pd.isna(bb_u) or pd.isna(kc_u) or kc_u == 0 or float(row.get('ATR', 0)) <= 0: return None
    if (bb_u < kc_u) and (bb_l > kc_l): return None
    squeeze_count = 0
    for j in range(max(0, len(df) - 11), len(df) - 1):
        r = df.iloc[j]
        if (float(r.get('BB_upper', 0)) < float(r.get('KC_upper', 0))
            and float(r.get('BB_lower', 0)) > float(r.get('KC_lower', 0))):
            squeeze_count += 1
        else:
            squeeze_count = 0
    if squeeze_count < 5: return None
    kc_mid = float(row.get('KC_mid', 0))
    if c > kc_mid: return {'strategy': 'h4_squeeze', 'signal': 'BUY'}
    return {'strategy': 'h4_squeeze', 'signal': 'SELL'}

def h4_sig_donchian(df):
    if len(df) < 55: return None
    row = df.iloc[-1]
    c = float(row['Close'])
    high50 = float(df['High'].iloc[-51:-1].max())
    low50 = float(df['Low'].iloc[-51:-1].min())
    if float(row.get('ATR', 0)) <= 0: return None
    if c > high50: return {'strategy': 'h4_donchian', 'signal': 'BUY'}
    if c < low50: return {'strategy': 'h4_donchian', 'signal': 'SELL'}
    return None

# --- R224 extended strategies ---
def h4_sig_ema_fast(df):
    if len(df) < 25: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    if 'EMA9' not in df.columns:
        ema9 = df['Close'].ewm(span=9, adjust=False).mean()
        e9, e9p = float(ema9.iloc[-1]), float(ema9.iloc[-2])
    else:
        e9, e9p = float(curr.get('EMA9', 0)), float(prev.get('EMA9', 0))
    e20, e20p = float(curr['EMA20']), float(prev['EMA20'])
    if pd.isna(e9) or pd.isna(e20) or float(curr.get('ATR', 0)) <= 0: return None
    if e9 > e20 and e9p <= e20p: return {'strategy': 'h4_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p: return {'strategy': 'h4_ema_fast', 'signal': 'SELL'}
    return None

def h4_sig_rsi_divergence(df):
    if len(df) < 30: return None
    curr = df.iloc[-1]
    if float(curr.get('ATR', 0)) <= 0: return None
    prices = df['Close'].values[-20:]
    rsi_vals = df['RSI14'].values[-20:] if 'RSI14' in df.columns else None
    if rsi_vals is None or np.any(np.isnan(rsi_vals)) or len(prices) < 10: return None
    p_min1, p_min2 = prices[:10].min(), prices[10:].min()
    r_min1, r_min2 = rsi_vals[:10].min(), rsi_vals[10:].min()
    if p_min2 < p_min1 and r_min2 > r_min1 and float(curr['RSI14']) < 40:
        return {'strategy': 'h4_rsi_div', 'signal': 'BUY'}
    p_max1, p_max2 = prices[:10].max(), prices[10:].max()
    r_max1, r_max2 = rsi_vals[:10].max(), rsi_vals[10:].max()
    if p_max2 > p_max1 and r_max2 < r_max1 and float(curr['RSI14']) > 60:
        return {'strategy': 'h4_rsi_div', 'signal': 'SELL'}
    return None

def h4_sig_adx_di_cross(df):
    if len(df) < 20: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    adx = float(curr.get('ADX', 0))
    if pd.isna(adx) or adx < 25: return None
    tr = pd.concat([df['High'] - df['Low'],
                    (df['High'] - df['Close'].shift(1)).abs(),
                    (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm[df['High'].diff() <= (-df['Low'].diff())] = 0
    minus_dm[(-df['Low'].diff()) <= df['High'].diff()] = 0
    atr_s = tr.ewm(span=14, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, np.nan)
    pdi_c, pdi_p = float(pdi.iloc[-1]), float(pdi.iloc[-2])
    mdi_c, mdi_p = float(mdi.iloc[-1]), float(mdi.iloc[-2])
    if pdi_c > mdi_c and pdi_p <= mdi_p: return {'strategy': 'h4_adx_di', 'signal': 'BUY'}
    if pdi_c < mdi_c and pdi_p >= mdi_p: return {'strategy': 'h4_adx_di', 'signal': 'SELL'}
    return None

def h4_sig_stochastic(df):
    if len(df) < 20: return None
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    stoch_k = 100 * (df['Close'] - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    k_c, d_c = float(stoch_k.iloc[-1]), float(stoch_d.iloc[-1])
    k_p, d_p = float(stoch_k.iloc[-2]), float(stoch_d.iloc[-2])
    if pd.isna(k_c) or pd.isna(d_c): return None
    if k_c > d_c and k_p <= d_p and k_c < 30: return {'strategy': 'h4_stoch', 'signal': 'BUY'}
    if k_c < d_c and k_p >= d_p and k_c > 70: return {'strategy': 'h4_stoch', 'signal': 'SELL'}
    return None

def h4_sig_ema_ribbon(df):
    if len(df) < 105: return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e20, e50, e100 = float(curr['EMA20']), float(curr['EMA50']), float(curr.get('EMA100', 0))
    e20p, e50p, e100p = float(prev['EMA20']), float(prev['EMA50']), float(prev.get('EMA100', 0))
    if pd.isna(e100) or e100 == 0: return None
    if (e20 > e50 > e100) and not (e20p > e50p > e100p):
        return {'strategy': 'h4_ema_ribbon', 'signal': 'BUY'}
    if (e20 < e50 < e100) and not (e20p < e50p < e100p):
        return {'strategy': 'h4_ema_ribbon', 'signal': 'SELL'}
    return None

def h4_sig_mean_revert(df):
    if len(df) < 55: return None
    curr = df.iloc[-1]
    c, ema50 = float(curr['Close']), float(curr['EMA50'])
    atr = float(curr.get('ATR', 0))
    if pd.isna(ema50) or atr <= 0: return None
    dev = (c - ema50) / atr
    if dev < -2.0: return {'strategy': 'h4_mean_rev', 'signal': 'BUY'}
    if dev > 2.0: return {'strategy': 'h4_mean_rev', 'signal': 'SELL'}
    return None

def h4_sig_momentum_breakout(df):
    if len(df) < 15: return None
    curr = df.iloc[-1]
    c, atr = float(curr['Close']), float(curr.get('ATR', 0))
    if atr <= 0: return None
    high10 = float(df['High'].iloc[-11:-1].max())
    low10 = float(df['Low'].iloc[-11:-1].min())
    if c > high10 + 0.5 * atr: return {'strategy': 'h4_momentum', 'signal': 'BUY'}
    if c < low10 - 0.5 * atr: return {'strategy': 'h4_momentum', 'signal': 'SELL'}
    return None

def h4_sig_inside_bar(df):
    if len(df) < 5: return None
    curr, prev1, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    c = float(curr['Close'])
    was_inside = (float(prev1['High']) <= float(prev2['High']) and
                  float(prev1['Low']) >= float(prev2['Low']))
    if not was_inside: return None
    if c > float(prev2['High']): return {'strategy': 'h4_inside_bar', 'signal': 'BUY'}
    if c < float(prev2['Low']): return {'strategy': 'h4_inside_bar', 'signal': 'SELL'}
    return None

H4_STRATEGIES = [
    ('h4_kc', h4_sig_kc_breakout),
    ('h4_ema_cross', h4_sig_ema_cross),
    ('h4_macd', h4_sig_macd_cross),
    ('h4_rsi', h4_sig_rsi),
    ('h4_cci', h4_sig_cci_momentum),
    ('h4_squeeze', h4_sig_squeeze),
    ('h4_donchian', h4_sig_donchian),
    ('h4_ema_fast', h4_sig_ema_fast),
    ('h4_rsi_div', h4_sig_rsi_divergence),
    ('h4_adx_di', h4_sig_adx_di_cross),
    ('h4_stoch', h4_sig_stochastic),
    ('h4_ema_ribbon', h4_sig_ema_ribbon),
    ('h4_mean_rev', h4_sig_mean_revert),
    ('h4_momentum', h4_sig_momentum_breakout),
    ('h4_inside_bar', h4_sig_inside_bar),
]


# ═══════════════════════════════════════════════════════════════
# Generic Timeframe Pipeline
# ═══════════════════════════════════════════════════════════════

def run_timeframe_pipeline(
    tf_name: str,
    df: pd.DataFrame,
    strategies: List[Tuple[str, Callable]],
    engine_cls,
    default_params: dict,
    sl_grid: list,
    tp_grid: list,
    trail_grid: list,
    max_hold_grid: list,
    part_label: str,
):
    """Run the full 10-phase pipeline for one timeframe."""
    t0 = time.time()
    strat_map = dict(strategies)

    def run_single(strat_name, sig_func, params_override=None, **extra_kw):
        p = dict(default_params)
        if params_override:
            p.update(params_override)
        p.update(extra_kw)
        engine = engine_cls(df, signal_funcs=[(strat_name, sig_func)], **p)
        trades = engine.run()
        return [t for t in trades if t.strategy == strat_name]

    def run_single_on_slice(strat_name, sig_func, df_slice, params_override=None):
        p = dict(default_params)
        if params_override:
            p.update(params_override)
        engine = engine_cls(df_slice, signal_funcs=[(strat_name, sig_func)], **p)
        trades = engine.run()
        return [t for t in trades if t.strategy == strat_name]

    # ─── Phase 1: Screening ──────────────────────────────────
    print_flush(f'\n{"="*80}\n{part_label} Phase 1: Strategy Screening\n{"="*80}')
    phase1 = {}
    viable = []
    for strat_name, sig_func in strategies:
        print_flush(f'\n  --- {strat_name} ---')
        trades = run_single(strat_name, sig_func)
        s = calc_stats(trades)
        print_flush(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  '
                     f'PnL=${s["pnl"]:.0f}  WR={s["win_rate"]:.1f}%  PF={s["profit_factor"]:.2f}  MaxDD=${s["max_dd"]:.0f}')
        eras = {}
        for era_name, (es, ee) in ERA_SEGMENTS.items():
            era_t = filter_period(trades, es, ee)
            era_s = calc_stats(era_t)
            eras[era_name] = era_s
            print_flush(f'    {era_name:<30} n={era_s["n"]:>4}  Sharpe={era_s["sharpe"]:.3f}')
        phase1[strat_name] = {'stats': s, 'eras': eras}
        if s['sharpe'] > 0.3 and s['n'] >= 30:
            viable.append(strat_name)
    save(f'{tf_name}_phase1_screening', phase1)
    save_progress(part_label, 'phase1', f'viable={viable}')
    print_flush(f'\n  Viable (Sharpe>0.3, n>=30): {viable}')

    # ─── Phase 2: 6-Fold + 10-Fold K-Fold ────────────────────
    print_flush(f'\n{"="*80}\n{part_label} Phase 2: K-Fold Validation\n{"="*80}')
    phase2 = {}
    for strat_name in viable:
        sig_func = strat_map[strat_name]
        trades = run_single(strat_name, sig_func)
        kf6 = kfold_6(trades)
        kf10 = kfold_10(trades)
        print_flush(f'\n  {strat_name}: 6-Fold={kf6.get("verdict","SKIP")}  10-Fold={kf10.get("verdict","SKIP")}')
        if kf6.get('folds'):
            for f in kf6['folds']:
                print_flush(f'    6F {f["fold"]}: n={f["n"]:>4}  Sharpe={f["sharpe"]:.3f}')
        phase2[strat_name] = {'kfold_6': kf6, 'kfold_10': kf10}
    save(f'{tf_name}_phase2_kfold', phase2)
    save_progress(part_label, 'phase2')

    kf_passers = [s for s in viable if phase2.get(s, {}).get('kfold_6', {}).get('verdict') == 'PASS']
    print_flush(f'\n  K-Fold passers: {kf_passers}')

    # ─── Phase 3: Extended Parameter Sweep ────────────────────
    if kf_passers:
        print_flush(f'\n{"="*80}\n{part_label} Phase 3: Extended Parameter Sweep\n{"="*80}')
        phase3 = {}
        for strat_name in kf_passers:
            sig_func = strat_map[strat_name]
            print_flush(f'\n  --- {strat_name} extended sweep ---')
            best_sharpe, best_params = -999, None
            sweep_results = []
            combo_count = 0
            total_combos = len(sl_grid) * len(tp_grid) * len(trail_grid) * len(max_hold_grid)
            for sl_m in sl_grid:
                for tp_m in tp_grid:
                    if tp_m < sl_m:
                        continue
                    for trail_a, trail_d in trail_grid:
                        for mh in max_hold_grid:
                            combo_count += 1
                            if combo_count % 50 == 0:
                                print_flush(f'    ... combo {combo_count}/{total_combos}')
                            trades = run_single(strat_name, sig_func,
                                                {'sl_atr_mult': sl_m, 'tp_atr_mult': tp_m,
                                                 'trailing_activate_atr': trail_a,
                                                 'trailing_distance_atr': trail_d,
                                                 'max_hold': mh})
                            s = calc_stats(trades)
                            sweep_results.append({
                                'sl': sl_m, 'tp': tp_m, 'trail_a': trail_a, 'trail_d': trail_d,
                                'max_hold': mh, **s
                            })
                            if s['sharpe'] > best_sharpe and s['n'] >= 20:
                                best_sharpe = s['sharpe']
                                best_params = {'sl': sl_m, 'tp': tp_m, 'trail_act': trail_a,
                                               'trail_dist': trail_d, 'max_hold': mh}

            sweep_results.sort(key=lambda x: x['sharpe'], reverse=True)
            for r in sweep_results[:10]:
                print_flush(f'    SL{r["sl"]}_TP{r["tp"]}_T{r["trail_a"]}/{r["trail_d"]}_MH{r["max_hold"]}  '
                            f'n={r["n"]:>5} Sh={r["sharpe"]:.3f} PnL=${r["pnl"]:.0f} PF={r["profit_factor"]:.2f}')
            print_flush(f'  Best: {best_params}  Sharpe={best_sharpe:.3f}')
            phase3[strat_name] = {'best_params': best_params, 'best_sharpe': best_sharpe,
                                  'top10': sweep_results[:10], 'total_combos': combo_count}
        save(f'{tf_name}_phase3_param_sweep', phase3)
        save_progress(part_label, 'phase3')
    else:
        phase3 = {}

    # ─── Phase 4: Walk-Forward Optimization ───────────────────
    deep_candidates = [s for s in kf_passers if phase3.get(s, {}).get('best_sharpe', 0) > 1.0]
    if not deep_candidates:
        deep_candidates = kf_passers[:5]

    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 4: Walk-Forward (6 periods)\n{"="*80}')
        phase4 = {}
        for strat_name in deep_candidates:
            print_flush(f'\n  --- {strat_name} Walk-Forward ---')
            sig_func = strat_map[strat_name]
            best_p = phase3.get(strat_name, {}).get('best_params', {})
            if not best_p:
                best_p = {'sl': default_params.get('sl_atr_mult', 3.0),
                          'tp': default_params.get('tp_atr_mult', 6.0),
                          'trail_act': 0.3, 'trail_dist': 0.08}

            wf_results, oos_sharpes = [], []
            for wf_i, (train_s, train_e, test_s, test_e) in enumerate(WF_CUTOFFS):
                best_sh, best_wf_p = -999, None
                for sl_m in [best_p.get('sl', 3.0) * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]:
                    for tp_m in [best_p.get('tp', 6.0) * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]:
                        sl_m, tp_m = round(sl_m, 1), round(tp_m, 1)
                        if tp_m < sl_m:
                            continue
                        df_train = df[(df.index >= pd.Timestamp(train_s, tz='UTC')) &
                                      (df.index < pd.Timestamp(train_e, tz='UTC'))].copy()
                        if len(df_train) < 100:
                            continue
                        params = {
                            'sl_atr_mult': sl_m, 'tp_atr_mult': tp_m,
                            'trailing_activate_atr': best_p.get('trail_act', 0.3),
                            'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                            'max_hold': best_p.get('max_hold', default_params.get('max_hold', 30)),
                        }
                        trades = run_single_on_slice(strat_name, sig_func, df_train, params)
                        s = calc_stats(trades)
                        if s['sharpe'] > best_sh and s['n'] >= 5:
                            best_sh = s['sharpe']
                            best_wf_p = params.copy()

                if best_wf_p is None:
                    best_wf_p = {'sl_atr_mult': best_p.get('sl', 3.0),
                                 'tp_atr_mult': best_p.get('tp', 6.0),
                                 'trailing_activate_atr': best_p.get('trail_act', 0.3),
                                 'trailing_distance_atr': best_p.get('trail_dist', 0.08)}

                df_test = df[(df.index >= pd.Timestamp(test_s, tz='UTC')) &
                             (df.index < pd.Timestamp(test_e, tz='UTC'))].copy()
                if len(df_test) < 50:
                    wf_results.append({'period': f'{test_s}->{test_e}', 'skip': True})
                    continue

                oos_trades = run_single_on_slice(strat_name, sig_func, df_test, best_wf_p)
                oos_stats = calc_stats(oos_trades)
                oos_sharpes.append(oos_stats['sharpe'])
                wf_results.append({
                    'period': f'{test_s}->{test_e}',
                    'train_best_params': best_wf_p,
                    'train_sharpe': round(best_sh, 3),
                    'oos_n': oos_stats['n'],
                    'oos_sharpe': oos_stats['sharpe'],
                    'oos_pnl': oos_stats['pnl'],
                })
                print_flush(f'    WF{wf_i+1} [{test_s}->{test_e}]: '
                            f'train_Sh={best_sh:.3f} -> OOS n={oos_stats["n"]} Sh={oos_stats["sharpe"]:.3f}')

            valid_oos = [s for s in oos_sharpes if not np.isnan(s)]
            positive_oos = sum(1 for s in valid_oos if s > 0)
            wf_pass_rate = positive_oos / max(len(valid_oos), 1)
            wf_verdict = 'PASS' if wf_pass_rate >= 0.67 else 'FAIL'
            print_flush(f'    WF Verdict: {wf_verdict} ({positive_oos}/{len(valid_oos)} positive)')
            phase4[strat_name] = {
                'walk_forward': wf_results,
                'oos_sharpes': [round(s, 3) for s in valid_oos],
                'wf_pass_rate': round(wf_pass_rate, 3),
                'wf_verdict': wf_verdict,
            }
        save(f'{tf_name}_phase4_walk_forward', phase4)
        save_progress(part_label, 'phase4')
    else:
        phase4 = {}

    # ─── Phase 5: Era Stability ───────────────────────────────
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 5: Era Stability\n{"="*80}')
        phase5 = {}
        for strat_name in deep_candidates:
            sig_func = strat_map[strat_name]
            best_p_raw = phase3.get(strat_name, {}).get('best_params', {})
            params_override = {}
            if best_p_raw:
                params_override = {
                    'sl_atr_mult': best_p_raw.get('sl', default_params.get('sl_atr_mult', 3.0)),
                    'tp_atr_mult': best_p_raw.get('tp', default_params.get('tp_atr_mult', 6.0)),
                    'trailing_activate_atr': best_p_raw.get('trail_act', 0.3),
                    'trailing_distance_atr': best_p_raw.get('trail_dist', 0.08),
                }
            trades = run_single(strat_name, sig_func, params_override)
            positive_eras = 0
            era_results = {}
            for era_name, (es, ee) in ERA_SEGMENTS.items():
                era_t = filter_period(trades, es, ee)
                s = calc_stats(era_t)
                era_results[era_name] = s
                if s['sharpe'] > 0:
                    positive_eras += 1
                print_flush(f'  {strat_name} {era_name:<30} n={s["n"]:>4}  Sharpe={s["sharpe"]:.3f}')
            era_verdict = 'PASS' if positive_eras >= 3 else 'FAIL'
            phase5[strat_name] = {'eras': era_results, 'positive_eras': positive_eras, 'era_verdict': era_verdict}
        save(f'{tf_name}_phase5_era_stability', phase5)
        save_progress(part_label, 'phase5')
    else:
        phase5 = {}

    # ─── Phase 6: Parameter Sensitivity (±20%, ±40%) ──────────
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 6: Parameter Sensitivity\n{"="*80}')
        phase6 = {}
        for strat_name in deep_candidates:
            sig_func = strat_map[strat_name]
            best_p_raw = phase3.get(strat_name, {}).get('best_params', {})
            if not best_p_raw:
                continue
            base_params = {
                'sl_atr_mult': best_p_raw.get('sl', 3.0),
                'tp_atr_mult': best_p_raw.get('tp', 6.0),
                'trailing_activate_atr': best_p_raw.get('trail_act', 0.3),
                'trailing_distance_atr': best_p_raw.get('trail_dist', 0.08),
            }
            base_trades = run_single(strat_name, sig_func, base_params)
            base_sharpe = calc_stats(base_trades)['sharpe']
            perturbations, sharpe_drops = [], []
            for param_name in ['sl_atr_mult', 'tp_atr_mult', 'trailing_activate_atr', 'trailing_distance_atr']:
                base_val = base_params.get(param_name, 0)
                if base_val == 0:
                    continue
                for factor in [0.6, 0.8, 1.2, 1.4]:
                    perturbed = dict(base_params)
                    perturbed[param_name] = round(base_val * factor, 4)
                    p_trades = run_single(strat_name, sig_func, perturbed)
                    p_sh = calc_stats(p_trades)['sharpe']
                    drop = base_sharpe - p_sh
                    drop_pct = (drop / max(abs(base_sharpe), 1e-9)) * 100
                    perturbations.append({
                        'param': param_name, 'base': base_val,
                        'perturbed': perturbed[param_name], 'factor': factor,
                        'sharpe': p_sh, 'drop_pct': round(drop_pct, 1),
                    })
                    sharpe_drops.append(abs(drop_pct))
            max_drop = max(sharpe_drops) if sharpe_drops else 0
            avg_drop = np.mean(sharpe_drops) if sharpe_drops else 0
            sens_verdict = 'STABLE' if max_drop < 40 else ('MODERATE' if max_drop < 70 else 'FRAGILE')
            print_flush(f'  {strat_name}: {sens_verdict} (max_drop={max_drop:.1f}%, avg_drop={avg_drop:.1f}%)')
            phase6[strat_name] = {
                'base_sharpe': base_sharpe, 'perturbations': perturbations,
                'max_drop_pct': round(max_drop, 1), 'avg_drop_pct': round(avg_drop, 1),
                'verdict': sens_verdict,
            }
        save(f'{tf_name}_phase6_sensitivity', phase6)
        save_progress(part_label, 'phase6')
    else:
        phase6 = {}

    # ─── Phase 7: Monte Carlo Bootstrap ───────────────────────
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 7: Monte Carlo ({N_BOOTSTRAP} resamples)\n{"="*80}')
        phase7 = {}
        for strat_name in deep_candidates:
            sig_func = strat_map[strat_name]
            best_p_raw = phase3.get(strat_name, {}).get('best_params', {})
            params_override = {}
            if best_p_raw:
                params_override = {
                    'sl_atr_mult': best_p_raw.get('sl', default_params.get('sl_atr_mult', 3.0)),
                    'tp_atr_mult': best_p_raw.get('tp', default_params.get('tp_atr_mult', 6.0)),
                    'trailing_activate_atr': best_p_raw.get('trail_act', 0.3),
                    'trailing_distance_atr': best_p_raw.get('trail_dist', 0.08),
                }
            trades = run_single(strat_name, sig_func, params_override)
            mc = monte_carlo_bootstrap(trades)
            print_flush(f'  {strat_name}: p={mc.get("p_value","N/A")}  '
                        f'median_Sh={mc.get("median_sharpe","N/A")}  -> {mc.get("mc_verdict","SKIP")}')
            phase7[strat_name] = mc
        save(f'{tf_name}_phase7_monte_carlo', phase7)
        save_progress(part_label, 'phase7')
    else:
        phase7 = {}

    # ─── Phase 8: Drawdown Stress ─────────────────────────────
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 8: Drawdown Stress Test\n{"="*80}')
        phase8 = {}
        for strat_name in deep_candidates:
            sig_func = strat_map[strat_name]
            best_p_raw = phase3.get(strat_name, {}).get('best_params', {})
            params_override = {}
            if best_p_raw:
                params_override = {
                    'sl_atr_mult': best_p_raw.get('sl', default_params.get('sl_atr_mult', 3.0)),
                    'tp_atr_mult': best_p_raw.get('tp', default_params.get('tp_atr_mult', 6.0)),
                    'trailing_activate_atr': best_p_raw.get('trail_act', 0.3),
                    'trailing_distance_atr': best_p_raw.get('trail_dist', 0.08),
                }
            trades = run_single(strat_name, sig_func, params_override)
            dd = drawdown_analysis(trades)
            print_flush(f'  {strat_name}: MaxDD=${dd.get("max_dd",0):.0f}  '
                        f'WorstStreak={dd.get("worst_losing_streak",0)}  Recovery={dd.get("recovery_trades",0)}')
            phase8[strat_name] = dd
        save(f'{tf_name}_phase8_drawdown', phase8)
        save_progress(part_label, 'phase8')
    else:
        phase8 = {}

    # ─── Phase 9: Slippage Testing ────────────────────────────
    winners = [s for s in deep_candidates
               if phase7.get(s, {}).get('mc_verdict') == 'PASS'
               or phase4.get(s, {}).get('wf_verdict') == 'PASS']
    if not winners:
        winners = deep_candidates[:3]

    if winners:
        print_flush(f'\n{"="*80}\n{part_label} Phase 9: Slippage Testing\n{"="*80}')
        phase9 = {}
        for strat_name in winners:
            sig_func = strat_map[strat_name]
            best_p_raw = phase3.get(strat_name, {}).get('best_params', {})
            slip_results = {}
            for slip_cfg in SLIPPAGE_CONFIGS:
                params = dict(default_params)
                if best_p_raw:
                    params.update({
                        'sl_atr_mult': best_p_raw.get('sl', default_params.get('sl_atr_mult', 3.0)),
                        'tp_atr_mult': best_p_raw.get('tp', default_params.get('tp_atr_mult', 6.0)),
                        'trailing_activate_atr': best_p_raw.get('trail_act', 0.3),
                        'trailing_distance_atr': best_p_raw.get('trail_dist', 0.08),
                    })
                params['slippage_model'] = slip_cfg['slippage_model']
                engine = engine_cls(df, signal_funcs=[(strat_name, sig_func)], **params)
                trades = engine.run()
                st = [t for t in trades if t.strategy == strat_name]
                s = calc_stats(st)
                slip_results[slip_cfg['name']] = s
                print_flush(f'  {strat_name} [{slip_cfg["name"]}]: n={s["n"]}  '
                            f'Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')

            no_slip_sh = slip_results.get('no_slippage', {}).get('sharpe', 0)
            real_slip_sh = slip_results.get('realistic_slippage', {}).get('sharpe', 0)
            degradation = (no_slip_sh - real_slip_sh) / max(abs(no_slip_sh), 1e-9) * 100
            slip_verdict = 'ROBUST' if degradation < 30 else ('ACCEPTABLE' if degradation < 60 else 'FRAGILE')
            print_flush(f'  {strat_name} slippage degradation: {degradation:.1f}% -> {slip_verdict}')
            phase9[strat_name] = {
                'results': slip_results,
                'degradation_pct': round(degradation, 1),
                'slip_verdict': slip_verdict,
            }
        save(f'{tf_name}_phase9_slippage', phase9)
        save_progress(part_label, 'phase9')
    else:
        phase9 = {}

    # ─── Phase 10: Final Multi-Gate Verdict ───────────────────
    print_flush(f'\n{"="*80}\n{part_label} Phase 10: Final Multi-Gate Verdict\n{"="*80}')
    phase10 = {}
    for strat_name in viable:
        p1_sh = phase1.get(strat_name, {}).get('stats', {}).get('sharpe', 0)
        kf6_v = phase2.get(strat_name, {}).get('kfold_6', {}).get('verdict', 'SKIP')
        kf10_v = phase2.get(strat_name, {}).get('kfold_10', {}).get('verdict', 'SKIP')
        wf_v = phase4.get(strat_name, {}).get('wf_verdict', 'N/A')
        era_v = phase5.get(strat_name, {}).get('era_verdict', 'N/A')
        mc_v = phase7.get(strat_name, {}).get('mc_verdict', 'N/A')
        sens_v = phase6.get(strat_name, {}).get('verdict', 'N/A')
        slip_v = phase9.get(strat_name, {}).get('slip_verdict', 'N/A')
        best_sh = phase3.get(strat_name, {}).get('best_sharpe', p1_sh)
        best_p = phase3.get(strat_name, {}).get('best_params', {})

        gates = {'kfold_6': kf6_v, 'kfold_10': kf10_v, 'walk_forward': wf_v,
                 'era_stability': era_v, 'monte_carlo': mc_v, 'sensitivity': sens_v,
                 'slippage': slip_v}
        pass_count = sum(1 for v in [kf6_v, wf_v, era_v, mc_v] if v == 'PASS')

        if pass_count >= 4 and sens_v != 'FRAGILE' and slip_v in ('ROBUST', 'ACCEPTABLE'):
            final = 'STRONG_PASS'
        elif pass_count >= 3 and mc_v == 'PASS':
            final = 'CONDITIONAL_PASS'
        elif pass_count >= 2:
            final = 'WEAK_PASS'
        elif kf6_v == 'PASS':
            final = 'MARGINAL'
        else:
            final = 'REJECT'

        phase10[strat_name] = {
            'baseline_sharpe': p1_sh,
            'best_sharpe': best_sh,
            'best_params': best_p,
            'gates': gates,
            'pass_count': pass_count,
            'final_verdict': final,
        }
        print_flush(f'  {strat_name:<20} KF6={kf6_v:<5} KF10={kf10_v:<5} WF={wf_v:<5} '
                     f'Era={era_v:<5} MC={mc_v:<5} Sens={sens_v:<8} Slip={slip_v:<10} -> {final}')

    save(f'{tf_name}_phase10_verdict', phase10)
    save_progress(part_label, 'phase10_done')

    elapsed = time.time() - t0
    print_flush(f'\n  {part_label} Total runtime: {elapsed:.0f}s ({elapsed/3600:.1f}h)')

    return {
        'phase1': phase1, 'phase2': phase2, 'phase3': phase3,
        'phase4': phase4, 'phase5': phase5, 'phase6': phase6,
        'phase7': phase7, 'phase8': phase8, 'phase9': phase9,
        'phase10': phase10, 'viable': viable, 'kf_passers': kf_passers,
        'deep_candidates': deep_candidates, 'winners': winners,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print_flush('=' * 80)
    print_flush('R230: 200-Hour Mega Overnight Experiment')
    print_flush(f'Started: {pd.Timestamp.now()}')
    print_flush('=' * 80)

    # ══════════════════════════════════════════════════════════
    # PART A: M30 Strategies
    # ══════════════════════════════════════════════════════════
    try:
        print_flush(f'\n{"#"*80}\n# PART A: M30 Strategy Universe (12 strategies)\n{"#"*80}')
        m30_df = load_m30_with_indicators()
        m30_results = run_timeframe_pipeline(
            tf_name='m30',
            df=m30_df,
            strategies=M30_STRATEGIES,
            engine_cls=M30BacktestEngine,
            default_params={
                'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 48, 'cooldown_bars': 4, 'spread_cost': SPREAD,
            },
            sl_grid=SL_GRID,
            tp_grid=TP_GRID,
            trail_grid=TRAIL_GRID,
            max_hold_grid=[24, 36, 48, 72, 96],
            part_label='PART_A_M30',
        )
        save_progress('PART_A', 'COMPLETE', f'winners={m30_results.get("winners", [])}')
    except Exception as e:
        print_flush(f'\n!!! PART A ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_A', 'ERROR', str(e))
        m30_results = {}

    # ══════════════════════════════════════════════════════════
    # PART B: H4 Strategies
    # ══════════════════════════════════════════════════════════
    try:
        print_flush(f'\n{"#"*80}\n# PART B: H4 Strategy Universe (15 strategies)\n{"#"*80}')
        h4_df = load_h4_with_indicators()
        h4_results = run_timeframe_pipeline(
            tf_name='h4',
            df=h4_df,
            strategies=H4_STRATEGIES,
            engine_cls=H4BacktestEngine,
            default_params={
                'sl_atr_mult': 3.0, 'tp_atr_mult': 6.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 30, 'cooldown_bars': 2, 'spread_cost': SPREAD,
            },
            sl_grid=SL_GRID,
            tp_grid=TP_GRID,
            trail_grid=TRAIL_GRID,
            max_hold_grid=[15, 20, 30, 45, 60],
            part_label='PART_B_H4',
        )
        save_progress('PART_B', 'COMPLETE', f'winners={h4_results.get("winners", [])}')
    except Exception as e:
        print_flush(f'\n!!! PART B ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_B', 'ERROR', str(e))
        h4_results = {}

    # ══════════════════════════════════════════════════════════
    # PART C: Cross-Timeframe Correlation & Portfolio
    # ══════════════════════════════════════════════════════════
    try:
        print_flush(f'\n{"#"*80}\n# PART C: Cross-Timeframe Portfolio Analysis\n{"#"*80}')

        all_winners = {}
        for tf_name, results, df_data, engine_cls, strats, def_params in [
            ('m30', m30_results, m30_df if 'm30_df' in dir() else None, M30BacktestEngine, M30_STRATEGIES, {
                'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 48, 'cooldown_bars': 4, 'spread_cost': SPREAD,
            }),
            ('h4', h4_results, h4_df if 'h4_df' in dir() else None, H4BacktestEngine, H4_STRATEGIES, {
                'sl_atr_mult': 3.0, 'tp_atr_mult': 6.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 30, 'cooldown_bars': 2, 'spread_cost': SPREAD,
            }),
        ]:
            if not results or df_data is None:
                continue
            strat_map = dict(strats)
            p10 = results.get('phase10', {})
            for sname, verdict_info in p10.items():
                if verdict_info.get('final_verdict') in ('STRONG_PASS', 'CONDITIONAL_PASS'):
                    sig_func = strat_map.get(sname)
                    if sig_func is None:
                        continue
                    best_p = verdict_info.get('best_params', {})
                    params = dict(def_params)
                    if best_p:
                        params.update({
                            'sl_atr_mult': best_p.get('sl', params.get('sl_atr_mult', 3.0)),
                            'tp_atr_mult': best_p.get('tp', params.get('tp_atr_mult', 6.0)),
                            'trailing_activate_atr': best_p.get('trail_act', 0.3),
                            'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                        })
                    engine = engine_cls(df_data, signal_funcs=[(sname, sig_func)], **params)
                    trades = engine.run()
                    st = [t for t in trades if t.strategy == sname]
                    daily = {}
                    for t in st:
                        day = pd.Timestamp(t.exit_time).date()
                        daily[day] = daily.get(day, 0) + t.pnl
                    all_winners[sname] = {'daily_pnl': pd.Series(daily), 'stats': calc_stats(st),
                                          'verdict': verdict_info.get('final_verdict')}

        if len(all_winners) >= 2:
            all_days = sorted(set().union(*[set(w['daily_pnl'].index) for w in all_winners.values()]))
            corr_df = pd.DataFrame({name: w['daily_pnl'].reindex(all_days, fill_value=0)
                                    for name, w in all_winners.items()})
            portfolio_corr = {}
            for s1, s2 in combinations(all_winners.keys(), 2):
                r = float(corr_df[s1].corr(corr_df[s2]))
                label = 'LOW' if abs(r) < 0.3 else ('MODERATE' if abs(r) < 0.6 else 'HIGH')
                portfolio_corr[f'{s1}_vs_{s2}'] = {'correlation': round(r, 3), 'label': label}
                print_flush(f'  {s1} vs {s2}: r={r:.3f} ({label})')

            combined_daily = corr_df.sum(axis=1)
            combined_pnls = combined_daily.values
            if len(combined_pnls) > 10:
                port_sharpe = float(combined_pnls.mean() / max(combined_pnls.std(ddof=1), 1e-9) * np.sqrt(252))
                port_pnl = float(combined_pnls.sum())
                cum = np.cumsum(combined_pnls)
                port_dd = float((np.maximum.accumulate(cum) - cum).max())
                print_flush(f'\n  Combined Portfolio: Sharpe={port_sharpe:.3f}  PnL=${port_pnl:.0f}  MaxDD=${port_dd:.0f}')
            else:
                port_sharpe, port_pnl, port_dd = 0, 0, 0

            save('part_c_portfolio', {
                'winners': {k: {'stats': v['stats'], 'verdict': v['verdict']} for k, v in all_winners.items()},
                'correlations': portfolio_corr,
                'combined_portfolio': {
                    'sharpe': round(port_sharpe, 3),
                    'total_pnl': round(port_pnl, 2),
                    'max_dd': round(port_dd, 2),
                },
            })
        else:
            print_flush('  Not enough winners for portfolio analysis')
            save('part_c_portfolio', {'note': 'insufficient winners'})

        save_progress('PART_C', 'COMPLETE')
    except Exception as e:
        print_flush(f'\n!!! PART C ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_C', 'ERROR', str(e))

    # ══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print_flush(f'\n{"#"*80}\n# FINAL MEGA SUMMARY\n{"#"*80}')

    final_summary = {'started': pd.Timestamp.now().isoformat()}
    for tf_name, results in [('m30', m30_results), ('h4', h4_results)]:
        if not results:
            continue
        p10 = results.get('phase10', {})
        tf_summary = {}
        for sname, info in p10.items():
            tf_summary[sname] = {
                'baseline_sharpe': info.get('baseline_sharpe'),
                'best_sharpe': info.get('best_sharpe'),
                'best_params': info.get('best_params'),
                'final_verdict': info.get('final_verdict'),
                'gates': info.get('gates'),
            }
            v = info.get('final_verdict', 'REJECT')
            marker = '***' if v == 'STRONG_PASS' else ('**' if v == 'CONDITIONAL_PASS' else '')
            print_flush(f'  {tf_name}/{sname:<20} Sharpe={info.get("best_sharpe", 0):.3f}  -> {v} {marker}')
        final_summary[tf_name] = tf_summary

    elapsed_total = time.time() - t_global
    final_summary['total_runtime_seconds'] = round(elapsed_total, 0)
    final_summary['total_runtime_hours'] = round(elapsed_total / 3600, 2)
    save('R230_final_summary', final_summary)

    print_flush(f'\n  Total runtime: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)')
    print_flush(f'  Finished: {pd.Timestamp.now()}')
    print_flush('=' * 80)


if __name__ == '__main__':
    main()
