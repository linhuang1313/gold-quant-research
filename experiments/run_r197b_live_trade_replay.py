#!/usr/bin/env python3
"""
R197b — Live Trade Replay with Different Trend Score Thresholds
================================================================
Takes actual live trades from gold_trade_log.json and replays them
against historical H1 data to compute what the trend_score was at
the time of each entry. Then shows how many trades would have been
filtered at each threshold, and what the PnL impact would be.

This is NOT a backtest — it uses your real trades and their real PnLs.
It only checks: "would trend_score have blocked this entry?"
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv, prepare_indicators_custom, H1_CSV_PATH

OUTPUT_DIR = Path("results/r197b_live_replay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 0.561, 0.01)]
TREND_GATED_STRATEGIES = {'keltner', 'macd', 'm15_rsi', 'orb', 'm5_rsi'}

# ═══════════════ Trend Score Computation ═══════════════
def compute_trend_score(h1_df, target_time):
    """Compute the trend_score at a specific time using H1 bars up to that point.
    Replicates the IntradayTrendMeter logic exactly."""
    target_ts = pd.Timestamp(target_time)
    if target_ts.tzinfo is None:
        target_ts = target_ts.tz_localize('UTC')
    
    today = target_ts.normalize()
    today_bars = h1_df[(h1_df.index >= today) & (h1_df.index <= target_ts)]
    
    if len(today_bars) < 2:
        return 0.5, 'neutral'
    
    latest = today_bars.iloc[-1]
    
    # 1. ADX component (weight 0.30)
    adx = float(latest.get('ADX', 20))
    if np.isnan(adx):
        adx = 20
    adx_score = min(adx / 40.0, 1.0)
    
    # 2. KC breakout ratio (weight 0.25)
    kc_upper = today_bars.get('KC_upper')
    kc_lower = today_bars.get('KC_lower')
    if kc_upper is not None and kc_lower is not None:
        breaks = ((today_bars['Close'] > kc_upper) | (today_bars['Close'] < kc_lower)).sum()
        kc_score = min(float(breaks) / len(today_bars), 1.0)
    else:
        kc_score = 0.0
    
    # 3. EMA alignment consistency (weight 0.25)
    ema9 = today_bars.get('EMA9')
    ema21 = today_bars.get('EMA21')
    ema100 = today_bars.get('EMA100')
    if ema9 is not None and ema21 is not None and ema100 is not None:
        bullish = (ema9 > ema21) & (ema21 > ema100)
        bearish = (ema9 < ema21) & (ema21 < ema100)
        aligned = (bullish | bearish).sum()
        ema_score = float(aligned) / len(today_bars)
    else:
        ema_score = 0.0
    
    # 4. Trend intensity (weight 0.20)
    day_open = float(today_bars.iloc[0]['Open'])
    day_close = float(latest['Close'])
    day_high = float(today_bars['High'].max())
    day_low = float(today_bars['Low'].min())
    day_range = day_high - day_low
    ti = abs(day_close - day_open) / day_range if day_range > 0.01 else 0.0
    
    score = round(0.30 * adx_score + 0.25 * kc_score + 0.25 * ema_score + 0.20 * ti, 3)
    
    if score >= 0.60:
        regime = 'trending'
    elif score >= 0.50:
        regime = 'neutral'
    else:
        regime = 'choppy'
    
    return score, regime


# ═══════════════ Load Data ═══════════════
print(f"{'='*100}")
print(f"  R197b — Live Trade Replay with Different Trend Score Thresholds")
print(f"{'='*100}")

# Load H1 data
print(f"\nLoading H1 data...")
h1 = load_csv(str(H1_CSV_PATH))
h1 = prepare_indicators_custom(h1)
print(f"  H1: {len(h1)} bars, {h1.index[0]} to {h1.index[-1]}")

# Load live trade log
_candidates = [
    Path(r"c:\Users\hlin2\gold-quant-trading\data\gold_trade_log.json"),
    Path("/root/gold-quant-research/data/gold_trade_log.json"),
    Path("/root/gold-quant-trading/data/gold_trade_log.json"),
    Path("data/gold_trade_log.json"),
]
TRADE_LOG_PATH = next((p for p in _candidates if p.exists()), _candidates[0])

print(f"Loading trade log from {TRADE_LOG_PATH}...")
with open(TRADE_LOG_PATH) as f:
    raw_trades = json.load(f)

# Pair OPEN/CLOSE into completed trades
print(f"  Raw entries: {len(raw_trades)}")

open_tickets = {}
completed_trades = []
for entry in raw_trades:
    action = entry.get('action', '')
    if action == 'OPEN':
        strategy = entry.get('strategy', '?')
        direction = entry.get('direction', '?')
        t = entry.get('time', '')
        key = f"{strategy}_{direction}_{t[:16]}"
        open_tickets[key] = entry
    elif action == 'CLOSE' and 'profit' in entry:
        strategy = entry.get('strategy', '?')
        direction = entry.get('direction', '?')
        ticket = entry.get('ticket', 0)
        
        matched_open = None
        for k, v in list(open_tickets.items()):
            if v.get('strategy') == strategy:
                matched_open = v
                del open_tickets[k]
                break
        
        completed_trades.append({
            'strategy': strategy,
            'direction': direction,
            'entry_time': matched_open['time'] if matched_open else entry.get('time', ''),
            'exit_time': entry.get('time', ''),
            'profit': float(entry.get('profit', 0)),
            'lots': float(entry.get('lots', 0)),
            'entry_price': float(matched_open.get('price', 0)) if matched_open else float(entry.get('open_price', 0)),
            'exit_price': float(entry.get('close_price', 0)),
            'reason': entry.get('reason', ''),
        })

print(f"  Completed trades: {len(completed_trades)}")

# Compute trend_score for each trade's entry time
print(f"\nComputing trend_score for each trade entry...")
for trade in completed_trades:
    entry_time = trade['entry_time']
    score, regime = compute_trend_score(h1, entry_time)
    trade['trend_score'] = score
    trade['trend_regime'] = regime

# ═══════════════ Analysis ═══════════════
print(f"\n{'='*100}")
print(f"  ANALYSIS: Impact of Different Trend Score Thresholds on Live Trades")
print(f"{'='*100}")

# Overall stats
all_pnl = sum(t['profit'] for t in completed_trades)
all_n = len(completed_trades)
all_wins = sum(1 for t in completed_trades if t['profit'] > 0)
print(f"\n  Total live trades: {all_n}")
print(f"  Total PnL: ${all_pnl:.2f}")
print(f"  Win rate: {all_wins/all_n*100:.1f}%")

# By strategy
from collections import Counter, defaultdict
by_strat = defaultdict(list)
for t in completed_trades:
    by_strat[t['strategy']].append(t)

print(f"\n  By strategy:")
for s, trades in sorted(by_strat.items(), key=lambda x: -len(x[1])):
    pnl = sum(t['profit'] for t in trades)
    n = len(trades)
    wr = sum(1 for t in trades if t['profit'] > 0) / n * 100 if n > 0 else 0
    gated = "YES (trend_score gated)" if s in TREND_GATED_STRATEGIES else "NO"
    print(f"    {s:<15} N={n:>4}  PnL=${pnl:>8.2f}  WR={wr:>5.1f}%  Gated={gated}")

# Score distribution
print(f"\n  Trend score distribution at entry time:")
scores = [t['trend_score'] for t in completed_trades]
for bucket_lo in np.arange(0, 1.0, 0.1):
    bucket_hi = bucket_lo + 0.1
    in_bucket = [t for t in completed_trades if bucket_lo <= t['trend_score'] < bucket_hi]
    if in_bucket:
        pnl = sum(t['profit'] for t in in_bucket)
        n = len(in_bucket)
        wr = sum(1 for t in in_bucket if t['profit'] > 0) / n * 100
        avg = pnl / n
        print(f"    score [{bucket_lo:.1f}-{bucket_hi:.1f}): N={n:>4}  PnL=${pnl:>8.2f}  WR={wr:>5.1f}%  Avg=${avg:>6.2f}")

# ═══════════════ Threshold Sweep on Live Trades ═══════════════
print(f"\n{'─'*100}")
print(f"  Threshold Sweep: What happens at each choppy_threshold?")
print(f"  (Only affects trend-gated strategies: {TREND_GATED_STRATEGIES})")
print(f"{'─'*100}")

header = f"  {'Threshold':<10} {'Kept':>5} {'Filtered':>8} {'Kept PnL':>10} {'Filt PnL':>10} {'Kept WR':>7} {'Filt WR':>7} {'Avg Kept':>9} {'Avg Filt':>9} {'Better?':>8}"
print(header)
print(f"  {'-'*95}")

sweep_results = {}
for thr in THRESHOLDS:
    kept = []
    filtered = []
    for t in completed_trades:
        if t['strategy'] in TREND_GATED_STRATEGIES and t['trend_score'] < thr:
            filtered.append(t)
        else:
            kept.append(t)
    
    kept_pnl = sum(t['profit'] for t in kept)
    filt_pnl = sum(t['profit'] for t in filtered)
    kept_n = len(kept)
    filt_n = len(filtered)
    kept_wr = sum(1 for t in kept if t['profit'] > 0) / kept_n * 100 if kept_n > 0 else 0
    filt_wr = sum(1 for t in filtered if t['profit'] > 0) / filt_n * 100 if filt_n > 0 else 0
    avg_kept = kept_pnl / kept_n if kept_n > 0 else 0
    avg_filt = filt_pnl / filt_n if filt_n > 0 else 0
    better = "YES" if filt_pnl < 0 else "no"
    
    sweep_results[str(thr)] = {
        'kept_n': kept_n, 'filt_n': filt_n, 'kept_pnl': round(kept_pnl, 2),
        'filt_pnl': round(filt_pnl, 2), 'kept_wr': round(kept_wr, 1),
        'filt_wr': round(filt_wr, 1), 'avg_kept': round(avg_kept, 2),
        'avg_filt': round(avg_filt, 2),
    }
    
    if filt_n > 0:
        print(f"  {thr:<10.2f} {kept_n:>5} {filt_n:>8} ${kept_pnl:>9.2f} ${filt_pnl:>9.2f} {kept_wr:>6.1f}% {filt_wr:>6.1f}% ${avg_kept:>8.2f} ${avg_filt:>8.2f} {better:>8}")

# ═══════════════ Detailed Trade-by-Trade for Current Threshold (0.50) ═══════════════
print(f"\n{'─'*100}")
print(f"  Detail: Trades FILTERED by current threshold (0.50)")
print(f"{'─'*100}")

filtered_050 = [t for t in completed_trades if t['strategy'] in TREND_GATED_STRATEGIES and t['trend_score'] < 0.50]
if filtered_050:
    print(f"  {'Time':<20} {'Strategy':<12} {'Dir':<5} {'Score':>6} {'Regime':<10} {'PnL':>8} {'Result':<6}")
    print(f"  {'-'*75}")
    for t in sorted(filtered_050, key=lambda x: x['entry_time']):
        result = "WIN" if t['profit'] > 0 else "LOSS"
        print(f"  {t['entry_time'][:19]:<20} {t['strategy']:<12} {t['direction']:<5} {t['trend_score']:>6.3f} {t['trend_regime']:<10} ${t['profit']:>7.2f} {result:<6}")
    total_filt = sum(t['profit'] for t in filtered_050)
    print(f"\n  Total filtered PnL: ${total_filt:.2f} ({len(filtered_050)} trades)")
    print(f"  -> {'GOOD filter (removed losses)' if total_filt < 0 else 'BAD filter (removed profits)'}")
else:
    print(f"  No trades were filtered at threshold 0.50")

# ═══════════════ Keltner-Specific Analysis ═══════════════
print(f"\n{'─'*100}")
print(f"  Keltner-Only Analysis (primary strategy)")
print(f"{'─'*100}")

keltner_trades = [t for t in completed_trades if t['strategy'] == 'keltner']
if keltner_trades:
    print(f"\n  {'Threshold':<10} {'Kept':>5} {'Filtered':>8} {'Kept PnL':>10} {'Filt PnL':>10} {'Kept Avg':>9} {'Filt Avg':>9}")
    print(f"  {'-'*65}")
    for thr in [0.0, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55]:
        kept = [t for t in keltner_trades if t['trend_score'] >= thr]
        filt = [t for t in keltner_trades if t['trend_score'] < thr]
        kp = sum(t['profit'] for t in kept)
        fp = sum(t['profit'] for t in filt)
        ka = kp / len(kept) if kept else 0
        fa = fp / len(filt) if filt else 0
        print(f"  {thr:<10.2f} {len(kept):>5} {len(filt):>8} ${kp:>9.2f} ${fp:>9.2f} ${ka:>8.2f} ${fa:>8.2f}")

# Save results
with open(OUTPUT_DIR / "r197b_results.json", 'w') as f:
    json.dump({
        'sweep': sweep_results,
        'trades': [{k: v for k, v in t.items()} for t in completed_trades],
    }, f, indent=2, default=str)

total_time = time.time() - t0
print(f"\n  Total runtime: {total_time:.0f}s")
print(f"{'='*100}")
