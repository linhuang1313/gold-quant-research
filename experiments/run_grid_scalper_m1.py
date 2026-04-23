"""
HF Trend Scalper on M1 data — 高频趋势做市策略回测 (M1版)
=========================================================
使用真正的 M1 数据回测，匹配截图中的交易频率。

先用 2024-2026 做快速验证，然后再跑全量。
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.engine import TradeRecord
from backtest.stats import aggregate_daily_pnl


def load_m1_csv(path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load M1 CSV in our standard format."""
    print(f"  Loading {path}...", flush=True)
    df = pd.read_csv(path)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.000')
    df = df.set_index('Gmt time').sort_index()

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    print(f"  Loaded {len(df):,} M1 bars: {df.index[0]} -> {df.index[-1]}", flush=True)
    return df


@dataclass
class ScalpPos:
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    lots: float
    entry_idx: int
    best_pnl: float = 0.0


class HFScalperM1:
    """
    M1 高频 Scalper 引擎。

    策略逻辑 (从截图逆向):
    1. EMA fast/slow 交叉确定短期方向
    2. 顺方向入场，每笔目标 ~$1.5-2.5 (per 0.01 lot)
    3. 方向反转立即平仓
    4. 连续盈利后加大手数 0.01 → 0.02 → 0.03
    5. 亚盘时段交易 (UTC 0-13)
    """

    def __init__(
        self,
        ema_fast: int = 5,
        ema_slow: int = 13,
        tp_dollars: float = 2.0,
        sl_dollars: float = 3.5,
        scale_up_after: int = 3,
        lot_levels: tuple = (0.01, 0.02, 0.03),
        spread: float = 0.30,
        cooldown_bars: int = 2,
        session_filter: bool = True,
        session_hours: tuple = tuple(range(0, 13)),
        max_trades_per_day: int = 20,
        max_hold_bars: int = 60,  # 60 M1 bars = 1 hour max hold
        require_bar_alignment: bool = True,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.tp_dollars = tp_dollars
        self.sl_dollars = sl_dollars
        self.scale_up_after = scale_up_after
        self.lot_levels = lot_levels
        self.spread = spread
        self.cooldown_bars = cooldown_bars
        self.session_filter = session_filter
        self.session_hours = set(session_hours)
        self.max_trades_per_day = max_trades_per_day
        self.max_hold_bars = max_hold_bars
        self.require_bar_alignment = require_bar_alignment

    def run(self, df: pd.DataFrame) -> Tuple[List[TradeRecord], List[float]]:
        # Indicators
        close = df['Close'].values
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        times = df.index

        trades: List[TradeRecord] = []
        equity = [2000.0]
        position: Optional[ScalpPos] = None
        last_close_idx = -999
        consecutive_wins = 0
        lot_idx = 0
        daily_count: Dict[str, int] = {}

        warmup = max(self.ema_fast, self.ema_slow) + 5
        n = len(df)
        pct_step = max(1, n // 20)

        for i in range(warmup, n):
            if i % pct_step == 0:
                print(f"    {i*100//n}%...", end=" ", flush=True)

            ts = times[i]
            c = close[i]
            h = highs[i]
            lo = lows[i]
            o = opens[i]
            day_key = str(ts.date())

            ef = ema_f[i]
            es = ema_s[i]
            direction = 'BUY' if ef > es else 'SELL' if ef < es else None

            # ── Check exits ──
            if position is not None:
                pos = position
                lot_mult = pos.lots / 0.01

                if pos.direction == 'BUY':
                    best_px = h
                    worst_px = lo
                    pnl_best = (best_px - pos.entry_price - self.spread) * pos.lots * 100
                    pnl_worst = (worst_px - pos.entry_price - self.spread) * pos.lots * 100
                    pnl_close = (c - pos.entry_price - self.spread) * pos.lots * 100
                else:
                    best_px = lo
                    worst_px = h
                    pnl_best = (pos.entry_price - best_px - self.spread) * pos.lots * 100
                    pnl_worst = (pos.entry_price - worst_px - self.spread) * pos.lots * 100
                    pnl_close = (pos.entry_price - c - self.spread) * pos.lots * 100

                tp_level = self.tp_dollars * lot_mult
                sl_level = self.sl_dollars * lot_mult
                bars_held = i - pos.entry_idx

                # TP
                if pnl_best >= tp_level:
                    pnl = tp_level
                    tr = self._make_trade(pos, c, ts, "TP", i, pnl)
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    consecutive_wins += 1
                    if consecutive_wins >= self.scale_up_after:
                        lot_idx = min(lot_idx + 1, len(self.lot_levels) - 1)
                    continue

                # SL
                if pnl_worst <= -sl_level:
                    pnl = -sl_level
                    tr = self._make_trade(pos, c, ts, "SL", i, pnl)
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    consecutive_wins = 0
                    lot_idx = max(0, lot_idx - 1)
                    continue

                # Max hold timeout
                if bars_held >= self.max_hold_bars:
                    pnl = pnl_close
                    tr = self._make_trade(pos, c, ts, "Timeout", i, pnl)
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    if pnl > 0:
                        consecutive_wins += 1
                    else:
                        consecutive_wins = 0
                        lot_idx = max(0, lot_idx - 1)
                    continue

                # Direction reversal
                if direction is not None and direction != pos.direction:
                    pnl = pnl_close
                    tr = self._make_trade(pos, c, ts, "Reversal", i, pnl)
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    if pnl > 0:
                        consecutive_wins += 1
                    else:
                        consecutive_wins = 0
                        lot_idx = max(0, lot_idx - 1)
                    # allow re-entry below

            # ── New Entry ──
            if position is not None:
                continue

            if i - last_close_idx < self.cooldown_bars:
                continue

            if self.session_filter and ts.hour not in self.session_hours:
                continue

            if daily_count.get(day_key, 0) >= self.max_trades_per_day:
                continue

            if direction is None:
                continue

            # Bar alignment: bar should move in signal direction
            if self.require_bar_alignment:
                bar_move = c - o
                if direction == 'BUY' and bar_move < 0:
                    continue
                if direction == 'SELL' and bar_move > 0:
                    continue

            lots = self.lot_levels[lot_idx]
            entry_px = c + self.spread / 2 if direction == 'BUY' else c - self.spread / 2

            position = ScalpPos(
                direction=direction,
                entry_price=entry_px,
                entry_time=ts,
                lots=lots,
                entry_idx=i,
            )
            daily_count[day_key] = daily_count.get(day_key, 0) + 1

        # Close remaining
        if position is not None:
            pnl = self._calc_pnl(position, close[-1])
            tr = self._make_trade(position, close[-1], times[-1], "EOD", n - 1, pnl)
            trades.append(tr)
            equity.append(equity[-1] + pnl)

        print(" done!", flush=True)
        return trades, equity

    @staticmethod
    def _ema(data, period):
        out = np.empty_like(data)
        out[0] = data[0]
        alpha = 2.0 / (period + 1)
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    def _calc_pnl(self, pos, exit_px):
        if pos.direction == 'BUY':
            return (exit_px - pos.entry_price - self.spread) * pos.lots * 100
        return (pos.entry_price - exit_px - self.spread) * pos.lots * 100

    def _make_trade(self, pos, exit_px, exit_time, reason, exit_idx, pnl=None):
        if pnl is None:
            pnl = self._calc_pnl(pos, exit_px)
        return TradeRecord(
            strategy="HFScalper",
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_px,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            lots=pos.lots,
            pnl=pnl,
            exit_reason=reason,
            bars_held=exit_idx - pos.entry_idx,
        )


def print_report(trades, equity, label):
    if not trades:
        print(f"  {label}: No trades")
        return {}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    wr = len(wins) / len(pnls) * 100
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0

    daily = aggregate_daily_pnl(trades)
    sharpe = 0
    if len(daily) > 1 and np.std(daily, ddof=1) > 0:
        sharpe = np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252)

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = peak - e
        if dd > max_dd: max_dd = dd

    by_reason = {}
    for t in trades:
        r = t.exit_reason
        if r not in by_reason:
            by_reason[r] = {'n': 0, 'pnl': 0}
        by_reason[r]['n'] += 1
        by_reason[r]['pnl'] += t.pnl

    by_lots = {}
    for t in trades:
        k = f"{t.lots:.2f}"
        if k not in by_lots:
            by_lots[k] = {'n': 0, 'pnl': 0, 'wins': 0}
        by_lots[k]['n'] += 1
        by_lots[k]['pnl'] += t.pnl
        if t.pnl > 0:
            by_lots[k]['wins'] += 1

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades):,}")
    print(f"  Total PnL: ${total:,.2f}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Win Rate: {wr:.1f}%")
    if avg_l > 0:
        print(f"  Avg Win: ${avg_w:.2f}  |  Avg Loss: ${avg_l:.2f}  |  RR: {avg_w/avg_l:.2f}")
    else:
        print(f"  Avg Win: ${avg_w:.2f}  |  Avg Loss: $0.00")
    print(f"  Max DD: ${max_dd:,.2f}")
    print(f"  Avg bars held: {np.mean([t.bars_held for t in trades]):.1f} (M1 bars)")

    print(f"\n  Exit reasons:")
    for r, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"    {r:>12}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}")

    print(f"\n  By lot size:")
    for k in sorted(by_lots.keys()):
        v = by_lots[k]
        print(f"    {k}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}, WR={v['wins']/max(1,v['n'])*100:.1f}%")

    year_pnl: Dict[int, List] = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        if y not in year_pnl:
            year_pnl[y] = [0, 0.0]
        year_pnl[y][0] += 1
        year_pnl[y][1] += t.pnl
    print(f"\n  Year-by-Year:")
    for y in sorted(year_pnl.keys()):
        n_y, p = year_pnl[y]
        print(f"    {y}: N={n_y:>6}, PnL=${p:>10,.2f}")

    return {'n': len(trades), 'pnl': total, 'sharpe': sharpe, 'wr': wr, 'max_dd': max_dd}


def main():
    t0 = time.time()
    out = Path("results/grid_scalper_m1")
    out.mkdir(parents=True, exist_ok=True)

    print("# HF Trend Scalper — M1 Backtest")
    print(f"# {pd.Timestamp.now()}\n")

    m1_path = "data/download/xauusd-m1-bid-2015-01-01-2026-04-10.csv"

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Quick test on 2024-2026 (faster iteration)
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Quick test (2024-2026)")
    print("=" * 60)

    df_quick = load_m1_csv(m1_path, start_date="2024-01-01")
    print()

    # A. Baseline
    print("-" * 40)
    print("A. Baseline: EMA(5,13), TP=$2, SL=$3.5")
    print("-" * 40)
    eng = HFScalperM1(
        ema_fast=5, ema_slow=13,
        tp_dollars=2.0, sl_dollars=3.5,
        spread=0.30,
        session_filter=True,
        session_hours=tuple(range(0, 13)),
        cooldown_bars=2,
        max_trades_per_day=20,
        max_hold_bars=60,
    )
    tr_a, eq_a = eng.run(df_quick)
    print_report(tr_a, eq_a, "Baseline M1 (2024-2026)")

    # B. Tighter TP
    print("\n" + "-" * 40)
    print("B. Tight TP: TP=$1.5, SL=$2.5")
    print("-" * 40)
    eng_b = HFScalperM1(
        ema_fast=5, ema_slow=13,
        tp_dollars=1.5, sl_dollars=2.5,
        spread=0.30,
        session_filter=True,
        cooldown_bars=1,
        max_trades_per_day=25,
        max_hold_bars=30,
    )
    tr_b, eq_b = eng_b.run(df_quick)
    print_report(tr_b, eq_b, "Tight TP=$1.5, SL=$2.5")

    # C. Wider TP/SL
    print("\n" + "-" * 40)
    print("C. Wider: TP=$3, SL=$5")
    print("-" * 40)
    eng_c = HFScalperM1(
        ema_fast=5, ema_slow=13,
        tp_dollars=3.0, sl_dollars=5.0,
        spread=0.30,
        session_filter=True,
        cooldown_bars=2,
        max_trades_per_day=20,
        max_hold_bars=90,
    )
    tr_c, eq_c = eng_c.run(df_quick)
    print_report(tr_c, eq_c, "Wider TP=$3, SL=$5")

    # D. EMA(3,8)
    print("\n" + "-" * 40)
    print("D. Faster EMA(3,8), TP=$1.5, SL=$3")
    print("-" * 40)
    eng_d = HFScalperM1(
        ema_fast=3, ema_slow=8,
        tp_dollars=1.5, sl_dollars=3.0,
        spread=0.30,
        session_filter=True,
        cooldown_bars=1,
        max_trades_per_day=25,
        max_hold_bars=30,
    )
    tr_d, eq_d = eng_d.run(df_quick)
    print_report(tr_d, eq_d, "EMA(3,8), TP=$1.5, SL=$3")

    # E. No bar alignment
    print("\n" + "-" * 40)
    print("E. No bar alignment requirement")
    print("-" * 40)
    eng_e = HFScalperM1(
        ema_fast=5, ema_slow=13,
        tp_dollars=2.0, sl_dollars=3.5,
        spread=0.30,
        session_filter=True,
        cooldown_bars=2,
        max_trades_per_day=20,
        max_hold_bars=60,
        require_bar_alignment=False,
    )
    tr_e, eq_e = eng_e.run(df_quick)
    print_report(tr_e, eq_e, "No bar alignment")

    # F. Full day (no session filter)
    print("\n" + "-" * 40)
    print("F. Full day trading")
    print("-" * 40)
    eng_f = HFScalperM1(
        ema_fast=5, ema_slow=13,
        tp_dollars=2.0, sl_dollars=3.5,
        spread=0.30,
        session_filter=False,
        cooldown_bars=2,
        max_trades_per_day=30,
        max_hold_bars=60,
    )
    tr_f, eq_f = eng_f.run(df_quick)
    print_report(tr_f, eq_f, "Full Day")

    # G. Asymmetric — Big TP, tight SL (trend capture mode)
    print("\n" + "-" * 40)
    print("G. Trend Capture: TP=$5, SL=$2")
    print("-" * 40)
    eng_g = HFScalperM1(
        ema_fast=5, ema_slow=13,
        tp_dollars=5.0, sl_dollars=2.0,
        spread=0.30,
        session_filter=True,
        cooldown_bars=2,
        max_trades_per_day=20,
        max_hold_bars=120,
    )
    tr_g, eq_g = eng_g.run(df_quick)
    print_report(tr_g, eq_g, "Trend Capture TP=$5, SL=$2")

    # H. TP/SL scan
    print("\n" + "=" * 60)
    print("H. TP/SL 参数扫描 (2024-2026)")
    print("=" * 60)

    best_sharpe = -999
    best_cfg = ""
    scan_results = []

    for tp in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for sl in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
            if sl < tp:
                continue
            eng_h = HFScalperM1(
                ema_fast=5, ema_slow=13,
                tp_dollars=tp, sl_dollars=sl,
                spread=0.30,
                session_filter=True,
                cooldown_bars=2,
                max_trades_per_day=20,
                max_hold_bars=60,
            )
            tr_h, eq_h = eng_h.run(df_quick)
            if tr_h:
                pnls_h = [t.pnl for t in tr_h]
                total_h = sum(pnls_h)
                wr_h = sum(1 for p in pnls_h if p > 0) / len(pnls_h) * 100
                daily_h = aggregate_daily_pnl(tr_h)
                sh = 0
                if len(daily_h) > 1 and np.std(daily_h, ddof=1) > 0:
                    sh = np.mean(daily_h) / np.std(daily_h, ddof=1) * np.sqrt(252)
                scan_results.append((tp, sl, len(tr_h), total_h, sh, wr_h))
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_cfg = f"TP=${tp}, SL=${sl}"

    print("\n  TP/SL Scan Results (sorted by Sharpe):")
    print(f"  {'TP':>6} {'SL':>6} {'Trades':>8} {'PnL':>12} {'Sharpe':>8} {'WR':>6}")
    print(f"  {'-'*52}")
    for tp, sl, n_t, pnl, sh, wr in sorted(scan_results, key=lambda x: -x[4]):
        marker = " <<<" if f"TP=${tp}, SL=${sl}" == best_cfg else ""
        print(f"  ${tp:>5.1f} ${sl:>5.1f} {n_t:>8,} ${pnl:>11,.2f} {sh:>8.2f} {wr:>5.1f}%{marker}")

    print(f"\n  >>> Best: {best_cfg} (Sharpe={best_sharpe:.2f})")

    elapsed = time.time() - t0
    print(f"\n\nTotal: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
