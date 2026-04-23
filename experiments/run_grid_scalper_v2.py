"""
HF Trend Scalper v2 — 高频趋势做市混合策略回测
================================================
基于完整交易记录逆向工程:
  - 主模式: 顺短期EMA方向快速scalping, 目标$1.5-2.5/笔 (0.01手)
  - 加码模式: 方向确认后手数递增 0.01→0.02→0.03
  - 趋势持有: 强趋势时持有至大波段，单笔可赚$20-35
  - 止损: 不硬止损，反向时快速平仓亏损并切换方向
  - 亚盘交易为主 (UTC 01:00-06:00)

关键参数（从截图提取）:
  - scalp TP: ~$1.5-2.5 (per 0.01 lot)
  - 方向判断: 短期EMA + 价格动量
  - 加码条件: 连续盈利 → 加大手数
  - 平仓: 利润达标 or 方向反转
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.runner import load_csv
from backtest.engine import TradeRecord
from backtest.stats import aggregate_daily_pnl


@dataclass
class ScalpPosition:
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    lots: float
    entry_idx: int
    # trailing
    best_pnl: float = 0.0


class HFTrendScalper:
    """
    高频趋势 Scalper 引擎。

    核心逻辑:
    1. 用短期 EMA(3) vs EMA(8) 判断方向
    2. 价格在方向上移动时入场 (scalp)
    3. 目标利润 tp_dollars 后平仓
    4. 方向反转时立即平仓亏损单
    5. 连续盈利后加大手数 (0.01 → 0.02 → 0.03)
    6. 强趋势信号时持有至更大利润

    M15 数据限制: 我们用 M15 的 Open/High/Low/Close 模拟。
    每根 M15 bar 最多触发一次入场。
    """

    def __init__(
        self,
        # 方向判断
        ema_fast: int = 3,
        ema_slow: int = 8,
        ema_trend: int = 50,       # 大趋势EMA，用于加码判断
        # Scalp 参数
        tp_dollars: float = 2.0,   # 每0.01手的止盈美元 ($2 = 2点)
        sl_dollars: float = 3.0,   # 每0.01手的止损美元 ($3 = 3点)
        # 加码
        scale_up_after: int = 3,   # 连续N笔盈利后手数翻倍
        lot_levels: tuple = (0.01, 0.02, 0.03),
        # 趋势持有
        trend_hold_atr_mult: float = 1.5,  # 趋势模式下TP为 ATR*mult
        trend_trigger_slope: float = 0.3,  # EMA斜率超过此值进入趋势模式
        # 交易管理
        spread: float = 0.30,
        max_open: int = 1,         # 同时最多持仓数
        cooldown_bars: int = 1,    # 两笔之间冷却
        # 时段
        session_filter: bool = True,
        session_hours: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        max_trades_per_day: int = 20,
        # ATR
        atr_period: int = 14,
        min_atr: float = 0.5,
        max_atr: float = 20.0,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        self.tp_dollars = tp_dollars
        self.sl_dollars = sl_dollars
        self.scale_up_after = scale_up_after
        self.lot_levels = lot_levels
        self.trend_hold_atr_mult = trend_hold_atr_mult
        self.trend_trigger_slope = trend_trigger_slope
        self.spread = spread
        self.max_open = max_open
        self.cooldown_bars = cooldown_bars
        self.session_filter = session_filter
        self.session_hours = set(session_hours)
        self.max_trades_per_day = max_trades_per_day
        self.atr_period = atr_period
        self.min_atr = min_atr
        self.max_atr = max_atr

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema_f'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_s'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_t'] = df['Close'].ewm(span=self.ema_trend, adjust=False).mean()

        # EMA slope (per bar)
        df['ema_f_slope'] = df['ema_f'].diff()
        df['ema_s_slope'] = df['ema_s'].diff()

        # ATR
        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': (df['High'] - df['Close'].shift(1)).abs(),
            'lc': (df['Low'] - df['Close'].shift(1)).abs(),
        }).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()

        # Direction signal
        df['ema_diff'] = df['ema_f'] - df['ema_s']
        df['trend_up'] = (df['ema_f'] > df['ema_s']) & (df['Close'] > df['ema_t'])
        df['trend_dn'] = (df['ema_f'] < df['ema_s']) & (df['Close'] < df['ema_t'])

        # Momentum (close vs open of current bar)
        df['bar_move'] = df['Close'] - df['Open']

        return df

    def run(self, df: pd.DataFrame) -> Tuple[List[TradeRecord], List[float]]:
        df = self._calc_indicators(df)

        trades: List[TradeRecord] = []
        equity = [2000.0]
        position: Optional[ScalpPosition] = None
        last_close_idx = -999
        consecutive_wins = 0
        current_lot_idx = 0
        daily_count: Dict[str, int] = {}

        warmup = max(self.ema_fast, self.ema_slow, self.ema_trend, self.atr_period) + 5
        n = len(df)
        pct_step = max(1, n // 10)

        for i in range(warmup, n):
            if i % pct_step == 0:
                print(f"    {i*100//n}%...", end="  ", flush=True)

            row = df.iloc[i]
            ts = df.index[i]
            close = row['Close']
            high = row['High']
            low = row['Low']
            atr = row['atr']
            day_key = str(ts.date())

            if pd.isna(atr) or atr < self.min_atr or atr > self.max_atr:
                if position is not None:
                    pnl = self._calc_pnl(position, close)
                    trades.append(self._close(position, close, ts, "ATR_Filter", i))
                    equity.append(equity[-1] + pnl)
                    position = None
                continue

            # ── Check exits on existing position ──
            if position is not None:
                pos = position
                lots = pos.lots
                lot_mult = lots / 0.01

                if pos.direction == 'BUY':
                    pnl_at_high = (high - pos.entry_price - self.spread) * lots * 100
                    pnl_at_low = (low - pos.entry_price - self.spread) * lots * 100
                    pnl_at_close = (close - pos.entry_price - self.spread) * lots * 100
                else:
                    pnl_at_high = (pos.entry_price - high - self.spread) * lots * 100
                    pnl_at_low = (pos.entry_price - low - self.spread) * lots * 100
                    pnl_at_close = (pos.entry_price - close - self.spread) * lots * 100

                pos.best_pnl = max(pos.best_pnl, pnl_at_high if pos.direction == 'BUY' else pnl_at_low)

                # TP hit
                tp_level = self.tp_dollars * lot_mult
                # In trend mode, use wider TP
                ema_diff = row['ema_diff']
                in_trend = abs(row.get('ema_f_slope', 0)) > self.trend_trigger_slope
                if in_trend:
                    tp_level = atr * self.trend_hold_atr_mult * lots * 100

                best_intrabar = pnl_at_high if pos.direction == 'BUY' else pnl_at_low
                if best_intrabar >= tp_level:
                    exit_px = pos.entry_price + (tp_level / (lots * 100) + self.spread) if pos.direction == 'BUY' \
                        else pos.entry_price - (tp_level / (lots * 100) + self.spread)
                    pnl = tp_level
                    tr = self._close(position, exit_px, ts, "TP", i)
                    tr.pnl = pnl
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    consecutive_wins += 1
                    if consecutive_wins >= self.scale_up_after:
                        current_lot_idx = min(current_lot_idx + 1, len(self.lot_levels) - 1)
                    continue

                # SL hit
                sl_level = self.sl_dollars * lot_mult
                worst_intrabar = pnl_at_low if pos.direction == 'BUY' else pnl_at_high
                if worst_intrabar <= -sl_level:
                    pnl = -sl_level
                    exit_px = pos.entry_price - (sl_level / (lots * 100) - self.spread) if pos.direction == 'BUY' \
                        else pos.entry_price + (sl_level / (lots * 100) - self.spread)
                    tr = self._close(position, exit_px, ts, "SL", i)
                    tr.pnl = pnl
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    consecutive_wins = 0
                    current_lot_idx = max(0, current_lot_idx - 1)
                    continue

                # Direction reversal → close
                direction = self._get_direction(row)
                if direction is not None and direction != pos.direction:
                    pnl = pnl_at_close
                    tr = self._close(position, close, ts, "Reversal", i)
                    tr.pnl = pnl
                    trades.append(tr)
                    equity.append(equity[-1] + pnl)
                    position = None
                    last_close_idx = i
                    if pnl > 0:
                        consecutive_wins += 1
                    else:
                        consecutive_wins = 0
                        current_lot_idx = max(0, current_lot_idx - 1)
                    # Don't continue — allow immediate re-entry in new direction

            # ── New entry ──
            if position is not None:
                continue

            if i - last_close_idx < self.cooldown_bars:
                continue

            if self.session_filter and ts.hour not in self.session_hours:
                continue

            if daily_count.get(day_key, 0) >= self.max_trades_per_day:
                continue

            direction = self._get_direction(row)
            if direction is None:
                continue

            # Require bar move aligned with direction
            bar_move = row['bar_move']
            if direction == 'BUY' and bar_move < 0:
                continue
            if direction == 'SELL' and bar_move > 0:
                continue

            lots = self.lot_levels[current_lot_idx]

            if direction == 'BUY':
                entry_px = close + self.spread
            else:
                entry_px = close - self.spread

            position = ScalpPosition(
                direction=direction,
                entry_price=entry_px,
                entry_time=ts,
                lots=lots,
                entry_idx=i,
            )
            daily_count[day_key] = daily_count.get(day_key, 0) + 1

        # Close remaining
        if position is not None:
            pnl = self._calc_pnl(position, df.iloc[-1]['Close'])
            trades.append(self._close(position, df.iloc[-1]['Close'], df.index[-1], "EOD", n-1))
            trades[-1].pnl = pnl
            equity.append(equity[-1] + pnl)

        print(" done!", flush=True)
        return trades, equity

    def _get_direction(self, row) -> Optional[str]:
        ema_diff = row.get('ema_diff', 0)
        if pd.isna(ema_diff):
            return None
        if ema_diff > 0:
            return 'BUY'
        elif ema_diff < 0:
            return 'SELL'
        return None

    def _calc_pnl(self, pos: ScalpPosition, exit_price: float) -> float:
        if pos.direction == 'BUY':
            return (exit_price - pos.entry_price - self.spread) * pos.lots * 100
        else:
            return (pos.entry_price - exit_price - self.spread) * pos.lots * 100

    def _close(self, pos: ScalpPosition, exit_price: float, exit_time, reason: str, exit_idx: int) -> TradeRecord:
        pnl = self._calc_pnl(pos, exit_price)
        return TradeRecord(
            strategy="HFScalper",
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            lots=pos.lots,
            pnl=pnl,
            exit_reason=reason,
            bars_held=exit_idx - pos.entry_idx,
        )


def print_report(trades: List[TradeRecord], equity: List[float], label: str):
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
    print(f"  Max DD: ${max_dd:,.2f}")
    print(f"  Avg bars held: {np.mean([t.bars_held for t in trades]):.1f}")

    print(f"\n  Exit reasons:")
    for r, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]['pnl'])):
        print(f"    {r:>12}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}")

    print(f"\n  By lot size:")
    for k in sorted(by_lots.keys()):
        v = by_lots[k]
        print(f"    {k}: N={v['n']:>6}, PnL=${v['pnl']:>10,.2f}, WR={v['wins']/max(1,v['n'])*100:.1f}%")

    year_pnl: Dict[int, Tuple[int, float]] = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        if y not in year_pnl:
            year_pnl[y] = [0, 0.0]
        year_pnl[y][0] += 1
        year_pnl[y][1] += t.pnl
    print(f"\n  Year-by-Year:")
    for y in sorted(year_pnl.keys()):
        n, p = year_pnl[y]
        print(f"    {y}: N={n:>6}, PnL=${p:>10,.2f}")

    return {'n': len(trades), 'pnl': total, 'sharpe': sharpe, 'wr': wr, 'max_dd': max_dd}


def main():
    t0 = time.time()
    out = Path("results/grid_scalper_v2")
    out.mkdir(parents=True, exist_ok=True)

    print("# HF Trend Scalper v2 — Backtest")
    print(f"# {pd.Timestamp.now()}\n")

    m15_path = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
    if not m15_path.exists():
        m15_path = Path("data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv")
    df = load_csv(str(m15_path))
    print(f"  M15: {len(df)} bars, {df.index[0]} -> {df.index[-1]}\n")

    # ═══════════════════════════════════════════════════════════
    # A. Baseline — 从截图逆向工程
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("A. Baseline — 截图逆向工程参数")
    print("=" * 60)
    eng = HFTrendScalper(
        ema_fast=3, ema_slow=8, ema_trend=50,
        tp_dollars=2.0, sl_dollars=3.0,
        scale_up_after=3, lot_levels=(0.01, 0.02, 0.03),
        trend_hold_atr_mult=1.5, trend_trigger_slope=0.3,
        spread=0.30, max_open=1, cooldown_bars=1,
        session_filter=True,
        session_hours=tuple(range(0, 13)),
        max_trades_per_day=20,
    )
    tr, eq = eng.run(df)
    print_report(tr, eq, "Baseline: EMA(3,8), TP=$2, SL=$3, 亚盘")

    # ═══════════════════════════════════════════════════════════
    # B. 全天交易
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. 全天交易")
    print("=" * 60)
    eng_b = HFTrendScalper(
        ema_fast=3, ema_slow=8, ema_trend=50,
        tp_dollars=2.0, sl_dollars=3.0,
        spread=0.30, session_filter=False,
        max_trades_per_day=30,
    )
    tr_b, eq_b = eng_b.run(df)
    print_report(tr_b, eq_b, "Full Day")

    # ═══════════════════════════════════════════════════════════
    # C. TP/SL 参数扫描
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C. TP/SL 扫描")
    print("=" * 60)

    configs = [
        (1.5, 2.0, "TP=$1.5, SL=$2"),
        (2.0, 3.0, "TP=$2.0, SL=$3 (baseline)"),
        (2.5, 3.5, "TP=$2.5, SL=$3.5"),
        (3.0, 4.0, "TP=$3.0, SL=$4"),
        (3.0, 5.0, "TP=$3.0, SL=$5"),
        (4.0, 5.0, "TP=$4.0, SL=$5"),
        (5.0, 6.0, "TP=$5.0, SL=$6"),
        (5.0, 8.0, "TP=$5.0, SL=$8"),
    ]

    best_sharpe = -999
    best_label = ""
    for tp, sl, label in configs:
        eng_c = HFTrendScalper(
            ema_fast=3, ema_slow=8, ema_trend=50,
            tp_dollars=tp, sl_dollars=sl,
            spread=0.30, session_filter=True,
            session_hours=tuple(range(0, 13)),
            max_trades_per_day=20,
        )
        tr_c, eq_c = eng_c.run(df)
        res = print_report(tr_c, eq_c, label)
        if res.get('sharpe', -999) > best_sharpe:
            best_sharpe = res['sharpe']
            best_label = label

    print(f"\n  >>> Best TP/SL: {best_label} (Sharpe={best_sharpe:.2f})")

    # ═══════════════════════════════════════════════════════════
    # D. EMA 周期扫描
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("D. EMA 周期扫描 (fixed TP=$3, SL=$5)")
    print("=" * 60)

    ema_configs = [
        (2, 5, "EMA(2,5)"),
        (3, 8, "EMA(3,8)"),
        (5, 13, "EMA(5,13)"),
        (5, 20, "EMA(5,20)"),
        (8, 21, "EMA(8,21)"),
    ]

    for fast, slow, label in ema_configs:
        eng_d = HFTrendScalper(
            ema_fast=fast, ema_slow=slow, ema_trend=50,
            tp_dollars=3.0, sl_dollars=5.0,
            spread=0.30, session_filter=True,
            session_hours=tuple(range(0, 13)),
            max_trades_per_day=20,
        )
        tr_d, eq_d = eng_d.run(df)
        print_report(tr_d, eq_d, label)

    # ═══════════════════════════════════════════════════════════
    # E. ATR自适应 TP/SL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("E. ATR自适应版本")
    print("=" * 60)

    # Use ATR-scaled TP/SL
    for atr_tp, atr_sl, label in [
        (0.3, 0.5, "TP=0.3*ATR, SL=0.5*ATR"),
        (0.5, 0.8, "TP=0.5*ATR, SL=0.8*ATR"),
        (0.5, 1.0, "TP=0.5*ATR, SL=1.0*ATR"),
        (0.8, 1.2, "TP=0.8*ATR, SL=1.2*ATR"),
    ]:
        # Compute average ATR to convert
        df_tmp = df.copy()
        tr_vals = pd.DataFrame({
            'hl': df_tmp['High'] - df_tmp['Low'],
            'hc': (df_tmp['High'] - df_tmp['Close'].shift(1)).abs(),
            'lc': (df_tmp['Low'] - df_tmp['Close'].shift(1)).abs(),
        }).max(axis=1)
        avg_atr = tr_vals.rolling(14).mean().median()

        tp_d = atr_tp * avg_atr
        sl_d = atr_sl * avg_atr

        eng_e = HFTrendScalper(
            ema_fast=3, ema_slow=8, ema_trend=50,
            tp_dollars=tp_d, sl_dollars=sl_d,
            spread=0.30, session_filter=True,
            session_hours=tuple(range(0, 13)),
            max_trades_per_day=20,
        )
        tr_e, eq_e = eng_e.run(df)
        print_report(tr_e, eq_e, f"{label} (TP=${tp_d:.1f}, SL=${sl_d:.1f})")

    elapsed = time.time() - t0
    print(f"\n\nTotal: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
