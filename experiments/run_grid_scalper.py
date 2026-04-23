"""
Grid Trend Scalper — 网格追踪策略回测
=========================================
基于量化机构实盘交易记录逆向工程的策略：
  - M15 级别判断短期方向（EMA交叉 + RSI）
  - 沿方向分批建仓，固定网格间距 ~$1.0
  - 每批最多4单，统一止盈平仓
  - 固定止损保护
  - 亚盘到欧盘前震荡时段最佳

使用 M15 数据回测（15分钟K线是最小可用数据）。
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
from backtest.stats import calc_stats, aggregate_daily_pnl


# ═════════════════════════════════════════════════════════════
# Grid Position
# ═════════════════════════════════════════════════════════════

@dataclass
class GridOrder:
    direction: str          # 'BUY' or 'SELL'
    entry_price: float
    entry_time: pd.Timestamp
    lots: float


@dataclass
class GridBatch:
    """A batch of up to N grid orders in the same direction."""
    direction: str
    orders: List[GridOrder] = field(default_factory=list)
    sl_price: float = 0.0
    tp_price: float = 0.0
    batch_open_time: pd.Timestamp = None

    @property
    def avg_price(self) -> float:
        if not self.orders:
            return 0
        return np.mean([o.entry_price for o in self.orders])

    @property
    def last_price(self) -> float:
        return self.orders[-1].entry_price if self.orders else 0

    @property
    def n_orders(self) -> int:
        return len(self.orders)

    def unrealized_pnl(self, price: float) -> float:
        total = 0
        for o in self.orders:
            if self.direction == 'BUY':
                total += (price - o.entry_price) * o.lots * 100
            else:
                total += (o.entry_price - price) * o.lots * 100
        return total


# ═════════════════════════════════════════════════════════════
# Grid Scalper Engine
# ═════════════════════════════════════════════════════════════

class GridScalperEngine:
    """
    网格追踪策略回测引擎。

    Parameters:
    -----------
    grid_spacing : 网格间距（美元），每隔多少加一单
    max_orders : 每批最大单数
    tp_from_last : 止盈距离（从最后一单算，美元）
    sl_from_avg : 止损距离（从均价算，美元）
    lots : 每单手数
    spread : 点差（美元）
    ema_fast : 快速EMA周期（M15 bars）
    ema_slow : 慢速EMA周期（M15 bars）
    rsi_period : RSI周期
    rsi_overbought : RSI超买阈值（做空信号）
    rsi_oversold : RSI超卖阈值（做多信号）
    session_filter : 是否启用交易时段过滤
    session_start_utc : 允许交易开始时间（UTC小时）
    session_end_utc : 允许交易结束时间（UTC小时）
    max_batches_per_day : 每天最大批次数
    cooldown_bars : 两批之间冷却期（M15 bars）
    """

    def __init__(
        self,
        grid_spacing: float = 1.0,
        max_orders: int = 4,
        tp_from_last: float = 1.2,
        sl_from_avg: float = 4.0,
        lots: float = 0.10,
        spread: float = 0.30,
        ema_fast: int = 5,
        ema_slow: int = 13,
        rsi_period: int = 14,
        rsi_overbought: float = 65,
        rsi_oversold: float = 35,
        session_filter: bool = True,
        session_start_utc: int = 0,
        session_end_utc: int = 13,
        max_batches_per_day: int = 8,
        cooldown_bars: int = 2,
        atr_period: int = 14,
        min_atr: float = 1.5,
        max_atr: float = 15.0,
    ):
        self.grid_spacing = grid_spacing
        self.max_orders = max_orders
        self.tp_from_last = tp_from_last
        self.sl_from_avg = sl_from_avg
        self.lots = lots
        self.spread = spread
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.session_filter = session_filter
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.max_batches_per_day = max_batches_per_day
        self.cooldown_bars = cooldown_bars
        self.atr_period = atr_period
        self.min_atr = min_atr
        self.max_atr = max_atr

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA, RSI, ATR on M15 data."""
        df = df.copy()
        df['ema_fast'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': (df['High'] - df['Close'].shift(1)).abs(),
            'lc': (df['Low'] - df['Close'].shift(1)).abs(),
        }).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()

        # EMA slope (direction strength)
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_prev'] = df['ema_diff'].shift(1)

        return df

    def _get_direction(self, row) -> Optional[str]:
        """Determine trade direction based on EMA crossover + RSI."""
        ema_diff = row['ema_diff']
        ema_diff_prev = row.get('ema_diff_prev', 0)
        rsi = row['rsi']

        if pd.isna(ema_diff) or pd.isna(rsi):
            return None

        # EMA golden cross or fast > slow + RSI not overbought
        if ema_diff > 0 and rsi < self.rsi_overbought:
            return 'BUY'
        # EMA death cross or fast < slow + RSI not oversold
        elif ema_diff < 0 and rsi > self.rsi_oversold:
            return 'SELL'

        return None

    def _in_session(self, ts: pd.Timestamp) -> bool:
        """Check if timestamp is within allowed trading session."""
        if not self.session_filter:
            return True
        hour = ts.hour
        if self.session_start_utc <= self.session_end_utc:
            return self.session_start_utc <= hour < self.session_end_utc
        else:
            return hour >= self.session_start_utc or hour < self.session_end_utc

    def run(self, df: pd.DataFrame) -> Tuple[List[TradeRecord], List[float]]:
        """Run backtest on M15 dataframe."""
        df = self._calc_indicators(df)

        trades: List[TradeRecord] = []
        equity = [2000.0]
        current_batch: Optional[GridBatch] = None
        last_batch_close_idx = -999
        daily_batch_count: Dict[str, int] = {}

        n_bars = len(df)
        print_pct = max(1, n_bars // 10)

        for i in range(max(self.ema_slow, self.rsi_period, self.atr_period) + 5, n_bars):
            if i % print_pct == 0:
                print(f"    {i*100//n_bars}%...", end="    ", flush=True)

            row = df.iloc[i]
            ts = df.index[i]
            price = row['Close']
            high = row['High']
            low = row['Low']
            atr = row['atr']
            day_key = str(ts.date())

            if pd.isna(atr) or atr < self.min_atr or atr > self.max_atr:
                continue

            # ── Check existing batch exits ──
            if current_batch is not None:
                batch = current_batch
                hit_tp = False
                hit_sl = False
                exit_price = 0
                exit_reason = ""

                if batch.direction == 'BUY':
                    if high >= batch.tp_price:
                        hit_tp = True
                        exit_price = batch.tp_price - self.spread
                        exit_reason = "GridTP"
                    elif low <= batch.sl_price:
                        hit_sl = True
                        exit_price = batch.sl_price - self.spread
                        exit_reason = "GridSL"
                else:
                    if low <= batch.tp_price:
                        hit_tp = True
                        exit_price = batch.tp_price + self.spread
                        exit_reason = "GridTP"
                    elif high >= batch.sl_price:
                        hit_sl = True
                        exit_price = batch.sl_price + self.spread
                        exit_reason = "GridSL"

                if hit_tp or hit_sl:
                    for order in batch.orders:
                        if batch.direction == 'BUY':
                            pnl = (exit_price - order.entry_price) * order.lots * 100
                        else:
                            pnl = (order.entry_price - exit_price) * order.lots * 100

                        bars_held = i - df.index.get_loc(order.entry_time) if order.entry_time in df.index else 1
                        trades.append(TradeRecord(
                            strategy="GridScalper",
                            direction=batch.direction,
                            entry_price=order.entry_price,
                            exit_price=exit_price,
                            entry_time=order.entry_time,
                            exit_time=ts,
                            lots=order.lots,
                            pnl=pnl,
                            exit_reason=exit_reason,
                            bars_held=bars_held,
                        ))
                        equity.append(equity[-1] + pnl)

                    current_batch = None
                    last_batch_close_idx = i
                    continue

                # ── Try to add more orders to the batch ──
                if batch.n_orders < self.max_orders:
                    last_entry = batch.last_price
                    if batch.direction == 'BUY':
                        next_entry_level = last_entry + self.grid_spacing
                        if high >= next_entry_level:
                            entry_px = next_entry_level + self.spread
                            batch.orders.append(GridOrder(
                                direction='BUY', entry_price=entry_px,
                                entry_time=ts, lots=self.lots,
                            ))
                            # Update TP/SL
                            batch.tp_price = entry_px + self.tp_from_last
                            batch.sl_price = batch.avg_price - self.sl_from_avg
                    else:
                        next_entry_level = last_entry - self.grid_spacing
                        if low <= next_entry_level:
                            entry_px = next_entry_level - self.spread
                            batch.orders.append(GridOrder(
                                direction='SELL', entry_price=entry_px,
                                entry_time=ts, lots=self.lots,
                            ))
                            batch.tp_price = entry_px - self.tp_from_last
                            batch.sl_price = batch.avg_price + self.sl_from_avg

                continue  # Don't open new batch while one is active

            # ── Open new batch ──
            if i - last_batch_close_idx < self.cooldown_bars:
                continue

            if not self._in_session(ts):
                continue

            if daily_batch_count.get(day_key, 0) >= self.max_batches_per_day:
                continue

            direction = self._get_direction(row)
            if direction is None:
                continue

            if direction == 'BUY':
                entry_px = price + self.spread
                tp = entry_px + self.tp_from_last
                sl = entry_px - self.sl_from_avg
            else:
                entry_px = price - self.spread
                tp = entry_px - self.tp_from_last
                sl = entry_px + self.sl_from_avg

            current_batch = GridBatch(
                direction=direction,
                orders=[GridOrder(direction=direction, entry_price=entry_px,
                                  entry_time=ts, lots=self.lots)],
                sl_price=sl,
                tp_price=tp,
                batch_open_time=ts,
            )
            daily_batch_count[day_key] = daily_batch_count.get(day_key, 0) + 1

        # Close any open batch at the end
        if current_batch is not None:
            last_price = df.iloc[-1]['Close']
            for order in current_batch.orders:
                if current_batch.direction == 'BUY':
                    pnl = (last_price - order.entry_price) * order.lots * 100
                else:
                    pnl = (order.entry_price - last_price) * order.lots * 100
                trades.append(TradeRecord(
                    strategy="GridScalper", direction=current_batch.direction,
                    entry_price=order.entry_price, exit_price=last_price,
                    entry_time=order.entry_time, exit_time=df.index[-1],
                    lots=order.lots, pnl=pnl, exit_reason="EOD",
                    bars_held=1,
                ))
                equity.append(equity[-1] + pnl)

        print(" done!", flush=True)
        return trades, equity


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════

def print_grid_report(trades: List[TradeRecord], equity: List[float], label: str = ""):
    """Print comprehensive report for grid scalper."""
    if not trades:
        print(f"  {label}: No trades")
        return {}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0

    daily_pnl = aggregate_daily_pnl(trades)
    sharpe = 0
    if len(daily_pnl) > 1 and np.std(daily_pnl, ddof=1) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl, ddof=1) * np.sqrt(252)

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        if dd > max_dd:
            max_dd = dd

    tp_trades = [t for t in trades if t.exit_reason == "GridTP"]
    sl_trades = [t for t in trades if t.exit_reason == "GridSL"]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades)}")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Avg Win: ${avg_win:.2f}  |  Avg Loss: ${avg_loss:.2f}  |  RR: {avg_win/avg_loss:.2f}" if avg_loss > 0 else "")
    print(f"  Max Drawdown: ${max_dd:.2f}")
    print(f"  TP exits: {len(tp_trades)} ({sum(t.pnl for t in tp_trades):+.2f})")
    print(f"  SL exits: {len(sl_trades)} ({sum(t.pnl for t in sl_trades):+.2f})")

    # By direction
    buys = [t for t in trades if t.direction == 'BUY']
    sells = [t for t in trades if t.direction == 'SELL']
    print(f"\n  BUY:  N={len(buys)}, PnL=${sum(t.pnl for t in buys):,.2f}, WR={len([t for t in buys if t.pnl>0])/max(1,len(buys))*100:.1f}%")
    print(f"  SELL: N={len(sells)}, PnL=${sum(t.pnl for t in sells):,.2f}, WR={len([t for t in sells if t.pnl>0])/max(1,len(sells))*100:.1f}%")

    # Per year
    year_pnl: Dict[int, float] = {}
    year_n: Dict[int, int] = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] = year_pnl.get(y, 0) + t.pnl
        year_n[y] = year_n.get(y, 0) + 1

    print(f"\n  Year-by-Year:")
    for y in sorted(year_pnl.keys()):
        print(f"    {y}: N={year_n[y]:>5}, PnL=${year_pnl[y]:>10,.2f}")

    return {
        'n': len(trades), 'pnl': total_pnl, 'sharpe': sharpe,
        'wr': wr, 'max_dd': max_dd, 'avg_win': avg_win, 'avg_loss': avg_loss,
    }


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("# Grid Trend Scalper — Backtest")
    print(f"# Started: {pd.Timestamp.now()}")

    # Load M15 data
    print("\nLoading M15 data...")
    m15_path = Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv")
    if not m15_path.exists():
        m15_path = Path("data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv")
    df = load_csv(str(m15_path))
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    # ─── Baseline: 从截图逆向工程的参数 ───
    print("\n" + "="*60)
    print("A. Baseline — 截图逆向工程参数")
    print("="*60)
    engine = GridScalperEngine(
        grid_spacing=1.0,
        max_orders=4,
        tp_from_last=1.2,
        sl_from_avg=4.0,
        lots=0.10,
        spread=0.30,
        ema_fast=5,
        ema_slow=13,
        rsi_period=14,
        rsi_overbought=65,
        rsi_oversold=35,
        session_filter=True,
        session_start_utc=0,
        session_end_utc=13,
        max_batches_per_day=8,
        cooldown_bars=2,
    )
    print("  Running baseline...")
    trades_a, equity_a = engine.run(df)
    res_a = print_grid_report(trades_a, equity_a, "Baseline (Grid=$1, TP=$1.2, SL=$4)")

    # ─── B: 放宽时段，全天交易 ───
    print("\n" + "="*60)
    print("B. 全天交易（不限时段）")
    print("="*60)
    engine_b = GridScalperEngine(
        grid_spacing=1.0,
        max_orders=4,
        tp_from_last=1.2,
        sl_from_avg=4.0,
        lots=0.10,
        spread=0.30,
        ema_fast=5,
        ema_slow=13,
        session_filter=False,
        max_batches_per_day=12,
        cooldown_bars=2,
    )
    print("  Running full-day...")
    trades_b, equity_b = engine_b.run(df)
    res_b = print_grid_report(trades_b, equity_b, "Full Day (no session filter)")

    # ─── C: 参数网格搜索 ───
    print("\n" + "="*60)
    print("C. 参数扫描")
    print("="*60)

    grid_params = [
        {"grid_spacing": 0.8, "tp_from_last": 1.0, "sl_from_avg": 3.5, "label": "Tight (G=0.8, TP=1.0, SL=3.5)"},
        {"grid_spacing": 1.0, "tp_from_last": 1.5, "sl_from_avg": 5.0, "label": "Wide TP (G=1.0, TP=1.5, SL=5.0)"},
        {"grid_spacing": 1.5, "tp_from_last": 1.5, "sl_from_avg": 5.0, "label": "Wide Grid (G=1.5, TP=1.5, SL=5.0)"},
        {"grid_spacing": 1.0, "tp_from_last": 1.2, "sl_from_avg": 3.0, "label": "Tight SL (G=1.0, TP=1.2, SL=3.0)"},
        {"grid_spacing": 0.5, "tp_from_last": 0.8, "sl_from_avg": 3.0, "label": "Micro (G=0.5, TP=0.8, SL=3.0)"},
        {"grid_spacing": 2.0, "tp_from_last": 2.0, "sl_from_avg": 6.0, "label": "Macro (G=2.0, TP=2.0, SL=6.0)"},
    ]

    best_sharpe = -999
    best_label = ""
    results = []

    for params in grid_params:
        label = params.pop("label")
        eng = GridScalperEngine(
            **params,
            max_orders=4,
            lots=0.10,
            spread=0.30,
            ema_fast=5,
            ema_slow=13,
            session_filter=True,
            session_start_utc=0,
            session_end_utc=13,
            max_batches_per_day=8,
            cooldown_bars=2,
        )
        print(f"\n  [{label}]")
        tr, eq = eng.run(df)
        res = print_grid_report(tr, eq, label)
        res['label'] = label
        results.append(res)
        if res.get('sharpe', 0) > best_sharpe:
            best_sharpe = res['sharpe']
            best_label = label

    print(f"\n  Best config: {best_label} (Sharpe={best_sharpe:.2f})")

    # ─── D: EMA 周期扫描 ───
    print("\n" + "="*60)
    print("D. EMA周期扫描")
    print("="*60)

    ema_combos = [
        (3, 8, "EMA(3,8)"),
        (5, 13, "EMA(5,13)"),
        (5, 20, "EMA(5,20)"),
        (8, 21, "EMA(8,21)"),
        (10, 30, "EMA(10,30)"),
    ]

    for fast, slow, label in ema_combos:
        eng = GridScalperEngine(
            grid_spacing=1.0, max_orders=4, tp_from_last=1.2, sl_from_avg=4.0,
            lots=0.10, spread=0.30,
            ema_fast=fast, ema_slow=slow,
            session_filter=True, session_start_utc=0, session_end_utc=13,
            max_batches_per_day=8, cooldown_bars=2,
        )
        print(f"\n  [{label}]")
        tr, eq = eng.run(df)
        print_grid_report(tr, eq, label)

    # ─── E: 方向判断对比（纯EMA vs 纯RSI vs 组合） ───
    print("\n" + "="*60)
    print("E. RSI 阈值扫描")
    print("="*60)

    rsi_combos = [
        (60, 40, "RSI(60/40)"),
        (65, 35, "RSI(65/35) — baseline"),
        (70, 30, "RSI(70/30)"),
        (55, 45, "RSI(55/45) — loose"),
    ]

    for ob, os_, label in rsi_combos:
        eng = GridScalperEngine(
            grid_spacing=1.0, max_orders=4, tp_from_last=1.2, sl_from_avg=4.0,
            lots=0.10, spread=0.30,
            ema_fast=5, ema_slow=13,
            rsi_overbought=ob, rsi_oversold=os_,
            session_filter=True, session_start_utc=0, session_end_utc=13,
            max_batches_per_day=8, cooldown_bars=2,
        )
        print(f"\n  [{label}]")
        tr, eq = eng.run(df)
        print_grid_report(tr, eq, label)

    elapsed = time.time() - t0
    print(f"\n\nTotal elapsed: {elapsed:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
