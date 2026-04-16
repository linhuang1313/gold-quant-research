"""
Research Config — 独立于实盘 config.py
========================================
回测引擎使用的所有常量，修改此文件不影响实盘系统。
值与实盘 config.py 保持对齐（截至 2026-04-16 L5.1 部署版本）。
"""

# ── 账户参数 ──
CAPITAL = 2000
LOT_SIZE = 0.03
RISK_PER_TRADE = 50
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 0.05
POINT_VALUE_PER_LOT = 100
STOP_LOSS_PIPS = 20
DAILY_MAX_LOSSES = 5
MAX_POSITIONS = 1
COOLDOWN_MINUTES = 30

# ── Trailing Stop ──
TRAILING_STOP_ENABLED = True
TRAILING_ACTIVATE_ATR = 0.28
TRAILING_DISTANCE_ATR = 0.06
V3_ATR_REGIME_ENABLED = True

# ── 策略定义 ──
STRATEGIES = {
    "keltner": {
        "enabled": True,
        "name": "Keltner通道突破",
        "stop_loss": 20,
        "take_profit": 35,
        "max_hold_bars": 5,
    },
    "macd": {
        "enabled": False,
        "name": "MACD+EMA100趋势",
        "stop_loss": 20,
        "take_profit": 50,
        "max_hold_bars": 20,
    },
    "orb": {
        "enabled": True,
        "name": "NY开盘区间突破",
        "max_hold_bars": 6,
    },
    "m15_rsi": {
        "enabled": True,
        "name": "M15 RSI均值回归",
        "max_hold_bars": 4,
    },
    "gap_fill": {
        "enabled": False,
        "name": "周一跳空回补",
        "max_hold_bars": 8,
    },
}

# ── ORB 参数 ──
ORB_ENABLED = True
ORB_NY_OPEN_HOUR_UTC = 14
ORB_RANGE_MINUTES = 15
ORB_EXPIRY_MINUTES = 120
ORB_SL_MULTIPLIER = 0.75
ORB_TP_MULTIPLIER = 3.0
ORB_SL_MIN_ATR_MULTIPLIER = 1.5

# ── 自动调仓 ──
AUTO_LOT_SIZING = True
MAX_LOT_CAP_BY_LOSSES = {0: 0.05, 1: 0.03, 2: 0.02, 3: 0.01}

# ── 趋势过滤 ──
INTRADAY_TREND_ENABLED = True
INTRADAY_TREND_THRESHOLD = 0.50
INTRADAY_TREND_KC_ONLY_THRESHOLD = 0.60
RSI_ADX_BLOCK_THRESHOLD = 40

# ── 其他 ──
ADD_POSITION_ENABLED = False
KELTNER_EXHAUSTION_RSI_LOW = 0
KELTNER_EXHAUSTION_RSI_HIGH = 100
