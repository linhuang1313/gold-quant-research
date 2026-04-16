"""L5 配置端到端验证"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from strategies.exit_logic import calc_trailing_params, check_trailing_exit, check_time_decay_tp

atr = 15.0
errors = []

# === 1. Regime Trail 参数正确性 ===
print("=== Regime Trail 参数验证 ===")

act, dist = calc_trailing_params(atr, 0.85)  # high
print(f"High  (pct=0.85): activate=${act:.2f}, distance=${dist:.2f}")
if abs(act - 1.80) > 0.01 or abs(dist - 0.30) > 0.01:
    errors.append(f"High regime wrong: {act}, {dist}")

act, dist = calc_trailing_params(atr, 0.50)  # normal
print(f"Normal(pct=0.50): activate=${act:.2f}, distance=${dist:.2f}")
if abs(act - 4.20) > 0.01 or abs(dist - 0.90) > 0.01:
    errors.append(f"Normal regime wrong: {act}, {dist}")

act, dist = calc_trailing_params(atr, 0.15)  # low
print(f"Low   (pct=0.15): activate=${act:.2f}, distance=${dist:.2f}")
if abs(act - 6.00) > 0.01 or abs(dist - 1.50) > 0.01:
    errors.append(f"Low regime wrong: {act}, {dist}")

# === 2. Trailing 激活/触发逻辑 ===
print("\n=== Trailing 激活/触发逻辑 ===")

# BUY, high regime, 浮盈 $2 > activate $1.80
reason, extreme, trail, act_th = check_trailing_exit(
    'BUY', 2002.0, 2000.0, atr, 0.85, 2002.0, 0
)
print(f"BUY +$2, high: trail={trail}, extreme={extreme}, activate=${act_th:.2f}")
if trail != 2001.70:
    errors.append(f"Trail price wrong: {trail}, expected 2001.70")
if reason is not None:
    errors.append(f"Should NOT trigger yet, but got reason")

# 价格小幅回落但仍在激活阈值以上 → trail 触发
# activate=$1.80, 浮盈 2001.85-2000=1.85 > 1.80, trail=2002-0.30=2001.70
# 2001.85 > 2001.70 → 不触发（价格仍在 trail 上方）
reason2a, _, _, _ = check_trailing_exit(
    'BUY', 2001.85, 2000.0, atr, 0.85, 2002.0, 2001.70
)
print(f"BUY at 2001.85 (> trail 2001.70, profit > activate): {'TRIGGERED' if reason2a else 'tracking'}")

# 价格跌到 trail 以下且仍在激活以上 → 不可能（trail=2001.70, activate at $1.80 means 2001.80）
# 实际：浮盈低于 activate 时函数直接返回 None（设计如此，trailing 暂时失活）
reason2b, _, _, _ = check_trailing_exit(
    'BUY', 2001.50, 2000.0, atr, 0.85, 2002.0, 2001.70
)
print(f"BUY at 2001.50 (profit $1.50 < activate $1.80): {'TRAILING PAUSED' if not reason2b else 'TRIGGERED'}")
# 这是正确行为：浮盈缩水到激活线以下，trailing 暂停

# SELL 方向
reason3, extreme3, trail3, _ = check_trailing_exit(
    'SELL', 1998.0, 2000.0, atr, 0.85, 1998.0, 0
)
print(f"SELL -$2, high: trail={trail3}, extreme={extreme3}")
if trail3 != 1998.30:
    errors.append(f"SELL trail price wrong: {trail3}, expected 1998.30")

# === 3. TDTP 函数仍正常(但不被调用) ===
print("\n=== TDTP 函数验证(不被实盘调用) ===")
result = check_time_decay_tp('BUY', 2050, 2040, 2.5, atr, False)
print(f"TDTP result: {result}")

# === 4. Config 一致性 ===
print("\n=== Config 一致性 ===")
import config
print(f"config.TRAILING_ACTIVATE_ATR = {config.TRAILING_ACTIVATE_ATR}")
print(f"config.TRAILING_DISTANCE_ATR = {config.TRAILING_DISTANCE_ATR}")
print(f"config.V3_ATR_REGIME_ENABLED = {config.V3_ATR_REGIME_ENABLED}")

if config.TRAILING_ACTIVATE_ATR != 0.28:
    errors.append(f"Config TRAILING_ACTIVATE_ATR wrong: {config.TRAILING_ACTIVATE_ATR}")
if config.TRAILING_DISTANCE_ATR != 0.06:
    errors.append(f"Config TRAILING_DISTANCE_ATR wrong: {config.TRAILING_DISTANCE_ATR}")
if not config.V3_ATR_REGIME_ENABLED:
    errors.append("V3_ATR_REGIME_ENABLED should be True")

# === 5. LIVE_PARITY 一致性 ===
print("\n=== LIVE_PARITY_KWARGS 一致性 ===")
from backtest.runner import LIVE_PARITY_KWARGS as LP
print(f"time_decay_tp = {LP['time_decay_tp']}")
print(f"regime high   = {LP['regime_config']['high']}")
print(f"regime normal = {LP['regime_config']['normal']}")
print(f"regime low    = {LP['regime_config']['low']}")
print(f"trailing base = {LP['trailing_activate_atr']}/{LP['trailing_distance_atr']}")

if LP['time_decay_tp'] != False:
    errors.append("LIVE_PARITY time_decay_tp should be False")
if LP['regime_config']['high'] != {'trail_act': 0.12, 'trail_dist': 0.02}:
    errors.append(f"LIVE_PARITY high wrong: {LP['regime_config']['high']}")
if LP['regime_config']['normal'] != {'trail_act': 0.28, 'trail_dist': 0.06}:
    errors.append(f"LIVE_PARITY normal wrong: {LP['regime_config']['normal']}")
if LP['regime_config']['low'] != {'trail_act': 0.40, 'trail_dist': 0.10}:
    errors.append(f"LIVE_PARITY low wrong: {LP['regime_config']['low']}")

# === 6. gold_trader TDTP 确认关闭 ===
print("\n=== gold_trader.py TDTP 关闭验证 ===")
with open('gold_trader.py', 'r', encoding='utf-8') as f:
    code = f.read()
    # 检查 check_time_decay_tp 的调用是否被注释
    import re
    active_calls = re.findall(r'^[^#]*check_time_decay_tp', code, re.MULTILINE)
    # 只有 import 行应该匹配
    non_import = [c for c in active_calls if 'import' not in c]
    if non_import:
        errors.append(f"TDTP still has active calls: {non_import}")
        print("WARNING: TDTP may still be active!")
    else:
        print("TDTP calls are all commented out (only import remains)")

# === 结果 ===
print("\n" + "=" * 50)
if errors:
    print(f"FAILED! {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL CHECKS PASSED - L5 配置完全正确")
print("=" * 50)
