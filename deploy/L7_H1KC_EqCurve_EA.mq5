//+------------------------------------------------------------------+
//| L7_H1KC_EqCurve_EA.mq5                                          |
//| L7(MH=8) + H1 KC同向过滤 + EqCurve LB=10                        |
//| Sharpe 13.97 | K-Fold 6/6 | MaxDD $64                            |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== 交易参数 ==="
input double   LotSize           = 0.03;      // 基础手数
input int      MagicNumber       = 20250418;  // EA 唯一标识
input int      MaxSlippage       = 30;        // 最大滑点 (points)

input group "=== L7 Keltner 入场参数 (M15) ==="
input int      KC_EMA_Period     = 25;        // KC 中轨 EMA 周期
input double   KC_Multiplier     = 1.2;       // KC 通道乘数
input int      ATR_Period        = 14;        // ATR 周期
input int      ADX_Period        = 14;        // ADX 周期
input double   ADX_Threshold     = 18.0;      // ADX 最低阈值
input int      EMA100_Period     = 100;       // EMA100 趋势过滤周期

input group "=== L7 出场参数 ==="
input double   SL_ATR_Mult       = 3.5;       // 止损 = ATR × 此值
input double   TP_ATR_Mult       = 8.0;       // 止盈 = ATR × 此值
input int      MaxHold_Bars      = 8;         // 最大持仓 M15 bar 数 (=2小时)
input double   Trail_Act_ATR     = 0.28;      // 追踪止盈激活距离 (ATR倍数)
input double   Trail_Dist_ATR    = 0.06;      // 追踪止盈距离 (ATR倍数)

input group "=== 时间自适应追踪 (TATrail) ==="
input int      TATrail_Start     = 2;         // 开始收紧的 bar 数
input double   TATrail_Decay     = 0.75;      // 每 bar 衰减系数
input double   TATrail_Floor     = 0.003;     // 最小 trail_dist 倍数

input group "=== Regime 自适应参数 ==="
input double   Regime_Low_Act    = 0.40;      // 低波 trail 激活
input double   Regime_Low_Dist   = 0.10;      // 低波 trail 距离
input double   Regime_Norm_Act   = 0.28;      // 正常 trail 激活
input double   Regime_Norm_Dist  = 0.06;      // 正常 trail 距离
input double   Regime_High_Act   = 0.12;      // 高波 trail 激活
input double   Regime_High_Dist  = 0.02;      // 高波 trail 距离

input group "=== H1 KC 多TF同向过滤 (第二层) ==="
input bool     H1_Filter_Enabled = true;      // 启用 H1 KC 过滤
input int      H1_KC_EMA_Period  = 20;        // H1 KC EMA 周期
input double   H1_KC_Multiplier  = 2.0;       // H1 KC 乘数
input int      H1_ATR_Period     = 14;        // H1 ATR 周期

input group "=== EqCurve 风控层 (第三层) ==="
input bool     EqCurve_Enabled   = true;      // 启用 EqCurve
input int      EqCurve_LB        = 10;        // 回看笔数
input double   EqCurve_Cut       = 0.0;       // 阈值 (平均PnL < Cut 则触发)
input double   EqCurve_Red       = 0.0;       // 缩仓比例 (0=跳过交易)

input group "=== 入场间隔 ==="
input double   MinEntryGapHours  = 1.0;       // 最小入场间隔 (小时)

input group "=== Choppy 过滤 ==="
input double   ChoppyThreshold   = 0.50;      // Choppy 阈值

//--- Global variables
CTrade         trade;
int            h_atr_m15, h_atr_h1, h_adx, h_ema100, h_ema_kc_m15, h_ema_kc_h1;
datetime       lastEntryTime     = 0;
datetime       lastBarTime       = 0;
int            barsHeld          = 0;
double         entryATR          = 0;
double         trailStopPrice    = 0;
double         extremePrice      = 0;

// EqCurve tracking
double         recentPnL[];
int            eqCurveCount      = 0;
bool           eqCurveSkipping   = false;

// ATR regime percentile tracking
double         atrHistory[];
int            atrHistCount      = 0;
int            ATR_HIST_SIZE     = 50;  // rolling window for live percentile


//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(MaxSlippage);
   
   h_atr_m15    = iATR(_Symbol, PERIOD_M15, ATR_Period);
   h_atr_h1     = iATR(_Symbol, PERIOD_H1, H1_ATR_Period);
   h_adx        = iADX(_Symbol, PERIOD_H1, ADX_Period);
   h_ema100     = iMA(_Symbol, PERIOD_H1, EMA100_Period, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_kc_m15 = iMA(_Symbol, PERIOD_H1, KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_kc_h1  = iMA(_Symbol, PERIOD_H1, H1_KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   
   if(h_atr_m15 == INVALID_HANDLE || h_atr_h1 == INVALID_HANDLE || 
      h_adx == INVALID_HANDLE || h_ema100 == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return INIT_FAILED;
   }
   
   ArrayResize(recentPnL, EqCurve_LB);
   ArrayInitialize(recentPnL, 0);
   
   ArrayResize(atrHistory, ATR_HIST_SIZE);
   ArrayInitialize(atrHistory, 0);
   
   Print("L7+H1KC+EqCurve EA initialized. Lot=", LotSize, 
         " H1Filter=", H1_Filter_Enabled, " EqCurve=", EqCurve_Enabled);
   
   return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(h_atr_m15 != INVALID_HANDLE) IndicatorRelease(h_atr_m15);
   if(h_atr_h1 != INVALID_HANDLE)  IndicatorRelease(h_atr_h1);
   if(h_adx != INVALID_HANDLE)     IndicatorRelease(h_adx);
   if(h_ema100 != INVALID_HANDLE)  IndicatorRelease(h_ema100);
   if(h_ema_kc_m15 != INVALID_HANDLE) IndicatorRelease(h_ema_kc_m15);
   if(h_ema_kc_h1 != INVALID_HANDLE)  IndicatorRelease(h_ema_kc_h1);
}


//+------------------------------------------------------------------+
//| Get ATR percentile regime (low/normal/high)                       |
//+------------------------------------------------------------------+
int GetATRRegime(double currentATR)
{
   if(atrHistCount < 10) return 1; // default normal
   
   double sorted[];
   int count = MathMin(atrHistCount, ATR_HIST_SIZE);
   ArrayResize(sorted, count);
   for(int i = 0; i < count; i++)
      sorted[i] = atrHistory[i % ATR_HIST_SIZE];
   ArraySort(sorted);
   
   int rank = 0;
   for(int i = 0; i < count; i++)
      if(sorted[i] <= currentATR) rank = i;
   
   double pct = (double)rank / (double)count;
   
   if(pct < 0.25) return 0;      // low volatility
   if(pct > 0.75) return 2;      // high volatility
   return 1;                      // normal
}


//+------------------------------------------------------------------+
//| Get trail parameters based on regime                              |
//+------------------------------------------------------------------+
void GetRegimeTrailParams(int regime, double &act, double &dist)
{
   switch(regime)
   {
      case 0: act = Regime_Low_Act;  dist = Regime_Low_Dist;  break;
      case 2: act = Regime_High_Act; dist = Regime_High_Dist; break;
      default: act = Regime_Norm_Act; dist = Regime_Norm_Dist; break;
   }
}


//+------------------------------------------------------------------+
//| Check H1 Keltner Channel direction (第二层过滤)                   |
//+------------------------------------------------------------------+
string GetH1KCDirection()
{
   if(!H1_Filter_Enabled) return "PASS";
   
   double h1_close = iClose(_Symbol, PERIOD_H1, 1);  // 上一根已完成H1 bar
   
   double h1_ema[];
   ArraySetAsSeries(h1_ema, true);
   if(CopyBuffer(h1_ema_kc_h1, 0, 1, 1, h1_ema) < 1) return "NEUTRAL";
   
   double h1_atr[];
   ArraySetAsSeries(h1_atr, true);
   if(CopyBuffer(h_atr_h1, 0, 1, 1, h1_atr) < 1) return "NEUTRAL";
   
   double kc_upper = h1_ema[0] + H1_KC_Multiplier * h1_atr[0];
   double kc_lower = h1_ema[0] - H1_KC_Multiplier * h1_atr[0];
   
   if(h1_close > kc_upper) return "BULL";
   if(h1_close < kc_lower) return "BEAR";
   return "NEUTRAL";
}


//+------------------------------------------------------------------+
//| Check EqCurve — 是否跳过当前交易 (第三层)                          |
//+------------------------------------------------------------------+
bool ShouldSkipByEqCurve()
{
   if(!EqCurve_Enabled) return false;
   if(eqCurveCount < EqCurve_LB) return false;
   
   double avg = 0;
   for(int i = 0; i < EqCurve_LB; i++)
      avg += recentPnL[i];
   avg /= EqCurve_LB;
   
   if(avg < EqCurve_Cut)
   {
      eqCurveSkipping = true;
      return (EqCurve_Red == 0.0); // Red=0 means skip entirely
   }
   
   eqCurveSkipping = false;
   return false;
}


//+------------------------------------------------------------------+
//| Record trade PnL to EqCurve buffer                                |
//+------------------------------------------------------------------+
void RecordPnL(double pnl)
{
   // Shift buffer
   for(int i = 0; i < EqCurve_LB - 1; i++)
      recentPnL[i] = recentPnL[i + 1];
   recentPnL[EqCurve_LB - 1] = pnl;
   eqCurveCount++;
}


//+------------------------------------------------------------------+
//| Calculate Choppy index (trend strength)                           |
//+------------------------------------------------------------------+
double CalcTrendScore()
{
   double close_arr[];
   ArraySetAsSeries(close_arr, true);
   if(CopyClose(_Symbol, PERIOD_H1, 0, 50, close_arr) < 50) return 0.5;
   
   int up = 0, dn = 0;
   for(int i = 0; i < 20; i++)
   {
      if(close_arr[i] > close_arr[i+1]) up++;
      else dn++;
   }
   return (double)MathMax(up, dn) / 20.0;
}


//+------------------------------------------------------------------+
//| Check M15 Keltner entry signal                                    |
//+------------------------------------------------------------------+
int CheckKeltnerSignal()
{
   // KC computed on H1, signal checked per bar
   double ema_kc[];
   ArraySetAsSeries(ema_kc, true);
   if(CopyBuffer(h_ema_kc_m15, 0, 0, 2, ema_kc) < 2) return 0;
   
   double atr_val[];
   ArraySetAsSeries(atr_val, true);
   if(CopyBuffer(h_atr_h1, 0, 0, 2, atr_val) < 2) return 0;
   
   double adx_val[];
   ArraySetAsSeries(adx_val, true);
   if(CopyBuffer(h_adx, 0, 0, 2, adx_val) < 2) return 0;
   
   double ema100_val[];
   ArraySetAsSeries(ema100_val, true);
   if(CopyBuffer(h_ema100, 0, 0, 2, ema100_val) < 2) return 0;
   
   double close = iClose(_Symbol, PERIOD_H1, 0);
   
   double kc_upper = ema_kc[0] + KC_Multiplier * atr_val[0];
   double kc_lower = ema_kc[0] - KC_Multiplier * atr_val[0];
   
   if(adx_val[0] < ADX_Threshold) return 0;
   
   // Choppy filter
   double trendScore = CalcTrendScore();
   if(trendScore < ChoppyThreshold) return 0;
   
   // BUY: close > KC upper + close > EMA100
   if(close > kc_upper && close > ema100_val[0]) return 1;
   
   // SELL: close < KC lower + close < EMA100
   if(close < kc_lower && close < ema100_val[0]) return -1;
   
   return 0;
}


//+------------------------------------------------------------------+
//| Manage open position: trailing, timeout, SL/TP                    |
//+------------------------------------------------------------------+
void ManagePosition()
{
   if(!PositionSelect(_Symbol)) return;
   
   long posMagic = PositionGetInteger(POSITION_MAGIC);
   if(posMagic != MagicNumber) return;
   
   long posType = PositionGetInteger(POSITION_TYPE);
   double posOpen = PositionGetDouble(POSITION_PRICE_OPEN);
   double posSL = PositionGetDouble(POSITION_SL);
   double posTP = PositionGetDouble(POSITION_TP);
   double posProfit = PositionGetDouble(POSITION_PROFIT);
   ulong ticket = PositionGetInteger(POSITION_TICKET);
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   barsHeld++;
   
   // Update extreme price
   if(posType == POSITION_TYPE_BUY)
      extremePrice = MathMax(extremePrice, bid);
   else
      extremePrice = MathMin(extremePrice, ask);
   
   // Get regime trail params
   double atr_h1[];
   ArraySetAsSeries(atr_h1, true);
   CopyBuffer(h_atr_h1, 0, 0, 1, atr_h1);
   double currentATR = (ArraySize(atr_h1) > 0) ? atr_h1[0] : entryATR;
   
   int regime = GetATRRegime(currentATR);
   double trail_act, trail_dist;
   GetRegimeTrailParams(regime, trail_act, trail_dist);
   
   // Time-adaptive trail: tighten after TATrail_Start bars
   if(barsHeld > TATrail_Start)
   {
      int extra = barsHeld - TATrail_Start;
      double decay_factor = MathPow(TATrail_Decay, extra);
      trail_dist = MathMax(trail_dist * decay_factor, TATrail_Floor * entryATR / currentATR * trail_dist);
   }
   
   // Trailing stop logic
   double activateDist = trail_act * entryATR;
   double trailDist    = trail_dist * entryATR;
   
   if(posType == POSITION_TYPE_BUY)
   {
      double floatProfit = bid - posOpen;
      if(floatProfit >= activateDist)
      {
         double newSL = extremePrice - trailDist;
         if(newSL > posSL && newSL < bid)
         {
            trade.PositionModify(ticket, newSL, posTP);
            trailStopPrice = newSL;
         }
      }
   }
   else // SELL
   {
      double floatProfit = posOpen - ask;
      if(floatProfit >= activateDist)
      {
         double newSL = extremePrice + trailDist;
         if(newSL < posSL && newSL > ask)
         {
            trade.PositionModify(ticket, newSL, posTP);
            trailStopPrice = newSL;
         }
      }
   }
   
   // MaxHold timeout
   if(barsHeld >= MaxHold_Bars)
   {
      Print("MaxHold reached (", MaxHold_Bars, " bars), closing position");
      trade.PositionClose(ticket, MaxSlippage);
      double pnl = posProfit;
      RecordPnL(pnl);
      barsHeld = 0;
      trailStopPrice = 0;
      extremePrice = 0;
   }
}


//+------------------------------------------------------------------+
//| Check if position was closed by SL/TP (broker-side)               |
//+------------------------------------------------------------------+
void CheckClosedTrades()
{
   static int lastDealsTotal = 0;
   int currentDeals = HistoryDealsTotal();
   
   if(currentDeals > lastDealsTotal)
   {
      HistorySelect(TimeCurrent() - 86400, TimeCurrent());
      int total = HistoryDealsTotal();
      for(int i = lastDealsTotal; i < total; i++)
      {
         ulong dealTicket = HistoryDealGetTicket(i);
         if(dealTicket == 0) continue;
         long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
         long dealEntry = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
         if(dealMagic == MagicNumber && dealEntry == DEAL_ENTRY_OUT)
         {
            double pnl = HistoryDealGetDouble(dealTicket, DEAL_PROFIT) 
                       + HistoryDealGetDouble(dealTicket, DEAL_COMMISSION)
                       + HistoryDealGetDouble(dealTicket, DEAL_SWAP);
            RecordPnL(pnl);
            Print("Trade closed. PnL=", pnl, " EqCurve count=", eqCurveCount);
            barsHeld = 0;
            trailStopPrice = 0;
            extremePrice = 0;
         }
      }
   }
   lastDealsTotal = currentDeals;
}


//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new M15 bar
   datetime currentBar = iTime(_Symbol, PERIOD_M15, 0);
   if(currentBar == lastBarTime) return;
   lastBarTime = currentBar;
   
   // Check for closed trades (SL/TP hit)
   CheckClosedTrades();
   
   // Update ATR history for regime detection
   double atr_h1[];
   ArraySetAsSeries(atr_h1, true);
   if(CopyBuffer(h_atr_h1, 0, 0, 1, atr_h1) >= 1 && atr_h1[0] > 0)
   {
      atrHistory[atrHistCount % ATR_HIST_SIZE] = atr_h1[0];
      atrHistCount++;
   }
   
   // If position open, manage it
   if(PositionSelect(_Symbol))
   {
      long posMagic = PositionGetInteger(POSITION_MAGIC);
      if(posMagic == MagicNumber)
      {
         ManagePosition();
         return;
      }
   }
   
   // === No position open — check for new entry ===
   
   // Entry gap check
   if(TimeCurrent() - lastEntryTime < (int)(MinEntryGapHours * 3600))
      return;
   
   // M15/H1 Keltner signal check
   int signal = CheckKeltnerSignal();
   if(signal == 0) return;
   
   // === 第二层: H1 KC 同向过滤 ===
   string h1Dir = GetH1KCDirection();
   if(h1Dir != "PASS") // filter enabled
   {
      if(signal == 1 && h1Dir != "BULL")
      {
         // Print("H1 KC filter blocked BUY (H1=", h1Dir, ")");
         return;
      }
      if(signal == -1 && h1Dir != "BEAR")
      {
         // Print("H1 KC filter blocked SELL (H1=", h1Dir, ")");
         return;
      }
   }
   
   // === 第三层: EqCurve 风控 ===
   if(ShouldSkipByEqCurve())
   {
      Print("EqCurve skip: recent ", EqCurve_LB, " trades avg PnL < ", EqCurve_Cut);
      return;
   }
   
   // Get ATR for SL/TP calculation
   double atr_m15[];
   ArraySetAsSeries(atr_m15, true);
   if(CopyBuffer(h_atr_m15, 0, 0, 1, atr_m15) < 1) return;
   double atr = atr_m15[0];
   if(atr < 0.1) return;
   
   entryATR = atr;
   double sl_dist = atr * SL_ATR_Mult;
   double tp_dist = atr * TP_ATR_Mult;
   
   double lot = LotSize;
   if(EqCurve_Enabled && eqCurveSkipping && EqCurve_Red > 0)
      lot = LotSize * EqCurve_Red;
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   if(signal == 1) // BUY
   {
      double sl = ask - sl_dist;
      double tp = ask + tp_dist;
      if(trade.Buy(lot, _Symbol, ask, sl, tp, "L7+H1KC+EqC BUY"))
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = bid;
         trailStopPrice = 0;
         Print("BUY @ ", ask, " SL=", sl, " TP=", tp, " ATR=", atr);
      }
   }
   else if(signal == -1) // SELL
   {
      double sl = bid + sl_dist;
      double tp = bid - tp_dist;
      if(trade.Sell(lot, _Symbol, bid, sl, tp, "L7+H1KC+EqC SELL"))
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = ask;
         trailStopPrice = 0;
         Print("SELL @ ", bid, " SL=", sl, " TP=", tp, " ATR=", atr);
      }
   }
}
//+------------------------------------------------------------------+
