//+------------------------------------------------------------------+
//| L8_BASE_EA.mq4                                                   |
//| L8_BASE + Cap80                                                   |
//| R41 K-Fold 6/6 PASS (CorrSh Mean=6.27, Min=1.14)                |
//| Chart: XAUUSD M15                                                |
//+------------------------------------------------------------------+
//| L8_BASE vs L7 差异:                                               |
//|   ADX: 18 -> 14 (更多信号)                                        |
//|   Normal Trail: 0.28/0.06 -> 0.14/0.025 (更紧锁利)               |
//|   High Trail: 0.12/0.02 -> 0.06/0.008 (高波动更紧)               |
//|   TATrail: ON -> OFF (不用时间衰减)                                |
//|   MaxHold: 8 -> 20 (给趋势更多空间)                                |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "3.00"
#property strict

//--- 交易参数
extern double LotSize          = 0.03;
extern int    MagicNumber      = 20250427;
extern int    MaxSlippage       = 30;

//--- Keltner 入场参数
extern int    KC_EMA_Period     = 25;
extern double KC_Multiplier     = 1.2;
extern int    ATR_Period        = 14;
extern int    ADX_Period        = 14;
extern double ADX_Threshold     = 14.0;     // L8: 14 (L7 was 18)
extern int    EMA100_Period     = 100;

//--- 出场参数
extern double SL_ATR_Mult       = 3.5;
extern double TP_ATR_Mult       = 8.0;
extern int    MaxHold_Bars      = 20;        // L8: 20 (L7 was 8)
extern double Trail_Act_ATR     = 0.14;      // L8: 0.14 (L7 was 0.28)
extern double Trail_Dist_ATR    = 0.025;     // L8: 0.025 (L7 was 0.06)

//--- Regime Trail
extern double Regime_Low_Act    = 0.22;      // L8
extern double Regime_Low_Dist   = 0.04;      // L8
extern double Regime_Norm_Act   = 0.14;      // L8
extern double Regime_Norm_Dist  = 0.025;     // L8
extern double Regime_High_Act   = 0.06;      // L8
extern double Regime_High_Dist  = 0.008;     // L8

//--- H1 KC Filter
extern bool   H1_Filter_Enabled = true;
extern int    H1_KC_EMA_Period  = 20;
extern double H1_KC_Multiplier  = 2.0;
extern int    H1_ATR_Period     = 14;

//--- EqCurve
extern bool   EqCurve_Enabled   = true;
extern int    EqCurve_LB        = 10;
extern double EqCurve_Cut       = 0.0;
extern double EqCurve_Red       = 0.0;

//--- 入场间隔
extern double MinEntryGapHours  = 1.0;

//--- Choppy
extern double ChoppyThreshold   = 0.50;

//=== EXECUTION EDGE ===
extern bool   KCBW_Enabled      = false;     // 实盘 OFF
extern int    KCBW_Lookback     = 5;
extern double KCBW_Min_Ratio    = 0.0;

extern bool   MaxLoss_Enabled   = true;
extern double MaxLoss_USD       = 80.0;      // Cap $80

extern bool   Session_Filter    = false;
extern int    Session_Skip_Start = 2;
extern int    Session_Skip_End   = 5;

//--- Global
datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;

double   recentPnL[];
int      eqCurveCount   = 0;
bool     eqCurveSkip    = false;

double   atrHistory[];
int      atrHistCount   = 0;
int      ATR_HIST_SIZE  = 50;

//+------------------------------------------------------------------+
int OnInit()
{
   ArrayResize(recentPnL, EqCurve_LB);
   ArrayInitialize(recentPnL, 0);
   ArrayResize(atrHistory, ATR_HIST_SIZE);
   ArrayInitialize(atrHistory, 0);

   Print("L8_BASE EA v3.0 | Lot=", LotSize,
         " ADX=", ADX_Threshold, " MH=", MaxHold_Bars,
         " Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " KCBW=", KCBW_Enabled, " Cap=", MaxLoss_USD);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int GetATRRegime(double currentATR)
{
   if(atrHistCount < 10) return 1;
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
   if(pct < 0.25) return 0;
   if(pct > 0.75) return 2;
   return 1;
}

//+------------------------------------------------------------------+
void GetRegimeTrailParams(int regime, double &act, double &dist)
{
   if(regime == 0)      { act = Regime_Low_Act;  dist = Regime_Low_Dist;  }
   else if(regime == 2) { act = Regime_High_Act; dist = Regime_High_Dist; }
   else                 { act = Regime_Norm_Act; dist = Regime_Norm_Dist; }
}

//+------------------------------------------------------------------+
string GetH1KCDirection()
{
   if(!H1_Filter_Enabled) return "PASS";

   double h1_close = iClose(Symbol(), PERIOD_H1, 1);
   double h1_ema   = iMA(Symbol(), PERIOD_H1, H1_KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   double h1_atr   = iATR(Symbol(), PERIOD_H1, H1_ATR_Period, 1);

   double kc_upper = h1_ema + H1_KC_Multiplier * h1_atr;
   double kc_lower = h1_ema - H1_KC_Multiplier * h1_atr;

   if(h1_close > kc_upper) return "BULL";
   if(h1_close < kc_lower) return "BEAR";
   return "NEUTRAL";
}

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
      eqCurveSkip = true;
      return (EqCurve_Red == 0.0);
   }
   eqCurveSkip = false;
   return false;
}

//+------------------------------------------------------------------+
void RecordPnL(double pnl)
{
   for(int i = 0; i < EqCurve_LB - 1; i++)
      recentPnL[i] = recentPnL[i + 1];
   recentPnL[EqCurve_LB - 1] = pnl;
   eqCurveCount++;
}

//+------------------------------------------------------------------+
double CalcTrendScore()
{
   int up = 0, dn = 0;
   for(int i = 0; i < 20; i++)
   {
      double c0 = iClose(Symbol(), PERIOD_H1, i);
      double c1 = iClose(Symbol(), PERIOD_H1, i + 1);
      if(c0 > c1) up++;
      else dn++;
   }
   return (double)MathMax(up, dn) / 20.0;
}

//+------------------------------------------------------------------+
bool IsKCBandwidthExpanding()
{
   if(!KCBW_Enabled) return true;

   double bw[];
   ArrayResize(bw, KCBW_Lookback + 1);

   for(int i = 0; i <= KCBW_Lookback; i++)
   {
      double ema_val = iMA(Symbol(), PERIOD_H1, KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, i + 1);
      double atr_val = iATR(Symbol(), PERIOD_H1, ATR_Period, i + 1);
      if(ema_val <= 0) return true;
      bw[i] = (KC_Multiplier * atr_val * 2.0) / ema_val;
   }

   double minBW = bw[1];
   for(int i = 2; i <= KCBW_Lookback; i++)
      if(bw[i] < minBW) minBW = bw[i];

   return (bw[0] > minBW);
}

//+------------------------------------------------------------------+
bool IsSessionFiltered()
{
   if(!Session_Filter) return false;
   int hour = TimeHour(TimeGMT());
   if(Session_Skip_Start <= Session_Skip_End)
      return (hour >= Session_Skip_Start && hour < Session_Skip_End);
   else
      return (hour >= Session_Skip_Start || hour < Session_Skip_End);
}

//+------------------------------------------------------------------+
int CheckKeltnerSignal()
{
   double ema_kc = iMA(Symbol(), PERIOD_H1, KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 0);
   double atr_h1 = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   double adx    = iADX(Symbol(), PERIOD_H1, ADX_Period, PRICE_CLOSE, MODE_MAIN, 0);
   double ema100 = iMA(Symbol(), PERIOD_H1, EMA100_Period, 0, MODE_EMA, PRICE_CLOSE, 0);
   double close  = iClose(Symbol(), PERIOD_H1, 0);

   double kc_upper = ema_kc + KC_Multiplier * atr_h1;
   double kc_lower = ema_kc - KC_Multiplier * atr_h1;

   if(adx < ADX_Threshold) return 0;

   double trendScore = CalcTrendScore();
   if(trendScore < ChoppyThreshold) return 0;

   if(close > kc_upper && close > ema100) return 1;
   if(close < kc_lower && close < ema100) return -1;
   return 0;
}

//+------------------------------------------------------------------+
void CheckMaxLossCap()
{
   if(!MaxLoss_Enabled || MaxLoss_USD <= 0) return;
   if(myTicket < 0) return;
   if(OrderSelect(myTicket, SELECT_BY_TICKET, MODE_TRADES) == false) return;
   if(OrderCloseTime() > 0) return;

   double profit = OrderProfit() + OrderCommission() + OrderSwap();
   if(profit < -MaxLoss_USD)
   {
      double price;
      if(OrderType() == OP_BUY)
         price = MarketInfo(Symbol(), MODE_BID);
      else
         price = MarketInfo(Symbol(), MODE_ASK);

      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrRed);
      if(closed)
      {
         Print("MaxLoss Cap! Loss=$", DoubleToStr(MathAbs(profit), 2),
               " > Cap=$", DoubleToStr(MaxLoss_USD, 2));
         RecordPnL(profit);
         barsHeld = 0;
         trailStopPrice = 0;
         extremePrice = 0;
         myTicket = -1;
      }
   }
}

//+------------------------------------------------------------------+
void ManagePosition()
{
   if(myTicket < 0) return;
   if(OrderSelect(myTicket, SELECT_BY_TICKET, MODE_TRADES) == false)
   {
      myTicket = -1;
      return;
   }
   if(OrderCloseTime() > 0)
   {
      double pnl = OrderProfit() + OrderCommission() + OrderSwap();
      RecordPnL(pnl);
      Print("Trade closed by SL/TP. PnL=$", DoubleToStr(pnl, 2));
      barsHeld = 0;
      trailStopPrice = 0;
      extremePrice = 0;
      myTicket = -1;
      return;
   }

   double posOpen = OrderOpenPrice();
   double posSL   = OrderStopLoss();
   double posTP   = OrderTakeProfit();
   int    posType = OrderType();

   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);

   barsHeld++;

   if(posType == OP_BUY)
      extremePrice = MathMax(extremePrice, bid);
   else
      extremePrice = MathMin(extremePrice, ask);

   double currentATR = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(currentATR <= 0) currentATR = entryATR;

   int regime = GetATRRegime(currentATR);
   double trail_act, trail_dist;
   GetRegimeTrailParams(regime, trail_act, trail_dist);

   // L8_BASE: NO TATrail (no time decay on trailing)

   double activateDist = trail_act * entryATR;
   double trailDist    = trail_dist * entryATR;

   if(posType == OP_BUY)
   {
      double floatProfit = bid - posOpen;
      if(floatProfit >= activateDist)
      {
         double newSL = extremePrice - trailDist;
         if(newSL > posSL && newSL < bid)
         {
            bool ok = OrderModify(myTicket, posOpen, newSL, posTP, 0, clrGreen);
            if(ok) trailStopPrice = newSL;
         }
      }
   }
   else
   {
      double floatProfit = posOpen - ask;
      if(floatProfit >= activateDist)
      {
         double newSL = extremePrice + trailDist;
         if(newSL < posSL && newSL > ask)
         {
            bool ok = OrderModify(myTicket, posOpen, newSL, posTP, 0, clrGreen);
            if(ok) trailStopPrice = newSL;
         }
      }
   }

   if(barsHeld >= MaxHold_Bars)
   {
      double price;
      if(posType == OP_BUY)
         price = MarketInfo(Symbol(), MODE_BID);
      else
         price = MarketInfo(Symbol(), MODE_ASK);

      Print("MaxHold reached (", MaxHold_Bars, " bars), closing");
      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrYellow);
      if(closed)
      {
         double pnl = OrderProfit() + OrderCommission() + OrderSwap();
         RecordPnL(pnl);
         barsHeld = 0;
         trailStopPrice = 0;
         extremePrice = 0;
         myTicket = -1;
      }
   }
}

//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            myTicket = OrderTicket();
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
void OnTick()
{
   CheckMaxLossCap();

   datetime currentBar = iTime(Symbol(), PERIOD_M15, 0);
   if(currentBar == lastBarTime) return;
   lastBarTime = currentBar;

   double h1_atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(h1_atr > 0)
   {
      atrHistory[atrHistCount % ATR_HIST_SIZE] = h1_atr;
      atrHistCount++;
   }

   if(HasOpenPosition())
   {
      ManagePosition();
      return;
   }

   if(TimeCurrent() - lastEntryTime < (int)(MinEntryGapHours * 3600))
      return;

   if(IsSessionFiltered())
      return;

   int signal = CheckKeltnerSignal();
   if(signal == 0) return;

   string h1Dir = GetH1KCDirection();
   if(h1Dir != "PASS")
   {
      if(signal == 1 && h1Dir != "BULL") return;
      if(signal == -1 && h1Dir != "BEAR") return;
   }

   if(!IsKCBandwidthExpanding())
      return;

   if(ShouldSkipByEqCurve())
   {
      Print("EqCurve skip");
      return;
   }

   double atr_m15 = iATR(Symbol(), PERIOD_M15, ATR_Period, 0);
   if(atr_m15 < 0.1) return;

   entryATR = atr_m15;
   double sl_dist = atr_m15 * SL_ATR_Mult;
   double tp_dist = atr_m15 * TP_ATR_Mult;

   double lot = LotSize;
   if(EqCurve_Enabled && eqCurveSkip && EqCurve_Red > 0)
      lot = LotSize * EqCurve_Red;

   double ask_price = MarketInfo(Symbol(), MODE_ASK);
   double bid_price = MarketInfo(Symbol(), MODE_BID);

   int ticket = -1;
   if(signal == 1)
   {
      double sl = ask_price - sl_dist;
      double tp = ask_price + tp_dist;
      ticket = OrderSend(Symbol(), OP_BUY, lot, ask_price, MaxSlippage,
                         sl, tp, "L8v3 BUY", MagicNumber, 0, clrBlue);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = bid_price;
         trailStopPrice = 0;
         myTicket = ticket;
         Print("BUY @ ", ask_price, " SL=", sl, " TP=", tp, " ATR=", atr_m15);
      }
   }
   else if(signal == -1)
   {
      double sl = bid_price + sl_dist;
      double tp = bid_price - tp_dist;
      ticket = OrderSend(Symbol(), OP_SELL, lot, bid_price, MaxSlippage,
                         sl, tp, "L8v3 SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = ask_price;
         trailStopPrice = 0;
         myTicket = ticket;
         Print("SELL @ ", bid_price, " SL=", sl, " TP=", tp, " ATR=", atr_m15);
      }
   }
}
//+------------------------------------------------------------------+
