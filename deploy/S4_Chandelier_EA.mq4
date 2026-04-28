//+------------------------------------------------------------------+
//| S4_Chandelier_EA.mq4                                             |
//| Chandelier Exit Flip — R45/R46 验证通过                           |
//| K-Fold 6/6, mean=6.35, min=6.02, L8相关性 -0.021                |
//| Chart: XAUUSD H1                                                 |
//+------------------------------------------------------------------+
//| 策略逻辑:                                                         |
//|   Chandelier Exit 反用为入场信号                                   |
//|   BUY: Close 从下穿越 (HH - mult*ATR) 线                         |
//|   SELL: Close 从上穿越 (LL + mult*ATR) 线                         |
//|   本质: 趋势反转 + 止损猎杀                                        |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

//--- 交易参数
extern double LotSize          = 0.01;     // 模拟盘先用最小手
extern int    MagicNumber      = 20260428; // 独立 Magic, 不与 L8 冲突
extern int    MaxSlippage      = 30;

//--- Chandelier 入场参数
extern int    CH_Period        = 10;       // HH/LL 回看周期
extern double CH_Mult          = 3.0;      // ATR 乘数
extern int    ATR_Period       = 14;

//--- 出场参数
extern double SL_ATR_Mult      = 3.0;
extern double TP_ATR_Mult      = 8.0;
extern int    MaxHold_Bars     = 20;
extern double Trail_Act_ATR    = 0.28;     // 追踪激活: 浮盈 >= 0.28*ATR
extern double Trail_Dist_ATR   = 0.06;     // 追踪距离: 0.06*ATR

//--- Cap (灾难保护)
extern bool   MaxLoss_Enabled  = true;
extern double MaxLoss_USD      = 80.0;     // R43 结论: Cap80

//--- 入场间隔
extern double MinEntryGapHours = 1.0;

//--- Global
datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;

double   prevChandLong  = 0;
double   prevChandShort = 0;
bool     prevAboveLong  = false;
bool     prevBelowShort = false;
bool     firstBar       = true;

//+------------------------------------------------------------------+
int OnInit()
{
   Print("S4 Chandelier EA v1.0 | Lot=", LotSize,
         " Period=", CH_Period, " Mult=", CH_Mult,
         " SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " MH=", MaxHold_Bars,
         " Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " Cap=", MaxLoss_USD);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void CalcChandelier(double &chandLong, double &chandShort)
{
   double hh = 0, ll = 999999;
   for(int i = 1; i <= CH_Period; i++)
   {
      double h = iHigh(Symbol(), PERIOD_H1, i);
      double l = iLow(Symbol(), PERIOD_H1, i);
      if(h > hh) hh = h;
      if(l < ll) ll = l;
   }
   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 1);
   chandLong  = hh - CH_Mult * atr;
   chandShort = ll + CH_Mult * atr;
}

//+------------------------------------------------------------------+
int CheckChandelierSignal()
{
   double chandLong, chandShort;
   CalcChandelier(chandLong, chandShort);

   double close = iClose(Symbol(), PERIOD_H1, 0);

   bool aboveLong  = (close > chandLong);
   bool belowShort = (close < chandShort);

   int signal = 0;

   if(!firstBar)
   {
      // Flip: 从下方穿越到上方 = BUY
      if(aboveLong && !prevAboveLong)
         signal = 1;
      // Flip: 从上方穿越到下方 = SELL
      if(belowShort && !prevBelowShort)
         signal = -1;
   }

   prevAboveLong  = aboveLong;
   prevBelowShort = belowShort;
   prevChandLong  = chandLong;
   prevChandShort = chandShort;
   firstBar = false;

   return signal;
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
         Print("CH MaxLoss Cap! Loss=$", DoubleToStr(MathAbs(profit), 2),
               " > Cap=$", DoubleToStr(MaxLoss_USD, 2));
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
      Print("CH trade closed by SL/TP. PnL=$", DoubleToStr(pnl, 2));
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

   double activateDist = Trail_Act_ATR * entryATR;
   double trailDist    = Trail_Dist_ATR * entryATR;

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

      Print("CH MaxHold (", MaxHold_Bars, " bars), closing");
      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrYellow);
      if(closed)
      {
         double pnl = OrderProfit() + OrderCommission() + OrderSwap();
         Print("CH MaxHold close PnL=$", DoubleToStr(pnl, 2));
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

   // H1 bar 驱动 (信号在 H1 上计算)
   datetime currentBar = iTime(Symbol(), PERIOD_H1, 0);
   if(currentBar == lastBarTime) return;
   lastBarTime = currentBar;

   if(HasOpenPosition())
   {
      ManagePosition();
      return;
   }

   if(TimeCurrent() - lastEntryTime < (int)(MinEntryGapHours * 3600))
      return;

   int signal = CheckChandelierSignal();
   if(signal == 0) return;

   double atr_h1 = iATR(Symbol(), PERIOD_H1, ATR_Period, 1);
   if(atr_h1 < 0.1) return;

   entryATR = atr_h1;
   double sl_dist = atr_h1 * SL_ATR_Mult;
   double tp_dist = atr_h1 * TP_ATR_Mult;

   double ask_price = MarketInfo(Symbol(), MODE_ASK);
   double bid_price = MarketInfo(Symbol(), MODE_BID);

   int ticket = -1;
   if(signal == 1)
   {
      double sl = ask_price - sl_dist;
      double tp = ask_price + tp_dist;
      ticket = OrderSend(Symbol(), OP_BUY, LotSize, ask_price, MaxSlippage,
                         sl, tp, "CH_v1 BUY", MagicNumber, 0, clrDodgerBlue);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = bid_price;
         trailStopPrice = 0;
         myTicket = ticket;
         Print("CH BUY @ ", ask_price, " SL=", sl, " TP=", tp, " ATR=", atr_h1);
      }
   }
   else if(signal == -1)
   {
      double sl = bid_price + sl_dist;
      double tp = bid_price - tp_dist;
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bid_price, MaxSlippage,
                         sl, tp, "CH_v1 SELL", MagicNumber, 0, clrOrangeRed);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0;
         extremePrice = ask_price;
         trailStopPrice = 0;
         myTicket = ticket;
         Print("CH SELL @ ", bid_price, " SL=", sl, " TP=", tp, " ATR=", atr_h1);
      }
   }
}
//+------------------------------------------------------------------+
