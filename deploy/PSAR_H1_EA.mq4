//+------------------------------------------------------------------+
//| PSAR_H1_EA.mq4                                                  |
//| Parabolic SAR H1 Direction Flip                                  |
//| R51: 6,000 combos, K-Fold 50/50 PASS, Sharpe 4.13              |
//| Chart: XAUUSD H1                                                |
//+------------------------------------------------------------------+
//| Best params from R51 brute-force:                                |
//|   PSAR: AF_Start=0.01, AF_Max=0.05                              |
//|   Exit: SL=4.5xATR, TP=16.0xATR, MaxHold=20 bars               |
//|   Trail: Act=0.20xATR, Dist=0.04xATR                            |
//| R52 recommended lot: 0.05                                        |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.01"
#property strict

#include "TradeLogger.mqh"

//--- 交易参数
extern double LotSize          = 0.03;       // R56 最优组合推荐
extern int    MagicNumber      = 20250432;
extern int    MaxSlippage       = 30;

//--- PSAR 参数 (H1 timeframe)
extern double PSAR_AF_Start     = 0.01;      // R51 最优
extern double PSAR_AF_Max       = 0.05;      // R51 最优
extern int    ATR_Period        = 14;

//--- 出场参数
extern double SL_ATR_Mult       = 4.5;       // R51 最优
extern double TP_ATR_Mult       = 16.0;      // R51 最优
extern int    MaxHold_Bars      = 20;        // H1 bars
extern double Trail_Act_ATR     = 0.20;      // R51 最优
extern double Trail_Dist_ATR    = 0.04;      // R51 最优

//--- 入场间隔 (H1 bar 间隔 ≥ 2, 对应回测 last_exit gap)
extern double MinEntryGapHours  = 2.0;

//--- Global
datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;
int      prevPSARDir    = 0;     // 1=bullish, -1=bearish, 0=unknown

//+------------------------------------------------------------------+
int OnInit()
{
   Print("PSAR_H1 EA v1.0 | Lot=", LotSize,
         " AF=", PSAR_AF_Start, "/", PSAR_AF_Max,
         " SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " MH=", MaxHold_Bars, " Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR);

   double sar_prev = iSAR(Symbol(), PERIOD_H1, PSAR_AF_Start, PSAR_AF_Max, 2);
   double close_prev = iClose(Symbol(), PERIOD_H1, 2);
   prevPSARDir = (close_prev > sar_prev) ? 1 : -1;

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int GetPSARDirection(int shift)
{
   double sar   = iSAR(Symbol(), PERIOD_H1, PSAR_AF_Start, PSAR_AF_Max, shift);
   double close = iClose(Symbol(), PERIOD_H1, shift);
   return (close > sar) ? 1 : -1;
}

//+------------------------------------------------------------------+
int CheckPSARSignal()
{
   int curDir  = GetPSARDirection(1);
   int prevDir = GetPSARDirection(2);

   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 1);
   if(atr < 0.1) return 0;

   if(prevDir == -1 && curDir == 1)  return 1;   // flip to bullish
   if(prevDir == 1  && curDir == -1) return -1;   // flip to bearish
   return 0;
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
      Print("PSAR closed by SL/TP. PnL=$", DoubleToStr(pnl, 2));
      LogTrade("PSAR_H1", myTicket, "SL/TP");
      barsHeld = 0; trailStopPrice = 0; extremePrice = 0; myTicket = -1;
      return;
   }

   double posOpen = OrderOpenPrice();
   double posSL   = OrderStopLoss();
   double posTP   = OrderTakeProfit();
   int    posType = OrderType();

   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);

   datetime currentH1 = iTime(Symbol(), PERIOD_H1, 0);
   static datetime lastH1Bar = 0;
   if(currentH1 != lastH1Bar)
   {
      barsHeld++;
      lastH1Bar = currentH1;
   }

   if(posType == OP_BUY)
      extremePrice = MathMax(extremePrice, bid);
   else
      extremePrice = MathMin(extremePrice, ask);

   double activateDist = Trail_Act_ATR * entryATR;
   double trailDist    = Trail_Dist_ATR * entryATR;

   if(posType == OP_BUY)
   {
      if(bid - posOpen >= activateDist)
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
      if(posOpen - ask >= activateDist)
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
      double price = (posType == OP_BUY) ? bid : ask;
      Print("PSAR MaxHold (", MaxHold_Bars, " H1 bars), closing");
      int ticketToLog = myTicket;
      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrYellow);
      if(closed)
      {
         LogTrade("PSAR_H1", ticketToLog, "Timeout");
         barsHeld = 0; trailStopPrice = 0; extremePrice = 0; myTicket = -1;
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
   datetime currentBar = iTime(Symbol(), PERIOD_H1, 0);
   if(currentBar == lastBarTime)
   {
      if(HasOpenPosition()) ManagePosition();
      return;
   }
   lastBarTime = currentBar;

   if(HasOpenPosition())
   {
      ManagePosition();
      return;
   }

   if(TimeCurrent() - lastEntryTime < (int)(MinEntryGapHours * 3600))
      return;

   int signal = CheckPSARSignal();
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
                         sl, tp, "PSAR BUY", MagicNumber, 0, clrBlue);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0; extremePrice = bid_price;
         trailStopPrice = 0; myTicket = ticket;
         Print("PSAR BUY @ ", ask_price, " SL=", sl, " TP=", tp, " ATR=", atr_h1);
      }
   }
   else if(signal == -1)
   {
      double sl = bid_price + sl_dist;
      double tp = bid_price - tp_dist;
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bid_price, MaxSlippage,
                         sl, tp, "PSAR SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0; extremePrice = ask_price;
         trailStopPrice = 0; myTicket = ticket;
         Print("PSAR SELL @ ", bid_price, " SL=", sl, " TP=", tp, " ATR=", atr_h1);
      }
   }
}
//+------------------------------------------------------------------+
