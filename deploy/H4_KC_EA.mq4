//+------------------------------------------------------------------+
//| H4_KC_EA.mq4                                                    |
//| H4 Keltner Channel Breakout                                     |
//| R51: 48,000 combos, K-Fold 50/50 PASS, Sharpe 5.12             |
//| Chart: XAUUSD H4                                                |
//+------------------------------------------------------------------+
//| Best params from R51 brute-force:                                |
//|   KC: EMA=15, Mult=2.5, ATR=14, ADX>=10                        |
//|   Exit: SL=4.5xATR, TP=6.0xATR, MaxHold=50 bars                |
//|   Trail: Act=0.20xATR, Dist=0.04xATR                            |
//| R52 recommended lot: 0.02                                        |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.01"
#property strict

#include "TradeLogger.mqh"

//--- 交易参数
extern double LotSize          = 0.01;       // R56 最优组合推荐
extern int    MagicNumber      = 20250431;
extern int    MaxSlippage       = 30;

//--- Keltner 入场参数 (H4 timeframe)
extern int    KC_EMA_Period     = 15;        // R51 最优
extern double KC_Multiplier     = 2.5;       // R51 最优
extern int    ATR_Period        = 14;
extern double ADX_Threshold     = 10.0;      // R51 最优

//--- 出场参数
extern double SL_ATR_Mult       = 4.5;       // R51 最优
extern double TP_ATR_Mult       = 6.0;       // R51 最优
extern int    MaxHold_Bars      = 50;        // H4 bars = ~8 trading days
extern double Trail_Act_ATR     = 0.20;      // R51 最优
extern double Trail_Dist_ATR    = 0.04;      // R51 最优

//--- 入场间隔
extern double MinEntryGapHours  = 4.0;       // H4级别至少1根bar

//--- Global
datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;

//+------------------------------------------------------------------+
int OnInit()
{
   Print("H4_KC EA v1.0 | Lot=", LotSize,
         " EMA=", KC_EMA_Period, " Mult=", KC_Multiplier,
         " ADX=", ADX_Threshold, " SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " MH=", MaxHold_Bars, " Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int CheckH4Signal()
{
   double ema   = iMA(Symbol(), PERIOD_H4, KC_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   double atr   = iATR(Symbol(), PERIOD_H4, ATR_Period, 1);
   double adx   = iADX(Symbol(), PERIOD_H4, ATR_Period, PRICE_CLOSE, MODE_MAIN, 1);
   double close = iClose(Symbol(), PERIOD_H4, 1);

   double kc_upper = ema + KC_Multiplier * atr;
   double kc_lower = ema - KC_Multiplier * atr;

   if(adx < ADX_Threshold) return 0;
   if(atr < 0.1) return 0;

   if(close > kc_upper) return 1;
   if(close < kc_lower) return -1;
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
      Print("H4_KC closed by SL/TP. PnL=$", DoubleToStr(pnl, 2));
      LogTrade("H4_KC", myTicket, "SL/TP");
      barsHeld = 0; trailStopPrice = 0; extremePrice = 0; myTicket = -1;
      return;
   }

   double posOpen = OrderOpenPrice();
   double posSL   = OrderStopLoss();
   double posTP   = OrderTakeProfit();
   int    posType = OrderType();

   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);

   datetime currentH4 = iTime(Symbol(), PERIOD_H4, 0);
   static datetime lastH4Bar = 0;
   if(currentH4 != lastH4Bar)
   {
      barsHeld++;
      lastH4Bar = currentH4;
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
      Print("H4_KC MaxHold (", MaxHold_Bars, " H4 bars), closing");
      int ticketToLog = myTicket;
      bool closed = OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrYellow);
      if(closed)
      {
         LogTrade("H4_KC", ticketToLog, "Timeout");
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
   datetime currentBar = iTime(Symbol(), PERIOD_H4, 0);
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

   int signal = CheckH4Signal();
   if(signal == 0) return;

   double atr_h4 = iATR(Symbol(), PERIOD_H4, ATR_Period, 1);
   if(atr_h4 < 0.1) return;

   entryATR = atr_h4;
   double sl_dist = atr_h4 * SL_ATR_Mult;
   double tp_dist = atr_h4 * TP_ATR_Mult;

   double ask_price = MarketInfo(Symbol(), MODE_ASK);
   double bid_price = MarketInfo(Symbol(), MODE_BID);

   int ticket = -1;
   if(signal == 1)
   {
      double sl = ask_price - sl_dist;
      double tp = ask_price + tp_dist;
      ticket = OrderSend(Symbol(), OP_BUY, LotSize, ask_price, MaxSlippage,
                         sl, tp, "H4KC BUY", MagicNumber, 0, clrBlue);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0; extremePrice = bid_price;
         trailStopPrice = 0; myTicket = ticket;
         Print("H4_KC BUY @ ", ask_price, " SL=", sl, " TP=", tp, " ATR=", atr_h4);
      }
   }
   else if(signal == -1)
   {
      double sl = bid_price + sl_dist;
      double tp = bid_price - tp_dist;
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bid_price, MaxSlippage,
                         sl, tp, "H4KC SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         lastEntryTime = TimeCurrent();
         barsHeld = 0; extremePrice = ask_price;
         trailStopPrice = 0; myTicket = ticket;
         Print("H4_KC SELL @ ", bid_price, " SL=", sl, " TP=", tp, " ATR=", atr_h4);
      }
   }
}
//+------------------------------------------------------------------+
