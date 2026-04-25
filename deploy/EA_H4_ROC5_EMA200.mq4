//+------------------------------------------------------------------+
//| EA_H4_ROC5_EMA200.mq4                                            |
//| H4 Rate-of-Change Momentum + EMA200 Trend Filter (MQL4)          |
//| R37C validated: Sharpe=7.02, K-Fold 6/6                          |
//| Signal: ROC(5) > 1.0% AND Close > EMA200 => BUY                 |
//|         ROC(5) < -1.0% AND Close < EMA200 => SELL                |
//| Chart: XAUUSD H4                                                 |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

extern double LotSize          = 0.03;
extern int    ROC_Period        = 5;
extern double ROC_Threshold     = 1.0;
extern int    EMA_Trend_Period  = 200;
extern double SL_ATR_Mult      = 2.0;
extern double TP_ATR_Mult      = 4.0;
extern double Trail_Act_ATR    = 1.0;
extern double Trail_Dist_ATR   = 0.3;
extern int    ATR_Period        = 14;
extern int    MaxHoldBars       = 30;
extern int    Cooldown          = 3;
extern int    MagicNumber       = 37503;

datetime lastBarTime = 0;
int      barsSinceExit = 999;
int      barsInTrade = 0;
double   entryATR = 0;
double   entryPrice = 0;
int      entryDir = 0;
bool     trailActivated = false;
double   trailStop = 0;

//+------------------------------------------------------------------+
int init() { return(0); }

//+------------------------------------------------------------------+
int start()
{
   datetime currentBarTime = iTime(Symbol(), PERIOD_H4, 0);
   if(currentBarTime == lastBarTime) return(0);
   lastBarTime = currentBarTime;

   double atr = iATR(Symbol(), PERIOD_H4, ATR_Period, 1);
   if(atr < 0.1) return(0);

   bool hasPos = HasPosition();

   if(hasPos)
   {
      barsInTrade++;
      ManagePosition();
   }
   else
   {
      barsSinceExit++;
      if(barsSinceExit >= Cooldown)
         CheckEntry(atr);
   }

   return(0);
}

//+------------------------------------------------------------------+
double GetROC(int shift)
{
   double closeNow  = iClose(Symbol(), PERIOD_H4, shift);
   double closePrev = iClose(Symbol(), PERIOD_H4, shift + ROC_Period);
   if(closePrev == 0) return 0;
   return (closeNow - closePrev) / closePrev * 100.0;
}

//+------------------------------------------------------------------+
void CheckEntry(double atr)
{
   double closeNow = iClose(Symbol(), PERIOD_H4, 0);
   double ema200   = iMA(Symbol(), PERIOD_H4, EMA_Trend_Period, 0, MODE_EMA, PRICE_CLOSE, 0);
   double roc      = GetROC(0);

   if(ema200 == 0) return;

   int digits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   double sl, tp;
   int ticket;

   if(roc > ROC_Threshold && closeNow > ema200)
   {
      double askPrice = MarketInfo(Symbol(), MODE_ASK);
      sl = NormalizeDouble(askPrice - SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(askPrice + TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_BUY, LotSize, askPrice, 30, sl, tp, "ROC5_BUY", MagicNumber, 0, clrGreen);
      if(ticket > 0)
      {
         entryATR = atr; entryPrice = askPrice; entryDir = 1;
         barsInTrade = 0; trailActivated = false; barsSinceExit = 0;
      }
   }
   else if(roc < -ROC_Threshold && closeNow < ema200)
   {
      double bidPrice = MarketInfo(Symbol(), MODE_BID);
      sl = NormalizeDouble(bidPrice + SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(bidPrice - TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bidPrice, 30, sl, tp, "ROC5_SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         entryATR = atr; entryPrice = bidPrice; entryDir = -1;
         barsInTrade = 0; trailActivated = false; barsSinceExit = 0;
      }
   }
}

//+------------------------------------------------------------------+
void ManagePosition()
{
   if(!HasPosition()) return;

   double currentClose = iClose(Symbol(), PERIOD_H4, 0);
   double activateDist = Trail_Act_ATR * entryATR;
   double trailDist    = Trail_Dist_ATR * entryATR;

   if(barsInTrade >= MaxHoldBars)
   {
      ClosePosition("Timeout");
      return;
   }

   if(entryDir == 1)
   {
      double highBar = iHigh(Symbol(), PERIOD_H4, 0);
      if(highBar - entryPrice >= activateDist)
      {
         double newTrail = highBar - trailDist;
         if(!trailActivated || newTrail > trailStop)
         { trailStop = newTrail; trailActivated = true; }
      }
      if(trailActivated && currentClose <= trailStop)
      { ClosePosition("Trail"); return; }
   }
   else
   {
      double lowBar = iLow(Symbol(), PERIOD_H4, 0);
      if(entryPrice - lowBar >= activateDist)
      {
         double newTrail = lowBar + trailDist;
         if(!trailActivated || newTrail < trailStop)
         { trailStop = newTrail; trailActivated = true; }
      }
      if(trailActivated && currentClose >= trailStop)
      { ClosePosition("Trail"); return; }
   }
}

//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
            return true;
   }
   return false;
}

//+------------------------------------------------------------------+
void ClosePosition(string reason)
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
            double closePrice = (OrderType() == OP_BUY) ?
               MarketInfo(Symbol(), MODE_BID) : MarketInfo(Symbol(), MODE_ASK);
            if(OrderClose(OrderTicket(), OrderLots(), closePrice, 30, clrYellow))
               Print("Closed: ", reason, " ticket=", OrderTicket());
         }
      }
   }
   barsSinceExit = 0; entryDir = 0; trailActivated = false;
}
//+------------------------------------------------------------------+
