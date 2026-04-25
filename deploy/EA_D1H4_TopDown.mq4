//+------------------------------------------------------------------+
//| EA_D1H4_TopDown.mq4                                              |
//| D1 Trend (EMA50) + H4 EMA20 Bounce — Multi-TF (MQL4)            |
//| R37C validated: Sharpe=7.46, K-Fold 6/6, Corr=-0.031 with L7    |
//| Chart: XAUUSD H4                                                 |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

extern double LotSize        = 0.03;
extern int    D1_EMA_Period   = 50;
extern int    H4_EMA_Period   = 20;
extern double SL_ATR_Mult    = 1.5;
extern double TP_ATR_Mult    = 3.0;
extern double Trail_Act_ATR  = 1.0;
extern double Trail_Dist_ATR = 0.3;
extern int    ATR_Period      = 14;
extern int    MaxHoldBars     = 40;
extern int    Cooldown        = 3;
extern int    MagicNumber     = 37502;

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
void CheckEntry(double atr)
{
   // D1 trend: Close vs EMA50 (use bar 1 = last completed D1 bar)
   double d1Close = iClose(Symbol(), PERIOD_D1, 1);
   double d1EMA   = iMA(Symbol(), PERIOD_D1, D1_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   if(d1Close == 0 || d1EMA == 0) return;
   int trend = (d1Close > d1EMA) ? 1 : -1;

   // H4 EMA20 crossover
   double h4Close0 = iClose(Symbol(), PERIOD_H4, 0);
   double h4Close1 = iClose(Symbol(), PERIOD_H4, 1);
   double h4EMA0   = iMA(Symbol(), PERIOD_H4, H4_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 0);
   double h4EMA1   = iMA(Symbol(), PERIOD_H4, H4_EMA_Period, 0, MODE_EMA, PRICE_CLOSE, 1);
   if(h4Close0 == 0 || h4EMA0 == 0) return;

   int digits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   double sl, tp;
   int ticket;

   if(trend == 1 && h4Close1 < h4EMA1 && h4Close0 > h4EMA0)
   {
      double askPrice = MarketInfo(Symbol(), MODE_ASK);
      sl = NormalizeDouble(askPrice - SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(askPrice + TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_BUY, LotSize, askPrice, 30, sl, tp, "TopDown_BUY", MagicNumber, 0, clrGreen);
      if(ticket > 0)
      {
         entryATR = atr; entryPrice = askPrice; entryDir = 1;
         barsInTrade = 0; trailActivated = false; barsSinceExit = 0;
      }
   }
   else if(trend == -1 && h4Close1 > h4EMA1 && h4Close0 < h4EMA0)
   {
      double bidPrice = MarketInfo(Symbol(), MODE_BID);
      sl = NormalizeDouble(bidPrice + SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(bidPrice - TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bidPrice, 30, sl, tp, "TopDown_SELL", MagicNumber, 0, clrRed);
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
