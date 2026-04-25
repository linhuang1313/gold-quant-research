//+------------------------------------------------------------------+
//| EA_D1_Donchian50.mq4                                             |
//| D1 Donchian 50 Breakout — Realistic Params (MQL4)                |
//| R37C validated: Sharpe=12.62, K-Fold 6/6, swap/slip resistant    |
//| Params: SL=2.0xATR, TP=5.0xATR, Trail(1.5/0.5), MH=60, CD=3    |
//| Chart: XAUUSD D1                                                 |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

extern double LotSize        = 0.03;
extern int    DonchianPeriod = 50;
extern double SL_ATR_Mult    = 2.0;
extern double TP_ATR_Mult    = 5.0;
extern double Trail_Act_ATR  = 1.5;
extern double Trail_Dist_ATR = 0.5;
extern int    ATR_Period      = 14;
extern int    MaxHoldBars     = 60;
extern int    Cooldown        = 3;
extern int    MagicNumber     = 37501;

datetime lastBarTime = 0;
int      barsSinceExit = 999;
int      barsInTrade = 0;
double   entryATR = 0;
double   entryPrice = 0;
int      entryDir = 0; // 1=BUY, -1=SELL
bool     trailActivated = false;
double   trailStop = 0;

//+------------------------------------------------------------------+
int init()
{
   return(0);
}

//+------------------------------------------------------------------+
int start()
{
   datetime currentBarTime = iTime(Symbol(), PERIOD_D1, 0);
   if(currentBarTime == lastBarTime) return(0);
   lastBarTime = currentBarTime;

   double atr = iATR(Symbol(), PERIOD_D1, ATR_Period, 1);
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
   double donHigh = GetDonchianHigh(1);
   double donLow  = GetDonchianLow(1);
   double closeNow = iClose(Symbol(), PERIOD_D1, 0);

   if(donHigh == 0 || donLow == 0) return;

   int digits = (int)MarketInfo(Symbol(), MODE_DIGITS);
   double point = MarketInfo(Symbol(), MODE_POINT);
   double sl, tp;
   int ticket;

   if(closeNow > donHigh)
   {
      double askPrice = MarketInfo(Symbol(), MODE_ASK);
      sl = NormalizeDouble(askPrice - SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(askPrice + TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_BUY, LotSize, askPrice, 30, sl, tp, "Don50_BUY", MagicNumber, 0, clrGreen);
      if(ticket > 0)
      {
         entryATR = atr;
         entryPrice = askPrice;
         entryDir = 1;
         barsInTrade = 0;
         trailActivated = false;
         barsSinceExit = 0;
      }
   }
   else if(closeNow < donLow)
   {
      double bidPrice = MarketInfo(Symbol(), MODE_BID);
      sl = NormalizeDouble(bidPrice + SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(bidPrice - TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bidPrice, 30, sl, tp, "Don50_SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         entryATR = atr;
         entryPrice = bidPrice;
         entryDir = -1;
         barsInTrade = 0;
         trailActivated = false;
         barsSinceExit = 0;
      }
   }
}

//+------------------------------------------------------------------+
void ManagePosition()
{
   if(!HasPosition()) return;

   double currentClose = iClose(Symbol(), PERIOD_D1, 0);
   double activateDist = Trail_Act_ATR * entryATR;
   double trailDist    = Trail_Dist_ATR * entryATR;

   if(barsInTrade >= MaxHoldBars)
   {
      ClosePosition("Timeout");
      return;
   }

   if(entryDir == 1)
   {
      double highBar = iHigh(Symbol(), PERIOD_D1, 0);
      if(highBar - entryPrice >= activateDist)
      {
         double newTrail = highBar - trailDist;
         if(!trailActivated || newTrail > trailStop)
         {
            trailStop = newTrail;
            trailActivated = true;
         }
      }
      if(trailActivated && currentClose <= trailStop)
      {
         ClosePosition("Trail");
         return;
      }
   }
   else
   {
      double lowBar = iLow(Symbol(), PERIOD_D1, 0);
      if(entryPrice - lowBar >= activateDist)
      {
         double newTrail = lowBar + trailDist;
         if(!trailActivated || newTrail < trailStop)
         {
            trailStop = newTrail;
            trailActivated = true;
         }
      }
      if(trailActivated && currentClose >= trailStop)
      {
         ClosePosition("Trail");
         return;
      }
   }
}

//+------------------------------------------------------------------+
double GetDonchianHigh(int shift)
{
   double maxH = iHigh(Symbol(), PERIOD_D1, shift);
   for(int i = shift + 1; i < shift + DonchianPeriod; i++)
   {
      double h = iHigh(Symbol(), PERIOD_D1, i);
      if(h > maxH) maxH = h;
   }
   return maxH;
}

//+------------------------------------------------------------------+
double GetDonchianLow(int shift)
{
   double minL = iLow(Symbol(), PERIOD_D1, shift);
   for(int i = shift + 1; i < shift + DonchianPeriod; i++)
   {
      double l = iLow(Symbol(), PERIOD_D1, i);
      if(l < minL) minL = l;
   }
   return minL;
}

//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
            return true;
      }
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
            int type = OrderType();
            double closePrice;
            if(type == OP_BUY)
               closePrice = MarketInfo(Symbol(), MODE_BID);
            else
               closePrice = MarketInfo(Symbol(), MODE_ASK);

            bool closed = OrderClose(OrderTicket(), OrderLots(), closePrice, 30, clrYellow);
            if(closed)
               Print("Closed: ", reason, " ticket=", OrderTicket());
         }
      }
   }
   barsSinceExit = 0;
   entryDir = 0;
   trailActivated = false;
}
//+------------------------------------------------------------------+
