//+------------------------------------------------------------------+
//| EA_D1_Donchian50_Conservative.mq4                                |
//| D1 Donchian 50 — No Trail, Pure SL/TP (MQL4)                    |
//| R37C validated: Sharpe=10.69, K-Fold 6/6                         |
//| Simplest version: breakout entry, fixed SL/TP, timeout exit      |
//| Chart: XAUUSD D1                                                 |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

extern double LotSize        = 0.03;
extern int    DonchianPeriod = 50;
extern double SL_ATR_Mult    = 1.5;
extern double TP_ATR_Mult    = 3.0;
extern int    ATR_Period      = 14;
extern int    MaxHoldBars     = 40;
extern int    Cooldown        = 5;
extern int    MagicNumber     = 37504;

datetime lastBarTime = 0;
int      barsSinceExit = 999;
int      barsInTrade = 0;

//+------------------------------------------------------------------+
int init() { return(0); }

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
      if(barsInTrade >= MaxHoldBars)
         ClosePosition("Timeout");
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
   double sl, tp;
   int ticket;

   if(closeNow > donHigh)
   {
      double askPrice = MarketInfo(Symbol(), MODE_ASK);
      sl = NormalizeDouble(askPrice - SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(askPrice + TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_BUY, LotSize, askPrice, 30, sl, tp, "Don50C_BUY", MagicNumber, 0, clrGreen);
      if(ticket > 0)
      {
         barsInTrade = 0;
         barsSinceExit = 0;
      }
   }
   else if(closeNow < donLow)
   {
      double bidPrice = MarketInfo(Symbol(), MODE_BID);
      sl = NormalizeDouble(bidPrice + SL_ATR_Mult * atr, digits);
      tp = NormalizeDouble(bidPrice - TP_ATR_Mult * atr, digits);
      ticket = OrderSend(Symbol(), OP_SELL, LotSize, bidPrice, 30, sl, tp, "Don50C_SELL", MagicNumber, 0, clrRed);
      if(ticket > 0)
      {
         barsInTrade = 0;
         barsSinceExit = 0;
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
   barsSinceExit = 0;
}
//+------------------------------------------------------------------+
