//+------------------------------------------------------------------+
//| EA_D1_Donchian50_Conservative.mq5                                |
//| D1 Donchian 50 — No Trailing Stop, Pure SL/TP                    |
//| R37C validated: Sharpe=10.69, K-Fold 6/6                         |
//| The simplest version: breakout entry, fixed SL/TP exit           |
//| Params: SL=1.5xATR, TP=3.0xATR, no trail, MH=40, CD=5          |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input double LotSize          = 0.03;
input int    DonchianPeriod   = 50;
input double SL_ATR_Mult      = 1.5;
input double TP_ATR_Mult      = 3.0;
input int    ATR_Period        = 14;
input int    MaxHoldBars       = 40;    // D1 bars
input int    Cooldown          = 5;     // D1 bars after exit
input int    MagicNumber       = 37504;

CTrade trade;

datetime lastBarTime = 0;
int      barsSinceExit = 999;
int      barsInTrade = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime currentBarTime = iTime(_Symbol, PERIOD_D1, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   double atr = GetATR(1);
   if(atr < 0.1) return;

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
}

//+------------------------------------------------------------------+
double GetATR(int shift)
{
   double buf[];
   int handle = iATR(_Symbol, PERIOD_D1, ATR_Period);
   if(handle == INVALID_HANDLE) return 0;
   ArraySetAsSeries(buf, true);
   CopyBuffer(handle, 0, shift, 1, buf);
   IndicatorRelease(handle);
   return buf[0];
}

//+------------------------------------------------------------------+
void CheckEntry(double atr)
{
   double donHigh = GetDonchianHigh(1);
   double donLow  = GetDonchianLow(1);
   double closeNow = iClose(_Symbol, PERIOD_D1, 0);

   if(donHigh == 0 || donLow == 0) return;

   double sl, tp;

   if(closeNow > donHigh)
   {
      sl = closeNow - SL_ATR_Mult * atr;
      tp = closeNow + TP_ATR_Mult * atr;
      if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "Don50C_BUY"))
      {
         barsInTrade = 0;
         barsSinceExit = 0;
      }
   }
   else if(closeNow < donLow)
   {
      sl = closeNow + SL_ATR_Mult * atr;
      tp = closeNow - TP_ATR_Mult * atr;
      if(trade.Sell(LotSize, _Symbol, 0, sl, tp, "Don50C_SELL"))
      {
         barsInTrade = 0;
         barsSinceExit = 0;
      }
   }
}

//+------------------------------------------------------------------+
double GetDonchianHigh(int shift)
{
   double highs[];
   ArraySetAsSeries(highs, true);
   if(CopyHigh(_Symbol, PERIOD_D1, shift, DonchianPeriod, highs) < DonchianPeriod)
      return 0;
   double maxH = highs[0];
   for(int i = 1; i < DonchianPeriod; i++)
      if(highs[i] > maxH) maxH = highs[i];
   return maxH;
}

//+------------------------------------------------------------------+
double GetDonchianLow(int shift)
{
   double lows[];
   ArraySetAsSeries(lows, true);
   if(CopyLow(_Symbol, PERIOD_D1, shift, DonchianPeriod, lows) < DonchianPeriod)
      return 0;
   double minL = lows[0];
   for(int i = 1; i < DonchianPeriod; i++)
      if(lows[i] < minL) minL = lows[i];
   return minL;
}

//+------------------------------------------------------------------+
bool HasPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
void ClosePosition(string reason)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == MagicNumber)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         trade.PositionClose(ticket);
         Print("Closed: ", reason, " ticket=", ticket);
      }
   }
   barsSinceExit = 0;
}
//+------------------------------------------------------------------+
