//+------------------------------------------------------------------+
//| EA_D1_Donchian50.mq5                                             |
//| D1 Donchian 50 Breakout — Realistic Params                       |
//| R37C validated: Sharpe=12.62, K-Fold 6/6, swap/slip resistant    |
//| Params: SL=2.0xATR, TP=5.0xATR, Trail(1.5/0.5), MH=60, CD=3    |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input double LotSize          = 0.03;
input int    DonchianPeriod   = 50;
input double SL_ATR_Mult      = 2.0;
input double TP_ATR_Mult      = 5.0;
input double Trail_Act_ATR    = 1.5;   // Trail activates at 1.5x ATR
input double Trail_Dist_ATR   = 0.5;   // Trail distance 0.5x ATR
input int    ATR_Period        = 14;
input int    MaxHoldBars       = 60;    // D1 bars
input int    Cooldown          = 3;     // D1 bars after exit
input int    MagicNumber       = 37501;

CTrade trade;

datetime lastBarTime = 0;
int      barsSinceExit = 999;
int      barsInTrade = 0;
double   entryATR = 0;
double   entryPrice = 0;
int      entryDir = 0; // 1=BUY, -1=SELL
bool     trailActivated = false;
double   trailStop = 0;

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
      ManagePosition(atr);
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
   double atrBuf[];
   int handle = iATR(_Symbol, PERIOD_D1, ATR_Period);
   if(handle == INVALID_HANDLE) return 0;
   ArraySetAsSeries(atrBuf, true);
   CopyBuffer(handle, 0, shift, 1, atrBuf);
   IndicatorRelease(handle);
   return atrBuf[0];
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
      if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "Don50_BUY"))
      {
         entryATR = atr;
         entryPrice = closeNow;
         entryDir = 1;
         barsInTrade = 0;
         trailActivated = false;
         barsSinceExit = 0;
      }
   }
   else if(closeNow < donLow)
   {
      sl = closeNow + SL_ATR_Mult * atr;
      tp = closeNow - TP_ATR_Mult * atr;
      if(trade.Sell(LotSize, _Symbol, 0, sl, tp, "Don50_SELL"))
      {
         entryATR = atr;
         entryPrice = closeNow;
         entryDir = -1;
         barsInTrade = 0;
         trailActivated = false;
         barsSinceExit = 0;
      }
   }
}

//+------------------------------------------------------------------+
void ManagePosition(double atr)
{
   if(!HasPosition()) return;

   double currentPrice = iClose(_Symbol, PERIOD_D1, 0);
   double activateDist = Trail_Act_ATR * entryATR;
   double trailDist    = Trail_Dist_ATR * entryATR;

   // MaxHold timeout
   if(barsInTrade >= MaxHoldBars)
   {
      ClosePosition("Timeout");
      return;
   }

   // Trailing stop logic
   if(entryDir == 1)
   {
      double highBar = iHigh(_Symbol, PERIOD_D1, 0);
      if(highBar - entryPrice >= activateDist)
      {
         double newTrail = highBar - trailDist;
         if(!trailActivated || newTrail > trailStop)
         {
            trailStop = newTrail;
            trailActivated = true;
         }
      }
      if(trailActivated && currentPrice <= trailStop)
      {
         ClosePosition("Trail");
         return;
      }
   }
   else
   {
      double lowBar = iLow(_Symbol, PERIOD_D1, 0);
      if(entryPrice - lowBar >= activateDist)
      {
         double newTrail = lowBar + trailDist;
         if(!trailActivated || newTrail < trailStop)
         {
            trailStop = newTrail;
            trailActivated = true;
         }
      }
      if(trailActivated && currentPrice >= trailStop)
      {
         ClosePosition("Trail");
         return;
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
   entryDir = 0;
   trailActivated = false;
}
//+------------------------------------------------------------------+
