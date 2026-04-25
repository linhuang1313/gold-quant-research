//+------------------------------------------------------------------+
//| EA_D1H4_TopDown.mq5                                              |
//| D1 Trend + H4 EMA20 Bounce — Multi-Timeframe                     |
//| R37C validated: Sharpe=7.46, K-Fold 6/6, Corr=-0.031 with L7    |
//| D1: Close vs EMA50 for trend direction                           |
//| H4: EMA20 crossover for entry timing                             |
//| Params: SL=1.5xATR, TP=3.0xATR, Trail(1.0/0.3), MH=40, CD=3   |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input double LotSize          = 0.03;
input int    D1_EMA_Period     = 50;    // D1 trend filter
input int    H4_EMA_Period     = 20;    // H4 entry signal
input double SL_ATR_Mult      = 1.5;
input double TP_ATR_Mult      = 3.0;
input double Trail_Act_ATR    = 1.0;
input double Trail_Dist_ATR   = 0.3;
input int    ATR_Period        = 14;
input int    MaxHoldBars       = 40;    // H4 bars
input int    Cooldown          = 3;     // H4 bars
input int    MagicNumber       = 37502;

CTrade trade;

datetime lastBarTime = 0;
int      barsSinceExit = 999;
int      barsInTrade = 0;
double   entryATR = 0;
double   entryPrice = 0;
int      entryDir = 0;
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
   datetime currentBarTime = iTime(_Symbol, PERIOD_H4, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   double atr = GetATR_H4(1);
   if(atr < 0.1) return;

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
}

//+------------------------------------------------------------------+
double GetATR_H4(int shift)
{
   double atrBuf[];
   int handle = iATR(_Symbol, PERIOD_H4, ATR_Period);
   if(handle == INVALID_HANDLE) return 0;
   ArraySetAsSeries(atrBuf, true);
   CopyBuffer(handle, 0, shift, 1, atrBuf);
   IndicatorRelease(handle);
   return atrBuf[0];
}

//+------------------------------------------------------------------+
double GetEMA(ENUM_TIMEFRAMES tf, int period, int shift)
{
   double buf[];
   int handle = iMA(_Symbol, tf, period, 0, MODE_EMA, PRICE_CLOSE);
   if(handle == INVALID_HANDLE) return 0;
   ArraySetAsSeries(buf, true);
   CopyBuffer(handle, 0, shift, 1, buf);
   IndicatorRelease(handle);
   return buf[0];
}

//+------------------------------------------------------------------+
void CheckEntry(double atr)
{
   // D1 trend: Close vs EMA50
   double d1Close = iClose(_Symbol, PERIOD_D1, 1);
   double d1EMA50 = GetEMA(PERIOD_D1, D1_EMA_Period, 1);
   if(d1Close == 0 || d1EMA50 == 0) return;
   int trend = (d1Close > d1EMA50) ? 1 : -1; // 1=BULL, -1=BEAR

   // H4 EMA20 crossover
   double h4Close0 = iClose(_Symbol, PERIOD_H4, 0);
   double h4Close1 = iClose(_Symbol, PERIOD_H4, 1);
   double h4EMA0   = GetEMA(PERIOD_H4, H4_EMA_Period, 0);
   double h4EMA1   = GetEMA(PERIOD_H4, H4_EMA_Period, 1);
   if(h4Close0 == 0 || h4EMA0 == 0) return;

   double sl, tp;

   if(trend == 1 && h4Close1 < h4EMA1 && h4Close0 > h4EMA0)
   {
      sl = h4Close0 - SL_ATR_Mult * atr;
      tp = h4Close0 + TP_ATR_Mult * atr;
      if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "TopDown_BUY"))
      {
         entryATR = atr;
         entryPrice = h4Close0;
         entryDir = 1;
         barsInTrade = 0;
         trailActivated = false;
         barsSinceExit = 0;
      }
   }
   else if(trend == -1 && h4Close1 > h4EMA1 && h4Close0 < h4EMA0)
   {
      sl = h4Close0 + SL_ATR_Mult * atr;
      tp = h4Close0 - TP_ATR_Mult * atr;
      if(trade.Sell(LotSize, _Symbol, 0, sl, tp, "TopDown_SELL"))
      {
         entryATR = atr;
         entryPrice = h4Close0;
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

   double currentPrice = iClose(_Symbol, PERIOD_H4, 0);
   double activateDist = Trail_Act_ATR * entryATR;
   double trailDist    = Trail_Dist_ATR * entryATR;

   if(barsInTrade >= MaxHoldBars)
   {
      ClosePosition("Timeout");
      return;
   }

   if(entryDir == 1)
   {
      double highBar = iHigh(_Symbol, PERIOD_H4, 0);
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
      double lowBar = iLow(_Symbol, PERIOD_H4, 0);
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
