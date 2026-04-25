//+------------------------------------------------------------------+
//| EA_H4_ROC5_EMA200.mq5                                            |
//| H4 Rate-of-Change Momentum + EMA200 Trend Filter                 |
//| R37C validated: Sharpe=7.02, K-Fold 6/6                          |
//| Signal: ROC(5) > 1.0 AND Close > EMA200 => BUY                  |
//|         ROC(5) < -1.0 AND Close < EMA200 => SELL                 |
//| Params: SL=2.0xATR, TP=4.0xATR, Trail(1.0/0.3), MH=30, CD=3   |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input double LotSize          = 0.03;
input int    ROC_Period        = 5;
input double ROC_Threshold     = 1.0;   // percent
input int    EMA_Trend_Period  = 200;
input double SL_ATR_Mult      = 2.0;
input double TP_ATR_Mult      = 4.0;
input double Trail_Act_ATR    = 1.0;
input double Trail_Dist_ATR   = 0.3;
input int    ATR_Period        = 14;
input int    MaxHoldBars       = 30;    // H4 bars
input int    Cooldown          = 3;
input int    MagicNumber       = 37503;

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

   double atr = GetATR(1);
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
double GetATR(int shift)
{
   double buf[];
   int handle = iATR(_Symbol, PERIOD_H4, ATR_Period);
   if(handle == INVALID_HANDLE) return 0;
   ArraySetAsSeries(buf, true);
   CopyBuffer(handle, 0, shift, 1, buf);
   IndicatorRelease(handle);
   return buf[0];
}

//+------------------------------------------------------------------+
double GetEMA(int period, int shift)
{
   double buf[];
   int handle = iMA(_Symbol, PERIOD_H4, period, 0, MODE_EMA, PRICE_CLOSE);
   if(handle == INVALID_HANDLE) return 0;
   ArraySetAsSeries(buf, true);
   CopyBuffer(handle, 0, shift, 1, buf);
   IndicatorRelease(handle);
   return buf[0];
}

//+------------------------------------------------------------------+
double GetROC(int shift)
{
   double closeNow = iClose(_Symbol, PERIOD_H4, shift);
   double closePrev = iClose(_Symbol, PERIOD_H4, shift + ROC_Period);
   if(closePrev == 0) return 0;
   return (closeNow - closePrev) / closePrev * 100.0;
}

//+------------------------------------------------------------------+
void CheckEntry(double atr)
{
   double closeNow = iClose(_Symbol, PERIOD_H4, 0);
   double ema200   = GetEMA(EMA_Trend_Period, 0);
   double roc      = GetROC(0);

   if(ema200 == 0) return;

   double sl, tp;

   if(roc > ROC_Threshold && closeNow > ema200)
   {
      sl = closeNow - SL_ATR_Mult * atr;
      tp = closeNow + TP_ATR_Mult * atr;
      if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "ROC5_BUY"))
      {
         entryATR = atr;
         entryPrice = closeNow;
         entryDir = 1;
         barsInTrade = 0;
         trailActivated = false;
         barsSinceExit = 0;
      }
   }
   else if(roc < -ROC_Threshold && closeNow < ema200)
   {
      sl = closeNow + SL_ATR_Mult * atr;
      tp = closeNow - TP_ATR_Mult * atr;
      if(trade.Sell(LotSize, _Symbol, 0, sl, tp, "ROC5_SELL"))
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
