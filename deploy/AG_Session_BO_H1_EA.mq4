//+------------------------------------------------------------------+
//| AG_Session_BO_H1_EA.mq4                                          |
//| XAGUSD H1 Session Breakout - NY Peak (12 GMT)                    |
//| R152: Migrated from XAUUSD, Sharpe 4.52, K-Fold 5/5, WF 6/6     |
//| Optimized SL/TP for silver volatility profile                     |
//| DEMO ACCOUNT ONLY                                                 |
//+------------------------------------------------------------------+
//| Silver-specific adjustments:                                       |
//|   ATR minimum: 0.005 (vs gold 0.1)                                |
//|   SL=8.0 ATR, TP=12.0 ATR (wider than gold, silver is noisier)   |
//|   Trail/MaxHold/Session: same as gold (12 GMT, 4-bar lookback)    |
//|   PV=5000 per lot (vs gold 100) - handled by MT4 automatically    |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

extern double LotSize          = 0.01;     // Conservative for demo
extern int    MagicNumber      = 20260506;
extern int    MaxSlippage       = 50;       // Wider for silver liquidity

extern int    Session_Hour_GMT  = 12;
extern int    Lookback_Bars     = 4;
extern int    ATR_Period        = 14;
extern double SL_ATR_Mult       = 8.0;     // R152: wider for silver
extern double TP_ATR_Mult       = 12.0;    // R152: wider for silver
extern int    MaxHold_Bars      = 20;
extern double Trail_Act_ATR     = 0.14;
extern double Trail_Dist_ATR    = 0.025;
extern double ATR_Minimum       = 0.005;   // Silver ATR floor

extern int    Broker_GMT_Offset = 2;

extern bool   RuleB_Enabled     = true;
extern double RuleB_ATR_Sigma   = 3.0;
extern int    RuleB_ATR_Lookback = 60;
extern int    RuleB_SkipBars    = 8;

datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;
int      ruleB_skipCount    = 0;
datetime ruleB_lastCheckBar = 0;

//+------------------------------------------------------------------+
bool IsExtremeMarket()
{
   if(!RuleB_Enabled) return false;
   double sum = 0, sum2 = 0;
   for(int i = 0; i < RuleB_ATR_Lookback; i++)
   {
      double v = iATR(Symbol(), PERIOD_H1, 14, i + 1);
      sum += v;
      sum2 += v * v;
   }
   double mean = sum / RuleB_ATR_Lookback;
   double var  = sum2 / RuleB_ATR_Lookback - mean * mean;
   double std  = MathSqrt(MathMax(var, 0.000001));
   double cur  = iATR(Symbol(), PERIOD_H1, 14, 0);
   return (cur > mean + RuleB_ATR_Sigma * std);
}

//+------------------------------------------------------------------+
int OnInit()
{
   Print("AG_Session_BO_H1 EA v1.00 [DEMO] | Symbol=", Symbol(),
         " | Lot=", LotSize,
         " | Session GMT=", Session_Hour_GMT,
         " | LB=", Lookback_Bars,
         " | SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " | Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " | ATR_Min=", ATR_Minimum,
         " | Broker GMT Offset=", Broker_GMT_Offset,
         " | RuleB=", RuleB_Enabled);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void ManageOpenTrade()
{
   if(myTicket < 0) return;
   if(!OrderSelect(myTicket, SELECT_BY_TICKET)) { myTicket = -1; return; }
   if(OrderCloseTime() > 0) { myTicket = -1; return; }

   double atr = entryATR;
   double spread = MarketInfo(Symbol(), MODE_SPREAD) * Point;
   int dir = (OrderType() == OP_BUY) ? 1 : -1;
   double entry = OrderOpenPrice();
   double bid = MarketInfo(Symbol(), MODE_BID);
   double ask = MarketInfo(Symbol(), MODE_ASK);
   double price = (dir == 1) ? bid : ask;

   double pnl = (dir == 1) ? (price - entry) : (entry - price);
   pnl -= spread;
   double tp_dist = TP_ATR_Mult * atr;
   double sl_dist = SL_ATR_Mult * atr;

   if(pnl >= tp_dist) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrGreen);
      myTicket = -1; return;
   }
   if(pnl <= -sl_dist) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrRed);
      myTicket = -1; return;
   }

   double act = Trail_Act_ATR * atr;
   double dist = Trail_Dist_ATR * atr;
   if(dir == 1) {
      if(bid > extremePrice) extremePrice = bid;
      if(bid - entry >= act) {
         double ts = extremePrice - dist;
         if(bid <= ts) {
            OrderClose(myTicket, OrderLots(), bid, MaxSlippage, clrYellow);
            myTicket = -1; return;
         }
      }
   } else {
      if(ask < extremePrice) extremePrice = ask;
      if(entry - ask >= act) {
         double ts = extremePrice + dist;
         if(ask >= ts) {
            OrderClose(myTicket, OrderLots(), ask, MaxSlippage, clrYellow);
            myTicket = -1; return;
         }
      }
   }

   barsHeld++;
   if(barsHeld >= MaxHold_Bars) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrWhite);
      myTicket = -1; return;
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime curBar = iTime(Symbol(), PERIOD_H1, 0);
   if(curBar == lastBarTime) return;
   lastBarTime = curBar;

   if(RuleB_Enabled && curBar != ruleB_lastCheckBar)
   {
      ruleB_lastCheckBar = curBar;
      if(IsExtremeMarket())
      {
         ruleB_skipCount = RuleB_SkipBars;
         Print("Rule B: ATR spike on ", Symbol(), ", skipping ", RuleB_SkipBars, " bars");
      }
      else if(ruleB_skipCount > 0)
      {
         ruleB_skipCount--;
      }
   }

   ManageOpenTrade();

   if(myTicket >= 0) return;
   if(ruleB_skipCount > 0) return;

   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(atr < ATR_Minimum) return;

   int serverHour = TimeHour(TimeCurrent());
   int gmtHour = serverHour - Broker_GMT_Offset;
   if(gmtHour < 0) gmtHour += 24;
   if(gmtHour >= 24) gmtHour -= 24;

   if(gmtHour != Session_Hour_GMT) return;

   int prevServerHour = TimeHour(iTime(Symbol(), PERIOD_H1, 1));
   int prevGmtHour = prevServerHour - Broker_GMT_Offset;
   if(prevGmtHour < 0) prevGmtHour += 24;
   if(prevGmtHour >= 24) prevGmtHour -= 24;
   if(prevGmtHour == Session_Hour_GMT) return;

   if(TimeCurrent() - lastEntryTime < 7200) return;

   double rangeHigh = -999999;
   double rangeLow  =  999999;
   for(int i = 1; i <= Lookback_Bars; i++) {
      double h = iHigh(Symbol(), PERIOD_H1, i);
      double l = iLow(Symbol(), PERIOD_H1, i);
      if(h > rangeHigh) rangeHigh = h;
      if(l < rangeLow)  rangeLow  = l;
   }

   double curClose = iClose(Symbol(), PERIOD_H1, 0);
   double ask = MarketInfo(Symbol(), MODE_ASK);
   double bid = MarketInfo(Symbol(), MODE_BID);

   if(curClose > rangeHigh) {
      myTicket = OrderSend(Symbol(), OP_BUY, LotSize, ask, MaxSlippage, 0, 0,
                           "AG_SESS_BO_BUY", MagicNumber, 0, clrBlue);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = bid;
         lastEntryTime = TimeCurrent();
         Print("AG Session BO BUY | Range[", rangeLow, "-", rangeHigh, "] Close=", curClose,
               " ATR=", atr);
      }
   }
   else if(curClose < rangeLow) {
      myTicket = OrderSend(Symbol(), OP_SELL, LotSize, bid, MaxSlippage, 0, 0,
                           "AG_SESS_BO_SELL", MagicNumber, 0, clrRed);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = ask;
         lastEntryTime = TimeCurrent();
         Print("AG Session BO SELL | Range[", rangeLow, "-", rangeHigh, "] Close=", curClose,
               " ATR=", atr);
      }
   }
}
//+------------------------------------------------------------------+
