//+------------------------------------------------------------------+
//| Session_BO_H1_EA.mq4                                             |
//| H1 Session Breakout - NY Peak (12-14 GMT)                        |
//| R55A: K-Fold 50/50 PASS, Sharpe 5.87, KF Mean 6.89              |
//| R144: +Rule B extreme protection, Sharpe +0.38                   |
//| Chart: XAUUSD H1                                                  |
//+------------------------------------------------------------------+
//| Best params from R55A brute-force:                                |
//|   Session: peak_12_14 (GMT 12:00 trigger)                        |
//|   Lookback: 4 bars                                                |
//|   Exit: SL=4.5xATR, TP=4.0xATR, MaxHold=20 bars                 |
//|   Trail: Act=0.14xATR, Dist=0.025xATR                            |
//| Rule B: R144 validated, skip 8 bars after 3-sigma ATR spike      |
//| R56 recommended lot: 0.04                                         |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.10"
#property strict

extern double LotSize          = 0.04;
extern int    MagicNumber      = 20250502;
extern int    MaxSlippage       = 30;

extern int    Session_Hour_GMT  = 12;
extern int    Lookback_Bars     = 4;
extern int    ATR_Period        = 14;
extern double SL_ATR_Mult       = 4.5;
extern double TP_ATR_Mult       = 4.0;
extern int    MaxHold_Bars      = 20;
extern double Trail_Act_ATR     = 0.14;
extern double Trail_Dist_ATR    = 0.025;

extern int    Broker_GMT_Offset = 2;

//--- Rule B extreme protection (R144: Sharpe +0.38)
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
   Print("Session_BO_H1 EA v1.10 | Lot=", LotSize,
         " | Session GMT=", Session_Hour_GMT,
         " | LB=", Lookback_Bars,
         " | SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " | Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " | Broker GMT Offset=", Broker_GMT_Offset,
         " | RuleB=", RuleB_Enabled, " Sigma=", RuleB_ATR_Sigma,
         " Skip=", RuleB_SkipBars);
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

   // Rule B: check extreme market each new H1 bar
   if(RuleB_Enabled && curBar != ruleB_lastCheckBar)
   {
      ruleB_lastCheckBar = curBar;
      if(IsExtremeMarket())
      {
         ruleB_skipCount = RuleB_SkipBars;
         Print("Rule B: ATR spike detected, skipping ", RuleB_SkipBars, " bars");
      }
      else if(ruleB_skipCount > 0)
      {
         ruleB_skipCount--;
      }
   }

   ManageOpenTrade();

   if(myTicket >= 0) return;

   // Rule B: skip entry during cooldown
   if(ruleB_skipCount > 0) return;

   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(atr < 0.1) return;

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
                           "SESS_BO_BUY", MagicNumber, 0, clrBlue);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = bid;
         lastEntryTime = TimeCurrent();
         Print("Session BO BUY | Range[", rangeLow, "-", rangeHigh, "] Close=", curClose);
      }
   }
   else if(curClose < rangeLow) {
      myTicket = OrderSend(Symbol(), OP_SELL, LotSize, bid, MaxSlippage, 0, 0,
                           "SESS_BO_SELL", MagicNumber, 0, clrRed);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = ask;
         lastEntryTime = TimeCurrent();
         Print("Session BO SELL | Range[", rangeLow, "-", rangeHigh, "] Close=", curClose);
      }
   }
}
//+------------------------------------------------------------------+
