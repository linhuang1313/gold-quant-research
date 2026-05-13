//+------------------------------------------------------------------+
//| TSMOM_H1_EA.mq4 v1.20                                            |
//| H1 Time Series Momentum                                          |
//| R53: K-Fold 50/50 PASS, Sharpe 5.33                              |
//| R144: +Rule B extreme protection, Sharpe +0.38                   |
//| R203: +History load check, +Diagnostic logging (silent fail fix)|
//| Chart: XAUUSD H1                                                  |
//+------------------------------------------------------------------+
//| Best params from R53 brute-force:                                 |
//|   Momentum: Fast=480, Slow=720, Weight=0.5/0.5                   |
//|   Exit: SL=4.5xATR, TP=6.0xATR, MaxHold=20 bars                 |
//|   Trail: Act=0.14xATR, Dist=0.025xATR                            |
//| Rule B: R144 validated, skip 8 bars after 3-sigma ATR spike      |
//| R56 recommended lot: 0.04                                         |
//+------------------------------------------------------------------+
//| R203 FIXES (silent fail diagnosed via run_r203b_tsmom_filter):   |
//|  - OnInit: force history download for Slow_Lookback+2 bars       |
//|  - OnInit: ABORT with alert if history insufficient              |
//|  - OnTick: log score state every 24 H1 bars                      |
//|  - MomentumScore: log when returning 0 due to data issue         |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.20"
#property strict

extern double LotSize          = 0.04;
extern int    MagicNumber      = 20250501;
extern int    MaxSlippage       = 30;

extern int    Fast_Lookback     = 480;
extern int    Slow_Lookback     = 720;
extern int    ATR_Period        = 14;
extern double SL_ATR_Mult       = 4.5;
extern double TP_ATR_Mult       = 6.0;
extern int    MaxHold_Bars      = 20;
extern double Trail_Act_ATR     = 0.14;
extern double Trail_Dist_ATR    = 0.025;
extern double MinEntryGapHours  = 2.0;

//--- Rule B extreme protection (R144: Sharpe +0.38)
extern bool   RuleB_Enabled     = true;
extern double RuleB_ATR_Sigma   = 3.0;
extern int    RuleB_ATR_Lookback = 60;
extern int    RuleB_SkipBars    = 8;

//--- R203 Diagnostic
extern bool   Diag_Verbose      = true;
extern int    Diag_PrintEvery_H1Bars = 24;  // print state every N bars

datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;
int      ruleB_skipCount    = 0;
datetime ruleB_lastCheckBar = 0;
int      diag_barCount      = 0;
bool     history_ok         = false;

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
// R203: Force-download history by accessing oldest required bar.
// Returns true if Bars >= Slow_Lookback + 2 AND the oldest required
// close price is available (>0). Retries a few times to allow MT4 to
// fetch history from broker.
bool EnsureHistoryLoaded(int max_retries = 30)
{
   int needed = Slow_Lookback + 2;
   for(int attempt = 0; attempt < max_retries; attempt++)
   {
      int barsAvail = iBars(Symbol(), PERIOD_H1);
      double oldest = iClose(Symbol(), PERIOD_H1, Slow_Lookback + 1);
      double fastest = iClose(Symbol(), PERIOD_H1, Fast_Lookback + 1);
      if(barsAvail >= needed && oldest > 0 && fastest > 0)
      {
         Print("TSMOM_H1 history loaded: Bars=", barsAvail,
               " needed=", needed,
               " oldest_close(", Slow_Lookback+1, ")=", DoubleToStr(oldest, 2));
         return true;
      }
      Print("TSMOM_H1 history not yet loaded: Bars=", barsAvail,
            " needed=", needed,
            " oldest_close=", DoubleToStr(oldest, 2),
            " (retry ", attempt+1, "/", max_retries, ")");
      Sleep(1000);
   }
   return false;
}

//+------------------------------------------------------------------+
int OnInit()
{
   Print("TSMOM_H1 EA v1.20 | Lot=", LotSize,
         " | Fast=", Fast_Lookback, " Slow=", Slow_Lookback,
         " | SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " | Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " | RuleB=", RuleB_Enabled, " Sigma=", RuleB_ATR_Sigma,
         " Skip=", RuleB_SkipBars,
         " | Magic=", MagicNumber);

   Print("TSMOM_H1 OnInit: requesting history (need Slow_Lookback+2=",
         Slow_Lookback+2, " H1 bars)...");
   history_ok = EnsureHistoryLoaded(30);
   if(!history_ok)
   {
      Alert("TSMOM_H1 ABORT: insufficient H1 history (need ",
            Slow_Lookback+2, " bars). Open H1 chart, scroll left to load history, ",
            "or increase Tools>Options>Charts>Max bars in chart.");
      return INIT_FAILED;
   }

   double s = MomentumScore();
   double p = PrevMomentumScore();
   Print("TSMOM_H1 init scores: cur=", DoubleToStr(s, 2),
         " prev=", DoubleToStr(p, 2),
         " close[0]=", DoubleToStr(iClose(Symbol(), PERIOD_H1, 0), 2),
         " close[", Fast_Lookback, "]=", DoubleToStr(iClose(Symbol(), PERIOD_H1, Fast_Lookback), 2),
         " close[", Slow_Lookback, "]=", DoubleToStr(iClose(Symbol(), PERIOD_H1, Slow_Lookback), 2));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
double MomentumScore()
{
   int barsAvail = iBars(Symbol(), PERIOD_H1);
   if(barsAvail < Slow_Lookback + 2)
   {
      if(Diag_Verbose) Print("TSMOM_H1 MomentumScore: Bars=", barsAvail,
                             " < needed=", Slow_Lookback+2, " -> return 0");
      return 0;
   }
   double cur = iClose(Symbol(), PERIOD_H1, 0);
   double fast_ref = iClose(Symbol(), PERIOD_H1, Fast_Lookback);
   double slow_ref = iClose(Symbol(), PERIOD_H1, Slow_Lookback);
   if(fast_ref <= 0 || slow_ref <= 0)
   {
      if(Diag_Verbose) Print("TSMOM_H1 MomentumScore: ref<=0 fast=",
                             DoubleToStr(fast_ref, 2),
                             " slow=", DoubleToStr(slow_ref, 2),
                             " -> return 0");
      return 0;
   }

   double score = 0;
   if(cur / fast_ref - 1.0 > 0) score += 0.5;
   else if(cur / fast_ref - 1.0 < 0) score -= 0.5;

   if(cur / slow_ref - 1.0 > 0) score += 0.5;
   else if(cur / slow_ref - 1.0 < 0) score -= 0.5;

   return score;
}

double PrevMomentumScore()
{
   int barsAvail = iBars(Symbol(), PERIOD_H1);
   if(barsAvail < Slow_Lookback + 3) return 0;
   double cur = iClose(Symbol(), PERIOD_H1, 1);
   double fast_ref = iClose(Symbol(), PERIOD_H1, Fast_Lookback + 1);
   double slow_ref = iClose(Symbol(), PERIOD_H1, Slow_Lookback + 1);
   if(fast_ref <= 0 || slow_ref <= 0) return 0;

   double score = 0;
   if(cur / fast_ref - 1.0 > 0) score += 0.5;
   else if(cur / fast_ref - 1.0 < 0) score -= 0.5;

   if(cur / slow_ref - 1.0 > 0) score += 0.5;
   else if(cur / slow_ref - 1.0 < 0) score -= 0.5;

   return score;
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

   double pnl_usd = (dir == 1) ? (price - entry) : (entry - price);
   pnl_usd -= spread;
   double tp_dist = TP_ATR_Mult * atr;
   double sl_dist = SL_ATR_Mult * atr;

   if(pnl_usd >= tp_dist) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrGreen);
      myTicket = -1; return;
   }
   if(pnl_usd <= -sl_dist) {
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

   double score = MomentumScore();
   if(dir == 1 && score < 0) {
      OrderClose(myTicket, OrderLots(), bid, MaxSlippage, clrOrange);
      myTicket = -1; return;
   }
   if(dir == -1 && score > 0) {
      OrderClose(myTicket, OrderLots(), ask, MaxSlippage, clrOrange);
      myTicket = -1; return;
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime curBar = iTime(Symbol(), PERIOD_H1, 0);
   if(curBar == lastBarTime) return;
   lastBarTime = curBar;

   if(!history_ok)
   {
      history_ok = EnsureHistoryLoaded(3);
      if(!history_ok) return;
   }

   // Rule B: check extreme market each new H1 bar
   if(RuleB_Enabled && curBar != ruleB_lastCheckBar)
   {
      ruleB_lastCheckBar = curBar;
      if(IsExtremeMarket())
      {
         ruleB_skipCount = RuleB_SkipBars;
         Print("TSMOM_H1 Rule B: ATR spike detected, skipping ", RuleB_SkipBars, " bars");
      }
      else if(ruleB_skipCount > 0)
      {
         ruleB_skipCount--;
      }
   }

   ManageOpenTrade();

   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   double score = MomentumScore();
   double prev = PrevMomentumScore();

   // R203 periodic diagnostic
   diag_barCount++;
   if(Diag_Verbose && (diag_barCount % Diag_PrintEvery_H1Bars == 0))
   {
      string st = (myTicket > 0) ? StringConcatenate("OPEN dir=", (OrderType()==OP_BUY?"BUY":"SELL"),
                                                     " bars=", barsHeld) : "FLAT";
      Print("TSMOM_H1 STATE: bar=", TimeToStr(curBar, TIME_DATE|TIME_MINUTES),
            " score=", DoubleToStr(score, 2),
            " prev=", DoubleToStr(prev, 2),
            " atr=", DoubleToStr(atr, 3),
            " rule_b_skip=", ruleB_skipCount,
            " ", st);
   }

   if(myTicket >= 0) return;

   if(ruleB_skipCount > 0) return;
   if(TimeCurrent() - lastEntryTime < MinEntryGapHours * 3600) return;
   if(atr < 0.1)
   {
      if(Diag_Verbose) Print("TSMOM_H1: ATR<0.1 reject (atr=", DoubleToStr(atr, 3), ")");
      return;
   }

   double ask = MarketInfo(Symbol(), MODE_ASK);
   double bid = MarketInfo(Symbol(), MODE_BID);

   if(score > 0 && prev <= 0) {
      Print("TSMOM_H1 SIGNAL BUY: score=", DoubleToStr(score, 2),
            " prev=", DoubleToStr(prev, 2),
            " atr=", DoubleToStr(atr, 3),
            " ask=", DoubleToStr(ask, 2));
      myTicket = OrderSend(Symbol(), OP_BUY, LotSize, ask, MaxSlippage, 0, 0,
                           "TSMOM_BUY", MagicNumber, 0, clrBlue);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = bid;
         lastEntryTime = TimeCurrent();
         Print("TSMOM_H1 BUY ticket=", myTicket, " entryATR=", DoubleToStr(atr, 3));
      } else {
         Print("TSMOM_H1 BUY OrderSend FAILED, err=", GetLastError());
      }
   }
   else if(score < 0 && prev >= 0) {
      Print("TSMOM_H1 SIGNAL SELL: score=", DoubleToStr(score, 2),
            " prev=", DoubleToStr(prev, 2),
            " atr=", DoubleToStr(atr, 3),
            " bid=", DoubleToStr(bid, 2));
      myTicket = OrderSend(Symbol(), OP_SELL, LotSize, bid, MaxSlippage, 0, 0,
                           "TSMOM_SELL", MagicNumber, 0, clrRed);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = ask;
         lastEntryTime = TimeCurrent();
         Print("TSMOM_H1 SELL ticket=", myTicket, " entryATR=", DoubleToStr(atr, 3));
      } else {
         Print("TSMOM_H1 SELL OrderSend FAILED, err=", GetLastError());
      }
   }
}
//+------------------------------------------------------------------+
