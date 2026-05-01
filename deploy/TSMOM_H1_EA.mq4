//+------------------------------------------------------------------+
//| TSMOM_H1_EA.mq4                                                 |
//| H1 Time Series Momentum                                          |
//| R53: K-Fold 50/50 PASS, Sharpe 5.33                              |
//| Chart: XAUUSD H1                                                  |
//+------------------------------------------------------------------+
//| Best params from R53 brute-force:                                 |
//|   Momentum: Fast=480, Slow=720, Weight=0.5/0.5                   |
//|   Exit: SL=4.5xATR, TP=6.0xATR, MaxHold=20 bars                 |
//|   Trail: Act=0.14xATR, Dist=0.025xATR                            |
//| R56 recommended lot: 0.04                                         |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
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

datetime lastEntryTime  = 0;
datetime lastBarTime    = 0;
int      barsHeld       = 0;
double   entryATR       = 0;
double   trailStopPrice = 0;
double   extremePrice   = 0;
int      myTicket       = -1;

//+------------------------------------------------------------------+
int OnInit()
{
   Print("TSMOM_H1 EA v1.0 | Lot=", LotSize,
         " | Fast=", Fast_Lookback, " Slow=", Slow_Lookback,
         " | SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " | Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
double MomentumScore()
{
   if(Bars < Slow_Lookback + 2) return 0;
   double cur = iClose(Symbol(), PERIOD_H1, 0);
   double fast_ref = iClose(Symbol(), PERIOD_H1, Fast_Lookback);
   double slow_ref = iClose(Symbol(), PERIOD_H1, Slow_Lookback);
   if(fast_ref <= 0 || slow_ref <= 0) return 0;

   double score = 0;
   if(cur / fast_ref - 1.0 > 0) score += 0.5;
   else if(cur / fast_ref - 1.0 < 0) score -= 0.5;

   if(cur / slow_ref - 1.0 > 0) score += 0.5;
   else if(cur / slow_ref - 1.0 < 0) score -= 0.5;

   return score;
}

double PrevMomentumScore()
{
   if(Bars < Slow_Lookback + 3) return 0;
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

   // TP
   if(pnl_usd >= tp_dist) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrGreen);
      myTicket = -1; return;
   }
   // SL
   if(pnl_usd <= -sl_dist) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrRed);
      myTicket = -1; return;
   }

   // Trailing stop
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

   // MaxHold
   barsHeld++;
   if(barsHeld >= MaxHold_Bars) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrWhite);
      myTicket = -1; return;
   }

   // Reversal exit
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

   ManageOpenTrade();

   if(myTicket >= 0) return;
   if(TimeCurrent() - lastEntryTime < MinEntryGapHours * 3600) return;

   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(atr < 0.1) return;

   double score = MomentumScore();
   double prev = PrevMomentumScore();

   double ask = MarketInfo(Symbol(), MODE_ASK);
   double bid = MarketInfo(Symbol(), MODE_BID);

   // BUY: score crosses above 0
   if(score > 0 && prev <= 0) {
      myTicket = OrderSend(Symbol(), OP_BUY, LotSize, ask, MaxSlippage, 0, 0,
                           "TSMOM_BUY", MagicNumber, 0, clrBlue);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = bid;
         lastEntryTime = TimeCurrent();
      }
   }
   // SELL: score crosses below 0
   else if(score < 0 && prev >= 0) {
      myTicket = OrderSend(Symbol(), OP_SELL, LotSize, bid, MaxSlippage, 0, 0,
                           "TSMOM_SELL", MagicNumber, 0, clrRed);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = ask;
         lastEntryTime = TimeCurrent();
      }
   }
}
//+------------------------------------------------------------------+
