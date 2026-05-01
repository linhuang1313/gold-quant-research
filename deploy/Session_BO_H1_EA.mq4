//+------------------------------------------------------------------+
//| Session_BO_H1_EA.mq4                                             |
//| H1 Session Breakout — NY Peak (12-14 GMT)                        |
//| R55A: K-Fold 50/50 PASS, Sharpe 5.87, KF Mean 6.89              |
//| Chart: XAUUSD H1                                                  |
//+------------------------------------------------------------------+
//| Best params from R55A brute-force:                                |
//|   Session: peak_12_14 (GMT 12:00 trigger)                        |
//|   Lookback: 4 bars (前4根H1的高低点构成区间)                     |
//|   Exit: SL=4.5xATR, TP=4.0xATR, MaxHold=20 bars                 |
//|   Trail: Act=0.14xATR, Dist=0.025xATR                            |
//| R56 recommended lot: 0.04                                         |
//+------------------------------------------------------------------+
#property copyright "Gold Quant Research"
#property version   "1.00"
#property strict

extern double LotSize          = 0.04;
extern int    MagicNumber      = 20250502;
extern int    MaxSlippage       = 30;

extern int    Session_Hour_GMT  = 12;       // GMT 12:00 触发
extern int    Lookback_Bars     = 4;        // 前4根H1 bar
extern int    ATR_Period        = 14;
extern double SL_ATR_Mult       = 4.5;
extern double TP_ATR_Mult       = 4.0;
extern int    MaxHold_Bars      = 20;
extern double Trail_Act_ATR     = 0.14;
extern double Trail_Dist_ATR    = 0.025;

extern int    Broker_GMT_Offset = 2;        // 经纪商服务器时间与GMT的偏移 (常见: FXTM=2, IC=2)

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
   Print("Session_BO_H1 EA v1.0 | Lot=", LotSize,
         " | Session GMT=", Session_Hour_GMT,
         " | LB=", Lookback_Bars,
         " | SL=", SL_ATR_Mult, " TP=", TP_ATR_Mult,
         " | Trail=", Trail_Act_ATR, "/", Trail_Dist_ATR,
         " | Broker GMT Offset=", Broker_GMT_Offset);
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

   // TP
   if(pnl >= tp_dist) {
      OrderClose(myTicket, OrderLots(), price, MaxSlippage, clrGreen);
      myTicket = -1; return;
   }
   // SL
   if(pnl <= -sl_dist) {
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
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime curBar = iTime(Symbol(), PERIOD_H1, 0);
   if(curBar == lastBarTime) return;
   lastBarTime = curBar;

   ManageOpenTrade();

   if(myTicket >= 0) return;

   // 检查ATR有效性
   double atr = iATR(Symbol(), PERIOD_H1, ATR_Period, 0);
   if(atr < 0.1) return;

   // 将经纪商服务器时间转为GMT
   int serverHour = TimeHour(TimeCurrent());
   int gmtHour = serverHour - Broker_GMT_Offset;
   if(gmtHour < 0) gmtHour += 24;
   if(gmtHour >= 24) gmtHour -= 24;

   // 仅在 Session_Hour_GMT 触发
   if(gmtHour != Session_Hour_GMT) return;

   // 确认是新时段开始（前一根bar不是同一时段小时）
   int prevServerHour = TimeHour(iTime(Symbol(), PERIOD_H1, 1));
   int prevGmtHour = prevServerHour - Broker_GMT_Offset;
   if(prevGmtHour < 0) prevGmtHour += 24;
   if(prevGmtHour >= 24) prevGmtHour -= 24;
   if(prevGmtHour == Session_Hour_GMT) return;

   // 入场间隔检查（至少隔2根H1 bar ≈ 2小时）
   if(TimeCurrent() - lastEntryTime < 7200) return;

   // 计算前 Lookback_Bars 根bar的高低区间
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

   // BUY: 当前收盘突破区间高点
   if(curClose > rangeHigh) {
      myTicket = OrderSend(Symbol(), OP_BUY, LotSize, ask, MaxSlippage, 0, 0,
                           "SESS_BO_BUY", MagicNumber, 0, clrBlue);
      if(myTicket > 0) {
         entryATR = atr; barsHeld = 0; extremePrice = bid;
         lastEntryTime = TimeCurrent();
         Print("Session BO BUY | Range[", rangeLow, "-", rangeHigh, "] Close=", curClose);
      }
   }
   // SELL: 当前收盘跌破区间低点
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
