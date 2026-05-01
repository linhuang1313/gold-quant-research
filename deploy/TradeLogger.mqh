//+------------------------------------------------------------------+
//| TradeLogger.mqh                                                  |
//| Shared include: append closed trade records to CSV               |
//| Usage: #include "TradeLogger.mqh" then call LogTrade() on close  |
//+------------------------------------------------------------------+
#property strict

//+------------------------------------------------------------------+
//| LogTrade — append one line to <ea_name>_trades.csv               |
//|   ea_name : short name for the EA (e.g. "PSAR_H1")              |
//|   ticket  : order ticket (must be already selected via           |
//|             OrderSelect before calling, or pass ticket to select)|
//|   reason  : exit reason string ("SL","TP","Trail","Timeout")     |
//+------------------------------------------------------------------+
void LogTrade(string ea_name, int ticket, string reason)
{
   if(!OrderSelect(ticket, SELECT_BY_TICKET, MODE_HISTORY))
   {
      if(!OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
         return;
   }

   string filename = ea_name + "_trades.csv";

   bool file_exists = false;
   int check = FileOpen(filename, FILE_READ | FILE_CSV | FILE_COMMON);
   if(check != INVALID_HANDLE)
   {
      file_exists = (FileSize(check) > 10);
      FileClose(check);
   }

   int handle = FileOpen(filename, FILE_READ | FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("TradeLogger: cannot open ", filename, " error=", GetLastError());
      return;
   }

   if(!file_exists)
   {
      FileWrite(handle,
         "ticket", "symbol", "type", "open_time", "close_time",
         "open_price", "close_price", "lots", "pnl", "commission",
         "swap", "reason", "magic", "comment");
   }

   FileSeek(handle, 0, SEEK_END);

   string type_str = (OrderType() == OP_BUY) ? "BUY" : "SELL";
   double pnl = OrderProfit() + OrderCommission() + OrderSwap();

   FileWrite(handle,
      IntegerToString(OrderTicket()),
      OrderSymbol(),
      type_str,
      TimeToStr(OrderOpenTime(), TIME_DATE | TIME_SECONDS),
      TimeToStr(OrderCloseTime(), TIME_DATE | TIME_SECONDS),
      DoubleToStr(OrderOpenPrice(), (int)MarketInfo(OrderSymbol(), MODE_DIGITS)),
      DoubleToStr(OrderClosePrice(), (int)MarketInfo(OrderSymbol(), MODE_DIGITS)),
      DoubleToStr(OrderLots(), 2),
      DoubleToStr(pnl, 2),
      DoubleToStr(OrderCommission(), 2),
      DoubleToStr(OrderSwap(), 2),
      reason,
      IntegerToString(OrderMagicNumber()),
      OrderComment()
   );

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| LogTradeByTicket — select from history and log                   |
//+------------------------------------------------------------------+
void LogTradeByTicket(string ea_name, int ticket, string reason)
{
   LogTrade(ea_name, ticket, reason);
}
//+------------------------------------------------------------------+
