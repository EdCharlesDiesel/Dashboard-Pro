//+------------------------------------------------------------------+
//|                                          HistoryAutoExport.mq4    |
//|   Auto-exports closed-trade history + account balance to an HTML |
//|   file on a timer, for Dashboard-Pro's Trade Journal auto-import. |
//|                                                                  |
//|   The HTML is shaped to match src/services/mt4_import.py:        |
//|     - a closed-trades table whose rows carry a 'buy'/'sell' type, |
//|       two timestamps and a trailing Profit column;               |
//|     - a summary row with 'Balance:' / 'Equity:' cells.           |
//|                                                                  |
//|   MT4 sandboxes file writes, so the file lands in the terminal's  |
//|   Files folder (or the shared Common\Files folder). Point the     |
//|   Trade Journal's auto-import path at that file.                  |
//+------------------------------------------------------------------+
#property strict
#property description "Periodically exports account history to an HTML statement for Dashboard-Pro."

input string ExportFileName  = "history.htm"; // Output file name (in MQL4\Files or Common\Files)
input int    RefreshSeconds  = 300;           // Export interval in seconds (300 = 5 min)
input bool   UseCommonFolder = true;          // true: Common\Files (path stable across terminals)

//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer((RefreshSeconds < 5) ? 5 : RefreshSeconds);
   ExportHistory();                 // write once immediately
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }
//+------------------------------------------------------------------+
void OnTimer()
  {
   ExportHistory();
  }
//+------------------------------------------------------------------+
//| Format a price with the instrument's digit count (fallback 5).   |
//+------------------------------------------------------------------+
string Px(double price, string sym)
  {
   int dg = (int)MarketInfo(sym, MODE_DIGITS);
   if(dg <= 0) dg = 5;
   return(DoubleToStr(price, dg));
  }
//+------------------------------------------------------------------+
//| Build a single closed-trade <tr> matching the parser's layout.   |
//| Columns: Ticket, OpenTime, Type, Size, Item, Price, S/L, T/P,    |
//|          CloseTime, Price, Commission, Taxes, Swap, Profit.      |
//+------------------------------------------------------------------+
string TradeRow()
  {
   string sym  = OrderSymbol();
   string type = (OrderType() == OP_BUY) ? "buy" : "sell";
   string r = "<tr>";
   r += "<td>" + IntegerToString(OrderTicket()) + "</td>";
   r += "<td>" + TimeToStr(OrderOpenTime(),  TIME_DATE|TIME_SECONDS) + "</td>";
   r += "<td>" + type + "</td>";
   r += "<td>" + DoubleToStr(OrderLots(), 2) + "</td>";
   r += "<td>" + sym + "</td>";
   r += "<td>" + Px(OrderOpenPrice(),  sym) + "</td>";
   r += "<td>" + Px(OrderStopLoss(),   sym) + "</td>";
   r += "<td>" + Px(OrderTakeProfit(), sym) + "</td>";
   r += "<td>" + TimeToStr(OrderCloseTime(), TIME_DATE|TIME_SECONDS) + "</td>";
   r += "<td>" + Px(OrderClosePrice(), sym) + "</td>";
   r += "<td>" + DoubleToStr(OrderCommission(), 2) + "</td>";
   r += "<td>0.00</td>";                                   // Taxes
   r += "<td>" + DoubleToStr(OrderSwap(), 2) + "</td>";
   r += "<td>" + DoubleToStr(OrderProfit(), 2) + "</td>";  // Profit must be the last numeric cell
   r += "</tr>";
   return(r);
  }
//+------------------------------------------------------------------+
//| Assemble the full HTML statement and write it to disk.           |
//+------------------------------------------------------------------+
void ExportHistory()
  {
   string html = "<!DOCTYPE html><html><head><meta charset=\"utf-8\"></head><body>";
   html += "<h2>Account " + IntegerToString(AccountNumber())
           + " — auto-export " + TimeToStr(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "</h2>";

   // ── Closed trades table ──────────────────────────────────────────
   html += "<table border=\"1\"><tr>"
           "<td>Ticket</td><td>Open Time</td><td>Type</td><td>Size</td><td>Item</td>"
           "<td>Price</td><td>S/L</td><td>T/P</td><td>Close Time</td><td>Price</td>"
           "<td>Commission</td><td>Taxes</td><td>Swap</td><td>Profit</td></tr>";

   int trades = 0;
   double closedPL = 0.0;
   for(int i = 0; i < OrdersHistoryTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
         continue;
      int t = OrderType();
      if(t != OP_BUY && t != OP_SELL)     // skip deposits / credits / cancelled pendings
         continue;
      html += TradeRow();
      closedPL += OrderProfit() + OrderSwap() + OrderCommission();
      trades++;
     }
   html += "</table>";

   // ── Summary (the parser reads Balance: / Equity: from here) ──────
   html += "<table border=\"1\">";
   html += "<tr><td>Closed Trade P/L:</td><td>" + DoubleToStr(closedPL, 2) + "</td></tr>";
   html += "<tr><td>Balance:</td><td>" + DoubleToStr(AccountBalance(), 2) + "</td>"
           "<td>Equity:</td><td>" + DoubleToStr(AccountEquity(), 2) + "</td>"
           "<td>Free Margin:</td><td>" + DoubleToStr(AccountFreeMargin(), 2) + "</td></tr>";
   html += "</table>";
   html += "</body></html>";

   // ── Write (FILE_BIN + FileWriteString = raw bytes, no CSV mangling)
   int flags = FILE_WRITE | FILE_BIN;
   if(UseCommonFolder)
      flags |= FILE_COMMON;

   int handle = FileOpen(ExportFileName, flags);
   if(handle == INVALID_HANDLE)
     {
      Print("HistoryAutoExport: cannot open ", ExportFileName, " (err ", GetLastError(), ")");
      return;
     }
   FileWriteString(handle, html);
   FileClose(handle);

   Comment("HistoryAutoExport\nLast write: ", TimeToStr(TimeCurrent(), TIME_SECONDS),
           "\nTrades: ", trades, "   Balance: ", DoubleToStr(AccountBalance(), 2),
           "\nFile: ", (UseCommonFolder ? "Common\\Files\\" : "MQL4\\Files\\"), ExportFileName);
  }
//+------------------------------------------------------------------+
