//+------------------------------------------------------------------+
//|                                         WicklessSignalPro.mq5    |
//|  Production wickless-candle signal indicator                     |
//|                                                                  |
//|  NON-REPAINTING: signals are evaluated ONLY on a fully closed    |
//|  bar (index rates_total-2). Nothing is drawn or alerted on the   |
//|  forming bar, so a signal never disappears after the fact.       |
//|                                                                  |
//|  DIRECTION CONVENTION -- READ THIS                               |
//|                                                                  |
//|  SIGNAL_FILL (default, matches the WicklessCandleLab test):      |
//|      Bull wickless (open==low)  -> SELL, target = open           |
//|      Bear wickless (open==high) -> BUY,  target = open           |
//|      Thesis: the missing wick gets filled.                       |
//|                                                                  |
//|  SIGNAL_CONTINUATION (the opposite thesis):                      |
//|      Bull wickless  -> BUY,  target = close + R * risk           |
//|      Bear wickless  -> SELL, target = close - R * risk           |
//|      Thesis: an absent wick signals one-sided pressure.          |
//|                                                                  |
//|  EMAIL SETUP                                                     |
//|      MT5 -> Tools -> Options -> Email. Fill in SMTP server,      |
//|      login, password, from and to. Tick "Enable". Press Test.    |
//|      Push notifications: Tools -> Options -> Notifications,      |
//|      paste the MetaQuotes ID from the mobile app.                |
//|      Email and push are disabled inside Strategy Tester.         |
//+------------------------------------------------------------------+
#property copyright "Wickless Signal Pro"
#property version   "1.10"
#property description "Wickless candle signals with BUY/SELL labels and email alerts. Non-repainting."
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

#property indicator_label1  "Wickless Buy"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrDodgerBlue
#property indicator_width1  3

#property indicator_label2  "Wickless Sell"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrOrangeRed
#property indicator_width2  3

//+------------------------------------------------------------------+
//| Enums                                                            |
//+------------------------------------------------------------------+
enum ENUM_WICK_MODE
  {
   WICK_OPEN_SIDE = 0,   // Open-side wick missing (bull: open==low)
   WICK_MARUBOZU  = 1    // Both wicks missing (full marubozu)
  };

enum ENUM_SIGNAL_MODE
  {
   SIGNAL_FILL         = 0,  // Fill thesis: bull wickless = SELL to the open
   SIGNAL_CONTINUATION = 1   // Continuation thesis: bull wickless = BUY
  };

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group                 "=== Detection ==="
input ENUM_WICK_MODE   InpMode            = WICK_OPEN_SIDE;  // Wickless definition
input int              InpTolerancePoints = 0;               // Wick tolerance (points)
input double           InpMinBodyATR      = 0.10;            // Min body size (ATR units)
input int              InpATRPeriod       = 14;              // ATR period

input group                 "=== Signal construction ==="
input ENUM_SIGNAL_MODE InpSignalMode      = SIGNAL_FILL;     // Trade thesis (see header!)
input double           InpStopBufferATR   = 0.50;            // Stop buffer beyond candle (ATR)
input double           InpContTargetR     = 2.0;             // Continuation mode: target in R
input double           InpMinRR           = 0.80;            // Min reward:risk to fire (0 = off)
input int              InpValidBars       = 20;              // Signal validity (bars)

input group                 "=== Filters ==="
input bool             InpUseSession      = false;           // Use session time filter
input int              InpSessionStartHr  = 17;              // Session start hour (SERVER time)
input int              InpSessionStartMin = 0;               // Session start minute
input int              InpSessionEndHr    = 20;              // Session end hour (SERVER time)
input int              InpSessionEndMin   = 0;               // Session end minute
input bool             InpTradeMon        = true;            // Monday
input bool             InpTradeTue        = true;            // Tuesday
input bool             InpTradeWed        = true;            // Wednesday
input bool             InpTradeThu        = true;            // Thursday
input bool             InpTradeFri        = true;            // Friday
input int              InpMaxSpreadPoints = 0;               // Max spread to fire (points, 0=off)
input double           InpMinATRPoints    = 0;               // Min ATR to fire (points, 0=off)

input group                 "=== Alerts ==="
input bool             InpAlertPopup      = true;            // Terminal popup alert
input bool             InpAlertSound      = true;            // Play sound
input string           InpSoundFile       = "alert2.wav";    // Sound file
input bool             InpAlertEmail      = true;            // Send email
input bool             InpAlertPush       = false;           // Send push notification
input string           InpSubjectPrefix   = "[Wickless]";    // Email subject prefix
input int              InpCooldownMinutes = 0;               // Min minutes between alerts (0=off)

input group                 "=== Risk sizing (informational) ==="
input bool             InpShowSizing      = true;            // Include lot size in alert
input double           InpRiskPercent     = 0.5;             // Risk per trade (% of balance)

input group                 "=== Display ==="
input bool             InpShowLabels      = true;            // Draw BUY/SELL text labels
input int              InpLabelFontSize   = 9;               // Label font size
input color            InpBuyColor        = clrDodgerBlue;   // Buy colour
input color            InpSellColor       = clrOrangeRed;    // Sell colour
input bool             InpShowLevels      = true;            // Draw entry/stop/target lines
input int              InpScanBars        = 500;             // History bars to mark on load
input int              InpMaxLabels       = 100;             // Max labels kept on chart

input group                 "=== Logging ==="
input bool             InpLogCSV          = true;            // Append signals to CSV

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
double   BufBuy[];
double   BufSell[];

string   g_prefix = "WSP_";
string   g_gvName;
datetime g_lastSignalBar = 0;
datetime g_lastAlertTime = 0;
string   g_objNames[];
bool     g_emailReady = false;
bool     g_pushReady  = false;

struct Signal
  {
   bool     valid;
   int      dir;            // +1 = BUY, -1 = SELL
   datetime barTime;
   double   entry;
   double   stop;
   double   target;
   double   risk;
   double   reward;
   double   rr;
   double   atr;
   string   pattern;        // "Bull wickless" / "Bear wickless"
  };

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, BufBuy,  INDICATOR_DATA);
   SetIndexBuffer(1, BufSell, INDICATOR_DATA);
   PlotIndexSetInteger(0, PLOT_ARROW, 233);
   PlotIndexSetInteger(1, PLOT_ARROW, 234);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_ARROW_SHIFT,  14);
   PlotIndexSetInteger(1, PLOT_ARROW_SHIFT, -14);
   ArraySetAsSeries(BufBuy,  false);
   ArraySetAsSeries(BufSell, false);

   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("WicklessSignalPro(%s,%dpt)",
                                   InpSignalMode == SIGNAL_FILL ? "fill" : "cont",
                                   InpTolerancePoints));
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

   if(InpATRPeriod < 2 || InpValidBars < 1)
     {
      Print("WicklessSignalPro: invalid ATR period or validity window.");
      return(INIT_PARAMETERS_INCORRECT);
     }

   //--- alert channel readiness
   g_emailReady = (bool)TerminalInfoInteger(TERMINAL_EMAIL_ENABLED);
   g_pushReady  = (bool)TerminalInfoInteger(TERMINAL_NOTIFICATIONS_ENABLED);

   if(InpAlertEmail && !g_emailReady)
      Print("WicklessSignalPro: WARNING - email requested but not enabled. "
            "Tools > Options > Email, fill in SMTP details and tick Enable.");
   if(InpAlertPush && !g_pushReady)
      Print("WicklessSignalPro: WARNING - push requested but MetaQuotes ID not set. "
            "Tools > Options > Notifications.");

   //--- restore last-alerted bar so a restart does not re-fire
   g_gvName = "WSP_last_" + _Symbol + "_" + IntegerToString((int)_Period);
   if(GlobalVariableCheck(g_gvName))
      g_lastSignalBar = (datetime)GlobalVariableGet(g_gvName);

   ArrayResize(g_objNames, 0);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, g_prefix);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| ATR at a single bar index (simple average of true range)         |
//+------------------------------------------------------------------+
double ATRAt(const int i, const double &high[], const double &low[], const double &close[])
  {
   if(i < InpATRPeriod)
      return(0.0);
   double sum = 0.0;
   for(int k = i - InpATRPeriod + 1; k <= i; k++)
     {
      double a = high[k] - low[k];
      double b = MathAbs(high[k] - close[k-1]);
      double c = MathAbs(low[k]  - close[k-1]);
      sum += MathMax(a, MathMax(b, c));
     }
   double atr = sum / InpATRPeriod;
   return(atr > 0.0 ? atr : 0.0);
  }

//+------------------------------------------------------------------+
//| Pattern detection                                                |
//+------------------------------------------------------------------+
bool IsWicklessBull(const int i, const double atr,
                    const double &open[], const double &high[],
                    const double &low[],  const double &close[])
  {
   double tol = InpTolerancePoints * _Point;
   if(close[i] <= open[i])                                   return(false);
   if((open[i] - low[i]) > tol)                              return(false);
   if(InpMode == WICK_MARUBOZU && (high[i] - close[i]) > tol) return(false);
   if((close[i] - open[i]) < InpMinBodyATR * atr)            return(false);
   return(true);
  }

bool IsWicklessBear(const int i, const double atr,
                    const double &open[], const double &high[],
                    const double &low[],  const double &close[])
  {
   double tol = InpTolerancePoints * _Point;
   if(close[i] >= open[i])                                   return(false);
   if((high[i] - open[i]) > tol)                             return(false);
   if(InpMode == WICK_MARUBOZU && (close[i] - low[i]) > tol) return(false);
   if((open[i] - close[i]) < InpMinBodyATR * atr)            return(false);
   return(true);
  }

//+------------------------------------------------------------------+
//| Session / day filter                                             |
//+------------------------------------------------------------------+
bool DayAllowed(const datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   switch(dt.day_of_week)
     {
      case 1: return(InpTradeMon);
      case 2: return(InpTradeTue);
      case 3: return(InpTradeWed);
      case 4: return(InpTradeThu);
      case 5: return(InpTradeFri);
      default: return(false);   // weekend bars
     }
  }

bool SessionAllowed(const datetime t)
  {
   if(!InpUseSession)
      return(true);
   MqlDateTime dt;
   TimeToStruct(t, dt);
   int now   = dt.hour * 60 + dt.min;
   int start = InpSessionStartHr * 60 + InpSessionStartMin;
   int end   = InpSessionEndHr   * 60 + InpSessionEndMin;
   if(start == end)
      return(true);
   if(start < end)
      return(now >= start && now < end);
   return(now >= start || now < end);   // wraps midnight
  }

//+------------------------------------------------------------------+
//| Build a signal from a closed bar. Returns valid=false if none.   |
//+------------------------------------------------------------------+
Signal BuildSignal(const int i,
                   const datetime &time[], const double &open[],
                   const double &high[],   const double &low[],
                   const double &close[])
  {
   Signal s;
   ZeroMemory(s);
   s.valid = false;

   double atr = ATRAt(i, high, low, close);
   if(atr <= 0.0)
      return(s);

   bool wb = IsWicklessBull(i, atr, open, high, low, close);
   bool ws = IsWicklessBear(i, atr, open, high, low, close);
   if(!wb && !ws)
      return(s);

   s.atr     = atr;
   s.barTime = time[i];
   s.entry   = close[i];
   s.pattern = wb ? "Bull wickless" : "Bear wickless";

   double buf = InpStopBufferATR * atr;

   if(InpSignalMode == SIGNAL_FILL)
     {
      if(wb)
        {
         //--- unfilled level sits BELOW at the open -> sell into it
         s.dir    = -1;
         s.target = open[i];
         s.stop   = high[i] + buf;
        }
      else
        {
         //--- unfilled level sits ABOVE at the open -> buy toward it
         s.dir    = 1;
         s.target = open[i];
         s.stop   = low[i] - buf;
        }
     }
   else // SIGNAL_CONTINUATION
     {
      if(wb)
        {
         s.dir    = 1;
         s.stop   = low[i] - buf;
         s.target = s.entry + InpContTargetR * MathAbs(s.entry - s.stop);
        }
      else
        {
         s.dir    = -1;
         s.stop   = high[i] + buf;
         s.target = s.entry - InpContTargetR * MathAbs(s.entry - s.stop);
        }
     }

   s.risk   = MathAbs(s.entry - s.stop);
   s.reward = MathAbs(s.target - s.entry);
   if(s.risk <= 0.0)
      return(s);
   s.rr = s.reward / s.risk;

   if(InpMinRR > 0.0 && s.rr < InpMinRR)
      return(s);

   s.valid = true;
   return(s);
  }

//+------------------------------------------------------------------+
//| Position size from risk % (informational only)                   |
//+------------------------------------------------------------------+
double LotsForRisk(const double riskPrice)
  {
   if(riskPrice <= 0.0)
      return(0.0);

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue <= 0.0 || tickSize <= 0.0)
      return(0.0);

   double lossPerLot = (riskPrice / tickSize) * tickValue;
   if(lossPerLot <= 0.0)
      return(0.0);

   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * InpRiskPercent / 100.0;
   double lots      = riskMoney / lossPerLot;

   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(stepLot > 0.0)
      lots = MathFloor(lots / stepLot) * stepLot;
   if(lots < minLot) lots = 0.0;          // below broker minimum
   if(lots > maxLot) lots = maxLot;
   return(lots);
  }

//+------------------------------------------------------------------+
//| Object bookkeeping                                               |
//+------------------------------------------------------------------+
void TrackObject(const string name)
  {
   int n = ArraySize(g_objNames);
   ArrayResize(g_objNames, n + 1);
   g_objNames[n] = name;

   int cap = MathMax(10, InpMaxLabels) * 4;   // labels + up to 3 lines each
   if(ArraySize(g_objNames) > cap)
     {
      int drop = ArraySize(g_objNames) - cap;
      for(int k = 0; k < drop; k++)
         ObjectDelete(0, g_objNames[k]);
      for(int k = 0; k < ArraySize(g_objNames) - drop; k++)
         g_objNames[k] = g_objNames[k + drop];
      ArrayResize(g_objNames, ArraySize(g_objNames) - drop);
     }
  }

void DrawLine(const string name, const datetime t1, const datetime t2,
              const double price, const color clr, const int style)
  {
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
   if(!ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price))
      return;
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   TrackObject(name);
  }

void DrawSignal(const Signal &s, const int i, const int total,
                const datetime &time[], const double &high[], const double &low[])
  {
   if(s.dir > 0) BufBuy[i]  = low[i];
   else          BufSell[i] = high[i];

   string stamp = IntegerToString((int)s.barTime);
   color  clr   = (s.dir > 0) ? InpBuyColor : InpSellColor;

   if(InpShowLabels)
     {
      string nm = g_prefix + "txt_" + stamp;
      if(ObjectFind(0, nm) >= 0)
         ObjectDelete(0, nm);
      double y = (s.dir > 0) ? low[i] - 2.2 * s.atr * 0.25 : high[i] + 2.2 * s.atr * 0.25;
      if(ObjectCreate(0, nm, OBJ_TEXT, 0, s.barTime, y))
        {
         ObjectSetString(0, nm, OBJPROP_TEXT,
                         StringFormat("%s  %.2fR", (s.dir > 0 ? "BUY" : "SELL"), s.rr));
         ObjectSetString(0, nm, OBJPROP_FONT, "Arial Black");
         ObjectSetInteger(0, nm, OBJPROP_FONTSIZE, InpLabelFontSize);
         ObjectSetInteger(0, nm, OBJPROP_COLOR, clr);
         ObjectSetInteger(0, nm, OBJPROP_ANCHOR,
                          s.dir > 0 ? ANCHOR_UPPER : ANCHOR_LOWER);
         ObjectSetInteger(0, nm, OBJPROP_SELECTABLE, false);
         TrackObject(nm);
        }
     }

   if(InpShowLevels)
     {
      int      j  = MathMin(total - 1, i + InpValidBars);
      datetime t2 = time[j];
      DrawLine(g_prefix + "e_" + stamp, s.barTime, t2, s.entry,  clrSilver,  STYLE_DOT);
      DrawLine(g_prefix + "s_" + stamp, s.barTime, t2, s.stop,   clrCrimson, STYLE_DASH);
      DrawLine(g_prefix + "t_" + stamp, s.barTime, t2, s.target, clrMediumSeaGreen, STYLE_SOLID);
     }
  }

//+------------------------------------------------------------------+
//| Alerts                                                           |
//+------------------------------------------------------------------+
void FireAlerts(const Signal &s)
  {
   string side  = (s.dir > 0) ? "BUY" : "SELL";
   string tfStr = EnumToString((ENUM_TIMEFRAMES)_Period);
   StringReplace(tfStr, "PERIOD_", "");

   string subject = StringFormat("%s %s %s %s @ %s",
                                 InpSubjectPrefix, side, _Symbol, tfStr,
                                 DoubleToString(s.entry, _Digits));

   double lots = InpShowSizing ? LotsForRisk(s.risk) : 0.0;

   string body = "";
   body += StringFormat("%s SIGNAL\r\n", side);
   body += StringFormat("Symbol      : %s %s\r\n", _Symbol, tfStr);
   body += StringFormat("Pattern     : %s (tol %d pt)\r\n", s.pattern, InpTolerancePoints);
   body += StringFormat("Thesis      : %s\r\n",
                        InpSignalMode == SIGNAL_FILL ? "fill the missing wick" : "continuation");
   body += StringFormat("Bar close   : %s (server)\r\n",
                        TimeToString(s.barTime, TIME_DATE | TIME_MINUTES));
   body += "----------------------------------------\r\n";
   body += StringFormat("Entry       : %s\r\n", DoubleToString(s.entry,  _Digits));
   body += StringFormat("Stop        : %s   (%.0f pt)\r\n",
                        DoubleToString(s.stop, _Digits), s.risk / _Point);
   body += StringFormat("Target      : %s   (%.0f pt)\r\n",
                        DoubleToString(s.target, _Digits), s.reward / _Point);
   body += StringFormat("R:R         : %.2f\r\n", s.rr);
   body += StringFormat("ATR(%d)      : %s\r\n", InpATRPeriod, DoubleToString(s.atr, _Digits));
   body += StringFormat("Valid for   : %d bars\r\n", InpValidBars);
   if(InpShowSizing)
     {
      if(lots > 0.0)
         body += StringFormat("Size @%.2f%% : %.2f lots (balance %.2f %s)\r\n",
                              InpRiskPercent, lots,
                              AccountInfoDouble(ACCOUNT_BALANCE),
                              AccountInfoString(ACCOUNT_CURRENCY));
      else
         body += "Size        : below broker minimum at this risk %\r\n";
     }
   body += StringFormat("Spread now  : %d pt\r\n",
                        (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   body += "----------------------------------------\r\n";
   body += "Indicator signal only. Verify before acting.\r\n";

   if(InpAlertPopup)
      Alert(subject);

   if(InpAlertSound)
      PlaySound(InpSoundFile);

   if(InpAlertEmail && !MQLInfoInteger(MQL_TESTER))
     {
      if(!TerminalInfoInteger(TERMINAL_EMAIL_ENABLED))
         Print("WicklessSignalPro: email not enabled in terminal settings, skipped.");
      else if(!SendMail(subject, body))
         Print("WicklessSignalPro: SendMail failed, error ", GetLastError());
      else
         Print("WicklessSignalPro: email sent - ", subject);
     }

   if(InpAlertPush && !MQLInfoInteger(MQL_TESTER))
     {
      string push = StringFormat("%s %s %s | E %s SL %s TP %s | %.2fR",
                                 side, _Symbol, tfStr,
                                 DoubleToString(s.entry,  _Digits),
                                 DoubleToString(s.stop,   _Digits),
                                 DoubleToString(s.target, _Digits),
                                 s.rr);
      if(!SendNotification(push))
         Print("WicklessSignalPro: SendNotification failed, error ", GetLastError());
     }

   if(InpLogCSV)
      LogSignal(s, side, lots);
  }

//+------------------------------------------------------------------+
//| Append the signal to a CSV for forward-test record keeping       |
//+------------------------------------------------------------------+
void LogSignal(const Signal &s, const string side, const double lots)
  {
   string fn = StringFormat("WicklessSignals_%s.csv", _Symbol);
   bool   isNew = !FileIsExist(fn);

   int fh = FileOpen(fn, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_SHARE_READ, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("WicklessSignalPro: CSV log failed, error ", GetLastError());
      return;
     }
   FileSeek(fh, 0, SEEK_END);
   if(isNew)
      FileWrite(fh, "bar_time", "symbol", "tf", "side", "pattern", "mode",
                "entry", "stop", "target", "risk_pt", "reward_pt", "rr",
                "atr", "spread_pt", "lots");

   string tfStr = EnumToString((ENUM_TIMEFRAMES)_Period);
   StringReplace(tfStr, "PERIOD_", "");

   FileWrite(fh,
             TimeToString(s.barTime, TIME_DATE | TIME_MINUTES),
             _Symbol,
             tfStr,
             side,
             s.pattern,
             (InpSignalMode == SIGNAL_FILL ? "fill" : "continuation"),
             DoubleToString(s.entry,  _Digits),
             DoubleToString(s.stop,   _Digits),
             DoubleToString(s.target, _Digits),
             DoubleToString(s.risk   / _Point, 0),
             DoubleToString(s.reward / _Point, 0),
             DoubleToString(s.rr, 2),
             DoubleToString(s.atr, _Digits),
             IntegerToString((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD)),
             DoubleToString(lots, 2));
   FileClose(fh);
  }

//+------------------------------------------------------------------+
//| Main                                                             |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(rates_total < InpATRPeriod + 5)
      return(0);

   ArraySetAsSeries(time,  false);
   ArraySetAsSeries(open,  false);
   ArraySetAsSeries(high,  false);
   ArraySetAsSeries(low,   false);
   ArraySetAsSeries(close, false);

   int closedBar = rates_total - 2;          // last FULLY CLOSED bar
   if(closedBar < InpATRPeriod + 1)
      return(rates_total);

   //--- first load / full recalc: mark history, no alerts ------------
   if(prev_calculated == 0)
     {
      ArrayInitialize(BufBuy,  EMPTY_VALUE);
      ArrayInitialize(BufSell, EMPTY_VALUE);
      ObjectsDeleteAll(0, g_prefix);
      ArrayResize(g_objNames, 0);

      int from = MathMax(InpATRPeriod + 1, rates_total - InpScanBars);
      for(int i = from; i <= closedBar; i++)
        {
         if(!DayAllowed(time[i]) || !SessionAllowed(time[i]))
            continue;
         Signal s = BuildSignal(i, time, open, high, low, close);
         if(s.valid)
            DrawSignal(s, i, rates_total, time, high, low);
        }
      ChartRedraw();
      return(rates_total);
     }

   //--- live: only act once, on a newly closed bar -------------------
   if(time[closedBar] == g_lastSignalBar)
      return(rates_total);

   Signal s = BuildSignal(closedBar, time, open, high, low, close);

   //--- record the bar regardless, so we evaluate it exactly once
   g_lastSignalBar = time[closedBar];
   GlobalVariableSet(g_gvName, (double)g_lastSignalBar);

   if(!s.valid)
      return(rates_total);

   //--- draw first: the chart mark should not depend on alert filters
   DrawSignal(s, closedBar, rates_total, time, high, low);
   ChartRedraw();

   //--- alert gating -------------------------------------------------
   if(!DayAllowed(s.barTime) || !SessionAllowed(s.barTime))
      return(rates_total);

   if(InpMaxSpreadPoints > 0 &&
      (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > InpMaxSpreadPoints)
     {
      Print("WicklessSignalPro: signal suppressed, spread above limit.");
      return(rates_total);
     }

   if(InpMinATRPoints > 0 && (s.atr / _Point) < InpMinATRPoints)
     {
      Print("WicklessSignalPro: signal suppressed, ATR below limit.");
      return(rates_total);
     }

   if(InpCooldownMinutes > 0 &&
      g_lastAlertTime > 0 &&
      (TimeCurrent() - g_lastAlertTime) < InpCooldownMinutes * 60)
     {
      Print("WicklessSignalPro: signal suppressed, cooldown active.");
      return(rates_total);
     }

   g_lastAlertTime = TimeCurrent();
   FireAlerts(s);

   return(rates_total);
  }
//+------------------------------------------------------------------+
