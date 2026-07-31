//+------------------------------------------------------------------+
//|                                  FixedRangeActivityProfile.mq5    |
//|   Fixed Range Activity/Volume Profile, anchored to session breaks |
//|                                                                  |
//|   MT5 port of Dashboard-Pro's src/core/volume_profile.py:         |
//|     - the range is a *session break* (previous FX day, or one of  |
//|       the named kill zones), never a hand-dragged selection, so   |
//|       the POC is reproducible;                                    |
//|     - the session still forming is never profiled;                |
//|     - the value area is the CBOT/TradingView two-row expansion,   |
//|       not a one-row greedy walk;                                  |
//|     - each bar's activity is spread across every price bin its    |
//|       high-low touches;                                           |
//|     - the buy/sell split is inferred from bar direction, not the  |
//|       tape.                                                       |
//|                                                                  |
//|   POC / VAH / VAL are also published on three DRAW_NONE buffers   |
//|   (0 = POC, 1 = VAH, 2 = VAL) so an EA can read them via iCustom. |
//+------------------------------------------------------------------+
#property copyright "Dashboard-Pro"
#property link      "https://github.com/EdCharlesDiesel/Dashboard-Pro"
#property version   "1.00"
#property description "Fixed Range Activity Profile - POC/VAH/VAL from the previous FX day or kill zone."

#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots   3

#property indicator_label1  "FRAP POC"
#property indicator_type1   DRAW_NONE
#property indicator_label2  "FRAP VAH"
#property indicator_type2   DRAW_NONE
#property indicator_label3  "FRAP VAL"
#property indicator_type3   DRAW_NONE

//--- The fixed range to profile.
enum ENUM_FRAP_RANGE
  {
   FRAP_PREV_FX_DAY,      // Previous FX day (rolls at the NY close)
   FRAP_TOKYO,            // Tokyo             00:00-03:00 UTC
   FRAP_LONDON_KZ,        // London kill zone  07:00-09:00 UTC
   FRAP_NY_KZ,            // New York kill zone 12:00-14:00 UTC
   FRAP_LONDON_CLOSE,     // London close      15:00-17:00 UTC
   FRAP_LAST_N_BARS       // Last N bars (manual escape hatch)
  };

//--- Where the per-bin activity number comes from.
enum ENUM_FRAP_ACTIVITY
  {
   FRAP_ACT_AUTO,         // Auto: real volume -> tick volume -> true range
   FRAP_ACT_REAL,         // Real volume (exchange instruments only)
   FRAP_ACT_TICK,         // Tick volume
   FRAP_ACT_RANGE         // True range proxy (matches the dashboard)
  };

//--- Which edge of the range the histogram grows from.
enum ENUM_FRAP_ANCHOR
  {
   FRAP_ANCHOR_START,     // Range start (TradingView / dashboard style)
   FRAP_ANCHOR_END        // Range end (leaves the range's candles clear)
  };

input group                "-- Fixed range --"
input ENUM_TIMEFRAMES      ProfileTimeframe     = PERIOD_CURRENT; // Timeframe the profile is built on
input ENUM_FRAP_RANGE      RangeMode            = FRAP_PREV_FX_DAY; // Fixed range
input int                  SessionsBack         = 1;      // Sessions back (1 = most recent completed)
input int                  LastNBars            = 24;     // Bars in range (Last N bars mode)
input int                  LookbackBars         = 1500;   // Bars of history scanned for the range
input int                  DayBreakHourUTC      = 21;     // FX day break hour, UTC (21 ~ 17:00 New York)

input group                "-- Profile --"
input int                  Bins                 = 48;     // Price bins
input double               ValueAreaPercent     = 70.0;   // Value area %
input ENUM_FRAP_ACTIVITY   ActivitySource       = FRAP_ACT_AUTO; // Activity source

input group                "-- Broker time --"
input bool                 AutoDetectUTCOffset  = true;   // Detect server-vs-UTC offset automatically
input int                  ServerUTCOffsetHours = 3;      // Manual offset: server time = UTC + this

input group                "-- Drawing --"
input ENUM_FRAP_ANCHOR     ProfileAnchor        = FRAP_ANCHOR_START; // Histogram anchor
input double               ProfileWidthPct      = 100.0;  // Widest row, % of the auto width
input bool                 ShowBuySellSplit     = true;   // Split each row into buy/sell
input bool                 ShowRangeBox         = true;   // Shade the fixed range itself
input bool                 ShowSessionBreaks    = true;   // Dotted FX-day separators
input bool                 ExtendLevelsRight    = true;   // Ray POC/VAH/VAL to the right edge
input bool                 ShowInfoPanel        = true;   // Corner readout
input ENUM_BASE_CORNER     PanelCorner          = CORNER_LEFT_UPPER; // Panel corner
input int                  PanelXDistance       = 12;     // Panel X offset (px)
input int                  PanelYDistance       = 18;     // Panel Y offset (px)
input color                BuyColor             = C'0,224,255';  // Buy rows (up-closing bars)
input color                SellColor            = C'255,92,138'; // Sell rows (down-closing bars)
input color                POCColor             = clrGold;       // POC line
input color                VAColor              = clrSilver;     // VAH / VAL lines
input color                BoxColor             = C'0,224,255';  // Range shading
input string               ObjectPrefix         = "FRAP_";       // Object name prefix

//--- Published levels (DRAW_NONE; read from an EA with iCustom).
double BufPOC[];
double BufVAH[];
double BufVAL[];

//--- Sanitised inputs.
int    g_bins            = 48;
double g_vaPct           = 0.70;
int    g_sessionsBack    = 1;
int    g_lastN           = 24;
int    g_lookback        = 1500;
int    g_breakHour       = 21;

//--- The profile itself (globals rather than a struct: MQL5 structs with
//--- dynamic arrays are non-POD and awkward to pass around).
double   g_edges[];
double   g_centers[];
double   g_total[];
double   g_buy[];
int      g_pocIdx  = 0;
int      g_vaLoIdx = 0;
int      g_vaHiIdx = 0;
double   g_poc = 0.0, g_vah = 0.0, g_val = 0.0;
double   g_rangeLow = 0.0, g_rangeHigh = 0.0;
double   g_buyShare = 0.5;
double   g_lastClose = 0.0;
int      g_barsInRange = 0;
bool     g_isProxy = false;
string   g_actKind = "";
datetime g_winStart = 0, g_winEnd = 0;
string   g_winLabel = "";
bool     g_haveProfile = false;
string   g_failReason = "";
int      g_utcOffsetSec = 0;

//--- FX-day break timestamps (server time) for the dotted separators.
datetime g_breaks[];

//--- Recompute trigger: the profile timeframe's last bar.
datetime g_lastProfileBar = 0;

//--- Redraw bookkeeping. `g_drawing` is a re-entrancy guard: Redraw() ends in
//--- ChartRedraw(), and OnChartEvent must never be able to loop back into it.
//--- `g_lastVisibleBars` throttles zoom redraws - row widths scale with the
//--- visible span, so scrolling (which leaves that count alone) needs nothing.
bool g_drawing         = false;
long g_lastVisibleBars = 0;

//--- Panel lines, so labels can stack downward from an upper corner and
//--- upward from a lower one.
#define FRAP_PANEL_LINES 10

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, BufPOC, INDICATOR_DATA);
   SetIndexBuffer(1, BufVAH, INDICATOR_DATA);
   SetIndexBuffer(2, BufVAL, INDICATOR_DATA);
   for(int p = 0; p < 3; p++)
     {
      PlotIndexSetDouble(p, PLOT_EMPTY_VALUE, EMPTY_VALUE);
      PlotIndexSetInteger(p, PLOT_DRAW_TYPE, DRAW_NONE);
     }
   ArraySetAsSeries(BufPOC, false);
   ArraySetAsSeries(BufVAH, false);
   ArraySetAsSeries(BufVAL, false);
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

   if(Bins < 2)
     {
      Print("FRAP: Bins must be >= 2.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(ValueAreaPercent < 10.0 || ValueAreaPercent > 99.0)
     {
      Print("FRAP: ValueAreaPercent must be between 10 and 99.");
      return(INIT_PARAMETERS_INCORRECT);
     }

   g_bins         = (int)MathMin(Bins, 250);
   g_vaPct        = ValueAreaPercent / 100.0;
   g_sessionsBack = (int)MathMax(1, SessionsBack);
   g_lastN        = (int)MathMax(2, LastNBars);
   g_lookback     = (int)MathMin(MathMax(LookbackBars, 120), 50000);
   g_breakHour    = (int)MathMin(MathMax(DayBreakHourUTC, 0), 23);

   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("FRAP %s (%d bins, %.0f%%)",
                                   RangeModeName(RangeMode), g_bins, ValueAreaPercent));
   g_lastProfileBar = 0;
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, ObjectPrefix);
   ChartRedraw();
  }
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
   if(rates_total < 2)
      return(0);
   ArraySetAsSeries(time, false);

   ENUM_TIMEFRAMES tf = ProfileTF();
   datetime lastBar = (datetime)SeriesInfoInteger(_Symbol, tf, SERIES_LASTBAR_DATE);

   bool rebuilt = false;
   if(prev_calculated == 0 || lastBar == 0 || lastBar != g_lastProfileBar)
     {
      g_lastProfileBar = lastBar;
      g_haveProfile = BuildProfile(tf);
      Redraw();
      rebuilt = true;
     }

   int start = (rebuilt || prev_calculated == 0) ? 0 : prev_calculated - 1;
   for(int i = start; i < rates_total; i++)
     {
      if(g_haveProfile && time[i] >= g_winStart)
        {
         BufPOC[i] = g_poc;
         BufVAH[i] = g_vah;
         BufVAL[i] = g_val;
        }
      else
        {
         BufPOC[i] = EMPTY_VALUE;
         BufVAH[i] = EMPTY_VALUE;
         BufVAL[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+
//| Redraw the chart on every user zoom/scroll: the histogram's row  |
//| widths are scaled to the visible span, so they must follow it.   |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam,
                  const string &sparam)
  {
   if(id != CHARTEVENT_CHART_CHANGE || !g_haveProfile || g_drawing)
      return;
   long vis = ChartGetInteger(0, CHART_VISIBLE_BARS);
   if(vis == g_lastVisibleBars)
      return;                       // scrolled, not zoomed - widths unchanged
   Redraw();
  }

//==================================================================//
//  Time                                                            //
//==================================================================//
ENUM_TIMEFRAMES ProfileTF()
  {
   return((ProfileTimeframe == PERIOD_CURRENT) ? (ENUM_TIMEFRAMES)Period() : ProfileTimeframe);
  }
//+------------------------------------------------------------------+
//| Seconds to subtract from a server timestamp to get UTC.          |
//|                                                                  |
//| MT5 bar times are broker server time (commonly UTC+2/+3), while  |
//| every session window here is defined in UTC. Getting this wrong  |
//| shifts the whole indicator by hours, so it is detected from the  |
//| terminal (TimeCurrent is server, TimeGMT is UTC) and rounded to  |
//| the nearest half hour.                                           |
//+------------------------------------------------------------------+
int UTCOffsetSeconds()
  {
   if(!AutoDetectUTCOffset)
      return(ServerUTCOffsetHours * 3600);
   long diff = (long)TimeCurrent() - (long)TimeGMT();
   return((int)(MathRound((double)diff / 1800.0) * 1800.0));
  }
//+------------------------------------------------------------------+
//| The FX day a UTC timestamp belongs to (as a day number).         |
//|                                                                  |
//| Bars from the break hour onward belong to the *next* calendar    |
//| day's session, which is why the whole timestamp is shifted back  |
//| before flooring: 21:00 Monday and 08:00 Tuesday are one FX day.  |
//+------------------------------------------------------------------+
long SessionDayNum(datetime utc, int breakHour)
  {
   long shifted = (long)utc - (long)breakHour * 3600;
   if(shifted < 0)
      shifted = 0;
   return(shifted / 86400);
  }
//+------------------------------------------------------------------+
double HourOfDay(datetime utc)
  {
   MqlDateTime st;
   TimeToStruct(utc, st);
   return((double)st.hour + (double)st.min / 60.0);
  }
//+------------------------------------------------------------------+
string RangeModeName(ENUM_FRAP_RANGE m)
  {
   switch(m)
     {
      case FRAP_PREV_FX_DAY:   return("Previous FX day");
      case FRAP_TOKYO:         return("Tokyo");
      case FRAP_LONDON_KZ:     return("London KZ");
      case FRAP_NY_KZ:         return("NY KZ");
      case FRAP_LONDON_CLOSE:  return("London Close");
      case FRAP_LAST_N_BARS:   return("Last N bars");
     }
   return("?");
  }
//+------------------------------------------------------------------+
//| Kill-zone windows, UTC. Mirrors SessionService in the dashboard  |
//| (src/services/session_service.py) - keep the two in step.        |
//+------------------------------------------------------------------+
bool SessionWindow(ENUM_FRAP_RANGE m, double &lo, double &hi)
  {
   switch(m)
     {
      case FRAP_TOKYO:        lo = 0.0;  hi = 3.0;  return(true);
      case FRAP_LONDON_KZ:    lo = 7.0;  hi = 9.0;  return(true);
      case FRAP_NY_KZ:        lo = 12.0; hi = 14.0; return(true);
      case FRAP_LONDON_CLOSE: lo = 15.0; hi = 17.0; return(true);
     }
   lo = 0.0;
   hi = 0.0;
   return(false);
  }

//==================================================================//
//  The profile                                                     //
//==================================================================//
//+------------------------------------------------------------------+
//| Resolve the fixed range, then bin it. Returns false (with        |
//| g_failReason set for the panel) when there is nothing to profile.|
//+------------------------------------------------------------------+
bool BuildProfile(ENUM_TIMEFRAMES tf)
  {
   g_failReason = "";
   ArrayFree(g_breaks);

   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, tf, 0, g_lookback, rates);
   if(copied < 10)
     {
      g_failReason = "Not enough history loaded on " + EnumToString(tf) +
                     " (got " + IntegerToString((int)MathMax(copied, 0)) + " bars).";
      return(false);
     }

   g_utcOffsetSec = UTCOffsetSeconds();
   g_lastClose = rates[copied - 1].close;

   //--- UTC timestamps and FX-day numbers for every loaded bar.
   datetime utc[];
   long     days[];
   ArrayResize(utc, copied);
   ArrayResize(days, copied);
   for(int i = 0; i < copied; i++)
     {
      utc[i]  = (datetime)((long)rates[i].time - g_utcOffsetSec);
      days[i] = SessionDayNum(utc[i], g_breakHour);
     }

   CollectSessionBreaks(rates, days, copied);

   int i0 = -1, i1 = -1;
   if(!ResolveRange(rates, utc, days, copied, i0, i1))
      return(false);
   if(i0 < 0 || i1 < i0)
      return(false);

   return(BinRange(rates, i0, i1));
  }
//+------------------------------------------------------------------+
//| Bar timestamps (server time) where a new FX day starts.          |
//+------------------------------------------------------------------+
void CollectSessionBreaks(const MqlRates &rates[], const long &days[], int n)
  {
   ArrayResize(g_breaks, 0);
   for(int i = 1; i < n; i++)
      if(days[i] != days[i - 1])
        {
         int k = ArraySize(g_breaks);
         ArrayResize(g_breaks, k + 1);
         g_breaks[k] = rates[i].time;
        }
  }
//+------------------------------------------------------------------+
//| Pick the bar span [i0, i1] for the configured range mode.        |
//|                                                                  |
//| The session still forming is never selected - its POC would move |
//| on every tick, which is the opposite of the reproducible level   |
//| this indicator exists to provide.                                |
//+------------------------------------------------------------------+
bool ResolveRange(const MqlRates &rates[], const datetime &utc[], const long &days[],
                  int n, int &i0, int &i1)
  {
   if(RangeMode == FRAP_LAST_N_BARS)
     {
      i0 = (int)MathMax(0, n - g_lastN);
      i1 = n - 1;
      g_winStart = rates[i0].time;
      g_winEnd   = rates[i1].time;
      g_winLabel = StringFormat("last %d bars", i1 - i0 + 1);
      return(true);
     }

   if(RangeMode == FRAP_PREV_FX_DAY)
     {
      //--- Unique FX days, oldest first. days[] is non-decreasing, so a
      //--- single pass is enough.
      long uniq[];
      int  nu = 0;
      for(int i = 0; i < n; i++)
         if(nu == 0 || days[i] != uniq[nu - 1])
           {
            ArrayResize(uniq, nu + 1);
            uniq[nu] = days[i];
            nu++;
           }
      //--- uniq[nu-1] is the day still forming; step back from there.
      int pos = nu - 1 - g_sessionsBack;
      if(pos < 0)
        {
         g_failReason = StringFormat("Only %d completed FX day(s) in %d bars of %s - "
                                     "lower 'Sessions back' or raise 'Lookback bars'.",
                                     (int)MathMax(nu - 1, 0), n, EnumToString(ProfileTF()));
         return(false);
        }
      long target = uniq[pos];
      i0 = -1;
      i1 = -1;
      for(int i = 0; i < n; i++)
         if(days[i] == target)
           {
            if(i0 < 0)
               i0 = i;
            i1 = i;
           }
      if(i0 < 0)
         return(false);
      g_winStart = rates[i0].time;
      g_winEnd   = rates[i1].time;
      g_winLabel = "FX day " + TimeToString((datetime)(target * 86400), TIME_DATE);
      return(true);
     }

   //--- Named kill zone: every contiguous occurrence, oldest first.
   double lo = 0.0, hi = 0.0;
   if(!SessionWindow(RangeMode, lo, hi))
      return(false);

   int  bStart[];
   int  bEnd[];
   int  nb = 0;
   long currentDay = 0;
   bool haveBlock = false;
   for(int i = 0; i < n; i++)
     {
      double h = HourOfDay(utc[i]);
      if(!(h >= lo && h < hi))
         continue;
      if(!haveBlock || days[i] != currentDay)
        {
         ArrayResize(bStart, nb + 1);
         ArrayResize(bEnd,   nb + 1);
         bStart[nb] = i;
         bEnd[nb]   = i;
         nb++;
         currentDay = days[i];
         haveBlock = true;
        }
      else
         bEnd[nb - 1] = i;
     }
   //--- A block whose last bar is also the last loaded bar is in progress.
   if(nb > 0 && bEnd[nb - 1] == n - 1)
      nb--;
   if(nb < g_sessionsBack)
     {
      g_failReason = StringFormat("Only %d completed %s window(s) in %d bars of %s - "
                                  "lower 'Sessions back' or raise 'Lookback bars'.",
                                  nb, RangeModeName(RangeMode), n, EnumToString(ProfileTF()));
      return(false);
     }
   i0 = bStart[nb - g_sessionsBack];
   i1 = bEnd[nb - g_sessionsBack];
   g_winStart = rates[i0].time;
   g_winEnd   = rates[i1].time;
   g_winLabel = RangeModeName(RangeMode) + " " +
                TimeToString((datetime)utc[i0], TIME_DATE);
   return(true);
  }
//+------------------------------------------------------------------+
//| Which activity number to use for this range.                     |
//|                                                                  |
//| Spot FX has no real volume (OTC), so the honest fallbacks are    |
//| tick volume, then per-bar true range. `is_proxy` in the dashboard|
//| is g_isProxy here: true means the bar lengths are ranges/ticks,  |
//| not contracts. The *prices* the POC identifies are meaningful    |
//| either way.                                                      |
//+------------------------------------------------------------------+
ENUM_FRAP_ACTIVITY ResolveActivitySource(const MqlRates &rates[], int i0, int i1)
  {
   double sumReal = 0.0, sumTick = 0.0;
   for(int i = i0; i <= i1; i++)
     {
      sumReal += (double)rates[i].real_volume;
      sumTick += (double)rates[i].tick_volume;
     }
   if(ActivitySource == FRAP_ACT_REAL && sumReal > 0.0)
      return(FRAP_ACT_REAL);
   if(ActivitySource == FRAP_ACT_TICK && sumTick > 0.0)
      return(FRAP_ACT_TICK);
   if(ActivitySource == FRAP_ACT_RANGE)
      return(FRAP_ACT_RANGE);
   if(ActivitySource == FRAP_ACT_AUTO)
     {
      if(sumReal > 0.0)
         return(FRAP_ACT_REAL);
      if(sumTick > 0.0)
         return(FRAP_ACT_TICK);
     }
   return(FRAP_ACT_RANGE);   // forced source was empty -> proxy
  }
//+------------------------------------------------------------------+
//| One bar's activity. `j` is the bar's position inside the range:  |
//| j == 0 has no previous close, so its true range is high-low      |
//| (same as the dashboard, which slices before computing TR).       |
//+------------------------------------------------------------------+
double BarActivity(const MqlRates &rates[], int i, int j, ENUM_FRAP_ACTIVITY src)
  {
   if(src == FRAP_ACT_REAL)
      return((double)rates[i].real_volume);
   if(src == FRAP_ACT_TICK)
      return((double)rates[i].tick_volume);
   double h = rates[i].high, l = rates[i].low;
   double tr = h - l;
   if(j > 0)
     {
      double pc = rates[i - 1].close;
      tr = MathMax(tr, MathMax(MathAbs(h - pc), MathAbs(l - pc)));
     }
   return(tr);
  }
//+------------------------------------------------------------------+
//| Index of the bin holding `x` - the last edge <= x, matching      |
//| numpy's searchsorted(edges, x, "right") - 1, then clipped.       |
//+------------------------------------------------------------------+
int BinIndex(const double &edges[], int bins, double x)
  {
   double w = edges[1] - edges[0];
   if(w <= 0.0)
      return(0);
   int i = (int)MathFloor((x - edges[0]) / w);
   if(i < 0)
      i = 0;
   if(i > bins - 1)
      i = bins - 1;
   while(i > 0 && edges[i] > x)
      i--;
   while(i < bins - 1 && edges[i + 1] <= x)
      i++;
   return(i);
  }
//+------------------------------------------------------------------+
//| Bin [i0, i1] into g_total / g_buy and derive POC, VAH, VAL.      |
//|                                                                  |
//| Each bar's activity is spread evenly across every price bin its  |
//| high-low touches - the standard distribution, and what keeps the |
//| MT5 profile agreeing with the dashboard's.                       |
//+------------------------------------------------------------------+
bool BinRange(const MqlRates &rates[], int i0, int i1)
  {
   double lo = rates[i0].low, hi = rates[i0].high;
   for(int i = i0; i <= i1; i++)
     {
      lo = MathMin(lo, rates[i].low);
      hi = MathMax(hi, rates[i].high);
     }
   if(!(hi > lo))
     {
      g_failReason = "The selected range has no price extent.";
      return(false);
     }

   ENUM_FRAP_ACTIVITY src = ResolveActivitySource(rates, i0, i1);
   g_isProxy = (src == FRAP_ACT_RANGE);
   g_actKind = (src == FRAP_ACT_REAL) ? "real volume"
               : (src == FRAP_ACT_TICK) ? "tick volume"
               : "true range proxy";
   if(src != ActivitySource && ActivitySource != FRAP_ACT_AUTO)
      g_actKind += " (requested source empty)";

   ArrayResize(g_edges,   g_bins + 1);
   ArrayResize(g_centers, g_bins);
   ArrayResize(g_total,   g_bins);
   ArrayResize(g_buy,     g_bins);
   ArrayInitialize(g_total, 0.0);
   ArrayInitialize(g_buy,   0.0);

   double step = (hi - lo) / (double)g_bins;
   for(int b = 0; b <= g_bins; b++)
      g_edges[b] = lo + step * b;
   for(int b = 0; b < g_bins; b++)
      g_centers[b] = (g_edges[b] + g_edges[b + 1]) / 2.0;

   double buySum = 0.0, allSum = 0.0;
   for(int i = i0; i <= i1; i++)
     {
      double v = BarActivity(rates, i, i - i0, src);
      if(!MathIsValidNumber(v) || v <= 0.0)
         continue;
      int loI = BinIndex(g_edges, g_bins, rates[i].low);
      int hiI = BinIndex(g_edges, g_bins, rates[i].high);
      int span = hiI - loI + 1;
      double share = v / (double)span;
      bool up = (rates[i].close >= rates[i].open);
      for(int b = loI; b <= hiI; b++)
        {
         g_total[b] += share;
         if(up)
            g_buy[b] += share;
        }
      allSum += v;
      if(up)
         buySum += v;
     }
   if(allSum <= 0.0)
     {
      g_failReason = "No activity recorded in the selected range.";
      return(false);
     }

   ValueArea(g_total, g_bins, g_vaPct, g_vaLoIdx, g_pocIdx, g_vaHiIdx);
   g_poc        = g_centers[g_pocIdx];
   g_vah        = g_edges[g_vaHiIdx + 1];
   g_val        = g_edges[g_vaLoIdx];
   g_rangeLow   = lo;
   g_rangeHigh  = hi;
   g_barsInRange = i1 - i0 + 1;
   g_buyShare   = buySum / allSum;
   return(true);
  }
//+------------------------------------------------------------------+
//| Value area: start at the POC and repeatedly annex whichever      |
//| *pair* of adjacent bins carries more activity, until `pct` of    |
//| the total is enclosed.                                           |
//|                                                                  |
//| Taking two rows at a time is the CBOT convention TradingView     |
//| follows - a one-row-at-a-time greedy walk drifts the band toward |
//| whichever side happens to have the noisier single bin. -1 is a   |
//| "no room left on this side" sentinel; activity is never < 0.     |
//+------------------------------------------------------------------+
void ValueArea(const double &v[], int n, double pct, int &outLo, int &outPoc, int &outHi)
  {
   outLo = 0;
   outPoc = 0;
   outHi = 0;
   if(n <= 0)
      return;

   int poc = ArrayMaximum(v, 0, n);
   if(poc < 0)
      poc = 0;
   outPoc = poc;
   outLo = poc;
   outHi = poc;

   double total = 0.0;
   for(int i = 0; i < n; i++)
      total += v[i];
   if(total <= 0.0)
      return;

   double target = total * pct;
   double acc = v[poc];
   int lo = poc, hi = poc;
   while(acc < target && (lo > 0 || hi < n - 1))
     {
      int upHi = hi;
      double up = -1.0;
      if(hi < n - 1)
        {
         upHi = (int)MathMin(hi + 2, n - 1);
         up = 0.0;
         for(int i = hi + 1; i <= upHi; i++)
            up += v[i];
        }
      int dnLo = lo;
      double dn = -1.0;
      if(lo > 0)
        {
         dnLo = (int)MathMax(lo - 2, 0);
         dn = 0.0;
         for(int i = dnLo; i < lo; i++)
            dn += v[i];
        }
      if(up < 0.0 && dn < 0.0)
         break;
      if(up >= dn)
        {
         acc += up;
         hi = upHi;
        }
      else
        {
         acc += dn;
         lo = dnLo;
        }
     }
   outLo = lo;
   outHi = hi;
  }

//==================================================================//
//  Reading the levels                                              //
//==================================================================//
//+------------------------------------------------------------------+
//| Classify price against the value area.                           |
//|                                                                  |
//| Acceptance *outside* the previous value area is directional      |
//| information; rotation *inside* it is the market agreeing with    |
//| yesterday's price, and is the classic chop warning.              |
//+------------------------------------------------------------------+
void ReadProfile(double price, string &state, string &lean, color &tint)
  {
   if(price > g_vah)
     {
      state = "ABOVE VALUE";
      lean  = "Bullish - accepting above prior value";
      tint  = clrLime;
      return;
     }
   if(price < g_val)
     {
      state = "BELOW VALUE";
      lean  = "Bearish - accepting below prior value";
      tint  = clrTomato;
      return;
     }
   state = "IN VALUE";
   bool nearPoc = (MathAbs(price - g_poc) <= BinWidth() * 1.5);
   lean = nearPoc ? "Neutral - rotating, sitting on the POC"
          : "Neutral - rotating inside prior value";
   tint = clrSilver;
  }
//+------------------------------------------------------------------+
double BinWidth()
  {
   if(ArraySize(g_edges) < 2)
      return(0.0);
   return(g_edges[1] - g_edges[0]);
  }
//+------------------------------------------------------------------+
//| One pip: 10 points on 3/5-digit quotes, 1 point otherwise.       |
//+------------------------------------------------------------------+
double PipSize()
  {
   return(_Point * ((_Digits == 3 || _Digits == 5) ? 10.0 : 1.0));
  }

//==================================================================//
//  Drawing                                                         //
//==================================================================//
//+------------------------------------------------------------------+
//| Linear blend of two colors; used to fake opacity, which chart    |
//| objects do not have. t = 1 returns `b`.                          |
//+------------------------------------------------------------------+
color Blend(color a, color b, double t)
  {
   uint ca = (uint)a, cb = (uint)b;
   int a0 = (int)(ca & 0xFF), a1 = (int)((ca >> 8) & 0xFF), a2 = (int)((ca >> 16) & 0xFF);
   int b0 = (int)(cb & 0xFF), b1 = (int)((cb >> 8) & 0xFF), b2 = (int)((cb >> 16) & 0xFF);
   int r0 = (int)MathRound(a0 + (b0 - a0) * t);
   int r1 = (int)MathRound(a1 + (b1 - a1) * t);
   int r2 = (int)MathRound(a2 + (b2 - a2) * t);
   return((color)((uint)r0 | ((uint)r1 << 8) | ((uint)r2 << 16)));
  }
//+------------------------------------------------------------------+
color ChartBackground()
  {
   return((color)ChartGetInteger(0, CHART_COLOR_BACKGROUND));
  }
//+------------------------------------------------------------------+
void MakeRect(string name, datetime t1, double p1, datetime t2, double p2, color c)
  {
   if(t2 <= t1 || p2 <= p1)
      return;
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 0, p1);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p2);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTED, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }
//+------------------------------------------------------------------+
void MakeLevel(string name, datetime t1, datetime t2, double price,
               color c, int width, ENUM_LINE_STYLE style, bool ray)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 0, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 1, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT, false);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, ray);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }
//+------------------------------------------------------------------+
void MakeText(string name, datetime t, double price, string text, color c)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t);
   ObjectSetDouble(0, name, OBJPROP_PRICE, 0, price);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }
//+------------------------------------------------------------------+
void MakeVLine(string name, datetime t, color c)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_VLINE, 0, t, 0);
   ObjectSetInteger(0, name, OBJPROP_TIME, 0, t);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }
//+------------------------------------------------------------------+
void MakeLabel(string name, int line, string text, color c)
  {
   bool right = (PanelCorner == CORNER_RIGHT_UPPER || PanelCorner == CORNER_RIGHT_LOWER);
   bool lower = (PanelCorner == CORNER_LEFT_LOWER  || PanelCorner == CORNER_RIGHT_LOWER);
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   int slot = lower ? (FRAP_PANEL_LINES - 1 - line) : line;
   ObjectSetInteger(0, name, OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, PanelXDistance);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, PanelYDistance + slot * 14);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,
                    right ? (lower ? ANCHOR_RIGHT_LOWER : ANCHOR_RIGHT_UPPER)
                    : (lower ? ANCHOR_LEFT_LOWER  : ANCHOR_LEFT_UPPER));
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }
//+------------------------------------------------------------------+
//| Rebuild every chart object from the current profile.             |
//+------------------------------------------------------------------+
void Redraw()
  {
   if(g_drawing)
      return;
   g_drawing = true;
   g_lastVisibleBars = ChartGetInteger(0, CHART_VISIBLE_BARS);

   ObjectsDeleteAll(0, ObjectPrefix);
   if(!g_haveProfile)
     {
      if(ShowInfoPanel)
        {
         MakeLabel(ObjectPrefix + "P0", 0, "FIXED RANGE ACTIVITY PROFILE", clrSilver);
         MakeLabel(ObjectPrefix + "P1", 1,
                   (g_failReason == "" ? "No profile available." : g_failReason), clrTomato);
        }
      ChartRedraw();
      g_drawing = false;
      return;
     }

   color bg = ChartBackground();
   datetime rightEdge = (datetime)SeriesInfoInteger(_Symbol, (ENUM_TIMEFRAMES)Period(),
                                                    SERIES_LASTBAR_DATE);
   if(rightEdge < g_winEnd)
      rightEdge = g_winEnd;

   //--- Row width: the range's own duration, with a floor proportional to the
   //--- visible span so a two-hour kill zone is not a sliver on a week of bars.
   double rangeDur = (double)((long)g_winEnd - (long)g_winStart);
   double visSpan  = (double)ChartGetInteger(0, CHART_VISIBLE_BARS) *
                     (double)PeriodSeconds((ENUM_TIMEFRAMES)Period());
   double maxWidth = MathMax(rangeDur * 0.9, visSpan * 0.12) * (ProfileWidthPct / 100.0);
   if(maxWidth < 1.0)
      maxWidth = (double)PeriodSeconds((ENUM_TIMEFRAMES)Period()) * 10.0;

   datetime anchor = (ProfileAnchor == FRAP_ANCHOR_START) ? g_winStart : g_winEnd;

   //--- Shaded box marking the fixed range itself.
   if(ShowRangeBox)
      MakeRect(ObjectPrefix + "BOX", g_winStart, g_rangeLow, g_winEnd, g_rangeHigh,
               Blend(BoxColor, bg, 0.90));

   //--- FX-day separators, limited to the neighbourhood of the range.
   if(ShowSessionBreaks)
     {
      datetime from = (datetime)((long)g_winStart - (long)MathMax(rangeDur * 2.0, 86400.0));
      int drawn = 0;
      for(int i = 0; i < ArraySize(g_breaks) && drawn < 40; i++)
         if(g_breaks[i] >= from)
           {
            MakeVLine(ObjectPrefix + "BRK" + IntegerToString(i), g_breaks[i],
                      Blend(clrSilver, bg, 0.55));
            drawn++;
           }
     }

   //--- Histogram rows. Rows outside the value area are dimmed, same as
   //--- TradingView: the eye should land on the 70% band, not on the tails.
   double peak = 0.0;
   for(int b = 0; b < g_bins; b++)
      peak = MathMax(peak, g_total[b]);
   if(peak > 0.0)
     {
      for(int b = 0; b < g_bins; b++)
        {
         double tot = g_total[b];
         if(tot <= 0.0)
            continue;
         bool inVA = (b >= g_vaLoIdx && b <= g_vaHiIdx);
         double dim = inVA ? 0.0 : 0.62;
         double pad = (g_edges[b + 1] - g_edges[b]) * 0.12;
         double y0 = g_edges[b] + pad / 2.0;
         double y1 = g_edges[b + 1] - pad / 2.0;
         double w  = maxWidth * (tot / peak);

         if(ShowBuySellSplit)
           {
            double bw = w * (g_buy[b] / tot);
            datetime xMid = (datetime)((long)anchor + (long)MathRound(bw));
            datetime xEnd = (datetime)((long)anchor + (long)MathRound(w));
            MakeRect(ObjectPrefix + "R" + IntegerToString(b) + "B",
                     anchor, y0, xMid, y1, Blend(BuyColor, bg, dim));
            MakeRect(ObjectPrefix + "R" + IntegerToString(b) + "S",
                     xMid, y0, xEnd, y1, Blend(SellColor, bg, dim));
           }
         else
           {
            datetime xEnd = (datetime)((long)anchor + (long)MathRound(w));
            color c = (g_buy[b] >= tot * 0.5) ? BuyColor : SellColor;
            MakeRect(ObjectPrefix + "R" + IntegerToString(b),
                     anchor, y0, xEnd, y1, Blend(c, bg, dim));
           }
        }
     }

   //--- POC / VAH / VAL carried forward past the range - the whole reason to
   //--- profile a *previous* session.
   int dg = _Digits;
   MakeLevel(ObjectPrefix + "POC", g_winStart, rightEdge, g_poc, POCColor, 2, STYLE_SOLID,
             ExtendLevelsRight);
   MakeLevel(ObjectPrefix + "VAH", g_winStart, rightEdge, g_vah, VAColor, 1, STYLE_DASH,
             ExtendLevelsRight);
   MakeLevel(ObjectPrefix + "VAL", g_winStart, rightEdge, g_val, VAColor, 1, STYLE_DASH,
             ExtendLevelsRight);
   MakeText(ObjectPrefix + "TPOC", rightEdge, g_poc, "  POC " + DoubleToString(g_poc, dg), POCColor);
   MakeText(ObjectPrefix + "TVAH", rightEdge, g_vah, "  VAH " + DoubleToString(g_vah, dg), VAColor);
   MakeText(ObjectPrefix + "TVAL", rightEdge, g_val, "  VAL " + DoubleToString(g_val, dg), VAColor);

   if(ShowInfoPanel)
      DrawPanel();
   ChartRedraw();
   g_drawing = false;
  }
//+------------------------------------------------------------------+
void DrawPanel()
  {
   string state, lean;
   color  tint;
   ReadProfile(g_lastClose, state, lean, tint);

   double pip = PipSize();
   int    dg  = _Digits;
   int    ln  = 0;

   MakeLabel(ObjectPrefix + "P0", ln++, "FIXED RANGE ACTIVITY PROFILE", clrSilver);
   MakeLabel(ObjectPrefix + "P1", ln++,
             StringFormat("Range   %s  (%d x %s bars)", g_winLabel, g_barsInRange,
                          TFName(ProfileTF())), clrSilver);
   MakeLabel(ObjectPrefix + "P2", ln++,
             StringFormat("Source  %s%s", g_actKind, (g_isProxy ? "  [proxy]" : "")),
             g_isProxy ? clrGoldenrod : clrSilver);
   MakeLabel(ObjectPrefix + "P3", ln++,
             StringFormat("POC     %s   (%+.1f pips away)", DoubleToString(g_poc, dg),
                          (g_lastClose - g_poc) / pip), POCColor);
   MakeLabel(ObjectPrefix + "P4", ln++,
             StringFormat("VAH     %s", DoubleToString(g_vah, dg)), VAColor);
   MakeLabel(ObjectPrefix + "P5", ln++,
             StringFormat("VAL     %s", DoubleToString(g_val, dg)), VAColor);
   MakeLabel(ObjectPrefix + "P6", ln++,
             StringFormat("Value area %.0f%%  =  %.1f pips wide", g_vaPct * 100.0,
                          (g_vah - g_val) / pip), clrSilver);
   MakeLabel(ObjectPrefix + "P7", ln++,
             StringFormat("Buy share  %.0f%%   (inferred from bar direction)",
                          g_buyShare * 100.0), clrSilver);
   MakeLabel(ObjectPrefix + "P8", ln++, state + " - " + lean, tint);
   if(g_barsInRange < 4)
      MakeLabel(ObjectPrefix + "P9", ln++,
                StringFormat("Warning: only %d bars in range - drop to a lower timeframe",
                             g_barsInRange), clrTomato);
  }
//+------------------------------------------------------------------+
string TFName(ENUM_TIMEFRAMES tf)
  {
   string s = EnumToString(tf);
   StringReplace(s, "PERIOD_", "");
   return(s);
  }
//+------------------------------------------------------------------+
