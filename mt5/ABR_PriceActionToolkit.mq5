//+------------------------------------------------------------------+
//|                                    ABR_PriceActionToolkit.mq5     |
//|   ABR Price Action Toolkit — structure, order blocks, trade plan  |
//|                                                                  |
//|   MT5 port of Dashboard-Pro's pages/abr_toolkit_tab.py            |
//|   (run_engine). Same pipeline, same numbers:                      |
//|     - swings are strict unique extremes over +/-SwingLen bars;    |
//|     - a close beyond the last confirmed swing is BoS (trend       |
//|       continues) or CHoCH (trend flips), graded Strong/Weak by    |
//|       penetration vs ATR;                                         |
//|     - the order block is the last opposing candle before the      |
//|       impulse, mitigated when price closes back through it, aged  |
//|       out after OBMaxAge bars, capped per side;                   |
//|     - auto trendlines through the last two swings, with break     |
//|       detection inside a lookback window;                         |
//|     - multi-timeframe EMA bias feeds a quality score and grade;   |
//|     - the trade plan is entry / SL / TP1-3 at fixed R multiples.  |
//|                                                                  |
//|   Signal / entry / SL / TP1-3 / quality are published on          |
//|   DRAW_NONE buffers (0..6) so an EA can read them via iCustom     |
//|   without duplicating the engine.                                 |
//|                                                                  |
//|   THIS IS AN INDICATOR. It never places, modifies or closes an    |
//|   order — it reads the chart and draws.                           |
//+------------------------------------------------------------------+
#property copyright "Dashboard-Pro"
#property link      "https://github.com/EdCharlesDiesel/Dashboard-Pro"
#property version   "1.00"
#property description "ABR Price Action Toolkit - BoS/CHoCH, order blocks, MTF bias, graded trade plan."

#property indicator_chart_window
#property indicator_buffers 7
#property indicator_plots   7

#property indicator_label1  "ABR Signal"
#property indicator_type1   DRAW_NONE
#property indicator_label2  "ABR Entry"
#property indicator_type2   DRAW_NONE
#property indicator_label3  "ABR SL"
#property indicator_type3   DRAW_NONE
#property indicator_label4  "ABR TP1"
#property indicator_type4   DRAW_NONE
#property indicator_label5  "ABR TP2"
#property indicator_type5   DRAW_NONE
#property indicator_label6  "ABR TP3"
#property indicator_type6   DRAW_NONE
#property indicator_label7  "ABR Quality"
#property indicator_type7   DRAW_NONE

input group                "-- Structure --"
input int                  SwingLength      = 5;      // Swing length (bars each side of a pivot)
input int                  MaxBars          = 1500;   // Bars analysed (0 = all available)
input int                  ATRPeriod        = 14;     // ATR period (break strength + SL buffer)
input double               StrongBreakATR   = 0.5;    // Break >= this x ATR is "Trend Strong"

input group                "-- Order blocks --"
input int                  OBSearchBars     = 15;     // Bars searched back for the origin candle
input int                  OBMaxAgeBars     = 300;    // Auto-expire an unmitigated OB after N bars
input int                  MaxLiveOBs       = 2;      // Live order blocks kept per side

input group                "-- Trendlines --"
input int                  TLLookbackBars   = 10;     // Bars in which a trendline break still counts

input group                "-- Multi-timeframe bias --"
input ENUM_TIMEFRAMES      HTF1             = PERIOD_H4; // HTF #1 (PERIOD_CURRENT disables)
input ENUM_TIMEFRAMES      HTF2             = PERIOD_D1; // HTF #2 (PERIOD_CURRENT disables)
input ENUM_TIMEFRAMES      HTF3             = PERIOD_W1; // HTF #3 (PERIOD_CURRENT disables)
input int                  MAFast           = 21;     // EMA fast (bias)
input int                  MASlow           = 50;     // EMA slow (bias)

input group                "-- Trade plan --"
input double               SLBufferATR      = 0.5;    // SL buffer beyond structure, in ATR
input double               TP1R             = 1.0;    // TP1 (R multiple)
input double               TP2R             = 2.0;    // TP2 (R multiple)
input double               TP3R             = 2.83;   // TP3 (R multiple)
input double               RiskPerTrade     = 30.0;   // Risk per trade (account currency)

input group                "-- Drawing --"
input bool                 ShowSwings       = true;   // Mark swing highs/lows
input bool                 ShowStructure    = true;   // Draw BoS / CHoCH levels
input bool                 ShowOrderBlocks  = true;   // Draw live order blocks
input bool                 ShowTrendlines   = true;   // Draw auto trendlines
input bool                 ShowTradePlan    = true;   // Draw entry / SL / TP lines
input bool                 ShowInfoPanel    = true;   // Corner readout
input ENUM_BASE_CORNER     PanelCorner      = CORNER_LEFT_UPPER; // Panel corner
input int                  PanelXDistance   = 12;     // Panel X offset (px)
input int                  PanelYDistance   = 18;     // Panel Y offset (px)
input color                BullColor        = C'0,224,120';  // Bullish structure
input color                BearColor        = C'255,92,138';  // Bearish structure
input color                OBBullColor      = C'0,60,30';    // Bullish order block fill
input color                OBBearColor      = C'70,15,25';   // Bearish order block fill
input color                TextColor        = clrWhite;      // Panel / entry text
input string               ObjectPrefix     = "ABR_";        // Object name prefix

//--- Published plan (DRAW_NONE; read from an EA with iCustom).
double BufSignal[];
double BufEntry[];
double BufSL[];
double BufTP1[];
double BufTP2[];
double BufTP3[];
double BufQuality[];

//+------------------------------------------------------------------+
//| Types                                                            |
//+------------------------------------------------------------------+
struct StructEvent
  {
   int      idx;         // bar index the break was confirmed on
   datetime time;
   string   kind;        // "BoS" | "CHoCH"
   int      direction;   // +1 bull / -1 bear
   double   level;       // the swing level that broke
   string   strength;    // "Trend Strong" | "Trend Weak"
  };

struct OrderBlock
  {
   int      idx;
   datetime time;
   bool     bullish;
   double   top;
   double   bottom;
   bool     mitigated;
   int      mitigated_idx;
  };

struct SwingPoint
  {
   int      idx;
   double   price;
  };

//--- Sanitised inputs.
int    g_swingLen = 5, g_obSearch = 15, g_obMaxAge = 300, g_maxOB = 2;
int    g_tlLookback = 10, g_maxBars = 1500, g_atrPeriod = 14;
double g_strongATR = 0.5, g_slBufATR = 0.5;

//--- Engine output.
int          g_trend = 0, g_signal = 0;
double       g_entry = 0.0, g_sl = 0.0, g_tp1 = 0.0, g_tp2 = 0.0, g_tp3 = 0.0;
double       g_lots = 0.0, g_slMoney = 0.0;
int          g_quality = 0;
string       g_grade = "-";
int          g_htfAligned = 0, g_curBias = 0;
bool         g_tlBreak = false;
int          g_htfBias[3];
ENUM_TIMEFRAMES g_htf[3];

StructEvent  g_events[];
OrderBlock   g_obs[];
SwingPoint   g_swingHi[];
SwingPoint   g_swingLo[];

//--- ATR series, copied once per recalculation (never per bar).
double g_atr[];
int    g_atrOffset = 0;

//--- Indicator handles.
int g_atrHandle = INVALID_HANDLE;
int g_maFastHTF[3], g_maSlowHTF[3];
int g_maFastCur = INVALID_HANDLE, g_maSlowCur = INVALID_HANDLE;

datetime g_lastBarTime = 0;

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_swingLen   = MathMax(1,   SwingLength);
   g_obSearch   = MathMax(1,   OBSearchBars);
   g_obMaxAge   = MathMax(1,   OBMaxAgeBars);
   g_maxOB      = MathMax(1,   MaxLiveOBs);
   g_tlLookback = MathMax(1,   TLLookbackBars);
   g_maxBars    = MathMax(0,   MaxBars);
   g_atrPeriod  = MathMax(2,   ATRPeriod);
   g_strongATR  = MathMax(0.0, StrongBreakATR);
   g_slBufATR   = MathMax(0.0, SLBufferATR);

   SetIndexBuffer(0, BufSignal,  INDICATOR_DATA);
   SetIndexBuffer(1, BufEntry,   INDICATOR_DATA);
   SetIndexBuffer(2, BufSL,      INDICATOR_DATA);
   SetIndexBuffer(3, BufTP1,     INDICATOR_DATA);
   SetIndexBuffer(4, BufTP2,     INDICATOR_DATA);
   SetIndexBuffer(5, BufTP3,     INDICATOR_DATA);
   SetIndexBuffer(6, BufQuality, INDICATOR_DATA);
   for(int b = 0; b < 7; b++)
      PlotIndexSetDouble(b, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME, "ABR Price Action Toolkit");

   g_atrHandle = iATR(_Symbol, PERIOD_CURRENT, g_atrPeriod);
   if(g_atrHandle == INVALID_HANDLE)
     {
      Print("ABR: could not create the ATR handle");
      return INIT_FAILED;
     }
   g_maFastCur = iMA(_Symbol, PERIOD_CURRENT, MAFast, 0, MODE_EMA, PRICE_CLOSE);
   g_maSlowCur = iMA(_Symbol, PERIOD_CURRENT, MASlow, 0, MODE_EMA, PRICE_CLOSE);

   g_htf[0] = HTF1;  g_htf[1] = HTF2;  g_htf[2] = HTF3;
   for(int i = 0; i < 3; i++)
     {
      g_htfBias[i]  = 0;
      g_maFastHTF[i] = INVALID_HANDLE;
      g_maSlowHTF[i] = INVALID_HANDLE;
      if(g_htf[i] == PERIOD_CURRENT)          // treated as "disabled"
         continue;
      g_maFastHTF[i] = iMA(_Symbol, g_htf[i], MAFast, 0, MODE_EMA, PRICE_CLOSE);
      g_maSlowHTF[i] = iMA(_Symbol, g_htf[i], MASlow, 0, MODE_EMA, PRICE_CLOSE);
     }
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Deinit — remove only this instance's objects                     |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, ObjectPrefix);
   if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
   if(g_maFastCur != INVALID_HANDLE) IndicatorRelease(g_maFastCur);
   if(g_maSlowCur != INVALID_HANDLE) IndicatorRelease(g_maSlowCur);
   for(int i = 0; i < 3; i++)
     {
      if(g_maFastHTF[i] != INVALID_HANDLE) IndicatorRelease(g_maFastHTF[i]);
      if(g_maSlowHTF[i] != INVALID_HANDLE) IndicatorRelease(g_maSlowHTF[i]);
     }
   ChartRedraw();
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
   if(rates_total < g_swingLen * 2 + 25)
      return rates_total;

   // The structure detector is a path-dependent state machine, so it is
   // recomputed in full rather than patched incrementally. The window is
   // bounded by MaxBars, and it only reruns when a new bar appears.
   if(prev_calculated > 0 && time[rates_total - 1] == g_lastBarTime)
      return rates_total;
   g_lastBarTime = time[rates_total - 1];

   // Oldest-first indexing, exactly like the pandas frame in the Python engine.
   ArraySetAsSeries(time,  false);
   ArraySetAsSeries(open,  false);
   ArraySetAsSeries(high,  false);
   ArraySetAsSeries(low,   false);
   ArraySetAsSeries(close, false);
   ArraySetAsSeries(BufSignal,  false);
   ArraySetAsSeries(BufEntry,   false);
   ArraySetAsSeries(BufSL,      false);
   ArraySetAsSeries(BufTP1,     false);
   ArraySetAsSeries(BufTP2,     false);
   ArraySetAsSeries(BufTP3,     false);
   ArraySetAsSeries(BufQuality, false);

   int n = rates_total;
   int start = (g_maxBars > 0 && n > g_maxBars) ? n - g_maxBars : 0;

   RunEngine(time, open, high, low, close, start, n);
   PublishBuffers(n);
   Redraw(time, close, start, n);
   return rates_total;
  }

//+------------------------------------------------------------------+
//| Expose the current plan on the DRAW_NONE buffers                 |
//+------------------------------------------------------------------+
void PublishBuffers(const int n)
  {
   for(int i = 0; i < n; i++)
     {
      BufSignal[i]  = EMPTY_VALUE;
      BufEntry[i]   = EMPTY_VALUE;
      BufSL[i]      = EMPTY_VALUE;
      BufTP1[i]     = EMPTY_VALUE;
      BufTP2[i]     = EMPTY_VALUE;
      BufTP3[i]     = EMPTY_VALUE;
      BufQuality[i] = EMPTY_VALUE;
     }
   int last = n - 1;
   BufSignal[last]  = (double)g_signal;
   BufQuality[last] = (double)g_quality;
   if(g_signal != 0 && g_entry > 0.0)
     {
      BufEntry[last] = g_entry;
      BufSL[last]    = g_sl;
      if(g_tp1 > 0.0) BufTP1[last] = g_tp1;
      if(g_tp2 > 0.0) BufTP2[last] = g_tp2;
      if(g_tp3 > 0.0) BufTP3[last] = g_tp3;
     }
  }

//+------------------------------------------------------------------+
//| Copy the whole ATR series once — the structure loop reads it      |
//| thousands of times, so a CopyBuffer per bar would crawl.          |
//+------------------------------------------------------------------+
bool LoadAtr(const int rates_total)
  {
   ArraySetAsSeries(g_atr, false);                 // oldest-first, like prices
   int copied = CopyBuffer(g_atrHandle, 0, 0, rates_total, g_atr);
   if(copied <= 0)
     {
      g_atrOffset = 0;
      ArrayResize(g_atr, 0);
      return false;
     }
   // Fewer bars than the chart holds means the MOST RECENT ones came back —
   // record the offset so bar indices still map correctly.
   g_atrOffset = rates_total - copied;
   return true;
  }

//+------------------------------------------------------------------+
//| ATR at an absolute bar index (oldest-first). 0.0 when unusable.  |
//+------------------------------------------------------------------+
double AtrAt(const int abs_idx)
  {
   int i = abs_idx - g_atrOffset;
   if(i < 0 || i >= ArraySize(g_atr)) return 0.0;
   double v = g_atr[i];
   return (v > 0.0 && v == v) ? v : 0.0;           // v==v filters NaN
  }

//+------------------------------------------------------------------+
//| EMA bias: +1 bull / -1 bear / 0 flat.                            |
//| Mirrors _ema_bias(): fast>slow AND close>fast (and the inverse). |
//+------------------------------------------------------------------+
int EmaBias(const int fastHandle, const int slowHandle, const ENUM_TIMEFRAMES tf)
  {
   if(fastHandle == INVALID_HANDLE || slowHandle == INVALID_HANDLE)
      return 0;
   double f[], s[];
   ArraySetAsSeries(f, true);
   ArraySetAsSeries(s, true);
   if(CopyBuffer(fastHandle, 0, 0, 1, f) < 1) return 0;
   if(CopyBuffer(slowHandle, 0, 0, 1, s) < 1) return 0;
   double c = iClose(_Symbol, tf, 0);
   if(c <= 0.0) return 0;
   if(f[0] > s[0] && c > f[0]) return  1;
   if(f[0] < s[0] && c < f[0]) return -1;
   return 0;
  }

//+------------------------------------------------------------------+
//| Engine — the faithful port of run_engine()                       |
//+------------------------------------------------------------------+
void RunEngine(const datetime &time[], const double &open[], const double &high[],
               const double &low[], const double &close[], const int start, const int n)
  {
   ArrayResize(g_events,  0);
   ArrayResize(g_obs,     0);
   ArrayResize(g_swingHi, 0);
   ArrayResize(g_swingLo, 0);
   g_trend = 0; g_signal = 0; g_tlBreak = false;
   g_entry = 0; g_sl = 0; g_tp1 = 0; g_tp2 = 0; g_tp3 = 0;
   g_lots = 0; g_slMoney = 0; g_quality = 0; g_grade = "-";

   int L = g_swingLen;
   LoadAtr(n);                      // one CopyBuffer for the whole recalculation

   //--- 1. pivots: strict unique extreme in the +/-L window ---------------
   bool isHi[], isLo[];
   ArrayResize(isHi, n);
   ArrayResize(isLo, n);
   for(int i = 0; i < n; i++) { isHi[i] = false; isLo[i] = false; }

   for(int i = MathMax(start + L, L); i < n - L; i++)
     {
      double hmax = high[i], lmin = low[i];
      int    hcnt = 0,       lcnt = 0;
      for(int k = i - L; k <= i + L; k++)
        {
         if(high[k] > hmax) hmax = high[k];
         if(low[k]  < lmin) lmin = low[k];
        }
      for(int k = i - L; k <= i + L; k++)
        {
         if(high[k] == high[i]) hcnt++;
         if(low[k]  == low[i])  lcnt++;
        }
      if(high[i] == hmax && hcnt == 1) isHi[i] = true;
      if(low[i]  == lmin && lcnt == 1) isLo[i] = true;
     }

   //--- 2. walk forward: confirm pivots, detect BoS / CHoCH, tag OBs ------
   double lastHi = 0.0, lastLo = 0.0;
   bool   hiAct = false, loAct = false, hasHi = false, hasLo = false;
   int    trend = 0;

   for(int j = start + L; j < n; j++)
     {
      int c = j - L;                       // the pivot becomes visible at bar j
      if(c >= start && c >= 0)
        {
         if(isHi[c])
           {
            lastHi = high[c]; hiAct = true; hasHi = true;
            SwingPoint sp; sp.idx = c; sp.price = high[c];
            int sz = ArraySize(g_swingHi); ArrayResize(g_swingHi, sz + 1); g_swingHi[sz] = sp;
           }
         if(isLo[c])
           {
            lastLo = low[c]; loAct = true; hasLo = true;
            SwingPoint sp; sp.idx = c; sp.price = low[c];
            int sz = ArraySize(g_swingLo); ArrayResize(g_swingLo, sz + 1); g_swingLo[sz] = sp;
           }
        }

      double a = AtrAt(j);

      //--- bullish break -------------------------------------------------
      if(hiAct && hasHi && close[j] > lastHi)
        {
         double pen = (a > 0.0) ? (close[j] - lastHi) / a : 0.0;
         StructEvent e;
         e.idx = j; e.time = time[j];
         e.kind = (trend == -1) ? "CHoCH" : "BoS";
         e.direction = 1; e.level = lastHi;
         e.strength = (pen >= g_strongATR) ? "Trend Strong" : "Trend Weak";
         int es = ArraySize(g_events); ArrayResize(g_events, es + 1); g_events[es] = e;

         // bullish OB = the last DOWN candle before the up impulse
         int stop = MathMax(j - g_obSearch, MathMax(start, 0));
         for(int b = j - 1; b >= stop; b--)
            if(close[b] < open[b])
              {
               OrderBlock ob;
               ob.idx = b; ob.time = time[b]; ob.bullish = true;
               ob.top = high[b]; ob.bottom = low[b];
               ob.mitigated = false; ob.mitigated_idx = -1;
               int os = ArraySize(g_obs); ArrayResize(g_obs, os + 1); g_obs[os] = ob;
               break;
              }
         trend = 1; hiAct = false;
        }

      //--- bearish break -------------------------------------------------
      if(loAct && hasLo && close[j] < lastLo)
        {
         double pen = (a > 0.0) ? (lastLo - close[j]) / a : 0.0;
         StructEvent e;
         e.idx = j; e.time = time[j];
         e.kind = (trend == 1) ? "CHoCH" : "BoS";
         e.direction = -1; e.level = lastLo;
         e.strength = (pen >= g_strongATR) ? "Trend Strong" : "Trend Weak";
         int es = ArraySize(g_events); ArrayResize(g_events, es + 1); g_events[es] = e;

         int stop = MathMax(j - g_obSearch, MathMax(start, 0));
         for(int b = j - 1; b >= stop; b--)
            if(close[b] > open[b])
              {
               OrderBlock ob;
               ob.idx = b; ob.time = time[b]; ob.bullish = false;
               ob.top = high[b]; ob.bottom = low[b];
               ob.mitigated = false; ob.mitigated_idx = -1;
               int os = ArraySize(g_obs); ArrayResize(g_obs, os + 1); g_obs[os] = ob;
               break;
              }
         trend = -1; loAct = false;
        }
     }

   g_trend  = trend;
   g_signal = trend;

   //--- 3. OB mitigation + age -------------------------------------------
   int obCount = ArraySize(g_obs);
   for(int i = 0; i < obCount; i++)
     {
      for(int j = g_obs[i].idx + 1; j < n; j++)
        {
         if(g_obs[i].bullish && close[j] < g_obs[i].bottom)
           { g_obs[i].mitigated = true; g_obs[i].mitigated_idx = j; break; }
         if(!g_obs[i].bullish && close[j] > g_obs[i].top)
           { g_obs[i].mitigated = true; g_obs[i].mitigated_idx = j; break; }
        }
      if(!g_obs[i].mitigated && (n - 1 - g_obs[i].idx) > g_obMaxAge)
        { g_obs[i].mitigated = true; g_obs[i].mitigated_idx = n - 1; }
     }

   //--- cap live blocks per side, keeping the most recent -----------------
   for(int side = 0; side < 2; side++)
     {
      bool wantBull = (side == 0);
      int alive[];
      ArrayResize(alive, 0);
      for(int i = 0; i < obCount; i++)
         if(g_obs[i].bullish == wantBull && !g_obs[i].mitigated)
           { int s = ArraySize(alive); ArrayResize(alive, s + 1); alive[s] = i; }
      int excess = ArraySize(alive) - g_maxOB;
      for(int k = 0; k < excess; k++)          // oldest first
        { g_obs[alive[k]].mitigated = true; g_obs[alive[k]].mitigated_idx = n - 1; }
     }

   //--- 4. MTF bias -------------------------------------------------------
   g_htfAligned = 0;
   for(int i = 0; i < 3; i++)
     {
      g_htfBias[i] = (g_htf[i] == PERIOD_CURRENT)
                     ? 0 : EmaBias(g_maFastHTF[i], g_maSlowHTF[i], g_htf[i]);
      if(g_signal != 0 && g_htfBias[i] == g_signal)
         g_htfAligned++;
     }
   g_curBias = EmaBias(g_maFastCur, g_maSlowCur, PERIOD_CURRENT);

   //--- 5. trendline break ------------------------------------------------
   int nHi = ArraySize(g_swingHi), nLo = ArraySize(g_swingLo);
   if(g_signal > 0 && nHi >= 2)
      g_tlBreak = BrokeRecently(close, g_swingHi[nHi - 2], g_swingHi[nHi - 1], false, start, n);
   else if(g_signal < 0 && nLo >= 2)
      g_tlBreak = BrokeRecently(close, g_swingLo[nLo - 2], g_swingLo[nLo - 1], true, start, n);

   //--- 6. trade plan -----------------------------------------------------
   double buf = AtrAt(n - 1) * g_slBufATR;

   if(g_signal < 0)
     {
      int ob = LatestLiveOB(false);
      double swingHi = (nHi > 0) ? g_swingHi[nHi - 1].price : high[n - 2];
      g_entry = (ob < 0) ? close[n - 1] : g_obs[ob].bottom;
      g_sl    = ((ob < 0) ? swingHi : MathMax(g_obs[ob].top, swingHi)) + buf;
      double r = g_sl - g_entry;
      if(r > 0.0)
        { g_tp1 = g_entry - r * TP1R; g_tp2 = g_entry - r * TP2R; g_tp3 = g_entry - r * TP3R; }
     }
   else if(g_signal > 0)
     {
      int ob = LatestLiveOB(true);
      double swingLo = (nLo > 0) ? g_swingLo[nLo - 1].price : low[n - 2];
      g_entry = (ob < 0) ? close[n - 1] : g_obs[ob].top;
      g_sl    = ((ob < 0) ? swingLo : MathMin(g_obs[ob].bottom, swingLo)) - buf;
      double r = g_entry - g_sl;
      if(r > 0.0)
        { g_tp1 = g_entry + r * TP1R; g_tp2 = g_entry + r * TP2R; g_tp3 = g_entry + r * TP3R; }
     }

   //--- 7. lot size from the BROKER's real contract spec -------------------
   // The Python page uses a static point_value table; reading the live symbol
   // instead means suffixed CFDs (AUDJPYi, XAUUSDm, ...) size correctly with
   // nothing to configure.
   double riskDist = MathAbs(g_sl - g_entry);
   if(riskDist > 0.0 && g_signal != 0)
     {
      double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tickSize > 0.0 && tickValue > 0.0)
        {
         double valuePerPricePerLot = tickValue / tickSize;
         double raw  = RiskPerTrade / (riskDist * valuePerPricePerLot);
         double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
         double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
         if(step <= 0.0) step = 0.01;
         double lots = MathFloor(raw / step) * step;   // round DOWN: never over-risk
         if(lots < vmin)                lots = vmin;
         if(vmax > 0.0 && lots > vmax)  lots = vmax;
         g_lots    = lots;
         g_slMoney = riskDist * valuePerPricePerLot * lots;
        }
     }

   //--- 8. quality + grade (identical formula to the dashboard) -----------
   if(g_signal != 0)
     {
      int q = 25 + g_htfAligned * 15;
      if(LatestLiveOB(g_signal > 0) >= 0) q += 15;
      if(g_tlBreak)                       q += 15;
      g_quality = q;
      g_grade = (q >= 85) ? "A+" : (q >= 70) ? "A" : (q >= 55) ? "B" : (q >= 40) ? "C" : "D";
     }
  }

//+------------------------------------------------------------------+
//| Index of the newest un-mitigated OB on a side, or -1             |
//+------------------------------------------------------------------+
int LatestLiveOB(const bool bullish)
  {
   for(int i = ArraySize(g_obs) - 1; i >= 0; i--)
      if(g_obs[i].bullish == bullish && !g_obs[i].mitigated)
         return i;
   return -1;
  }

//+------------------------------------------------------------------+
//| Value of the line through two swings, projected to bar j         |
//+------------------------------------------------------------------+
double TLValue(const SwingPoint &p_old, const SwingPoint &p_new, const int j)
  {
   if(p_new.idx == p_old.idx) return p_new.price;
   double slope = (p_new.price - p_old.price) / (double)(p_new.idx - p_old.idx);
   return p_new.price + slope * (j - p_new.idx);
  }

//+------------------------------------------------------------------+
//| Did price cross the trendline inside the lookback window?        |
//+------------------------------------------------------------------+
bool BrokeRecently(const double &close[], const SwingPoint &p_old, const SwingPoint &p_new,
                   const bool downward, const int start, const int n)
  {
   int floorIdx = MathMax(n - 1 - g_tlLookback, MathMax(start + 1, 1));
   for(int j = n - 1; j > floorIdx; j--)
     {
      double v     = TLValue(p_old, p_new, j);
      double vPrev = TLValue(p_old, p_new, j - 1);
      if(downward  && close[j] < v && close[j - 1] >= vPrev) return true;
      if(!downward && close[j] > v && close[j - 1] <= vPrev) return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Drawing                                                          |
//+------------------------------------------------------------------+
void Redraw(const datetime &time[], const double &close[], const int start, const int n)
  {
   ObjectsDeleteAll(0, ObjectPrefix);

   if(ShowSwings)
     {
      int nHi = ArraySize(g_swingHi);
      for(int i = MathMax(0, nHi - 12); i < nHi; i++)
         Arrow(ObjectPrefix + "sh" + (string)i, time[g_swingHi[i].idx],
               g_swingHi[i].price, 234, BearColor);        // marker above a swing high
      int nLo = ArraySize(g_swingLo);
      for(int i = MathMax(0, nLo - 12); i < nLo; i++)
         Arrow(ObjectPrefix + "sl" + (string)i, time[g_swingLo[i].idx],
               g_swingLo[i].price, 233, BullColor);        // marker below a swing low
     }

   if(ShowStructure)
     {
      int ne = ArraySize(g_events);
      for(int i = MathMax(0, ne - 8); i < ne; i++)
        {
         color col = (g_events[i].direction > 0) ? BullColor : BearColor;
         string id = ObjectPrefix + "ev" + (string)i;
         int fromIdx = MathMax(start, g_events[i].idx - 20);
         Segment(id, time[fromIdx], g_events[i].level,
                 time[g_events[i].idx], g_events[i].level, col,
                 (g_events[i].kind == "CHoCH") ? STYLE_SOLID : STYLE_DASH, false);
         Label(id + "t", time[g_events[i].idx], g_events[i].level,
               g_events[i].kind + " (" + g_events[i].strength + ")", col);
        }
     }

   if(ShowOrderBlocks)
     {
      int no = ArraySize(g_obs);
      for(int i = 0; i < no; i++)
        {
         if(g_obs[i].mitigated) continue;                  // only live blocks
         datetime t2 = time[n - 1] + (datetime)(PeriodSeconds() * 8);
         Rect(ObjectPrefix + "ob" + (string)i, time[g_obs[i].idx], g_obs[i].top,
              t2, g_obs[i].bottom, g_obs[i].bullish ? OBBullColor : OBBearColor);
         Label(ObjectPrefix + "obt" + (string)i, time[g_obs[i].idx],
               g_obs[i].bullish ? g_obs[i].bottom : g_obs[i].top,
               g_obs[i].bullish ? "Bull OB" : "Bear OB",
               g_obs[i].bullish ? BullColor : BearColor);
        }
     }

   if(ShowTrendlines)
     {
      int nHi = ArraySize(g_swingHi), nLo = ArraySize(g_swingLo);
      if(nHi >= 2)
         Segment(ObjectPrefix + "tlhi",
                 time[g_swingHi[nHi - 2].idx], g_swingHi[nHi - 2].price,
                 time[g_swingHi[nHi - 1].idx], g_swingHi[nHi - 1].price,
                 BearColor, STYLE_DOT, true);
      if(nLo >= 2)
         Segment(ObjectPrefix + "tllo",
                 time[g_swingLo[nLo - 2].idx], g_swingLo[nLo - 2].price,
                 time[g_swingLo[nLo - 1].idx], g_swingLo[nLo - 1].price,
                 BullColor, STYLE_DOT, true);
     }

   if(ShowTradePlan && g_signal != 0 && g_entry > 0.0)
     {
      PriceLine(ObjectPrefix + "entry", g_entry, TextColor, STYLE_SOLID, "ABR Entry");
      PriceLine(ObjectPrefix + "sl",    g_sl,    BearColor, STYLE_DASH,  "ABR SL");
      if(g_tp1 > 0.0) PriceLine(ObjectPrefix + "tp1", g_tp1, BullColor, STYLE_DOT, "ABR TP1");
      if(g_tp2 > 0.0) PriceLine(ObjectPrefix + "tp2", g_tp2, BullColor, STYLE_DOT, "ABR TP2");
      if(g_tp3 > 0.0) PriceLine(ObjectPrefix + "tp3", g_tp3, BullColor, STYLE_DOT, "ABR TP3");
     }

   if(ShowInfoPanel) DrawPanel();
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Info panel                                                       |
//+------------------------------------------------------------------+
void DrawPanel()
  {
   string dir  = (g_signal > 0) ? "BUY" : (g_signal < 0) ? "SELL" : "NONE";
   color  dcol = (g_signal > 0) ? BullColor : (g_signal < 0) ? BearColor : clrGray;

   string htf = "";
   for(int i = 0; i < 3; i++)
     {
      if(g_htf[i] == PERIOD_CURRENT) continue;
      htf += TFName(g_htf[i]) + ":" + BiasGlyph(g_htfBias[i]) + "  ";
     }
   if(htf == "") htf = "(disabled)";

   int liveBull = 0, liveBear = 0;
   for(int i = 0; i < ArraySize(g_obs); i++)
      if(!g_obs[i].mitigated)
        { if(g_obs[i].bullish) liveBull++; else liveBear++; }

   int y = PanelYDistance;
   PanelLine("h",   y, "ABR PRICE ACTION TOOLKIT", TextColor);                 y += 16;
   PanelLine("s",   y, _Symbol + "   " + TFName((ENUM_TIMEFRAMES)Period()), clrSilver); y += 18;
   PanelLine("sig", y, "Signal:  " + dir + "    Grade " + g_grade +
                       "  (" + (string)g_quality + "/100)", dcol);             y += 16;
   PanelLine("tr",  y, "Structure: " + BiasGlyph(g_trend) +
                       "    EMA bias: " + BiasGlyph(g_curBias), clrSilver);    y += 16;
   PanelLine("htf", y, "HTF:  " + htf + "(" + (string)g_htfAligned + " aligned)",
             clrSilver);                                                       y += 16;
   PanelLine("ob",  y, "Live OBs: " + (string)liveBull + " bull / " +
                       (string)liveBear + " bear    TL break: " +
                       (g_tlBreak ? "YES" : "no"), clrSilver);                 y += 18;

   if(g_signal != 0 && g_entry > 0.0)
     {
      PanelLine("e",   y, "Entry: " + Px(g_entry), TextColor);                  y += 16;
      PanelLine("sl2", y, "SL:    " + Px(g_sl) + "   (" +
                          Px(MathAbs(g_entry - g_sl)) + ")", BearColor);        y += 16;
      PanelLine("t1",  y, "TP1:   " + Px(g_tp1), BullColor);                    y += 16;
      PanelLine("t2",  y, "TP2:   " + Px(g_tp2), BullColor);                    y += 16;
      PanelLine("t3",  y, "TP3:   " + Px(g_tp3), BullColor);                    y += 18;
      PanelLine("lot", y, "Lots:  " + DoubleToString(g_lots, 2) + "    risk " +
                          DoubleToString(g_slMoney, 2) + " " +
                          AccountInfoString(ACCOUNT_CURRENCY), clrGold);        y += 16;
     }
   else
     { PanelLine("no", y, "No structural signal yet.", clrGray);                y += 16; }

   y += 4;
   PanelLine("dis", y, "Indicator only - places no orders.", clrDimGray);
  }

//+------------------------------------------------------------------+
//| Object helpers                                                   |
//+------------------------------------------------------------------+
void PanelLine(const string tag, const int y, const string text, const color col)
  {
   string id = ObjectPrefix + "pnl_" + tag;
   if(!ObjectCreate(0, id, OBJ_LABEL, 0, 0, 0)) return;
   ObjectSetInteger(0, id, OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, id, OBJPROP_XDISTANCE, PanelXDistance);
   ObjectSetInteger(0, id, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(0, id, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, id, OBJPROP_HIDDEN, true);
   ObjectSetString(0, id, OBJPROP_FONT, "Consolas");
   ObjectSetString(0, id, OBJPROP_TEXT, text);
  }

void Arrow(const string id, const datetime t, const double p, const int code, const color col)
  {
   if(!ObjectCreate(0, id, OBJ_ARROW, 0, t, p)) return;
   ObjectSetInteger(0, id, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, id, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, id, OBJPROP_HIDDEN, true);
  }

void Segment(const string id, const datetime t1, const double p1,
             const datetime t2, const double p2, const color col,
             const ENUM_LINE_STYLE style, const bool rayRight)
  {
   if(!ObjectCreate(0, id, OBJ_TREND, 0, t1, p1, t2, p2)) return;
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_STYLE, style);
   ObjectSetInteger(0, id, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, id, OBJPROP_RAY_RIGHT, rayRight);
   ObjectSetInteger(0, id, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, id, OBJPROP_HIDDEN, true);
  }

void Rect(const string id, const datetime t1, const double p1,
          const datetime t2, const double p2, const color col)
  {
   if(!ObjectCreate(0, id, OBJ_RECTANGLE, 0, t1, p1, t2, p2)) return;
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_FILL, true);
   ObjectSetInteger(0, id, OBJPROP_BACK, true);
   ObjectSetInteger(0, id, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, id, OBJPROP_HIDDEN, true);
  }

void Label(const string id, const datetime t, const double p,
           const string text, const color col)
  {
   if(!ObjectCreate(0, id, OBJ_TEXT, 0, t, p)) return;
   ObjectSetString(0, id, OBJPROP_TEXT, " " + text);
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_FONTSIZE, 7);
   ObjectSetInteger(0, id, OBJPROP_ANCHOR, ANCHOR_LEFT);
   ObjectSetInteger(0, id, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, id, OBJPROP_HIDDEN, true);
  }

void PriceLine(const string id, const double p, const color col,
               const ENUM_LINE_STYLE style, const string text)
  {
   if(!ObjectCreate(0, id, OBJ_HLINE, 0, 0, p)) return;
   ObjectSetInteger(0, id, OBJPROP_COLOR, col);
   ObjectSetInteger(0, id, OBJPROP_STYLE, style);
   ObjectSetInteger(0, id, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, id, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, id, OBJPROP_HIDDEN, true);
   ObjectSetString(0, id, OBJPROP_TEXT, text);
  }

string Px(const double p) { return DoubleToString(p, _Digits); }

string BiasGlyph(const int b) { return (b > 0) ? "UP" : (b < 0) ? "DOWN" : "flat"; }

string TFName(const ENUM_TIMEFRAMES tf)
  {
   string s = EnumToString(tf);
   StringReplace(s, "PERIOD_", "");
   return s;
  }
//+------------------------------------------------------------------+
