//+------------------------------------------------------------------+
//|                                                  FiboRibbon.mq5  |
//|   The moving-average stack from the desk's own MT5 charts.       |
//+------------------------------------------------------------------+
//  Recreated from the terminal's saved chart profiles, not guessed. The
//  parameters below were read straight out of
//    MQL5\Profiles\Charts\(usd) AUDUSD\chart02.chr
//  and the identical stack appears on (usd) NZDUSD, so it is the trader's
//  standard template rather than a one-off:
//
//    EMA 200  on Close      period=200 method=1(EMA)  apply=1(PRICE_CLOSE)
//    EMA  55  on Close      period=55  method=1       apply=1
//    EMA  21  on Close      period=21  method=1       apply=1
//    EMA   5  on Close      period=5   method=1       apply=1
//    SMA   8  on OPEN       period=8   method=0(SMA)  apply=2(PRICE_OPEN)
//
//  Two other indicators sit on those charts and are deliberately NOT in here:
//    - PivotsNew.ex5   compiled only, no .mq5 source on this machine, so it
//                      cannot be reproduced faithfully. BiasedPivots.mq5 does
//                      ship with source and is a candidate replacement.
//    - Watermark.ex5   cosmetic, nothing to compute.
//
//  Why the 8-SMA is on the OPEN and everything else on the CLOSE: that is what
//  the chart file says. It is not a typo to "fix" — a fast MA of the open
//  against a fast EMA of the close is a standard crossover trigger, and the
//  pair only behaves as intended if the applied prices stay different.
//
//  BUFFER ORDER (stable — signal code reads these by index via iCustom):
//    0 = EMA200   1 = EMA55   2 = EMA21   3 = EMA5   4 = SMA8(Open)
//    5 = bull cross arrow      6 = bear cross arrow
//  New plots are APPENDED, never renumbered: consumers read by index.
//
//  THE CROSS SIGNAL (buffers 5/6)
//  A cross of the trigger pair, gated by the ribbon it sits inside:
//    bull = EMA5 crosses UP through SMA8(open), AND EMA21 > EMA55 > EMA200,
//           AND close > EMA21
//    bear = the mirror image
//  The gate is the point. A 5/8 cross on its own fires constantly in chop; the
//  ribbon ordering is what says the fast cross is going *with* the larger trend
//  rather than against it.
//
//  Signals are evaluated on CLOSED bars only. Testing the forming bar would
//  repaint — an arrow that appears mid-bar and vanishes before the close is
//  worse than no arrow, because it back-tests as a winner and trades as a loss.
//+------------------------------------------------------------------+
#property copyright "Dashboard-Pro"
#property version   "1.10"
#property indicator_chart_window
#property indicator_buffers 7
#property indicator_plots   7

//--- EMA 200
#property indicator_label1  "EMA200"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrCrimson
#property indicator_width1  2
//--- EMA 55
#property indicator_label2  "EMA55"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrOrange
#property indicator_width2  1
//--- EMA 21
#property indicator_label3  "EMA21"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrDodgerBlue
#property indicator_width3  1
//--- EMA 5
#property indicator_label4  "EMA5"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrLime
#property indicator_width4  1
//--- SMA 8 on Open
#property indicator_label5  "SMA8(Open)"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrGold
#property indicator_style5  STYLE_DOT
#property indicator_width5  1
//--- bull cross (gated)
#property indicator_label6  "Bull cross"
#property indicator_type6   DRAW_ARROW
#property indicator_color6  clrLime
#property indicator_width6  2
//--- bear cross (gated)
#property indicator_label7  "Bear cross"
#property indicator_type7   DRAW_ARROW
#property indicator_color7  clrRed
#property indicator_width7  2

//--- inputs (defaults are the values read from the chart profile) -----------
input int                InpSlowPeriod   = 200;            // EMA slow period
input int                InpMidPeriod    = 55;             // EMA mid period
input int                InpFastPeriod   = 21;             // EMA fast period
input int                InpTriggerEma   = 5;              // EMA trigger period
input int                InpTriggerSma   = 8;              // SMA trigger period
input ENUM_APPLIED_PRICE InpEmaPrice     = PRICE_CLOSE;    // applied price, EMAs
input ENUM_APPLIED_PRICE InpSmaPrice     = PRICE_OPEN;     // applied price, SMA8
input int                InpShift        = 0;              // horizontal shift
input bool               InpRequireStack = true;           // gate crosses on ribbon order
input int                InpArrowGapPts  = 40;             // arrow offset from bar, points

//--- buffers ---------------------------------------------------------------
double Ema200Buf[];
double Ema55Buf[];
double Ema21Buf[];
double Ema5Buf[];
double Sma8Buf[];
double BullBuf[];
double BearBuf[];

//--- handles ---------------------------------------------------------------
int hEma200 = INVALID_HANDLE;
int hEma55  = INVALID_HANDLE;
int hEma21  = INVALID_HANDLE;
int hEma5   = INVALID_HANDLE;
int hSma8   = INVALID_HANDLE;

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, Ema200Buf, INDICATOR_DATA);
   SetIndexBuffer(1, Ema55Buf,  INDICATOR_DATA);
   SetIndexBuffer(2, Ema21Buf,  INDICATOR_DATA);
   SetIndexBuffer(3, Ema5Buf,   INDICATOR_DATA);
   SetIndexBuffer(4, Sma8Buf,   INDICATOR_DATA);
   SetIndexBuffer(5, BullBuf,   INDICATOR_DATA);
   SetIndexBuffer(6, BearBuf,   INDICATOR_DATA);

   for(int i = 0; i < 5; i++)
      PlotIndexSetInteger(i, PLOT_SHIFT, InpShift);

   //--- arrows: 233 up / 234 down, empty where there is no signal
   PlotIndexSetInteger(5, PLOT_ARROW, 233);
   PlotIndexSetInteger(6, PLOT_ARROW, 234);
   PlotIndexSetDouble(5, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(6, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   //--- the gate needs the slow EMA, so nothing can fire before it exists
   PlotIndexSetInteger(5, PLOT_DRAW_BEGIN, InpSlowPeriod + 1);
   PlotIndexSetInteger(6, PLOT_DRAW_BEGIN, InpSlowPeriod + 1);

   //--- don't draw the leading bars where the average isn't defined yet
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpSlowPeriod);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, InpMidPeriod);
   PlotIndexSetInteger(2, PLOT_DRAW_BEGIN, InpFastPeriod);
   PlotIndexSetInteger(3, PLOT_DRAW_BEGIN, InpTriggerEma);
   PlotIndexSetInteger(4, PLOT_DRAW_BEGIN, InpTriggerSma);

   hEma200 = iMA(_Symbol, _Period, InpSlowPeriod,  0, MODE_EMA, InpEmaPrice);
   hEma55  = iMA(_Symbol, _Period, InpMidPeriod,   0, MODE_EMA, InpEmaPrice);
   hEma21  = iMA(_Symbol, _Period, InpFastPeriod,  0, MODE_EMA, InpEmaPrice);
   hEma5   = iMA(_Symbol, _Period, InpTriggerEma,  0, MODE_EMA, InpEmaPrice);
   hSma8   = iMA(_Symbol, _Period, InpTriggerSma,  0, MODE_SMA, InpSmaPrice);

   if(hEma200 == INVALID_HANDLE || hEma55 == INVALID_HANDLE ||
      hEma21  == INVALID_HANDLE || hEma5  == INVALID_HANDLE ||
      hSma8   == INVALID_HANDLE)
     {
      Print("FiboRibbon: could not create an iMA handle, error ", GetLastError());
      return(INIT_FAILED);
     }

   IndicatorSetString(INDICATOR_SHORTNAME,
                      StringFormat("FiboRibbon (%d/%d/%d/%d EMA + %d SMA open)",
                                   InpTriggerEma, InpFastPeriod, InpMidPeriod,
                                   InpSlowPeriod, InpTriggerSma));
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits + 1);
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(hEma200 != INVALID_HANDLE) IndicatorRelease(hEma200);
   if(hEma55  != INVALID_HANDLE) IndicatorRelease(hEma55);
   if(hEma21  != INVALID_HANDLE) IndicatorRelease(hEma21);
   if(hEma5   != INVALID_HANDLE) IndicatorRelease(hEma5);
   if(hSma8   != INVALID_HANDLE) IndicatorRelease(hSma8);
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
   //--- the slowest average decides when there is anything to draw at all
   if(rates_total < InpSlowPeriod)
      return(0);

   //--- recopy only what changed; on the first pass copy everything
   int to_copy;
   if(prev_calculated > rates_total || prev_calculated <= 0)
      to_copy = rates_total;
   else
     {
      to_copy = rates_total - prev_calculated;
      to_copy++;                       // the last bar is still forming
     }

   //--- a failed copy means the source indicator hasn't finished calculating;
   //--- return 0 so MT5 asks again on the next tick rather than leaving a gap
   if(CopyBuffer(hEma200, 0, 0, to_copy, Ema200Buf) <= 0) return(0);
   if(CopyBuffer(hEma55,  0, 0, to_copy, Ema55Buf)  <= 0) return(0);
   if(CopyBuffer(hEma21,  0, 0, to_copy, Ema21Buf)  <= 0) return(0);
   if(CopyBuffer(hEma5,   0, 0, to_copy, Ema5Buf)   <= 0) return(0);
   if(CopyBuffer(hSma8,   0, 0, to_copy, Sma8Buf)   <= 0) return(0);

   //--- cross detection ----------------------------------------------------
   //  Buffers here are plain arrays (index 0 = oldest), so bar i-1 is the bar
   //  before i. The gate needs EMA200, which only exists from InpSlowPeriod on.
   int start = (prev_calculated <= 0) ? InpSlowPeriod + 1 : prev_calculated - 1;
   if(start < InpSlowPeriod + 1)
      start = InpSlowPeriod + 1;
   if(start < 1)
      start = 1;

   double gap = InpArrowGapPts * _Point;

   for(int i = start; i < rates_total; i++)
     {
      BullBuf[i] = EMPTY_VALUE;
      BearBuf[i] = EMPTY_VALUE;

      //--- the newest bar is still forming; judging it would repaint
      if(i == rates_total - 1)
         continue;

      //--- the trigger pair crossing between the two closed bars
      bool crossUp   = (Ema5Buf[i]   >  Sma8Buf[i]) &&
                       (Ema5Buf[i-1] <= Sma8Buf[i-1]);
      bool crossDown = (Ema5Buf[i]   <  Sma8Buf[i]) &&
                       (Ema5Buf[i-1] >= Sma8Buf[i-1]);

      //--- the ribbon it sits inside: fast over mid over slow, price with it
      bool stackUp   = (Ema21Buf[i] > Ema55Buf[i]) &&
                       (Ema55Buf[i] > Ema200Buf[i]) &&
                       (close[i]    > Ema21Buf[i]);
      bool stackDown = (Ema21Buf[i] < Ema55Buf[i]) &&
                       (Ema55Buf[i] < Ema200Buf[i]) &&
                       (close[i]    < Ema21Buf[i]);

      if(!InpRequireStack)
        {
         stackUp   = true;
         stackDown = true;
        }

      if(crossUp && stackUp)
         BullBuf[i] = low[i] - gap;
      else if(crossDown && stackDown)
         BearBuf[i] = high[i] + gap;
     }

   return(rates_total);
  }
//+------------------------------------------------------------------+
