//+------------------------------------------------------------------+
//|                                        OscillatorConsensus.mq5   |
//|   Four oscillators, one overbought/oversold answer.              |
//+------------------------------------------------------------------+
//  A single oscillator whipsaws. Four agreeing is worth a look. This plots
//  RSI, Stochastic %K, Williams %R and CCI on one 0-100 scale, plus a
//  histogram counting how many are simultaneously beyond their band.
//
//  NORMALISATION (so four different scales can share one sub-window)
//    RSI          0..100 already.
//    Stochastic   0..100 already.
//    Williams %R  natively -100..0  -> plotted as W%R + 100.
//    CCI          unbounded         -> plotted as 50 + CCI/4, clipped to 0..100,
//                                      so the classic +/-100 lands on 75/25.
//  The count is computed from the RAW values, never the normalised ones:
//  clipping CCI would otherwise silently cap an extreme reading, and a CCI of
//  400 would count the same as one of 101.
//
//  BANDS -- deliberately the classic mean-reversion levels, NOT the 40/60 in
//  the Python app's config.py. That 40/60 is a trend-PULLBACK band (buy the dip
//  to 40 in an uptrend); this indicator answers "is this stretched", which is a
//  mean-reversion question. Measured 2026-08-15: EUR/USD at RSI 61.3 reads
//  "overbought" on 60 and "mid-range" on 70. Both are defensible; they are
//  answers to different questions. All eight bands are inputs -- change them
//  here, not in the app.
//
//  BUFFER ORDER (stable — signal code reads these by index via iCustom):
//    0 = RSI                     1 = Stoch %K
//    2 = Williams %R + 100       3 = CCI normalised
//    4 = histogram baseline (constant 50)
//    5 = histogram top (50 + count * 12.5)
//    6 = CONSENSUS COUNT, -4..+4  <-- this is the one to read from iCustom
//  New plots are APPENDED, never renumbered: consumers read by index.
//
//  WHY THE HISTOGRAM IS DRAWN FROM TWO BUFFERS
//  The count is -4..+4 and the pane is 0..100. Plotted raw with DRAW_HISTOGRAM
//  it would be a 4%-tall smudge against the zero line, and DRAW_HISTOGRAM always
//  draws *from zero*, so it cannot simply be shifted up. DRAW_HISTOGRAM2 draws
//  between two buffers instead, so buffer 4 pins a baseline at the pane's
//  midpoint and buffer 5 carries the scaled value: -4..+4 maps to 0..100 and the
//  bars grow up or down from 50 the way a signed count should. Buffer 6 keeps
//  the honest integer for machines, so nothing downstream has to undo the
//  scaling.
//
//  READING IT: the histogram is the indicator. Three or four bars' worth above
//  the midline means three or four oscillators agree the market is stretched up;
//  the same below, stretched down. One or two is noise.
//
//  THE COLOURED BANDS
//    GREEN line (30) = oversold, the buy side.
//    RED   line (70) = overbought, the sell side.
//    Grey dotted (50) = the histogram's zero. A reference, not a decision.
//  A line being touched is a CONDITION, not a trigger: markets stay oversold
//  for weeks in a downtrend, which is exactly why the histogram exists. Green
//  with three or four bars below the midline is the read worth acting on;
//  green on its own is just a low number.
//
//  NO ALERTS BY DESIGN. The desk's alerting lives in alert_service.py; a second
//  channel firing from the terminal would double every notification.
//+------------------------------------------------------------------+
#property copyright "Dashboard-Pro"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 7
#property indicator_plots   5

#property indicator_minimum 0
#property indicator_maximum 100

//--- plot 0: RSI
#property indicator_label1  "RSI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  1

//--- plot 1: Stochastic %K
#property indicator_label2  "Stoch %K"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrOrange
#property indicator_width2  1

//--- plot 2: Williams %R (shifted)
#property indicator_label3  "Williams %R"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrMediumPurple
#property indicator_width3  1

//--- plot 3: CCI (normalised)
#property indicator_label4  "CCI"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrGold
#property indicator_width4  1

//--- plot 4: the consensus histogram (drawn between buffers 4 and 5)
#property indicator_label5  "Consensus"
#property indicator_type5   DRAW_HISTOGRAM2
#property indicator_color5  clrLime
#property indicator_width5  3

input int    RSIPeriod       = 14;      // RSI period
input int    StochKPeriod    = 14;      // Stochastic %K period
input int    StochDPeriod    = 3;       // Stochastic %D period
input int    StochSlowing    = 3;       // Stochastic slowing
input int    WPRPeriod       = 14;      // Williams %R period
input int    CCIPeriod       = 20;      // CCI period

input double RSIOverbought   = 70.0;    // RSI overbought (classic; app uses 60)
input double RSIOversold     = 30.0;    // RSI oversold   (classic; app uses 40)
input double StochOverbought = 80.0;    // Stochastic overbought
input double StochOversold   = 20.0;    // Stochastic oversold
input double WPROverbought   = -20.0;   // Williams %R overbought (raw -100..0)
input double WPROversold     = -80.0;   // Williams %R oversold   (raw -100..0)
input double CCIOverbought   = 100.0;   // CCI overbought
input double CCIOversold     = -100.0;  // CCI oversold

//--- the pane midpoint the signed histogram grows from
#define HIST_BASE   50.0
//--- 4 oscillators * 12.5 = 50, so a full +/-4 fills the pane exactly
#define HIST_SCALE  12.5

double BufRSI[], BufStoch[], BufWPR[], BufCCI[];
double BufHistBase[], BufHistTop[], BufCount[];

int hRSI, hStoch, hWPR, hCCI;

int OnInit()
  {
   SetIndexBuffer(0, BufRSI,      INDICATOR_DATA);
   SetIndexBuffer(1, BufStoch,    INDICATOR_DATA);
   SetIndexBuffer(2, BufWPR,      INDICATOR_DATA);
   SetIndexBuffer(3, BufCCI,      INDICATOR_DATA);
   SetIndexBuffer(4, BufHistBase, INDICATOR_DATA);
   SetIndexBuffer(5, BufHistTop,  INDICATOR_DATA);
   //--- the honest integer, for iCustom rather than for the eye
   SetIndexBuffer(6, BufCount,    INDICATOR_CALCULATIONS);

   hRSI   = iRSI(NULL, 0, RSIPeriod, PRICE_CLOSE);
   hStoch = iStochastic(NULL, 0, StochKPeriod, StochDPeriod, StochSlowing,
                        MODE_SMA, STO_LOWHIGH);
   hWPR   = iWPR(NULL, 0, WPRPeriod);
   hCCI   = iCCI(NULL, 0, CCIPeriod, PRICE_TYPICAL);

   if(hRSI == INVALID_HANDLE || hStoch == INVALID_HANDLE ||
      hWPR == INVALID_HANDLE || hCCI == INVALID_HANDLE)
     {
      Print("OscillatorConsensus: could not create an indicator handle");
      return(INIT_FAILED);
     }

   IndicatorSetString(INDICATOR_SHORTNAME, "OscillatorConsensus");
   IndicatorSetInteger(INDICATOR_DIGITS, 1);

   //--- THE BANDS ARE THE POINT, SO THEY ARE COLOURED, NOT HAIRLINES.
   //  Green at the oversold band, red at the overbought one, both solid and
   //  2px so they read at a glance from across the desk. The midline stays a
   //  dim dotted grey: it is the histogram's zero, a reference rather than a
   //  level anyone acts on, and drawing it as loudly as the other two would
   //  make the pane read as three equal decisions instead of one.
   IndicatorSetInteger(INDICATOR_LEVELS, 3);

   //--- level 0: oversold -> the BUY side
   IndicatorSetDouble (INDICATOR_LEVELVALUE, 0, RSIOversold);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 0, clrLime);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 0, STYLE_SOLID);
   IndicatorSetInteger(INDICATOR_LEVELWIDTH, 0, 2);
   IndicatorSetString (INDICATOR_LEVELTEXT,  0, "OVERSOLD / buy zone");

   //--- level 1: the histogram's zero, deliberately quiet
   IndicatorSetDouble (INDICATOR_LEVELVALUE, 1, HIST_BASE);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 1, clrDimGray);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 1, STYLE_DOT);
   IndicatorSetInteger(INDICATOR_LEVELWIDTH, 1, 1);
   IndicatorSetString (INDICATOR_LEVELTEXT,  1, "");

   //--- level 2: overbought -> the SELL side
   IndicatorSetDouble (INDICATOR_LEVELVALUE, 2, RSIOverbought);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 2, clrRed);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 2, STYLE_SOLID);
   IndicatorSetInteger(INDICATOR_LEVELWIDTH, 2, 2);
   IndicatorSetString (INDICATOR_LEVELTEXT,  2, "OVERBOUGHT / sell zone");

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| CCI is unbounded; squeeze it onto the shared 0-100 axis so the    |
//| classic +/-100 lands on 75/25. Only the PLOT is clipped -- the    |
//| count reads the raw value.                                       |
//+------------------------------------------------------------------+
double NormaliseCCI(const double raw)
  {
   double v = 50.0 + raw / 4.0;
   if(v > 100.0) v = 100.0;
   if(v <   0.0) v =   0.0;
   return(v);
  }

//+------------------------------------------------------------------+
//| Signed count of oscillators beyond their band, from RAW values.   |
//+------------------------------------------------------------------+
int Consensus(const double rsi, const double stoch,
              const double wpr, const double cci)
  {
   int score = 0;
   if(rsi   > RSIOverbought)   score++;
   if(stoch > StochOverbought) score++;
   if(wpr   > WPROverbought)   score++;
   if(cci   > CCIOverbought)   score++;

   if(rsi   < RSIOversold)     score--;
   if(stoch < StochOversold)   score--;
   if(wpr   < WPROversold)     score--;
   if(cci   < CCIOversold)     score--;
   return(score);
  }

int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
  {
   int need = MathMax(MathMax(RSIPeriod, StochKPeriod),
                      MathMax(WPRPeriod, CCIPeriod)) + StochSlowing + 1;
   if(rates_total < need)
      return(0);

   double rsi[], stoch[], wpr[], cci[];
   if(CopyBuffer(hRSI,   0, 0, rates_total, rsi)   <= 0) return(0);
   if(CopyBuffer(hStoch, 0, 0, rates_total, stoch) <= 0) return(0);
   if(CopyBuffer(hWPR,   0, 0, rates_total, wpr)   <= 0) return(0);
   if(CopyBuffer(hCCI,   0, 0, rates_total, cci)   <= 0) return(0);

   //--- recompute the last known bar too: it was still forming last tick
   int start = (prev_calculated > 1) ? prev_calculated - 1 : 0;

   for(int i = start; i < rates_total; i++)
     {
      BufRSI[i]   = rsi[i];
      BufStoch[i] = stoch[i];
      BufWPR[i]   = wpr[i] + 100.0;          // -100..0 -> 0..100
      BufCCI[i]   = NormaliseCCI(cci[i]);

      int score = Consensus(rsi[i], stoch[i], wpr[i], cci[i]);

      BufCount[i]    = (double)score;
      BufHistBase[i] = HIST_BASE;
      BufHistTop[i]  = HIST_BASE + score * HIST_SCALE;
     }
   return(rates_total);
  }
//+------------------------------------------------------------------+
