//+------------------------------------------------------------------+
//|                                                  SessionsAMD.mq5 |
//|  Shades the Asia / London / New York trading sessions on the    |
//|  chart in light colors, and flags a simplified Accumulation -   |
//|  Manipulation - Distribution (AMD) zone based on the Asia range.|
//|                                                                  |
//|  INSTALL:                                                        |
//|  1. Open MetaEditor (Tools > MetaQuotes Language Editor, or F4   |
//|     from MT5).                                                   |
//|  2. File > Open Data Folder > MQL5 > Indicators, drop this file  |
//|     there (or File > New > Indicator, paste this code in).       |
//|  3. Compile (F7).                                                |
//|  4. In MT5, drag "SessionsAMD" from Navigator > Indicators onto  |
//|     any chart.                                                   |
//|                                                                  |
//|  All session times are in the input parameters and are           |
//|  interpreted as BROKER/SERVER time. Adjust the *_Start / *_End   |
//|  inputs (and GMTOffsetHours if you just want it for labeling)    |
//|  to match your broker's server-time offset from GMT.             |
//+------------------------------------------------------------------+
#property copyright "Dashboard Pro"
#property link      ""
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0

//--- Session toggles
input bool   Show_Asia            = true;
input bool   Show_London          = true;
input bool   Show_NewYork         = true;
input bool   Show_AMD             = true;     // Accumulation/Manipulation/Distribution overlay

//--- Session hours (BROKER/SERVER TIME, 24h). Defaults are a common
//    approximation - adjust to match your broker's GMT offset.
input int    Asia_StartHour       = 0;
input int    Asia_EndHour         = 8;

input int    London_StartHour     = 8;
input int    London_EndHour       = 13;

input int    NewYork_StartHour    = 13;
input int    NewYork_EndHour      = 21;

//--- Light colors for session shading (kept soft/pastel on purpose)
input color  Asia_Color           = C'255,235,240';  // light pink
input color  London_Color         = C'225,235,255';  // light blue
input color  NewYork_Color        = C'225,250,230';  // light green

//--- AMD overlay colors
input color  Accumulation_Color   = C'255,250,205';  // light yellow (Asia range box)
input color  Manipulation_Color   = C'255,220,220';  // light red (liquidity-sweep box)
input color  Distribution_Color   = C'210,245,215';  // light green (breakout/expansion box)

input int    Lookback_Days        = 5;   // how many past days to draw sessions/AMD for

//--- how far manipulation must poke beyond the Asia range before it counts
input double Manipulation_MinPips = 2.0;

#define OBJ_PREFIX "SessAMD_"

//+------------------------------------------------------------------+
int OnInit()
  {
   IndicatorSetString(INDICATOR_SHORTNAME,"SessionsAMD");
   EventSetTimer(30); // periodic redraw so today's boxes keep extending
   DrawAll();
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0,OBJ_PREFIX);
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   DrawAll();
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
   // Redraw whenever a new bar forms (cheap enough at H1/M15/M5 timeframes)
   static datetime lastBarTime = 0;
   if(rates_total>0 && time[rates_total-1]!=lastBarTime)
     {
      lastBarTime = time[rates_total-1];
      DrawAll();
     }
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Master redraw                                                    |
//+------------------------------------------------------------------+
void DrawAll()
  {
   ObjectsDeleteAll(0,OBJ_PREFIX);

   datetime now = TimeCurrent();
   MqlDateTime dtNow;
   TimeToStruct(now,dtNow);

   for(int d=0; d<=Lookback_Days; d++)
     {
      datetime dayStart = DayStart(now - d*86400);

      if(Show_Asia)
         DrawSessionBox(dayStart, Asia_StartHour,   Asia_EndHour,   Asia_Color,   "Asia_"   + IntegerToString(d));
      if(Show_London)
         DrawSessionBox(dayStart, London_StartHour, London_EndHour, London_Color, "London_" + IntegerToString(d));
      if(Show_NewYork)
         DrawSessionBox(dayStart, NewYork_StartHour,NewYork_EndHour,NewYork_Color,"NY_"     + IntegerToString(d));

      if(Show_AMD)
         DrawAMD(dayStart, d);
     }
  }

//+------------------------------------------------------------------+
datetime DayStart(datetime t)
  {
   MqlDateTime s;
   TimeToStruct(t,s);
   s.hour=0; s.min=0; s.sec=0;
   return(StructToTime(s));
  }

//+------------------------------------------------------------------+
//| Draw one light-colored session background rectangle              |
//+------------------------------------------------------------------+
void DrawSessionBox(datetime dayStart,int startHour,int endHour,color clr,string tag)
  {
   datetime t1 = dayStart + startHour*3600;
   datetime t2 = dayStart + endHour*3600;
   if(t2<=t1) t2 += 86400; // session crosses midnight

   double hi,lo;
   if(!RangeHighLow(t1,t2,hi,lo))
     {
      // no bars yet in this window (e.g. still forming) - use visible chart range as fallback
      hi = ChartGetDouble(0,CHART_PRICE_MAX,0);
      lo = ChartGetDouble(0,CHART_PRICE_MIN,0);
      if(hi<=lo) return;
     }
   else
     {
      double pad = (hi-lo)*0.15;
      hi += pad; lo -= pad;
     }

   string name = OBJ_PREFIX + tag;
   ObjectCreate(0,name,OBJ_RECTANGLE,0,t1,hi,t2,lo);
   ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(0,name,OBJPROP_FILL,true);
   ObjectSetInteger(0,name,OBJPROP_BACK,true);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
   ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
   ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_SOLID);
  }

//+------------------------------------------------------------------+
//| Get high/low of bars between t1 and t2 on the current timeframe   |
//+------------------------------------------------------------------+
bool RangeHighLow(datetime t1,datetime t2,double &hi,double &lo)
  {
   int i1 = iBarShift(_Symbol,_Period,t1,true);
   int i2 = iBarShift(_Symbol,_Period,t2,true);
   if(i1<0 || i2<0) return(false);
   int from = MathMin(i1,i2);
   int to   = MathMax(i1,i2);

   hi = -DBL_MAX; lo = DBL_MAX;
   bool found=false;
   for(int i=to;i<=from;i++)
     {
      double h = iHigh(_Symbol,_Period,i);
      double l = iLow(_Symbol,_Period,i);
      if(h<=0 || l<=0) continue;
      if(h>hi) hi=h;
      if(l<lo) lo=l;
      found=true;
     }
   return(found);
  }

//+------------------------------------------------------------------+
//| Simplified ICT-style Accumulation / Manipulation / Distribution   |
//|                                                                    |
//| Accumulation = Asia session range (the "resting liquidity" box)   |
//| Manipulation = first wick during/after London open that pokes     |
//|                 through the Asia high or low by more than         |
//|                 Manipulation_MinPips, then price closes back      |
//|                 inside the Asia range.                            |
//| Distribution = the directional expansion away from that sweep,    |
//|                 drawn from the sweep bar through to the current   |
//|                 last bar of the day (extends live as bars form).  |
//+------------------------------------------------------------------+
void DrawAMD(datetime dayStart,int dayIndex)
  {
   datetime asiaStart = dayStart + Asia_StartHour*3600;
   datetime asiaEnd   = dayStart + Asia_EndHour*3600;
   if(asiaEnd<=asiaStart) asiaEnd += 86400;

   double asiaHi,asiaLo;
   if(!RangeHighLow(asiaStart,asiaEnd,asiaHi,asiaLo))
      return; // no Asia data yet for this day

   // 1) Accumulation box (Asia range itself, light yellow)
   string accName = OBJ_PREFIX + "Acc_" + IntegerToString(dayIndex);
   ObjectCreate(0,accName,OBJ_RECTANGLE,0,asiaStart,asiaHi,asiaEnd,asiaLo);
   ObjectSetInteger(0,accName,OBJPROP_COLOR,Accumulation_Color);
   ObjectSetInteger(0,accName,OBJPROP_FILL,true);
   ObjectSetInteger(0,accName,OBJPROP_BACK,true);
   ObjectSetInteger(0,accName,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,accName,OBJPROP_HIDDEN,true);

   string accLbl = OBJ_PREFIX + "AccLbl_" + IntegerToString(dayIndex);
   ObjectCreate(0,accLbl,OBJ_TEXT,0,asiaStart,asiaHi);
   ObjectSetString(0,accLbl,OBJPROP_TEXT,"Accumulation");
   ObjectSetInteger(0,accLbl,OBJPROP_COLOR,clrDarkGoldenrod);
   ObjectSetInteger(0,accLbl,OBJPROP_FONTSIZE,7);
   ObjectSetInteger(0,accLbl,OBJPROP_HIDDEN,true);

   // 2) Look for a manipulation sweep after Asia closes (through end of NY)
   double pip = _Point * (( _Digits==3 || _Digits==5) ? 10 : 1);
   double minPoke = Manipulation_MinPips*pip;

   datetime scanStart = asiaEnd;
   datetime scanEnd   = dayStart + NewYork_EndHour*3600;
   if(scanEnd<=scanStart) scanEnd += 86400;

   int iFrom = iBarShift(_Symbol,_Period,scanEnd,true);
   int iTo   = iBarShift(_Symbol,_Period,scanStart,true);
   if(iFrom<0 || iTo<0) return;
   if(iFrom>iTo){int tmp=iFrom; iFrom=iTo; iTo=tmp;}

   int sweepBar = -1; bool sweepUp=false; // sweepUp = swept the HIGH (bearish manip), false = swept the LOW (bullish manip)
   for(int i=iTo;i>=iFrom;i--) // walk forward in time (older index -> newer)
     {
      double h=iHigh(_Symbol,_Period,i);
      double l=iLow(_Symbol,_Period,i);
      double c=iClose(_Symbol,_Period,i);
      if(h<=0||l<=0) continue;

      if(h > asiaHi + minPoke && c < asiaHi)
        { sweepBar=i; sweepUp=true; break; }
      if(l < asiaLo - minPoke && c > asiaLo)
        { sweepBar=i; sweepUp=false; break; }
     }

   if(sweepBar<0) return; // no manipulation detected yet for this day

   datetime sweepTime = iTime(_Symbol,_Period,sweepBar);
   double sweepHi = sweepUp ? iHigh(_Symbol,_Period,sweepBar) : asiaHi;
   double sweepLo = sweepUp ? asiaLo : iLow(_Symbol,_Period,sweepBar);

   string manName = OBJ_PREFIX + "Man_" + IntegerToString(dayIndex);
   ObjectCreate(0,manName,OBJ_RECTANGLE,0,sweepTime,sweepHi,sweepTime+ (int)PeriodSeconds()*2,sweepLo);
   ObjectSetInteger(0,manName,OBJPROP_COLOR,Manipulation_Color);
   ObjectSetInteger(0,manName,OBJPROP_FILL,true);
   ObjectSetInteger(0,manName,OBJPROP_BACK,true);
   ObjectSetInteger(0,manName,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,manName,OBJPROP_HIDDEN,true);

   string manLbl = OBJ_PREFIX + "ManLbl_" + IntegerToString(dayIndex);
   ObjectCreate(0,manLbl,OBJ_TEXT,0,sweepTime,sweepUp?sweepHi:sweepLo);
   ObjectSetString(0,manLbl,OBJPROP_TEXT,"Manipulation");
   ObjectSetInteger(0,manLbl,OBJPROP_COLOR,clrCrimson);
   ObjectSetInteger(0,manLbl,OBJPROP_FONTSIZE,7);
   ObjectSetInteger(0,manLbl,OBJPROP_HIDDEN,true);

   // 3) Distribution = expansion from the sweep bar to the most recent bar of the scan window
   int iLast = iBarShift(_Symbol,_Period,MathMin(scanEnd,TimeCurrent()),true);
   if(iLast<0 || iLast>=sweepBar) return;

   double distHi=-DBL_MAX, distLo=DBL_MAX;
   for(int i=sweepBar;i>=iLast;i--)
     {
      double h=iHigh(_Symbol,_Period,i);
      double l=iLow(_Symbol,_Period,i);
      if(h<=0||l<=0) continue;
      if(h>distHi) distHi=h;
      if(l<distLo) distLo=l;
     }
   if(distHi<=distLo) return;

   datetime distStart = sweepTime;
   datetime distEnd   = iTime(_Symbol,_Period,iLast) + (int)PeriodSeconds();

   string distName = OBJ_PREFIX + "Dist_" + IntegerToString(dayIndex);
   ObjectCreate(0,distName,OBJ_RECTANGLE,0,distStart,distHi,distEnd,distLo);
   ObjectSetInteger(0,distName,OBJPROP_COLOR,Distribution_Color);
   ObjectSetInteger(0,distName,OBJPROP_FILL,true);
   ObjectSetInteger(0,distName,OBJPROP_BACK,true);
   ObjectSetInteger(0,distName,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,distName,OBJPROP_HIDDEN,true);

   string distLbl = OBJ_PREFIX + "DistLbl_" + IntegerToString(dayIndex);
   ObjectCreate(0,distLbl,OBJ_TEXT,0,distEnd,sweepUp?distLo:distHi);
   ObjectSetString(0,distLbl,OBJPROP_TEXT,"Distribution");
   ObjectSetInteger(0,distLbl,OBJPROP_COLOR,clrSeaGreen);
   ObjectSetInteger(0,distLbl,OBJPROP_FONTSIZE,7);
   ObjectSetInteger(0,distLbl,OBJPROP_HIDDEN,true);
  }
//+------------------------------------------------------------------+
