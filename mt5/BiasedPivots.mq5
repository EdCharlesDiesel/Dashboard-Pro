//+------------------------------------------------------------------+
//|                                                  WyattsPivots.mq4|
//+------------------------------------------------------------------+

#property copyright "Copyright, FX BOOTCAMP TRAINING LLC"
#property indicator_chart_window
#property indicator_buffers   13


#property indicator_color1 clrNONE
#property indicator_color2 clrNONE
#property indicator_color3 clrNONE
#property indicator_color4 clrNONE
#property indicator_color5 clrNONE
#property indicator_color6 clrNONE
#property indicator_color7 clrNONE
#property indicator_color8 clrNONE
#property indicator_color9 clrNONE
#property indicator_color10 clrNONE
#property indicator_color11 clrNONE
#property indicator_color12 clrNONE
#property indicator_color13 clrNONE


enum indicatorMode
  {
   bullishMode       = 0, // Bullish Mode
   bearishMode       = 1, // Bearish Mode
  };



enum timeFrames
{
Daily       = 1,
Weekly      = 2,
Monthly     = 3,
};



input indicatorMode  modeInd = bullishMode; // Indicator Mode

input int              CountPeriods=20;
input timeFrames       TimePeriod=Daily;
input bool             PlotPivots=true;
input bool             PlotFuturePivots=true;
input bool             PlotPivotLabels=true;
input bool             PlotPivotPrices=true;
input ENUM_LINE_STYLE  StylePivots=STYLE_SOLID;
input int              WidthPivots=2;
input color            ColorRes=clrGray;
input color            ColorPP=clrGray;
input color            ColorSup=clrGray;
input bool             PlotMidpoints=true;
input ENUM_LINE_STYLE  StyleMidpoints=STYLE_DASH;
input int              WidthMidpoints=1;

input bool             PlotZones=true;
input color            ColorBuyZone=clrDarkSeaGreen;
input color            ColorSellZone=clrMistyRose;



input color            ColorM35=clrGray;
input color            ColorM02=clrGray;
input int              TaillePolice=7;
string   period;

datetime timestart,
         timeend;

double   openValue,
         closeValue,
         highValue,
         lowValue;

double   PP,         // Pivot Levels
         R1,
         R2,
         R3,
         S1,
         S2,
         S3,
         M0,
         M1,
         M2,
         M3,
         M4,
         M5,
         f214,       // Fibot Levels
         f236,
         f382,
         f50,
         f618,
         f764,
         f786,
         rangeopen1, // OHLC Levels
         rangeopen2,
         rangeclose1,
         rangeclose2;

int      shift;

string NotesFont                                  = "Cambria";
int NotesLocation_x                            = 10;
int NotesLocation_y                            = 10;

color  NotesColor                                 = Black;


double  buffer_PP[];
double  buffer_R1[];
double  buffer_R2[];
double  buffer_R3[];
double  buffer_S1[];
double  buffer_S2[];
double  buffer_S3[];
double  buffer_M0[];
double  buffer_M1[];
double  buffer_M2[];
double  buffer_M3[];
double  buffer_M4[];
double  buffer_M5[];



//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool PlotTrend(const long              chart_ID=0,
               string                  nameP="trendline",
               const int               subwindow=0,
               datetime                time1=0,
               double                  price1=0,
               datetime                time2=0,
               double                  price2=0,
               const color             clr=clrBlack,
               const ENUM_LINE_STYLE   style=STYLE_SOLID,
               const int               width=2,
               const bool              back=true,
               const bool              selection=false,
               const bool              ray=false,
               const bool              hidden=false)
  {
  
     string name = "Market_ "+nameP;
     
   ResetLastError();
   ObjectDelete(0,name);
   if(!ObjectCreate(chart_ID,name,OBJ_TREND,subwindow,time1,price1,time2,price2))
     {
      Print(__FUNCTION__,": failed to create arrow = ",GetLastError());
      return(false);
     }
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,width);
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_RAY,ray);
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);
   return(true);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool PlotRectangle(const long        chart_ID=0,
                   string            nameP="rectangle",
                   const int         subwindow=0,
                   datetime          time1=0,
                   double            price1=1,
                   datetime          time2=0,
                   double            price2=0,
                   const color       clr=clrGray,
                   const bool        back=true,
                   const bool        selection=false,
                   const bool        hidden=true)
  {
  
   
   string name = "Market_ "+nameP;
   ObjectDelete(0,name);
   if(!ObjectCreate(chart_ID,name,OBJ_RECTANGLE,subwindow,time1,price1,time2,price2))
     {
      Print(__FUNCTION__,": failed to create arrow = ",GetLastError());
      return(false);
     }
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);
   ObjectSetInteger(chart_ID,name,OBJPROP_FILL,true);
   return(true);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool PlotText(const long        chart_ID=0,
              string            nameP="text",
              const int         subwindow=0,
              datetime          time1=0,
              double            price1=0,
              const string      text="text",
              const string      font="Arial",
              const int         font_size=10,
              const color       clr=clrGray,
              const ENUM_ANCHOR_POINT anchor = ANCHOR_RIGHT_UPPER,
              const bool        back=true,
              const bool        selection=false,
              const bool        hidden=true)
  {
  
string name = "Market_ "+nameP;

   ObjectDelete(0,name);
   ResetLastError();
   if(!ObjectCreate(chart_ID,name,OBJ_TEXT,subwindow,time1,price1))
     {
      Print(__FUNCTION__,": failed to create arrow = ",GetLastError());
      return(false);
     }
   ObjectSetString(chart_ID,name,OBJPROP_TEXT,text);
   ObjectSetString(chart_ID,name,OBJPROP_FONT,font);
   ObjectSetInteger(chart_ID,name,OBJPROP_FONTSIZE,TaillePolice);
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,anchor);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);
   return(true);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void LevelsDraw(int      shft,
                datetime tmestrt,
                datetime tmend,
                string   name,
                bool     future)
  {
   highValue  = iHigh(NULL,timePeriodConverter(),shft);
   lowValue   = iLow(NULL,timePeriodConverter(),shft);
   openValue  = iOpen(NULL,timePeriodConverter(),shft);
   if(future==false)
     {
      closeValue = iClose(NULL,timePeriodConverter(),shft+1);
     }
   else
     {
      closeValue = SymbolInfoDouble(Symbol(),SYMBOL_BID);
     }

   PP  = (highValue+lowValue+closeValue)/3.0;


   R1 = 2*PP-lowValue;
   R2 = PP+(highValue - lowValue);
   R3 = 0;

   S1 = 2*PP-highValue;
   S2 = PP-(highValue - lowValue);
   S3 = 0;

   M0=0;
   M1=0.5*(S1+S2);
   M2=0.5*(PP+S1);
   M3=0.5*(PP+R1);
   M4=0.5*(R1+R2);
   M5=0;


   buffer_PP[shft] = PP;
   buffer_R1[shft] = R1;
   buffer_R2[shft] = R2;
   buffer_R3[shft] = R3;
   buffer_S1[shft] = S1;
   buffer_S2[shft] = S2;
   buffer_S3[shft] = S3;
   buffer_M0[shft] = M0;
   buffer_M1[shft] = M1;
   buffer_M2[shft] = M2;
   buffer_M3[shft] = M3;
   buffer_M4[shft] = M4;
   buffer_M5[shft] = M5;


   if (  highValue  == 0 || lowValue == 0 ) printf(" Refresh MT5 TimeFrame Data of "+Symbol());
   else
   {
   rangeopen1  = (openValue-lowValue)/((highValue-lowValue)/100);
   rangeopen2  = 100-((openValue-lowValue)/((highValue-lowValue)/100));
   rangeclose1 = (closeValue-lowValue)/((highValue-lowValue)/100);
   rangeclose2 = 100-((closeValue-lowValue)/((highValue-lowValue)/100));
   }
   
   if(PlotPivots)
     {
      if(modeInd == bullishMode)
        {
         PlotTrend(ChartID(),"R2"+name,0,tmestrt,R2,tmend,R2,ColorRes,StylePivots,WidthPivots+1);
        }

      PlotTrend(0,"PP"+name,0,tmestrt,PP,tmend,PP,ColorPP,StylePivots,WidthPivots);

      if(modeInd == bearishMode)
        {
         PlotTrend(ChartID(),"S2"+name,0,tmestrt,S2,tmend,S2,ColorSup,StylePivots,WidthPivots+1);
        }

      if(PlotPivotLabels)
        {
         if(modeInd == bullishMode)
           {
            PlotText(ChartID(),"R2L"+name,0,tmend,R2,"R2 : Bullish Profit Zone - Agressive","Arial",8,ColorRes,ANCHOR_RIGHT_UPPER);
           }
         PlotText(0,"PPL"+name,0,tmend,PP,"PP","Arial",8,ColorPP,ANCHOR_RIGHT_UPPER);

         if(modeInd == bearishMode)
           {
            PlotText(ChartID(),"S2L"+name,0,tmend,S2,"S2 : Bearish Profit Zone - Agressive","Arial",8,ColorSup,ANCHOR_RIGHT_UPPER);
           }

        }
      if(PlotPivotPrices)
        {
         if(modeInd == bullishMode)
           {
            PlotText(ChartID(),"R2P"+name,0,tmestrt,R2,DoubleToString(R2,4),"Arial",8,ColorRes,ANCHOR_LEFT_UPPER);
           }

         PlotText(0,"PPP"+name,0,tmestrt,PP,DoubleToString(PP,4),"Arial",8,ColorPP,ANCHOR_LEFT_UPPER);
         if(modeInd == bearishMode)
           {
            PlotText(ChartID(),"S2P"+name,0,tmestrt,S2,DoubleToString(S2,4),"Arial",8,ColorSup,ANCHOR_LEFT_UPPER);
           }
        }
     }

   if(PlotMidpoints)
     {
      if(modeInd == bearishMode)
        {
         PlotTrend(ChartID(),"M0"+name,0,tmestrt,M0,tmend,M0,ColorM02,StyleMidpoints,WidthMidpoints);
         PlotTrend(ChartID(),"M1"+name,0,tmestrt,M1,tmend,M1,ColorM02,StyleMidpoints,WidthMidpoints);
        }

      if(modeInd == bullishMode)
        {
         PlotTrend(ChartID(),"M4"+name,0,tmestrt,M4,tmend,M4,ColorM35,StyleMidpoints,WidthMidpoints);
         PlotTrend(ChartID(),"M5"+name,0,tmestrt,M5,tmend,M5,ColorM35,StyleMidpoints,WidthMidpoints);
        }

      if(PlotPivotLabels)
        {
         if(modeInd == bearishMode)
           {
            PlotText(ChartID(),"M0L"+name,0,tmend,M0,"M0","Arial",8,ColorSup,ANCHOR_RIGHT_UPPER);
            PlotText(ChartID(),"M1L"+name,0,tmend,M1,"M1 : Bearish Profit Target - Conservative","Arial",8,ColorSup,ANCHOR_RIGHT_UPPER);
            PlotText(ChartID(),"M3L"+name,0,tmend,(M3+(100*_Point)),"M3 : Bearish Selling Zone","Arial",8,ColorRes,ANCHOR_RIGHT_UPPER);

           }
         if(modeInd == bullishMode)
           {
            PlotText(ChartID(),"M2L"+name,0,tmend,M2,"M2 : Bullish Buying Zone","Arial",8,ColorSup,ANCHOR_RIGHT_UPPER);
            PlotText(ChartID(),"M4L"+name,0,tmend,M4,"M4 : Bullish Profit Target - Conservative","Arial",8,ColorRes,ANCHOR_RIGHT_UPPER);
           }

        }
      if(PlotPivotPrices)
        {
         if(modeInd == bearishMode)
           {
            PlotText(ChartID(),"M0P"+name,0,tmestrt,M0,DoubleToString(M0,4),"Arial",8,ColorSup,ANCHOR_LEFT_UPPER);
            PlotText(ChartID(),"M1P"+name,0,tmestrt,M1,DoubleToString(M1,4),"Arial",8,ColorSup,ANCHOR_LEFT_UPPER);
           }
         if(modeInd == bullishMode)
           {
            PlotText(ChartID(),"M4P"+name,0,tmestrt,M4,DoubleToString(M4,4),"Arial",8,ColorRes,ANCHOR_LEFT_UPPER);
           }
        }
     }

   if(PlotZones)
     {
      if(modeInd == bullishMode)
        {
         PlotRectangle(ChartID(),"BZ"+name,0,tmestrt,PP,tmend,M2,ColorBuyZone);
         if(future == true)
           {
            draw_obj("CP","Copyright "+IntegerToString(TimeYearMQL4(TimeCurrent()))+", FX BOOTCAMP TRAINING LLC",TaillePolice+4,clrDarkBlue,4,NotesLocation_x+400,NotesLocation_y+15,NotesFont);
            draw_obj("tradingZone","Projected Buy Zone : "+DoubleToString(PP,_Digits)+" - "+DoubleToString(M2,_Digits),TaillePolice+4,clrGreen,4,NotesLocation_x+400,NotesLocation_y+50,NotesFont);
           }

        }

      if(modeInd == bearishMode)
        {
         PlotRectangle(ChartID(),"SZ"+name,0,tmestrt,PP,tmend,M3,ColorSellZone);
         if(future == true)
           {
            draw_obj("CP","Copyright "+IntegerToString(TimeYearMQL4(TimeCurrent()))+", FX BOOTCAMP TRAINING LLC",TaillePolice+4,clrDarkBlue,4,NotesLocation_x+400,NotesLocation_y+15,NotesFont);
            draw_obj("tradingZone","Projected Sell Zone : "+DoubleToString(PP,_Digits)+" - "+DoubleToString(M3,_Digits),TaillePolice+4,clrRed,4,NotesLocation_x+400,NotesLocation_y+50,NotesFont);
           }

        }

     }






  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   cleanChart(); 

   SetIndexBuffer(0,buffer_PP);
   SetIndexBuffer(1,buffer_R1);
   SetIndexBuffer(2,buffer_R2);
   SetIndexBuffer(3,buffer_R3);
   SetIndexBuffer(4,buffer_S1);
   SetIndexBuffer(5,buffer_S2);
   SetIndexBuffer(6,buffer_S3);
   SetIndexBuffer(7,buffer_M0);
   SetIndexBuffer(8,buffer_M1);
   SetIndexBuffer(9,buffer_M2);
   SetIndexBuffer(10,buffer_M3);
   SetIndexBuffer(11,buffer_M4);
   SetIndexBuffer(12,buffer_M5);
   


    ArraySetAsSeries(buffer_PP,true);
     ArraySetAsSeries(buffer_R1,true);
         ArraySetAsSeries(buffer_R2,true);
             ArraySetAsSeries(buffer_R3,true);
                 ArraySetAsSeries(buffer_S1,true);
                     ArraySetAsSeries(buffer_S2,true);
                         ArraySetAsSeries(buffer_S3,true);
                             ArraySetAsSeries(buffer_M0,true);
                                 ArraySetAsSeries(buffer_M1,true);
                                     ArraySetAsSeries(buffer_M2,true);
                                        ArraySetAsSeries(buffer_M3,true);
                                          ArraySetAsSeries(buffer_M4,true);
                                             ArraySetAsSeries(buffer_M5,true);
   if(TimePeriod==Daily)
     {
      period="D1";
     }
   if(TimePeriod==Weekly)
     {
      period="W1";
     }
   if(TimePeriod==Monthly)
     {
      period="MN1";
     }
     

                                        
   return(INIT_SUCCEEDED);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   cleanChart(); 
   Comment("");
 }

//+------------------------------------------------------------------+
//|                                                                  |
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

    int zoneFind = 0;
    if( NewBarOrdersCechking() == true )
    {                                      
            cleanChart(); 
            for(shift=CountPeriods-1; shift>=0; shift--)
              {
               
               timestart = iTime(NULL,timePeriodConverter(),shift);
               timeend   = iTime(NULL,timePeriodConverter(),shift)+findNextFutureDate();;
               LevelsDraw(shift,timestart,timeend,period+IntegerToString(zoneFind),false);
               ChartRedraw();
              zoneFind++;
              }
         
            if(PlotFuturePivots)
              {
               timestart=iTime(NULL,timePeriodConverter(),0)+findNextFutureDate();
               timeend=iTime(NULL,timePeriodConverter(),0)+(findNextFutureDate()*2);
               LevelsDraw(0,timestart,timeend,"F"+period,true);
               ChartRedraw();
              }
  }
 
//--- return value of prev_calculated for next call
   return(rates_total);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool draw_obj(string nameP,string text,int size,color clr,int cor,int x,int y,string font)
  {
//----

string name = "Market_ "+nameP;

int chart_ID                  = 0;
double            angle       = 0.0;
ENUM_ANCHOR_POINT anchor      = ANCHOR_LEFT_UPPER;
bool              back        = false;
bool              selection   = false;
bool              hidden      = true;
long              z_order     = 0;
ENUM_BASE_CORNER  corner      = CORNER_LEFT_UPPER;
int               sub_window  = 0;
     
//--- reset the error value 
   ResetLastError(); 
   ObjectDelete(chart_ID,name);
//--- create a text label 
   if(!ObjectCreate(chart_ID,name,OBJ_LABEL,sub_window,0,0)) 
     { 
      Print(__FUNCTION__, 
            ": failed to create text label! Error code = ",GetLastError()); 
      return(false); 
     } 
//--- set label coordinates 
   ObjectSetInteger(chart_ID,name,OBJPROP_XDISTANCE,x); 
   ObjectSetInteger(chart_ID,name,OBJPROP_YDISTANCE,y); 
//--- set the chart's corner, relative to which point coordinates are defined 
   ObjectSetInteger(chart_ID,name,OBJPROP_CORNER,corner); 
//--- set the text 
   ObjectSetString(chart_ID,name,OBJPROP_TEXT,text); 
//--- set text font 
   ObjectSetString(chart_ID,name,OBJPROP_FONT,font); 
//--- set font size 
   ObjectSetInteger(chart_ID,name,OBJPROP_FONTSIZE,size); 
//--- set the slope angle of the text 
   ObjectSetDouble(chart_ID,name,OBJPROP_ANGLE,angle); 
//--- set anchor type 
   ObjectSetInteger(chart_ID,name,OBJPROP_ANCHOR,anchor); 
//--- set color 
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr); 
//--- display in the foreground (false) or background (true) 
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back); 
//--- enable (true) or disable (false) the mode of moving the label by mouse 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection); 
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection); 
//--- hide (true) or display (false) graphical object name in the object list 
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden); 
//--- set the priority for receiving the event of a mouse click in the chart 
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order); 
//--- successful execution 
   return(true); 
   
  }


/* ====================================== */
/* ====================================== */
/* ====================================== */

ENUM_TIMEFRAMES timePeriodConverter()
{
ENUM_TIMEFRAMES tmf = 0;

  if ( TimePeriod == Daily   ) tmf = PERIOD_D1;
  if ( TimePeriod == Weekly  ) tmf = PERIOD_W1;
  if ( TimePeriod == Monthly ) tmf = PERIOD_MN1;
  
  return(tmf); 
}

/* ====================================== */
/* ====================================== */
/* ====================================== */


int TimeYearMQL4(datetime date)
  {
   MqlDateTime tm;
   TimeToStruct(date,tm);
   return(tm.year);
  }
  
/* ====================================== */
/* ====================================== */
/* ====================================== */

long findNextFutureDate()
{
long val = 0;

long month = 604800*4;
  if ( TimePeriod == Daily   ) val  = 86400;
  if ( TimePeriod == Weekly  ) val  = 604800;
  if ( TimePeriod == Monthly ) val  = month;
  
  return(val);
}

/* ====================================== */
/* ====================================== */
/* ====================================== */

void cleanChart()
{
   for (int MyObjectsCount =ObjectsTotal(0)-1; MyObjectsCount>=0; MyObjectsCount--)
   {
      string name  = ObjectName(0,MyObjectsCount);
      if (StringFind(name,"Market_",0)>=0)
        ObjectDelete(0,name);
   }
}

/* ================================================================= */
/* ================================================================= */
/* ================================================================= */
/* ================================================================= */



bool NewBarOrdersCechking()
{
static datetime dt = 0;

datetime Time[];
   // number of elements to copy
ArraySetAsSeries(Time,true);
CopyTime(_Symbol,_Period,0,Bars(_Symbol,_Period),Time);

if (dt != Time[0] )
{
dt =  Time[0]; Sleep(100); // wait for tick
return(true);
}
return(false);
}