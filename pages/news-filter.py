import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.core import secrets
import pandas as pd
import requests
from datetime import datetime, timedelta, date, time
import pytz
import json
import time as time_module
from typing import List, Dict, Optional, Tuple

st.set_page_config(
    page_title="News Filter · Trading System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{font-family:'JetBrains Mono','Fira Code',monospace;}
    .stApp{background:var(--background-color);}
    section[data-testid="stSidebar"]{background:var(--secondary-background-color)!important;border-right:1px solid var(--border,#2a2a2a);}
    .card{background:var(--secondary-background-color);border:1px solid var(--border,#2a2a2a);border-radius:12px;padding:18px 20px;margin-bottom:14px;}
    .card-header{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#9a9a9a;margin-bottom:12px;}
    .hero{background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-40%;right:-5%;width:300px;height:300px;background:radial-gradient(circle,rgba(248,81,73,.06) 0%,transparent 70%);border-radius:50%;}

    /* Safe/Danger banner */
    .safe-banner{background:#0d2f1a;border:2px solid #00a32a;border-radius:14px;padding:24px 28px;text-align:center;margin-bottom:20px;}
    .danger-banner{background:#2f0d0d;border:2px solid #8b2d2d;border-radius:14px;padding:24px 28px;text-align:center;margin-bottom:20px;animation:pulse 2s infinite;}
    .warning-banner{background:#2f1f0d;border:2px solid #9e6a03;border-radius:14px;padding:24px 28px;text-align:center;margin-bottom:20px;}
    @keyframes pulse{0%{border-color:#8b2d2d;}50%{border-color:#ff3344;}100%{border-color:#8b2d2d;}}

    .banner-icon{font-size:42px;margin-bottom:8px;}
    .banner-title{font-size:22px;font-weight:700;margin-bottom:6px;}
    .banner-sub{font-size:13px;color:#9a9a9a;}

    /* Event rows */
    .event-card{border-radius:10px;padding:14px 16px;margin-bottom:8px;border-left:4px solid transparent;}
    .event-high   {background:#1e0d0d;border-left-color:#ff3344;}
    .event-medium {background:#1e1a0d;border-left-color:#ffcc00;}
    .event-low    {background:#0a0a0a;border-left-color:#2a2a2a;}
    .event-imminent{background:#2f0d0d;border:2px solid #ff3344;border-left:4px solid #ff3344;animation:pulse 1.5s infinite;}
    .event-critical{background:#2f0d0d;border:3px solid #ff3344;border-left:4px solid #ff3344;animation:pulse 0.8s infinite;}

    .event-time{font-size:14px;font-weight:700;color:#e6e6e6;font-family:monospace;}
    .event-name{font-size:14px;font-weight:600;color:#e6e6e6;margin:2px 0;}
    .event-country{font-size:11px;color:#9a9a9a;}
    .event-meta{display:flex;gap:12px;margin-top:6px;font-size:12px;color:#9a9a9a;}
    .event-meta span{display:flex;align-items:center;gap:4px;}

    .impact-badge-high  {background:#4a0d0d;color:#ff3344;border:1px solid #8b2d2d;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}
    .impact-badge-medium{background:#4a2d0d;color:#ffcc00;border:1px solid #9e6a03;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}
    .impact-badge-low   {background:#2a2a2a;color:#9a9a9a;border:1px solid #2a2a2a;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}
    .impact-badge-holiday{background:#0d2040;color:#00ff41;border:1px solid #1f4f8a;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}

    .countdown-box{background:#000000;border:1px solid #2a2a2a;border-radius:8px;padding:12px 14px;text-align:center;}
    .countdown-val{font-size:28px;font-weight:700;font-family:monospace;}
    .countdown-lbl{font-size:10px;color:#9a9a9a;margin-top:2px;letter-spacing:.05em;text-transform:uppercase;}

    .metric-box{background:var(--background-color);border:1px solid var(--border,#2a2a2a);border-radius:8px;padding:12px 14px;text-align:center;}
    .metric-value{font-size:20px;font-weight:700;color:#e6e6e6;}
    .metric-label{font-size:10px;color:#9a9a9a;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}

    .day-header{font-size:13px;font-weight:700;color:#00ff41;letter-spacing:.05em;text-transform:uppercase;padding:12px 0 6px;border-bottom:1px solid #2a2a2a;margin-bottom:10px;}
    .now-line{background:#00ff41;height:2px;border-radius:2px;margin:8px 0;position:relative;}
    .now-label{position:absolute;right:0;top:-10px;font-size:10px;color:#00ff41;font-weight:700;}

    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}

    .alert-box{background:#2f0d0d;border:2px solid #ff3344;border-radius:8px;padding:12px;margin:8px 0;}
    .alert-icon{font-size:20px;}
    .alert-text{color:#ffa198;font-size:13px;font-weight:600;}
    .alert-time{color:#ff3344;font-family:monospace;font-size:13px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CURRENCY → COUNTRY MAPPING
# ══════════════════════════════════════════════════════════════════
CCY_MAP = {
    "USD": {"flag": "🇺🇸", "name": "United States", "ff": "USD"},
    "EUR": {"flag": "🇪🇺", "name": "Euro Zone", "ff": "EUR"},
    "GBP": {"flag": "🇬🇧", "name": "United Kingdom", "ff": "GBP"},
    "AUD": {"flag": "🇦🇺", "name": "Australia", "ff": "AUD"},
    "NZD": {"flag": "🇳🇿", "name": "New Zealand", "ff": "NZD"},
    "JPY": {"flag": "🇯🇵", "name": "Japan", "ff": "JPY"},
    "CHF": {"flag": "🇨🇭", "name": "Switzerland", "ff": "CHF"},
    "CAD": {"flag": "🇨🇦", "name": "Canada", "ff": "CAD"},
    "ZAR": {"flag": "🇿🇦", "name": "South Africa", "ff": "ZAR"},
    "CNY": {"flag": "🇨🇳", "name": "China", "ff": "CNY"},
}

FLAG = {v["ff"]: v["flag"] for v in CCY_MAP.values()}

IMPACT_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Holiday": 3}

# Critical events that always need special attention
CRITICAL_EVENTS = [
    "NFP", "Nonfarm Payrolls", "CPI", "Consumer Price Index",
    "FOMC", "Federal Funds Rate", "GDP", "Gross Domestic Product",
    "Retail Sales", "Unemployment Rate", "ISM Manufacturing",
    "Interest Rate Decision", "ECB", "BOE", "BOJ", "RBA", "RBNZ"
]


# ══════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ff_calendar(week: str = "thisweek"):
    """
    Fetch Forex Factory calendar JSON with retry logic.
    week = 'thisweek' | 'nextweek'
    Returns list of raw event dicts.
    """
    url = f"https://nfs.faireconomy.media/ff_calendar_{week}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data:  # Ensure data is not empty
                return data
        except requests.exceptions.Timeout:
            if attempt < 2:
                time_module.sleep(1)
            else:
                st.warning(f"Forex Factory request timed out after 3 attempts")
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time_module.sleep(1)
            else:
                st.warning(f"Failed to fetch Forex Factory data: {str(e)}")
        except json.JSONDecodeError:
            if attempt < 2:
                time_module.sleep(1)
            else:
                st.warning("Invalid JSON response from Forex Factory")
    return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fmp_calendar(api_key: str, from_date: str, to_date: str):
    """Fallback: Financial Modeling Prep economic calendar (free tier)."""
    if not api_key or api_key == "YOUR_KEY":
        return []

    url = (f"https://financialmodelingprep.com/api/v3/economic_calendar"
           f"?from={from_date}&to={to_date}&apikey={api_key}")

    for attempt in range(2):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            raw = r.json()

            # Validate response structure
            if not isinstance(raw, list):
                return []

            # Normalise to our schema
            out = []
            for e in raw:
                if not e.get("event") or not e.get("country"):
                    continue

                impact = "High" if e.get("impact") == "3" else \
                    "Medium" if e.get("impact") == "2" else "Low"
                out.append({
                    "title": e.get("event", ""),
                    "country": e.get("country", ""),
                    "date": e.get("date", "")[:10],
                    "time": e.get("date", "")[11:16] if len(e.get("date", "")) > 10 else "00:00",
                    "impact": impact,
                    "forecast": e.get("estimate", ""),
                    "previous": e.get("previous", ""),
                    "actual": e.get("actual", ""),
                })
            return out
        except requests.exceptions.RequestException:
            if attempt < 1:
                time_module.sleep(1)
            else:
                st.warning("FMP API request failed")
        except (json.JSONDecodeError, ValueError) as e:
            st.warning(f"FMP data parsing error: {str(e)}")
            break
    return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_finnhub_calendar(api_key: str, from_date: str, to_date: str):
    """Finnhub economic calendar — free tier, 60 calls/min. Times are UTC."""
    if not api_key:
        return []

    iso_to_ccy = {
        "US": "USD", "GB": "GBP", "EU": "EUR", "EMU": "EUR",
        "JP": "JPY", "AU": "AUD", "NZ": "NZD", "CA": "CAD",
        "CH": "CHF", "ZA": "ZAR", "CN": "CNY",
    }

    url = (f"https://finnhub.io/api/v1/calendar/economic"
           f"?from={from_date}&to={to_date}&token={api_key}")

    for attempt in range(2):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            raw = r.json().get("economicCalendar", [])

            out = []
            utc = pytz.utc

            for e in raw:
                t_str = e.get("time", "")
                if not t_str:
                    continue

                try:
                    utc_dt = utc.localize(datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S"))
                except (ValueError, TypeError):
                    utc_dt = None

                iso = e.get("country", "").upper()
                ccy = iso_to_ccy.get(iso, iso)

                out.append({
                    "_utc_dt": utc_dt,
                    "date_str": t_str[:10],
                    "time_str": t_str[11:16],
                    "title": e.get("event", ""),
                    "country": ccy,
                    "impact": e.get("impact", "low").capitalize(),
                    "forecast": str(e.get("estimate") or "—"),
                    "previous": str(e.get("prev") or "—"),
                    "actual": str(e.get("actual") or ""),
                    "flag": FLAG.get(ccy, "🌐"),
                })
            return out
        except requests.exceptions.RequestException:
            if attempt < 1:
                time_module.sleep(1)
            else:
                st.warning("Finnhub API request failed")
        except (json.JSONDecodeError, ValueError) as e:
            st.warning(f"Finnhub data parsing error: {str(e)}")
            break
    return []


# ══════════════════════════════════════════════════════════════════
# DATA VALIDATION & NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def validate_event(event: Dict) -> bool:
    """Validate event data structure and content."""
    required_fields = ['title', 'country', 'impact']
    for field in required_fields:
        if field not in event or not event[field]:
            return False

    # Validate impact level
    if event['impact'] not in ['High', 'Medium', 'Low', 'Holiday']:
        return False

    # Validate date format if present
    if 'date_str' in event and event['date_str']:
        try:
            datetime.strptime(event['date_str'], "%Y-%m-%d")
        except ValueError:
            return False

    return True


def is_critical_event(title: str) -> bool:
    """Check if event is considered critical for trading."""
    title_upper = title.upper()
    return any(crit.upper() in title_upper for crit in CRITICAL_EVENTS)


def normalise_finnhub(raw: list, tz_target) -> list:
    """Convert Finnhub UTC datetimes to the user's local timezone."""
    out = []
    for e in raw:
        utc_dt = e.get("_utc_dt")
        local_dt = utc_dt.astimezone(tz_target) if utc_dt else None
        entry = {k: v for k, v in e.items() if k != "_utc_dt"}
        entry["datetime_local"] = local_dt
        entry["is_critical"] = is_critical_event(entry.get("title", ""))
        out.append(entry)
    return out


def normalise_ff(raw: list, tz_target) -> list:
    """
    Parse FF calendar JSON and add datetime objects in the user's timezone.
    FF times are in America/New_York (ET).
    Enhanced with better error handling for edge cases.
    """
    et = pytz.timezone("America/New_York")
    events = []

    for e in raw:
        try:
            # Validate required fields
            if not e.get("date") or not e.get("title") or not e.get("country"):
                continue

            date_str = e.get("date", "")
            time_str = e.get("time", "")
            title = e.get("title", "")
            country = e.get("country", "").upper()
            impact = e.get("impact", "Low").capitalize()
            forecast = e.get("forecast", "") or "—"
            previous = e.get("previous", "") or "—"
            actual = e.get("actual", "") or ""

            # Parse datetime with multiple format support
            try:
                date_part = date_str[:10]
                base_date = datetime.strptime(date_part, "%Y-%m-%d")

                if time_str and time_str.lower() != "all day" and time_str.strip():
                    # Try multiple time formats
                    parsed_time = None
                    for fmt in ["%I:%M%p", "%H:%M", "%I:%M %p", "%I:%M:%S%p"]:
                        try:
                            parsed_time = datetime.strptime(time_str.strip(), fmt).time()
                            break
                        except ValueError:
                            continue

                    if parsed_time:
                        naive_dt = datetime.combine(base_date, parsed_time)
                    else:
                        st.warning(f"Could not parse time format: {time_str}")
                        naive_dt = base_date
                else:
                    naive_dt = base_date
                    if not time_str:
                        impact = "Holiday"

                local_dt = et.localize(naive_dt)
                target_dt = local_dt.astimezone(tz_target)
            except Exception as e:
                if 'debug_mode' in st.session_state and st.session_state.debug_mode:
                    st.warning(f"Error parsing event '{title}': {str(e)}")
                target_dt = None

            events.append({
                "datetime_local": target_dt,
                "date_str": date_part,
                "time_str": time_str,
                "title": title,
                "country": country,
                "impact": impact,
                "forecast": forecast,
                "previous": previous,
                "actual": actual,
                "flag": FLAG.get(country, "🌐"),
                "is_critical": is_critical_event(title),
            })
        except Exception as e:
            if 'debug_mode' in st.session_state and st.session_state.debug_mode:
                st.warning(f"Error processing event: {str(e)}")
            continue

    return events


def events_near_window(events: list, entry_dt: datetime, window_min: int = 60) -> list:
    """Return High-impact and critical events within ±window_min of entry_dt."""
    out = []
    for e in events:
        # Check for high impact OR critical events
        if e["impact"] == "High" or e.get("is_critical", False):
            if e["datetime_local"] is None:
                continue
            diff = abs((e["datetime_local"] - entry_dt).total_seconds() / 60)
            if diff <= window_min:
                out.append((diff, e))
    return sorted(out, key=lambda x: x[0])


# ══════════════════════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════

def check_alerts(events: List[Dict], user_tz) -> List[Dict]:
    """Generate alerts for critical events and high-impact events."""
    alerts = []
    now = datetime.now(user_tz)

    for e in events:
        if not e.get("datetime_local"):
            continue

        time_diff = (e["datetime_local"] - now).total_seconds() / 60

        # Critical events within 2 hours
        if e.get("is_critical", False) and time_diff <= 120 and time_diff > 0:
            alerts.append({
                "type": "critical",
                "severity": "high",
                "message": f"CRITICAL: {e['title']} in {int(time_diff)} min",
                "event": e,
                "minutes_away": int(time_diff)
            })
        # High impact within 30 minutes
        elif e["impact"] == "High" and time_diff <= 30 and time_diff > 0:
            alerts.append({
                "type": "high_impact",
                "severity": "medium",
                "message": f"HIGH IMPACT: {e['title']} in {int(time_diff)} min",
                "event": e,
                "minutes_away": int(time_diff)
            })
        # High impact within 1 hour
        elif e["impact"] == "High" and time_diff <= 60 and time_diff > 30:
            alerts.append({
                "type": "high_impact",
                "severity": "low",
                "message": f"Upcoming: {e['title']} in {int(time_diff)} min",
                "event": e,
                "minutes_away": int(time_diff)
            })

    return sorted(alerts, key=lambda x: x["minutes_away"])


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
           <style>
               span[data-baseweb="tag"]{
                   background-color:#00ff41 !important;
               }
               span[data-baseweb="tag"] span{
                   color:#000000 !important;
                   font-weight:600 !important;
               }
               span[data-baseweb="tag"] svg{
                   fill:#000000 !important;
               }
               span[data-baseweb="tag"] [role="button"]:hover{
                   background-color:#00cc34 !important;
               }
           </style>
           """, unsafe_allow_html=True)
    st.markdown("### 📰 News Filter")

    # Timezone picker
    tz_options = [
        "Africa/Johannesburg",
        "Europe/London",
        "America/New_York",
        "Europe/Frankfurt",
        "Asia/Tokyo",
        "Australia/Sydney",
        "America/Chicago",
        "America/Los_Angeles",
        "UTC",
    ]
    selected_tz_str = st.selectbox("🕐 Your Timezone", tz_options, index=0)
    user_tz = pytz.timezone(selected_tz_str)

    st.markdown("---")
    st.markdown("**🎯 Entry Window Check**")
    entry_date = st.date_input("Planned Entry Date", value=date.today())
    entry_time_input = st.time_input("Planned Entry Time (local)", value=time(9, 0))
    buffer_min = st.slider("News Buffer (minutes)", 15, 120, 60, 5)

    st.markdown("---")
    st.markdown("**🔍 Filters**")
    impact_filter = st.multiselect("Show Impact Levels",
                                   ["High", "Medium", "Low", "Holiday"],
                                   default=["High", "Medium"])

    ccy_filter = st.multiselect("Filter by Currency",
                                list(CCY_MAP.keys()), default=list(CCY_MAP.keys()))

    show_week = st.radio("Calendar Range", ["This Week", "Next Week", "Both"], index=0)

    st.markdown("---")
    st.markdown("**🔧 Settings**")
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)

    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = True
    st.session_state.alert_system = st.checkbox("Enable Alert System", value=st.session_state.alert_system)

    st.markdown("---")
    st.markdown("**🔑 API Keys**")
    fmp_key = secrets.fmp_api_key()
    if fmp_key:
        st.success("FMP ✓ loaded from secrets", icon="🔑")
    else:
        st.caption("FMP key not found in secrets.toml")
    finnhub_key = secrets.finnhub_api_key()
    if finnhub_key:
        st.success("Finnhub ✓ loaded from secrets", icon="🔑")
    else:
        st.caption("Finnhub key not found in secrets.toml")

    if st.button("🔄 Refresh Calendar", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# FETCH & NORMALISE
# ══════════════════════════════════════════════════════════════════

# Date ranges for fallback APIs
_today = date.today()
_wk_start = _today - timedelta(days=_today.weekday())
_wk_end = _wk_start + timedelta(days=6)
_nw_start = _wk_start + timedelta(days=7)
_nw_end = _nw_start + timedelta(days=6)
if show_week == "This Week":
    _fb_from, _fb_to = _wk_start.isoformat(), _wk_end.isoformat()
elif show_week == "Next Week":
    _fb_from, _fb_to = _nw_start.isoformat(), _nw_end.isoformat()
else:
    _fb_from, _fb_to = _wk_start.isoformat(), _nw_end.isoformat()

data_source = "No Data"

# Primary: Forex Factory mirror
with st.spinner("📡 Fetching economic calendar from Forex Factory…"):
    raw_this = fetch_ff_calendar("thisweek") if show_week in ["This Week", "Both"] else []
    raw_next = fetch_ff_calendar("nextweek") if show_week in ["Next Week", "Both"] else []
    raw_all = raw_this + raw_next

events_all = normalise_ff(raw_all, user_tz)
if events_all:
    data_source = "Forex Factory"

# Fallback 1: Finnhub
if not events_all and finnhub_key:
    with st.spinner("FF unavailable — trying Finnhub…"):
        fh_raw = fetch_finnhub_calendar(finnhub_key, _fb_from, _fb_to)
        if fh_raw:
            events_all = normalise_finnhub(fh_raw, user_tz)
            data_source = "Finnhub"

# Fallback 2: FMP
if not events_all and fmp_key:
    with st.spinner("Trying FMP fallback…"):
        fmp_raw = fetch_fmp_calendar(fmp_key, _fb_from, _fb_to)
        if fmp_raw:
            events_all = fmp_raw
            data_source = "FMP"

# Validate all events
events_all = [e for e in events_all if validate_event(e)]

# Filter by currency and impact
filtered = [
    e for e in events_all
    if e["impact"] in impact_filter
       and (e["country"] in ccy_filter or e["country"] == "")
]
filtered.sort(key=lambda e: (
    e["datetime_local"] or datetime(2099, 1, 1, tzinfo=user_tz),
    IMPACT_ORDER.get(e["impact"], 99)
))

# ── Now & entry window ─────────────────────────────────────────────────────────
now_local = datetime.now(user_tz)
entry_naive = datetime.combine(entry_date, entry_time_input)
entry_local = user_tz.localize(entry_naive)

# Today's high-impact events
today_str = now_local.strftime("%Y-%m-%d")
high_today = [e for e in events_all if e["impact"] == "High" and e["date_str"] == today_str]
critical_today = [e for e in events_all if e.get("is_critical", False) and e["date_str"] == today_str]
upcoming_high = [e for e in high_today
                 if e["datetime_local"] and e["datetime_local"] > now_local]

# Events near entry window
near_entry = events_near_window(events_all, entry_local, buffer_min)

# Next high-impact event
next_high = upcoming_high[0] if upcoming_high else None
mins_to_next = int((next_high["datetime_local"] - now_local).total_seconds() / 60) if next_high else None

# Safe-to-trade logic
SAFE = not near_entry
DANGER = bool(near_entry)

# Generate alerts if system is enabled
alerts = []
if st.session_state.alert_system:
    alerts = check_alerts(events_all, user_tz)

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="font-size:26px;font-weight:700;color:#e6e6e6;">📰 Economic News Filter</div>
      <div style="color:#9a9a9a;font-size:14px;margin-top:6px;">High-Impact Event Tracker · Red Folder News · Checklist Item #2</div>
      <div style="font-size:13px;color:#00ff41;font-weight:500;margin-top:4px;">
        🕐 {now_local.strftime('%A, %d %B %Y  |  %H:%M')} {selected_tz_str} · Source: {data_source}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#9a9a9a;">Entry Window</div>
      <div style="font-size:22px;font-weight:700;color:#e6e6e6;">{entry_local.strftime('%d %b  %H:%M')}</div>
      <div style="font-size:12px;color:#9a9a9a;">±{buffer_min} min buffer</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ALERTS DISPLAY
# ══════════════════════════════════════════════════════════════════
if alerts and st.session_state.alert_system:
    st.markdown("### 🚨 Active Alerts")
    alert_cols = st.columns(min(3, len(alerts)))

    for idx, alert in enumerate(alerts[:6]):  # Show max 6 alerts
        with alert_cols[idx % 3]:
            severity_color = "#ff3344" if alert["severity"] == "high" else "#ffcc00" if alert[
                                                                                            "severity"] == "medium" else "#9a9a9a"
            st.markdown(f"""
            <div class="alert-box" style="border-color:{severity_color};">
                <div class="alert-icon">{'🔴' if alert['type'] == 'critical' else '🟡'}</div>
                <div class="alert-text" style="color:{severity_color};">{alert['message']}</div>
                <div class="alert-time">{alert['event']['title']}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SAFE / DANGER BANNER
# ══════════════════════════════════════════════════════════════════
if SAFE:
    st.markdown(f"""
    <div class="safe-banner">
      <div class="banner-icon">✅</div>
      <div class="banner-title" style="color:#00ff66;">CLEAR TO TRADE</div>
      <div class="banner-sub">No high-impact or critical news within {buffer_min} minutes of your entry window
        ({entry_local.strftime('%H:%M')} {selected_tz_str.split('/')[-1]})</div>
    </div>
    """, unsafe_allow_html=True)
else:
    conflict_names = " · ".join([f"{e['flag']} {e['title']}" for _, e in near_entry[:3]])
    conflict_times = " · ".join([
        e["datetime_local"].strftime("%H:%M") if e["datetime_local"] else "?"
        for _, e in near_entry[:3]
    ])
    st.markdown(f"""
    <div class="danger-banner">
      <div class="banner-icon">🚨</div>
      <div class="banner-title" style="color:#ff3344;">⛔ DO NOT ENTER — HIGH-IMPACT NEWS NEARBY</div>
      <div class="banner-sub" style="color:#ffa198;font-weight:600;margin-bottom:6px;">{conflict_names}</div>
      <div class="banner-sub">Scheduled at {conflict_times} — within {buffer_min} min of your entry
        ({entry_local.strftime('%H:%M')})</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
all_high = [e for e in events_all if e["impact"] == "High"]
all_med = [e for e in events_all if e["impact"] == "Medium"]
total_evts = len(events_all)

kpis = [
    (k1, len(all_high), "🔴 High Impact", "#ff3344"),
    (k2, len(all_med), "🟡 Medium", "#ffcc00"),
    (k3, len(high_today), "Today High", "#ff3344"),
    (k4, len(critical_today), "Critical Today", "#ff3344" if critical_today else "#00ff66"),
    (k5, len(upcoming_high), "Upcoming High", "#ffcc00" if upcoming_high else "#00ff66"),
    (k6, len(near_entry), f"Near Entry ±{buffer_min}m", "#ff3344" if near_entry else "#00ff66"),
    (k7, f"{mins_to_next}m" if mins_to_next is not None else "None", "Next High Event",
     "#ff3344" if mins_to_next and mins_to_next < 60 else "#ffcc00" if mins_to_next else "#00ff66"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════════════════
# LEFT: FULL CALENDAR
# ══════════════════════════════════════════════════════════════════
with left:
    st.markdown("### 📅 Economic Calendar")

    # Export button
    if filtered:
        csv_data = pd.DataFrame([{
            'Date': e['date_str'],
            'Time': e.get('time_str', ''),
            'Country': e['country'],
            'Event': e['title'],
            'Impact': e['impact'],
            'Forecast': e.get('forecast', ''),
            'Previous': e.get('previous', ''),
            'Actual': e.get('actual', '')
        } for e in filtered]).to_csv(index=False)

        st.download_button(
            label="📥 Download Calendar CSV",
            data=csv_data,
            file_name=f"economic_calendar_{_today}.csv",
            mime="text/csv",
            key="download_calendar"
        )

    if not filtered:
        if not raw_all:
            st.warning("""
**No calendar data loaded.**

The Forex Factory feed couldn't be reached. This may be due to:
- No internet connection
- FF temporarily blocking the request

**Options:**
1. Click **🔄 Refresh Calendar** in the sidebar
2. Add your free **FMP API key** in the sidebar as fallback
3. Add your free **Finnhub API key** in the sidebar as fallback
4. Use the manual event tracker below
            """)
        else:
            st.info("No events match your current filters.")
    else:
        # Group by date
        from itertools import groupby


        def get_date(e):
            return e["date_str"]


        groups = {}
        for e in filtered:
            groups.setdefault(e["date_str"], []).append(e)

        for day_str, day_events in sorted(groups.items()):
            try:
                day_label = datetime.strptime(day_str, "%Y-%m-%d").strftime("%A, %d %B %Y")
            except Exception:
                day_label = day_str
            is_today = (day_str == today_str)
            day_color = "#00ff41" if is_today else "#9a9a9a"
            today_marker = " 📍 TODAY" if is_today else ""

            st.markdown(f'<div class="day-header" style="color:{day_color};">{day_label}{today_marker}</div>',
                        unsafe_allow_html=True)

            past_now = False
            for e in day_events:
                # "NOW" marker
                if is_today and not past_now and e["datetime_local"] and e["datetime_local"] > now_local:
                    st.markdown(f'<div class="now-line"><span class="now-label">NOW ▼</span></div>',
                                unsafe_allow_html=True)
                    past_now = True

                # Card styling
                is_near = any(ev is e for _, ev in near_entry)
                is_critical = e.get("is_critical", False)

                if is_near and is_critical:
                    card_cls = "event-critical"
                elif is_near:
                    card_cls = "event-imminent"
                elif e["impact"] == "High":
                    card_cls = "event-high"
                elif e["impact"] == "Medium":
                    card_cls = "event-medium"
                else:
                    card_cls = "event-low"

                impact_badge = {
                    "High": '<span class="impact-badge-high">🔴 HIGH</span>',
                    "Medium": '<span class="impact-badge-medium">🟡 MEDIUM</span>',
                    "Low": '<span class="impact-badge-low">⚪ LOW</span>',
                    "Holiday": '<span class="impact-badge-holiday">📅 HOLIDAY</span>',
                }.get(e["impact"], "")

                critical_badge = '<span class="impact-badge-high" style="background:#4a0000;">⚠️ CRITICAL</span>' if is_critical else ""

                # Time display
                if e["datetime_local"]:
                    t_display = e["datetime_local"].strftime("%H:%M")
                    is_past = e["datetime_local"] < now_local and is_today
                elif e["time_str"].lower() == "all day":
                    t_display = "All Day"
                    is_past = False
                else:
                    t_display = e["time_str"] or "—"
                    is_past = False

                past_style = "opacity:0.45;" if is_past else ""
                near_warn = '<span style="color:#ff3344;font-weight:700;font-size:12px;margin-left:8px;">⚠️ WITHIN WINDOW</span>' if is_near else ""

                # Actual vs forecast
                actual_str = ""
                if e.get("actual"):
                    actual_str = f'<span>✅ Actual: <strong style="color:#e6e6e6">{e["actual"]}</strong></span>'
                forecast_str = f'<span>📊 Forecast: {e["forecast"]}</span>' if e["forecast"] != "—" else ""
                prev_str = f'<span>🕐 Previous: {e["previous"]}</span>' if e["previous"] != "—" else ""

                # Minutes away
                mins_away = ""
                if e["datetime_local"] and not is_past and is_today:
                    delta = int((e["datetime_local"] - now_local).total_seconds() / 60)
                    if delta < 60:
                        mins_away = f'<span style="color:#ff3344;font-weight:700;">⏱ In {delta} min</span>'
                    elif delta < 180:
                        mins_away = f'<span style="color:#ffcc00;">⏱ In {delta // 60}h {delta % 60}m</span>'

                st.markdown(f"""
                <div class="event-card {card_cls}" style="{past_style}">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
                    <div style="flex:1;">
                      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                        <span class="event-time">{t_display}</span>
                        {impact_badge}
                        {critical_badge}
                        <span style="font-size:15px;">{e['flag']}</span>
                        <span style="font-size:12px;color:#9a9a9a;font-weight:600;">{e['country']}</span>
                        {near_warn}
                      </div>
                      <div class="event-name">{e['title']}</div>
                      <div class="event-meta">
                        {forecast_str}{prev_str}{actual_str}
                      </div>
                    </div>
                    <div style="text-align:right;min-width:80px;">{mins_away}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT: PANELS
# ══════════════════════════════════════════════════════════════════
with right:
    # ── Countdown to next high-impact event ───────────────────────
    st.markdown('<div class="card"><div class="card-header">⏱️ Countdown to Next High-Impact Event</div>',
                unsafe_allow_html=True)
    if next_high:
        delta = next_high["datetime_local"] - now_local
        hrs = int(delta.total_seconds() // 3600)
        mins = int((delta.total_seconds() % 3600) // 60)
        c1, c2, c3 = st.columns(3)
        count_color = "#ff3344" if hrs == 0 and mins < 60 else "#ffcc00" if hrs < 3 else "#00ff66"
        with c1:
            st.markdown(
                f'<div class="countdown-box"><div class="countdown-val" style="color:{count_color};">{hrs:02d}</div><div class="countdown-lbl">Hours</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="countdown-box"><div class="countdown-val" style="color:{count_color};">{mins:02d}</div><div class="countdown-lbl">Minutes</div></div>',
                unsafe_allow_html=True)
        with c3:
            status_emoji = "🔴" if hrs == 0 and mins < 60 else "🟡" if hrs < 3 else "🟢"
            st.markdown(
                f'<div class="countdown-box"><div class="countdown-val" style="color:{count_color};">{status_emoji}</div><div class="countdown-lbl">Status</div></div>',
                unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top:12px;padding:10px;background:#000000;border-radius:8px;">
          <div style="font-size:13px;font-weight:700;color:#e6e6e6;">{next_high['flag']} {next_high['title']}</div>
          <div style="font-size:12px;color:#9a9a9a;margin-top:3px;">
            {next_high['country']} · {next_high['datetime_local'].strftime('%H:%M %Z') if next_high['datetime_local'] else '?'}
          </div>
          <div style="font-size:11px;color:#9a9a9a;margin-top:3px;">
            Forecast: {next_high['forecast']} · Previous: {next_high['previous']}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center;padding:20px;color:#00ff66;font-size:16px;font-weight:700;">🟢 No upcoming high-impact events today</div>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Entry window conflict list ─────────────────────────────────
    st.markdown('<div class="card"><div class="card-header">🎯 Entry Window Conflict Check</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:12px;color:#9a9a9a;margin-bottom:10px;">Entry: <strong style="color:#e6e6e6;">{entry_local.strftime("%d %b %H:%M")}</strong> ±{buffer_min} min buffer</div>',
        unsafe_allow_html=True)
    if near_entry:
        for diff_min, e in near_entry:
            direction = "before" if e["datetime_local"] < entry_local else "after"
            is_critical = e.get("is_critical", False)
            border_color = "#ff3344" if is_critical else "#8b2d2d"
            bg_color = "#2f0d0d" if is_critical else "#1e0d0d"

            st.markdown(f"""
            <div style="background:{bg_color};border:1px solid {border_color};border-radius:8px;padding:10px 12px;margin-bottom:8px;">
              <div style="font-size:13px;font-weight:700;color:#ff3344;">
                {'⚠️ CRITICAL: ' if is_critical else '🚨 '}{e['flag']} {e['title']}
              </div>
              <div style="font-size:12px;color:#9a9a9a;margin-top:3px;">
                {e['country']} · {e['datetime_local'].strftime('%H:%M') if e['datetime_local'] else '?'}
                · <strong style="color:#ffa198;">{int(diff_min)} min {direction} entry</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#ff3344;font-weight:600;padding:6px 0;">⛔ Wait for all events to clear before entering.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#0d2f1a;border:1px solid #00a32a;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:18px;font-weight:700;color:#00ff66;">✅ Entry Window is Clear</div>
          <div style="font-size:12px;color:#9a9a9a;margin-top:6px;">No high-impact or critical events within {buffer_min} minutes of {entry_local.strftime('%H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Today's high-impact summary ────────────────────────────────
    st.markdown('<div class="card"><div class="card-header">🔴 Today\'s High-Impact Events</div>',
                unsafe_allow_html=True)
    if high_today:
        for e in high_today:
            if e["datetime_local"]:
                t_str = e["datetime_local"].strftime("%H:%M")
                is_past = e["datetime_local"] < now_local
                is_upc = e["datetime_local"] > now_local
            else:
                t_str, is_past, is_upc = e["time_str"], False, False

            status_dot = "🔴" if is_upc and (e["datetime_local"] - now_local).total_seconds() < 3600 \
                else "🟡" if is_upc else "⚫"
            op = "opacity:0.5;" if is_past else ""
            is_critical = e.get("is_critical", False)
            critical_marker = " ⚠️" if is_critical else ""

            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 4px;border-bottom:1px solid #2a2a2a;{op}">
              <div>
                <span style="font-size:12px;">{status_dot} {e['flag']}{critical_marker}</span>
                <span style="font-size:13px;font-weight:600;color:#e6e6e6;margin-left:6px;">{e['title']}</span>
                <div style="font-size:11px;color:#9a9a9a;margin-top:2px;">{e['country']} · F: {e['forecast']} · P: {e['previous']}</div>
              </div>
              <div style="font-size:13px;font-weight:700;font-family:monospace;color:#ff3344;">{t_str}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center;padding:16px;color:#00ff66;font-weight:600;">🟢 No high-impact events today</div>',
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Critical events section ────────────────────────────────────
    if critical_today:
        st.markdown('<div class="card"><div class="card-header">⚠️ Today\'s Critical Events</div>',
                    unsafe_allow_html=True)
        for e in critical_today:
            if e["datetime_local"]:
                t_str = e["datetime_local"].strftime("%H:%M")
                is_past = e["datetime_local"] < now_local
            else:
                t_str, is_past = "—", False

            op = "opacity:0.5;" if is_past else ""

            st.markdown(f"""
            <div style="background:#2f0d0d;border:2px solid #ff3344;border-radius:8px;padding:10px 12px;margin-bottom:8px;{op}">
              <div style="font-size:13px;font-weight:700;color:#ff3344;">⚠️ {e['flag']} {e['title']}</div>
              <div style="font-size:12px;color:#9a9a9a;margin-top:3px;">
                {e['country']} · {t_str} · Impact: {e['impact']}
              </div>
              <div style="font-size:11px;color:#9a9a9a;margin-top:2px;">
                Forecast: {e['forecast']} · Previous: {e['previous']}
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Session danger zones ───────────────────────────────────────
    st.markdown('<div class="card"><div class="card-header">🕐 Session Danger Windows (Local Time)</div>',
                unsafe_allow_html=True)
    sessions = {
        "London Open": ("07:00", "09:30", "Europe/London"),
        "NY Open": ("08:00", "10:30", "America/New_York"),
        "London Close": ("15:00", "16:00", "Europe/London"),
        "NFP / CPI": ("08:30", "08:30", "America/New_York"),
        "FOMC": ("14:00", "14:30", "America/New_York"),
        "Tokyo Open": ("09:00", "10:00", "Asia/Tokyo"),
    }
    for sess_name, (t_open, t_close, sess_tz_str) in sessions.items():
        sess_tz = pytz.timezone(sess_tz_str)
        try:
            t_o = datetime.strptime(t_open, "%H:%M").replace(
                year=now_local.year, month=now_local.month, day=now_local.day)
            t_c = datetime.strptime(t_close, "%H:%M").replace(
                year=now_local.year, month=now_local.month, day=now_local.day)
            local_open = sess_tz.localize(t_o).astimezone(user_tz)
            local_close = sess_tz.localize(t_c).astimezone(user_tz)
            time_str = f"{local_open.strftime('%H:%M')}–{local_close.strftime('%H:%M')}"
        except Exception:
            time_str = f"{t_open} ({sess_tz_str})"

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:6px 4px;border-bottom:1px solid #2a2a2a;font-size:12px;">
          <span style="color:#e6e6e6;">{sess_name}</span>
          <span style="color:#ffcc00;font-family:monospace;font-weight:600;">{time_str}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DEBUG PANEL
# ══════════════════════════════════════════════════════════════════
if st.session_state.debug_mode:
    st.markdown("---")
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Data Source:**", data_source)
        st.write("**Total Raw Events:**", len(raw_all) if raw_all else 0)
        st.write("**Total Valid Events:**", len(events_all))
        st.write("**Filtered Events:**", len(filtered))
        st.write("**High Impact Today:**", len(high_today))
        st.write("**Critical Today:**", len(critical_today))
        st.write("**Near Entry:**", len(near_entry))
        st.write("**Active Alerts:**", len(alerts))

        if events_all:
            st.json({'sample_event': events_all[0] if events_all else {}})

        if st.button("Clear Cache & Refresh"):
            st.cache_data.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# MANUAL EVENT TRACKER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### ✏️ Manual Event Tracker")
st.caption("Add events manually if live feed is unavailable or you want to track custom events.")

if "manual_events" not in st.session_state:
    st.session_state.manual_events = []

with st.expander("➕ Add Manual Event", expanded=False):
    mc1, mc2, mc3, mc4, mc5 = st.columns([2, 2, 2, 2, 1])
    with mc1:
        m_title = st.text_input("Event Name", placeholder="e.g. US CPI")
    with mc2:
        m_country = st.selectbox("Currency", list(CCY_MAP.keys()))
    with mc3:
        m_date = st.date_input("Date", value=date.today(), key="m_date")
    with mc4:
        m_time = st.time_input("Time (local)", value=time(14, 30), key="m_time")
    with mc5:
        m_impact = st.selectbox("Impact", ["High", "Medium", "Low"])
        if st.button("➕ Add", use_container_width=True):
            if m_title:  # Validate title is not empty
                st.session_state.manual_events.append({
                    "datetime_local": user_tz.localize(datetime.combine(m_date, m_time)),
                    "date_str": m_date.isoformat(),
                    "time_str": m_time.strftime("%H:%M"),
                    "title": m_title,
                    "country": m_country,
                    "impact": m_impact,
                    "forecast": "—",
                    "previous": "—",
                    "actual": "",
                    "flag": CCY_MAP[m_country]["flag"],
                    "is_critical": is_critical_event(m_title),
                })
                st.success(f"Added: {m_title}")
            else:
                st.error("Please enter an event name")

if st.session_state.manual_events:
    st.markdown("**Manual Events:**")
    for idx, e in enumerate(st.session_state.manual_events):
        c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
        with c1:
            st.write(f"{e['flag']} {e['title']}")
        with c2:
            st.write(e["datetime_local"].strftime("%d %b %H:%M") if e["datetime_local"] else "?")
        with c3:
            badge_color = "#ff3344" if e["impact"] == "High" else "#ffcc00" if e["impact"] == "Medium" else "#9a9a9a"
            st.markdown(f'<span style="color:{badge_color};font-weight:700;">{e["impact"]}</span>',
                        unsafe_allow_html=True)
        with c4:
            if e.get("is_critical"):
                st.markdown('<span style="color:#ff3344;font-weight:700;">⚠️ CRITICAL</span>', unsafe_allow_html=True)
        with c5:
            if st.button("🗑️", key=f"del_manual_{idx}"):
                st.session_state.manual_events.pop(idx)
                st.rerun()

    if st.button("🗑️ Clear All Manual Events", use_container_width=True):
        st.session_state.manual_events = []
        st.rerun()

# Footer
st.markdown("""
<div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #2a2a2a;">
  📰 News Filter · Economic Calendar · Sources: Forex Factory / Finnhub / FMP · Refresh every 15 min · Not financial advice
</div>
""", unsafe_allow_html=True)