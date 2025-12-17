import streamlit as st

import inspect

# ---- Plotly rendering helper (prevents StreamlitDuplicateElementId) ----
_PLOTLY_SEQ = 0
def st_plotly(fig, base_key: str | None = None):
    """Render Plotly figure with an always-unique Streamlit element key."""
    global _PLOTLY_SEQ
    _PLOTLY_SEQ += 1
    if base_key is None:
        # Use caller function name as a stable key prefix
        base_key = inspect.stack()[1].function
    st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_{_PLOTLY_SEQ}")
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import requests
import time

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="JAMS Capital | Risk Management Terminal",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Bloomberg-like CSS
# =========================
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

.main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
.block-container {
    background-color: #000000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    padding: 0.5rem !important;
}

h1, h2, h3 {
    color: #FF9500 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    margin: 15px 0 10px 0 !important;
}

h1 {
    text-align: center;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px;
}

p, div, span, label, td, th,
.stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
    color: #FFFFFF !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 400 !important;
}

/* =========================
   Input controls (fix white-on-white dropdowns)
   ========================= */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #FF9500 !important;
}
div[data-baseweb="select"] span {
    color: #000000 !important;
}
div[data-baseweb="select"] input {
    color: #000000 !important;
}
ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #FF9500 !important;
}
ul[role="listbox"] span {
    color: #000000 !important;
}
div[data-baseweb="tag"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333333 !important;
}
div[data-baseweb="tag"] span {
    color: #FFFFFF !important;
}

.dataframe {
    background-color: #000000 !important;
    border: 1px solid #333333 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

.dataframe th {
    background-color: #1a1a1a !important;
    color: #FF9500 !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding: 6px !important;
    border: 1px solid #333333 !important;
}

.dataframe td {
    color: #FFFFFF !important;
    background-color: #000000 !important;
    text-align: center !important;
    padding: 6px !important;
    border: 1px solid #333333 !important;
    font-weight: 400 !important;
}

.stButton button {
    background-color: #FF9500 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
}

.risk-score-large {
    font-size: 6rem !important;
    font-weight: 700 !important;
    text-align: center;
    margin: 20px 0 !important;
    color: #FFFFFF !important;
}

.terminal-line {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    line-height: 1.45 !important;
    margin: 3px 0 !important;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid #FF9500;
    background: #111;
    font-size: 0.85rem;
}

.hr {
    height: 1px;
    background: #333;
    margin: 10px 0 18px 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Selectbox / dropdown: black text on white background */
div[data-baseweb="select"] * { color: #000000 !important; }
div[data-baseweb="select"] > div { background-color: #FFFFFF !important; border: 2px solid #FF9500 !important; }
div[data-baseweb="select"] svg { fill: #000000 !important; }

/* Date input text */
div[data-testid="stDateInput"] input { color: #000000 !important; background-color: #FFFFFF !important; border: 2px solid #FF9500 !important; }

/* Number input text */
div[data-testid="stNumberInput"] input { color: #000000 !important; }

</style>
""", unsafe_allow_html=True)

# =========================
# Utilities
# =========================
def _to_date(x) -> date:
    if isinstance(x, date):
        return x
    return pd.to_datetime(x).date()

def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    s = s.astype(float)
    mu = s.rolling(window, min_periods=max(20, window//5)).mean()
    sd = s.rolling(window, min_periods=max(20, window//5)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)

def regime_from_z(z: float) -> str:
    if pd.isna(z):
        return "INSUFFICIENT"
    if z >= 1.0:
        return "ELEVATED"
    if z <= -1.0:
        return "DEPRESSED"
    return "NORMAL"

def signal_from_z(z: float, direction: str) -> float:
    if pd.isna(z):
        return np.nan
    zc = float(np.clip(z, -3.0, 3.0))
    if direction == "higher_worse":
        return (zc + 3.0) / 6.0 * 100.0
    if direction == "lower_worse":
        return (3.0 - zc) / 6.0 * 100.0
    return (abs(zc) / 3.0) * 100.0

def plot_timeseries(df: pd.DataFrame, title: str, y_cols, y2_cols=None, y_title="Value", y2_title="", height=420) -> go.Figure:
    y2_cols = y2_cols or []
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    for c in y2_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c, yaxis="y2"))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color="#FF9500", family="IBM Plex Mono", size=14)),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#FFFFFF", family="IBM Plex Mono", size=10),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#FF9500",
            borderwidth=1,
            font=dict(color="#FFFFFF", family="IBM Plex Mono", size=10)
        ),
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor="#333333",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                bgcolor="#FF9500",
                activecolor="#FFFFFF",
                font=dict(color="#000000", family="IBM Plex Mono", size=10),
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            )
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#333333", title=y_title),
    )
    if y2_cols:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title=y2_title))
    return fig

# =========================
# Data Providers
# =========================
@st.cache_data(ttl=3600)
@st.cache_data(ttl=900)
def fetch_fred_series(series_id: str, start: date, end: date) -> pd.Series:
    """Fetch a FRED series as a daily time series.

    Streamlit Cloud frequently triggers urllib/pandas URL fetch issues (403/HTTPError) because pandas.read_csv(url)
    uses urllib without a browser-like User-Agent. We therefore fetch via requests with explicit headers, then parse.
    """
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_s}&coed={end_s}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JAMS-Risk-Terminal/1.0; +https://streamlit.app)",
        "Accept": "text/csv,application/csv;q=0.9,*/*;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        # Return empty series so the app can render an informative warning instead of hard-crashing.
        return pd.Series(dtype=float, name=series_id)

    df.columns = [c.strip() for c in df.columns]
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col].sort_index()
    s.name = series_id
    return s

@st.cache_data(ttl=900)
def fetch_stooq_prices(ticker: str, start: date, end: date) -> pd.Series:
    """Fetch daily adjusted close from Stooq (free, no key). Used as fallback when Yahoo is rate-limited on Streamlit Cloud."""
    # Stooq tickers are typically like 'spy.us', 'hyg.us'
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty or "Date" not in df.columns:
        return pd.Series(dtype=float)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Stooq uses 'Close' (not adjusted). Good enough for ratio stability; we can use Close.
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    return df["Close"].astype(float)

def _stooq_map(t: str) -> str:
    """Map common US ETFs to Stooq symbols."""
    m = {
        "SPY": "spy.us",
        "HYG": "hyg.us",
        "TLT": "tlt.us",
        "RSP": "rsp.us",
        "XLU": "xlu.us",
        "XLK": "xlk.us",
        "IEF": "ief.us",
        "SHY": "shy.us",
        "LQD": "lqd.us",
    }
    return m.get(t.upper(), f"{t.lower()}.us")



@st.cache_data(ttl=900)
def fetch_yf_prices(tickers, start: date, end: date) -> pd.DataFrame:
    """Fetch daily prices for tickers.

    On Streamlit Cloud, Yahoo often rate-limits (YFRateLimitError) causing empty/missing series.
    We therefore (1) try yfinance; (2) fill missing tickers via Stooq fallback.
    """
    tickers = [t.upper() for t in tickers]
    px = pd.DataFrame()
    try:
        df = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="column",
            threads=False,
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                # Expect ('Close', TICKER)
                if "Close" in df.columns.get_level_values(0):
                    px = df["Close"].copy()
            else:
                # Single ticker
                if "Close" in df.columns:
                    px = df[["Close"]].rename(columns={"Close": tickers[0]})
    except Exception:
        # Fall through to Stooq fill
        px = pd.DataFrame()

    # Normalize index/shape
    if px is None or px.empty:
        px = pd.DataFrame(index=pd.to_datetime([]))

    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px.sort_index()

    # Identify missing tickers (absent or all-NaN)
    missing = []
    for t in tickers:
        if t not in px.columns:
            missing.append(t)
        else:
            s = pd.to_numeric(px[t], errors="coerce")
            if s.dropna().empty:
                missing.append(t)

    # Fill missing tickers from Stooq
    if missing:
        for t in missing:
            stq = fetch_stooq_prices(_stooq_map(t), start, end)
            if not stq.empty:
                px[t] = stq

    # Clean final frame
    px = px.dropna(how="all").ffill()
    return px

@st.cache_data(ttl=3600)
def fetch_cot_sp500_legacy_net_spec(start: date, end: date, market_code: str = "13874+") -> pd.Series:
    """
    Robust COT fetch:
      - Try PRE API
      - Fall back to CFTC historical compressed ZIP (deacotYYYY.zip)
    Returns weekly series indexed by report date.
    """
    # 1) Try PRE API
    try:
        base = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        where = (
            f"cftc_contract_market_code='{market_code}' "
            f"AND report_date_as_yyyy_mm_dd >= '{start.strftime('%Y-%m-%d')}' "
            f"AND report_date_as_yyyy_mm_dd <= '{end.strftime('%Y-%m-%d')}'"
        )
        params = {"$where": where, "$order": "report_date_as_yyyy_mm_dd asc", "$limit": 5000}
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if rows:
            df = pd.DataFrame(rows)
            df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
            long_col = "noncomm_positions_long_all" if "noncomm_positions_long_all" in df.columns else "noncomm_positions_long"
            short_col = "noncomm_positions_short_all" if "noncomm_positions_short_all" in df.columns else "noncomm_positions_short"
            df[long_col] = pd.to_numeric(df[long_col], errors="coerce")
            df[short_col] = pd.to_numeric(df[short_col], errors="coerce")
            s = (df[long_col] - df[short_col]).rename("COT_NET_NONCOMM")
            return pd.Series(s.values, index=df["report_date_as_yyyy_mm_dd"]).dropna().sort_index()
    except Exception:
        pass

    # 2) Fallback ZIP
    import io, zipfile

    def _load_year(y: int) -> pd.DataFrame:
        url = f"https://www.cftc.gov/files/dea/history/deacot{y}.zip"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        members = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))] or z.namelist()
        name = members[0]
        raw = z.open(name).read().decode("utf-8", errors="replace")
        dfy = pd.read_csv(io.StringIO(raw))
        dfy.columns = [c.strip() for c in dfy.columns]
        return dfy

    frames = []
    for y in range(start.year, end.year + 1):
        try:
            frames.append(_load_year(y))
        except Exception:
            continue
    if not frames:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    df = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in df.columns}

    # date col
    date_col = None
    for k in ["report_date_as_yyyy-mm-dd", "report_date_as_yyyy_mm_dd"]:
        if k in cols:
            date_col = cols[k]
            break
    if date_col is None:
        for k, v in cols.items():
            if "report_date" in k:
                date_col = v
                break
    if date_col is None:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    mkt_col = None
    for k, v in cols.items():
        if "cftc_contract_market_code" in k:
            mkt_col = v
            break
    if mkt_col is None:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[(df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)]
    df = df[df[mkt_col].astype(str).str.strip().isin([market_code, market_code.replace("+", "")])]

    long_col = None
    short_col = None
    for k, v in cols.items():
        if long_col is None and "noncommercial" in k and "long" in k:
            long_col = v
        if short_col is None and "noncommercial" in k and "short" in k:
            short_col = v
    if long_col is None or short_col is None:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    df[long_col] = pd.to_numeric(df[long_col], errors="coerce")
    df[short_col] = pd.to_numeric(df[short_col], errors="coerce")
    net = (df[long_col] - df[short_col]).rename("COT_NET_NONCOMM")
    s = pd.Series(net.values, index=df[date_col]).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s

# =========================
# Core Modules
# =========================
def build_modules(start: date, end: date, z_window: int) -> dict:
    out = {}

    def _ensure_nonempty(s: pd.Series, name: str) -> pd.Series:
        """Ensure downstream code does not crash if a data vendor returns nothing."""
        if s is None or getattr(s, "empty", True):
            idx = pd.date_range(start, end, freq="B")
            return pd.Series(index=idx, dtype=float, name=name)
        s = s.copy()
        s.name = name
        return s

    # HY OAS
    hy = _ensure_nonempty(fetch_fred_series("BAMLH0A0HYM2", start, end).ffill(), "HY_OAS")
    df = pd.DataFrame({"HY_OAS": hy})
    df["Z"] = zscore(df["HY_OAS"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "higher_worse"))
    out["HY_OAS"] = df

    # HYG/SPY
    px = fetch_yf_prices(["HYG", "SPY"], start, end)
    ratio = (px["HYG"] / px["SPY"]).rename("HYG_SPY")
    df = pd.DataFrame({"HYG_SPY": ratio}).ffill()
    df["Z"] = zscore(df["HYG_SPY"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "lower_worse"))
    out["HYG_SPY"] = df

    # SOFR - 3M TBill
    sofr = _ensure_nonempty(fetch_fred_series("SOFR", start, end).ffill(), "SOFR")
    tb3 = _ensure_nonempty(fetch_fred_series("DTB3", start, end).ffill(), "DTB3")
    spread = (sofr - tb3).rename("SOFR_MINUS_TB3M")
    df = pd.DataFrame({"SOFR_MINUS_TB3M": spread}).ffill()
    df["Z"] = zscore(df["SOFR_MINUS_TB3M"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "higher_worse"))
    out["SOFR_TBILL"] = df

    # 10Y Breakeven
    be = fetch_fred_series("T10YIE", start, end).ffill()
    df = pd.DataFrame({"T10YIE": be})
    df["Z"] = zscore(df["T10YIE"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "higher_worse"))
    out["BREAKEVEN_10Y"] = df

    # COT (weekly -> daily ffill)
    cot = fetch_cot_sp500_legacy_net_spec(start, end).sort_index()
    cot_daily = cot.reindex(pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")).ffill()
    df = pd.DataFrame({"COT_NET_NONCOMM": cot_daily})
    # Use ~2 years of weekly equivalent after ffill
    df["Z"] = zscore(df["COT_NET_NONCOMM"], max(60, z_window // 3))
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "both_worse"))
    out["COT_SP500"] = df

    return out

def composite_snapshot(modules: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for name, df in modules.items():
        dfx = df[df.index <= as_of]
        if dfx.empty:
            continue
        last = dfx.iloc[-1]
        level_col = [c for c in df.columns if c not in ["Z", "SIGNAL"]][0]
        rows.append({
            "MODULE": name,
            "REGIME": regime_from_z(last["Z"]),
            "Z": np.round(float(last["Z"]), 2) if pd.notna(last["Z"]) else np.nan,
            "SIGNAL(0-100)": np.round(float(last["SIGNAL"]), 1) if pd.notna(last["SIGNAL"]) else np.nan,
            "LEVEL": np.round(float(last[level_col]), 4) if pd.notna(last[level_col]) else np.nan
        })
    if not rows:
        return pd.DataFrame(columns=["MODULE", "REGIME", "Z", "SIGNAL(0-100)", "LEVEL"])
    return pd.DataFrame(rows).sort_values("SIGNAL(0-100)", ascending=False)

# =========================
# Risk State Engine (Policy Layer)
# =========================
RISK_STATES = {
    "I":  {"name": "STRUCTURAL STABILITY", "name_zh": "結構穩定期"},
    "II": {"name": "INTERNAL FRAGILITY", "name_zh": "內部脆弱"},
    "III":{"name": "EARLY CREDIT DETERIORATION", "name_zh": "信用惡化初期"},
    "IV": {"name": "LIQUIDITY PRESSURE", "name_zh": "資金壓力浮現"},
    "V":  {"name": "SYSTEMIC RISK REGIME", "name_zh": "系統性風險"},
    "VII":{"name": "STRUCTURAL REPAIR", "name_zh": "結構修復期"},
}

def _latest(df: pd.DataFrame, col: str, as_of: pd.Timestamp):
    dfx = df[df.index <= as_of]
    if dfx.empty or col not in dfx.columns:
        return np.nan
    return dfx[col].iloc[-1]

def compute_v1_signals(v1_df: pd.DataFrame, z_window: int) -> pd.DataFrame:
    """Build V1 proxy signal frame (level, z, signal) to feed the risk-state engine."""
    df = v1_df.copy()
    out = pd.DataFrame(index=df.index)

    out["HYG_TLT"] = df["HYG_TLT"]
    out["HYG_TLT_Z"] = zscore(out["HYG_TLT"], z_window)
    out["HYG_TLT_SIGNAL"] = out["HYG_TLT_Z"].apply(lambda x: signal_from_z(x, "lower_worse"))

    out["USD_JPY_PROXY"] = df["USD_JPY_PROXY"]
    out["USD_JPY_Z"] = zscore(out["USD_JPY_PROXY"], z_window)
    out["USD_JPY_SIGNAL"] = out["USD_JPY_Z"].apply(lambda x: signal_from_z(x, "lower_worse"))

    out["RSP_SPY"] = df["RSP_SPY"]
    out["RSP_SPY_Z"] = zscore(out["RSP_SPY"], z_window)
    out["RSP_SPY_SIGNAL"] = out["RSP_SPY_Z"].apply(lambda x: signal_from_z(x, "lower_worse"))

    out["XLU_XLK"] = df["XLU_XLK"]
    out["XLU_XLK_Z"] = zscore(out["XLU_XLK"], z_window)
    out["XLU_XLK_SIGNAL"] = out["XLU_XLK_Z"].apply(lambda x: signal_from_z(x, "higher_worse"))

    return out

def classify_risk_state(modules: dict, v1_sig: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Rule-based risk-state classification with positioning amplifiers."""
    hy_sig   = _latest(modules["HY_OAS"], "SIGNAL", as_of) if "HY_OAS" in modules else np.nan
    hy_z     = _latest(modules["HY_OAS"], "Z", as_of) if "HY_OAS" in modules else np.nan

    hygsp_sig = _latest(modules["HYG_SPY"], "SIGNAL", as_of) if "HYG_SPY" in modules else np.nan
    hygsp_z   = _latest(modules["HYG_SPY"], "Z", as_of) if "HYG_SPY" in modules else np.nan

    sofr_sig = _latest(modules["SOFR_TBILL"], "SIGNAL", as_of) if "SOFR_TBILL" in modules else np.nan
    sofr_z   = _latest(modules["SOFR_TBILL"], "Z", as_of) if "SOFR_TBILL" in modules else np.nan

    be_z     = _latest(modules["BREAKEVEN_10Y"], "Z", as_of) if "BREAKEVEN_10Y" in modules else np.nan

    cot_z    = _latest(modules["COT_SP500"], "Z", as_of) if "COT_SP500" in modules else np.nan

    v1x = v1_sig[v1_sig.index <= as_of]
    if v1x.empty:
        v1_credit = v1_fx = v1_breadth = v1_def = np.nan
        v1_mean = np.nan
    else:
        v1_credit  = v1x["HYG_TLT_SIGNAL"].iloc[-1]
        v1_fx      = v1x["USD_JPY_SIGNAL"].iloc[-1]
        v1_breadth = v1x["RSP_SPY_SIGNAL"].iloc[-1]
        v1_def     = v1x["XLU_XLK_SIGNAL"].iloc[-1]
        v1_mean    = np.nanmean([v1_credit, v1_fx, v1_breadth, v1_def])

    HI = 70.0
    OK = 55.0

    amp = []
    if pd.notna(cot_z) and cot_z >= 1.5 and ((pd.notna(hy_sig) and hy_sig >= HI) or (pd.notna(sofr_sig) and sofr_sig >= HI) or (pd.notna(hygsp_sig) and hygsp_sig >= HI)):
        amp.append("POSITIONING_AMPLIFIER: crowded long + rising stress (assume speed/gaps)")
    if pd.notna(cot_z) and cot_z <= -1.5 and (pd.notna(hy_sig) and hy_sig < OK) and (pd.notna(sofr_sig) and sofr_sig < OK):
        amp.append("RELIEF_AMPLIFIER: crowded short + easing stress (short-cover risk)")

    repair = (pd.notna(hy_z) and hy_z < 0) and (pd.notna(sofr_z) and sofr_z < 0) and (pd.notna(hygsp_z) and hygsp_z < 0)
    still_defensive = pd.notna(v1_def) and v1_def >= HI

    systemic = (pd.notna(hy_sig) and hy_sig >= HI) and (pd.notna(sofr_sig) and sofr_sig >= HI) and (
        (pd.notna(hygsp_sig) and hygsp_sig >= HI) or (pd.notna(v1_fx) and v1_fx >= HI) or (pd.notna(v1_credit) and v1_credit >= HI)
    )

    liquidity_led = (pd.notna(sofr_sig) and sofr_sig >= HI) and (pd.isna(hy_sig) or hy_sig < HI)
    credit_early = ((pd.notna(hy_sig) and hy_sig >= HI) or (pd.notna(hygsp_sig) and hygsp_sig >= HI)) and (pd.isna(sofr_sig) or sofr_sig < HI)
    internal_fragility = ((pd.notna(v1_breadth) and v1_breadth >= HI) or (pd.notna(v1_def) and v1_def >= HI)) and (
        (pd.isna(hy_sig) or hy_sig < OK) and (pd.isna(sofr_sig) or sofr_sig < OK)
    )
    stability = ((pd.isna(hy_sig) or hy_sig < OK) and (pd.isna(sofr_sig) or sofr_sig < OK) and (pd.isna(hygsp_sig) or hygsp_sig < OK) and
                 (pd.isna(v1_mean) or v1_mean < OK))

    if systemic:
        code = "V"
    elif liquidity_led:
        code = "IV"
    elif credit_early:
        code = "III"
    elif repair and still_defensive:
        code = "VII"
    elif internal_fragility:
        code = "II"
    elif stability:
        code = "I"
    else:
        code = "II" if (pd.notna(v1_mean) and v1_mean >= OK) else "III" if (pd.notna(hy_sig) and hy_sig >= OK) else "I"

    key = []
    if pd.notna(hy_sig) and hy_sig >= HI: key.append("HY credit stress elevated")
    if pd.notna(hygsp_sig) and hygsp_sig >= HI: key.append("Credit underperforming equities (HYG/SPY)")
    if pd.notna(sofr_sig) and sofr_sig >= HI: key.append("Funding stress elevated (SOFR–TBill)")
    if pd.notna(v1_fx) and v1_fx >= HI: key.append("Risk-off FX impulse (JPY strength proxy)")
    if pd.notna(v1_breadth) and v1_breadth >= HI: key.append("Breadth deterioration (RSP/SPY)")
    if pd.notna(v1_def) and v1_def >= HI: key.append("Defensive rotation (XLU/XLK)")
    if pd.notna(be_z):
        if be_z <= -1.0: key.append("Macro tilt: recession/disinflation risk (breakevens low)")
        if be_z >=  1.0: key.append("Macro tilt: inflation persistence risk (breakevens high)")

    state = RISK_STATES[code]
    return {
        "code": code,
        "name": state["name"],
        "name_zh": state["name_zh"],
        "amplifiers": amp,
        "key_drivers": "; ".join(key) if key else "Mixed signals: monitor cross-confirmation and persistence."
    }

def exposure_policy_for_state(code: str) -> dict:
    """Suggested posture ranges; customize to JAMS policy as needed."""
    if code == "I":
        return {"gross": "120–180% NAV", "net": "30–80% NAV", "hedge": "0–10% notional", "liquidity": "5–12% NAV",
                "notes": "Allow risk to work; avoid over-hedging. Focus on alpha, not protection."}
    if code == "II":
        return {"gross": "90–150% NAV", "net": "20–60% NAV", "hedge": "5–20% notional", "liquidity": "8–18% NAV",
                "notes": "Fragility without system stress; prioritize optionality and quality."}
    if code == "III":
        return {"gross": "70–120% NAV", "net": "0–40% NAV", "hedge": "15–35% notional", "liquidity": "12–25% NAV",
                "notes": "Reduce sensitivity before forced repricing; tighten limits."}
    if code == "IV":
        return {"gross": "50–100% NAV", "net": "-10–25% NAV", "hedge": "25–50% notional", "liquidity": "20–35% NAV",
                "notes": "Funding stress is nonlinear; prioritize resilience and liquidity."}
    if code == "V":
        return {"gross": "20–60% NAV", "net": "-20–10% NAV", "hedge": "40–80% notional", "liquidity": "30–60% NAV",
                "notes": "Capital preservation regime; correlations rise and liquidity degrades."}
    if code == "VII":
        return {"gross": "60–120% NAV", "net": "0–50% NAV", "hedge": "10–30% notional", "liquidity": "15–30% NAV",
                "notes": "System repairing; re-risk incrementally with confirmation."}
    return {"gross": "70–120% NAV", "net": "0–40% NAV", "hedge": "15–35% notional", "liquidity": "12–25% NAV",
            "notes": "Mixed regime; default conservative."}

def render_policy_block(state: dict):
    pol = exposure_policy_for_state(state["code"])
    st.markdown("## RISK STATE (POLICY LAYER)")
    st.markdown(
        f"<div class='terminal-line'><span class='badge'>STATE {state['code']}</span> {state['name']} | {state['name_zh']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='terminal-line'>KEY DRIVERS: {state['key_drivers']}</div>", unsafe_allow_html=True)
    if state["amplifiers"]:
        for a in state["amplifiers"]:
            st.markdown(f"<div class='terminal-line'>AMPLIFIER: {a}</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <table class="dataframe" style="width:100%; border-collapse:collapse;">
            <tr>
                <th>GROSS EXPOSURE</th><th>NET EXPOSURE</th><th>HEDGE INTENSITY</th><th>LIQUIDITY BUFFER</th>
            </tr>
            <tr>
                <td>{pol['gross']}</td><td>{pol['net']}</td><td>{pol['hedge']}</td><td>{pol['liquidity']}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<div class='terminal-line'>NOTES: {pol['notes']}</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

def render_policy_appendix_expander():
    with st.expander("POLICY APPENDIX (EN + 繁體中文)", expanded=False):
        st.markdown("<div class='terminal-line'>This appendix constrains behavior; it does not mandate trades. Risk states require cross-module confirmation and persistence.</div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>本附錄用於約束行為而非下達交易指令；風險狀態需跨模組確認與持續性。</div>", unsafe_allow_html=True)
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>STATE I / 結構穩定期：允許風險運作，避免過度避險。</div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>STATE II / 內部脆弱：提高選擇性與彈性，避免把指數平靜當作安全證據。</div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>STATE III / 信用惡化初期：主動降敏感度並提高避險與流動性。</div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>STATE IV / 資金壓力浮現：優先韌性與流動性，偏好凸性避險。</div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>STATE V / 系統性風險：資本保護優先，總曝險下限、避險高強度、現金最大化。</div>", unsafe_allow_html=True)
        st.markdown("<div class='terminal-line'>STATE VII / 結構修復期：逐步加回風險，避險緩慢撤除。</div>", unsafe_allow_html=True)



# =========================
# App
# =========================

# =========================
# Chart Renderers (for Tabs + Custom Dashboard)
# =========================
def _section_divider(title: str):
    st.markdown(f"<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='terminal-line'><b>{title}</b></div>", unsafe_allow_html=True)

def render_chart_hyg_tlt_v1(v1_df: pd.DataFrame, z_window: int):
    df = pd.DataFrame({"HYG_TLT": v1_df["HYG_TLT"]})
    df["Z"] = zscore(df["HYG_TLT"], int(z_window))
    fig = plot_timeseries(df, "CREDIT STRESS (V1): HYG/TLT", ["HYG_TLT"], ["Z"], "Ratio", "Z")
    st_plotly(fig)

def render_chart_usd_jpy_v1(v1_df: pd.DataFrame, z_window: int):
    df = pd.DataFrame({"USD_JPY_PROXY": v1_df["USD_JPY_PROXY"]})
    df["Z"] = zscore(df["USD_JPY_PROXY"], int(z_window))
    fig = plot_timeseries(df, "CURRENCY STRESS (V1): USD/JPY Proxy (UUP/FXY)", ["USD_JPY_PROXY"], ["Z"], "Proxy Ratio", "Z")
    st_plotly(fig)

def render_chart_rsp_spy_v1(v1_df: pd.DataFrame, z_window: int):
    df = pd.DataFrame({"RSP_SPY": v1_df["RSP_SPY"]})
    df["Z"] = zscore(df["RSP_SPY"], int(z_window))
    fig = plot_timeseries(df, "MARKET BREADTH (V1): RSP/SPY", ["RSP_SPY"], ["Z"], "Ratio", "Z")
    st_plotly(fig)

def render_chart_xlu_xlk_v1(v1_df: pd.DataFrame, z_window: int):
    df = pd.DataFrame({"XLU_XLK": v1_df["XLU_XLK"]})
    df["Z"] = zscore(df["XLU_XLK"], int(z_window))
    fig = plot_timeseries(df, "DEFENSIVE ROTATION (V1): XLU/XLK", ["XLU_XLK"], ["Z"], "Ratio", "Z")
    st_plotly(fig)

def render_chart_hy_oas(modules: dict):
    df = modules["HY_OAS"].copy()
    # Module column is named HY_OAS (not VALUE)
    fig = plot_timeseries(df, "HIGH-YIELD OAS (FRED): Credit Stress", ["HY_OAS"], ["Z"], "bps", "Z")
    st_plotly(fig)

def render_chart_hyg_spy(modules: dict):
    df = modules["HYG_SPY"].copy()
    fig = plot_timeseries(df, "CREDIT vs EQUITY: HYG/SPY", ["HYG_SPY"], ["Z"], "Ratio", "Z")
    st_plotly(fig)

def render_chart_sofr_tbill(modules: dict):
    df = modules["SOFR_TBILL"].copy()
    fig = plot_timeseries(df, "FUNDING STRESS: SOFR - 3M T-Bill", ["SOFR_MINUS_TB3M"], ["Z"], "Rate Spread", "Z")
    st_plotly(fig)

def render_chart_breakeven_10y(modules: dict):
    df = modules["BREAKEVEN_10Y"].copy()
    # FRED series T10YIE is stored under column 'T10YIE' in build_modules()
    level_col = "T10YIE" if "T10YIE" in df.columns else next((c for c in df.columns if c not in ["Z", "SIGNAL"]), None)
    if level_col is None:
        st.warning("Breakeven module has no level column to plot.")
        return
    fig = plot_timeseries(df, "10Y BREAKEVEN INFLATION", [level_col], ["Z"], "Rate", "Z")
    st_plotly(fig)

def render_chart_cot_sp500(modules: dict):
    df = modules["COT_SP500"].copy()
    fig = plot_timeseries(df, "COT: S&P 500 Net Speculative Positioning (Weekly)", ["COT_NET_NONCOMM"], ["Z"], "Contracts (Net)", "Z")
    st_plotly(fig)

def build_chart_registry(v1_df: pd.DataFrame, modules: dict, z_window: int):
    """Returns a mapping of display-name -> callable that renders the chart.
    This registry powers both the standard tabs and the Custom Dashboard tab."""
    return {
        "CREDIT STRESS HYG/TLT (V1)": lambda: render_chart_hyg_tlt_v1(v1_df, z_window),
        "CURRENCY STRESS USD/JPY (V1)": lambda: render_chart_usd_jpy_v1(v1_df, z_window),
        "MARKET BREADTH RSP/SPY (V1)": lambda: render_chart_rsp_spy_v1(v1_df, z_window),
        "DEFENSIVE ROTATION XLU/XLK (V1)": lambda: render_chart_xlu_xlk_v1(v1_df, z_window),
        "HY OAS (FRED)": lambda: render_chart_hy_oas(modules),
        "HYG/SPY": lambda: render_chart_hyg_spy(modules),
        "SOFR - 3M T-Bill": lambda: render_chart_sofr_tbill(modules),
        "10Y Breakeven": lambda: render_chart_breakeven_10y(modules),
        "COT S&P 500": lambda: render_chart_cot_sp500(modules),
    }

def render_custom_dashboard(chart_registry: dict):
    """Custom Dashboard: user-selectable vertical stack of charts with clean alignment."""
    st.markdown("## CUSTOM DASHBOARD")
    st.markdown(
        "<div class='terminal-line'>Select the charts you want on this page. Charts render top-to-bottom with consistent sizing and interactive zoom.</div>",
        unsafe_allow_html=True
    )

    all_charts = list(chart_registry.keys())

    presets = {
        "None (manual selection)": [],
        "Equities Desk": ["MARKET BREADTH RSP/SPY (V1)", "DEFENSIVE ROTATION XLU/XLK (V1)", "HYG/SPY", "HY OAS (FRED)"],
        "Credit Desk": ["HY OAS (FRED)", "HYG/SPY", "SOFR - 3M T-Bill", "COT S&P 500"],
        "Macro / Rates Desk": ["SOFR - 3M T-Bill", "CURRENCY STRESS USD/JPY (V1)", "10Y Breakeven", "HY OAS (FRED)"],
        "Risk (Full Stack)": all_charts,
    }

    preset_name = st.selectbox("Preset view", list(presets.keys()), index=0)
    default_sel = presets.get(preset_name, [])

    selection = st.multiselect("Charts to display (top-to-bottom)", all_charts, default=default_sel)

    if not selection:
        st.info("Select at least one chart to render the Custom Dashboard.")
        return

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Render charts in the exact order selected to avoid overlap and preserve visual flow.
    for name in selection:
        with st.container():
            st.markdown(f"<div class='terminal-line'><b>{name}</b></div>", unsafe_allow_html=True)
            chart_registry[name]()
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

def main():
    st.markdown("# JAMS CAPITAL RISK MANAGEMENT TERMINAL")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.2, 1.2, 1.4])
    with c1:
        if st.button("REFRESH DATA"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        auto_refresh = st.checkbox("AUTO REFRESH", value=False)
    with c3:
        lookback = st.selectbox("DEFAULT LOOKBACK", ["6M", "1Y", "3Y", "5Y", "10Y", "MAX"], index=2)
    with c4:
        z_window = st.selectbox("Z-SCORE WINDOW", [63, 126, 252, 504], index=2, help="Trading days. 252 = ~1Y.")
    with c5:
        st.write(f"LAST UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    today = datetime.utcnow().date()
    if lookback == "6M":
        default_start = today - timedelta(days=183)
    elif lookback == "1Y":
        default_start = today - timedelta(days=365)
    elif lookback == "3Y":
        default_start = today - timedelta(days=365*3)
    elif lookback == "5Y":
        default_start = today - timedelta(days=365*5)
    elif lookback == "10Y":
        default_start = today - timedelta(days=365*10)
    else:
        default_start = date(1990, 1, 1)

    d1, d2, d3 = st.columns([1.2, 1.2, 2.6])
    with d1:
        start_date = st.date_input("START DATE", value=default_start)
    with d2:
        end_date = st.date_input("END DATE", value=today)
    with d3:
        st.markdown(
            "<div class='terminal-line'>CHARTS SUPPORT PAN/ZOOM + RANGE SLIDER + QUICK RANGE BUTTONS (1M/3M/6M/1Y/3Y/ALL).</div>",
            unsafe_allow_html=True
        )

    start_date = _to_date(start_date)
    end_date = _to_date(end_date)
    if end_date <= start_date:
        st.error("END DATE must be after START DATE.")
        st.stop()

    # Build core modules
    modules = build_modules(start_date, end_date, int(z_window))

    st.markdown("## DATA INPUTS")
    st.markdown("<div class='terminal-line'>All modules are sourced programmatically (FRED, Yahoo Finance, CFTC). No uploads required.</div>", unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Snapshot table
    as_of = pd.to_datetime(end_date)
    snap_df = composite_snapshot(modules, as_of)

    left, right = st.columns([1.15, 2.85])
    with left:
        st.markdown("## SIGNAL SNAPSHOT")
        if snap_df.empty:
            st.markdown("<div class='terminal-line'>DATA INSUFFICIENT.</div>", unsafe_allow_html=True)
        else:
            # themed HTML table
            header = "<tr><th>MODULE</th><th>REGIME</th><th>Z</th><th>SIGNAL(0-100)</th></tr>"
            body = ""
            for _, r in snap_df[["MODULE", "REGIME", "Z", "SIGNAL(0-100)"]].iterrows():
                body += f"<tr><td>{r['MODULE']}</td><td>{r['REGIME']}</td><td>{r['Z']}</td><td>{r['SIGNAL(0-100)']}</td></tr>"
            st.markdown(f"<table class='dataframe' style='width:100%; border-collapse:collapse;'>{header}{body}</table>", unsafe_allow_html=True)

    with right:
        st.markdown("## EXECUTIVE SUMMARY (SIGNALIZED)")
        if snap_df.empty:
            msg = "DATA INSUFFICIENT: Unable to compute signals for the selected range."
        else:
            worst = snap_df.iloc[0]["MODULE"]
            comp = float(np.nanmean(snap_df["SIGNAL(0-100)"]))
            comp10 = float(np.clip(round(comp / 10.0, 1), 0.0, 10.0))
            msg = (
                f"RISK STATE: composite signal is {comp10:.1f}/10 (avg across modules). "
                f"Highest-pressure module: {worst}. Validate directionality in the tabs below."
            )
        st.markdown(f"<div class='terminal-line'>{msg}</div>", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    
    # -------------------------

    
    # -------------------------
    # V1 proxies (required for Policy Layer)
    # -------------------------
    try:
        v1_px = fetch_yf_prices(["HYG", "TLT", "UUP", "FXY", "RSP", "SPY", "XLU", "XLK"], start_date, end_date).ffill()
        v1_hyg_tlt = (v1_px["HYG"] / v1_px["TLT"]).rename("HYG_TLT")
        v1_usd_jpy = (v1_px["UUP"] / v1_px["FXY"]).rename("USD_JPY_PROXY")
        v1_rsp_spy = (v1_px["RSP"] / v1_px["SPY"]).rename("RSP_SPY")
        v1_def = (v1_px["XLU"] / v1_px["XLK"]).rename("XLU_XLK")
        v1_df = pd.concat([v1_hyg_tlt, v1_usd_jpy, v1_rsp_spy, v1_def], axis=1).dropna()
    except Exception as e:
        # Keep the app running even if Yahoo fetch is temporarily unavailable
        v1_df = pd.DataFrame()
        st.warning(f"V1 proxy fetch failed (policy layer will degrade gracefully): {e}")

# -------------------------
    # Policy Layer: Risk State + Suggested Posture (auto-labeled)
    # -------------------------
    v1_sig = compute_v1_signals(v1_df, int(z_window)) if not v1_df.empty else pd.DataFrame()
    state = classify_risk_state(modules, v1_sig, pd.to_datetime(end_date))

    # Risk State (Policy Layer) - Render
    # -------------------------
    render_policy_block(state)
    render_policy_appendix_expander()

# -------------------------
    # V1 proxy tabs (excluding VIX and forward risk score, per your request)
    # -------------------------
    v1_px = fetch_yf_prices(["HYG", "TLT", "UUP", "FXY", "RSP", "SPY", "XLU", "XLK"], start_date, end_date).ffill()
    v1_hyg_tlt = (v1_px["HYG"] / v1_px["TLT"]).rename("HYG_TLT")
    v1_usd_jpy = (v1_px["UUP"] / v1_px["FXY"]).rename("USD_JPY_PROXY")
    v1_rsp_spy = (v1_px["RSP"] / v1_px["SPY"]).rename("RSP_SPY")
    v1_def = (v1_px["XLU"] / v1_px["XLK"]).rename("XLU_XLK")

    v1_df = pd.concat([v1_hyg_tlt, v1_usd_jpy, v1_rsp_spy, v1_def], axis=1).dropna()

    tabs = st.tabs([
        "CUSTOM DASHBOARD",
        "CREDIT STRESS HYG/TLT (V1)",
        "CURRENCY STRESS USD/JPY (V1)",
        "MARKET BREADTH RSP/SPY (V1)",
        "DEFENSIVE ROTATION XLU/XLK (V1)",
        "HY OAS (FRED)",
        "HYG/SPY",
        "SOFR - 3M T-Bill",
        "10Y Breakeven",
        "COT S&P 500"
    ])

    
    # Build chart registry for standard tabs and the Custom Dashboard
    chart_registry = build_chart_registry(v1_df, modules, int(z_window))

    with tabs[0]:
        render_custom_dashboard(chart_registry)

    with tabs[1]:
        chart_registry["CREDIT STRESS HYG/TLT (V1)"]()

    with tabs[2]:
        chart_registry["CURRENCY STRESS USD/JPY (V1)"]()

    with tabs[3]:
        chart_registry["MARKET BREADTH RSP/SPY (V1)"]()

    with tabs[4]:
        chart_registry["DEFENSIVE ROTATION XLU/XLK (V1)"]()

    with tabs[5]:
        chart_registry["HY OAS (FRED)"]()

    with tabs[6]:
        chart_registry["HYG/SPY"]()

    with tabs[7]:
        chart_registry["SOFR - 3M T-Bill"]()

    with tabs[8]:
        chart_registry["10Y Breakeven"]()

    with tabs[9]:
        chart_registry["COT S&P 500"]()
        st.markdown("<div class='terminal-line'>COT is weekly (reported by CFTC). For visualization and scoring, values are forward-filled to daily frequency.</div>", unsafe_allow_html=True)

    # Auto refresh (optional). Keep inside main() so it has access to UI state.
    if auto_refresh:
        st.caption("Auto refresh enabled (refreshes every 60 seconds).")
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()
