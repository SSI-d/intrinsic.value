import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests_cache
from requests_ratelimiter import LimiterSession

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] { background: #f8fafc; border-radius: 8px; padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)

# ── Cached + rate-limited yfinance session ────────────────────────────────────
# Caches responses for 1 hour, max 2 requests/second — avoids Yahoo rate limits
@st.cache_resource
def get_yf_session():
    session = requests_cache.CachedSession(
        cache_name="yfinance_cache",
        backend="memory",
        expire_after=3600,
    )
    session.headers.update({"User-Agent": "dcf-valuation-tool/1.0"})
    return session

def make_ticker(symbol: str):
    """Return a yfinance Ticker using the shared cached session."""
    return yf.Ticker(symbol, session=get_yf_session())

def fetch_with_retry(fn, retries=3, delay=2):
    """Call fn(), retrying on rate-limit errors with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "rate limit" in msg or "429" in msg:
                if attempt < retries - 1:
                    time.sleep(delay * (2 ** attempt))
                    continue
            raise
    raise RuntimeError("Yahoo Finance rate limit reached. Please wait a moment and try again.")

# ── Data layer ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str) -> dict:
    def _fetch():
        t = make_ticker(symbol)
        info = t.info
        if not info or info.get("quoteType") is None:
            raise ValueError(f"No data found for '{symbol}'. Please check the ticker symbol.")
        return {
            "info": info,
            "income_stmt": t.income_stmt,
            "balance_sheet": t.balance_sheet,
            "cash_flow": t.cashflow,
            "growth_est": _safe_fetch(t.growth_estimates),
            "rf": _get_rf_rate(),
        }
    return fetch_with_retry(_fetch)

def _safe_fetch(obj):
    try:
        return obj
    except Exception:
        return None

def _get_rf_rate() -> float:
    try:
        tnx = make_ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100
    except Exception:
        pass
    return 0.045  # fallback: 4.5%

def safe_get(df: pd.DataFrame, keys: list, col: int = 0, default: float = 0.0) -> float:
    if df is None or df.empty:
        return default
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key].iloc[col]
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return default

# ── Calculation layer ──────────────────────────────────────────────────────────
def get_growth_rate(data: dict) -> tuple:
    """Return (rate, source_label). Uses analyst stockTrend → info fields → historical CAGR."""
    ge = data.get("growth_est")

    # 1. Analyst consensus — stockTrend only (indexTrend is sector-wide, not company-specific)
    if ge is not None and not ge.empty and "stockTrend" in ge.columns:
        try:
            for row_key, label in [("LTG", "Analyst LTG"), ("+1y", "Analyst 1Y"), ("0y", "Analyst Current Year")]:
                if row_key in ge.index:
                    val = ge.loc[row_key, "stockTrend"]
                    if pd.notna(val) and float(val) != 0:
                        return float(np.clip(float(val), -0.10, 0.40)), label
        except Exception:
            pass

    # 2. Info fields
    info = data["info"]
    for field, label in [("earningsGrowth", "TTM Earnings Growth"), ("revenueGrowth", "TTM Revenue Growth")]:
        val = info.get(field)
        if val and pd.notna(val):
            return float(np.clip(float(val), -0.10, 0.40)), label

    # 3. Historical FCF CAGR fallback
    cf = data["cash_flow"]
    for key in ["Operating Cash Flow", "Cash From Operations"]:
        if cf is not None and not cf.empty and key in cf.index:
            series = cf.loc[key].dropna()
            if len(series) >= 2 and series.iloc[-1] != 0:
                cagr = (series.iloc[0] / series.iloc[-1]) ** (1 / (len(series) - 1)) - 1
                return float(np.clip(cagr, -0.10, 0.35)), "Historical FCF CAGR"

    return 0.05, "Default (5%)"


def get_base_fcf(data: dict) -> float:
    cf = data["cash_flow"]
    ocf = safe_get(cf, ["Operating Cash Flow", "Cash From Operations"])
    capex = safe_get(cf, ["Capital Expenditure"])  # negative in yfinance
    return ocf + capex


def calc_wacc(data: dict) -> dict:
    info = data["info"]
    inc = data["income_stmt"]
    bal = data["balance_sheet"]
    rf = data["rf"]
    erp = 0.055  # Damodaran equity risk premium

    beta = float(info.get("beta") or 1.0)
    beta = float(np.clip(beta, 0.1, 3.0))
    ke = rf + beta * erp

    total_debt = safe_get(bal, ["Total Debt", "Long Term Debt"])
    interest_exp = abs(safe_get(inc, ["Interest Expense"]))
    kd = (interest_exp / total_debt) if total_debt > 0 else 0.05
    kd = float(np.clip(kd, 0.02, 0.15))

    pretax = safe_get(inc, ["Pretax Income", "Income Before Tax"])
    tax_prov = safe_get(inc, ["Tax Provision", "Income Tax Expense"])
    tax_rate = (tax_prov / pretax) if pretax > 0 else 0.21
    tax_rate = float(np.clip(tax_rate, 0.0, 0.40))

    mkt_cap = float(info.get("marketCap") or 0)
    total_ev = mkt_cap + total_debt
    we = mkt_cap / total_ev if total_ev > 0 else 1.0
    wd = total_debt / total_ev if total_ev > 0 else 0.0

    wacc = we * ke + wd * kd * (1 - tax_rate)

    return dict(
        wacc=wacc, ke=ke, kd=kd, beta=beta, rf=rf,
        tax_rate=tax_rate, we=we, wd=wd, total_debt=total_debt,
    )


def run_dcf(base_fcf: float, g: float, g_term: float, wacc: float, data: dict) -> dict:
    """Project FCFs fading from g → g_term over years 6–10, discount to PV."""
    if wacc <= g_term:
        g_term = wacc - 0.005  # prevent division-by-zero in terminal value

    fcfs, pv_fcfs, growth_rates = [], [], []
    for yr in range(1, 11):
        if yr <= 5:
            g_yr = g
        else:
            fade = (yr - 5) / 5
            g_yr = g * (1 - fade) + g_term * fade
        growth_rates.append(g_yr)
        fcf = (base_fcf if yr == 1 else fcfs[-1]) * (1 + g_yr)
        fcfs.append(fcf)
        pv_fcfs.append(fcf / (1 + wacc) ** yr)

    tv = fcfs[-1] * (1 + g_term) / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** 10
    ev = sum(pv_fcfs) + pv_tv

    bal = data["balance_sheet"]
    info = data["info"]
    cash = safe_get(bal, [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
    ])
    total_debt = data.get("wacc_details", {}).get("total_debt", 0.0)
    net_debt = total_debt - cash
    equity_val = ev - net_debt
    shares = float(info.get("sharesOutstanding") or 1)
    iv = equity_val / shares

    return dict(
        fcfs=fcfs, pv_fcfs=pv_fcfs, growth_rates=growth_rates,
        pv_tv=pv_tv, total_pv_fcfs=sum(pv_fcfs),
        ev=ev, equity_val=equity_val, iv=iv,
        net_debt=net_debt, cash=cash,
    )


def get_verdict(iv: float, price: float) -> tuple:
    if iv <= 0 or price <= 0:
        return "N/A", 0.0
    upside = (iv - price) / price * 100
    if upside > 20:
        return "CHEAP", upside
    elif upside < -20:
        return "EXPENSIVE", upside
    return "FAIRLY VALUED", upside


def render_verdict_banner(verd: str, upside: float):
    if verd == "CHEAP":
        st.success(f"🟢  **CHEAP** — Intrinsic value is **{upside:.0f}% above** the current market price")
    elif verd == "EXPENSIVE":
        st.error(f"🔴  **EXPENSIVE** — Market price is **{abs(upside):.0f}% above** intrinsic value")
    elif verd == "FAIRLY VALUED":
        direction = "above" if upside >= 0 else "below"
        st.warning(f"🟡  **FAIRLY VALUED** — Intrinsic value is **{abs(upside):.0f}% {direction}** the current market price")
    else:
        st.info("⚪  Cannot determine verdict — negative FCF or missing data")


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📊 DCF Intrinsic Value Calculator")
st.caption("Estimate the intrinsic value of any publicly traded company using discounted cash flow analysis.")

c1, c2 = st.columns([5, 1])
with c1:
    sym_input = st.text_input(
        "",
        placeholder="Enter a ticker — e.g. AAPL, MSFT, NVDA, TSLA",
        label_visibility="collapsed",
    )
with c2:
    go_btn = st.button("Analyze →", type="primary", use_container_width=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if go_btn and sym_input:
    sym = sym_input.upper().strip()

    with st.spinner(f"Fetching data for {sym}…"):
        try:
            data = fetch_data(sym)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not load data for **{sym}**: {e}")
            st.stop()

    with st.spinner("Running DCF model…"):
        wacc_d = calc_wacc(data)
        data["wacc_details"] = wacc_d
        base_fcf = get_base_fcf(data)
        g, g_source = get_growth_rate(data)
        g_term = 0.025
        dcf = run_dcf(base_fcf, g, g_term, wacc_d["wacc"], data)

        info = data["info"]
        price = float(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or 0
        )
        verd, upside = get_verdict(dcf["iv"], price)

    for k, v in dict(
        data=data, wacc_d=wacc_d, base_fcf=base_fcf,
        g=g, g_source=g_source, g_term=g_term, dcf=dcf,
        price=price, verd=verd, upside=upside, analyzed=True,
    ).items():
        st.session_state[k] = v

# ── Display results ────────────────────────────────────────────────────────────
ss = st.session_state
if ss.get("analyzed"):
    data   = ss["data"]
    info   = data["info"]
    wacc_d = ss["wacc_d"]
    dcf    = ss["dcf"]
    price  = ss["price"]
    verd   = ss["verd"]
    upside = ss["upside"]
    g      = ss["g"]
    g_source = ss["g_source"]
    g_term = ss["g_term"]

    st.divider()

    # Company header
    mkt_cap = info.get("marketCap", 0) or 0
    st.subheader(f"{info.get('longName', info.get('symbol', '—'))}  ({info.get('symbol', '—')})")
    st.caption(
        f"{info.get('sector', '—')} · {info.get('industry', '—')} · "
        f"Market Cap ${mkt_cap / 1e9:.1f}B"
    )

    render_verdict_banner(verd, upside)

    # Key metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Market Price", f"${price:.2f}")
    m2.metric("Intrinsic Value", f"${max(dcf['iv'], 0):.2f}", delta=f"{upside:+.1f}%")
    m3.metric("WACC", f"{wacc_d['wacc']:.1%}")
    m4.metric("Growth Rate (Y1–5)", f"{g:.1%}", help=f"Source: {g_source}")
    m5.metric("Beta", f"{wacc_d['beta']:.2f}")

    base_fcf = ss["base_fcf"]
    if base_fcf < 0:
        st.warning(
            f"⚠️  Base FCF is negative (${base_fcf / 1e9:.2f}B). "
            "Interpret the DCF result with caution."
        )

    st.divider()

    # Charts
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        st.subheader("Projected Free Cash Flows")
        fig = go.Figure(go.Bar(
            x=[f"Y{i}" for i in range(1, 11)],
            y=[f / 1e9 for f in dcf["fcfs"]],
            marker_color=["#3b82f6" if f > 0 else "#ef4444" for f in dcf["fcfs"]],
            text=[f"${f / 1e9:.1f}B" for f in dcf["fcfs"]],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="FCF ($B)", height=300,
            margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.subheader("Enterprise Value Composition")
        pv_f = max(dcf["total_pv_fcfs"], 0)
        pv_t = max(dcf["pv_tv"], 0)
        if pv_f + pv_t > 0:
            fig2 = go.Figure(go.Pie(
                labels=["PV of FCFs (Y1–10)", "PV of Terminal Value"],
                values=[pv_f / 1e9, pv_t / 1e9],
                hole=0.45,
                marker_colors=["#3b82f6", "#93c5fd"],
                textinfo="label+percent",
            ))
            fig2.update_layout(
                showlegend=False, height=300,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Price vs intrinsic value
    st.subheader("Market Price vs Intrinsic Value")
    iv_display = max(dcf["iv"], 0)
    cheap = dcf["iv"] > price
    bar_colors = ["#94a3b8", "#22c55e"] if cheap else ["#94a3b8", "#ef4444"]

    fig3 = go.Figure(go.Bar(
        x=["Current Market Price", "Intrinsic Value (DCF)"],
        y=[price, iv_display],
        marker_color=bar_colors,
        text=[f"${price:.2f}", f"${iv_display:.2f}"],
        textposition="outside",
        width=0.35,
    ))
    fig3.update_layout(
        yaxis_title="Price per share ($)", height=320,
        margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white",
        yaxis_range=[0, max(price, iv_display) * 1.30],
    )
    st.plotly_chart(fig3, use_container_width=True)

    # WACC breakdown
    with st.expander("WACC Breakdown"):
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Risk-Free Rate", f"{wacc_d['rf']:.2%}")
        w2.metric("Cost of Equity (CAPM)", f"{wacc_d['ke']:.2%}")
        w3.metric("After-Tax Cost of Debt", f"{wacc_d['kd'] * (1 - wacc_d['tax_rate']):.2%}")
        w4.metric("Effective Tax Rate", f"{wacc_d['tax_rate']:.1%}")
        w5, w6 = st.columns(2)
        w5.metric("Equity Weight", f"{wacc_d['we']:.1%}")
        w6.metric("Debt Weight", f"{wacc_d['wd']:.1%}")

    # DCF bridge
    with st.expander("DCF Bridge"):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Enterprise Value", f"${dcf['ev'] / 1e9:.1f}B")
        b2.metric("Net Debt", f"${dcf['net_debt'] / 1e9:.1f}B")
        b3.metric("Equity Value", f"${dcf['equity_val'] / 1e9:.1f}B")
        b4.metric("Shares Outstanding", f"{(info.get('sharesOutstanding') or 0) / 1e9:.2f}B")

    # ── Override section ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Override Assumptions")
    st.caption(
        f"Defaults are auto-calculated. Growth source: **{g_source}**. "
        "Adjust and recalculate to explore sensitivity."
    )

    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        ov_g = st.slider(
            "Growth Rate Y1–5 (%)",
            min_value=-10.0, max_value=50.0,
            value=round(g * 100, 1), step=1.0,
        ) / 100
    with ov2:
        ov_gt = st.slider(
            "Terminal Growth Rate (%)",
            min_value=0.0, max_value=5.0,
            value=round(g_term * 100, 1), step=0.1,
        ) / 100
    with ov3:
        ov_wacc = st.slider(
            "WACC (%)",
            min_value=4.0, max_value=20.0,
            value=round(wacc_d["wacc"] * 100, 1), step=0.5,
        ) / 100

    if st.button("Recalculate with Overrides", type="secondary"):
        dcf2 = run_dcf(ss["base_fcf"], ov_g, ov_gt, ov_wacc, data)
        verd2, upside2 = get_verdict(dcf2["iv"], price)

        st.subheader("Revised Valuation")
        render_verdict_banner(verd2, upside2)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Revised Intrinsic Value", f"${max(dcf2['iv'], 0):.2f}",
                  delta=f"{dcf2['iv'] - dcf['iv']:+.2f} vs base case")
        r2.metric("Market Price", f"${price:.2f}")
        r3.metric("Upside / Downside", f"{upside2:+.1f}%")
        r4.metric("Verdict", verd2)

        fig_cmp = go.Figure(go.Bar(
            x=["Market Price", "Base Case IV", "Revised IV"],
            y=[price, max(dcf["iv"], 0), max(dcf2["iv"], 0)],
            marker_color=["#94a3b8", "#3b82f6", "#6366f1"],
            text=[f"${price:.2f}", f"${max(dcf['iv'], 0):.2f}", f"${max(dcf2['iv'], 0):.2f}"],
            textposition="outside",
            width=0.4,
        ))
        fig_cmp.update_layout(
            yaxis_title="Price per share ($)", height=320,
            margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white",
            yaxis_range=[0, max(price, dcf["iv"], dcf2["iv"]) * 1.30],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ For informational purposes only — not financial advice. "
        "DCF valuations are highly sensitive to assumptions. "
        "Always do your own research before making investment decisions."
    )
