import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] { background: #f8fafc; border-radius: 8px; padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Financial Modeling Prep API Key",
        type="password",
        placeholder="Paste your free API key here",
        help="Get a free key at financialmodelingprep.com/developer/docs",
    )
    st.caption(
        "Get a **free API key** at [financialmodelingprep.com](https://financialmodelingprep.com/developer/docs). "
        "The free tier allows 250 requests/day — plenty for personal use."
    )
    st.divider()
    st.caption("Risk-free rate is sourced from the US 10Y Treasury yield via FMP.")

# ── Data layer ─────────────────────────────────────────────────────────────────
BASE = "https://financialmodelingprep.com/api"

def fmp_get(endpoint: str, api_key: str, params: dict = None) -> list | dict:
    """Make a single FMP API call and return parsed JSON."""
    p = {"apikey": api_key, **(params or {})}
    r = requests.get(f"{BASE}{endpoint}", params=p, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        if "message" in data and "limit" in data["message"].lower():
            raise ValueError("API limit reached. Upgrade your FMP plan or try again tomorrow.")
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str, api_key: str) -> dict:
    profile_data = fmp_get(f"/v3/profile/{symbol}", api_key)
    if not profile_data:
        raise ValueError(f"No data found for '{symbol}'. Please check the ticker symbol.")
    profile = profile_data[0]

    income   = fmp_get(f"/v3/income-statement/{symbol}",       api_key, {"limit": 4})
    balance  = fmp_get(f"/v3/balance-sheet-statement/{symbol}", api_key, {"limit": 4})
    cashflow = fmp_get(f"/v3/cash-flow-statement/{symbol}",    api_key, {"limit": 4})

    try:
        estimates = fmp_get(f"/v3/analyst-estimates/{symbol}", api_key, {"limit": 3})
    except Exception:
        estimates = []

    # 10Y Treasury yield for risk-free rate
    try:
        treasury = fmp_get("/v4/treasury", api_key, {
            "from": "2026-01-01", "to": "2026-03-15"
        })
        rf = float(sorted(treasury, key=lambda x: x["date"])[-1].get("year10", 4.3)) / 100
    except Exception:
        rf = 0.043  # fallback ~4.3%

    return dict(profile=profile, income=income, balance=balance,
                cashflow=cashflow, estimates=estimates, rf=rf)


# ── Calculation layer ──────────────────────────────────────────────────────────
def _f(records: list, key: str, index: int = 0, default: float = 0.0) -> float:
    """Safely extract a float from a FMP records list."""
    if not records or len(records) <= index:
        return default
    val = records[index].get(key)
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def get_base_fcf(data: dict) -> float:
    cf = data["cashflow"]
    if not cf:
        return 0.0
    fcf = cf[0].get("freeCashFlow")
    if fcf is not None:
        return float(fcf)
    # Fallback: operatingCashFlow + capitalExpenditure (capex is negative in FMP)
    ocf   = _f(cf, "operatingCashFlow")
    capex = _f(cf, "capitalExpenditure")
    return ocf + capex


def get_growth_rate(data: dict) -> tuple:
    estimates = data.get("estimates", [])
    income    = data.get("income",    [])
    cashflow  = data.get("cashflow",  [])

    # 1. Analyst revenue estimate (next FY vs current FY)
    if estimates and income:
        curr_rev = _f(income, "revenue", 0)
        next_rev = _f(estimates, "revenueAvg", 0)
        if curr_rev > 0 and next_rev > 0:
            g = (next_rev - curr_rev) / curr_rev
            return float(np.clip(g, -0.10, 0.40)), "Analyst Revenue Estimate (1Y)"

    # 2. Historical revenue CAGR
    if len(income) >= 2:
        r0 = _f(income, "revenue", 0)
        rn = _f(income, "revenue", len(income) - 1)
        if r0 > 0 and rn > 0:
            cagr = (r0 / rn) ** (1 / (len(income) - 1)) - 1
            return float(np.clip(cagr, -0.10, 0.35)), "Historical Revenue CAGR"

    # 3. Historical FCF CAGR
    if len(cashflow) >= 2:
        f0 = get_base_fcf({"cashflow": [cashflow[0]]})
        fn = get_base_fcf({"cashflow": [cashflow[-1]]})
        if f0 > 0 and fn > 0:
            cagr = (f0 / fn) ** (1 / (len(cashflow) - 1)) - 1
            return float(np.clip(cagr, -0.10, 0.35)), "Historical FCF CAGR"

    return 0.05, "Default (5%)"


def calc_wacc(data: dict) -> dict:
    profile  = data["profile"]
    income   = data["income"]
    balance  = data["balance"]
    rf       = data["rf"]
    erp      = 0.055  # Damodaran equity risk premium

    beta = float(profile.get("beta") or 1.0)
    beta = float(np.clip(beta, 0.1, 3.0))
    ke   = rf + beta * erp

    total_debt  = _f(balance, "totalDebt")
    interest_exp = abs(_f(income, "interestExpense"))
    kd = (interest_exp / total_debt) if total_debt > 0 else 0.05
    kd = float(np.clip(kd, 0.02, 0.15))

    pretax   = _f(income, "incomeBeforeTax")
    tax_prov = _f(income, "incomeTaxExpense")
    tax_rate = (tax_prov / pretax) if pretax > 0 else 0.21
    tax_rate = float(np.clip(tax_rate, 0.0, 0.40))

    mkt_cap  = float(profile.get("mktCap") or 0)
    total_ev = mkt_cap + total_debt
    we = mkt_cap  / total_ev if total_ev > 0 else 1.0
    wd = total_debt / total_ev if total_ev > 0 else 0.0

    wacc = we * ke + wd * kd * (1 - tax_rate)

    return dict(wacc=wacc, ke=ke, kd=kd, beta=beta, rf=rf,
                tax_rate=tax_rate, we=we, wd=wd, total_debt=total_debt)


def run_dcf(base_fcf: float, g: float, g_term: float, wacc: float, data: dict) -> dict:
    if wacc <= g_term:
        g_term = wacc - 0.005

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

    tv    = fcfs[-1] * (1 + g_term) / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** 10
    ev    = sum(pv_fcfs) + pv_tv

    balance  = data["balance"]
    profile  = data["profile"]
    cash     = _f(balance, "cashAndShortTermInvestments") or _f(balance, "cashAndCashEquivalents")
    total_debt = data.get("wacc_details", {}).get("total_debt", 0.0)
    net_debt   = total_debt - cash
    equity_val = ev - net_debt
    shares     = float(profile.get("sharesOutstanding") or 1)
    iv         = equity_val / shares

    return dict(fcfs=fcfs, pv_fcfs=pv_fcfs, growth_rates=growth_rates,
                pv_tv=pv_tv, total_pv_fcfs=sum(pv_fcfs),
                ev=ev, equity_val=equity_val, iv=iv,
                net_debt=net_debt, cash=cash)


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

# API key gate
if not api_key:
    st.info(
        "👈  **Enter your free FMP API key in the sidebar to get started.**  \n"
        "Get one in 30 seconds at [financialmodelingprep.com/developer/docs](https://financialmodelingprep.com/developer/docs)"
    )
    st.stop()

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
            data = fetch_data(sym, api_key)
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

        profile = data["profile"]
        price   = float(profile.get("price") or 0)
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
    data     = ss["data"]
    profile  = data["profile"]
    wacc_d   = ss["wacc_d"]
    dcf      = ss["dcf"]
    price    = ss["price"]
    verd     = ss["verd"]
    upside   = ss["upside"]
    g        = ss["g"]
    g_source = ss["g_source"]
    g_term   = ss["g_term"]
    base_fcf = ss["base_fcf"]

    st.divider()

    # Company header
    mkt_cap = float(profile.get("mktCap") or 0)
    st.subheader(f"{profile.get('companyName', profile.get('symbol', '—'))}  ({profile.get('symbol', '—')})")
    st.caption(
        f"{profile.get('sector', '—')} · {profile.get('industry', '—')} · "
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
    iv_display  = max(dcf["iv"], 0)
    cheap       = dcf["iv"] > price
    bar_colors  = ["#94a3b8", "#22c55e"] if cheap else ["#94a3b8", "#ef4444"]

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
        w1.metric("Risk-Free Rate",        f"{wacc_d['rf']:.2%}")
        w2.metric("Cost of Equity (CAPM)", f"{wacc_d['ke']:.2%}")
        w3.metric("After-Tax Cost of Debt",f"{wacc_d['kd'] * (1 - wacc_d['tax_rate']):.2%}")
        w4.metric("Effective Tax Rate",    f"{wacc_d['tax_rate']:.1%}")
        w5, w6 = st.columns(2)
        w5.metric("Equity Weight", f"{wacc_d['we']:.1%}")
        w6.metric("Debt Weight",   f"{wacc_d['wd']:.1%}")

    # DCF bridge
    with st.expander("DCF Bridge"):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Enterprise Value", f"${dcf['ev'] / 1e9:.1f}B")
        b2.metric("Net Debt",         f"${dcf['net_debt'] / 1e9:.1f}B")
        b3.metric("Equity Value",     f"${dcf['equity_val'] / 1e9:.1f}B")
        shares_out = float(profile.get("sharesOutstanding") or 0)
        b4.metric("Shares Outstanding", f"{shares_out / 1e9:.2f}B")

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
        r2.metric("Market Price",      f"${price:.2f}")
        r3.metric("Upside / Downside", f"{upside2:+.1f}%")
        r4.metric("Verdict",           verd2)

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
