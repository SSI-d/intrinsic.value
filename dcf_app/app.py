import streamlit as st
import requests
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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Alpha Vantage API Key",
        type="password",
        placeholder="Paste your free API key here",
    )
    st.caption(
        "Get a **free API key** (500 calls/day, no credit card) at "
        "[alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)"
    )

# ── Data layer ─────────────────────────────────────────────────────────────────
AV = "https://www.alphavantage.co/query"

def av_get(function: str, symbol: str, api_key: str) -> dict:
    r = requests.get(AV, params={
        "function": function, "symbol": symbol, "apikey": api_key
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "Error Message" in data:
        raise ValueError(f"Ticker '{symbol}' not found.")
    if "Note" in data or "Information" in data:
        raise ValueError(
            "API rate limit reached. "
            "Get a free key at alphavantage.co/support/#api-key (500 calls/day, no card needed)."
        )
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str, api_key: str) -> dict:
    overview  = av_get("OVERVIEW",          symbol, api_key)
    quote_raw = av_get("GLOBAL_QUOTE",      symbol, api_key)
    income_r  = av_get("INCOME_STATEMENT",  symbol, api_key)
    balance_r = av_get("BALANCE_SHEET",     symbol, api_key)
    cf_r      = av_get("CASH_FLOW",         symbol, api_key)

    if not overview.get("Symbol"):
        raise ValueError(f"No data found for '{symbol}'. Check the ticker symbol.")

    return dict(
        overview  = overview,
        price     = float(quote_raw.get("Global Quote", {}).get("05. price", 0) or 0),
        income    = income_r.get("annualReports",  []),
        balance   = balance_r.get("annualReports", []),
        cashflow  = cf_r.get("annualReports",      []),
        rf        = 0.043,   # US 10Y Treasury approx — user can override WACC
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _f(records: list, key: str, idx: int = 0, default: float = 0.0) -> float:
    """Safely extract a float from an AV annual-reports list."""
    if not records or len(records) <= idx:
        return default
    val = records[idx].get(key)
    try:
        v = float(val)
        return v if v == v else default   # NaN guard
    except (TypeError, ValueError):
        return default


# ── Calculations ───────────────────────────────────────────────────────────────
def get_base_fcf(data: dict) -> float:
    cf  = data["cashflow"]
    ocf = _f(cf, "operatingCashflow")
    capex = _f(cf, "capitalExpenditures")   # positive in AV → subtract
    return ocf - capex


def get_growth_rate(data: dict) -> tuple:
    income  = data["income"]
    cashflow = data["cashflow"]
    overview = data["overview"]

    # 1. Historical revenue CAGR (up to 4 years)
    n = min(len(income), 4)
    if n >= 2:
        r0 = _f(income, "totalRevenue", 0)
        rn = _f(income, "totalRevenue", n - 1)
        if r0 > 0 and rn > 0:
            cagr = (r0 / rn) ** (1 / (n - 1)) - 1
            return float(np.clip(cagr, -0.10, 0.40)), f"Historical Revenue CAGR ({n}Y)"

    # 2. Historical FCF CAGR
    fcfs = []
    for i in range(min(len(cashflow), 4)):
        ocf   = _f(cashflow, "operatingCashflow",   i)
        capex = _f(cashflow, "capitalExpenditures",  i)
        fcfs.append(ocf - capex)
    if len(fcfs) >= 2 and fcfs[-1] > 0 and fcfs[0] > 0:
        cagr = (fcfs[0] / fcfs[-1]) ** (1 / (len(fcfs) - 1)) - 1
        return float(np.clip(cagr, -0.10, 0.35)), "Historical FCF CAGR"

    return 0.05, "Default (5%)"


def calc_wacc(data: dict) -> dict:
    ov      = data["overview"]
    income  = data["income"]
    balance = data["balance"]
    rf      = data["rf"]
    erp     = 0.055   # Damodaran equity risk premium

    beta = float(ov.get("Beta") or 1.0)
    beta = float(np.clip(beta, 0.1, 3.0))
    ke   = rf + beta * erp

    total_debt   = _f(balance, "shortLongTermDebtTotal")
    interest_exp = abs(_f(income, "interestExpense"))
    kd = (interest_exp / total_debt) if total_debt > 0 else 0.05
    kd = float(np.clip(kd, 0.02, 0.15))

    pretax   = _f(income, "incomeBeforeTax")
    tax_prov = _f(income, "incomeTaxExpense")
    tax_rate = (tax_prov / pretax) if pretax > 0 else 0.21
    tax_rate = float(np.clip(tax_rate, 0.0, 0.40))

    mkt_cap  = float(ov.get("MarketCapitalization") or 0)
    total_ev = mkt_cap + total_debt
    we = mkt_cap   / total_ev if total_ev > 0 else 1.0
    wd = total_debt / total_ev if total_ev > 0 else 0.0

    wacc = we * ke + wd * kd * (1 - tax_rate)
    return dict(wacc=wacc, ke=ke, kd=kd, beta=beta, rf=rf,
                tax_rate=tax_rate, we=we, wd=wd, total_debt=total_debt)


def run_dcf(base_fcf: float, g: float, g_term: float, wacc: float, data: dict) -> dict:
    if wacc <= g_term:
        g_term = wacc - 0.005

    fcfs, pv_fcfs, growth_rates = [], [], []
    for yr in range(1, 11):
        g_yr = g if yr <= 5 else g * (1 - (yr-5)/5) + g_term * ((yr-5)/5)
        growth_rates.append(g_yr)
        fcf = (base_fcf if yr == 1 else fcfs[-1]) * (1 + g_yr)
        fcfs.append(fcf)
        pv_fcfs.append(fcf / (1 + wacc) ** yr)

    tv    = fcfs[-1] * (1 + g_term) / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** 10
    ev    = sum(pv_fcfs) + pv_tv

    bal        = data["balance"]
    ov         = data["overview"]
    cash       = _f(bal, "cashAndShortTermInvestments") or _f(bal, "cashAndCashEquivalentsAtCarryingValue")
    total_debt = data.get("wacc_details", {}).get("total_debt", 0.0)
    net_debt   = total_debt - cash
    equity_val = ev - net_debt
    shares     = float(ov.get("SharesOutstanding") or
                       _f(data["balance"], "commonStockSharesOutstanding") or 1)
    iv = equity_val / shares

    return dict(fcfs=fcfs, pv_fcfs=pv_fcfs, growth_rates=growth_rates,
                pv_tv=pv_tv, total_pv_fcfs=sum(pv_fcfs),
                ev=ev, equity_val=equity_val, iv=iv,
                net_debt=net_debt, cash=cash)


def get_verdict(iv: float, price: float) -> tuple:
    if iv <= 0 or price <= 0:
        return "N/A", 0.0
    upside = (iv - price) / price * 100
    if   upside >  20: return "CHEAP",        upside
    elif upside < -20: return "EXPENSIVE",    upside
    else:              return "FAIRLY VALUED", upside


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

if not api_key:
    st.info(
        "👈  **Enter your free Alpha Vantage API key in the sidebar to get started.**\n\n"
        "Get one for free (no credit card) at "
        "[alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) — takes 30 seconds."
    )
    st.stop()

c1, c2 = st.columns([5, 1])
with c1:
    sym_input = st.text_input("", placeholder="Enter a ticker — e.g. AAPL, MSFT, NVDA, TSLA",
                               label_visibility="collapsed")
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
        base_fcf        = get_base_fcf(data)
        g, g_source     = get_growth_rate(data)
        g_term          = 0.025
        dcf             = run_dcf(base_fcf, g, g_term, wacc_d["wacc"], data)
        price           = data["price"]
        verd, upside    = get_verdict(dcf["iv"], price)

    for k, v in dict(data=data, wacc_d=wacc_d, base_fcf=base_fcf, g=g,
                     g_source=g_source, g_term=g_term, dcf=dcf,
                     price=price, verd=verd, upside=upside, analyzed=True).items():
        st.session_state[k] = v

# ── Display results ────────────────────────────────────────────────────────────
ss = st.session_state
if ss.get("analyzed"):
    data     = ss["data"]
    ov       = data["overview"]
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

    mkt_cap = float(ov.get("MarketCapitalization") or 0)
    st.subheader(f"{ov.get('Name', ov.get('Symbol', '—'))}  ({ov.get('Symbol', '—')})")
    st.caption(f"{ov.get('Sector','—')} · {ov.get('Industry','—')} · Market Cap ${mkt_cap/1e9:.1f}B")

    render_verdict_banner(verd, upside)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Market Price",       f"${price:.2f}")
    m2.metric("Intrinsic Value",    f"${max(dcf['iv'],0):.2f}", delta=f"{upside:+.1f}%")
    m3.metric("WACC",               f"{wacc_d['wacc']:.1%}")
    m4.metric("Growth Rate (Y1–5)", f"{g:.1%}", help=f"Source: {g_source}")
    m5.metric("Beta",               f"{wacc_d['beta']:.2f}")

    if base_fcf < 0:
        st.warning(f"⚠️  Base FCF is negative (${base_fcf/1e9:.2f}B). Interpret with caution.")

    st.divider()

    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.subheader("Projected Free Cash Flows")
        fig = go.Figure(go.Bar(
            x=[f"Y{i}" for i in range(1,11)],
            y=[f/1e9 for f in dcf["fcfs"]],
            marker_color=["#3b82f6" if f>0 else "#ef4444" for f in dcf["fcfs"]],
            text=[f"${f/1e9:.1f}B" for f in dcf["fcfs"]],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title="FCF ($B)", height=300,
                          margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.subheader("Enterprise Value Composition")
        pv_f, pv_t = max(dcf["total_pv_fcfs"],0), max(dcf["pv_tv"],0)
        if pv_f + pv_t > 0:
            fig2 = go.Figure(go.Pie(
                labels=["PV of FCFs (Y1–10)","PV of Terminal Value"],
                values=[pv_f/1e9, pv_t/1e9], hole=0.45,
                marker_colors=["#3b82f6","#93c5fd"], textinfo="label+percent",
            ))
            fig2.update_layout(showlegend=False, height=300, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Market Price vs Intrinsic Value")
    iv_d = max(dcf["iv"], 0)
    fig3 = go.Figure(go.Bar(
        x=["Current Market Price","Intrinsic Value (DCF)"],
        y=[price, iv_d],
        marker_color=["#94a3b8","#22c55e"] if dcf["iv"]>price else ["#94a3b8","#ef4444"],
        text=[f"${price:.2f}", f"${iv_d:.2f}"],
        textposition="outside", width=0.35,
    ))
    fig3.update_layout(yaxis_title="Price per share ($)", height=320,
                       margin=dict(l=0,r=0,t=20,b=0), plot_bgcolor="white",
                       yaxis_range=[0, max(price,iv_d)*1.30])
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("WACC Breakdown"):
        w1,w2,w3,w4 = st.columns(4)
        w1.metric("Risk-Free Rate",         f"{wacc_d['rf']:.2%}")
        w2.metric("Cost of Equity (CAPM)",  f"{wacc_d['ke']:.2%}")
        w3.metric("After-Tax Cost of Debt", f"{wacc_d['kd']*(1-wacc_d['tax_rate']):.2%}")
        w4.metric("Effective Tax Rate",     f"{wacc_d['tax_rate']:.1%}")
        w5,w6 = st.columns(2)
        w5.metric("Equity Weight", f"{wacc_d['we']:.1%}")
        w6.metric("Debt Weight",   f"{wacc_d['wd']:.1%}")

    with st.expander("DCF Bridge"):
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Enterprise Value", f"${dcf['ev']/1e9:.1f}B")
        b2.metric("Net Debt",         f"${dcf['net_debt']/1e9:.1f}B")
        b3.metric("Equity Value",     f"${dcf['equity_val']/1e9:.1f}B")
        shares = float(ov.get("SharesOutstanding") or 0)
        b4.metric("Shares Outstanding", f"{shares/1e9:.2f}B")

    # ── Override ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Override Assumptions")
    st.caption(f"Defaults auto-calculated. Growth source: **{g_source}**.")

    ov1,ov2,ov3 = st.columns(3)
    with ov1:
        ov_g = st.slider("Growth Rate Y1–5 (%)", -10.0, 50.0,
                         round(g*100,1), step=1.0) / 100
    with ov2:
        ov_gt = st.slider("Terminal Growth Rate (%)", 0.0, 5.0,
                          round(g_term*100,1), step=0.1) / 100
    with ov3:
        ov_wacc = st.slider("WACC (%)", 4.0, 20.0,
                            round(wacc_d["wacc"]*100,1), step=0.5) / 100

    if st.button("Recalculate with Overrides", type="secondary"):
        dcf2 = run_dcf(ss["base_fcf"], ov_g, ov_gt, ov_wacc, data)
        verd2, upside2 = get_verdict(dcf2["iv"], price)
        st.subheader("Revised Valuation")
        render_verdict_banner(verd2, upside2)

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Revised Intrinsic Value", f"${max(dcf2['iv'],0):.2f}",
                  delta=f"{dcf2['iv']-dcf['iv']:+.2f} vs base case")
        r2.metric("Market Price",      f"${price:.2f}")
        r3.metric("Upside / Downside", f"{upside2:+.1f}%")
        r4.metric("Verdict",           verd2)

        fig_cmp = go.Figure(go.Bar(
            x=["Market Price","Base Case IV","Revised IV"],
            y=[price, max(dcf["iv"],0), max(dcf2["iv"],0)],
            marker_color=["#94a3b8","#3b82f6","#6366f1"],
            text=[f"${price:.2f}", f"${max(dcf['iv'],0):.2f}", f"${max(dcf2['iv'],0):.2f}"],
            textposition="outside", width=0.4,
        ))
        fig_cmp.update_layout(
            yaxis_title="Price per share ($)", height=320,
            margin=dict(l=0,r=0,t=20,b=0), plot_bgcolor="white",
            yaxis_range=[0, max(price, dcf["iv"], dcf2["iv"])*1.30],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()
    st.caption("⚠️ For informational purposes only — not financial advice. "
               "Always do your own research before making investment decisions.")
