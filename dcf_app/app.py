import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go

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

# ── Constants ──────────────────────────────────────────────────────────────────
SEC_HEADERS = {"User-Agent": "dcf-tool contact@example.com"}
YF_HEADERS  = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ── Data layer — SEC EDGAR (free, no key) ─────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def get_cik_map() -> dict:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS, timeout=20,
    )
    r.raise_for_status()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}


def ticker_to_cik(symbol: str) -> str:
    cik_map = get_cik_map()
    cik = cik_map.get(symbol.upper())
    if not cik:
        raise ValueError(f"Ticker '{symbol}' not found in SEC EDGAR. Only US-listed companies are supported.")
    return cik


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str) -> dict:
    cik = ticker_to_cik(symbol)

    # Company name & sector
    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS, timeout=20,
    ).json()

    # XBRL financial facts
    facts_r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=SEC_HEADERS, timeout=30,
    ).json()

    # Price + beta from Yahoo Finance chart API (2y weekly history vs S&P 500)
    price = 0.0
    beta  = 1.0
    try:
        def yf_closes(sym, interval="1wk", range_="2y"):
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                params={"interval": interval, "range": range_},
                headers=YF_HEADERS, timeout=15,
            ).json()
            result = r["chart"]["result"][0]
            closes = result["indicators"]["adjclose"][0]["adjclose"]
            return result["meta"]["regularMarketPrice"], closes

        price, s_closes = yf_closes(symbol)
        _,     m_closes = yf_closes("^GSPC")

        n    = min(len(s_closes), len(m_closes))
        s    = np.array(s_closes[-n:], dtype=float)
        m    = np.array(m_closes[-n:], dtype=float)
        mask = ~(np.isnan(s) | np.isnan(m))
        s, m = s[mask], m[mask]
        if len(s) > 10:
            s_ret = np.diff(np.log(s))
            m_ret = np.diff(np.log(m))
            beta  = float(np.clip(np.cov(s_ret, m_ret)[0,1] / np.var(m_ret), 0.1, 3.0))
    except Exception:
        pass

    return dict(cik=cik, sub=sub, facts=facts_r, price=price, beta=beta)


# ── XBRL extraction helpers ────────────────────────────────────────────────────
def get_annual(us_gaap: dict, *tags, n: int = 4) -> list:
    """Return up to n most-recent annual values, trying tags in order."""
    for tag in tags:
        tag_data = us_gaap.get(tag, {}).get("units", {})
        raw = tag_data.get("USD", tag_data.get("shares", []))
        annual = [v for v in raw if v.get("form") == "10-K" and v.get("fp") == "FY"]
        seen, out = set(), []
        for v in sorted(annual, key=lambda x: x["end"], reverse=True):
            fy = v.get("fy")
            if fy not in seen:
                seen.add(fy)
                out.append(float(v["val"]))
            if len(out) >= n:
                break
        if out:
            return out
    return []


def extract_financials(data: dict) -> dict:
    us_gaap = data["facts"]["facts"]["us-gaap"]

    revenue = get_annual(us_gaap,
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax")

    ocf = get_annual(us_gaap, "NetCashProvidedByUsedInOperatingActivities")

    capex = get_annual(us_gaap,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditureDiscontinuedOperations")

    interest = get_annual(us_gaap,
        "InterestExpense", "InterestAndDebtExpense",
        "InterestExpenseDebt")

    pretax = get_annual(us_gaap,
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments")

    tax = get_annual(us_gaap,
        "IncomeTaxExpenseBenefit", "IncomeTaxesPaid")

    lt_debt = get_annual(us_gaap,
        "LongTermDebt", "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations")

    st_debt = get_annual(us_gaap,
        "ShortTermBorrowings", "DebtCurrent",
        "LongTermDebtCurrent")

    cash = get_annual(us_gaap,
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsPeriodIncreaseDecrease")

    shares = get_annual(us_gaap,
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding")

    net_income = get_annual(us_gaap, "NetIncomeLoss")

    equity = get_annual(us_gaap,
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")

    dividends = get_annual(us_gaap,
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
        "PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndPreferredStock")

    return dict(
        revenue=revenue, ocf=ocf, capex=capex,
        interest=interest, pretax=pretax, tax=tax,
        lt_debt=lt_debt, st_debt=st_debt, cash=cash, shares=shares,
        net_income=net_income, equity=equity, dividends=dividends,
    )


def _first(lst: list, default: float = 0.0) -> float:
    return float(lst[0]) if lst else default


# ── Calculation layer ──────────────────────────────────────────────────────────
def get_base_fcf(fin: dict) -> float:
    ocf   = _first(fin["ocf"])
    capex = _first(fin["capex"])
    return ocf - capex   # capex is positive in EDGAR


def get_growth_rate(fin: dict) -> tuple:
    """Return (blended_g, label, g_hist, g_sust).
    g_hist  = historical revenue CAGR (backward-looking)
    g_sust  = sustainable growth rate ROE × retention ratio (fundamental)
    blended = average of the two when both available.
    """
    rev = fin["revenue"]
    ocf = fin["ocf"]
    cap = fin["capex"]

    # ── 1. Historical revenue CAGR ──────────────────────────────────────────
    g_hist = None
    n = min(len(rev), 4)
    if n >= 2 and rev[n-1] > 0 and rev[0] > 0:
        g_hist = float(np.clip((rev[0] / rev[n-1]) ** (1/(n-1)) - 1, -0.10, 0.40))
    else:
        fcfs = [o - c for o, c in zip(ocf, cap)]
        if len(fcfs) >= 2 and fcfs[-1] > 0 and fcfs[0] > 0:
            g_hist = float(np.clip((fcfs[0] / fcfs[-1]) ** (1/(len(fcfs)-1)) - 1, -0.10, 0.35))

    # ── 2. Sustainable growth rate = ROE × retention ratio ─────────────────
    # ROE = Net Income / Equity
    # Retention ratio = 1 − (Dividends / Net Income)
    g_sust = None
    ni    = _first(fin.get("net_income", []))
    eq    = _first(fin.get("equity",     []))
    divs  = _first(fin.get("dividends",  []))
    if ni > 0 and eq > 0:
        roe       = ni / eq
        payout    = min(divs / ni, 0.99) if divs > 0 else 0.0
        retention = 1.0 - payout
        g_sust    = float(np.clip(roe * retention, 0.0, 0.40))

    # ── 3. Blend ────────────────────────────────────────────────────────────
    available = [x for x in [g_hist, g_sust] if x is not None]
    if available:
        g = float(np.mean(available))
        parts = []
        if g_hist is not None: parts.append(f"Rev CAGR {g_hist:.1%}")
        if g_sust is not None: parts.append(f"SGR {g_sust:.1%}")
        label = "Avg of " + " & ".join(parts)
    else:
        g, label = 0.05, "Default (5%)"

    return g, label, g_hist, g_sust


def calc_wacc(fin: dict, price: float, beta: float = 1.0, rf: float = 0.043) -> dict:
    erp = 0.055
    beta = float(np.clip(beta, 0.1, 3.0))
    ke   = rf + beta * erp

    total_debt   = _first(fin["lt_debt"]) + _first(fin["st_debt"])
    interest_exp = _first(fin["interest"])
    kd = (interest_exp / total_debt) if total_debt > 0 else 0.05
    kd = float(np.clip(kd, 0.02, 0.15))

    pretax   = _first(fin["pretax"])
    tax_prov = _first(fin["tax"])
    tax_rate = (tax_prov / pretax) if pretax > 0 else 0.21
    tax_rate = float(np.clip(tax_rate, 0.0, 0.40))

    shares   = _first(fin["shares"])
    mkt_cap  = price * shares
    total_ev = mkt_cap + total_debt
    we = mkt_cap   / total_ev if total_ev > 0 else 1.0
    wd = total_debt / total_ev if total_ev > 0 else 0.0

    wacc = we * ke + wd * kd * (1 - tax_rate)

    return dict(wacc=wacc, ke=ke, kd=kd, beta=beta, rf=rf,
                tax_rate=tax_rate, we=we, wd=wd,
                total_debt=total_debt, mkt_cap=mkt_cap)


def run_dcf(base_fcf: float, g: float, g_term: float, wacc: float, fin: dict) -> dict:
    if wacc <= g_term:
        g_term = wacc - 0.005

    fcfs, pv_fcfs, growth_rates = [], [], []
    for yr in range(1, 11):
        g_yr = g if yr <= 5 else g * (1-(yr-5)/5) + g_term * ((yr-5)/5)
        growth_rates.append(g_yr)
        fcf = (base_fcf if yr == 1 else fcfs[-1]) * (1 + g_yr)
        fcfs.append(fcf)
        pv_fcfs.append(fcf / (1 + wacc) ** yr)

    tv    = fcfs[-1] * (1 + g_term) / (wacc - g_term)
    pv_tv = tv / (1 + wacc) ** 10
    ev    = sum(pv_fcfs) + pv_tv

    cash       = _first(fin["cash"])
    total_debt = _first(fin["lt_debt"]) + _first(fin["st_debt"])
    net_debt   = total_debt - cash
    equity_val = ev - net_debt
    shares     = _first(fin["shares"])
    iv         = equity_val / shares if shares > 0 else 0.0

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
st.caption(
    "Estimate the intrinsic value of any US-listed company using discounted cash flow analysis. "
    "Data sourced from SEC EDGAR — no API key required."
)

c1, c2 = st.columns([5, 1])
with c1:
    sym_input = st.text_input("", placeholder="Enter a ticker — e.g. AAPL, MSFT, NVDA, TSLA",
                               label_visibility="collapsed")
with c2:
    go_btn = st.button("Analyze →", type="primary", use_container_width=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if go_btn and sym_input:
    sym = sym_input.upper().strip()
    with st.spinner(f"Fetching SEC filings for {sym}…"):
        try:
            data = fetch_data(sym)
        except Exception as e:
            st.error(f"Could not load data for **{sym}**: {e}")
            st.stop()

    with st.spinner("Running DCF model…"):
        fin            = extract_financials(data)
        price          = data["price"]
        wacc_d         = calc_wacc(fin, price, beta=data["beta"])
        base_fcf       = get_base_fcf(fin)
        g, g_source, g_hist, g_sust = get_growth_rate(fin)
        g_term         = 0.025
        dcf            = run_dcf(base_fcf, g, g_term, wacc_d["wacc"], fin)
        verd, upside   = get_verdict(dcf["iv"], price)

    for k, v in dict(data=data, fin=fin, wacc_d=wacc_d, base_fcf=base_fcf,
                     g=g, g_source=g_source, g_hist=g_hist, g_sust=g_sust,
                     g_term=g_term, dcf=dcf,
                     price=price, verd=verd, upside=upside, analyzed=True).items():
        st.session_state[k] = v

# ── Display results ────────────────────────────────────────────────────────────
ss = st.session_state
if ss.get("analyzed"):
    data     = ss["data"]
    fin      = ss["fin"]
    wacc_d   = ss["wacc_d"]
    dcf      = ss["dcf"]
    price    = ss["price"]
    verd     = ss["verd"]
    upside   = ss["upside"]
    g        = ss["g"]
    g_source = ss["g_source"]
    g_hist   = ss["g_hist"]
    g_sust   = ss["g_sust"]
    g_term   = ss["g_term"]
    base_fcf = ss["base_fcf"]

    sub      = data["sub"]
    name     = sub.get("name", sub.get("entityType", "—"))
    sic_desc = sub.get("sicDescription", "—")
    mkt_cap  = wacc_d["mkt_cap"]

    st.divider()
    st.subheader(f"{name}  ({sub.get('tickers', ['—'])[0] if sub.get('tickers') else '—'})")
    st.caption(f"{sic_desc} · Market Cap ${mkt_cap/1e9:.1f}B")

    render_verdict_banner(verd, upside)

    # Key metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Market Price",       f"${price:.2f}")
    m2.metric("Intrinsic Value",    f"${max(dcf['iv'],0):.2f}", delta=f"{upside:+.1f}%")
    m3.metric("WACC",               f"{wacc_d['wacc']:.1%}")
    m4.metric("Growth Rate (Y1–5)", f"{g:.1%}", help=f"Source: {g_source}")
    m5.metric("Base FCF",           f"${base_fcf/1e9:.1f}B")

    if base_fcf < 0:
        st.warning(f"⚠️  Base FCF is negative (${base_fcf/1e9:.2f}B). Interpret the result with caution.")

    if price == 0:
        st.warning("⚠️  Could not fetch the current price automatically. Use the override below to enter it.")

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

    # ── WACC Breakdown ────────────────────────────────────────────────────────
    st.subheader("WACC Breakdown")
    st.caption(
        f"WACC = {wacc_d['we']:.0%} × {wacc_d['ke']:.2%} (equity) "
        f"+ {wacc_d['wd']:.0%} × {wacc_d['kd']*(1-wacc_d['tax_rate']):.2%} (after-tax debt) "
        f"= **{wacc_d['wacc']:.2%}**"
    )
    w1,w2,w3,w4,w5,w6 = st.columns(6)
    w1.metric("Risk-Free Rate",         f"{wacc_d['rf']:.2%}", help="US 10Y Treasury yield")
    w2.metric("Beta",                   f"{wacc_d['beta']:.2f}", help="2Y weekly returns vs S&P 500")
    w3.metric("Cost of Equity (CAPM)",  f"{wacc_d['ke']:.2%}", help="Rf + β × 5.5% ERP")
    w4.metric("After-Tax Cost of Debt", f"{wacc_d['kd']*(1-wacc_d['tax_rate']):.2%}", help="Interest / Debt × (1 − tax rate)")
    w5.metric("Equity Weight",          f"{wacc_d['we']:.1%}")
    w6.metric("Debt Weight",            f"{wacc_d['wd']:.1%}")

    # ── Growth Rate Breakdown ─────────────────────────────────────────────────
    st.subheader("Growth Rate Breakdown")
    st.caption(
        "Two independent estimates are blended into the Y1–5 growth rate. "
        "Historical CAGR is backward-looking; Sustainable Growth Rate (SGR = ROE × retention ratio) is fundamentals-based."
    )
    ga, gb, gc = st.columns(3)
    ga.metric("Historical Revenue CAGR",
              f"{g_hist:.1%}" if g_hist is not None else "n/a",
              help="Revenue compound annual growth rate from last 4 annual filings")
    gb.metric("Sustainable Growth Rate",
              f"{g_sust:.1%}" if g_sust is not None else "n/a",
              help="SGR = ROE × (1 − dividend payout ratio)")
    gc.metric("Blended Rate (used in DCF)", f"{g:.1%}",
              help="Simple average of the two estimates above")

    # ── DCF Bridge ────────────────────────────────────────────────────────────
    with st.expander("DCF Bridge"):
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Enterprise Value", f"${dcf['ev']/1e9:.1f}B")
        b2.metric("Net Debt",         f"${dcf['net_debt']/1e9:.1f}B")
        b3.metric("Equity Value",     f"${dcf['equity_val']/1e9:.1f}B")
        shares = _first(fin["shares"])
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
        dcf2 = run_dcf(ss["base_fcf"], ov_g, ov_gt, ov_wacc, fin)
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
    st.caption(
        "⚠️ For informational purposes only — not financial advice. "
        "Data from SEC EDGAR (US companies only). "
        "Always do your own research before making investment decisions."
    )
