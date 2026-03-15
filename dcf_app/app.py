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

    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=SEC_HEADERS, timeout=20,
    ).json()

    facts_r = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers=SEC_HEADERS, timeout=30,
    ).json()

    price = 0.0
    beta  = 1.0
    rf    = 0.045   # safety fallback
    erp   = 0.055   # safety fallback
    try:
        def yf_closes(sym, interval, range_):
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                params={"interval": interval, "range": range_},
                headers=YF_HEADERS, timeout=15,
            ).json()
            result = r["chart"]["result"][0]
            closes = result["indicators"]["adjclose"][0]["adjclose"]
            return result["meta"]["regularMarketPrice"], closes

        # Risk-free rate: 10Y US Treasury (^TNX)
        tnx_price, _ = yf_closes("%5ETNX", "1d", "5d")
        rf = float(tnx_price) / 100

        # Current price + beta (2Y weekly vs S&P 500)
        price, s_closes = yf_closes(symbol,    "1wk", "2y")
        _,     m_closes = yf_closes("%5EGSPC", "1wk", "2y")
        n    = min(len(s_closes), len(m_closes))
        s    = np.array(s_closes[-n:], dtype=float)
        m    = np.array(m_closes[-n:], dtype=float)
        mask = ~(np.isnan(s) | np.isnan(m))
        s, m = s[mask], m[mask]
        if len(s) > 10:
            s_ret = np.diff(np.log(s))
            m_ret = np.diff(np.log(m))
            beta  = float(np.clip(np.cov(s_ret, m_ret)[0,1] / np.var(m_ret), 0.1, 3.0))

        # ERP: 20Y S&P 500 Total Return (^SP500TR) minus rf — dividends included, no hardcoded yield
        _, sp_tr = yf_closes("%5ESP500TR", "1mo", "20y")
        sp_tr = [c for c in sp_tr if c is not None]
        if len(sp_tr) >= 24:
            years        = len(sp_tr) / 12
            total_return = (sp_tr[-1] / sp_tr[0]) ** (1 / years) - 1
            erp          = float(np.clip(total_return - rf, 0.02, 0.12))
    except Exception:
        pass

    return dict(cik=cik, sub=sub, facts=facts_r, price=price, beta=beta, erp=erp, rf=rf)


# ── Market data: multiples, margins, 52W range ─────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(symbol: str) -> dict:
    """Current market multiples and margins from Yahoo Finance quoteSummary."""
    try:
        url    = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "defaultKeyStatistics,financialData,summaryDetail"}
        r = requests.get(url, params=params, headers=YF_HEADERS, timeout=15).json()
        res = r.get("quoteSummary", {}).get("result", [{}])
        if not res:
            return {}
        res = res[0]

        def _v(d, k):
            obj = d.get(k, {})
            return obj.get("raw") if isinstance(obj, dict) else None

        ks = res.get("defaultKeyStatistics", {})
        fd = res.get("financialData", {})
        sd = res.get("summaryDetail", {})

        return {
            "trailing_pe":       _v(sd, "trailingPE"),
            "forward_pe":        _v(sd, "forwardPE"),
            "price_to_book":     _v(ks, "priceToBook"),
            "ev_to_ebitda":      _v(ks, "enterpriseToEbitda"),
            "ev_to_revenue":     _v(ks, "enterpriseToRevenue"),
            "w52_high":          _v(sd, "fiftyTwoWeekHigh"),
            "w52_low":           _v(sd, "fiftyTwoWeekLow"),
            "revenue_growth":    _v(fd, "revenueGrowth"),
            "gross_margins":     _v(fd, "grossMargins"),
            "operating_margins": _v(fd, "operatingMargins"),
            "profit_margins":    _v(fd, "profitMargins"),
            "dividend_yield":    _v(sd, "dividendYield"),
        }
    except Exception:
        return {}


# ── Peer comparison via Yahoo Finance recommendations ──────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peers(symbol: str) -> list:
    """Return up to 5 comparable companies with key multiples."""
    try:
        r = requests.get(
            f"https://query2.finance.yahoo.com/v6/finance/recommendationsbyticker/{symbol}",
            headers=YF_HEADERS, timeout=10,
        ).json()
        rec = r.get("finance", {}).get("result", [{}])
        if not rec:
            return []
        syms = [
            x["symbol"] for x in rec[0].get("recommendedSymbols", [])
            if x.get("symbol") and x["symbol"] != symbol
        ][:5]
    except Exception:
        return []

    peers = []
    for sym in syms:
        try:
            url    = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
            params = {"modules": "defaultKeyStatistics,financialData,summaryDetail,price"}
            r2 = requests.get(url, params=params, headers=YF_HEADERS, timeout=10).json()
            res = r2.get("quoteSummary", {}).get("result", [{}])
            if not res:
                continue
            res = res[0]

            def _v(d, k):
                obj = d.get(k, {})
                return obj.get("raw") if isinstance(obj, dict) else None

            ks = res.get("defaultKeyStatistics", {})
            fd = res.get("financialData", {})
            sd = res.get("summaryDetail", {})
            pr = res.get("price", {})

            sname = pr.get("shortName", sym) or sym
            peers.append({
                "symbol":        sym,
                "name":          (sname[:22] + "…") if len(sname) > 22 else sname,
                "price":         _v(pr, "regularMarketPrice"),
                "mkt_cap_b":     (_v(pr, "marketCap") or 0) / 1e9,
                "trailing_pe":   _v(sd, "trailingPE"),
                "ev_to_ebitda":  _v(ks, "enterpriseToEbitda"),
                "ev_to_revenue": _v(ks, "enterpriseToRevenue"),
                "revenue_growth":_v(fd, "revenueGrowth"),
                "gross_margins": _v(fd, "grossMargins"),
                "profit_margins":_v(fd, "profitMargins"),
            })
        except Exception:
            continue

    return peers


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
        "Revenues", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax")
    ocf     = get_annual(us_gaap, "NetCashProvidedByUsedInOperatingActivities")
    capex   = get_annual(us_gaap,
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditureDiscontinuedOperations")
    interest = get_annual(us_gaap,
        "InterestExpense", "InterestAndDebtExpense", "InterestExpenseDebt")
    pretax  = get_annual(us_gaap,
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments")
    tax     = get_annual(us_gaap, "IncomeTaxExpenseBenefit", "IncomeTaxesPaid")
    lt_debt = get_annual(us_gaap, "LongTermDebt", "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations")
    st_debt = get_annual(us_gaap, "ShortTermBorrowings", "DebtCurrent", "LongTermDebtCurrent")
    cash    = get_annual(us_gaap,
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsPeriodIncreaseDecrease")
    shares  = get_annual(us_gaap,
        "CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding")
    net_income = get_annual(us_gaap, "NetIncomeLoss")
    equity  = get_annual(us_gaap,
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    dividends = get_annual(us_gaap,
        "PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
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
    return _first(fin["ocf"]) - _first(fin["capex"])


def get_growth_rate(fin: dict) -> tuple:
    """Return (blended_g, label, g_hist, g_sust)."""
    rev = fin["revenue"]
    ocf = fin["ocf"]
    cap = fin["capex"]

    g_hist = None
    n = min(len(rev), 4)
    if n >= 2 and rev[n-1] > 0 and rev[0] > 0:
        g_hist = float(np.clip((rev[0] / rev[n-1]) ** (1/(n-1)) - 1, -0.10, 0.40))
    else:
        fcfs = [o - c for o, c in zip(ocf, cap)]
        if len(fcfs) >= 2 and fcfs[-1] > 0 and fcfs[0] > 0:
            g_hist = float(np.clip((fcfs[0] / fcfs[-1]) ** (1/(len(fcfs)-1)) - 1, -0.10, 0.35))

    g_sust = None
    ni   = _first(fin.get("net_income", []))
    eq   = _first(fin.get("equity",     []))
    divs = _first(fin.get("dividends",  []))
    if ni > 0 and eq > 0:
        roe       = ni / eq
        payout    = min(divs / ni, 0.99) if divs > 0 else 0.0
        retention = 1.0 - payout
        g_sust    = float(np.clip(roe * retention, 0.0, 0.40))

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


def calc_wacc(fin: dict, price: float, beta: float = 1.0,
              rf: float = 0.043, erp: float = 0.055) -> dict:
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
    we = mkt_cap    / total_ev if total_ev > 0 else 1.0
    wd = total_debt / total_ev if total_ev > 0 else 0.0

    wacc = we * ke + wd * kd * (1 - tax_rate)

    return dict(wacc=wacc, ke=ke, kd=kd, beta=beta, rf=rf, erp=erp,
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


# ── Display utility helpers ────────────────────────────────────────────────────
def _fmt_x(v):
    """Format a valuation multiple, e.g. 24.5×"""
    return f"{v:.1f}×" if v is not None else "N/A"

def _fmt_pct(v):
    """Format a fraction as percentage, e.g. 34.2%"""
    return f"{v:.1%}" if v is not None else "N/A"


# ── Chart helpers ──────────────────────────────────────────────────────────────
def plot_52w_range(price: float, low: float, high: float) -> go.Figure:
    """Gauge showing where current price sits in the 52-week range."""
    pct  = (price - low) / (high - low) * 100 if high > low else 50
    color = "#22c55e" if pct < 35 else ("#ef4444" if pct > 75 else "#f59e0b")

    fig = go.Figure()
    # Background track
    fig.add_shape(type="line", x0=low, x1=high, y0=0, y1=0,
                  line=dict(color="#e2e8f0", width=16), layer="below")
    # Filled portion up to current price
    fig.add_shape(type="line", x0=low, x1=price, y0=0, y1=0,
                  line=dict(color=color, width=16), layer="below")
    # Endpoint + current price markers
    fig.add_trace(go.Scatter(
        x=[low, price, high], y=[0, 0, 0],
        mode="markers+text",
        marker=dict(size=[10, 24, 10],
                    color=["#64748b", color, "#64748b"],
                    symbol=["circle", "diamond", "circle"]),
        text=[f"52W Low<br><b>${low:.2f}</b>",
              f"<b>${price:.2f}</b><br>({pct:.0f}th pctile)",
              f"52W High<br><b>${high:.2f}</b>"],
        textposition=["bottom left", "top center", "bottom right"],
        textfont=dict(size=11),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=130, showlegend=False,
        yaxis=dict(visible=False, range=[-0.8, 0.8]),
        xaxis=dict(visible=False, range=[low * 0.93, high * 1.07]),
        margin=dict(l=70, r=70, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def plot_ev_waterfall(dcf: dict) -> go.Figure:
    """Waterfall bridge: PV FCFs + PV TV → EV → minus net debt → Equity Value."""
    pv_f = dcf["total_pv_fcfs"] / 1e9
    pv_t = dcf["pv_tv"] / 1e9
    nd   = dcf["net_debt"] / 1e9   # positive = net debt, negative = net cash
    ev   = dcf["ev"] / 1e9
    eq   = dcf["equity_val"] / 1e9

    nd_label = f"Net Cash (+${abs(nd):.1f}B)" if nd < 0 else f"Net Debt (−${nd:.1f}B)"

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total", "relative", "total"],
        x=["PV of FCFs\n(Yrs 1–10)", "PV of Terminal\nValue", "Enterprise\nValue",
           nd_label, "Equity\nValue"],
        y=[pv_f, pv_t, 0, -nd, 0],
        connector={"line": {"color": "#cbd5e1", "width": 1}},
        decreasing={"marker": {"color": "#ef4444", "line": {"color": "#dc2626", "width": 1}}},
        increasing={"marker": {"color": "#22c55e", "line": {"color": "#16a34a", "width": 1}}},
        totals={"marker": {"color": "#3b82f6", "line": {"color": "#2563eb", "width": 1}}},
        text=[f"${pv_f:.1f}B", f"${pv_t:.1f}B", f"${ev:.1f}B",
              f"{'−' if nd > 0 else '+'}${abs(nd):.1f}B", f"${eq:.1f}B"],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        yaxis_title="Value ($B)",
        height=360, plot_bgcolor="white",
        margin=dict(l=0, r=0, t=20, b=0),
        waterfallgap=0.4,
    )
    return fig


def plot_growth_fade(growth_rates: list, g_term: float) -> go.Figure:
    """Line chart showing how the applied growth rate fades from Y1 to Y10."""
    years = [f"Y{i}" for i in range(1, 11)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=[r * 100 for r in growth_rates],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=3),
        marker=dict(size=9, color="#3b82f6"),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        name="Applied Growth Rate",
    ))
    fig.add_hline(
        y=g_term * 100, line_dash="dot", line_color="#94a3b8",
        annotation_text=f"Terminal rate: {g_term:.1%}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        yaxis_title="Growth Rate (%)", showlegend=False,
        height=220, plot_bgcolor="white",
        margin=dict(l=0, r=80, t=10, b=0),
    )
    return fig


def plot_historical_bars(values: list, label: str, color: str):
    """Bar chart of last 4 annual values (oldest → latest)."""
    vals = list(reversed(values[:4]))
    n = len(vals)
    if n == 0:
        return None
    years = [f"FY−{n-1-i}" if i < n-1 else "Latest FY" for i in range(n)]
    fig = go.Figure(go.Bar(
        x=years, y=[v / 1e9 for v in vals],
        marker_color=[color if v >= 0 else "#ef4444" for v in vals],
        text=[f"${v/1e9:.1f}B" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title=f"{label} ($B)", height=230,
        plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def plot_peer_ev_ebitda(peers: list, main_sym: str, main_val) -> go.Figure:
    """Horizontal bar chart comparing EV/EBITDA of subject company vs peers."""
    labels, values, colors = [], [], []
    if main_val is not None:
        labels.append(f"▶ {main_sym}")
        values.append(round(main_val, 1))
        colors.append("#3b82f6")
    for p in peers:
        v = p.get("ev_to_ebitda")
        if v is not None:
            labels.append(p["symbol"])
            values.append(round(v, 1))
            colors.append("#94a3b8")
    if not labels:
        return None
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[str(v) for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="EV / EBITDA (×)",
        height=max(180, len(labels) * 46),
        plot_bgcolor="white",
        margin=dict(l=0, r=50, t=10, b=0),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📊 DCF Intrinsic Value Calculator")
st.caption(
    "Estimate the intrinsic value of any US-listed company using discounted cash flow analysis. "
    "All inputs — WACC, beta, ERP, growth rate — are calculated live from public data. "
    "Sources: SEC EDGAR · Yahoo Finance."
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
        wacc_d         = calc_wacc(fin, price, beta=data["beta"],
                                   rf=data["rf"], erp=data["erp"])
        base_fcf       = get_base_fcf(fin)
        g, g_source, g_hist, g_sust = get_growth_rate(fin)
        g_term         = 0.025
        dcf            = run_dcf(base_fcf, g, g_term, wacc_d["wacc"], fin)
        verd, upside   = get_verdict(dcf["iv"], price)

    with st.spinner("Loading market multiples & peer data…"):
        mkt_data = fetch_market_data(sym)
        peers    = fetch_peers(sym)

    for k, v in dict(
        data=data, fin=fin, wacc_d=wacc_d, base_fcf=base_fcf,
        g=g, g_source=g_source, g_hist=g_hist, g_sust=g_sust,
        g_term=g_term, dcf=dcf, price=price, verd=verd, upside=upside,
        mkt_data=mkt_data, peers=peers, analyzed=True,
    ).items():
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
    mkt_data = ss.get("mkt_data", {})
    peers    = ss.get("peers", [])

    sub      = data["sub"]
    name     = sub.get("name", "—")
    sic_desc = sub.get("sicDescription", "—")
    ticker   = (sub.get("tickers") or ["—"])[0]
    mkt_cap  = wacc_d["mkt_cap"]

    # ── Header ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"{name}  ({ticker})")
    st.caption(f"{sic_desc} · Market Cap ${mkt_cap/1e9:.1f}B")
    render_verdict_banner(verd, upside)

    # ── §1 Key Metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Market Price",       f"${price:.2f}")
    m2.metric("Intrinsic Value",    f"${max(dcf['iv'],0):.2f}", delta=f"{upside:+.1f}%")
    m3.metric("WACC",               f"{wacc_d['wacc']:.1%}")
    m4.metric("Growth Rate (Y1–5)", f"{g:.1%}", help=f"Source: {g_source}")
    m5.metric("Base FCF",           f"${base_fcf/1e9:.1f}B")

    if base_fcf < 0:
        st.warning("⚠️  Base FCF is negative. The model projects losses first — interpret the result with caution.")
    if price == 0:
        st.warning("⚠️  Could not fetch the current price. Use the override sliders below to enter it manually.")

    st.divider()

    # ── §2 Current Market Context ─────────────────────────────────────────────
    st.subheader("Current Market Context")
    st.caption(
        "Live market data from Yahoo Finance — how the market is pricing this company right now, "
        "independent of our DCF model."
    )

    # 52-week range gauge
    w52h = mkt_data.get("w52_high")
    w52l = mkt_data.get("w52_low")
    if w52h and w52l and price > 0 and w52h > w52l:
        pct_in_range = (price - w52l) / (w52h - w52l) * 100
        pos_label = "near its 52-week low" if pct_in_range < 25 else (
                    "near its 52-week high" if pct_in_range > 75 else
                    "in the middle of its 52-week range")
        st.caption(
            f"📍 Currently trading at the **{pct_in_range:.0f}th percentile** of its 52-week range — {pos_label}."
        )
        st.plotly_chart(plot_52w_range(price, w52l, w52h), use_container_width=True)

    # Market multiples
    cx1, cx2, cx3, cx4 = st.columns(4)
    cx1.metric("Trailing P/E",  _fmt_x(mkt_data.get("trailing_pe")),
               help="Current price ÷ trailing 12-month EPS. Reflects what the market pays per dollar of past earnings.")
    cx2.metric("Forward P/E",   _fmt_x(mkt_data.get("forward_pe")),
               help="Current price ÷ consensus next-12-month EPS forecast. A lower forward P/E vs trailing implies expected earnings growth.")
    cx3.metric("EV / EBITDA",   _fmt_x(mkt_data.get("ev_to_ebitda")),
               help="Enterprise Value ÷ EBITDA. Widely used acquisition multiple; useful for comparing companies with different capital structures.")
    cx4.metric("EV / Revenue",  _fmt_x(mkt_data.get("ev_to_revenue")),
               help="Enterprise Value ÷ Revenue. Useful for high-growth companies where earnings may be low or negative.")

    # Margin & growth metrics
    cx5, cx6, cx7, cx8 = st.columns(4)
    cx5.metric("Revenue Growth (YoY)",  _fmt_pct(mkt_data.get("revenue_growth")),
               help="Year-over-year revenue growth (trailing twelve months vs prior year).")
    cx6.metric("Gross Margin",          _fmt_pct(mkt_data.get("gross_margins")),
               help="(Revenue − Cost of Goods Sold) ÷ Revenue. Measures how efficiently the company produces its goods/services.")
    cx7.metric("Operating Margin",      _fmt_pct(mkt_data.get("operating_margins")),
               help="Operating Income ÷ Revenue. Shows profitability before interest and taxes — the core business margin.")
    cx8.metric("Net Profit Margin",     _fmt_pct(mkt_data.get("profit_margins")),
               help="Net Income ÷ Revenue. Bottom-line profitability after all expenses, interest, and taxes.")

    dy = mkt_data.get("dividend_yield")
    if dy and dy > 0:
        st.caption(f"💰 Dividend yield: **{dy:.2%}**")

    st.divider()

    # ── §3 DCF Projections ────────────────────────────────────────────────────
    st.subheader("DCF Projections")
    st.caption(
        "The model projects free cash flow for 10 years, then adds a terminal value using the Gordon Growth Model. "
        "Growth fades from the Y1–5 rate toward the long-run terminal rate in years 6–10."
    )

    ch1, ch2 = st.columns([3, 2])
    with ch1:
        fig_fcf = go.Figure(go.Bar(
            x=[f"Y{i}" for i in range(1, 11)],
            y=[f / 1e9 for f in dcf["fcfs"]],
            marker_color=["#3b82f6" if f > 0 else "#ef4444" for f in dcf["fcfs"]],
            text=[f"${f/1e9:.1f}B" for f in dcf["fcfs"]],
            textposition="outside",
        ))
        fig_fcf.update_layout(
            yaxis_title="FCF ($B)", height=280,
            margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
        )
        st.markdown("**Projected Free Cash Flows**")
        st.plotly_chart(fig_fcf, use_container_width=True)

    with ch2:
        pv_f, pv_t = max(dcf["total_pv_fcfs"], 0), max(dcf["pv_tv"], 0)
        if pv_f + pv_t > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["PV of FCFs (Y1–10)", "PV of Terminal Value"],
                values=[pv_f / 1e9, pv_t / 1e9], hole=0.45,
                marker_colors=["#3b82f6", "#93c5fd"], textinfo="label+percent",
            ))
            fig_pie.update_layout(
                showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.markdown("**Enterprise Value Composition**")
            st.plotly_chart(fig_pie, use_container_width=True)

    # Growth rate fade chart
    st.caption(
        "**How the growth rate evolves year by year** — flat at the forecast rate through Y5, "
        "then linearly fading to the long-run terminal rate by Y10:"
    )
    st.plotly_chart(plot_growth_fade(dcf["growth_rates"], g_term), use_container_width=True)

    st.divider()

    # ── §4 From Cash Flows to Intrinsic Value — The Bridge ────────────────────
    st.subheader("From Cash Flows to Intrinsic Value — The Bridge")
    st.caption(
        "This waterfall shows exactly how the model moves from projected cash flows to a per-share price. "
        "**Step 1:** Sum the discounted FCFs + terminal value → Enterprise Value. "
        "**Step 2:** Subtract net debt (or add net cash) → Equity Value. "
        "**Step 3:** Divide by shares outstanding → Intrinsic Value per share."
    )

    wf_col, iv_col = st.columns([3, 2])
    with wf_col:
        st.plotly_chart(plot_ev_waterfall(dcf), use_container_width=True)

    with iv_col:
        shares_out = _first(fin["shares"])
        iv_d = max(dcf["iv"], 0)
        st.markdown("**Intrinsic Value per Share**")
        st.markdown(
            f"**Equity Value** ${dcf['equity_val']/1e9:.1f}B  "
            f"÷  **{shares_out/1e9:.2f}B shares**  =  **${iv_d:.2f} / share**"
        )
        st.markdown("")
        fig_iv = go.Figure(go.Bar(
            x=["Market Price", "Intrinsic Value (DCF)"],
            y=[price, iv_d],
            marker_color=["#94a3b8", "#22c55e"] if dcf["iv"] > price else ["#94a3b8", "#ef4444"],
            text=[f"${price:.2f}", f"${iv_d:.2f}"],
            textposition="outside", width=0.4,
        ))
        fig_iv.update_layout(
            yaxis_title="$ per share", height=280,
            margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white",
            yaxis_range=[0, max(price, iv_d) * 1.35],
        )
        st.plotly_chart(fig_iv, use_container_width=True)

    st.divider()

    # ── §5 WACC Breakdown ──────────────────────────────────────────────────────
    st.subheader("WACC Breakdown")
    st.caption(
        f"**Formula:** WACC = Equity weight ({wacc_d['we']:.0%}) × Ke ({wacc_d['ke']:.2%}) "
        f"+ Debt weight ({wacc_d['wd']:.0%}) × Kd after-tax ({wacc_d['kd']*(1-wacc_d['tax_rate']):.2%}) "
        f"= **{wacc_d['wacc']:.2%}**. "
        "A higher WACC discounts future cash flows more heavily, reducing intrinsic value."
    )
    w1, w2, w3, w4, w5, w6, w7 = st.columns(7)
    w1.metric("Risk-Free Rate",
              f"{wacc_d['rf']:.2%}",
              help="Live 10Y US Treasury yield (^TNX). The 'guaranteed' return — the baseline for all risky investments.")
    w2.metric("Equity Risk Premium",
              f"{wacc_d['erp']:.2%}",
              help="20Y annualised S&P 500 Total Return (^SP500TR) minus the risk-free rate. "
                   "Represents the extra return investors historically demanded for holding equities over bonds.")
    w3.metric("Beta (β)",
              f"{wacc_d['beta']:.2f}",
              help="Calculated from 2Y weekly returns vs S&P 500. β = 1.0 means the stock moves in line with the market. "
                   "β > 1 = more volatile (riskier); β < 1 = less volatile.")
    w4.metric("Cost of Equity (Ke)",
              f"{wacc_d['ke']:.2%}",
              help="CAPM: Rf + β × ERP. The minimum return equity investors require to hold this stock.")
    w5.metric("After-Tax Cost of Debt",
              f"{wacc_d['kd']*(1-wacc_d['tax_rate']):.2%}",
              help="(Interest Expense ÷ Total Debt) × (1 − Tax Rate). "
                   "Debt is cheaper than equity because interest is tax-deductible.")
    w6.metric("Equity Weight",
              f"{wacc_d['we']:.1%}",
              help="Market cap ÷ (Market cap + Total debt). How much of the firm's capital comes from equity.")
    w7.metric("Debt Weight",
              f"{wacc_d['wd']:.1%}",
              help="Total debt ÷ (Market cap + Total debt). How much of the firm's capital comes from debt.")

    st.divider()

    # ── §6 Growth Rate Breakdown ───────────────────────────────────────────────
    st.subheader("Growth Rate Breakdown")
    st.caption(
        "Two independent methods are blended into the Y1–5 growth rate. "
        "**Historical Revenue CAGR** is backward-looking (from 4 years of SEC 10-K filings). "
        "**Sustainable Growth Rate (SGR = ROE × retention ratio)** is fundamentals-based — "
        "it answers: at the current profitability and reinvestment rate, how fast can this business organically grow?"
    )
    ga, gb, gc = st.columns(3)
    ga.metric("Historical Revenue CAGR",
              f"{g_hist:.1%}" if g_hist is not None else "n/a",
              help="Compound annual growth rate of revenue over the last 4 annual 10-K filings.")
    gb.metric("Sustainable Growth Rate (SGR)",
              f"{g_sust:.1%}" if g_sust is not None else "n/a",
              help="SGR = ROE × (1 − dividend payout ratio). Represents how fast the company can grow "
                   "using only internally generated funds, at the current return on equity.")
    gc.metric("Blended Rate used in DCF (Y1–5)",
              f"{g:.1%}",
              help="Simple average of both estimates above. Using two methods reduces dependence on any single assumption.")

    st.divider()

    # ── §7 Historical Financial Trends ────────────────────────────────────────
    st.subheader("Historical Financial Trends")
    st.caption(
        "Last 4 years of reported figures from SEC EDGAR 10-K filings — "
        "provides context for whether the growth rate assumptions are grounded in historical reality."
    )
    ht1, ht2 = st.columns(2)
    with ht1:
        fig_rev_h = plot_historical_bars(fin["revenue"], "Revenue", "#3b82f6")
        if fig_rev_h:
            st.markdown("**Revenue (last 4 fiscal years)**")
            st.plotly_chart(fig_rev_h, use_container_width=True)
        else:
            st.caption("Revenue history unavailable.")

    with ht2:
        fcf_hist = [o - c for o, c in zip(fin["ocf"], fin["capex"])]
        fig_fcf_h = plot_historical_bars(fcf_hist, "Free Cash Flow", "#22c55e")
        if fig_fcf_h:
            st.markdown("**Free Cash Flow = OCF − CapEx (last 4 fiscal years)**")
            st.plotly_chart(fig_fcf_h, use_container_width=True)
        else:
            st.caption("FCF history unavailable.")

    st.divider()

    # ── §8 Peer Comparison ────────────────────────────────────────────────────
    st.subheader("Peer Comparison")
    if peers:
        st.caption(
            f"Top comparable companies identified by Yahoo Finance. "
            f"The highlighted **▶ {ticker}** row shows this company's current live multiples."
        )

        # Build comparison table rows
        main_row = {
            "Symbol":        f"▶ {ticker}",
            "Company":       name[:22],
            "Mkt Cap ($B)":  f"{mkt_cap/1e9:.1f}",
            "P/E":           _fmt_x(mkt_data.get("trailing_pe")),
            "EV/EBITDA":     _fmt_x(mkt_data.get("ev_to_ebitda")),
            "EV/Revenue":    _fmt_x(mkt_data.get("ev_to_revenue")),
            "Rev Growth":    _fmt_pct(mkt_data.get("revenue_growth")),
            "Gross Margin":  _fmt_pct(mkt_data.get("gross_margins")),
            "Net Margin":    _fmt_pct(mkt_data.get("profit_margins")),
        }
        peer_rows = []
        for p in peers:
            peer_rows.append({
                "Symbol":       p["symbol"],
                "Company":      p["name"],
                "Mkt Cap ($B)": f"{p['mkt_cap_b']:.1f}",
                "P/E":          _fmt_x(p.get("trailing_pe")),
                "EV/EBITDA":    _fmt_x(p.get("ev_to_ebitda")),
                "EV/Revenue":   _fmt_x(p.get("ev_to_revenue")),
                "Rev Growth":   _fmt_pct(p.get("revenue_growth")),
                "Gross Margin": _fmt_pct(p.get("gross_margins")),
                "Net Margin":   _fmt_pct(p.get("profit_margins")),
            })

        all_rows = [main_row] + peer_rows
        headers  = list(main_row.keys())
        cell_vals = [[r[h] for r in all_rows] for h in headers]
        n_peers   = len(all_rows)
        row_colors = [["#dbeafe"] + ["#f8fafc" if i % 2 == 0 else "white" for i in range(n_peers - 1)]
                      for _ in headers]

        fig_tbl = go.Figure(go.Table(
            header=dict(
                values=headers,
                fill_color="#1e3a5f",
                font=dict(color="white", size=12),
                align="center",
                height=36,
            ),
            cells=dict(
                values=cell_vals,
                fill_color=row_colors,
                font=dict(size=12),
                align="center",
                height=32,
            ),
        ))
        fig_tbl.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=36 + 32 * n_peers + 30,
        )
        st.plotly_chart(fig_tbl, use_container_width=True)

        # EV/EBITDA bar chart
        fig_peer = plot_peer_ev_ebitda(peers, ticker, mkt_data.get("ev_to_ebitda"))
        if fig_peer:
            st.caption(
                "**EV/EBITDA comparison** — a higher multiple typically signals the market expects faster growth "
                "or lower risk relative to peers. If this company trades at a premium, the DCF should justify it with higher FCF growth."
            )
            st.plotly_chart(fig_peer, use_container_width=True)

    else:
        st.info("Peer comparison data is unavailable for this ticker.")

    # ── §9 Detailed DCF Numbers (expander) ────────────────────────────────────
    with st.expander("📋  Detailed DCF Numbers"):
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Enterprise Value",   f"${dcf['ev']/1e9:.1f}B")
        b2.metric("Net Debt",           f"${dcf['net_debt']/1e9:.1f}B",
                  help="Total Debt − Cash. Positive = net debt; negative = net cash position.")
        b3.metric("Equity Value",       f"${dcf['equity_val']/1e9:.1f}B")
        b4.metric("Shares Outstanding", f"{_first(fin['shares'])/1e9:.2f}B")

        years_   = [f"Y{i}" for i in range(1, 11)]
        gr_      = [f"{r:.1%}" for r in dcf["growth_rates"]]
        fcfs_    = [f"${f/1e9:.2f}B" for f in dcf["fcfs"]]
        pv_fcfs_ = [f"${p/1e9:.2f}B" for p in dcf["pv_fcfs"]]

        fig_dtbl = go.Figure(go.Table(
            header=dict(
                values=["Year", "Growth Rate", "FCF (nominal)", "PV of FCF"],
                fill_color="#1e3a5f",
                font=dict(color="white", size=11),
                align="center",
            ),
            cells=dict(
                values=[years_, gr_, fcfs_, pv_fcfs_],
                fill_color=[["#f8fafc", "white"] * 5],
                font=dict(size=11),
                align="center",
            ),
        ))
        fig_dtbl.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380)
        st.plotly_chart(fig_dtbl, use_container_width=True)

    # ── §10 Override Assumptions ──────────────────────────────────────────────
    st.divider()
    st.subheader("Override Assumptions")
    st.caption(
        f"All defaults are auto-calculated from live data. Growth source: **{g_source}**. "
        "Drag sliders to stress-test different scenarios."
    )

    ov1, ov2, ov3 = st.columns(3)
    with ov1:
        ov_g = st.slider("Growth Rate Y1–5 (%)", -10.0, 50.0,
                         round(g * 100, 1), step=1.0) / 100
    with ov2:
        ov_gt = st.slider("Terminal Growth Rate (%)", 0.0, 5.0,
                          round(g_term * 100, 1), step=0.1) / 100
    with ov3:
        ov_wacc = st.slider("WACC (%)", 4.0, 20.0,
                            round(wacc_d["wacc"] * 100, 1), step=0.5) / 100

    if st.button("Recalculate with Overrides", type="secondary"):
        dcf2 = run_dcf(ss["base_fcf"], ov_g, ov_gt, ov_wacc, fin)
        verd2, upside2 = get_verdict(dcf2["iv"], price)
        st.subheader("Revised Valuation")
        render_verdict_banner(verd2, upside2)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Revised Intrinsic Value", f"${max(dcf2['iv'],0):.2f}",
                  delta=f"{dcf2['iv']-dcf['iv']:+.2f} vs base case")
        r2.metric("Market Price",      f"${price:.2f}")
        r3.metric("Upside / Downside", f"{upside2:+.1f}%")
        r4.metric("Verdict",           verd2)

        fig_cmp = go.Figure(go.Bar(
            x=["Market Price", "Base Case IV", "Revised IV"],
            y=[price, max(dcf["iv"], 0), max(dcf2["iv"], 0)],
            marker_color=["#94a3b8", "#3b82f6", "#6366f1"],
            text=[f"${price:.2f}", f"${max(dcf['iv'],0):.2f}", f"${max(dcf2['iv'],0):.2f}"],
            textposition="outside", width=0.4,
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
        "Data from SEC EDGAR and Yahoo Finance (US-listed companies only). "
        "Always do your own research before making investment decisions."
    )
