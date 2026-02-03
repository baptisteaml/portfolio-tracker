"""
Portfolio Dashboard - Hedge Fund Style
A comprehensive portfolio tracker with trading, analytics, and benchmarking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f2937; margin-bottom: 0; }
    .positive { color: #10b981 !important; }
    .negative { color: #ef4444 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

DATA_FILE = "portfolio_data.json"


def load_portfolio_data() -> dict:
    """Load portfolio data from JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_portfolio_data(data: dict):
    """Save portfolio data to JSON file."""
    data["updated_at"] = datetime.now().isoformat()
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


@st.cache_data(ttl=300)
def fetch_fx_rates() -> Dict[str, float]:
    """Fetch current FX rates to USD."""
    fx_tickers = ["HKDUSD=X", "EURUSD=X", "DKKUSD=X"]
    rates = {"USD": 1.0}

    for ticker in fx_tickers:
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="1d")
            if not hist.empty:
                currency = ticker.replace("USD=X", "")
                rates[currency] = float(hist['Close'].iloc[-1])
        except:
            pass

    # Fallback rates if fetch fails
    if "HKD" not in rates:
        rates["HKD"] = 0.128
    if "EUR" not in rates:
        rates["EUR"] = 1.08
    if "DKK" not in rates:
        rates["DKK"] = 0.145

    return rates


@st.cache_data(ttl=300)
def fetch_prices_with_prev(tickers: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Fetch current and previous day prices for tickers."""
    current_prices = {}
    prev_prices = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) >= 2:
                current_prices[ticker] = float(hist['Close'].iloc[-1])
                prev_prices[ticker] = float(hist['Close'].iloc[-2])
            elif len(hist) == 1:
                current_prices[ticker] = float(hist['Close'].iloc[-1])
                prev_prices[ticker] = current_prices[ticker]
        except Exception as e:
            pass

    return current_prices, prev_prices


@st.cache_data(ttl=600)
def fetch_price_history(tickers: List[str], start_date: str) -> pd.DataFrame:
    """Fetch price history from start date."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, start=start_date, progress=False)
        if len(tickers) == 1:
            prices = data['Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data['Close']
        return prices
    except:
        return pd.DataFrame()


def get_all_tickers(data: dict) -> List[str]:
    """Get all tickers from portfolio."""
    tickers = []
    for strategy in data.get("strategies", {}).values():
        for holding in strategy.get("holdings", []):
            tickers.append(holding["ticker"])
    return tickers


def calculate_portfolio_value(data: dict, prices: Dict[str, float], fx_rates: Dict[str, float]) -> Tuple[float, Dict]:
    """Calculate portfolio value with currency conversion."""
    total_value = data.get("cash", 0)
    holdings_detail = {}

    for strategy_name, strategy in data.get("strategies", {}).items():
        for holding in strategy.get("holdings", []):
            ticker = holding["ticker"]
            shares = holding["shares"]
            cost_basis = holding.get("cost_basis", 0)
            currency = holding.get("currency", "USD")

            if ticker in prices:
                local_price = prices[ticker]
                fx_rate = fx_rates.get(currency, 1.0)
                usd_price = local_price * fx_rate
                usd_value = shares * usd_price
                usd_cost = shares * cost_basis * fx_rate

                holdings_detail[ticker] = {
                    "ticker": ticker,
                    "shares": shares,
                    "local_price": local_price,
                    "currency": currency,
                    "fx_rate": fx_rate,
                    "usd_price": usd_price,
                    "usd_value": usd_value,
                    "cost_basis": cost_basis,
                    "usd_cost": usd_cost,
                    "pnl": usd_value - usd_cost,
                    "pnl_pct": ((usd_value - usd_cost) / usd_cost * 100) if usd_cost > 0 else 0,
                    "strategy": strategy_name
                }
                total_value += usd_value

    return total_value, holdings_detail


def calculate_daily_changes(data: dict, current_prices: Dict, prev_prices: Dict, fx_rates: Dict) -> Dict:
    """Calculate daily changes for all holdings."""
    changes = {}

    for strategy_name, strategy in data.get("strategies", {}).items():
        for holding in strategy.get("holdings", []):
            ticker = holding["ticker"]
            shares = holding["shares"]
            currency = holding.get("currency", "USD")
            fx_rate = fx_rates.get(currency, 1.0)

            if ticker in current_prices and ticker in prev_prices:
                curr = current_prices[ticker] * fx_rate
                prev = prev_prices[ticker] * fx_rate

                daily_change_pct = ((curr - prev) / prev * 100) if prev > 0 else 0
                daily_pnl = shares * (curr - prev)

                changes[ticker] = {
                    "ticker": ticker,
                    "shares": shares,
                    "current_price": curr,
                    "prev_price": prev,
                    "daily_change_pct": daily_change_pct,
                    "daily_pnl": daily_pnl,
                    "strategy": strategy_name,
                    "currency": currency
                }

    return changes


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Calculate annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(values: pd.Series) -> float:
    """Calculate maximum drawdown."""
    if values.empty:
        return 0
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    return drawdown.min() * 100


# =============================================================================
# TRADING FUNCTIONS
# =============================================================================

def execute_buy(data: dict, ticker: str, shares: float, price: float, strategy: str, currency: str = "USD") -> Tuple[bool, str]:
    """Execute a buy order."""
    fx_rates = fetch_fx_rates()
    fx_rate = fx_rates.get(currency, 1.0)
    cost_usd = shares * price * fx_rate

    if cost_usd > data.get("cash", 0):
        return False, f"Insufficient cash. Need ${cost_usd:,.2f}, have ${data.get('cash', 0):,.2f}"

    data["cash"] = data.get("cash", 0) - cost_usd

    if strategy not in data["strategies"]:
        data["strategies"][strategy] = {"description": "", "holdings": []}

    existing = None
    for h in data["strategies"][strategy]["holdings"]:
        if h["ticker"] == ticker:
            existing = h
            break

    if existing:
        total_shares = existing["shares"] + shares
        total_cost = (existing["shares"] * existing.get("cost_basis", 0)) + (shares * price)
        existing["shares"] = total_shares
        existing["cost_basis"] = total_cost / total_shares
    else:
        data["strategies"][strategy]["holdings"].append({
            "ticker": ticker,
            "shares": shares,
            "cost_basis": price,
            "currency": currency
        })

    data["trades"].append({
        "date": datetime.now().isoformat(),
        "action": "BUY",
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "currency": currency,
        "total_usd": cost_usd,
        "strategy": strategy
    })

    save_portfolio_data(data)
    return True, f"Bought {shares} shares of {ticker} at {currency} {price:.2f}"


def execute_sell(data: dict, ticker: str, shares: float, price: float, strategy: str) -> Tuple[bool, str]:
    """Execute a sell order."""
    if strategy not in data["strategies"]:
        return False, f"Strategy '{strategy}' not found"

    holding = None
    holding_idx = None
    for i, h in enumerate(data["strategies"][strategy]["holdings"]):
        if h["ticker"] == ticker:
            holding = h
            holding_idx = i
            break

    if not holding:
        return False, f"{ticker} not found in {strategy}"

    if shares > holding["shares"]:
        return False, f"Cannot sell {shares} shares. Only have {holding['shares']:.2f}"

    currency = holding.get("currency", "USD")
    fx_rates = fetch_fx_rates()
    fx_rate = fx_rates.get(currency, 1.0)

    proceeds_usd = shares * price * fx_rate
    cost_basis_usd = shares * holding.get("cost_basis", 0) * fx_rate
    realized_pnl = proceeds_usd - cost_basis_usd

    data["cash"] = data.get("cash", 0) + proceeds_usd

    if shares >= holding["shares"]:
        data["strategies"][strategy]["holdings"].pop(holding_idx)
    else:
        holding["shares"] -= shares

    data["trades"].append({
        "date": datetime.now().isoformat(),
        "action": "SELL",
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "currency": currency,
        "total_usd": proceeds_usd,
        "realized_pnl": realized_pnl,
        "strategy": strategy
    })

    save_portfolio_data(data)
    return True, f"Sold {shares} shares of {ticker}. Realized P&L: ${realized_pnl:,.2f}"


# =============================================================================
# UI PAGES
# =============================================================================

def render_sidebar(data: dict):
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## 📊 Portfolio Dashboard")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Daily Summary", "Holdings", "All Positions", "Performance", "Trading", "Trade History"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        tickers = get_all_tickers(data)
        if tickers:
            fx_rates = fetch_fx_rates()
            prices, _ = fetch_prices_with_prev(tickers)
            total_value, _ = calculate_portfolio_value(data, prices, fx_rates)

            st.metric("Portfolio Value", f"${total_value:,.2f}")
            st.metric("Cash", f"${data.get('cash', 0):,.2f}")

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        return page


def render_dashboard(data: dict):
    """Render the main dashboard page."""
    st.markdown("# 📈 Portfolio Dashboard")

    tickers = get_all_tickers(data)
    if not tickers:
        st.warning("No holdings found. Go to Trading to add positions.")
        return

    fx_rates = fetch_fx_rates()
    current_prices, prev_prices = fetch_prices_with_prev(tickers)
    total_value, holdings_detail = calculate_portfolio_value(data, current_prices, fx_rates)

    # Calculate previous day value
    prev_value, _ = calculate_portfolio_value(data, prev_prices, fx_rates)
    prev_value += data.get("cash", 0) - data.get("cash", 0)  # Cash doesn't change daily
    daily_change = total_value - prev_value
    daily_change_pct = (daily_change / prev_value * 100) if prev_value > 0 else 0

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    starting_capital = data.get("starting_capital", 129626)
    total_return = ((total_value - starting_capital) / starting_capital * 100)

    with col1:
        st.metric("Portfolio Value", f"${total_value:,.2f}", f"{daily_change_pct:+.2f}% today")
    with col2:
        st.metric("Cash", f"${data.get('cash', 0):,.2f}", f"{(data.get('cash', 0)/total_value*100):.1f}%")
    with col3:
        holdings_value = sum(h["usd_value"] for h in holdings_detail.values())
        st.metric("Invested", f"${holdings_value:,.2f}")
    with col4:
        total_pnl = sum(h["pnl"] for h in holdings_detail.values())
        st.metric("Unrealized P&L", f"${total_pnl:,.2f}")
    with col5:
        st.metric("Total Return", f"{total_return:+.2f}%", f"from ${starting_capital:,.0f}")

    st.markdown("---")

    # =====================================================
    # RETURNS GRID - DTD, MTD, YTD, All-Time
    # =====================================================
    st.subheader("Returns Overview")

    hist_data = data.get("historical_values", [])
    if hist_data:
        hist_df_returns = pd.DataFrame(hist_data)
        period_returns = calculate_period_returns(data, hist_df_returns, total_value)

        # Create returns grid with all benchmarks
        returns_grid = pd.DataFrame({
            "": ["Your Portfolio", "S&P 500 TR", "S&P 500 EW", "MSCI World", "MSCI World EW"],
            "DTD": [
                f"{period_returns['dtd']['portfolio']:+.2f}%",
                f"{period_returns['dtd']['sp500_tr']:+.2f}%",
                f"{period_returns['dtd']['sp500_ew']:+.2f}%",
                f"{period_returns['dtd']['msci_world']:+.2f}%",
                f"{period_returns['dtd'].get('msci_world_ew', 0):+.2f}%"
            ],
            "MTD": [
                f"{period_returns['mtd']['portfolio']:+.2f}%",
                f"{period_returns['mtd']['sp500_tr']:+.2f}%",
                f"{period_returns['mtd']['sp500_ew']:+.2f}%",
                f"{period_returns['mtd']['msci_world']:+.2f}%",
                f"{period_returns['mtd'].get('msci_world_ew', 0):+.2f}%"
            ],
            "YTD": [
                f"{period_returns['ytd']['portfolio']:+.2f}%",
                f"{period_returns['ytd']['sp500_tr']:+.2f}%",
                f"{period_returns['ytd']['sp500_ew']:+.2f}%",
                f"{period_returns['ytd']['msci_world']:+.2f}%",
                f"{period_returns['ytd'].get('msci_world_ew', 0):+.2f}%"
            ],
            "All-Time*": [
                f"{period_returns['all_time']['portfolio']:+.2f}%",
                f"{period_returns['all_time']['sp500_tr']:+.2f}%",
                f"{period_returns['all_time']['sp500_ew']:+.2f}%",
                f"{period_returns['all_time']['msci_world']:+.2f}%",
                f"{period_returns['all_time'].get('msci_world_ew', 0):+.2f}%"
            ]
        })

        st.dataframe(returns_grid, use_container_width=True, hide_index=True, height=215)
        st.caption("*MSCI World EW All-Time calculated from Oct 2024 (ETF launch)")

    st.markdown("---")

    # Strategy allocation
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Strategy Allocation")

        strategy_values = {}
        for h in holdings_detail.values():
            strat = h["strategy"]
            strategy_values[strat] = strategy_values.get(strat, 0) + h["usd_value"]

        # Add cash
        strategy_values["Cash"] = data.get("cash", 0)

        strat_df = pd.DataFrame([
            {"Strategy": k, "Value": v, "Weight": v/total_value*100}
            for k, v in strategy_values.items()
        ])

        fig = px.pie(strat_df, values="Value", names="Strategy", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Strategy Returns")
        for strat_name, strat_data in data.get("strategies", {}).items():
            strat_holdings = [h for h in holdings_detail.values() if h["strategy"] == strat_name]
            if strat_holdings:
                strat_value = sum(h["usd_value"] for h in strat_holdings)
                strat_cost = sum(h["usd_cost"] for h in strat_holdings)
                strat_return = ((strat_value - strat_cost) / strat_cost * 100) if strat_cost > 0 else 0
                color = "🟢" if strat_return >= 0 else "🔴"
                st.markdown(f"**{strat_name}**: {color} {strat_return:+.2f}%")

    st.markdown("---")

    # Historical performance chart
    st.subheader("Portfolio Performance vs Benchmarks")

    hist_data = data.get("historical_values", [])
    if hist_data:
        hist_df = pd.DataFrame(hist_data)
        hist_df['date'] = pd.to_datetime(hist_df['date'])

        # Normalize to starting value
        start_val = hist_df['portfolio_value'].iloc[0]
        start_sp = hist_df['sp500_tr'].iloc[0]
        start_msci = hist_df['msci_world'].iloc[0]
        start_spew = hist_df['sp500_ew'].iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df['date'], y=(hist_df['portfolio_value']/start_val-1)*100,
                                 name="Your Portfolio", line=dict(color="#3b82f6", width=3)))
        fig.add_trace(go.Scatter(x=hist_df['date'], y=(hist_df['sp500_tr']/start_sp-1)*100,
                                 name="S&P 500 TR", line=dict(color="#10b981", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=hist_df['date'], y=(hist_df['msci_world']/start_msci-1)*100,
                                 name="MSCI World", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=hist_df['date'], y=(hist_df['sp500_ew']/start_spew-1)*100,
                                 name="S&P 500 EW", line=dict(color="#8b5cf6", width=2, dash="dash")))

        # Add MSCI World EW if available (from Oct 2024)
        if 'msci_world_ew' in hist_df.columns:
            msci_ew_data = hist_df[hist_df['msci_world_ew'].notna()]
            if not msci_ew_data.empty:
                start_msci_ew = msci_ew_data['msci_world_ew'].iloc[0]
                fig.add_trace(go.Scatter(
                    x=msci_ew_data['date'],
                    y=(msci_ew_data['msci_world_ew']/start_msci_ew-1)*100,
                    name="MSCI World EW",
                    line=dict(color="#ec4899", width=2, dash="dot")
                ))

        fig.update_layout(
            yaxis_title="Return (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=300)
def fetch_benchmark_prices(start_date: str) -> pd.DataFrame:
    """Fetch benchmark price history."""
    benchmark_tickers = ["^SP500TR", "RSP", "URTH", "ACWI"]  # Added ACWI for MSCI World EW
    try:
        data = yf.download(benchmark_tickers, start=start_date, progress=False)
        if not data.empty:
            return data['Close']
    except:
        pass
    return pd.DataFrame()


def calculate_period_returns(data: dict, hist_df: pd.DataFrame, current_value: float) -> dict:
    """Calculate returns for different time periods."""
    today = datetime.now().date()

    # Get historical data points
    hist_df = hist_df.copy()
    hist_df['date'] = pd.to_datetime(hist_df['date']).dt.date
    hist_df = hist_df.sort_values('date')

    returns = {}

    # Fetch live benchmark prices for accurate DTD/MTD - go back further for MTD
    benchmark_prices = fetch_benchmark_prices((today - timedelta(days=45)).strftime("%Y-%m-%d"))

    # Get current and previous benchmark values from live data
    if not benchmark_prices.empty and len(benchmark_prices) >= 2:
        # Current values (last row)
        curr_sp500 = float(benchmark_prices['^SP500TR'].iloc[-1]) if '^SP500TR' in benchmark_prices.columns else 0
        curr_rsp = float(benchmark_prices['RSP'].iloc[-1]) if 'RSP' in benchmark_prices.columns else 0
        curr_urth = float(benchmark_prices['URTH'].iloc[-1]) if 'URTH' in benchmark_prices.columns else 0
        curr_acwi = float(benchmark_prices['ACWI'].iloc[-1]) if 'ACWI' in benchmark_prices.columns else 0

        # Previous day values (second to last row)
        prev_sp500 = float(benchmark_prices['^SP500TR'].iloc[-2]) if '^SP500TR' in benchmark_prices.columns else 0
        prev_rsp = float(benchmark_prices['RSP'].iloc[-2]) if 'RSP' in benchmark_prices.columns else 0
        prev_urth = float(benchmark_prices['URTH'].iloc[-2]) if 'URTH' in benchmark_prices.columns else 0
        prev_acwi = float(benchmark_prices['ACWI'].iloc[-2]) if 'ACWI' in benchmark_prices.columns else 0

        # DTD from live data
        returns['dtd'] = {
            'sp500_tr': ((curr_sp500 - prev_sp500) / prev_sp500 * 100) if prev_sp500 > 0 else 0,
            'sp500_ew': ((curr_rsp - prev_rsp) / prev_rsp * 100) if prev_rsp > 0 else 0,
            'msci_world': ((curr_urth - prev_urth) / prev_urth * 100) if prev_urth > 0 else 0,
            'msci_world_ew': ((curr_acwi - prev_acwi) / prev_acwi * 100) if prev_acwi > 0 else 0,
        }

        # MTD from live data - find first trading day of month
        month_start = today.replace(day=1)
        # Convert index to date for comparison
        benchmark_prices_copy = benchmark_prices.copy()
        benchmark_prices_copy['_date'] = benchmark_prices_copy.index.date
        month_prices = benchmark_prices_copy[benchmark_prices_copy['_date'] >= month_start]

        if len(month_prices) > 0:
            month_start_sp500 = float(month_prices['^SP500TR'].iloc[0]) if '^SP500TR' in month_prices.columns else curr_sp500
            month_start_rsp = float(month_prices['RSP'].iloc[0]) if 'RSP' in month_prices.columns else curr_rsp
            month_start_urth = float(month_prices['URTH'].iloc[0]) if 'URTH' in month_prices.columns else curr_urth
            month_start_acwi = float(month_prices['ACWI'].iloc[0]) if 'ACWI' in month_prices.columns else curr_acwi
        else:
            month_start_sp500, month_start_rsp, month_start_urth, month_start_acwi = curr_sp500, curr_rsp, curr_urth, curr_acwi

        returns['mtd'] = {
            'sp500_tr': ((curr_sp500 - month_start_sp500) / month_start_sp500 * 100) if month_start_sp500 > 0 else 0,
            'sp500_ew': ((curr_rsp - month_start_rsp) / month_start_rsp * 100) if month_start_rsp > 0 else 0,
            'msci_world': ((curr_urth - month_start_urth) / month_start_urth * 100) if month_start_urth > 0 else 0,
            'msci_world_ew': ((curr_acwi - month_start_acwi) / month_start_acwi * 100) if month_start_acwi > 0 else 0,
        }
    else:
        returns['dtd'] = {'sp500_tr': 0, 'sp500_ew': 0, 'msci_world': 0, 'msci_world_ew': 0}
        returns['mtd'] = {'sp500_tr': 0, 'sp500_ew': 0, 'msci_world': 0, 'msci_world_ew': 0}

    # Portfolio DTD - from last historical entry
    last_hist = hist_df.iloc[-1] if len(hist_df) > 0 else None
    if last_hist is not None:
        returns['dtd']['portfolio'] = ((current_value - last_hist['portfolio_value']) / last_hist['portfolio_value']) * 100
    else:
        returns['dtd']['portfolio'] = 0

    # Portfolio MTD - find closest entry to start of month
    month_start = today.replace(day=1)
    month_data = hist_df[hist_df['date'] <= month_start]
    if len(month_data) > 0:
        month_start_data = month_data.iloc[-1]
        returns['mtd']['portfolio'] = ((current_value - month_start_data['portfolio_value']) / month_start_data['portfolio_value']) * 100
    else:
        returns['mtd']['portfolio'] = returns['dtd']['portfolio']

    # YTD - use Jan 1 entry or closest after
    year_start = today.replace(month=1, day=1)
    # Find entry on or after Jan 1
    ytd_data = hist_df[hist_df['date'] >= year_start]
    if len(ytd_data) > 0:
        year_start_data = ytd_data.iloc[0]  # First entry of the year
    else:
        # Fallback to last entry of previous year
        year_data = hist_df[hist_df['date'] < year_start]
        year_start_data = year_data.iloc[-1] if len(year_data) > 0 else hist_df.iloc[0]

    returns['ytd'] = {
        'portfolio': ((current_value - year_start_data['portfolio_value']) / year_start_data['portfolio_value']) * 100,
        'sp500_tr': ((hist_df.iloc[-1]['sp500_tr'] - year_start_data['sp500_tr']) / year_start_data['sp500_tr']) * 100 if year_start_data['sp500_tr'] > 0 else 0,
        'sp500_ew': ((hist_df.iloc[-1]['sp500_ew'] - year_start_data['sp500_ew']) / year_start_data['sp500_ew']) * 100 if year_start_data['sp500_ew'] > 0 else 0,
        'msci_world': ((hist_df.iloc[-1]['msci_world'] - year_start_data['msci_world']) / year_start_data['msci_world']) * 100 if year_start_data['msci_world'] > 0 else 0,
        'msci_world_ew': ((hist_df.iloc[-1].get('msci_world_ew', 0) - year_start_data.get('msci_world_ew', 0)) / year_start_data.get('msci_world_ew', 1) * 100) if year_start_data.get('msci_world_ew', 0) > 0 else 0,
    }

    # All time (from inception)
    inception_data = hist_df.iloc[0]

    # For MSCI World EW, use first available data point (Oct 2024)
    msci_ew_data = hist_df[hist_df['msci_world_ew'].notna()] if 'msci_world_ew' in hist_df.columns else pd.DataFrame()
    if len(msci_ew_data) > 0:
        msci_ew_inception = msci_ew_data.iloc[0]
        msci_ew_all_time = ((hist_df.iloc[-1]['msci_world_ew'] - msci_ew_inception['msci_world_ew']) / msci_ew_inception['msci_world_ew']) * 100
    else:
        msci_ew_all_time = 0

    returns['all_time'] = {
        'portfolio': ((current_value - inception_data['portfolio_value']) / inception_data['portfolio_value']) * 100,
        'sp500_tr': ((hist_df.iloc[-1]['sp500_tr'] - inception_data['sp500_tr']) / inception_data['sp500_tr']) * 100,
        'sp500_ew': ((hist_df.iloc[-1]['sp500_ew'] - inception_data['sp500_ew']) / inception_data['sp500_ew']) * 100,
        'msci_world': ((hist_df.iloc[-1]['msci_world'] - inception_data['msci_world']) / inception_data['msci_world']) * 100,
        'msci_world_ew': msci_ew_all_time,
    }

    return returns


def calculate_period_attribution(data: dict, hist_df: pd.DataFrame, start_date, end_date, fx_rates: dict) -> dict:
    """Calculate strategy attribution for a specific time period using historical data."""
    tickers = get_all_tickers(data)

    # Fetch price history for the period
    try:
        price_history = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), progress=False)
        if price_history.empty:
            return {}

        if len(tickers) == 1:
            prices = price_history['Close'].to_frame()
            prices.columns = tickers
        else:
            prices = price_history['Close']

        if len(prices) < 2:
            return {}

        start_prices = prices.iloc[0]
        end_prices = prices.iloc[-1]

    except:
        return {}

    # Calculate attribution by strategy
    attribution = {}
    total_start_value = 0
    total_end_value = 0

    for strategy_name, strategy in data.get("strategies", {}).items():
        strat_start = 0
        strat_end = 0
        holdings_attr = []

        for holding in strategy.get("holdings", []):
            ticker = holding["ticker"]
            shares = holding["shares"]
            currency = holding.get("currency", "USD")
            fx_rate = fx_rates.get(currency, 1.0)

            if ticker in start_prices.index and ticker in end_prices.index:
                start_val = shares * start_prices[ticker] * fx_rate
                end_val = shares * end_prices[ticker] * fx_rate

                strat_start += start_val
                strat_end += end_val

                pnl = end_val - start_val
                pct_change = ((end_val - start_val) / start_val * 100) if start_val > 0 else 0

                holdings_attr.append({
                    'ticker': ticker,
                    'start_value': start_val,
                    'end_value': end_val,
                    'pnl': pnl,
                    'pct_change': pct_change
                })

        total_start_value += strat_start
        total_end_value += strat_end

        attribution[strategy_name] = {
            'start_value': strat_start,
            'end_value': strat_end,
            'pnl': strat_end - strat_start,
            'pct_return': ((strat_end - strat_start) / strat_start * 100) if strat_start > 0 else 0,
            'holdings': holdings_attr
        }

    # Calculate attribution percentage (contribution to total return)
    for strat in attribution:
        attribution[strat]['attribution_pct'] = (attribution[strat]['pnl'] / total_start_value * 100) if total_start_value > 0 else 0

    return attribution


def render_daily_summary(data: dict):
    """Render daily summary page with returns grid and period attribution."""
    st.markdown("# 📅 Daily Summary")
    st.markdown(f"**{datetime.now().strftime('%A, %B %d, %Y')}**")

    tickers = get_all_tickers(data)
    if not tickers:
        st.warning("No holdings found.")
        return

    fx_rates = fetch_fx_rates()
    current_prices, prev_prices = fetch_prices_with_prev(tickers)
    total_value, holdings_detail = calculate_portfolio_value(data, current_prices, fx_rates)
    prev_total, _ = calculate_portfolio_value(data, prev_prices, fx_rates)
    daily_changes = calculate_daily_changes(data, current_prices, prev_prices, fx_rates)

    # Daily P&L summary - cleaner metrics
    total_daily_pnl = sum(c["daily_pnl"] for c in daily_changes.values())
    daily_pct = (total_daily_pnl / prev_total * 100) if prev_total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "normal" if total_daily_pnl >= 0 else "inverse"
        st.metric("Today's P&L", f"${total_daily_pnl:,.2f}", f"{daily_pct:+.2f}%", delta_color=color)
    with col2:
        st.metric("Portfolio Value", f"${total_value:,.2f}")
    with col3:
        st.metric("Previous Close", f"${prev_total:,.2f}")
    with col4:
        starting_capital = data.get("starting_capital", 129626)
        total_return = ((total_value - starting_capital) / starting_capital * 100)
        st.metric("Total Return", f"{total_return:+.1f}%")

    st.markdown("---")

    # =====================================================
    # RETURNS GRID - DTD, MTD, YTD, All-Time
    # =====================================================
    st.subheader("Returns Overview")

    hist_data = data.get("historical_values", [])
    if hist_data:
        hist_df = pd.DataFrame(hist_data)
        period_returns = calculate_period_returns(data, hist_df, total_value)

        # Create returns grid with all benchmarks including MSCI World EW
        returns_grid = pd.DataFrame({
            "": ["Your Portfolio", "S&P 500 TR", "S&P 500 EW", "MSCI World", "MSCI World EW"],
            "DTD": [
                f"{period_returns['dtd']['portfolio']:+.2f}%",
                f"{period_returns['dtd']['sp500_tr']:+.2f}%",
                f"{period_returns['dtd']['sp500_ew']:+.2f}%",
                f"{period_returns['dtd']['msci_world']:+.2f}%",
                f"{period_returns['dtd'].get('msci_world_ew', 0):+.2f}%"
            ],
            "MTD": [
                f"{period_returns['mtd']['portfolio']:+.2f}%",
                f"{period_returns['mtd']['sp500_tr']:+.2f}%",
                f"{period_returns['mtd']['sp500_ew']:+.2f}%",
                f"{period_returns['mtd']['msci_world']:+.2f}%",
                f"{period_returns['mtd'].get('msci_world_ew', 0):+.2f}%"
            ],
            "YTD": [
                f"{period_returns['ytd']['portfolio']:+.2f}%",
                f"{period_returns['ytd']['sp500_tr']:+.2f}%",
                f"{period_returns['ytd']['sp500_ew']:+.2f}%",
                f"{period_returns['ytd']['msci_world']:+.2f}%",
                f"{period_returns['ytd'].get('msci_world_ew', 0):+.2f}%"
            ],
            "All-Time*": [
                f"{period_returns['all_time']['portfolio']:+.2f}%",
                f"{period_returns['all_time']['sp500_tr']:+.2f}%",
                f"{period_returns['all_time']['sp500_ew']:+.2f}%",
                f"{period_returns['all_time']['msci_world']:+.2f}%",
                f"{period_returns['all_time'].get('msci_world_ew', 0):+.2f}%"
            ]
        })

        # Style the returns grid with colors
        st.dataframe(returns_grid, use_container_width=True, hide_index=True, height=215)
        st.caption("*MSCI World EW All-Time calculated from Oct 2024 (ETF launch)")

        # Show alpha (outperformance vs S&P 500)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            alpha_dtd = period_returns['dtd']['portfolio'] - period_returns['dtd']['sp500_tr']
            st.metric("DTD Alpha", f"{alpha_dtd:+.2f}%", help="vs S&P 500 TR")
        with col2:
            alpha_mtd = period_returns['mtd']['portfolio'] - period_returns['mtd']['sp500_tr']
            st.metric("MTD Alpha", f"{alpha_mtd:+.2f}%", help="vs S&P 500 TR")
        with col3:
            alpha_ytd = period_returns['ytd']['portfolio'] - period_returns['ytd']['sp500_tr']
            st.metric("YTD Alpha", f"{alpha_ytd:+.2f}%", help="vs S&P 500 TR")
        with col4:
            alpha_all = period_returns['all_time']['portfolio'] - period_returns['all_time']['sp500_tr']
            st.metric("All-Time Alpha", f"{alpha_all:+.2f}%", help="vs S&P 500 TR")

    st.markdown("---")

    # =====================================================
    # PERIOD ATTRIBUTION ANALYSIS
    # =====================================================
    st.subheader("Attribution Analysis")

    # Time period selector
    period_options = {
        "Today (DTD)": 0,
        "Week to Date": 7,
        "Month to Date (MTD)": 30,
        "Quarter to Date": 90,
        "Year to Date (YTD)": 365,
        "2025": "2025",
        "2024": "2024",
        "2023": "2023",
        "All Time": "all"
    }

    selected_period = st.selectbox("Select Time Period", list(period_options.keys()), index=4)
    period_value = period_options[selected_period]

    # Calculate date range
    today = datetime.now().date()
    if period_value == "all":
        start_date = datetime.strptime(data.get("inception_date", "2022-10-10"), "%Y-%m-%d").date()
        end_date = today
    elif isinstance(period_value, str) and period_value.isdigit():
        # Specific year
        year = int(period_value)
        start_date = datetime(year, 1, 1).date()
        if year == today.year:
            end_date = today
        else:
            end_date = datetime(year, 12, 31).date()
    elif period_value == 0:
        # Today only - use daily changes we already have
        start_date = today - timedelta(days=1)
        end_date = today
    else:
        start_date = today - timedelta(days=period_value)
        end_date = today

    st.caption(f"Period: {start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}")

    # For "Today", use the already calculated daily changes
    if period_value == 0:
        strat_changes = {}
        for ticker, change in daily_changes.items():
            strat = change["strategy"]
            if strat not in strat_changes:
                strat_changes[strat] = {"pnl": 0, "holdings": []}
            strat_changes[strat]["pnl"] += change["daily_pnl"]
            strat_changes[strat]["holdings"].append(change)

        for strat in strat_changes:
            strat_changes[strat]["attribution_pct"] = (strat_changes[strat]["pnl"] / prev_total * 100) if prev_total > 0 else 0

        attribution = strat_changes
    else:
        # Calculate period attribution
        attribution = calculate_period_attribution(data, hist_df if hist_data else pd.DataFrame(), start_date, end_date, fx_rates)

    if attribution:
        # Strategy attribution chart
        col1, col2 = st.columns([3, 2])

        with col1:
            attr_data = []
            for strat, info in sorted(attribution.items(), key=lambda x: x[1].get('pnl', 0), reverse=True):
                pnl = info.get('pnl', 0)
                attr_pct = info.get('attribution_pct', 0)
                attr_data.append({
                    "Strategy": strat,
                    "P&L": pnl,
                    "Attribution": attr_pct
                })

            attr_df = pd.DataFrame(attr_data)

            fig = px.bar(
                attr_df, x="Strategy", y="Attribution",
                color="Attribution",
                color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],
                color_continuous_midpoint=0,
                text=attr_df["Attribution"].apply(lambda x: f"{x:+.2f}%")
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                margin=dict(t=20, b=20),
                showlegend=False,
                yaxis_title="Attribution (%)",
                xaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Strategy Breakdown**")
            total_period_pnl = sum(info.get('pnl', 0) for info in attribution.values())
            st.markdown(f"**Total P&L: ${total_period_pnl:,.2f}**")
            st.markdown("")

            for strat, info in sorted(attribution.items(), key=lambda x: x[1].get('pnl', 0), reverse=True):
                pnl = info.get('pnl', 0)
                attr_pct = info.get('attribution_pct', 0)
                icon = "🟢" if pnl >= 0 else "🔴"
                st.markdown(f"{icon} **{strat}**")
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;${pnl:,.2f} ({attr_pct:+.2f}%)")

        # Holdings breakdown for selected period
        st.markdown("---")
        st.subheader("Top Contributors")

        all_holdings = []
        for strat, info in attribution.items():
            for h in info.get('holdings', []):
                h['strategy'] = strat
                all_holdings.append(h)

        if all_holdings:
            # Sort by P&L
            sorted_holdings = sorted(all_holdings, key=lambda x: x.get('pnl', 0), reverse=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top Winners**")
                winners = [h for h in sorted_holdings if h.get('pnl', 0) > 0][:5]
                for h in winners:
                    pnl = h.get('pnl', 0)
                    pct = h.get('pct_change', h.get('daily_change_pct', 0))
                    st.markdown(f"**{h['ticker']}** ({h['strategy']})")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;+${pnl:,.2f} ({pct:+.2f}%)")

            with col2:
                st.markdown("**Top Losers**")
                losers = [h for h in sorted_holdings if h.get('pnl', 0) < 0]
                losers = sorted(losers, key=lambda x: x.get('pnl', 0))[:5]
                for h in losers:
                    pnl = h.get('pnl', 0)
                    pct = h.get('pct_change', h.get('daily_change_pct', 0))
                    st.markdown(f"**{h['ticker']}** ({h['strategy']})")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;${pnl:,.2f} ({pct:+.2f}%)")

    st.markdown("---")

    # All positions daily change table - cleaner format
    st.subheader("All Positions - Today's Change")

    sorted_changes = sorted(daily_changes.values(), key=lambda x: x["daily_pnl"], reverse=True)

    all_changes = pd.DataFrame([
        {
            "Ticker": c["ticker"],
            "Strategy": c["strategy"],
            "Shares": f"{c['shares']:.2f}",
            "Change %": f"{c['daily_change_pct']:+.2f}%",
            "Daily P&L": f"${c['daily_pnl']:,.2f}"
        }
        for c in sorted_changes
    ])

    row_height = 35
    header_height = 38
    table_height = header_height + (len(all_changes) * row_height) + 10
    st.dataframe(all_changes, use_container_width=True, hide_index=True, height=table_height)


def render_all_positions(data: dict):
    """Render all positions chart."""
    st.markdown("# 📊 All Positions")

    tickers = get_all_tickers(data)
    if not tickers:
        st.warning("No holdings found.")
        return

    fx_rates = fetch_fx_rates()
    prices, _ = fetch_prices_with_prev(tickers)
    total_value, holdings_detail = calculate_portfolio_value(data, prices, fx_rates)

    # Create treemap data
    treemap_data = []
    for h in holdings_detail.values():
        treemap_data.append({
            "Ticker": h["ticker"],
            "Strategy": h["strategy"],
            "Value": h["usd_value"],
            "P&L %": h["pnl_pct"],
            "P&L": h["pnl"]
        })

    # Add cash
    treemap_data.append({
        "Ticker": "CASH",
        "Strategy": "Cash",
        "Value": data.get("cash", 0),
        "P&L %": 0,
        "P&L": 0
    })

    df = pd.DataFrame(treemap_data)

    # Treemap
    st.subheader("Portfolio Treemap")
    fig = px.treemap(
        df,
        path=["Strategy", "Ticker"],
        values="Value",
        color="P&L %",
        color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],
        color_continuous_midpoint=0,
        hover_data={"Value": ":$,.0f", "P&L": ":$,.0f", "P&L %": ":.2f%"}
    )
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Bar chart of all positions
    st.subheader("Position Values")

    df_sorted = df[df["Ticker"] != "CASH"].sort_values("Value", ascending=True)

    fig = px.bar(
        df_sorted,
        y="Ticker",
        x="Value",
        color="Strategy",
        orientation="h",
        text=df_sorted["Value"].apply(lambda x: f"${x:,.0f}")
    )
    fig.update_layout(
        height=max(400, len(df_sorted) * 25),
        margin=dict(t=30),
        yaxis_title="",
        xaxis_title="Value (USD)"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed table - NO SCROLL
    st.subheader("Position Details")

    detail_df = pd.DataFrame([
        {
            "Ticker": h["ticker"],
            "Strategy": h["strategy"],
            "Shares": f"{h['shares']:.2f}",
            "Price (USD)": f"${h['usd_price']:.2f}",
            "Value": f"${h['usd_value']:,.2f}",
            "Cost Basis": f"${h['cost_basis']:.2f}",
            "P&L": f"${h['pnl']:,.2f}",
            "Return": f"{h['pnl_pct']:+.2f}%",
            "Currency": h["currency"],
            "Weight": f"{(h['usd_value']/total_value*100):.2f}%"
        }
        for h in sorted(holdings_detail.values(), key=lambda x: x["usd_value"], reverse=True)
    ])

    # Display without scroll
    row_height = 35
    header_height = 38
    table_height = header_height + (len(detail_df) * row_height) + 10
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=table_height)


def render_holdings(data: dict):
    """Render holdings by strategy."""
    st.markdown("# 📋 Holdings by Strategy")

    tickers = get_all_tickers(data)
    if not tickers:
        st.warning("No holdings found.")
        return

    fx_rates = fetch_fx_rates()
    prices, _ = fetch_prices_with_prev(tickers)
    total_value, holdings_detail = calculate_portfolio_value(data, prices, fx_rates)

    for strategy_name, strategy in data.get("strategies", {}).items():
        strat_holdings = [h for h in holdings_detail.values() if h["strategy"] == strategy_name]
        if not strat_holdings:
            continue

        strat_value = sum(h["usd_value"] for h in strat_holdings)
        strat_cost = sum(h["usd_cost"] for h in strat_holdings)
        strat_pnl = strat_value - strat_cost
        strat_return = ((strat_pnl / strat_cost) * 100) if strat_cost > 0 else 0

        st.subheader(f"📁 {strategy_name}")
        st.caption(strategy.get("description", ""))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Value", f"${strat_value:,.2f}")
        with col2:
            st.metric("Weight", f"{(strat_value/total_value*100):.1f}%")
        with col3:
            st.metric("P&L", f"${strat_pnl:,.2f}")
        with col4:
            st.metric("Return", f"{strat_return:+.2f}%")

        rows = []
        for h in strat_holdings:
            rows.append({
                "Ticker": h["ticker"],
                "Shares": f"{h['shares']:.2f}",
                "Price": f"${h['usd_price']:.2f}",
                "Value": f"${h['usd_value']:,.2f}",
                "Cost": f"${h['cost_basis']:.2f}",
                "P&L": f"${h['pnl']:,.2f}",
                "Return": f"{h['pnl_pct']:+.2f}%",
                "Currency": h["currency"]
            })

        rows_df = pd.DataFrame(rows)
        row_height = 35
        header_height = 38
        table_height = header_height + (len(rows_df) * row_height) + 10
        st.dataframe(rows_df, use_container_width=True, hide_index=True, height=table_height)
        st.markdown("---")


def render_performance(data: dict):
    """Render performance analytics."""
    st.markdown("# 📊 Performance Analytics")

    hist_data = data.get("historical_values", [])
    if not hist_data:
        st.warning("No historical data available.")
        return

    hist_df = pd.DataFrame(hist_data)
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')

    # Calculate returns
    hist_df['portfolio_return'] = hist_df['portfolio_value'].pct_change()
    hist_df['sp500_return'] = hist_df['sp500_tr'].pct_change()
    hist_df['msci_return'] = hist_df['msci_world'].pct_change()
    hist_df['spew_return'] = hist_df['sp500_ew'].pct_change()
    if 'msci_world_ew' in hist_df.columns:
        hist_df['msci_ew_return'] = hist_df['msci_world_ew'].pct_change()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    total_return = ((hist_df['portfolio_value'].iloc[-1] / hist_df['portfolio_value'].iloc[0]) - 1) * 100
    sp_return = ((hist_df['sp500_tr'].iloc[-1] / hist_df['sp500_tr'].iloc[0]) - 1) * 100

    portfolio_sharpe = calculate_sharpe_ratio(hist_df['portfolio_return'].dropna())
    max_dd = calculate_max_drawdown(hist_df['portfolio_value'])
    volatility = hist_df['portfolio_return'].std() * np.sqrt(12) * 100  # Monthly to annual

    with col1:
        st.metric("Total Return", f"{total_return:+.2f}%", f"vs S&P: {total_return - sp_return:+.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
    with col3:
        st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
    with col4:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")

    st.markdown("---")

    # Cumulative returns chart
    st.subheader("Cumulative Returns")

    # Normalize to 100
    fig = go.Figure()
    base = hist_df['portfolio_value'].iloc[0]
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=(hist_df['portfolio_value'] / base) * 100,
        name="Your Portfolio",
        line=dict(color="#3b82f6", width=3)
    ))

    base_sp = hist_df['sp500_tr'].iloc[0]
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=(hist_df['sp500_tr'] / base_sp) * 100,
        name="S&P 500 TR",
        line=dict(color="#10b981", width=2, dash="dash")
    ))

    base_msci = hist_df['msci_world'].iloc[0]
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=(hist_df['msci_world'] / base_msci) * 100,
        name="MSCI World",
        line=dict(color="#f59e0b", width=2, dash="dash")
    ))

    base_spew = hist_df['sp500_ew'].iloc[0]
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=(hist_df['sp500_ew'] / base_spew) * 100,
        name="S&P 500 EW",
        line=dict(color="#8b5cf6", width=2, dash="dash")
    ))

    # Add MSCI World EW if available
    if 'msci_world_ew' in hist_df.columns:
        msci_ew_data = hist_df[hist_df['msci_world_ew'].notna()]
        if not msci_ew_data.empty:
            base_msci_ew = msci_ew_data['msci_world_ew'].iloc[0]
            fig.add_trace(go.Scatter(
                x=msci_ew_data['date'],
                y=(msci_ew_data['msci_world_ew'] / base_msci_ew) * 100,
                name="MSCI World EW",
                line=dict(color="#ec4899", width=2, dash="dot")
            ))

    fig.update_layout(
        yaxis_title="Value (Indexed to 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Comparison table
    st.subheader("Benchmark Comparison")

    sp_sharpe = calculate_sharpe_ratio(hist_df['sp500_return'].dropna())
    msci_sharpe = calculate_sharpe_ratio(hist_df['msci_return'].dropna())
    spew_sharpe = calculate_sharpe_ratio(hist_df['spew_return'].dropna())

    comparison_data = [
        {
            "": "Your Portfolio",
            "Total Return": f"{total_return:+.2f}%",
            "Sharpe Ratio": f"{portfolio_sharpe:.2f}",
            "Volatility": f"{volatility:.2f}%",
            "Max Drawdown": f"{max_dd:.2f}%"
        },
        {
            "": "S&P 500 TR",
            "Total Return": f"{sp_return:+.2f}%",
            "Sharpe Ratio": f"{sp_sharpe:.2f}",
            "Volatility": f"{hist_df['sp500_return'].std() * np.sqrt(12) * 100:.2f}%",
            "Max Drawdown": f"{calculate_max_drawdown(hist_df['sp500_tr']):.2f}%"
        },
        {
            "": "S&P 500 EW",
            "Total Return": f"{((hist_df['sp500_ew'].iloc[-1] / hist_df['sp500_ew'].iloc[0]) - 1) * 100:+.2f}%",
            "Sharpe Ratio": f"{spew_sharpe:.2f}",
            "Volatility": f"{hist_df['spew_return'].std() * np.sqrt(12) * 100:.2f}%",
            "Max Drawdown": f"{calculate_max_drawdown(hist_df['sp500_ew']):.2f}%"
        },
        {
            "": "MSCI World",
            "Total Return": f"{((hist_df['msci_world'].iloc[-1] / hist_df['msci_world'].iloc[0]) - 1) * 100:+.2f}%",
            "Sharpe Ratio": f"{msci_sharpe:.2f}",
            "Volatility": f"{hist_df['msci_return'].std() * np.sqrt(12) * 100:.2f}%",
            "Max Drawdown": f"{calculate_max_drawdown(hist_df['msci_world']):.2f}%"
        }
    ]

    # Add MSCI World EW if available
    if 'msci_world_ew' in hist_df.columns:
        msci_ew_data = hist_df[hist_df['msci_world_ew'].notna()]
        if len(msci_ew_data) > 1:
            msci_ew_return = ((msci_ew_data['msci_world_ew'].iloc[-1] / msci_ew_data['msci_world_ew'].iloc[0]) - 1) * 100
            msci_ew_sharpe = calculate_sharpe_ratio(hist_df['msci_ew_return'].dropna())
            comparison_data.append({
                "": "MSCI World EW",
                "Total Return": f"{msci_ew_return:+.2f}%",
                "Sharpe Ratio": f"{msci_ew_sharpe:.2f}",
                "Volatility": f"{hist_df['msci_ew_return'].std() * np.sqrt(12) * 100:.2f}%",
                "Max Drawdown": f"{calculate_max_drawdown(msci_ew_data['msci_world_ew']):.2f}%"
            })

    comparison = pd.DataFrame(comparison_data)
    st.dataframe(comparison, use_container_width=True, hide_index=True)


def render_trading(data: dict):
    """Render trading page."""
    st.markdown("# 💹 Trading")

    tab1, tab2, tab3 = st.tabs(["Buy", "Sell", "Cash"])

    with tab1:
        st.subheader("Buy Stock")
        col1, col2 = st.columns(2)

        with col1:
            buy_ticker = st.text_input("Ticker", placeholder="e.g., AAPL").upper()
            buy_shares = st.number_input("Shares", min_value=0.0, step=1.0, key="buy_shares")
            buy_currency = st.selectbox("Currency", ["USD", "EUR", "HKD", "DKK"], key="buy_currency")

            strategies = list(data.get("strategies", {}).keys())
            if strategies:
                buy_strategy = st.selectbox("Strategy", strategies)
            else:
                buy_strategy = st.text_input("New Strategy Name")

        with col2:
            if buy_ticker:
                try:
                    stock = yf.Ticker(buy_ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        st.metric("Current Price", f"{buy_currency} {current_price:.2f}")
                        buy_price = st.number_input("Buy Price", value=current_price, min_value=0.01, step=0.01)

                        fx_rates = fetch_fx_rates()
                        total_usd = buy_shares * buy_price * fx_rates.get(buy_currency, 1.0)
                        st.metric("Total (USD)", f"${total_usd:,.2f}")
                        st.metric("Available Cash", f"${data.get('cash', 0):,.2f}")
                    else:
                        buy_price = st.number_input("Buy Price", min_value=0.01, step=0.01)
                except:
                    buy_price = st.number_input("Buy Price", min_value=0.01, step=0.01)
            else:
                buy_price = 0

        if st.button("Execute Buy", type="primary"):
            if buy_ticker and buy_shares > 0 and buy_price > 0 and buy_strategy:
                success, msg = execute_buy(data, buy_ticker, buy_shares, buy_price, buy_strategy, buy_currency)
                if success:
                    st.success(msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)

    with tab2:
        st.subheader("Sell Stock")

        holdings_list = []
        for strat_name, strat in data.get("strategies", {}).items():
            for h in strat.get("holdings", []):
                holdings_list.append({"ticker": h["ticker"], "shares": h["shares"], "strategy": strat_name})

        if not holdings_list:
            st.info("No holdings to sell")
        else:
            col1, col2 = st.columns(2)

            with col1:
                options = [f"{h['ticker']} ({h['strategy']})" for h in holdings_list]
                selected = st.selectbox("Select Holding", options)

                if selected:
                    idx = options.index(selected)
                    holding = holdings_list[idx]
                    st.info(f"Available: {holding['shares']:.4f} shares")
                    sell_shares = st.number_input("Shares to Sell", min_value=0.0, max_value=float(holding['shares']), step=1.0)

            with col2:
                if selected:
                    try:
                        stock = yf.Ticker(holding['ticker'])
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                            st.metric("Current Price", f"${current_price:.2f}")
                            sell_price = st.number_input("Sell Price", value=current_price, min_value=0.01, step=0.01)
                        else:
                            sell_price = st.number_input("Sell Price", min_value=0.01, step=0.01)
                    except:
                        sell_price = st.number_input("Sell Price", min_value=0.01, step=0.01)

            if st.button("Execute Sell", type="primary"):
                if selected and sell_shares > 0:
                    success, msg = execute_sell(data, holding['ticker'], sell_shares, sell_price, holding['strategy'])
                    if success:
                        st.success(msg)
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(msg)

    with tab3:
        st.subheader("Cash Management")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Current Cash", f"${data.get('cash', 0):,.2f}")

        with col2:
            action = st.radio("Action", ["Deposit", "Withdraw"], horizontal=True)
            amount = st.number_input("Amount", min_value=0.0, step=100.0)

            if st.button("Submit"):
                if amount > 0:
                    if action == "Withdraw" and amount > data.get("cash", 0):
                        st.error("Insufficient cash")
                    else:
                        data["cash"] = data.get("cash", 0) + (amount if action == "Deposit" else -amount)
                        data["trades"].append({
                            "date": datetime.now().isoformat(),
                            "action": action.upper(),
                            "ticker": "CASH",
                            "total_usd": amount if action == "Deposit" else -amount
                        })
                        save_portfolio_data(data)
                        st.success(f"{'Added' if action == 'Deposit' else 'Withdrew'} ${amount:,.2f}")
                        st.rerun()


def render_trade_history(data: dict):
    """Render trade history."""
    st.markdown("# 📜 Trade History")

    trades = data.get("trades", [])
    if not trades:
        st.info("No trades recorded")
        return

    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')

    row_height = 35
    header_height = 38
    table_height = header_height + (len(df) * row_height) + 10
    st.dataframe(df, use_container_width=True, hide_index=True, height=table_height)

    # Summary
    buy_trades = [t for t in trades if t.get('action') == 'BUY']
    sell_trades = [t for t in trades if t.get('action') == 'SELL']
    realized_pnl = sum(t.get('realized_pnl', 0) for t in sell_trades)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", len(trades))
    with col2:
        st.metric("Buy Orders", len(buy_trades))
    with col3:
        st.metric("Realized P&L", f"${realized_pnl:,.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    data = load_portfolio_data()
    page = render_sidebar(data)

    if page == "Dashboard":
        render_dashboard(data)
    elif page == "Daily Summary":
        render_daily_summary(data)
    elif page == "Holdings":
        render_holdings(data)
    elif page == "All Positions":
        render_all_positions(data)
    elif page == "Performance":
        render_performance(data)
    elif page == "Trading":
        render_trading(data)
    elif page == "Trade History":
        render_trade_history(data)


if __name__ == "__main__":
    main()
