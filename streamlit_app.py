from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app.main import (
    POPULAR_STOCKS,
    build_price_predictions,
    fetch_yahoo_history,
    fetch_yahoo_quote,
    fetch_yahoo_search,
    normalize_symbol,
)


st.set_page_config(
    page_title="Indian Stock Predictor",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at 8% 18%, rgba(159, 216, 247, .45), transparent 28%),
          linear-gradient(135deg, #7fbee0 0%, #69abd2 45%, #3f86ae 100%);
        color: #f7f8ff;
      }
      .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
      }
      [data-testid="stMetric"] {
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 24px;
        padding: 18px;
        background: linear-gradient(180deg, rgba(45, 52, 72, .95), rgba(21, 25, 37, .96));
        box-shadow: 0 24px 60px rgba(23, 42, 58, .22);
      }
      [data-testid="stMetricLabel"] {
        color: #aeb5c7;
      }
      .dark-card {
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 32px;
        padding: 24px;
        background: linear-gradient(180deg, rgba(45, 52, 72, .96), rgba(21, 25, 37, .96));
        box-shadow: 0 24px 60px rgba(23, 42, 58, .22);
      }
      .stock-chip {
        display: inline-block;
        margin: 0 8px 8px 0;
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 999px;
        padding: 8px 12px;
        background: rgba(21, 25, 37, .7);
        color: #f7f8ff;
        font-weight: 700;
      }
      .muted {
        color: #aeb5c7;
      }
      .prediction-card {
        border-radius: 24px;
        padding: 18px;
        background: rgba(15, 19, 30, .76);
        border: 1px solid rgba(255, 255, 255, .12);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value: float | None, currency: str = "INR") -> str:
    if value is None:
        return "--"
    symbol = "₹" if currency == "INR" else currency
    return f"{symbol}{value:,.2f}"


def percent(value: float | None) -> str:
    if value is None:
        return "--"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


@st.cache_data(ttl=300, show_spinner=False)
def cached_search(query: str) -> list[dict]:
    if not query.strip():
        return POPULAR_STOCKS
    return fetch_yahoo_search(query)


@st.cache_data(ttl=60, show_spinner=False)
def cached_quote(symbol: str) -> dict:
    return fetch_yahoo_quote(normalize_symbol(symbol))


@st.cache_data(ttl=300, show_spinner=False)
def cached_history(symbol: str) -> dict:
    return fetch_yahoo_history(normalize_symbol(symbol))


@st.cache_data(ttl=900, show_spinner=False)
def cached_predictions(symbol: str) -> dict:
    history = fetch_yahoo_history(normalize_symbol(symbol), range_value="5y", range_label="Last 5 years")
    return build_price_predictions(history)


def history_frame(history: dict) -> pd.DataFrame:
    rows = history["points"]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")


st.title("Indian Stock Price Predictor")
st.caption("Search NSE/BSE stocks, view live price analytics, and train ML forecasts for 5, 15, and 30 trading days.")

with st.container():
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    search_query = st.text_input(
        "Search Indian stock",
        placeholder="Try Reliance, TCS, INFY, SBIN...",
        help="Leave blank and click Search to see the default Indian stock list.",
    )
    search_clicked = st.button("Search Stocks", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if search_clicked or "matches" not in st.session_state:
    with st.spinner("Loading stock list..."):
        st.session_state.matches = cached_search(search_query)

matches = st.session_state.matches
if not matches:
    st.warning("No Indian stock matches found.")
    st.stop()

options = {
    f"{item.get('name', item['symbol'])} ({item['symbol']})": item["symbol"]
    for item in matches
}

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.subheader("All Stocks")
    st.caption("Select a stock from the search results.")
    selected_label = st.selectbox("Stock", list(options.keys()), label_visibility="collapsed")
    selected_symbol = options[selected_label]
    st.markdown(
        "".join(
            f'<span class="stock-chip">{item["symbol"]}</span>'
            for item in matches[:16]
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with st.spinner(f"Fetching data for {selected_symbol}..."):
    quote = cached_quote(selected_symbol)
    history = cached_history(selected_symbol)

currency = quote.get("currency", "INR")
analytics = history["analytics"]
hist_df = history_frame(history)

with right:
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.subheader(quote.get("short_name") or quote["symbol"])
    st.caption(f"{quote['symbol']} · {quote.get('exchange') or 'NSE/BSE'} · Yahoo Finance")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Current Price", money(quote["current_price"], currency), percent(quote.get("change_percent")))
    metric_cols[1].metric("Previous Close", money(quote.get("previous_close"), currency))
    metric_cols[2].metric("6M Return", percent(analytics["period_return_percent"]))
    st.line_chart(hist_df["close"], height=260)
    st.markdown("</div>", unsafe_allow_html=True)

analytics_cols = st.columns(4)
analytics_cols[0].metric("Trend", analytics["trend"])
analytics_cols[1].metric("Volatility", percent(analytics["annualized_volatility_percent"]))
analytics_cols[2].metric("History Low", money(analytics["period_low"], currency))
analytics_cols[3].metric("History High", money(analytics["period_high"], currency))

st.markdown('<div class="dark-card">', unsafe_allow_html=True)
st.subheader("ML Price Prediction")
st.caption("The app trains Random Forest, Extra Trees, and Gradient Boosting models, then chooses the lowest validation error for each horizon.")

if st.button("Train Prediction Model", type="primary", use_container_width=True):
    with st.spinner("Training models for 5, 15, and 30 trading days..."):
        st.session_state.predictions = cached_predictions(selected_symbol)

prediction_result = st.session_state.get("predictions")
if prediction_result and prediction_result["symbol"] == normalize_symbol(selected_symbol):
    prediction_cols = st.columns(3)
    for column, item in zip(prediction_cols, prediction_result["predictions"]):
        with column:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.metric(
                f"{item['horizon_days']} Days",
                money(item["predicted_price"], prediction_result["currency"]),
                percent(item["change_percent"]),
            )
            st.caption(f"{item['model']} · {item['confidence']} confidence")
            st.caption(f"Validation error: {item['validation_mape_percent']:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Click Train Prediction Model to generate 5, 15, and 30 day forecasts.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="dark-card">', unsafe_allow_html=True)
st.subheader("Validate Your Expected Price")
validate_cols = st.columns([1, 1, 1])
expected_price = validate_cols[0].number_input("Expected Price", min_value=0.0, step=0.1)
tolerance = validate_cols[1].number_input("Tolerance %", min_value=0.0, max_value=100.0, value=0.5, step=0.1)

if validate_cols[2].button("Validate", use_container_width=True):
    current_price = float(quote["current_price"])
    difference = expected_price - current_price
    difference_percent = abs(difference) / current_price * 100
    if difference_percent <= tolerance:
        st.success(f"Valid. Your price is within {tolerance:.2f}% of live market price.")
    else:
        st.error(f"Outside tolerance. Difference is {difference_percent:.4f}% from market.")
st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Predictions are statistical estimates from historical data, not financial advice. "
    f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
)
