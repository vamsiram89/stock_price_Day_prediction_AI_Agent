from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app.main import (
    POPULAR_STOCKS,
    PREDICTION_HORIZONS,
    build_price_predictions,
    fetch_yahoo_history,
    fetch_yahoo_quote,
    fetch_yahoo_search,
    normalize_symbol,
    scan_under_500_stocks,
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
def cached_predictions(symbol: str, prediction_version: str = "horizons-5-15-30-v3") -> dict:
    # prediction_version is part of Streamlit's cache key so older 5/15/30-day
    # results cannot be reused after changing the forecast horizons.
    history = fetch_yahoo_history(normalize_symbol(symbol), range_value="5y", range_label="Last 5 years")
    return build_price_predictions(history)



@st.cache_data(ttl=1800, show_spinner=False)
def cached_under_500_scan(scan_version: str = "nse-under-500-score-v1") -> dict:
    return scan_under_500_stocks(max_price=500.0)

def history_frame(history: dict) -> pd.DataFrame:
    rows = history["points"]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")



st.markdown('<div class="eyebrow">AI-powered market research · NSE watchlist</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Indian Equity Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Discover NSE stocks trading at ₹500 or below, compare technical strength, and explore machine-learning forecasts.</div>', unsafe_allow_html=True)

scan_col, scan_info_col = st.columns([1, 2.2], vertical_alignment="center")
with scan_col:
    run_scan = st.button("⚡ Scan stocks under ₹500", type="primary", use_container_width=True)
with scan_info_col:
    st.caption("Live-price eligibility · 6-month trend and volatility · average volume · transparent 0–100 screening score")

if run_scan:
    try:
        with st.spinner("Screening the NSE watchlist with the latest available market data..."):
            st.session_state["under500_scan_results"] = cached_under_500_scan()
        st.success(f"Scan complete. {st.session_state['under500_scan_results']['eligible_count']} stocks currently meet the ₹500 price filter.")
    except Exception as exc:
        st.error(f"Stock scan could not be completed: {exc}")

scan_result = st.session_state.get("under500_scan_results")
if scan_result:
    st.markdown("### Under-₹500 opportunities")
    st.caption(f"Scanned {scan_result['scanned_universe']} NSE symbols · {scan_result['eligible_count']} eligible · Data: {scan_result['provider']} · Score is a screening aid, not a buy recommendation.")
    if scan_result["stocks"]:
        top_cols = st.columns(3)
        for col, item in zip(top_cols, scan_result["stocks"][:3]):
            with col:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"**{item['name']}**")
                st.metric("Last available price", money(item["current_price"]), f"Score {item['screen_score']:.1f}/100")
                st.caption(f"{item['symbol']} · {item['sector']} · {item['trend']}")
                st.markdown("</div>", unsafe_allow_html=True)
        scan_frame = pd.DataFrame([
            {
                "Company": item["name"], "Symbol": item["symbol"], "Sector": item["sector"],
                "Price (₹)": item["current_price"], "6M return": f"{item['six_month_return_percent']:+.2f}%",
                "Annual volatility": f"{item['annualized_volatility_percent']:.2f}%",
                "Avg. daily volume": item["average_daily_volume"], "Screen score": item["screen_score"],
                "Assessment": item["screen_label"],
            }
            for item in scan_result["stocks"]
        ])
        st.dataframe(scan_frame, use_container_width=True, hide_index=True)
        score_frame = pd.DataFrame([
            {"Stock": item["symbol"].replace(".NS", ""), "Score": item["screen_score"]}
            for item in scan_result["stocks"][:10]
        ]).set_index("Stock")
        st.markdown("#### Top screening scores")
        st.bar_chart(score_frame, use_container_width=True)
        st.info("To view a stock's detailed chart and 5/15/30-day forecasts, select its symbol in the stock selector below. Scan results are cached for 30 minutes.")
    else:
        st.warning("No stocks in the selected NSE watchlist are currently priced at ₹500 or below. Try scanning again later.")
    if scan_result.get("failed_symbols"):
        st.caption(f"Market data was unavailable for {len(scan_result['failed_symbols'])} symbols during this scan.")


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
if scan_result and scan_result.get("stocks"):
    combined = {item["symbol"]: item for item in matches}
    for item in scan_result["stocks"]:
        combined[item["symbol"]] = item
    matches = list(combined.values())
    st.session_state.matches = matches

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
st.caption("Generate separate 5, 15, and 30 trading-day forecasts. For each horizon, the app compares Random Forest, Extra Trees, and Gradient Boosting using a time-ordered validation split.")

prediction_state_key = f"predictions_{normalize_symbol(selected_symbol)}"

if st.button("Generate 5 / 15 / 30 Day Predictions", type="primary", use_container_width=True):
    st.session_state.pop(prediction_state_key, None)
    try:
        with st.spinner("Training and validating the three prediction horizons..."):
            trained_result = cached_predictions(selected_symbol, prediction_version="horizons-5-15-30-v3")
        actual_horizons = [item.get("horizon_days") for item in trained_result.get("predictions", [])]
        expected_horizons = list(PREDICTION_HORIZONS)
        if actual_horizons != expected_horizons:
            raise ValueError(
                f"Unexpected prediction horizons returned: {actual_horizons}. "
                f"Expected: {expected_horizons}."
            )
        st.session_state[prediction_state_key] = trained_result
        st.success("All three prediction horizons were generated successfully.")
    except Exception as exc:
        st.session_state.pop(prediction_state_key, None)
        st.error(f"Prediction could not be generated: {exc}")

prediction_result = st.session_state.get(prediction_state_key)
if prediction_result:
    prediction_items = prediction_result["predictions"]
    prediction_cols = st.columns(3)
    for column, item in zip(prediction_cols, prediction_items):
        with column:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.metric(
                f"{item['horizon_days']} Trading Days",
                money(item["predicted_price"], prediction_result["currency"]),
                percent(item["change_percent"]),
            )
            st.caption(f"Selected model: {item['model']}")
            st.caption(f"Validation MAPE: {item['validation_mape_percent']:.2f}%")
            st.caption(f"Validation label: {item['confidence']} (not a probability)")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Forecast comparison")
    comparison = pd.DataFrame(
        [
            {
                "Horizon": f"{item['horizon_days']} trading days",
                "Predicted price": money(item["predicted_price"], prediction_result["currency"]),
                "Expected change": percent(item["change_percent"]),
                "Selected model": item["model"],
                "Validation MAPE": f"{item['validation_mape_percent']:.2f}%",
                "Error label": item["confidence"],
            }
            for item in prediction_items
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    chart_rows = [
        {"Forecast point": "Current", "Estimated price": prediction_result["current_price"]}
    ] + [
        {
            "Forecast point": f"+{item['horizon_days']} days",
            "Estimated price": item["predicted_price"],
        }
        for item in prediction_items
    ]
    forecast_chart = pd.DataFrame(chart_rows).set_index("Forecast point")
    st.markdown("#### Current price vs. model forecasts")
    st.line_chart(forecast_chart, use_container_width=True)

    st.caption(
        f"Model trained using {prediction_result['trained_on_sessions']:,} historical trading sessions. "
        "Validation MAPE is measured on a historical holdout period; it is not a guarantee of future accuracy."
    )
else:
    st.info(
        "Select a stock, then click **Generate 5 / 15 / 30 Day Predictions**. "
        "The three forecast cards, comparison table, and chart will appear here."
    )
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
