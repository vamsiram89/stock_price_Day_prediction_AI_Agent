from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
import json

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

PREDICTION_HORIZONS = (5, 15, 30)

POPULAR_STOCKS = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Technology"},
    {"symbol": "INFY.NS", "name": "Infosys", "sector": "Technology"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"},
    {"symbol": "ITC.NS", "name": "ITC", "sector": "Consumer"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "Consumer"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Infrastructure"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India", "sector": "Auto"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Auto"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical", "sector": "Pharma"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "Technology"},
    {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corp", "sector": "Energy"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financials"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Financials"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Consumer"},
    {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Consumer"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Materials"},
    {"symbol": "NESTLEIND.NS", "name": "Nestle India", "sector": "Consumer"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corp", "sector": "Utilities"},
    {"symbol": "NTPC.NS", "name": "NTPC", "sector": "Utilities"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Energy"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Materials"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Materials"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "Technology"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "sector": "Technology"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Diversified"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports", "sector": "Infrastructure"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "sector": "Materials"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "sector": "Pharma"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Labs", "sector": "Pharma"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Pharma"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Auto"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "sector": "Auto"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Auto"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "sector": "Auto"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Materials"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "Consumer"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "sector": "Healthcare"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Banking"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy"},
    {"symbol": "IOC.NS", "name": "Indian Oil Corp", "sector": "Energy"},
    {"symbol": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "sector": "Insurance"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Insurance"},
    {"symbol": "SHRIRAMFIN.NS", "name": "Shriram Finance", "sector": "Financials"},
]



# A liquid, diversified NSE watchlist to screen dynamically. Eligibility is always
# determined from the latest downloaded market price, never from a hard-coded price.
UNDER_500_UNIVERSE = [
    {"symbol": "ITC.NS", "name": "ITC", "sector": "Consumer"},
    {"symbol": "ONGC.NS", "name": "ONGC", "sector": "Energy"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid", "sector": "Utilities"},
    {"symbol": "NTPC.NS", "name": "NTPC", "sector": "Utilities"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Energy"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Metals"},
    {"symbol": "IOC.NS", "name": "Indian Oil", "sector": "Energy"},
    {"symbol": "BPCL.NS", "name": "BPCL", "sector": "Energy"},
    {"symbol": "PNB.NS", "name": "Punjab National Bank", "sector": "Banking"},
    {"symbol": "BANKBARODA.NS", "name": "Bank of Baroda", "sector": "Banking"},
    {"symbol": "CANBK.NS", "name": "Canara Bank", "sector": "Banking"},
    {"symbol": "UNIONBANK.NS", "name": "Union Bank", "sector": "Banking"},
    {"symbol": "IDFCFIRSTB.NS", "name": "IDFC First Bank", "sector": "Banking"},
    {"symbol": "FEDERALBNK.NS", "name": "Federal Bank", "sector": "Banking"},
    {"symbol": "YESBANK.NS", "name": "Yes Bank", "sector": "Banking"},
    {"symbol": "IRFC.NS", "name": "IRFC", "sector": "Financials"},
    {"symbol": "NHPC.NS", "name": "NHPC", "sector": "Utilities"},
    {"symbol": "SJVN.NS", "name": "SJVN", "sector": "Utilities"},
    {"symbol": "IREDA.NS", "name": "IREDA", "sector": "Financials"},
    {"symbol": "GAIL.NS", "name": "GAIL", "sector": "Energy"},
    {"symbol": "NMDC.NS", "name": "NMDC", "sector": "Metals"},
    {"symbol": "HFCL.NS", "name": "HFCL", "sector": "Telecom"},
    {"symbol": "BHEL.NS", "name": "BHEL", "sector": "Industrials"},
    {"symbol": "REC.NS", "name": "REC", "sector": "Financials"},
    {"symbol": "PFC.NS", "name": "Power Finance Corp", "sector": "Financials"},
    {"symbol": "JIOFIN.NS", "name": "Jio Financial Services", "sector": "Financials"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "Technology"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Auto"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco", "sector": "Metals"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "sector": "Pharma"},
]

app = FastAPI(title="Indian Stock Price Finder API", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/quote/{symbol}")
def get_quote(symbol: str) -> dict[str, Any]:
    clean_symbol = normalize_symbol(symbol)
    quote_data = fetch_yahoo_quote(clean_symbol)
    return quote_data


@app.get("/api/search")
def search_stocks(q: str = Query(default="", max_length=80)) -> list[dict[str, Any]]:
    if not q.strip():
        return POPULAR_STOCKS
    return fetch_yahoo_search(q)


@app.get("/api/stocks")
def list_stocks() -> list[dict[str, Any]]:
    return POPULAR_STOCKS


@app.get("/api/history/{symbol}")
def get_history(symbol: str) -> dict[str, Any]:
    clean_symbol = normalize_symbol(symbol)
    return fetch_yahoo_history(clean_symbol)


@app.get("/api/predict/{symbol}")
def predict_prices(symbol: str) -> dict[str, Any]:
    clean_symbol = normalize_symbol(symbol)
    history = fetch_yahoo_history(clean_symbol, range_value="5y", range_label="Last 5 years")
    return build_price_predictions(history)


@app.get("/api/validate/{symbol}")
def validate_price(
    symbol: str,
    expected_price: float = Query(gt=0),
    tolerance_percent: float = Query(default=0.5, ge=0, le=100),
) -> dict[str, Any]:
    quote_data = get_quote(symbol)
    current_price = quote_data["current_price"]
    difference = expected_price - current_price
    difference_percent = abs(difference) / current_price * 100
    is_valid = difference_percent <= tolerance_percent

    return {
        **quote_data,
        "expected_price": round(expected_price, 4),
        "difference": round(difference, 4),
        "difference_percent": round(difference_percent, 4),
        "tolerance_percent": tolerance_percent,
        "is_valid": is_valid,
        "verdict": "MATCH" if is_valid else "OUTSIDE_TOLERANCE",
    }


def normalize_symbol(symbol: str) -> str:
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise HTTPException(status_code=400, detail="Stock symbol is required.")
    if len(clean_symbol) > 20:
        raise HTTPException(status_code=400, detail="Stock symbol is too long.")
    if "." not in clean_symbol and not clean_symbol.startswith("^"):
        return f"{clean_symbol}.NS"
    return clean_symbol


def fetch_yahoo_search(query: str) -> list[dict[str, Any]]:
    encoded_query = quote(query.strip(), safe="")
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded_query}&quotesCount=12&newsCount=0"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not search market data provider. Check your internet connection and try again.",
        ) from exc

    matches = []
    seen_symbols = set()
    for quote_item in payload.get("quotes", []):
        symbol = quote_item.get("symbol")
        exchange = quote_item.get("exchDisp") or quote_item.get("exchange")
        quote_type = quote_item.get("quoteType")

        if not symbol or symbol in seen_symbols:
            continue
        if not is_indian_equity(symbol, exchange, quote_type):
            continue

        seen_symbols.add(symbol)
        matches.append(
            {
                "symbol": symbol,
                "name": quote_item.get("shortname") or quote_item.get("longname") or symbol,
                "exchange": exchange,
                "type": quote_type,
            }
        )

    if not matches:
        fallback_symbol = normalize_symbol(query)
        matches.append(
            {
                "symbol": fallback_symbol,
                "name": fallback_symbol,
                "exchange": "NSE" if fallback_symbol.endswith(".NS") else "BSE",
                "type": "EQUITY",
            }
        )

    return sorted(matches, key=lambda match: search_rank(match, query))


def is_indian_equity(symbol: str, exchange: str | None, quote_type: str | None) -> bool:
    indian_symbol = symbol.endswith((".NS", ".BO"))
    indian_exchange = exchange in {"NSE", "BSE", "NSI", "BSE - BSE Real Time Price"}
    equity_like = quote_type in {None, "EQUITY", "ETF", "INDEX"}
    return (indian_symbol or indian_exchange) and equity_like


def search_rank(match: dict[str, Any], query: str) -> tuple[int, int, str]:
    normalized_query = query.strip().upper()
    symbol = match["symbol"].upper()
    base_symbol = symbol.split(".", maxsplit=1)[0]
    exchange = str(match.get("exchange") or "").upper()

    exact_symbol = 0 if base_symbol == normalized_query else 1
    exchange_rank = 0 if symbol.endswith(".NS") or exchange in {"NSE", "NSI"} else 1
    return exact_symbol, exchange_rank, symbol


def fetch_yahoo_quote(symbol: str) -> dict[str, Any]:
    encoded_symbol = quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range=1d&interval=1m"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not reach market data provider. Check your internet connection and try again.",
        ) from exc

    result = payload.get("chart", {}).get("result")
    if not result:
        raise HTTPException(status_code=404, detail=f"No market data found for {symbol}.")

    chart = result[0]
    meta = chart.get("meta", {})
    current_price = meta.get("regularMarketPrice")
    previous_close = meta.get("chartPreviousClose") or meta.get("previousClose")

    if current_price is None:
        current_price = latest_close(chart)
    if current_price is None:
        raise HTTPException(status_code=404, detail=f"No current price found for {symbol}.")

    current_price = float(current_price)
    previous_close = float(previous_close) if previous_close is not None else None
    change = current_price - previous_close if previous_close else None
    change_percent = (change / previous_close * 100) if previous_close else None
    quote_time = meta.get("regularMarketTime")

    return {
        "symbol": meta.get("symbol", symbol),
        "short_name": meta.get("shortName") or meta.get("longName"),
        "exchange": meta.get("exchangeName"),
        "currency": meta.get("currency", "USD"),
        "current_price": round(current_price, 4),
        "previous_close": round(previous_close, 4) if previous_close is not None else None,
        "change": round(change, 4) if change is not None else None,
        "change_percent": round(change_percent, 4) if change_percent is not None else None,
        "market_state": meta.get("marketState", "UNKNOWN"),
        "quote_time": format_epoch(quote_time),
        "provider": "Yahoo Finance",
    }


def fetch_yahoo_history(
    symbol: str,
    range_value: str = "6mo",
    range_label: str = "Last 6 months",
) -> dict[str, Any]:
    encoded_symbol = quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range={range_value}&interval=1d"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not reach historical market data provider. Check your internet connection and try again.",
        ) from exc

    result = payload.get("chart", {}).get("result")
    if not result:
        raise HTTPException(status_code=404, detail=f"No historical market data found for {symbol}.")

    chart = result[0]
    meta = chart.get("meta", {})
    timestamps = chart.get("timestamp", [])
    quote_blocks = chart.get("indicators", {}).get("quote", [])
    quote_block = quote_blocks[0] if quote_blocks else {}
    closes = quote_block.get("close", [])
    highs = quote_block.get("high", [])
    lows = quote_block.get("low", [])
    opens = quote_block.get("open", [])
    volumes = quote_block.get("volume", [])
    points = []

    for index, timestamp in enumerate(timestamps):
        close = closes[index] if index < len(closes) else None
        if close is None:
            continue
        high = highs[index] if index < len(highs) else None
        low = lows[index] if index < len(lows) else None
        open_price = opens[index] if index < len(opens) else None
        volume = volumes[index] if index < len(volumes) else None
        points.append(
            {
                "date": format_epoch(timestamp),
                "open": round(float(open_price), 4) if open_price is not None else None,
                "close": round(float(close), 4),
                "high": round(float(high), 4) if high is not None else None,
                "low": round(float(low), 4) if low is not None else None,
                "volume": int(volume) if volume is not None else None,
            }
        )

    if len(points) < 2:
        raise HTTPException(status_code=404, detail=f"Not enough historical close prices found for {symbol}.")

    return {
        "symbol": meta.get("symbol", symbol),
        "currency": meta.get("currency", "INR"),
        "range_label": range_label,
        "points": points,
        "analytics": calculate_analytics(points),
        "provider": "Yahoo Finance",
    }


def calculate_analytics(points: list[dict[str, Any]]) -> dict[str, Any]:
    closes = [float(point["close"]) for point in points]
    highs = [float(point["high"]) for point in points if point.get("high") is not None]
    lows = [float(point["low"]) for point in points if point.get("low") is not None]
    first_close = closes[0]
    last_close = closes[-1]
    period_return_percent = (last_close - first_close) / first_close * 100
    daily_returns = [
        (closes[index] - closes[index - 1]) / closes[index - 1]
        for index in range(1, len(closes))
    ]
    average_daily_return = average(daily_returns)
    daily_volatility = standard_deviation(daily_returns)
    sma_20 = average(closes[-20:])
    sma_50 = average(closes[-50:])

    if last_close >= sma_20 >= sma_50:
        trend = "Bullish"
    elif last_close <= sma_20 <= sma_50:
        trend = "Bearish"
    else:
        trend = "Sideways"

    return {
        "first_close": round(first_close, 4),
        "last_close": round(last_close, 4),
        "average_close": round(average(closes), 4),
        "period_high": round(max(highs or closes), 4),
        "period_low": round(min(lows or closes), 4),
        "period_return_percent": round(period_return_percent, 4),
        "average_daily_return_percent": round(average_daily_return * 100, 4),
        "annualized_volatility_percent": round(daily_volatility * (252 ** 0.5) * 100, 4),
        "sma_20": round(sma_20, 4),
        "sma_50": round(sma_50, 4),
        "trend": trend,
        "best_day_percent": round(max(daily_returns) * 100, 4),
        "worst_day_percent": round(min(daily_returns) * 100, 4),
    }


def average(values: list[float]) -> float:
    if not values:
        return 0
    return sum(values) / len(values)


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0
    mean = average(values)
    variance = average([(value - mean) ** 2 for value in values])
    return variance ** 0.5



def scan_under_500_stocks(max_price: float = 500.0) -> dict[str, Any]:
    """Screen a curated NSE watchlist using current Yahoo Finance data.

    The score is a transparent technical screening score, not a return forecast.
    Stocks without usable data or priced above max_price are excluded.
    """
    screened = []
    failures = []
    for stock in UNDER_500_UNIVERSE:
        symbol = stock["symbol"]
        try:
            history = fetch_yahoo_history(symbol, range_value="6mo", range_label="Last 6 months")
            points = history["points"]
            analytics = history["analytics"]
            current_price = float(points[-1]["close"])
            if current_price <= 0 or current_price > max_price:
                continue

            recent_volumes = [
                float(point["volume"])
                for point in points[-20:]
                if point.get("volume") is not None and float(point["volume"]) > 0
            ]
            avg_volume = average(recent_volumes) if recent_volumes else 0.0
            trend_score = 35.0 if analytics["trend"] == "Bullish" else 20.0 if analytics["trend"] == "Sideways" else 5.0
            momentum_score = max(0.0, min(25.0, float(analytics["period_return_percent"]) * 1.25 + 12.5))
            volatility_score = max(0.0, 20.0 - min(float(analytics["annualized_volatility_percent"]), 40.0) * 0.5)
            liquidity_score = min(20.0, float(np.log10(avg_volume + 1.0)) * 2.0)
            score = round(trend_score + momentum_score + volatility_score + liquidity_score, 1)

            screened.append(
                {
                    **stock,
                    "current_price": round(current_price, 2),
                    "trend": analytics["trend"],
                    "six_month_return_percent": round(float(analytics["period_return_percent"]), 2),
                    "annualized_volatility_percent": round(float(analytics["annualized_volatility_percent"]), 2),
                    "average_daily_volume": int(avg_volume),
                    "screen_score": score,
                    "screen_label": "Higher score" if score >= 70 else "Watch" if score >= 55 else "Needs review",
                    "symbol_for_search": symbol,
                }
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "reason": str(exc)})

    screened.sort(key=lambda item: (item["screen_score"], item["average_daily_volume"]), reverse=True)
    return {
        "max_price": float(max_price),
        "scanned_universe": len(UNDER_500_UNIVERSE),
        "eligible_count": len(screened),
        "stocks": screened,
        "provider": "Yahoo Finance",
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "failed_symbols": [item["symbol"] for item in failures],
    }


def build_price_predictions(history: dict[str, Any]) -> dict[str, Any]:
    points = history["points"]
    closes = np.array([float(point["close"]) for point in points], dtype=float)
    if len(closes) < 90:
        raise HTTPException(
            status_code=422,
            detail="At least 90 historical trading sessions are required for prediction.",
        )

    feature_rows = build_feature_rows(points)
    latest_features = feature_rows[-1:].copy()
    horizons = list(PREDICTION_HORIZONS)
    predictions = []

    for horizon in horizons:
        model_result = train_best_prediction_model(feature_rows, closes, horizon)
        predicted_price = max(0, float(model_result["prediction"]))
        current_price = float(closes[-1])
        change_percent = (predicted_price - current_price) / current_price * 100
        predictions.append(
            {
                "horizon_days": horizon,
                "predicted_price": round(predicted_price, 4),
                "change_percent": round(change_percent, 4),
                "model": model_result["model_name"],
                "validation_mape_percent": round(model_result["mape_percent"], 4),
                "confidence": confidence_label(model_result["mape_percent"]),
            }
        )

    return {
        "symbol": history["symbol"],
        "currency": history["currency"],
        "current_price": round(float(closes[-1]), 4),
        "trained_on_sessions": len(points),
        "algorithm": "Best of Random Forest, Extra Trees, and Gradient Boosting",
        "features": [
            "lagged closes",
            "daily returns",
            "moving averages",
            "rolling volatility",
            "high-low range",
            "volume change",
        ],
        "predictions": predictions,
        "provider": history["provider"],
    }


def build_feature_rows(points: list[dict[str, Any]]) -> np.ndarray:
    closes = np.array([float(point["close"]) for point in points], dtype=float)
    highs = np.array([float(point.get("high") or point["close"]) for point in points], dtype=float)
    lows = np.array([float(point.get("low") or point["close"]) for point in points], dtype=float)
    opens = np.array([float(point.get("open") or point["close"]) for point in points], dtype=float)
    volumes = np.array([float(point.get("volume") or 0) for point in points], dtype=float)
    rows = []

    for index in range(len(points)):
        close = closes[index]
        previous_close = closes[index - 1] if index > 0 else close
        daily_return = safe_ratio(close - previous_close, previous_close)
        open_return = safe_ratio(close - opens[index], opens[index])
        high_low_range = safe_ratio(highs[index] - lows[index], close)
        volume_change = safe_ratio(volumes[index] - volumes[index - 1], volumes[index - 1]) if index > 0 else 0

        rows.append(
            [
                close,
                lag_value(closes, index, 1),
                lag_value(closes, index, 2),
                lag_value(closes, index, 5),
                lag_value(closes, index, 10),
                rolling_mean(closes, index, 5),
                rolling_mean(closes, index, 10),
                rolling_mean(closes, index, 20),
                rolling_mean(closes, index, 50),
                rolling_std(closes, index, 10),
                rolling_std(closes, index, 20),
                daily_return,
                open_return,
                high_low_range,
                volume_change,
            ]
        )

    return np.nan_to_num(np.array(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def train_best_prediction_model(features: np.ndarray, closes: np.ndarray, horizon: int) -> dict[str, Any]:
    usable_count = len(closes) - horizon
    x_data = features[:usable_count]
    y_data = closes[horizon:]
    if len(x_data) < 75:
        raise HTTPException(status_code=422, detail=f"Not enough history to train {horizon}-day model.")

    split_index = max(60, int(len(x_data) * 0.8))
    if split_index >= len(x_data):
        split_index = len(x_data) - 10

    x_train, x_valid = x_data[:split_index], x_data[split_index:]
    y_train, y_valid = y_data[:split_index], y_data[split_index:]
    candidates = [
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=7,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=90,
                learning_rate=0.045,
                max_depth=3,
                random_state=42,
            ),
        ),
    ]

    best_model = None
    best_name = ""
    best_mape = float("inf")

    for model_name, model in candidates:
        model.fit(x_train, y_train)
        validation_predictions = model.predict(x_valid)
        mape = float(mean_absolute_percentage_error(y_valid, validation_predictions) * 100)
        if mape < best_mape:
            best_mape = mape
            best_model = model
            best_name = model_name

    if best_model is None:
        raise HTTPException(status_code=500, detail="Could not train prediction model.")

    best_model.fit(x_data, y_data)
    prediction = float(best_model.predict(features[-1:])[0])
    return {
        "model_name": best_name,
        "mape_percent": best_mape,
        "prediction": prediction,
    }


def lag_value(values: np.ndarray, index: int, lag: int) -> float:
    return float(values[index - lag]) if index >= lag else float(values[0])


def rolling_mean(values: np.ndarray, index: int, window: int) -> float:
    start = max(0, index - window + 1)
    return float(np.mean(values[start : index + 1]))


def rolling_std(values: np.ndarray, index: int, window: int) -> float:
    start = max(0, index - window + 1)
    return float(np.std(values[start : index + 1]))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0
    return float(numerator / denominator)


def confidence_label(mape_percent: float) -> str:
    if mape_percent <= 3:
        return "High"
    if mape_percent <= 7:
        return "Medium"
    return "Low"


def latest_close(chart: dict[str, Any]) -> float | None:
    quote_blocks = chart.get("indicators", {}).get("quote", [])
    if not quote_blocks:
        return None

    closes = quote_blocks[0].get("close", [])
    for price in reversed(closes):
        if price is not None:
            return float(price)
    return None


def format_epoch(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
