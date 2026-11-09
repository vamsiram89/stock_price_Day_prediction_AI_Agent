# Indian Stock Price Finder

A small web app for searching Indian NSE/BSE stocks, checking live prices, validating an expected price, and viewing 6-month historical analytics.

## Run With Streamlit

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL printed in the terminal, usually http://localhost:8501.

## Optional FastAPI Backend

```powershell
python run.py
```

Open the URL printed in the terminal, usually http://127.0.0.1:8000.

## What It Does

- Shows the latest available market price for a stock symbol.
- Searches Indian stocks by company name or symbol.
- Displays previous close, absolute change, percent change, market state, and quote time.
- Shows 6-month return, trend, annualized volatility, history high/low, and recent close-price bars.
- Predicts 5, 15, and 30 trading day prices using the best validation result from Random Forest, Extra Trees, and Gradient Boosting.
- Lets you enter an expected price and tolerance to validate whether your price matches the live market price.

Market data is fetched from Yahoo Finance's public chart endpoint.


## Under-₹500 NSE Stock Screener

- Screens a curated watchlist of 30 NSE-listed symbols using the latest available daily close.
- Filters stocks at or below ₹500 and ranks eligible candidates using a transparent technical score based on trend, six-month momentum, annualized volatility, and average daily volume.
- Shows ranked candidates, a comparison table, and a top-score chart in the Streamlit dashboard.
- The scan is cached for 30 minutes to reduce repeated market-data requests.

The screening score is a research aid, not a buy/sell recommendation. Prices can change and the watchlist is not the full NSE universe.
