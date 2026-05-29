const quoteForm = document.querySelector("#quoteForm");
const validateForm = document.querySelector("#validateForm");
const symbolInput = document.querySelector("#symbolInput");
const searchResults = document.querySelector("#searchResults");
const clearSearch = document.querySelector("#clearSearch");
const expectedPriceInput = document.querySelector("#expectedPrice");
const toleranceInput = document.querySelector("#tolerance");
const statusBadge = document.querySelector("#statusBadge");
const currentPrice = document.querySelector("#currentPrice");
const quoteMeta = document.querySelector("#quoteMeta");
const previousClose = document.querySelector("#previousClose");
const marketState = document.querySelector("#marketState");
const exchange = document.querySelector("#exchange");
const quoteTime = document.querySelector("#quoteTime");
const changePill = document.querySelector("#changePill");
const validationResult = document.querySelector("#validationResult");
const analyticsMeta = document.querySelector("#analyticsMeta");
const periodReturn = document.querySelector("#periodReturn");
const trendSignal = document.querySelector("#trendSignal");
const volatility = document.querySelector("#volatility");
const historyRange = document.querySelector("#historyRange");
const historyBars = document.querySelector("#historyBars");
const predictionMeta = document.querySelector("#predictionMeta");
const predictionResults = document.querySelector("#predictionResults");

let latestQuote = null;

quoteForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await searchStocks();
});

validateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await validateExpectedPrice();
});

clearSearch.addEventListener("click", async () => {
  symbolInput.value = "";
  await searchStocks();
});

searchResults.addEventListener("click", async (event) => {
  const resultButton = event.target.closest("[data-symbol]");
  if (!resultButton) {
    return;
  }

  symbolInput.value = resultButton.dataset.symbol;
  markActiveResult(resultButton.dataset.symbol);
  await loadQuote(resultButton.dataset.symbol);
});

window.addEventListener("load", async () => {
  symbolInput.value = "";
  await searchStocks();
});

async function searchStocks() {
  const query = symbolInput.value.trim();
  const url = query ? `/api/search?q=${encodeURIComponent(query)}` : "/api/stocks";

  setBusy(true, query ? "Searching" : "Loading all");
  resetValidation();

  try {
    const matches = await fetchJson(url);
    renderSearchResults(matches);

    if (matches.length) {
      await loadQuote(matches[0].symbol);
      statusBadge.textContent = query ? "Search loaded" : "All stocks";
    } else {
      clearQuote();
      showWarning("No Indian stocks found. Try TCS, Reliance, INFY, or SBIN.");
    }
  } catch (error) {
    showWarning(error.message);
    showEmptySearch("Stock list could not be loaded.");
  } finally {
    setBusy(false);
  }
}

async function loadQuote(selectedSymbol) {
  const symbol = normalizeIndianSymbol(selectedSymbol || symbolInput.value);
  if (!symbol) {
    showWarning("Enter a stock symbol first.");
    return;
  }

  setBusy(true, "Fetching");
  resetValidation();

  try {
    const quote = await fetchJson(`/api/quote/${encodeURIComponent(symbol)}`);
    latestQuote = quote;
    renderQuote(quote);
    await loadAnalytics(symbol);
    await loadPredictions(symbol);
    statusBadge.textContent = "Live";
  } catch (error) {
    showWarning(error.message);
  } finally {
    setBusy(false);
  }
}

async function loadAnalytics(symbol) {
  try {
    const history = await fetchJson(`/api/history/${encodeURIComponent(symbol)}`);
    renderAnalytics(history);
  } catch (error) {
    clearAnalytics();
    analyticsMeta.textContent = error.message;
  }
}

async function loadPredictions(symbol) {
  predictionMeta.textContent = "Training models...";
  predictionResults.innerHTML = '<div class="prediction-empty">Training Random Forest, Extra Trees, and Gradient Boosting.</div>';

  try {
    const result = await fetchJson(`/api/predict/${encodeURIComponent(symbol)}`);
    renderPredictions(result);
  } catch (error) {
    predictionMeta.textContent = error.message;
    predictionResults.innerHTML = '<div class="prediction-empty">Prediction is not available for this stock.</div>';
  }
}

async function validateExpectedPrice() {
  const symbol = normalizeIndianSymbol(latestQuote?.symbol || symbolInput.value);
  const expectedPrice = expectedPriceInput.value;
  const tolerance = toleranceInput.value || "0.5";

  if (!symbol || !expectedPrice) {
    showWarning("Enter both a stock symbol and expected price.");
    return;
  }

  setBusy(true, "Checking");

  try {
    const params = new URLSearchParams({
      expected_price: expectedPrice,
      tolerance_percent: tolerance,
    });
    const result = await fetchJson(`/api/validate/${encodeURIComponent(symbol)}?${params}`);
    latestQuote = result;
    renderQuote(result);
    renderValidation(result);
    statusBadge.textContent = "Checked";
  } catch (error) {
    showWarning(error.message);
  } finally {
    setBusy(false);
  }
}

function renderQuote(quote) {
  const currency = quote.currency || "INR";
  currentPrice.textContent = formatMoney(quote.current_price, currency);
  previousClose.textContent = quote.previous_close == null ? "--" : formatMoney(quote.previous_close, currency);
  marketState.textContent = quote.market_state || "--";
  exchange.textContent = quote.exchange || "--";
  quoteTime.textContent = quote.quote_time ? formatTime(quote.quote_time) : "--";
  quoteMeta.textContent = `${quote.short_name || quote.symbol} · ${quote.symbol}`;

  const change = quote.change;
  const changePercent = quote.change_percent;
  if (change == null || changePercent == null) {
    changePill.className = "change-pill neutral";
    changePill.textContent = "--";
    return;
  }

  const isPositive = change >= 0;
  changePill.className = `change-pill ${isPositive ? "positive" : "negative"}`;
  changePill.textContent = `${isPositive ? "+" : ""}${change.toFixed(2)} (${isPositive ? "+" : ""}${changePercent.toFixed(2)}%)`;
}

function renderAnalytics(history) {
  const currency = history.currency || "INR";
  const analytics = history.analytics;
  analyticsMeta.textContent = `${history.symbol} · ${history.range_label} · ${history.points.length} sessions`;
  periodReturn.textContent = formatPercent(analytics.period_return_percent);
  periodReturn.className = analytics.period_return_percent >= 0 ? "positive-text" : "negative-text";
  trendSignal.textContent = analytics.trend;
  volatility.textContent = `${analytics.annualized_volatility_percent.toFixed(2)}%`;
  historyRange.textContent = `${formatMoney(analytics.period_low, currency)} / ${formatMoney(analytics.period_high, currency)}`;
  renderHistoryBars(history.points, currency);
}

function renderPredictions(result) {
  const currency = result.currency || "INR";
  predictionMeta.textContent = `${result.symbol} · ${result.trained_on_sessions} sessions · ${result.algorithm}`;
  predictionResults.innerHTML = "";

  const fragment = document.createDocumentFragment();
  result.predictions.forEach((prediction) => {
    const card = document.createElement("article");
    const isPositive = prediction.change_percent >= 0;
    card.className = "prediction-item";
    card.innerHTML = `
      <span>${prediction.horizon_days} Days</span>
      <strong>${formatMoney(prediction.predicted_price, currency)}</strong>
      <small class="${isPositive ? "positive-text" : "negative-text"}">${formatPercent(prediction.change_percent)} expected</small>
      <em>${escapeHtml(prediction.model)} · ${prediction.confidence} confidence</em>
      <em>Validation error ${prediction.validation_mape_percent.toFixed(2)}%</em>
    `;
    fragment.appendChild(card);
  });

  predictionResults.appendChild(fragment);
}


function renderHistoryBars(points, currency) {
  historyBars.innerHTML = "";

  const visiblePoints = points.slice(-24);
  if (!visiblePoints.length) {
    historyBars.innerHTML = '<div class="empty-results">No historical prices available.</div>';
    return;
  }

  const closes = visiblePoints.map((point) => point.close);
  const minClose = Math.min(...closes);
  const maxClose = Math.max(...closes);
  const span = maxClose - minClose || 1;
  const fragment = document.createDocumentFragment();

  visiblePoints.forEach((point) => {
    const bar = document.createElement("div");
    const height = 18 + ((point.close - minClose) / span) * 82;
    bar.className = "history-bar";
    bar.style.height = `${height}%`;
    bar.title = `${formatDate(point.date)} · ${formatMoney(point.close, currency)}`;
    fragment.appendChild(bar);
  });

  historyBars.appendChild(fragment);
}

function renderSearchResults(matches) {
  searchResults.innerHTML = "";

  if (!matches.length) {
    showEmptySearch("No Indian stock matches found.");
    return;
  }

  const fragment = document.createDocumentFragment();
  matches.forEach((match, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `stock-result ${index === 0 ? "active" : ""}`;
    button.dataset.symbol = match.symbol;
    button.innerHTML = `
      <span class="logo">${escapeHtml((match.name || match.symbol).slice(0, 1))}</span>
      <span class="stock-copy">
        <strong>${escapeHtml(match.name || match.symbol)}</strong>
        <small>${escapeHtml(match.symbol)} · ${escapeHtml(match.sector || match.exchange || "NSE/BSE")}</small>
      </span>
      <span class="result-action">Load</span>
    `;
    fragment.appendChild(button);
  });

  searchResults.appendChild(fragment);
}

function showEmptySearch(message) {
  searchResults.innerHTML = `<div class="empty-results">${escapeHtml(message)}</div>`;
}

function markActiveResult(symbol) {
  searchResults.querySelectorAll(".stock-result").forEach((button) => {
    button.classList.toggle("active", button.dataset.symbol === symbol);
  });
}

function clearQuote() {
  latestQuote = null;
  currentPrice.textContent = "--";
  previousClose.textContent = "--";
  marketState.textContent = "--";
  exchange.textContent = "--";
  quoteTime.textContent = "--";
  quoteMeta.textContent = "Press Search to see Indian stocks.";
  changePill.className = "change-pill neutral";
  changePill.textContent = "--";
  clearAnalytics();
  clearPredictions();
}

function clearAnalytics() {
  analyticsMeta.textContent = "Choose a stock to load analytics.";
  periodReturn.textContent = "--";
  periodReturn.className = "";
  trendSignal.textContent = "--";
  volatility.textContent = "--";
  historyRange.textContent = "--";
  historyBars.innerHTML = "";
}

function clearPredictions() {
  predictionMeta.textContent = "Choose a stock to train the model.";
  predictionResults.innerHTML = '<div class="prediction-empty">Predictions will appear here.</div>';
}

function renderValidation(result) {
  validationResult.className = `validation-result ${result.is_valid ? "valid" : "invalid"}`;
  validationResult.textContent = result.is_valid
    ? `Valid: within ${result.tolerance_percent}% of live price. Difference ${Math.abs(result.difference).toFixed(4)}.`
    : `Outside tolerance: ${result.difference_percent.toFixed(4)}% away from market.`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

function showWarning(message) {
  statusBadge.textContent = "Attention";
  validationResult.className = "validation-result warn";
  validationResult.textContent = message;
}

function setBusy(isBusy, label = "Working") {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
  if (isBusy) {
    statusBadge.textContent = label;
  }
}

function resetValidation() {
  validationResult.className = "validation-result idle";
  validationResult.textContent = "Validation result will appear here.";
}

function normalizeIndianSymbol(value) {
  const symbol = value.trim().toUpperCase();
  if (!symbol || symbol.includes(".") || symbol.startsWith("^")) {
    return symbol;
  }
  return `${symbol}.NS`;
}

function formatMoney(value, currency) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency,
    maximumFractionDigits: value >= 100 ? 2 : 4,
  }).format(value);
}

function formatTime(value) {
  return new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
  }).format(new Date(value));
}

function formatPercent(value) {
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
