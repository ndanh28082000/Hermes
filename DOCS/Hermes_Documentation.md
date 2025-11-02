# Hermes — System Documentation

## Overview
Hermes is an AI-assisted logistics analytics app built with Streamlit. It accepts natural language questions from a user, translates them to a structured query plan via a small LLM-based pipeline, executes analyses on shipment data, and returns a textual summary plus visualizations.

This document describes the data layout, the system architecture, the runtime workflow, and how query understanding and data summarization work.

---

## Table of Contents
- Data structure
- System architecture
- Runtime workflow
- Query understanding and generation
- Data summarization and execution
- Forecasting (ARIMA)
- How to run & dependencies
- Notes and next steps

---

## Data structure
Location: `Data/shipments.csv`

The project expects a shipping dataset that contains one row per shipment or event. The codebase uses these fields (observed/assumed):

- `date` (datetime): timestamp of the shipment event. Parsed to `pd.Timestamp` by `Utils.analytics.load_data()`.
- `route` (str): route identifier, used for route-level analysis.
- `warehouse` (str): warehouse identifier for facility / location analysis.
- `delivery_time` (float): delivery or processing time (days) used in `warehouse` analyses.
- `delay_minutes` (float): delay in minutes, used as primary delay metric across analyses.
- `delay_reason` (str): categorical reason for a delay (e.g., "weather", "customs") used for reason summaries.

Note: The above list is an inferred/typical schema. If your CSV uses different column names, update either the CSV or the code in `Utils/analytics.py` / `Utils/data_executor.py` accordingly.

`Utils/analytics.load_data(file_path)` reads the CSV and converts the `date` column to datetime. Many downstream functions rely on `date` being a datetime dtype.

---

## System architecture
High-level components (files and responsibilities):

- `App/app.py` — Streamlit entry point. Loads data, manages session state (including LLM instance), displays chat UI, and dispatches user questions to the query handler.
- `LLM/llm_handler.py` — Thin wrapper for the LLM API (Hugging Face router in current implementation). Exposes `.ask(prompt)` which returns the model's text.
- `Utils/query_parser.py` — Orchestrates LLM calls to produce a structured query plan (JSON schema). Also contains fallback, rule-based parsing and a `handle_query()` function that picks the correct action.
- `Utils/data_executor.py` — Core execution engine: given a normalized plan and the dataframe, it runs groupbys, aggregations and plotting and returns (text, fig).
- `Utils/analytics.py` — Helper analysis functions and local utilities used for exploration and testing (legacy pieces remain alongside `data_executor` logic).
- `Forecast/forecast.py` — Forecasting module. Contains `DelayPredictor` which uses ARIMA to model monthly average delays and predict the next period.
- `Data/shipments.csv` — The primary dataset.

Design notes:
- Dependency injection: the LLM handler is now initialized in `App/app.py` (single instance, stored in `st.session_state.llm_handler`) and passed into `handle_query` — this ensures the LLM client is constructed once per app session rather than per request.
- Forecasting is separated into its own package (`Forecast`) to keep modeling code isolated and testable.
- `Utils/data_executor.execute_query_plan()` is the deterministic, non-LLM backend that actually summarizes and visualizes data.

---

## Runtime workflow (user request -> answer)
1. User types a question in the Streamlit chat UI in `App/app.py`.
2. `app.py` builds a small conversation `context` from `st.session_state.chat_history` and calls `handle_query(user_input, df, st.session_state.llm_handler, chat_context=context)`.
3. `handle_query` calls `generate_query_plan(user_input, llm_handler, context=chat_context)`.
   - `generate_query_plan` composes a detailed prompt describing expected schema and examples, calls `llm_handler.ask(prompt)`, and attempts to parse the returned JSON.
   - If JSON parse fails or the model returns an incomplete plan, a rule-based fallback runs to infer common intents (predict, warehouse thresholds, top-k, route analysis, delay reasons, averages, etc.).
   - The final plan is a dictionary with keys such as `intent`, `metric`, `period`, `aggregation`, `threshold`, `top_k`, and optional nested fields (`comparison`, `trend`).
4. Back in `handle_query`, if the intent is `predict_delay`, the code will use the `Forecast.DelayPredictor` instance:
   - The predictor is initialized and trained only once (the instance is stored as `handle_query.predictor`).
   - Training computes monthly averages and fits an ARIMA(1,1,0) model when enough points exist. Otherwise, a mean-based fallback is used.
   - The predictor forecasts the next period and produces a plot.
5. For non-forecasting intents, `handle_query` forwards the plan and `df` to `Utils.data_executor.execute_query_plan(plan, df)`.
   - `execute_query_plan` applies period filtering, conducts groupby/aggregation operations, renders a matplotlib figure, and builds a textual summary.
6. The Streamlit app receives `(text, fig)` and displays the text and embedded chart. The chat entry is appended to session history.

---

## Query understanding (LLM prompt + fallback)
Approach:
- Prompt-first: the LLM is asked to return a strict JSON describing the analysis plan using a clear schema. The prompt contains capability lists and many examples to increase robustness to natural language variety.
- Parse-first: If the LLM returns valid JSON, that becomes the plan. The prompt was recently expanded with many examples and nested fields to capture more nuanced requests (comparisons, trends, time intervals, etc.).
- Fallback rules: If the LLM output is missing, malformed or returns `unknown`, a deterministic rule-based extractor runs to detect common intents. It uses simple regexes and keyword checks (e.g., `predict`, `warehouse`, `top`, `route`, `reason`, numerical thresholds like `above 5 days`).

Key implementation details:
- `generate_query_plan()` in `Utils/query_parser.py` is the place that unifies LLM and fallback outputs into a single `plan` dictionary.
- The LLM prompt instructs the model to return only valid JSON that follows the plan schema. The prompt includes multiple examples covering: thresholds, top-k ranking, comparisons, trend detection, and predictive requests.
- The rule-based fallback also attempts to extract numeric thresholds from natural language using regex and normalized text methods.

Why this hybrid approach?
- LLMs are great at interpreting natural language and supplying high-level intent plus nuanced parameters.
- A deterministic fallback ensures the system still works when the LLM is offline, rate-limited, or returns an unexpected format.

---

## Data summarization and execution
`Utils/data_executor.execute_query_plan(plan, df)` maps `intent` to concrete operations. Typical patterns:

- GroupBy + aggregation (mean, sum, count) for metrics such as `delay_minutes` and `delivery_time`.
- Period filtering via helper `_filter_by_period()` which understands terms like "last week", "last month", named months, and simple ranges.
- Visualizations: `matplotlib` figures are created (bar/line plots) and returned alongside textual summaries.
- Error handling: empty results return friendly messages and no figure.

Examples of supported analyses:
- Route with most delays — group by `route`, sum `delay_minutes` and return the top route and bar chart.
- Delay reason summary — group by `delay_reason`, sum `delay_minutes`.
- Average delay by month — compute monthly averages and plot trend.
- Warehouse filters — compute average `delivery_time` by `warehouse` and filter by threshold.
- Top-k warehouses — ranking by mean delivery time.

Design consideration: the executor is intentionally simple and transparent (Pandas + matplotlib) so results are traceable, debuggable, and easy to extend.

---

## Forecasting (ARIMA)
- Forecasting code is in `Forecast/forecast.py` as `DelayPredictor`.
- Behavior:
  - Computes monthly averages (grouped by `year_month` derived from `date`).
  - If 3+ monthly points exist, fits an ARIMA(1,1,0) model using `statsmodels`.
  - Forecasts 1-step ahead (next month) using the fitted model; if the model fitting fails or there are insufficient points, returns a mean-based heuristic (mean * 1.05).
  - `plot_prediction()` returns a matplotlib figure showing historical monthly averages (line) and a bar chart with current mean and predicted value.
- The predictor is trained once and the instance is reused across queries (trained lazily on first predict request). This avoids retraining on every request.

Notes:
- ARIMA is a good lightweight baseline for univariate time series. For stronger forecasting, consider seasonal ARIMA (`SARIMAX`) or `pmdarima.auto_arima` to pick orders automatically.

---

## How to run & dependencies
1. Install dependencies (PowerShell example):

```powershell
python -m pip install -r requirements.txt
```

2. Set required environment variables (example):

```powershell
$env:HF_TOKEN = "<your-hf-token>"
```

3. Run the Streamlit app from the repo root:

```powershell
streamlit run App\app.py
```

4. Open the provided Streamlit URL in your browser and interact via the chat input.

Notes:
- `requirements.txt` contains: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`, `requests`, `streamlit`.
- The LLM handler uses `HF_TOKEN` to authenticate with the Hugging Face router URL. If you don't want to call the LLM while testing, you can either:
  - Replace calls to the LLM with a mocked response in `Utils/query_parser.py`, or
  - Use the rule-based fallback by temporarily forcing generation to fail.

---

## Notes, caveats and next steps
- Column names: The code expects specific column names such as `date`, `route`, `warehouse`, `delivery_time`, `delay_minutes`, `delay_reason`. If your CSV uses different names, update either the CSV or the code.
- Forecast tuning: ARIMA order is fixed at (1,1,0). If your series show seasonality, consider seasonal models or `pmdarima.auto_arima`.
- LLM robustness: The prompt has been expanded with many examples and nested schema fields to handle natural language better. You may still see malformed JSON occasionally; improve robustness by stricter parsing, schema validation, or returning an extra `raw` field.
- Testing: For reproducible checks, add unit tests for `Forecast.DelayPredictor` using a small synthetic time series and for `Utils/data_executor.execute_query_plan` with a trimmed test CSV.

---

## Quick developer checklist
- [ ] Confirm `Data/shipments.csv` matches the expected schema.
- [ ] Install requirements and set `HF_TOKEN`.
- [ ] Run `streamlit run App\app.py` and try the intent "Predict average delay next week" and a few examples like:
  - "List warehouses with average delivery time above 5 days"
  - "Which route has the most delays last week?"
  - "Why are shipments being delayed on route X?"

If you want, I can also:
- Add a short unit test for `Forecast.DelayPredictor` (synthetic series).
- Add an optional Streamlit control to re-train the predictor or choose ARIMA order.
- Integrate `pmdarima` auto-arima if you'd like an automatic order-selection flow.

---

End of document.
