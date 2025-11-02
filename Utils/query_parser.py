import json
import re
import unicodedata
from LLM.llm_handler import LlmHandler
from Utils.data_executor import execute_query_plan
from Forecast.forecast import DelayPredictor

# ===============================================================
# 🔧 Helper: Normalize text for regex
# ===============================================================
def _normalize_text(text: str):
    """
    Normalize text for easier regex matching (remove accents, special characters, newlines).
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.replace(",", ".")
    text = re.sub(r"[^a-zA-Z0-9\s\.><=]", " ", text)
    return text.strip().lower()

# ===============================================================
# 🔍 Helper: Extract threshold from text
# ===============================================================
def _extract_threshold_from_text(text: str):
    """
    Find threshold in the question, e.g.:
    "above 5 days" -> 5.0
    "greater than 10" -> 10.0
    "more than 4.5d" -> 4.5
    "> 6 days" -> 6.0
    """
    text = _normalize_text(text)
    print("[DEBUG] Normalized text for threshold detection:", text)

    pattern = r"(?:above|over|greater\s+than|more\s+than|>\s*)(\d+(?:\.\d+)?)\s*(?:days?|d)?"
    match = re.search(pattern, text)
    if match:
        value = float(match.group(1))
        print(f"[DEBUG] Threshold extracted: {value}")
        return value

    print("[DEBUG] No threshold matched in:", text)
    return None

# ===============================================================
# 🧠 Generate Query Plan (LLM + fallback)
# ===============================================================
def generate_query_plan(user_input: str, llm_handler: LlmHandler, context: str = ""):
    prompt = f"""
You are an expert logistics data analyst assistant. Analyze the user's question and return a structured query plan.
Think step by step:
1. Understand the main goal (prediction, comparison, analysis, etc.)
2. Identify key metrics and conditions
3. Choose the most appropriate analysis method

Core Capabilities:
1. Route Analysis:
   - Identify problematic routes ("which routes have delays", "problematic shipping paths")
   - Analyze specific routes ("how is route A performing", "delays on path B")
   - Compare routes ("compare delays between routes", "which path is faster")

2. Delay Analysis:
   - Overall delays ("how bad are our delays", "average delay time")
   - Delay patterns ("when do delays usually happen", "peak delay times")
   - Delay predictions ("will we have delays next week", "forecast delivery times")
   - Delay reasons ("why are shipments delayed", "main causes of delay")

3. Warehouse Performance:
   - Processing times ("how fast are warehouses working", "processing speed")
   - Efficiency comparison ("which warehouse is fastest", "slowest facilities")
   - Time thresholds ("warehouses taking over 3 days", "quick processing locations")
   - Rankings ("top performing warehouses", "worst delay offenders")

4. Trend Analysis:
   - Time patterns ("are delays getting worse", "improvement over time")
   - Seasonal analysis ("summer vs winter performance", "monthly trends")
   - Future projections ("next month's outlook", "upcoming delay risks")

Return a JSON plan with these fields:
{{
  "intent": string,      // Main analysis type
  "metric": string,      // What to measure (delay_minutes, delivery_time, etc.)
  "period": string,      // Time period (daily, weekly, monthly, specific dates)
  "aggregation": string, // How to aggregate (mean, max, min, sum, count)
  "threshold": float,    // Numeric threshold for filtering
  "top_k": int,         // For rankings/limits
  "comparison": {{       // For comparative analysis
    "type": string,     // "time", "location", "route"
    "targets": array    // What to compare
  }},
  "trend": {{           // For trend analysis
    "interval": string, // "daily", "weekly", "monthly"
    "direction": string // "increasing", "decreasing", "stable"
  }}
}}

Examples of Natural Language Understanding:

User: "Have our deliveries gotten any faster in the past month?"
→ {{
  "intent": "trend_analysis",
  "metric": "delivery_time",
  "period": "last_month",
  "aggregation": "mean",
  "trend": {{"interval": "daily", "direction": "decreasing"}}
}}

User: "Which warehouses are struggling with long processing times?"
→ {{
  "intent": "warehouse_filter_by_time",
  "metric": "processing_time",
  "aggregation": "mean",
  "threshold": null,
  "top_k": 5,
  "comparison": {{"type": "location", "targets": ["all_warehouses"]}}
}}

User: "Why do we keep having delays on the Boston route?"
→ {{
  "intent": "delay_reason_summary",
  "metric": "delay_minutes",
  "period": "last_3_months",
  "comparison": {{"type": "route", "targets": ["Boston"]}}
}}

User: "I think our summer performance was better than winter, is that true?"
→ {{
  "intent": "compare_average_delay",
  "metric": "delay_minutes",
  "aggregation": "mean",
  "comparison": {{
    "type": "time",
    "targets": ["summer_months", "winter_months"]
  }}
}}

User: "Can you predict if we'll have any major delays next week?"
→ {{
  "intent": "predict_delay",
  "metric": "delay_minutes",
  "period": "next_week",
  "threshold": null
}}

User: "{user_input}"
Context: "{context}"

Remember to:
1. Handle variations in how users express time periods
2. Understand implicit metrics from context
3. Detect comparisons and trends
4. Include all relevant fields for the analysis type
5. Return ONLY valid JSON with appropriate fields filled
User: "{user_input}"
Context: "{context}"
"""

    # --- 1️⃣ call LLM ---
    try:
        response = llm_handler.ask(prompt)
    except Exception as e:
        print("[DEBUG] LLM request failed:", e)
        response = "{}"

    # --- 2️⃣ Parse JSON or fallback ---
    try:
        plan = json.loads(response)
    except Exception:
        print("[DEBUG] Invalid JSON from LLM:", response)
        plan = {"intent": "unknown", "raw": response}

    # --- 3️⃣ If LLM did not return threshold, extract it ---
    if plan.get("threshold") is None:
        threshold = _extract_threshold_from_text(user_input)
        if threshold:
            plan["threshold"] = threshold

    # --- 4️⃣ Fallback rule-based intent detection ---
    if plan.get("intent") in [None, "", "unknown"]:
        q = user_input.lower()
        plan = {"intent": "unknown", "metric": "delay_minutes", "period": None, "aggregation": "mean"}

        # --- Predict delay (highest priority) ---
        if any(k in q for k in ["predict", "forecast", "estimate"]) or "next week" in q:
            plan["intent"] = "predict_delay"
            plan["metric"] = "delay_minutes"

        # --- Warehouse filters ---
        elif "warehouse" in q and any(k in q for k in ["above", "greater", "more", ">"]):
            plan["intent"] = "warehouse_filter_by_time"
            plan["metric"] = "delivery_time"
            plan["threshold"] = _extract_threshold_from_text(q)

        # --- Top warehouses ---
        elif "top" in q and "warehouse" in q:
            plan["intent"] = "top_warehouses_by_processing_time"
            plan["metric"] = "delivery_time"
            match = re.search(r"top\s*(\d+)", q)
            plan["top_k"] = int(match.group(1)) if match else 3

        # --- Route with most delays ---
        elif "route" in q and "delay" in q:
            plan["intent"] = "route_with_most_delays"

        # --- Delay reasons ---
        elif "reason" in q:
            plan["intent"] = "delay_reason_summary"

        # --- Average / trend ---
        elif "average" in q and "delay" in q:
            plan["intent"] = "average_delay"

    print("[DEBUG] Final Query Plan:", plan)
    return plan

# ===============================================================
# 💬 Handle Query — Call LLM → Execute Query Plan
# ===============================================================
def handle_query(user_input, df, llm_handler: LlmHandler, chat_context=""):
    plan = generate_query_plan(user_input, llm_handler, context=chat_context)

    # 🚀 If intent is "predict_delay", use our forecast model
    if plan.get("intent") == "predict_delay":
        metric = plan.get("metric", "delay_minutes")
        
        # Get or initialize the predictor
        if not hasattr(handle_query, 'predictor'):
            handle_query.predictor = DelayPredictor()
            handle_query.predictor.train(df, metric)
            
        # Get prediction and visualization
        fig, current, predicted = handle_query.predictor.plot_prediction()
        text = f"🔮 Predicted average delivery delay for next week: **{predicted:.2f} minutes** (based on trend)"
        return text, fig

    # Other intents are handled as usual
    return execute_query_plan(plan, df)
