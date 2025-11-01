import json
import re
import unicodedata
from LLM.llm_handler import LlmHandler
from Utils.data_executor import execute_query_plan

llm = LlmHandler()

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
def generate_query_plan(user_input: str, context: str = ""):
    prompt = f"""
You are an intelligent logistics data analyst.
Return JSON describing how to analyze data based on the user's question.

Valid intents:
- "route_with_most_delays"
- "delay_reason_summary"
- "average_delay"
- "compare_average_delay"
- "trend_analysis"
- "predict_delay"
- "warehouse_filter_by_time"
- "top_warehouses_by_processing_time"

Schema:
{{
  "intent": string,
  "metric": string,
  "period": string|null,
  "aggregation": string|null,
  "threshold": float|null,
  "top_k": int|null
}}

Examples:
User: "List warehouses with average delivery time above 5 days."
→ {{"intent": "warehouse_filter_by_time", "metric": "delivery_time", "threshold": 5.0, "aggregation": "mean"}}

User: "Show the top 3 warehouses with the highest processing time."
→ {{"intent": "top_warehouses_by_processing_time", "metric": "delivery_time", "top_k": 3, "aggregation": "mean"}}

User: "Predict average delay next week."
→ {{"intent": "predict_delay", "metric": "delay_minutes"}}

Return ONLY valid JSON.
User: "{user_input}"
Context: "{context}"
"""

    # --- 1️⃣ call LLM ---
    try:
        response = llm.ask(prompt)
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
def handle_query(user_input, df, chat_context=""):
    plan = generate_query_plan(user_input, context=chat_context)

    # 🚀 If intent is "predict_delay", use Linear Regression to make a real prediction
    if plan.get("intent") == "predict_delay":
        from sklearn.linear_model import LinearRegression
        import numpy as np
        import matplotlib.pyplot as plt

        metric = plan.get("metric", "delay_minutes")

        # Calculate average delay by month
        df["month_num"] = df["date"].dt.month
        monthly_avg = df.groupby("month_num")[metric].mean()

        if len(monthly_avg) >= 2:
            X = np.arange(len(monthly_avg)).reshape(-1, 1)
            y = monthly_avg.values
            model = LinearRegression().fit(X, y)
            next_pred = model.predict([[len(monthly_avg)]])[0]
            pred = next_pred
            mean = np.mean(y)
        else:
            mean = df[metric].mean()
            pred = mean * 1.05

        # Visualize current vs predicted
        fig, ax = plt.subplots()
        plt.bar(["Current", "Predicted"], [mean, pred], color=["gray", "orange"])
        ax.set_ylabel("Average Delay (minutes)")
        ax.set_title("Predicted Average Delay for Next Week")
        plt.tight_layout()

        text = f"🔮 Predicted average delivery delay for next week: **{pred:.2f} minutes** (based on trend)"
        return text, fig

    # Other intents are handled as usual
    return execute_query_plan(plan, df)
