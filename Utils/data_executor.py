import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# ===============================================================
# Helper — Filter DataFrame by period
# ===============================================================
def _filter_by_period(df, period):
    if not period:
        return df

    period = period.lower()
    now = datetime.now()

    try:
        if "this week" in period:
            start = now - timedelta(days=now.weekday())
            filtered = df[df["date"] >= start]
            return filtered if not filtered.empty else df

        if "last week" in period:
            start = now - timedelta(days=7)
            filtered = df[df["date"] >= start]
            return filtered if not filtered.empty else df

        if "last month" in period or "past month" in period:
            last_month = (now.month - 1) or 12
            filtered = df[df["date"].dt.month == last_month]
            return filtered if not filtered.empty else df

        # Named months
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        for name, num in month_map.items():
            if name in period:
                filtered = df[df["date"].dt.month == num]
                return filtered if not filtered.empty else df

    except Exception:
        pass

    return df


# ===============================================================
# Core — Execute Query Plan
# ===============================================================
def execute_query_plan(plan: dict, df: pd.DataFrame):
    intent = plan.get("intent", "")
    metric = plan.get("metric", "delay_minutes")
    period = plan.get("period", None)
    aggregation = plan.get("aggregation", "mean")

    df["month"] = df["date"].dt.month_name()
    df_filtered = _filter_by_period(df, period)

    # ==========================================
    # ROUTE WITH MOST DELAYS
    # ==========================================
    if intent == "route_with_most_delays":
        agg = df_filtered.groupby("route")[metric].sum().sort_values(ascending=False)
        if agg.empty:
            return (f"⚠️ No shipment data found for '{period or 'all time'}'.", None)

        top_route, val = agg.index[0], agg.iloc[0]
        fig, ax = plt.subplots()
        agg.plot(kind="bar", ax=ax, title="Total Delay Minutes by Route", color="skyblue")
        ax.set_ylabel("Total Delay (minutes)")
        plt.tight_layout()
        text = f"🚚 Route with the most delays ({period or 'all time'}): **{top_route}** ({val:.0f} total minutes)"
        return text, fig

    # ==========================================
    # DELAY REASON SUMMARY
    # ==========================================
    if intent == "delay_reason_summary":
        reason_sum = df_filtered.groupby("delay_reason")[metric].sum().sort_values(ascending=False)
        if reason_sum.empty:
            return (f"⚠️ No delay reason data available for '{period or 'selected period'}'.", None)

        fig, ax = plt.subplots()
        reason_sum.plot(kind="bar", ax=ax, title="Total Delay by Reason", color="lightcoral")
        ax.set_ylabel("Total Delay (minutes)")
        plt.tight_layout()
        text = "🕒 Total delay by reason:\n" + "\n".join(
            [f"  - {r}: {v:.0f}" for r, v in reason_sum.items()]
        )
        return text, fig

    # ==========================================
    # AVERAGE DELAY / TREND / COMPARISON
    # ==========================================
    if intent in ["average_delay", "compare_average_delay", "trend_analysis"]:
        df_filtered["month"] = df_filtered["date"].dt.month_name()
        avg = df_filtered.groupby("month")[metric].mean().sort_index()
        if avg.empty:
            return (f"⚠️ No delay data available for '{period or 'selected timeframe'}'.", None)

        fig, ax = plt.subplots()
        avg.plot(kind="line", marker="o", ax=ax, color="green", title="Average Delay by Month")
        ax.set_ylabel("Average Delay (minutes)")
        plt.tight_layout()
        text = "📊 Average delay by month:\n" + "\n".join(
            [f"  - {m}: {v:.2f} minutes" for m, v in avg.items()]
        )

        if len(avg) >= 2:
            diff = avg.iloc[-1] - avg.iloc[-2]
            trend = "increased" if diff > 0 else "decreased"
            text += f"\n📈 Delay {trend} by {abs(diff):.2f} minutes between {avg.index[-2]} and {avg.index[-1]}."

        return text, fig

    # ==========================================
    # PREDICT DELAY (simple mock +5%)
    # ==========================================
    if intent == "predict_delay":
        mean = df[metric].mean()
        pred = mean * 1.05
        fig, ax = plt.subplots()
        plt.bar(["Current", "Predicted"], [mean, pred], color=["gray", "orange"])
        ax.set_ylabel("Delay (minutes)")
        ax.set_title("Predicted Average Delay Next Week")
        plt.tight_layout()
        text = f"🔮 Predicted average delay for next week: **{pred:.1f} minutes** (based on +5% trend)"
        return text, fig

    # ==========================================
    # WAREHOUSE FILTER BY AVG DELIVERY TIME
    # ==========================================
    if intent == "warehouse_filter_by_time":
        threshold = plan.get("threshold", 0)
        if threshold is None:
            return ("⚠️ No threshold specified (e.g. 'above 5 days').", None)

        avg_delivery = (
            df_filtered.groupby("warehouse")["delivery_time"]
            .mean()
            .sort_values(ascending=False)
        )
        filtered = avg_delivery[avg_delivery > threshold]

        if filtered.empty:
            return (
                f"⚠️ No warehouses found with average delivery time above {threshold} days.",
                None,
            )

        fig, ax = plt.subplots()
        filtered.plot(kind="bar", ax=ax, color="purple", title=f"Warehouses (> {threshold} days)")
        ax.set_ylabel("Average Delivery Time (days)")
        plt.tight_layout()
        text = f"🏭 Warehouses with average delivery time above {threshold} days:\n" + "\n".join(
            [f"  - {wh}: {val:.2f} days" for wh, val in filtered.items()]
        )
        return text, fig

    # ==========================================
    # TOP WAREHOUSES BY PROCESSING TIME
    # ==========================================
    if intent == "top_warehouses_by_processing_time":
        top_k = plan.get("top_k", 3)

        avg_delivery = (
            df_filtered.groupby("warehouse")["delivery_time"]
            .mean()
            .sort_values(ascending=False)
        )

        if avg_delivery.empty:
            return ("⚠️ No warehouse data available for this period.", None)

        top = avg_delivery.head(top_k)

        fig, ax = plt.subplots()
        top.plot(kind="bar", ax=ax, color="orange", title=f"Top {top_k} Warehouses by Processing Time")
        ax.set_ylabel("Average Delivery Time (days)")
        plt.tight_layout()

        text = f"🏭 Top {top_k} warehouses with the highest processing time:\n"
        for wh, val in top.items():
            text += f"  - {wh}: {val:.2f} days\n"

        return text, fig

    # ==========================================
    # FALLBACK / UNKNOWN INTENT
    # ==========================================
    return (
        f"❓ Sorry, I couldn’t compute that yet. (intent='{intent or 'unknown'}', period='{period or 'none'}')",
        None,
    )
