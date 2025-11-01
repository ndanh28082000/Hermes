import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =====================================================
# Load data
# =====================================================
def load_data(file_path="D:/AI/Hermes/Data/shipments.csv"):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# =====================================================
# "The route with the most delays
# =====================================================
def route_with_most_delays(df, period=None):
    """
    Return the route with the highest total delay during the specified period. Period examples: 'last week', 'October', '2024-10-01 to 2024-10-31
    """
    df_filtered = _filter_by_period(df, period)

    delay_sum = (
        df_filtered.groupby("route")["delay_minutes"]
        .sum()
        .sort_values(ascending=False)
    )

    best_route = delay_sum.idxmax()
    result_text = f"🚚 Route with the most delays ({period or 'all time'}): {best_route} ({delay_sum.max():.0f} total minutes)"

    fig, ax = plt.subplots()
    delay_sum.plot(kind="bar", ax=ax, title="Total Delay Minutes by Route")
    plt.tight_layout()
    return result_text, fig


# =====================================================
# Top warehouse by processing time
# =====================================================
def top_warehouses_by_processing_time(df, top_k=3, period=None):
    df_filtered = _filter_by_period(df, period)

    avg_time = (
        df_filtered.groupby("warehouse")["delivery_time"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
    )

    result_text = f"🏭 Top {top_k} warehouses by processing time ({period or 'all time'}):\n"
    for wh, val in avg_time.items():
        result_text += f"  - {wh}: {val:.2f} days\n"

    fig, ax = plt.subplots()
    avg_time.plot(kind="bar", ax=ax, title=f"Top {top_k} Warehouses (Avg Delivery Time)")
    plt.tight_layout()
    return result_text, fig


# =====================================================
# Average delay by month
# =====================================================
def average_delay_by_month(df, period=None):
    df_filtered = _filter_by_period(df, period)
    avg_delay = df_filtered.groupby(df_filtered["date"].dt.to_period("M"))[
        "delay_minutes"
    ].mean()

    result_text = f"📊 Average delay by month ({period or 'all time'}):\n"
    for period_label, val in avg_delay.items():
        result_text += f"  - {period_label}: {val:.1f} min\n"

    fig, ax = plt.subplots()
    avg_delay.plot(kind="line", ax=ax, marker="o", title="Average Delay by Month")
    plt.tight_layout()
    return result_text, fig


# =====================================================
# Delay reason summary
# =====================================================
def delay_reason_summary(df, period=None):
    df_filtered = _filter_by_period(df, period)
    delay_reason = (
        df_filtered.groupby("delay_reason")["delay_minutes"]
        .sum()
        .sort_values(ascending=False)
    )

    result_text = f"🛠️ Total delay minutes by reason ({period or 'all time'}):\n"
    for reason, total in delay_reason.items():
        result_text += f"  - {reason}: {total:.0f} min\n"

    fig, ax = plt.subplots()
    delay_reason.plot(kind="bar", ax=ax, title="Delay Minutes by Reason")
    plt.tight_layout()
    return result_text, fig


# =====================================================
# Predict average delay for next week (Simple Linear Regression)
# =====================================================
from sklearn.linear_model import LinearRegression
import numpy as np


def predict_next_week_delay(df):
    df = df.copy()
    df["week"] = df["date"].dt.isocalendar().week
    weekly_avg = df.groupby("week")["delay_minutes"].mean().reset_index()

    X = weekly_avg["week"].values.reshape(-1, 1)
    y = weekly_avg["delay_minutes"].values
    model = LinearRegression()
    model.fit(X, y)

    next_week = np.array([[weekly_avg["week"].max() + 1]])
    pred = model.predict(next_week)[0]

    result_text = f"📈 Predicted average delay for next week: {pred:.1f} minutes"

    fig, ax = plt.subplots()
    ax.plot(weekly_avg["week"], weekly_avg["delay_minutes"], marker="o", label="Historical")
    ax.plot(next_week, [pred], "r*", markersize=12, label="Predicted")
    ax.set_title("Weekly Average Delay Prediction")
    ax.legend()
    plt.tight_layout()
    return result_text, fig


# =====================================================
# Helper: Filter DataFrame by period
# =====================================================
def _filter_by_period(df, period):
    if not period:
        return df

    period = period.lower().strip()
    now = datetime.now()

    # "last week"
    if "last week" in period:
        start = now - timedelta(days=7)
        return df[df["date"] >= start]

    # "last month"
    if "last month" in period:
        start = now - timedelta(days=30)
        return df[df["date"] >= start]

    # "next week" / "next month"
    if "next" in period:
        return df

    # "October" or "2024-10"
    try:
        if re.search(r"\d{4}-\d{2}", period):  # YYYY-MM
            month = datetime.strptime(period, "%Y-%m")
            return df[df["date"].dt.month == month.month]
        elif any(month in period for month in [
            "january","february","march","april","may","june",
            "july","august","september","october","november","december"
        ]):
            for i, m in enumerate([
                "january","february","march","april","may","june",
                "july","august","september","october","november","december"
            ], 1):
                if m in period:
                    return df[df["date"].dt.month == i]
    except Exception:
        pass

    # if no match, return the entire DataFrame
    return df


# =====================================================
# TEST LOCAL
# =====================================================
if __name__ == "__main__":
    df = load_data("D:/AI/Hermes/Data/shipments.csv")

    for func, kwargs in [
        (route_with_most_delays, {"period": "last week"}),
        (top_warehouses_by_processing_time, {"top_k": 5}),
        (average_delay_by_month, {"period": "October"}),
        (delay_reason_summary, {"period": "October"}),
        (predict_next_week_delay, {}),
    ]:
        text, fig = func(df, **kwargs)
        print(text)
        plt.show()
