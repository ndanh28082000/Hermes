import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- Config ---
num_rows = 1000
start_date = datetime(2024, 8, 28)
end_date = datetime(2025, 11, 30)

routes = ["Route A", "Route B", "Route C", "Route D"]
warehouses = ["WH1", "WH2", "WH3", "WH4"]
delay_reasons = ["Weather", "Traffic", "Staff", "None"]

# --- Generate random data ---
data = []
for i in range(1, num_rows + 1):
    route = random.choice(routes)
    warehouse = random.choice(warehouses)
    delay_reason = random.choice(delay_reasons)
    date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    delivery_time = round(random.uniform(3.0, 8.0), 1)
    
    if delay_reason == "None":
        delay_minutes = 0
    else:
        delay_minutes = random.randint(10, 180)
    
    data.append([i, route, warehouse, delivery_time, delay_minutes, delay_reason, date.date()])

# --- Create DataFrame ---
df = pd.DataFrame(data, columns=[
    "id", "route", "warehouse", "delivery_time", "delay_minutes", "delay_reason", "date"
])

# --- Save to CSV ---
df.to_csv("D:/AI/Hermes/Data/shipments.csv", index=False)
print("Created shipments.csv with", len(df), "rows.")
df.head()
