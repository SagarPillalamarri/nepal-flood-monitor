import pandas as pd
import requests

r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": 27.62, "longitude": 85.36,
    "start_date": "2024-01-01", "end_date": "2024-12-31",
    "hourly": "total_column_integrated_water_vapour",
    "models": "era5", "timezone": "Asia/Kathmandu"}, timeout=120)
print("HTTP", r.status_code)
h = r.json()["hourly"]
s = pd.Series(h["total_column_integrated_water_vapour"], index=pd.to_datetime(h["time"]))
print("total hours:", len(s), "| missing:", s.isna().sum())
print("missing hours per month:")
print(s.isna().groupby(s.index.month).sum().to_string())