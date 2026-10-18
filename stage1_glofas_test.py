"""Which GloFAS point actually responds to rain over our basin?"""
import pandas as pd
from config import DATA_PROCESSED

df = pd.read_csv(DATA_PROCESSED / "stage1_daily.csv", parse_dates=["date"]).set_index("date")
monsoon = df[df.index.month.isin([6, 7, 8, 9])]
rain3 = monsoon["precip_mm"].rolling(3).sum()

print("Correlation with 3-day basin rain (monsoon days), by lag:")
for col in ["q_north", "q_west"]:
    corrs = [rain3.corr(monsoon[col].shift(-lag)) for lag in range(4)]
    print(f"  {col}: " + "  ".join(f"lag{l}={c:.2f}" for l, c in enumerate(corrs)))

print("\nSept 2024 daily discharge (m3/s):")
print(df.loc["2024-09-24":"2024-10-02", ["precip_mm", "q_north", "q_west"]].round(1))