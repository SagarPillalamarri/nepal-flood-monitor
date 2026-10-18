"""
Stage 1 check: is the downloaded data complete, physically sensible,
and does it show the September 2024 flood?

FAIL = something is broken, fix before moving on.
WARN = data works but has a weakness worth noting in the report.

Run:  python stage1_check.py
"""
import sys

import pandas as pd

from config import (CASE_STUDY_END, CASE_STUDY_START, DATA_PROCESSED,
                    END_DATE, GLOFAS_CHOSEN, START_DATE)

results = []


def report(status, msg):
    results.append(status)
    print(f"[{status}] {msg}")


def pct_rank(series, value):
    """Share of values in the series that are below `value`, as a percentage."""
    return (series < value).mean() * 100


def main():
    path = DATA_PROCESSED / "stage1_daily.csv"
    if not path.exists():
        print(f"[FAIL] {path} not found. Run stage1_fetch.py first.")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    q_cols = [c for c in df.columns if c.startswith("q_")]
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}\n")

    # 1. Coverage
    expected = pd.date_range(START_DATE, END_DATE, freq="D")
    missing = expected.difference(df.index)
    dupes = df.index.duplicated().sum()
    if len(missing) == 0 and dupes == 0:
        report("PASS", f"Full daily coverage {START_DATE} to {END_DATE}, no duplicates")
    else:
        report("FAIL", f"{len(missing)} missing days, {dupes} duplicate days")

    # 2. Nulls per column
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct == 0:
            continue
        status = "FAIL" if null_pct > 5 else "WARN"
        report(status, f"{col}: {null_pct:.1f}% missing")
    if not any(df[c].isna().any() for c in df.columns):
        report("PASS", "No missing values")

    # 3. Physical ranges
    bad_rain = ((df["precip_mm"] < 0) | (df["precip_mm"] > 500)).sum()
    report("PASS" if bad_rain == 0 else "FAIL", f"Rainfall in 0-500 mm/day ({bad_rain} bad values)")
    sm_cols = ["sm_0_7", "sm_7_28", "sm_28_100"]
    bad_sm = ((df[sm_cols] < 0) | (df[sm_cols] > 1)).sum().sum()
    report("PASS" if bad_sm == 0 else "FAIL", f"Soil moisture in 0-1 m3/m3 ({bad_sm} bad values)")
    bad_q = (df[q_cols] < 0).sum().sum()
    report("PASS" if bad_q == 0 else "FAIL", f"Discharge non-negative ({bad_q} bad values)")

    # 4. Monsoon signal: wet season should be far wetter than winter
    monsoon = df[df.index.month.isin([6, 7, 8, 9])]["precip_mm"].mean()
    winter = df[df.index.month.isin([12, 1, 2])]["precip_mm"].mean()
    ratio = monsoon / max(winter, 0.01)
    report("PASS" if ratio > 3 else "WARN",
           f"Monsoon rain {monsoon:.1f} mm/day vs winter {winter:.1f} mm/day (x{ratio:.0f})")

    # 5. Does the September 2024 flood show up?
    print(f"\n--- Case study: {CASE_STUDY_START} to {CASE_STUDY_END} ---")
    event = slice(CASE_STUDY_START, CASE_STUDY_END)

    rain3 = df["precip_mm"].rolling(3).sum()
    event_rain3 = rain3[event].max()
    rank = pct_rank(rain3.dropna(), event_rain3)
    report("PASS" if rank >= 99 else "WARN",
           f"Max 3-day basin rainfall {event_rain3:.0f} mm, higher than {rank:.2f}% of all 3-day windows")
    if rank < 99:
        print("       ERA5 may be smoothing the storm. Note this; DHM gauges will matter more.")

    print(f"\nGloFAS points (configured point: {GLOFAS_CHOSEN}):")
    for col in q_cols:
        s = df[col].dropna()
        peak = s[event].max()
        r = pct_rank(s, peak)
        print(f"  {col:10s} mean {s.mean():7.1f} m3/s | event peak {peak:7.1f} m3/s | "
              f"higher than {r:.2f}% of days")
    if GLOFAS_CHOSEN not in q_cols:
        report("FAIL", f"Configured GloFAS point {GLOFAS_CHOSEN} is missing from the data")
    else:
        chosen = df[GLOFAS_CHOSEN].dropna()
        chosen_rank = pct_rank(chosen, chosen[event].max())
        report("PASS" if chosen_rank >= 99 else "WARN",
               f"Configured GloFAS point: {GLOFAS_CHOSEN} "
               f"(event peak above {chosen_rank:.2f}% of days)")
        if chosen.mean() < 5:
            report("WARN", f"{GLOFAS_CHOSEN} mean flow is very low; GloFAS may have snapped to a tiny stream")

    # Summary
    n_fail, n_warn = results.count("FAIL"), results.count("WARN")
    print(f"\nSummary: {results.count('PASS')} PASS, {n_warn} WARN, {n_fail} FAIL")
    if n_fail:
        print("Fix the FAILs before Stage 2.")
        sys.exit(1)
    print("Stage 1 OK. Paste this output back into the chat.")


if __name__ == "__main__":
    main()
