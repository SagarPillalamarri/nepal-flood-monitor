# Rainfall interval diagnosis

| Metric | Original 00:00–23:00 | Shifted 01:00–24:00 |
|---|---:|---:|
| Complete days | 3652 | 3651 |
| Median signed difference (mm) | 0 | 0 |
| 95th percentile signed difference (mm) | 1 | 1.7763568394e-15 |
| 95th percentile absolute difference (mm) | 1.64333333333 | 3.5527136788e-15 |
| Maximum absolute difference (mm) | 15.3111111111 | 2.84217094304e-14 |
| Days with absolute difference >1 mm (rounded to 8 decimals) | 365 | 0 |
| Unrounded floating-point >1 mm count | 366 | 0 |
| Mismatches above 1e-6 mm | 1874 | 0 |

The correction changes only the precipitation grouping clock. Stored timestamps, accepted daily totals, features.py and models remain unchanged; live mode stays paused. All nine points still require 24 valid hours. Existing adjacent-year caches supply midnight across year boundaries. The final cached day, 2010-12-31, lacks the following midnight and remains missing. No filling or additional download was performed.

The >1 mm count rounds to eight decimals to avoid counting one exactly-1-mm difference as greater than 1 due to floating-point noise.

[Open-Meteo documentation](https://open-meteo.com/en/docs/historical-weather-api) defines precipitation as the preceding-hour sum.
