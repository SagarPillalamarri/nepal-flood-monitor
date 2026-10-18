"""Offline test of precipitation interval endpoints; accepted totals read only."""
import json
from pyspark.sql import functions as f
from floodwatch.core import config as c
from floodwatch.spark.spark_runtime import session
from floodwatch.spark.spark_pipeline import LAKE, OUT, aggregate_daily


def main():
    spark = session('Rainfall interval diagnosis')
    try:
        raw = spark.read.parquet(str(LAKE/'raw')).where("variable='precipitation' AND source='stage1_best_match'")
        accepted = spark.read.option('header',True).option('inferSchema',True).csv(str(c.STAGE1B_FILE)).select(f.to_date('date').alias('date'),'precip_mm')
        rows = []
        comparisons = None
        for shift in (0,1):
            hourly = raw.withColumn('timestamp',f.col('timestamp')-f.expr(f'INTERVAL {shift} HOURS'))
            daily = aggregate_daily(hourly, precipitation_end_hour=False).select('date',f.col('value').alias('hourly_mm'))
            pair = daily.join(accepted,'date').where('hourly_mm IS NOT NULL').withColumn('difference',f.col('hourly_mm')-f.col('precip_mm')).cache()
            stats = pair.agg(f.count('*').alias('days'),f.percentile('difference',.5).alias('median'),f.percentile('difference',.95).alias('p95'),f.percentile(f.abs('difference'),.95).alias('p95_absolute'),f.max(f.abs('difference')).alias('max_absolute'),f.sum((f.round(f.abs('difference'),8)>1).cast('int')).alias('days_over_1mm'),f.sum((f.abs('difference')>1).cast('int')).alias('days_over_1mm_unrounded'),f.sum((f.abs('difference')>1e-6).cast('int')).alias('mismatches')).first().asDict()
            stats['shift_hours'] = shift
            rows.append(stats)
            print(json.dumps(stats),flush=True)
            pair.orderBy(f.abs('difference').desc()).limit(15).toPandas().to_csv(OUT/f'rain_largest_shift_{shift}.csv',index=False)
            pair.orderBy('date').toPandas().to_csv(OUT/f'rain_differences_shift_{shift}.csv',index=False)
            pair.unpersist()
        (OUT/'rainfall_diagnosis.json').write_text(json.dumps(rows,indent=2))
        lines = ['# Rainfall interval diagnosis','',
                 '| Metric | Original 00:00–23:00 | Shifted 01:00–24:00 |',
                 '|---|---:|---:|']
        for key,label in [('days','Complete days'),('median','Median signed difference (mm)'),('p95','95th percentile signed difference (mm)'),('p95_absolute','95th percentile absolute difference (mm)'),('max_absolute','Maximum absolute difference (mm)'),('days_over_1mm','Days with absolute difference >1 mm (rounded to 8 decimals)'),('days_over_1mm_unrounded','Unrounded floating-point >1 mm count'),('mismatches','Mismatches above 1e-6 mm')]:
            lines.append(f'| {label} | {rows[0][key]:.12g} | {rows[1][key]:.12g} |')
        lines += ['', 'The correction changes only the precipitation grouping clock. Stored timestamps, accepted daily totals, features.py and models remain unchanged; live mode stays paused. All nine points still require 24 valid hours. Existing adjacent-year caches supply midnight across year boundaries. The final cached day, 2010-12-31, lacks the following midnight and remains missing. No filling or additional download was performed.', '', 'The >1 mm count rounds to eight decimals to avoid counting one exactly-1-mm difference as greater than 1 due to floating-point noise.', '', '[Open-Meteo documentation](https://open-meteo.com/en/docs/historical-weather-api) defines precipitation as the preceding-hour sum.', '']
        (OUT/'rainfall_diagnosis.md').write_text('\n'.join(lines))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
