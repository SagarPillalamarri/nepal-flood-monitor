"""Spark lake ingestion, complete-day aggregation, windows and exact seasonal ranks.

No scoring/model files are written. Daily-only Stage 1 supplements are identified
explicitly when hourly recovery is limited; they are never represented as hourly.
"""
import contextlib
import io
import json
from pathlib import Path
from floodwatch.core import config as c
from floodwatch.spark.spark_runtime import session

LAKE = c.ROOT/'data/lake'
OUT = c.ROOT/'outputs/spark'


def inventory():
    files = []
    for root in ('data/raw', 'data/live', 'outputs/live_preflight'):
        for path in sorted((c.ROOT/root).rglob('*.json')):
            data = json.loads(path.read_text())
            points = data if isinstance(data, list) else [data]
            hourly = [p for p in points if isinstance(p, dict) and isinstance(p.get('hourly'), dict)]
            if not hourly:
                continue
            files.append(dict(path=str(path), bytes=path.stat().st_size, points=len(hourly),
                              variables=sorted(set(v for p in hourly for v in p['hourly'] if v != 'time'))))
    return files


def ingest(spark):
    from pyspark.sql import functions as f, types as t
    files = inventory()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'hourly_inventory.json').write_text(json.dumps(files, indent=2))
    point = t.StructType([t.StructField('latitude', t.DoubleType()), t.StructField('longitude', t.DoubleType()),
                          t.StructField('hourly', t.MapType(t.StringType(), t.ArrayType(t.StringType())))])
    texts = spark.read.option('wholetext', True).text([x['path'] for x in files]).withColumn('source_file', f.input_file_name())
    docs = texts.withColumn('points', f.when(f.trim('value').startswith('['), f.from_json('value', t.ArrayType(point)))
                            .otherwise(f.array(f.from_json('value', point))))
    points = docs.select('source_file', f.size('points').alias('n_points'), f.posexplode('points').alias('point_index','point'))
    lat = f.array(*[f.lit(x[0]) for x in c.BASIN_POINTS])
    lon = f.array(*[f.lit(x[1]) for x in c.BASIN_POINTS])
    points = (points.withColumn('lat', f.when(f.col('n_points') == 9, f.element_at(lat, f.col('point_index')+1)).otherwise(f.lit(c.BASIN_POINTS[0][0])))
              .withColumn('lon', f.when(f.col('n_points') == 9, f.element_at(lon, f.col('point_index')+1)).otherwise(f.lit(c.BASIN_POINTS[0][1]))))
    variables = points.select('*', f.explode(f.map_entries('point.hourly')).alias('entry')).where("entry.key <> 'time'")
    hours = variables.select('source_file','lat','lon', f.col('point.latitude').alias('grid_lat'), f.col('point.longitude').alias('grid_lon'),
                             f.col('entry.key').alias('variable'), f.col('entry.value').alias('values'),
                             f.posexplode(f.col('point.hourly')['time']).alias('hour_index','local_timestamp'))
    hours = (hours.withColumn('timestamp', f.to_timestamp('local_timestamp'))
             .withColumn('value', f.element_at('values', f.col('hour_index')+1).cast('double'))
             .withColumn('year', f.year('timestamp'))
             .withColumn('source', f.when(f.col('source_file').contains('/stage1b/'), 'era5_tcwv')
                         .when(f.col('source_file').contains('/spark_weather/'), 'stage1_best_match')
                         .when(f.col('source_file').contains('/live_preflight/'), 'gfs_probe').otherwise('other_hourly_cache'))
             .select('timestamp','lat','lon','variable','value','year','source','source_file','grid_lat','grid_lon'))
    hours.repartition('variable','year').write.mode('overwrite').partitionBy('variable','year').option('compression','snappy').parquet(str(LAKE/'raw'))
    raw = spark.read.parquet(str(LAKE/'raw'))
    count = raw.count()
    counts = [r.asDict() for r in raw.groupBy('source','variable').agg(f.count('*').alias('rows'),f.count('value').alias('non_null')).collect()]
    parquet_files = list((LAKE/'raw').rglob('*.parquet'))
    partitions = {str(p.parent.relative_to(LAKE/'raw')) for p in parquet_files}
    stats = dict(rows=count, partitions=len(partitions), parquet_files=len(parquet_files),
                 parquet_bytes=sum(p.stat().st_size for p in parquet_files),
                 raw_json_bytes=sum(x['bytes'] for x in files), cached_hourly_files=len(files), variables=counts)
    (OUT/'lake_stats.json').write_text(json.dumps(stats, indent=2))
    print('LAKE:', json.dumps(stats, indent=2), flush=True)
    return raw


def aggregate_daily(raw, precipitation_end_hour=True):
    from pyspark.sql import functions as f
    # Rainfall timestamps label the END of the preceding accumulation hour.
    # Moving the grouping clock back one hour assigns next midnight to today.
    local_clock = f.when(f.col('variable') == 'precipitation', f.col('timestamp')-f.expr('INTERVAL 1 HOUR')).otherwise(f.col('timestamp')) if precipitation_end_hour else f.col('timestamp')
    point = (raw.withColumn('date', f.to_date(local_clock))
             .groupBy('date','lat','lon','variable','source')
             .agg(f.countDistinct('timestamp').alias('hour_count'),f.count('value').alias('valid_hours'),
                  f.avg('value').alias('hour_mean'),f.sum('value').alias('hour_sum'))
             .withColumn('complete', (f.col('hour_count') == 24) & (f.col('valid_hours') == 24))
             .withColumn('point_daily', f.when(f.col('complete'), f.when(f.col('variable') == 'precipitation', f.col('hour_sum')).otherwise(f.col('hour_mean')))))
    return (point.groupBy('date','variable','source')
            .agg(f.count('*').alias('n_points'),f.sum(f.col('complete').cast('int')).alias('complete_points'),
                 f.avg('point_daily').alias('available_mean'),f.max('point_daily').alias('available_max'))
            .withColumn('value', f.when((f.col('n_points') == 9) & (f.col('complete_points') == 9), f.col('available_mean')))
            .withColumn('max_point_value', f.when((f.col('n_points') == 9) & (f.col('complete_points') == 9), f.col('available_max'))))


def daily_features(spark, daily):
    from pyspark.sql import functions as f, Window
    base = (spark.read.option('header', True).option('inferSchema', True).csv(str(c.STAGE1B_FILE))
            .withColumn('date', f.to_date('date')))
    # Daily Stage 1 data remain legitimate cached measurements, explicitly
    # distinguished from newly recovered hourly aggregates at every date.
    mapping = {'precipitation':'precip_mm','soil_moisture_0_to_7cm':'sm_0_7',
               'soil_moisture_7_to_28cm':'sm_7_28','soil_moisture_28_to_100cm':'sm_28_100',
               'total_column_integrated_water_vapour':'tcwv_mm'}
    for variable, column in mapping.items():
        source = 'era5_tcwv' if column == 'tcwv_mm' else 'stage1_best_match'
        recovered = daily.where((f.col('variable') == variable) & (f.col('source') == source)).select('date', f.col('value').alias('recovered'), f.lit(True).alias('has_hourly'))
        if column == 'precip_mm':
            base = (base.join(recovered,'date','left')
                    .withColumn('hourly_precip_mm',f.col('recovered'))
                    .withColumn('precip_mm_provenance',f.lit('cached_stage1_daily_api_total'))
                    .drop('recovered','has_hourly'))
            continue
        base = (base.join(recovered,'date','left')
                .withColumn(column, f.when(f.col('has_hourly'), f.col('recovered')).otherwise(f.col(column)))
                .withColumn(column+'_provenance', f.when(f.col('has_hourly'), f.lit('hourly_complete_day')).otherwise(f.lit('cached_stage1_daily')))
                .drop('recovered','has_hourly'))
    maxrain = daily.where((f.col('variable') == 'precipitation') & (f.col('source') == 'stage1_best_match')).select('date',f.col('max_point_value').alias('recovered'),f.lit(True).alias('has_hourly'))
    base = base.join(maxrain,'date','left').withColumn('hourly_precip_max_point_mm',f.col('recovered')).drop('recovered','has_hourly')
    base = base.withColumn('_day',f.datediff('date',f.lit('2000-01-01')))
    for days in (1,3,7,14):
        window = Window.orderBy('_day').rangeBetween(-(days-1),0)
        base = base.withColumn(f'rain_{days}d', f.when(f.count('precip_mm').over(window) == days,f.sum('precip_mm').over(window)))
        base = base.withColumn(f'hourly_rain_{days}d', f.when(f.count('hourly_precip_mm').over(window) == days,f.sum('hourly_precip_mm').over(window)))
    signals = {**{f'rain_{d}d':f'rain_{d}d_pct' for d in (1,3,7,14)},'tcwv_mm':'tcwv_pct','sm_0_7':'sm_0_7_pct','precip_max_point_mm':'rain_max_point_pct'}
    base = base.withColumn('calendar_day',f.dayofyear(f.to_date(f.concat(f.lit('2000-'),f.date_format('date','MM-dd'))))).cache()
    base.count()
    long = base.select('date','calendar_day', f.explode(f.array(*[f.struct(f.lit(name).alias('signal'),f.round(f.col(name),8).alias('x')) for name in signals])).alias('s')).select('date','calendar_day','s.*')
    reference = long.where((f.col('date') <= '2019-12-31') & (f.col('date') >= '2001-01-01') & f.month('date').between(6,10) & f.col('x').isNotNull())
    reference.write.mode('overwrite').parquet(str(LAKE/'seasonal_reference'))
    # Only 153 calendar bins and small frozen reference pools are broadcast.
    days = base.where(f.month('date').between(6,10)).select('calendar_day').distinct().withColumnRenamed('calendar_day','query_day')
    pools = (days.crossJoin(reference.select('calendar_day','signal','x'))
             .where(f.least(f.abs(f.col('calendar_day')-f.col('query_day')),366-f.abs(f.col('calendar_day')-f.col('query_day'))) <= 15)
             .groupBy('query_day','signal').agg(f.collect_list('x').alias('pool')))
    ranked = (long.where(f.month('date').between(6,10)).alias('q').join(f.broadcast(pools).alias('p'), (f.col('q.calendar_day') == f.col('p.query_day')) & (f.col('q.signal') == f.col('p.signal')))
              .select(f.col('q.date').alias('date'), f.col('q.signal').alias('signal'), f.when(f.col('q.x').isNotNull() & (f.size('pool') >= 30),
                       f.expr('100.0 * (size(filter(pool, p -> p < x)) + size(filter(pool, p -> p <= x))) / (2 * size(pool))')).alias('percentile')))
    pct = ranked.groupBy('date').pivot('signal',list(signals)).agg(f.first('percentile'))
    for name, output in signals.items():
        pct = pct.withColumnRenamed(name, output)
    result = base.join(pct,'date','left').drop('_day','calendar_day').withColumn('year',f.year('date'))
    result.write.mode('overwrite').partitionBy('year').parquet(str(LAKE/'features_daily'))
    base.unpersist()
    return spark.read.parquet(str(LAKE/'features_daily'))


def queries(spark, features):
    features.createOrReplaceTempView('features_daily')
    events = spark.read.option('header',True).option('inferSchema',True).csv(str(c.STAGE2_EVENTS))
    events.createOrReplaceTempView('flood_events')
    labels = spark.read.option('header',True).option('inferSchema',True).csv(str(c.STAGE2_DAILY))
    labels.createOrReplaceTempView('flood_daily')
    sql = (c.ROOT/'floodwatch/spark/spark_queries.sql').read_text()
    for block in sql.split('-- QUERY: ')[1:]:
        name, statement = block.split('\n',1)
        result = spark.sql(statement.strip().rstrip(';'))
        result.toPandas().to_csv(OUT/f'{name.strip()}.csv',index=False)
        print(f'QUERY {name}',flush=True)
        result.show(30,truncate=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spark = session()
    try:
        import sys
        raw = spark.read.parquet(str(LAKE/'raw')) if '--reuse-raw' in sys.argv else ingest(spark)
        daily = aggregate_daily(raw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            daily.explain(mode='formatted')
        (OUT/'daily_physical_plan.txt').write_text(buf.getvalue())
        print(buf.getvalue(),flush=True)
        print('SHUFFLES: Exchange hashpartitioning(date,lat,lon,variable,source) aggregates hourly point-days; a second Exchange hashpartitioning(date,variable,source) combines the nine points.',flush=True)
        daily.write.mode('overwrite').parquet(str(LAKE/'daily_aggregates'))
        daily = spark.read.parquet(str(LAKE/'daily_aggregates'))
        features = daily_features(spark,daily)
        queries(spark,features)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
