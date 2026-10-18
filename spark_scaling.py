"""Measured local[*] hourly-to-daily scaling; independent synthetic replicas.

Replicas are generated in Spark, not written as expanded raw files. The exact
same daily aggregation is forced to completion through counts and checksums.
"""
import json
import statistics
import time
from pyspark.sql import functions as f
from spark_runtime import session
from spark_pipeline import LAKE, OUT, aggregate_daily


def main():
    spark = session('Bagmati scaling benchmark')
    # Fifty replicas exceed a 3 GB heap with ten large shuffle partitions.
    # Keep one bounded-memory configuration for every reported data scale.
    spark.conf.set('spark.sql.shuffle.partitions','100')
    spark.conf.set('spark.sql.adaptive.coalescePartitions.enabled','false')
    try:
        raw = spark.read.parquet(str(LAKE/'raw')).select('timestamp','lat','lon','variable','value','source')
        n = raw.count()
        def run(scale):
            replicas = spark.range(scale).select(f.col('id').alias('replica_id'))
            expanded = (raw.crossJoin(f.broadcast(replicas))
                        .withColumn('source',f.concat_ws('#replica=',f.col('source'),f.col('replica_id'))))
            start = time.perf_counter()
            result = aggregate_daily(expanded).agg(f.count('*').alias('daily_rows'),f.count('value').alias('valid_daily_rows'),
                       f.sum('value').alias('checksum'),f.sum(f.col('value')*f.col('value')).alias('squared_checksum')).first().asDict()
            elapsed = time.perf_counter()-start
            return dict(scale=scale,input_rows=n*scale,seconds=elapsed,**result)
        print('Untimed JVM/OS-cache warm-up at 1x',flush=True)
        warmup = run(1)
        results = []
        for scale in (1,10,50):
            for trial in range(1,4):
                row = run(scale)
                row['trial'] = trial
                assert row['daily_rows'] == warmup['daily_rows']*scale
                assert row['valid_daily_rows'] == warmup['valid_daily_rows']*scale
                assert abs(row['checksum']/scale-warmup['checksum']) <= max(1e-6,abs(warmup['checksum'])*1e-10)
                results.append(row)
                (OUT/'scaling_trials.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(row),flush=True)
        groups = {scale:[r['seconds'] for r in results if r['scale']==scale] for scale in (1,10,50)}
        baseline = statistics.median(groups[1])
        lines = ['# Spark local scaling benchmark','',f'Spark {spark.version}; `{spark.sparkContext.master}`; {spark.sparkContext.defaultParallelism} CPU slots; Java 17; 3 GB driver heap; 100 shuffle partitions at every scale; adaptive partition coalescing disabled.',
                 '', 'Workload: Parquet scan → shifted complete point-day aggregation → nine-point basin aggregation → forced count and value checksums. Three measured trials per scale after one untimed 1x warm-up; median wall-clock time is reported.',
                 '', '| Data scale | Input rows | Median seconds | Trial seconds | Runtime vs 1x |', '|---|---:|---:|---|---:|']
        for scale in (1,10,50):
            median = statistics.median(groups[scale])
            trials = ', '.join(f'{x:.3f}' for x in groups[scale])
            lines.append(f'| {scale}x | {n*scale:,} | {median:.3f} | {trials} | {median/baseline:.2f}x |')
        lines += ['', 'The 10x/50x datasets are synthetic independent replicas, separated by a replica suffix in the source grouping key so that aggregation cannot collapse them. No new observations are implied. Replicas are generated in memory from the same Parquet files; expanded datasets are not stored on disk. Timings include Parquet reads, replication, shuffles and aggregation, but exclude session startup, raw JSON conversion, seasonal percentiles, output writes and model scoring. OS file-cache effects remain; this is a local aggregation benchmark, not an end-to-end pipeline or cluster scalability claim. All trials verify row and value-checksum scaling.', '', 'Initial attempt: 50x failed with Java heap exhaustion using the pipeline default of 10 shuffle partitions and a 3 GB heap. Its logs and completed smaller trials are preserved as scaling_initial_10_partitions*. The reported table reruns every scale with 100 shuffle partitions and the same 3 GB heap. This is a real memory limitation of the initial configuration, not a successful initial 50x result.', '']
        (OUT/'scaling.md').write_text('\n'.join(lines))
        print('\n'.join(lines),flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
