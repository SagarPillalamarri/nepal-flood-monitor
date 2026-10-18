# Spark local scaling benchmark

Spark 4.0.1; `local[*]`; 10 CPU slots; Java 17; 3 GB driver heap; 100 shuffle partitions at every scale; adaptive partition coalescing disabled.

Workload: Parquet scan → shifted complete point-day aggregation → nine-point basin aggregation → forced count and value checksums. Three measured trials per scale after one untimed 1x warm-up; median wall-clock time is reported.

| Data scale | Input rows | Median seconds | Trial seconds | Runtime vs 1x |
|---|---:|---:|---|---:|
| 1x | 5,128,008 | 5.328 | 5.152, 5.342, 5.328 | 1.00x |
| 10x | 51,280,080 | 30.452 | 30.452, 31.693, 29.721 | 5.72x |
| 50x | 256,400,400 | 235.842 | 293.033, 206.815, 235.842 | 44.27x |

The 10x/50x datasets are synthetic independent replicas, separated by a replica suffix in the source grouping key so that aggregation cannot collapse them. No new observations are implied. Replicas are generated in memory from the same Parquet files; expanded datasets are not stored on disk. Timings include Parquet reads, replication, shuffles and aggregation, but exclude session startup, raw JSON conversion, seasonal percentiles, output writes and model scoring. OS file-cache effects remain; this is a local aggregation benchmark, not an end-to-end pipeline or cluster scalability claim. All trials verify row and value-checksum scaling.

Initial attempt: 50x failed with Java heap exhaustion using the pipeline default of 10 shuffle partitions and a 3 GB heap. Its logs and completed smaller trials are preserved as scaling_initial_10_partitions*. The reported table reruns every scale with 100 shuffle partitions and the same 3 GB heap. This is a real memory limitation of the initial configuration, not a successful initial 50x result.
