"""Project-local Spark setup; no system Java or shell profile modifications."""
import os
from pathlib import Path

JAVA = Path('/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home')
if not JAVA.exists():
    raise RuntimeError('Install Java 17 first: brew install openjdk@17')
os.environ['JAVA_HOME'] = str(JAVA)
os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')


def session(name='Bagmati data lake'):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master('local[*]').appName(name)
             .config('spark.sql.session.timeZone', 'Asia/Kathmandu')
             .config('spark.sql.shuffle.partitions', str(max(8, os.cpu_count() or 8)))
             .config('spark.sql.parquet.compression.codec', 'snappy')
             .config('spark.driver.memory', '3g')
             .config('spark.ui.port', '4040').getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    print(f'Spark {spark.version}; master={spark.sparkContext.master}; UI={spark.sparkContext.uiWebUrl}', flush=True)
    return spark
