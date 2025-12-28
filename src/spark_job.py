import os
import argparse
import logging
import pandas as pd
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from detectors.statistical_detector import StatisticalAnomalyDetector

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/spark_anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def build_spark(master: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("BigDataPJ-Spark-AnomalyDetection")
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark


def normalize_methods(methods: List[str]) -> List[str]:
    mapped = []
    for m in methods:
        if m == 'ensemble_stat':
            mapped.append('ensemble')
        elif m in ['zscore', 'modified_zscore', 'iqr', 'moving_average', 'ema', 'seasonal', 'grubbs']:
            mapped.append(m)
    return list(dict.fromkeys(mapped))


def detect_group_impl(pdf: pd.DataFrame, method: str, detector: StatisticalAnomalyDetector) -> pd.DataFrame:
    pdf = pdf.sort_values('timestamp')
    res = detector.detect(pdf[['timestamp', 'value']].copy(), method=method)
    res['cmdb_id'] = pdf['cmdb_id'].iloc[0]
    res['kpi_name'] = pdf['kpi_name'].iloc[0]
    return res[['timestamp', 'value', 'cmdb_id', 'kpi_name', 'is_anomaly', 'anomaly_score']]


def run_spark_job(data_dir: str, output_dir: str, methods: List[str], master: str, partitions: int):
    spark = build_spark(master)
    detector = StatisticalAnomalyDetector(contamination=0.1)

    pattern = os.path.join(data_dir, "**", "metric", "*", "*.csv")
    df = spark.read.option("header", True).csv(pattern)

    if 'timestamp' in df.columns:
        df = df.withColumn('timestamp', F.to_timestamp(
            F.from_unixtime(F.col('timestamp').cast('long'))))

    required_cols = ['timestamp', 'value', 'cmdb_id', 'kpi_name']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")

    if partitions and partitions > 0:
        df = df.repartition(partitions, F.col('cmdb_id'), F.col('kpi_name'))

    logger.info(f"读取数据分区数: {df.rdd.getNumPartitions()}")

    schema = T.StructType([
        T.StructField('timestamp', T.TimestampType(), False),
        T.StructField('value', T.DoubleType(), True),
        T.StructField('cmdb_id', T.StringType(), False),
        T.StructField('kpi_name', T.StringType(), False),
        T.StructField('is_anomaly', T.BooleanType(), False),
        T.StructField('anomaly_score', T.DoubleType(), False),
    ])

    def detect_group(pdf: pd.DataFrame, method: str) -> pd.DataFrame:
        return detect_group_impl(pdf, method, detector)

    methods = normalize_methods(methods)
    os.makedirs(os.path.join(output_dir, 'spark_results'), exist_ok=True)

    for method in methods:
        pandas_udf = F.pandas_udf(
            lambda pdf: detect_group(pdf, method), schema)
        result = (
            df.groupBy('cmdb_id', 'kpi_name')
              .apply(pandas_udf)
        )

        out_path = os.path.join(output_dir, 'spark_results', method)
        (
            result.write
                  .mode('overwrite')
                  .option('header', True)
                  .partitionBy('cmdb_id', 'kpi_name')
                  .csv(out_path)
        )

        summary = (
            result.groupBy('cmdb_id', 'kpi_name')
            .agg(F.sum(F.col('is_anomaly').cast('int')).alias('anomaly_count'),
                 F.count(F.lit(1)).alias('total'))
        )

        summary_path = os.path.join(
            output_dir, 'spark_results', f'summary_{method}.csv')
        (
            summary.coalesce(1)
                   .write
                   .mode('overwrite')
                   .option('header', True)
                   .csv(summary_path)
        )

    spark.stop()


def parse_args():
    parser = argparse.ArgumentParser(description='Spark 分布式异常检测')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['ensemble_stat'])
    parser.add_argument('--master', type=str, default='local[*]')
    parser.add_argument('--partitions', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    run_spark_job(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        methods=args.methods,
        master=args.master,
        partitions=args.partitions,
    )


if __name__ == '__main__':
    main()
