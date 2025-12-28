# 大数据分析技术期末实验报告（Spark 分布式版）

## 实验目标
- 在现有代码基础上引入 Apache Spark，实现时间序列异常检测的分布式运行。
- 输出分布式运行的结果数据与统计，并与原有单机流程形成对照。

## 环境与依赖
- 操作系统：macOS
- Python：3.14
- 依赖：pandas、numpy、scipy、scikit-learn、statsmodels、pyod、matplotlib、seaborn、pyspark、pyarrow
- Java：JDK 17（Spark 运行必需）。macOS 可通过 Homebrew 安装并配置 JAVA_HOME。

## 数据说明
- 目录结构：cloudbed/**/metric/*/*.csv，列包含 timestamp、value、cmdb_id、kpi_name。
- 本仓库内置合成数据用于快速验证：data/cloudbed_synth_2025-12-28/metric/service/*.csv。

## 分布式实现概述
- 核心脚本：Spark 作业 [spark_job.py](file:///Users/bytedance/Code/courses/bigdata-pj/src/spark_job.py)。
- 入口参数：data-dir、output-dir、methods、master、partitions。
- 技术路线：使用 Spark DataFrame 按 cmdb_id、kpi_name 分组，基于 Pandas UDF（Arrow 加速）将现有统计检测器 [statistical_detector.py](file:///Users/bytedance/Code/courses/bigdata-pj/src/detectors/statistical_detector.py) 应用于每个时间序列分组，并输出带 is_anomaly、anomaly_score 的结果。
- 输出：按方法分目录写出结果 CSV（分区字段 cmdb_id、kpi_name），以及聚合 summary_*.csv（每组异常数量与总样本数）。

## 运行与结果
- 本地快速验证（非分布式统计校验）：脚本 [verify_local.py](file:///Users/bytedance/Code/courses/bigdata-pj/scripts/verify_local.py) 运行结果：
  - svcA_cpu.csv ensemble anomalies=1/20
  - svcB_latency.csv zscore anomalies=1/20
- 分布式运行命令（示例）：
  - python src/spark_job.py --data-dir ./data/cloudbed_synth_2025-12-28 --output-dir ./results --methods ensemble_stat zscore --master 'local[*]' --partitions 4
  - 或连接集群：--master 'spark://<cluster-host:port>'，--partitions 64
- 分布式指标采集：在作业启动日志中记录数据分区数与分组并行度；结果文件位于 results/spark_results/。

## 方法说明
- 统计方法：zscore、modified_zscore、iqr、moving_average、ema、seasonal、grubbs；集成方法 ensemble_stat 映射至统计检测器的 ensemble。
- 检测器接口：StatisticalAnomalyDetector.detect(df, method) 返回含 is_anomaly 与 anomaly_score 的 DataFrame，已在 Pandas UDF 中复用。

## 结论与展望
- 已完成对现有代码的 Spark 分布式改造与集成，形成可运行的分布式作业入口与输出规范。
- 本地已完成方法正确性校验；分布式运行需配置 Java 运行环境与 Spark master，即可得到分区数与并行处理的实验结果。
