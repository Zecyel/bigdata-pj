# Time Series Anomaly Detection

A comprehensive Python framework for detecting anomalies in time series data from IoT monitoring systems. This project supports both statistical and machine learning-based detection methods, with integration for Apache IoTDB.

## Project Overview

This system analyzes time series data from microservice monitoring metrics (JVM, containers, services, nodes, and Istio) to identify anomalous behavior. It provides multiple detection algorithms, visualization tools, and reporting capabilities.

## Features

- **Multiple Detection Methods**:
  - Statistical: Z-score, Modified Z-score, IQR, Moving Average, EMA, Seasonal Decomposition, Grubbs Test
  - Machine Learning: Isolation Forest, LOF, KNN, One-Class SVM, HBOS, PCA-based Autoencoder
  - Ensemble methods combining multiple algorithms

- **Data Management**:
  - CSV data loading from multiple cloudbed directories
  - Apache IoTDB integration for efficient time series storage
  - Support for multiple metric categories (JVM, container, service, node, istio)

- **Visualization & Reporting**:
  - Time series plots with anomaly highlights
  - Method comparison visualizations
  - Distribution analysis
  - Heatmaps showing detection patterns
  - CSV export of detected anomalies
  - Statistical summary reports

## Project Structure

```
.
|-- src/
|   |-- detectors/
|   |   |-- statistical_detector.py    # Statistical anomaly detection methods
|   |   |-- ml_detector.py             # Machine learning detection methods
|   |-- utils/
|   |   |-- data_loader.py             # Data loading from CSV files
|   |   |-- iotdb_connector.py         # IoTDB integration
|   |   |-- visualizer.py              # Visualization and reporting
|   |-- main.py                        # Main pipeline script
|-- example.py                         # Simple example script
|-- requirements.txt                   # Python dependencies
|-- results/                           # Output directory (generated)
|-- logs/                              # Log files (generated)
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Apache IoTDB is running (if using IoTDB features):
```bash
# IoTDB should be running on port 6667
```

## Usage

### Quick Start Example

Run the simple example script to analyze the first available cloudbed:

```bash
python example.py
```

This will:
- Load data from the first cloudbed
- Run multiple detection methods (Statistical Ensemble, Isolation Forest, Z-score)
- Generate visualizations and reports in the `results/` directory

### Advanced Usage

Use the main pipeline script for more control:

```bash
# Basic analysis with default methods
python src/main.py --cloudbed 2022-03-20-cloudbed1

# Analyze specific service and KPI
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --service "adservice.ts:8088" \
    --kpi "jvm_threads_current"

# Use specific detection methods
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --methods zscore iqr isolation_forest lof

# Compare multiple methods
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --methods zscore modified_zscore iqr isolation_forest \
    --compare

# Ingest data to IoTDB
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --use-iotdb --ingest-to-iotdb
```

### Available Detection Methods

**Statistical Methods**:
- `zscore`: Standard Z-score method
- `modified_zscore`: Robust Z-score using MAD
- `iqr`: Interquartile Range method
- `moving_average`: Moving average deviation
- `ema`: Exponential Moving Average deviation
- `seasonal`: Seasonal decomposition residuals
- `grubbs`: Grubbs test for outliers
- `ensemble_stat`: Ensemble of statistical methods

**Machine Learning Methods**:
- `isolation_forest`: Isolation Forest algorithm
- `lof`: Local Outlier Factor
- `knn`: K-Nearest Neighbors
- `ocsvm`: One-Class SVM
- `hbos`: Histogram-Based Outlier Score
- `autoencoder`: PCA-based reconstruction error
- `ensemble_ml`: Ensemble of ML methods

## Command Line Arguments

```
--data-dir DIR          Directory containing cloudbed data (default: .)
--output-dir DIR        Output directory for results (default: results)
--cloudbed NAME         Cloudbed name to analyze (default: 2022-03-20-cloudbed1)
--service ID            Service ID to analyze (default: first available)
--kpi NAME              KPI name to analyze (default: first available)
--methods M1 M2 ...     Detection methods to use (default: ensemble_stat isolation_forest)
--compare               Compare multiple methods
--use-iotdb             Use IoTDB for data storage
--iotdb-host HOST       IoTDB host (default: 127.0.0.1)
--iotdb-port PORT       IoTDB port (default: 6667)
--ingest-to-iotdb       Ingest data to IoTDB
```

## Output Files

The system generates several types of output files in the `results/` directory:

- `*_plot.png`: Time series visualization with anomalies highlighted
- `*_anomalies.csv`: CSV file containing detected anomalies
- `*_report.txt`: Statistical summary report
- `methods_comparison.png`: Comparison of multiple detection methods
- `anomaly_heatmap.png`: Heatmap showing detection patterns
- `comparison_report.csv`: Comparison statistics for different methods

## Algorithm Details

### Statistical Methods

1. **Z-score**: Detects points beyond k standard deviations from mean
2. **Modified Z-score**: Uses median and MAD for robustness
3. **IQR**: Identifies outliers beyond Q1-1.5*IQR and Q3+1.5*IQR
4. **Moving Average**: Detects deviation from rolling window average
5. **Seasonal Decomposition**: Analyzes residuals after removing trend and seasonality

### Machine Learning Methods

1. **Isolation Forest**: Isolates anomalies using random forests
2. **LOF**: Detects local density-based outliers
3. **KNN**: Uses distance to k-nearest neighbors
4. **One-Class SVM**: Learns boundary of normal data distribution
5. **HBOS**: Histogram-based outlier scoring

All ML methods use engineered features including:
- Current value and statistical measures (mean, std, min, max, median)
- First-order differences
- Lag features (t-1, t-2, t-5)

## Data Format

Input CSV files should have the following format:
```csv
timestamp,cmdb_id,kpi_name,value
1647766800,adservice.ts:8088,jvm_threads_current,17.0
```

Where:
- `timestamp`: Unix timestamp in seconds
- `cmdb_id`: Service identifier
- `kpi_name`: Metric name
- `value`: Metric value

## Performance Considerations

- For large datasets, use IoTDB integration for efficient storage and querying
- ML methods with feature extraction may be slower on very long time series
- Consider using ensemble methods for better accuracy at the cost of computation time
- Statistical methods are faster but may be less accurate for complex patterns

## Troubleshooting

1. **No data found**: Check that cloudbed directories contain metric/category/csv files
2. **IoTDB connection failed**: Verify IoTDB is running on the specified port
3. **Import errors**: Ensure all dependencies are installed via requirements.txt
4. **Memory issues**: Process smaller time windows or individual services

## License

This project is for educational purposes as part of a Big Data course.

## References

- Statistical anomaly detection methods are based on standard statistical theory
- ML methods use implementations from scikit-learn and PyOD libraries
- IoTDB integration uses the official Apache IoTDB Python client
