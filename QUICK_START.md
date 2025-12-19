# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Code

### Option 1: Simple Example (Recommended for First Run)

```bash
python example.py
```

This will:
- Automatically load the first cloudbed data
- Run 3 different detection methods
- Generate all visualizations and reports in `results/` folder

### Option 2: Main Script with More Options

```bash
# Basic run with default settings
python src/main.py

# Analyze specific cloudbed
python src/main.py --cloudbed 2022-03-20-cloudbed1

# Compare multiple methods
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --methods zscore iqr isolation_forest lof --compare

# Analyze specific service and metric
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --service "adservice.ts:8088" \
    --kpi "jvm_threads_current"
```

## Available Detection Methods

### Statistical (Fast, Simple)
- `zscore` - Standard deviation based
- `modified_zscore` - Robust version using MAD
- `iqr` - Interquartile range method
- `moving_average` - Moving average deviation
- `ema` - Exponential moving average
- `seasonal` - Seasonal decomposition
- `ensemble_stat` - Combines all statistical methods

### Machine Learning (More Accurate, Slower)
- `isolation_forest` - Tree-based isolation (recommended)
- `lof` - Local Outlier Factor
- `knn` - K-Nearest Neighbors
- `ocsvm` - One-Class SVM
- `hbos` - Histogram-based
- `ensemble_ml` - Combines all ML methods

## Output Files

All results are saved in the `results/` directory:
- `*_plot.png` - Visualization of time series with anomalies
- `*_anomalies.csv` - Detected anomaly data points
- `*_report.txt` - Statistical summary
- `methods_comparison.png` - Compare different methods
- `anomaly_heatmap.png` - Pattern heatmap

## IoTDB Integration (Optional)

If you want to use IoTDB for efficient storage:

```bash
# First, ingest data into IoTDB
python src/main.py --cloudbed 2022-03-20-cloudbed1 \
    --use-iotdb --ingest-to-iotdb

# Then run analysis using IoTDB
python src/main.py --cloudbed 2022-03-20-cloudbed1 --use-iotdb
```

## Tips

1. Start with `example.py` to see basic functionality
2. Use `ensemble_stat` for fast, reliable results
3. Use `isolation_forest` for better accuracy with ML
4. Use `--compare` to evaluate multiple methods side-by-side
5. Check `results/` folder for all outputs

## Common Use Cases

### Quick Analysis
```bash
python example.py
```

### Compare All Statistical Methods
```bash
python src/main.py --methods zscore modified_zscore iqr \
    moving_average ema --compare
```

### Best Accuracy
```bash
python src/main.py --methods ensemble_ml isolation_forest lof --compare
```

### Specific Service Analysis
```bash
# First, list available services
python -c "
import sys; sys.path.append('src')
from utils.data_loader import TimeSeriesDataLoader
loader = TimeSeriesDataLoader('.')
data = loader.load_all_cloudbeds()
df = list(data.values())[0]
print('Services:', loader.get_all_services(df)[:5])
print('KPIs:', loader.get_all_kpis(df)[:5])
"

# Then analyze specific one
python src/main.py --service "SERVICE_NAME" --kpi "KPI_NAME"
```
