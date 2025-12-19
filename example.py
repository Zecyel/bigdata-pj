"""
Simple example script for time series anomaly detection.
"""

import sys
sys.path.append('src')

import logging
from utils.data_loader import TimeSeriesDataLoader
from detectors.statistical_detector import StatisticalAnomalyDetector
from detectors.ml_detector import MLAnomalyDetector
from utils.visualizer import AnomalyVisualizer, AnomalyReporter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a simple example."""
    logger.info("Starting anomaly detection example...")

    # Initialize components
    data_loader = TimeSeriesDataLoader('.')
    stat_detector = StatisticalAnomalyDetector(contamination=0.1)
    ml_detector = MLAnomalyDetector(contamination=0.1)
    visualizer = AnomalyVisualizer('results')
    reporter = AnomalyReporter('results')

    # Load data
    logger.info("Loading cloudbed data...")
    cloudbed_data = data_loader.load_all_cloudbeds()

    if not cloudbed_data:
        logger.error("No cloudbed data found!")
        return

    # Use first cloudbed
    cloudbed_name = list(cloudbed_data.keys())[0]
    df = cloudbed_data[cloudbed_name]
    logger.info(f"Using cloudbed: {cloudbed_name}")

    # Get first service and KPI
    services = data_loader.get_all_services(df)
    service_id = services[0]
    logger.info(f"Analyzing service: {service_id}")

    kpis = data_loader.get_all_kpis(df, service_id)
    kpi_name = kpis[0]
    logger.info(f"Analyzing KPI: {kpi_name}")

    # Get time series
    ts_data = data_loader.get_service_timeseries(df, service_id, kpi_name)
    logger.info(f"Time series length: {len(ts_data)}")

    # Run detection methods
    results = {}

    # Statistical ensemble
    logger.info("Running statistical ensemble...")
    results['Statistical Ensemble'] = stat_detector.detect(ts_data, method='ensemble')

    # Isolation Forest
    logger.info("Running Isolation Forest...")
    results['Isolation Forest'] = ml_detector.detect(ts_data, method='isolation_forest')

    # Z-score
    logger.info("Running Z-score...")
    results['Z-score'] = stat_detector.detect(ts_data, method='zscore')

    # Generate reports
    logger.info("Generating reports...")
    for method, result_df in results.items():
        # Summary report
        summary = reporter.generate_summary_report(result_df, method)
        logger.info(f"{method} - Anomalies: {summary['anomaly_count']} "
                   f"({summary['anomaly_rate']:.2%})")

        # Export anomalies
        csv_path = f"results/{method.replace(' ', '_')}_anomalies.csv"
        reporter.export_anomalies_csv(result_df, csv_path)

        # Visualize
        plot_path = f"results/{method.replace(' ', '_')}_plot.png"
        title = f"{cloudbed_name} - {service_id} - {kpi_name} ({method})"
        visualizer.plot_timeseries_with_anomalies(result_df, title, plot_path)

    # Compare methods
    logger.info("Creating comparison visualizations...")
    comparison_dict = {method: result_df['is_anomaly'].values
                      for method, result_df in results.items()}

    # Comparison plot
    visualizer.plot_multiple_methods_comparison(
        ts_data, comparison_dict,
        f"{cloudbed_name} - {service_id} - {kpi_name}",
        "results/methods_comparison.png"
    )

    # Heatmap
    visualizer.plot_anomaly_heatmap(
        comparison_dict,
        ts_data['timestamp'],
        "results/anomaly_heatmap.png"
    )

    # Comparison report
    comparison_df = reporter.generate_comparison_report(
        comparison_dict, ts_data,
        "results/comparison_report.csv"
    )
    logger.info("\nComparison Report:")
    logger.info(comparison_df.to_string())

    logger.info("\nAnalysis completed! Check the 'results' directory for outputs.")


if __name__ == '__main__':
    main()
