"""
Main script for time series anomaly detection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
from typing import List, Optional
import pandas as pd

from utils.data_loader import TimeSeriesDataLoader
from utils.iotdb_connector import IoTDBConnector
from detectors.statistical_detector import StatisticalAnomalyDetector
from detectors.ml_detector import MLAnomalyDetector
from utils.visualizer import AnomalyVisualizer, AnomalyReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    """Main pipeline for time series anomaly detection."""

    def __init__(self, data_dir: str, output_dir: str = "results",
                 use_iotdb: bool = False, iotdb_host: str = "127.0.0.1",
                 iotdb_port: int = 6667):
        """
        Initialize pipeline.

        Args:
            data_dir: Directory containing cloudbed data
            output_dir: Directory for results
            use_iotdb: Whether to use IoTDB
            iotdb_host: IoTDB host
            iotdb_port: IoTDB port
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_iotdb = use_iotdb

        # Initialize components
        self.data_loader = TimeSeriesDataLoader(data_dir)
        self.stat_detector = StatisticalAnomalyDetector(contamination=0.1)
        self.ml_detector = MLAnomalyDetector(contamination=0.1)
        self.visualizer = AnomalyVisualizer(output_dir)
        self.reporter = AnomalyReporter(output_dir)

        self.iotdb_connector = None
        if use_iotdb:
            self.iotdb_connector = IoTDBConnector(iotdb_host, iotdb_port)

    def load_data(self, date_pattern: Optional[str] = None) -> dict:
        """
        Load data from cloudbed directories.

        Args:
            date_pattern: Optional date pattern filter

        Returns:
            Dictionary of cloudbed DataFrames
        """
        logger.info("Loading data from CSV files...")
        cloudbed_data = self.data_loader.load_all_cloudbeds(date_pattern)
        logger.info(f"Loaded {len(cloudbed_data)} cloudbeds")
        return cloudbed_data

    def ingest_to_iotdb(self, cloudbed_name: str):
        """
        Ingest data into IoTDB.

        Args:
            cloudbed_name: Name of cloudbed to ingest
        """
        if not self.iotdb_connector:
            logger.error("IoTDB connector not initialized")
            return

        logger.info(f"Ingesting {cloudbed_name} to IoTDB...")
        self.iotdb_connector.connect()
        self.iotdb_connector.load_csv_to_iotdb(self.data_loader, cloudbed_name)
        self.iotdb_connector.disconnect()
        logger.info("Data ingestion completed")

    def detect_anomalies(self, df: pd.DataFrame, service_id: str, kpi_name: str,
                        methods: List[str], window: int = 10) -> dict:
        """
        Detect anomalies using specified methods.

        Args:
            df: DataFrame with time series data
            service_id: Service identifier
            kpi_name: KPI metric name
            methods: List of detection methods
            window: Window size for ML methods

        Returns:
            Dictionary of detection results
        """
        logger.info(f"Detecting anomalies for {service_id} - {kpi_name}")

        # Get time series
        ts_data = self.data_loader.get_service_timeseries(df, service_id, kpi_name)

        if ts_data.empty:
            logger.warning(f"No data found for {service_id} - {kpi_name}")
            return {}

        results = {}

        # Statistical methods
        stat_methods = ['zscore', 'modified_zscore', 'iqr', 'moving_average',
                       'ema', 'seasonal', 'grubbs']
        for method in methods:
            if method in stat_methods:
                logger.info(f"Running {method}...")
                result_df = self.stat_detector.detect(ts_data, method=method)
                results[method] = result_df

        # ML methods
        ml_methods = ['isolation_forest', 'lof', 'knn', 'ocsvm', 'hbos', 'autoencoder']
        for method in methods:
            if method in ml_methods:
                logger.info(f"Running {method}...")
                result_df = self.ml_detector.detect(ts_data, method=method, window=window)
                results[method] = result_df

        # Ensemble methods
        if 'ensemble_stat' in methods:
            logger.info("Running statistical ensemble...")
            result_df = self.stat_detector.detect(ts_data, method='ensemble')
            results['ensemble_stat'] = result_df

        if 'ensemble_ml' in methods:
            logger.info("Running ML ensemble...")
            result_df = self.ml_detector.detect(ts_data, method='ensemble', window=window)
            results['ensemble_ml'] = result_df

        return results

    def run_analysis(self, cloudbed_name: str, service_id: Optional[str] = None,
                    kpi_name: Optional[str] = None, methods: Optional[List[str]] = None,
                    compare_methods: bool = False):
        """
        Run complete anomaly detection analysis.

        Args:
            cloudbed_name: Name of cloudbed to analyze
            service_id: Specific service ID (optional, will use first if not specified)
            kpi_name: Specific KPI name (optional, will use first if not specified)
            methods: List of detection methods to use
            compare_methods: Whether to compare multiple methods
        """
        if methods is None:
            methods = ['ensemble_stat', 'isolation_forest']

        # Load data
        cloudbed_data = self.load_data()

        if cloudbed_name not in cloudbed_data:
            logger.error(f"Cloudbed {cloudbed_name} not found")
            return

        df = cloudbed_data[cloudbed_name]

        # Select service and KPI
        if service_id is None:
            services = self.data_loader.get_all_services(df)
            service_id = services[0]
            logger.info(f"Using service: {service_id}")

        if kpi_name is None:
            kpis = self.data_loader.get_all_kpis(df, service_id)
            kpi_name = kpis[0]
            logger.info(f"Using KPI: {kpi_name}")

        # Detect anomalies
        results = self.detect_anomalies(df, service_id, kpi_name, methods)

        if not results:
            logger.error("No detection results")
            return

        # Generate reports and visualizations
        logger.info("Generating reports and visualizations...")

        for method, result_df in results.items():
            # Generate summary report
            report_path = os.path.join(self.output_dir,
                                      f"{cloudbed_name}_{service_id}_{kpi_name}_{method}_report.txt")
            self.reporter.generate_summary_report(result_df, method, report_path)

            # Export anomalies
            csv_path = os.path.join(self.output_dir,
                                   f"{cloudbed_name}_{service_id}_{kpi_name}_{method}_anomalies.csv")
            self.reporter.export_anomalies_csv(result_df, csv_path)

            # Visualize
            plot_path = os.path.join(self.output_dir,
                                    f"{cloudbed_name}_{service_id}_{kpi_name}_{method}.png")
            title = f"{cloudbed_name} - {service_id} - {kpi_name} ({method})"
            self.visualizer.plot_timeseries_with_anomalies(result_df, title, plot_path)

        # Compare methods if requested
        if compare_methods and len(results) > 1:
            logger.info("Comparing methods...")

            # Get original time series
            ts_data = self.data_loader.get_service_timeseries(df, service_id, kpi_name)

            # Create comparison dict
            comparison_dict = {method: result_df['is_anomaly'].values
                             for method, result_df in results.items()}

            # Comparison plot
            comp_plot_path = os.path.join(self.output_dir,
                                         f"{cloudbed_name}_{service_id}_{kpi_name}_comparison.png")
            title = f"{cloudbed_name} - {service_id} - {kpi_name} - Methods Comparison"
            self.visualizer.plot_multiple_methods_comparison(ts_data, comparison_dict,
                                                            title, comp_plot_path)

            # Heatmap
            heatmap_path = os.path.join(self.output_dir,
                                       f"{cloudbed_name}_{service_id}_{kpi_name}_heatmap.png")
            self.visualizer.plot_anomaly_heatmap(comparison_dict, ts_data['timestamp'],
                                                heatmap_path)

            # Comparison report
            comp_report_path = os.path.join(self.output_dir,
                                           f"{cloudbed_name}_{service_id}_{kpi_name}_comparison.csv")
            self.reporter.generate_comparison_report(comparison_dict, ts_data, comp_report_path)

        logger.info("Analysis completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')

    parser.add_argument('--data-dir', type=str, default='.',
                       help='Directory containing cloudbed data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--cloudbed', type=str, default='2022-03-20-cloudbed1',
                       help='Cloudbed name to analyze')
    parser.add_argument('--service', type=str, default=None,
                       help='Service ID to analyze')
    parser.add_argument('--kpi', type=str, default=None,
                       help='KPI name to analyze')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['ensemble_stat', 'isolation_forest'],
                       help='Detection methods to use')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple methods')
    parser.add_argument('--use-iotdb', action='store_true',
                       help='Use IoTDB for data storage')
    parser.add_argument('--iotdb-host', type=str, default='127.0.0.1',
                       help='IoTDB host')
    parser.add_argument('--iotdb-port', type=int, default=6667,
                       help='IoTDB port')
    parser.add_argument('--ingest-to-iotdb', action='store_true',
                       help='Ingest data to IoTDB')

    args = parser.parse_args()

    # Create pipeline
    pipeline = AnomalyDetectionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_iotdb=args.use_iotdb,
        iotdb_host=args.iotdb_host,
        iotdb_port=args.iotdb_port
    )

    # Ingest to IoTDB if requested
    if args.ingest_to_iotdb:
        pipeline.ingest_to_iotdb(args.cloudbed)
        return

    # Run analysis
    pipeline.run_analysis(
        cloudbed_name=args.cloudbed,
        service_id=args.service,
        kpi_name=args.kpi,
        methods=args.methods,
        compare_methods=args.compare
    )


if __name__ == '__main__':
    main()
