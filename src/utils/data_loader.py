"""
Data loader module for time series data from CSV files.
Supports loading multiple metrics from cloudbed directories.
"""

import os
import pandas as pd
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataLoader:
    """Load and preprocess time series data from CSV files."""

    def __init__(self, data_dir: str):
        """
        Initialize data loader.

        Args:
            data_dir: Root directory containing cloudbed data
        """
        self.data_dir = data_dir
        self.metric_categories = ['container', 'istio', 'jvm', 'node', 'service']

    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single CSV file and parse timestamps.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with parsed timestamps
        """
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def load_metric_category(self, cloudbed_dir: str, category: str) -> pd.DataFrame:
        """
        Load all CSV files from a metric category.

        Args:
            cloudbed_dir: Cloudbed directory path
            category: Metric category (jvm, container, etc.)

        Returns:
            Combined DataFrame of all metrics in category
        """
        category_path = os.path.join(cloudbed_dir, 'metric', category)
        if not os.path.exists(category_path):
            logger.warning(f"Category path not found: {category_path}")
            return pd.DataFrame()

        all_data = []
        for filename in os.listdir(category_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(category_path, filename)
                df = self.load_csv_file(file_path)
                if not df.empty:
                    all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(all_data)} files from {category}, total rows: {len(combined)}")
            return combined
        return pd.DataFrame()

    def load_cloudbed(self, name: str, date_pattern: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from all cloudbed directories.

        Args:
            date_pattern: Optional pattern to filter directories (e.g., '2022-03-20')

        Returns:
            Dictionary mapping cloudbed names to DataFrames
        """
        cloudbed_data = {}

        for dirname in os.listdir(self.data_dir):
            if dirname == name:
                if date_pattern and date_pattern not in dirname:
                    continue

                cloudbed_path = os.path.join(self.data_dir, dirname)
                if os.path.isdir(cloudbed_path):
                    all_categories = []
                    for category in self.metric_categories:
                        df = self.load_metric_category(cloudbed_path, category)
                        if not df.empty:
                            all_categories.append(df)

                    if all_categories:
                        combined = pd.concat(all_categories, ignore_index=True)
                        cloudbed_data[dirname] = combined
                        logger.info(f"Loaded {dirname}: {len(combined)} records")

        return cloudbed_data

    def load_all_cloudbeds(self, date_pattern: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from all cloudbed directories.

        Args:
            date_pattern: Optional pattern to filter directories (e.g., '2022-03-20')

        Returns:
            Dictionary mapping cloudbed names to DataFrames
        """
        cloudbed_data = {}

        for dirname in os.listdir(self.data_dir):
            if 'cloudbed' in dirname:
                if date_pattern and date_pattern not in dirname:
                    continue

                cloudbed_path = os.path.join(self.data_dir, dirname)
                if os.path.isdir(cloudbed_path):
                    all_categories = []
                    for category in self.metric_categories:
                        df = self.load_metric_category(cloudbed_path, category)
                        if not df.empty:
                            all_categories.append(df)

                    if all_categories:
                        combined = pd.concat(all_categories, ignore_index=True)
                        cloudbed_data[dirname] = combined
                        logger.info(f"Loaded {dirname}: {len(combined)} records")

        return cloudbed_data

    def get_service_timeseries(self, df: pd.DataFrame, cmdb_id: str, kpi_name: str) -> pd.DataFrame:
        """
        Extract time series for a specific service and KPI.

        Args:
            df: DataFrame containing all data
            cmdb_id: Service identifier
            kpi_name: KPI metric name

        Returns:
            Filtered and sorted DataFrame
        """
        filtered = df[(df['cmdb_id'] == cmdb_id) & (df['kpi_name'] == kpi_name)].copy()
        filtered = filtered.sort_values('timestamp').reset_index(drop=True)
        return filtered[['timestamp', 'value']]

    def get_all_services(self, df: pd.DataFrame) -> List[str]:
        """Get list of unique service IDs."""
        return df['cmdb_id'].unique().tolist()

    def get_all_kpis(self, df: pd.DataFrame, cmdb_id: Optional[str] = None) -> List[str]:
        """
        Get list of unique KPI names.

        Args:
            df: DataFrame containing data
            cmdb_id: Optional service ID to filter KPIs for specific service

        Returns:
            List of KPI names
        """
        if cmdb_id:
            return df[df['cmdb_id'] == cmdb_id]['kpi_name'].unique().tolist()
        return df['kpi_name'].unique().tolist()

    def resample_timeseries(self, df: pd.DataFrame, freq: str = '5T') -> pd.DataFrame:
        """
        Resample time series to a specific frequency.

        Args:
            df: DataFrame with timestamp and value columns
            freq: Pandas frequency string (e.g., '5T' for 5 minutes)

        Returns:
            Resampled DataFrame
        """
        df = df.set_index('timestamp')
        resampled = df.resample(freq).mean()
        return resampled.reset_index()
