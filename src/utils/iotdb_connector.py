"""
IoTDB connector for storing and querying time series data.
"""

from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType, TSEncoding, Compressor
from iotdb.utils.Tablet import Tablet
import pandas as pd
import numpy as np
from typing import List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class IoTDBConnector:
    """Connector for Apache IoTDB time series database."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6667,
                 username: str = "root", password: str = "root"):
        """
        Initialize IoTDB connection.

        Args:
            host: IoTDB server host
            port: IoTDB server port
            username: Database username
            password: Database password
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.session = None

    def connect(self):
        """Establish connection to IoTDB."""
        try:
            self.session = Session(self.host, self.port, self.username, self.password)
            self.session.open(False)
            logger.info(f"Connected to IoTDB at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to IoTDB: {e}")
            raise

    def disconnect(self):
        """Close IoTDB connection."""
        if self.session:
            self.session.close()
            logger.info("Disconnected from IoTDB")

    def create_timeseries(self, path: str, data_type: TSDataType = TSDataType.DOUBLE,
                         encoding: TSEncoding = TSEncoding.GORILLA,
                         compressor: Compressor = Compressor.SNAPPY):
        """
        Create a time series in IoTDB.

        Args:
            path: Time series path (e.g., 'root.cloudbed.service.metric')
            data_type: Data type for the time series
            encoding: Encoding method
            compressor: Compression algorithm
        """
        try:
            self.session.create_time_series(path, data_type, encoding, compressor)
            logger.info(f"Created time series: {path}")
        except Exception as e:
            if "already exists" not in str(e):
                logger.error(f"Failed to create time series {path}: {e}")

    def insert_dataframe(self, df: pd.DataFrame, storage_group: str,
                        device: str, measurement: str):
        """
        Insert DataFrame into IoTDB.

        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            storage_group: Storage group name (e.g., 'root.cloudbed')
            device: Device name (e.g., 'service1')
            measurement: Measurement name (e.g., 'cpu_usage')
        """
        try:
            # Create storage group if not exists
            try:
                self.session.set_storage_group(storage_group)
            except Exception:
                pass  # Storage group may already exist

            # Sanitize device/measurement segments to satisfy IoTDB path rules
            safe_device = self.sanitize_path_segment(device)
            safe_measurement = self.sanitize_path_segment(measurement)

            # Create time series path
            path = f"{storage_group}.{safe_device}.{safe_measurement}"
            logger.debug(f"Using IoTDB path {path}")
            self.create_timeseries(path)

            # Prepare data
            timestamps = df['timestamp'].astype(np.int64) // 10**6  # Convert to milliseconds
            values = df['value'].values

            # Insert data in batches via Tablet to satisfy IoTDB API expectations
            batch_size = 1000
            device_id = f"{storage_group}.{safe_device}"
            measurements_schema = [safe_measurement]
            data_types_schema = [TSDataType.DOUBLE]

            for i in range(0, len(timestamps), batch_size):
                batch_timestamps = timestamps[i:i+batch_size].tolist()
                batch_values = values[i:i+batch_size].tolist()

                tablet = Tablet(device_id, measurements_schema, data_types_schema)
                for row_idx, (ts, val) in enumerate(zip(batch_timestamps, batch_values)):
                    tablet.timestamps[row_idx] = int(ts)
                    tablet.values[safe_measurement][row_idx] = float(val)
                    tablet.row_size += 1

                self.session.insert_tablet(tablet)

            logger.info(f"Inserted {len(df)} records to {path}")

        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            raise

    def insert_bulk_data(self, data_dict: dict, storage_group: str = "root.cloudbed"):
        """
        Insert multiple time series from a dictionary.

        Args:
            data_dict: Dict with keys as (device, measurement) tuples and values as DataFrames
            storage_group: Storage group name
        """
        for (device, measurement), df in data_dict.items():
            self.insert_dataframe(df, storage_group, device, measurement)

    def query_timeseries(self, path: str, start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Query time series data from IoTDB.

        Args:
            path: Time series path
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with timestamp and value columns
        """
        try:
            query = f"SELECT {path} FROM {path}"
            if start_time and end_time:
                query += f" WHERE time >= {start_time} AND time <= {end_time}"

            session_data_set = self.session.execute_query_statement(query)
            timestamps = []
            values = []

            while session_data_set.has_next():
                record = session_data_set.next()
                timestamps.append(record.get_timestamp())
                values.append(record.get_fields()[0])

            session_data_set.close_operation_handle()

            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='ms'),
                'value': values
            })
            return df

        except Exception as e:
            logger.error(f"Failed to query {path}: {e}")
            return pd.DataFrame()

    def load_csv_to_iotdb(self, csv_data_loader, cloudbed_name: str):
        """
        Load CSV data into IoTDB using the data loader.

        Args:
            csv_data_loader: TimeSeriesDataLoader instance
            cloudbed_name: Name of cloudbed to load
        """
        cloudbed_data = csv_data_loader.load_cloudbed(cloudbed_name)

        if cloudbed_name not in cloudbed_data:
            logger.error(f"Cloudbed {cloudbed_name} not found")
            return

        df = cloudbed_data[cloudbed_name]
        services = csv_data_loader.get_all_services(df)

        for service in services:
            kpis = csv_data_loader.get_all_kpis(df, service)
            for kpi in kpis:
                ts_data = csv_data_loader.get_service_timeseries(df, service, kpi)
                if not ts_data.empty:
                    # Sanitize names for IoTDB
                    device = self.sanitize_path_segment(service)
                    measurement = self.sanitize_path_segment(kpi)
                    self.insert_dataframe(ts_data, "root.cloudbed", device, measurement)

        logger.info(f"Loaded {cloudbed_name} into IoTDB")

    @staticmethod
    def sanitize_path_segment(name: str) -> str:
        """Replace IoTDB-illegal path characters with underscores."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or "unnamed"
