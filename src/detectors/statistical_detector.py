"""
Statistical methods for time series anomaly detection.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticalAnomalyDetector:
    """Statistical methods for detecting anomalies in time series data."""

    def __init__(self, contamination: float = 0.1):
        """
        Initialize detector.

        Args:
            contamination: Expected proportion of anomalies in the data
        """
        self.contamination = contamination

    def zscore_detection(self, values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies using Z-score method.

        Args:
            values: Time series values
            threshold: Number of standard deviations for anomaly threshold

        Returns:
            Boolean array indicating anomalies (True = anomaly)
        """
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return np.zeros(len(values), dtype=bool)

        z_scores = np.abs((values - mean) / std)
        anomalies = z_scores > threshold

        logger.info(f"Z-score detection: {np.sum(anomalies)} anomalies found")
        return anomalies

    def modified_zscore_detection(self, values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        """
        Detect anomalies using Modified Z-score (robust to outliers).

        Args:
            values: Time series values
            threshold: Threshold for modified Z-score

        Returns:
            Boolean array indicating anomalies
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            return np.zeros(len(values), dtype=bool)

        modified_z_scores = 0.6745 * (values - median) / mad
        anomalies = np.abs(modified_z_scores) > threshold

        logger.info(f"Modified Z-score detection: {np.sum(anomalies)} anomalies found")
        return anomalies

    def iqr_detection(self, values: np.ndarray, k: float = 1.5) -> np.ndarray:
        """
        Detect anomalies using Interquartile Range (IQR) method.

        Args:
            values: Time series values
            k: IQR multiplier (typically 1.5 for outliers, 3 for extreme outliers)

        Returns:
            Boolean array indicating anomalies
        """
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        anomalies = (values < lower_bound) | (values > upper_bound)

        logger.info(f"IQR detection: {np.sum(anomalies)} anomalies found")
        return anomalies

    def moving_average_detection(self, values: np.ndarray, window: int = 10,
                                 threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies based on deviation from moving average.

        Args:
            values: Time series values
            window: Window size for moving average
            threshold: Number of standard deviations for threshold

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window:
            logger.warning(f"Data length {len(values)} < window {window}")
            return np.zeros(len(values), dtype=bool)

        # Calculate moving average and std
        ma = pd.Series(values).rolling(window=window, center=True).mean().values
        std = pd.Series(values).rolling(window=window, center=True).std().values

        # Handle NaN values at boundaries
        ma = np.nan_to_num(ma, nan=np.nanmean(ma))
        std = np.nan_to_num(std, nan=np.nanstd(std))
        std[std == 0] = 1  # Avoid division by zero

        # Calculate deviation
        deviation = np.abs(values - ma) / std
        anomalies = deviation > threshold

        logger.info(f"Moving average detection: {np.sum(anomalies)} anomalies found")
        return anomalies

    def exponential_moving_average_detection(self, values: np.ndarray, span: int = 10,
                                            threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies based on exponential moving average.

        Args:
            values: Time series values
            span: Span for EMA calculation
            threshold: Number of standard deviations for threshold

        Returns:
            Boolean array indicating anomalies
        """
        ema = pd.Series(values).ewm(span=span).mean().values
        residuals = values - ema
        std = np.std(residuals)

        if std == 0:
            return np.zeros(len(values), dtype=bool)

        anomalies = np.abs(residuals) > threshold * std

        logger.info(f"EMA detection: {np.sum(anomalies)} anomalies found")
        return anomalies

    def seasonal_decomposition_detection(self, values: np.ndarray, period: int = 24,
                                        threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies in residuals after seasonal decomposition.

        Args:
            values: Time series values
            period: Seasonal period
            threshold: Number of standard deviations for threshold

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < 2 * period:
            logger.warning(f"Data length {len(values)} too short for period {period}")
            return self.zscore_detection(values, threshold)

        from statsmodels.tsa.seasonal import seasonal_decompose

        try:
            # Create series with frequency
            ts = pd.Series(values)
            decomposition = seasonal_decompose(ts, model='additive', period=period,
                                              extrapolate_trend='freq')
            residuals = decomposition.resid.values

            # Remove NaN values
            residuals = np.nan_to_num(residuals, nan=0)

            # Detect anomalies in residuals
            mean_resid = np.mean(residuals)
            std_resid = np.std(residuals)

            if std_resid == 0:
                return np.zeros(len(values), dtype=bool)

            anomalies = np.abs(residuals - mean_resid) > threshold * std_resid

            logger.info(f"Seasonal decomposition detection: {np.sum(anomalies)} anomalies found")
            return anomalies

        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
            return self.zscore_detection(values, threshold)

    def grubbs_test(self, values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Detect anomalies using Grubbs test (for single outlier detection).

        Args:
            values: Time series values
            alpha: Significance level

        Returns:
            Boolean array indicating anomalies
        """
        anomalies = np.zeros(len(values), dtype=bool)
        data = values.copy()

        while len(data) > 2:
            mean = np.mean(data)
            std = np.std(data)

            if std == 0:
                break

            # Calculate Grubbs statistic
            abs_val_minus_mean = np.abs(data - mean)
            max_idx = np.argmax(abs_val_minus_mean)
            G = abs_val_minus_mean[max_idx] / std

            # Critical value
            n = len(data)
            t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_critical = ((n - 1) * np.sqrt(np.square(t_dist))) / \
                        np.sqrt(n * (n - 2 + np.square(t_dist)))

            if G > G_critical:
                # Find original index
                orig_idx = np.where(values == data[max_idx])[0][0]
                anomalies[orig_idx] = True
                data = np.delete(data, max_idx)
            else:
                break

        logger.info(f"Grubbs test: {np.sum(anomalies)} anomalies found")
        return anomalies

    def ensemble_detection(self, values: np.ndarray, methods: Optional[list] = None,
                          voting_threshold: float = 0.5) -> np.ndarray:
        """
        Ensemble method combining multiple detection techniques.

        Args:
            values: Time series values
            methods: List of methods to use (default: all statistical methods)
            voting_threshold: Proportion of methods that must agree (0.5 = majority)

        Returns:
            Boolean array indicating anomalies
        """
        if methods is None:
            methods = ['zscore', 'modified_zscore', 'iqr', 'moving_average', 'ema']

        results = []

        if 'zscore' in methods:
            results.append(self.zscore_detection(values))
        if 'modified_zscore' in methods:
            results.append(self.modified_zscore_detection(values))
        if 'iqr' in methods:
            results.append(self.iqr_detection(values))
        if 'moving_average' in methods:
            results.append(self.moving_average_detection(values))
        if 'ema' in methods:
            results.append(self.exponential_moving_average_detection(values))

        if not results:
            logger.error("No detection methods selected")
            return np.zeros(len(values), dtype=bool)

        # Voting
        results_array = np.array(results)
        votes = np.sum(results_array, axis=0)
        anomalies = votes >= (len(methods) * voting_threshold)

        logger.info(f"Ensemble detection: {np.sum(anomalies)} anomalies found")
        return anomalies

    def detect(self, df: pd.DataFrame, method: str = 'ensemble', **kwargs) -> pd.DataFrame:
        """
        Main detection method that adds anomaly labels to DataFrame.

        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            method: Detection method to use
            **kwargs: Additional parameters for specific methods

        Returns:
            DataFrame with added 'is_anomaly' and 'anomaly_score' columns
        """
        values = df['value'].values

        if method == 'zscore':
            anomalies = self.zscore_detection(values, **kwargs)
        elif method == 'modified_zscore':
            anomalies = self.modified_zscore_detection(values, **kwargs)
        elif method == 'iqr':
            anomalies = self.iqr_detection(values, **kwargs)
        elif method == 'moving_average':
            anomalies = self.moving_average_detection(values, **kwargs)
        elif method == 'ema':
            anomalies = self.exponential_moving_average_detection(values, **kwargs)
        elif method == 'seasonal':
            anomalies = self.seasonal_decomposition_detection(values, **kwargs)
        elif method == 'grubbs':
            anomalies = self.grubbs_test(values, **kwargs)
        elif method == 'ensemble':
            anomalies = self.ensemble_detection(values, **kwargs)
        else:
            logger.error(f"Unknown method: {method}")
            anomalies = np.zeros(len(values), dtype=bool)

        result = df.copy()
        result['is_anomaly'] = anomalies
        result['anomaly_score'] = anomalies.astype(float)

        return result
