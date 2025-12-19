"""
Machine learning-based anomaly detection for time series.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MLAnomalyDetector:
    """Machine learning methods for anomaly detection."""

    def __init__(self, contamination: float = 0.1):
        """
        Initialize ML detector.

        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = None

    def create_features(self, values: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Create features from time series for ML models.

        Args:
            values: Time series values
            window: Window size for feature extraction

        Returns:
            Feature matrix
        """
        n = len(values)
        features = []

        for i in range(window, n):
            window_data = values[i-window:i]

            feature_vector = [
                values[i],                          # Current value
                np.mean(window_data),               # Mean
                np.std(window_data),                # Std
                np.min(window_data),                # Min
                np.max(window_data),                # Max
                np.median(window_data),             # Median
                values[i] - values[i-1],            # First difference
                values[i] - np.mean(window_data),   # Deviation from mean
            ]

            # Add lag features
            for lag in [1, 2, 5]:
                if i >= lag:
                    feature_vector.append(values[i-lag])

            features.append(feature_vector)

        return np.array(features)

    def isolation_forest_detection(self, values: np.ndarray, window: int = 10,
                                   n_estimators: int = 100) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.

        Args:
            values: Time series values
            window: Window size for feature extraction
            n_estimators: Number of trees

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window + 1:
            logger.warning(f"Data too short for window size {window}")
            return np.zeros(len(values), dtype=bool)

        # Create features
        X = self.create_features(values, window)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model = IsolationForest(contamination=self.contamination,
                               n_estimators=n_estimators,
                               random_state=42)
        predictions = model.fit_predict(X_scaled)

        # Convert to boolean array (pad with False for initial window)
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[window:] = (predictions == -1)

        logger.info(f"Isolation Forest: {np.sum(anomalies)} anomalies found")
        return anomalies

    def local_outlier_factor_detection(self, values: np.ndarray, window: int = 10,
                                       n_neighbors: int = 20) -> np.ndarray:
        """
        Detect anomalies using Local Outlier Factor.

        Args:
            values: Time series values
            window: Window size for feature extraction
            n_neighbors: Number of neighbors

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window + 1:
            logger.warning(f"Data too short for window size {window}")
            return np.zeros(len(values), dtype=bool)

        # Create features
        X = self.create_features(values, window)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model = LOF(contamination=self.contamination, n_neighbors=n_neighbors)
        model.fit(X_scaled)
        predictions = model.labels_

        # Convert to boolean array
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[window:] = (predictions == 1)

        logger.info(f"LOF: {np.sum(anomalies)} anomalies found")
        return anomalies

    def knn_detection(self, values: np.ndarray, window: int = 10,
                     n_neighbors: int = 5) -> np.ndarray:
        """
        Detect anomalies using K-Nearest Neighbors.

        Args:
            values: Time series values
            window: Window size for feature extraction
            n_neighbors: Number of neighbors

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window + 1:
            logger.warning(f"Data too short for window size {window}")
            return np.zeros(len(values), dtype=bool)

        # Create features
        X = self.create_features(values, window)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model = KNN(contamination=self.contamination, n_neighbors=n_neighbors)
        model.fit(X_scaled)
        predictions = model.labels_

        # Convert to boolean array
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[window:] = (predictions == 1)

        logger.info(f"KNN: {np.sum(anomalies)} anomalies found")
        return anomalies

    def one_class_svm_detection(self, values: np.ndarray, window: int = 10,
                               nu: float = 0.1, kernel: str = 'rbf') -> np.ndarray:
        """
        Detect anomalies using One-Class SVM.

        Args:
            values: Time series values
            window: Window size for feature extraction
            nu: Upper bound on fraction of outliers
            kernel: Kernel type

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window + 1:
            logger.warning(f"Data too short for window size {window}")
            return np.zeros(len(values), dtype=bool)

        # Create features
        X = self.create_features(values, window)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model = OneClassSVM(nu=nu, kernel=kernel, gamma='auto')
        predictions = model.fit_predict(X_scaled)

        # Convert to boolean array
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[window:] = (predictions == -1)

        logger.info(f"One-Class SVM: {np.sum(anomalies)} anomalies found")
        return anomalies

    def hbos_detection(self, values: np.ndarray, window: int = 10,
                      n_bins: int = 10) -> np.ndarray:
        """
        Detect anomalies using Histogram-Based Outlier Score.

        Args:
            values: Time series values
            window: Window size for feature extraction
            n_bins: Number of histogram bins

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window + 1:
            logger.warning(f"Data too short for window size {window}")
            return np.zeros(len(values), dtype=bool)

        # Create features
        X = self.create_features(values, window)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        model = HBOS(contamination=self.contamination, n_bins=n_bins)
        model.fit(X_scaled)
        predictions = model.labels_

        # Convert to boolean array
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[window:] = (predictions == 1)

        logger.info(f"HBOS: {np.sum(anomalies)} anomalies found")
        return anomalies

    def autoencoder_detection(self, values: np.ndarray, window: int = 10,
                            threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect anomalies using simple reconstruction error approach.

        Args:
            values: Time series values
            window: Window size
            threshold_percentile: Percentile for anomaly threshold

        Returns:
            Boolean array indicating anomalies
        """
        if len(values) < window + 1:
            logger.warning(f"Data too short for window size {window}")
            return np.zeros(len(values), dtype=bool)

        # Create sliding windows
        X = self.create_features(values, window)

        # Use PCA as a simple form of dimensionality reduction/reconstruction
        n_components = min(3, X.shape[1])
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Calculate reconstruction error
        reconstruction_errors = np.mean((X - X_reconstructed) ** 2, axis=1)

        # Threshold based on percentile
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomaly_indices = reconstruction_errors > threshold

        # Convert to boolean array
        anomalies = np.zeros(len(values), dtype=bool)
        anomalies[window:] = anomaly_indices

        logger.info(f"Autoencoder (PCA): {np.sum(anomalies)} anomalies found")
        return anomalies

    def ensemble_ml_detection(self, values: np.ndarray, window: int = 10,
                            methods: Optional[list] = None,
                            voting_threshold: float = 0.5) -> np.ndarray:
        """
        Ensemble of multiple ML methods.

        Args:
            values: Time series values
            window: Window size for feature extraction
            methods: List of methods to use
            voting_threshold: Proportion of methods that must agree

        Returns:
            Boolean array indicating anomalies
        """
        if methods is None:
            methods = ['isolation_forest', 'lof', 'knn']

        results = []

        if 'isolation_forest' in methods:
            results.append(self.isolation_forest_detection(values, window))
        if 'lof' in methods:
            results.append(self.local_outlier_factor_detection(values, window))
        if 'knn' in methods:
            results.append(self.knn_detection(values, window))
        if 'ocsvm' in methods:
            results.append(self.one_class_svm_detection(values, window))
        if 'hbos' in methods:
            results.append(self.hbos_detection(values, window))
        if 'autoencoder' in methods:
            results.append(self.autoencoder_detection(values, window))

        if not results:
            logger.error("No ML methods selected")
            return np.zeros(len(values), dtype=bool)

        # Voting
        results_array = np.array(results)
        votes = np.sum(results_array, axis=0)
        anomalies = votes >= (len(methods) * voting_threshold)

        logger.info(f"ML Ensemble: {np.sum(anomalies)} anomalies found")
        return anomalies

    def detect(self, df: pd.DataFrame, method: str = 'isolation_forest',
               window: int = 10, **kwargs) -> pd.DataFrame:
        """
        Main detection method that adds anomaly labels to DataFrame.

        Args:
            df: DataFrame with 'timestamp' and 'value' columns
            method: Detection method to use
            window: Window size for feature extraction
            **kwargs: Additional parameters for specific methods

        Returns:
            DataFrame with added 'is_anomaly' and 'anomaly_score' columns
        """
        values = df['value'].values

        if method == 'isolation_forest':
            anomalies = self.isolation_forest_detection(values, window, **kwargs)
        elif method == 'lof':
            anomalies = self.local_outlier_factor_detection(values, window, **kwargs)
        elif method == 'knn':
            anomalies = self.knn_detection(values, window, **kwargs)
        elif method == 'ocsvm':
            anomalies = self.one_class_svm_detection(values, window, **kwargs)
        elif method == 'hbos':
            anomalies = self.hbos_detection(values, window, **kwargs)
        elif method == 'autoencoder':
            anomalies = self.autoencoder_detection(values, window, **kwargs)
        elif method == 'ensemble':
            anomalies = self.ensemble_ml_detection(values, window, **kwargs)
        else:
            logger.error(f"Unknown method: {method}")
            anomalies = np.zeros(len(values), dtype=bool)

        result = df.copy()
        result['is_anomaly'] = anomalies
        result['anomaly_score'] = anomalies.astype(float)

        return result
