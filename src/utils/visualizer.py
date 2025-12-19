"""
Visualization and reporting module for anomaly detection results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import logging
import os

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AnomalyVisualizer:
    """Visualize time series and detected anomalies."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_timeseries_with_anomalies(self, df: pd.DataFrame, title: str,
                                      save_path: Optional[str] = None):
        """
        Plot time series with anomalies highlighted.

        Args:
            df: DataFrame with 'timestamp', 'value', and 'is_anomaly' columns
            title: Plot title
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot normal points
        normal = df[~df['is_anomaly']]
        ax.plot(normal['timestamp'], normal['value'], 'b-', label='Normal', alpha=0.7)

        # Plot anomalies
        anomalies = df[df['is_anomaly']]
        if not anomalies.empty:
            ax.scatter(anomalies['timestamp'], anomalies['value'],
                      color='red', s=100, label='Anomaly', zorder=5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'timeseries_anomalies.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_multiple_methods_comparison(self, df: pd.DataFrame, results: dict,
                                        title: str, save_path: Optional[str] = None):
        """
        Compare results from multiple detection methods.

        Args:
            df: Original DataFrame with 'timestamp' and 'value'
            results: Dict mapping method names to anomaly boolean arrays
            title: Plot title
            save_path: Path to save figure
        """
        n_methods = len(results)
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(15, 4 * (n_methods + 1)))

        if n_methods == 0:
            axes = [axes]

        # Plot original time series
        axes[0].plot(df['timestamp'], df['value'], 'b-', alpha=0.7)
        axes[0].set_title(f'{title} - Original Time Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Plot each method's results
        for idx, (method_name, anomalies) in enumerate(results.items(), 1):
            axes[idx].plot(df['timestamp'], df['value'], 'b-', alpha=0.3)

            if np.any(anomalies):
                anomaly_df = df[anomalies]
                axes[idx].scatter(anomaly_df['timestamp'], anomaly_df['value'],
                                color='red', s=50, label=f'Anomaly ({np.sum(anomalies)})')

            axes[idx].set_title(f'{method_name}')
            axes[idx].set_ylabel('Value')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time')

        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'methods_comparison.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_anomaly_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot distribution of normal vs anomalous values.

        Args:
            df: DataFrame with 'value' and 'is_anomaly' columns
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        normal_values = df[~df['is_anomaly']]['value']
        anomaly_values = df[df['is_anomaly']]['value']

        axes[0].hist(normal_values, bins=50, alpha=0.7, label='Normal', color='blue')
        if len(anomaly_values) > 0:
            axes[0].hist(anomaly_values, bins=20, alpha=0.7, label='Anomaly', color='red')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Value Distribution')
        axes[0].legend()

        # Box plot
        data_to_plot = [normal_values, anomaly_values] if len(anomaly_values) > 0 else [normal_values]
        labels = ['Normal', 'Anomaly'] if len(anomaly_values) > 0 else ['Normal']
        axes[1].boxplot(data_to_plot, labels=labels)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Value Distribution Box Plot')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'anomaly_distribution.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_anomaly_heatmap(self, results_dict: dict, timestamps: pd.Series,
                           save_path: Optional[str] = None):
        """
        Create heatmap showing when different methods detect anomalies.

        Args:
            results_dict: Dict mapping method names to anomaly boolean arrays
            timestamps: Time series timestamps
            save_path: Path to save figure
        """
        # Create matrix
        methods = list(results_dict.keys())
        matrix = np.array([results_dict[m].astype(int) for m in methods])

        fig, ax = plt.subplots(figsize=(15, len(methods) * 0.5 + 2))

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')

        # Set ticks
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods)

        # Set x-axis to show time intervals
        n_ticks = min(10, len(timestamps))
        tick_positions = np.linspace(0, len(timestamps) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(timestamps.iloc[i])[:16] for i in tick_positions],
                          rotation=45, ha='right')

        ax.set_xlabel('Time')
        ax.set_title('Anomaly Detection Heatmap (Red = Anomaly, Green = Normal)')

        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'anomaly_heatmap.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()


class AnomalyReporter:
    """Generate reports for anomaly detection results."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_summary_report(self, df: pd.DataFrame, method: str,
                               save_path: Optional[str] = None) -> dict:
        """
        Generate summary statistics for anomaly detection results.

        Args:
            df: DataFrame with 'is_anomaly' column
            method: Detection method name
            save_path: Path to save report

        Returns:
            Dictionary with summary statistics
        """
        total_points = len(df)
        anomaly_count = df['is_anomaly'].sum()
        anomaly_rate = anomaly_count / total_points if total_points > 0 else 0

        normal_values = df[~df['is_anomaly']]['value']
        anomaly_values = df[df['is_anomaly']]['value']

        summary = {
            'method': method,
            'total_points': total_points,
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'normal_mean': float(normal_values.mean()) if len(normal_values) > 0 else 0,
            'normal_std': float(normal_values.std()) if len(normal_values) > 0 else 0,
            'anomaly_mean': float(anomaly_values.mean()) if len(anomaly_values) > 0 else 0,
            'anomaly_std': float(anomaly_values.std()) if len(anomaly_values) > 0 else 0,
        }

        if save_path:
            with open(save_path, 'w') as f:
                f.write(f"Anomaly Detection Summary Report\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Method: {summary['method']}\n")
                f.write(f"Total data points: {summary['total_points']}\n")
                f.write(f"Anomalies detected: {summary['anomaly_count']}\n")
                f.write(f"Anomaly rate: {summary['anomaly_rate']:.2%}\n\n")
                f.write(f"Normal values:\n")
                f.write(f"  Mean: {summary['normal_mean']:.4f}\n")
                f.write(f"  Std: {summary['normal_std']:.4f}\n\n")
                f.write(f"Anomalous values:\n")
                f.write(f"  Mean: {summary['anomaly_mean']:.4f}\n")
                f.write(f"  Std: {summary['anomaly_std']:.4f}\n")

            logger.info(f"Saved summary report to {save_path}")

        return summary

    def export_anomalies_csv(self, df: pd.DataFrame, save_path: str):
        """
        Export detected anomalies to CSV.

        Args:
            df: DataFrame with anomaly detection results
            save_path: Path to save CSV
        """
        anomalies = df[df['is_anomaly']].copy()
        anomalies.to_csv(save_path, index=False)
        logger.info(f"Exported {len(anomalies)} anomalies to {save_path}")

    def generate_comparison_report(self, results_dict: dict, df: pd.DataFrame,
                                  save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate comparison report for multiple methods.

        Args:
            results_dict: Dict mapping method names to anomaly boolean arrays
            df: Original DataFrame
            save_path: Path to save report

        Returns:
            DataFrame with comparison statistics
        """
        comparison_data = []

        for method, anomalies in results_dict.items():
            temp_df = df.copy()
            temp_df['is_anomaly'] = anomalies
            summary = self.generate_summary_report(temp_df, method)
            comparison_data.append(summary)

        comparison_df = pd.DataFrame(comparison_data)

        if save_path:
            comparison_df.to_csv(save_path, index=False)
            logger.info(f"Saved comparison report to {save_path}")

        return comparison_df
