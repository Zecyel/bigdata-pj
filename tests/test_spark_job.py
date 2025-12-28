import os
import pandas as pd
from detectors.statistical_detector import StatisticalAnomalyDetector
from src.spark_job import detect_group_impl

def test_detect_group_impl_cpu():
    path = os.path.join('data', 'cloudbed_synth_2025-12-28', 'metric', 'service', 'svcA_cpu.csv')
    pdf = pd.read_csv(path)
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'], unit='s')
    det = StatisticalAnomalyDetector(contamination=0.1)
    res = detect_group_impl(pdf, 'ensemble', det)
    assert 'is_anomaly' in res.columns
    assert res['is_anomaly'].sum() > 0

def test_detect_group_impl_latency():
    path = os.path.join('data', 'cloudbed_synth_2025-12-28', 'metric', 'service', 'svcB_latency.csv')
    pdf = pd.read_csv(path)
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'], unit='s')
    det = StatisticalAnomalyDetector(contamination=0.1)
    res = detect_group_impl(pdf, 'zscore', det)
    assert 'anomaly_score' in res.columns
    assert res['is_anomaly'].sum() > 0
