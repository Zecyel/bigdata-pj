import os
import importlib.util
import pandas as pd

def load_statistical_detector():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'detectors', 'statistical_detector.py')
    spec = importlib.util.spec_from_file_location('statistical_detector', base)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StatisticalAnomalyDetector


def verify_one(file_path: str, method: str):
    pdf = pd.read_csv(file_path)
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'], unit='s')
    StatDet = load_statistical_detector()
    det = StatDet(contamination=0.1)
    res = det.detect(pdf[['timestamp', 'value']].copy(), method=method)
    res['cmdb_id'] = pdf['cmdb_id']
    res['kpi_name'] = pdf['kpi_name']
    anomalies = int(res['is_anomaly'].sum())
    total = int(len(res))
    print(f"{os.path.basename(file_path)} {method} anomalies={anomalies}/{total}")


def main():
    base = os.path.join('data', 'cloudbed_synth_2025-12-28', 'metric', 'service')
    verify_one(os.path.join(base, 'svcA_cpu.csv'), 'ensemble')
    verify_one(os.path.join(base, 'svcB_latency.csv'), 'zscore')


if __name__ == '__main__':
    main()
