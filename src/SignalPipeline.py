import pandas as pd
import numpy as np

from src.AnomalyDetector import AnomalyDetector
from src.BaselineCorrector import BaselineCorrector
from src.MeanPeakAnalyzer import MeanPeakAnalyzer


class SignalPipeline:
    def __init__(self, file_path, contamination=0.005):
        self.file_path = file_path
        self.contamination = contamination
        self.data = self.load_data()
        self.time = self.data.iloc[:, 0].values
        self.interval = np.mean(np.diff(self.time))
        self.fs = 1.0 / self.interval
        self.baseline_corrector = BaselineCorrector(fs=self.fs)
        self.anomaly_detector = AnomalyDetector(contamination=self.contamination)
        self.df_processed = self.process_data()

    def load_data(self):
        df = pd.read_csv(self.file_path, sep='\t', decimal=',', encoding='windows-1251')
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)
        return df

    def apply_baseline_correction(self, df):
        corrected_df = df.copy()
        for sensor in corrected_df.columns[1:]:
            data = corrected_df[sensor].values
            baseline = self.baseline_corrector.baseline_median(data)
            corrected_df[sensor] = data - baseline
        return corrected_df

    def detect_and_remove_anomalies(self, df):
        cleaned_df = df.copy()
        for sensor in cleaned_df.columns[1:]:
            sensor_data = cleaned_df[sensor]
            cleaned_data, anomalies = self.anomaly_detector.detect_anomalies(
                self.time, sensor_data, sensor, plot=False)
            cleaned_df[sensor] = cleaned_data

            if not anomalies.empty:
                print(f"\n{sensor}:")
                print(f"Удалено аномалий: {len(anomalies)}")
                print(f"Процент удаленных точек: {len(anomalies) / len(df) * 100:.2f}%")
        return cleaned_df

    def process_data(self):
        # 1. Коррекция базовой линии
        df_corrected = self.apply_baseline_correction(self.data)

        # 2. Обнаружение и удаление аномалий
        df_cleaned = self.detect_and_remove_anomalies(df_corrected)

        return df_cleaned

    def run(self):
        analyzer = MeanPeakAnalyzer(self.df_processed)
        return analyzer.analyze()


if __name__ == "__main__":
    file_path = 'resources/клин 600 701.txt'
    pipeline = SignalPipeline(file_path, contamination=0.005)
    results = pipeline.run()