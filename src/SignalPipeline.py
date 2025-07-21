import pandas as pd
import numpy as np

from BaselineCorrector import BaselineCorrector
from MeanPeakAnalyzer import MeanPeakAnalyzer
from AnomalyDetector import AnomalyDetector


class SignalPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.time = self.data.iloc[:, 0].values
        self.interval = np.mean(np.diff(self.time))
        self.fs = 1.0 / self.interval

        self.anomaly_detector = AnomalyDetector(contamination=0.005)
        self.baseline_corrector = BaselineCorrector(fs=self.fs)

        self.df_cleaned = self.remove_anomalies()
        self.df_corrected = self.apply_baseline_correction(self.df_cleaned)

    def load_data(self):
        df = pd.read_csv(
            self.file_path,
            sep='\t',
            decimal=',',
            encoding='windows-1251'
        )
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def remove_anomalies(self):
        cleaned_df = self.data.copy()
        time_values = cleaned_df.iloc[:, 0]

        for sensor in cleaned_df.columns[1:]:
            sensor_data = cleaned_df[sensor]
            cleaned_series, _ = self.anomaly_detector.detect_anomalies(
                time_values,
                sensor_data,
                sensor,
                plot=False
            )
            cleaned_df[sensor] = cleaned_series

        print("Аномалии успешно удалены для всех датчиков")
        return cleaned_df

    def apply_baseline_correction(self, df):
        corrected_df = df.copy()
        for sensor in corrected_df.columns[1:]:
            data = corrected_df[sensor].values
            baseline = self.baseline_corrector.baseline_median(data)
            corrected_df[sensor] = data - baseline
        return corrected_df

    def run(self):
        analyzer = MeanPeakAnalyzer(self.df_corrected, self.fs)
        analyzer.analyze()

if __name__ == "__main__":
    file_path = 'resources/клин 600 701.txt'
    pipeline = SignalPipeline(file_path)
    pipeline.run()