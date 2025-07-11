import pandas as pd
import numpy as np

from src.BaselineCorrector import BaselineCorrector
from src.MeanPeakAnalyzer import MeanPeakAnalyzer

class SignalPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.time = self.data.iloc[:, 0].values
        self.interval = np.mean(np.diff(self.time))
        self.fs = 1.0 / self.interval
        self.baseline_corrector = BaselineCorrector(fs=self.fs)
        self.df_corrected = self.apply_baseline_correction()

    def load_data(self):
        df = pd.read_csv(self.file_path, sep='\t', decimal=',', encoding='windows-1251')
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)
        return df

    def apply_baseline_correction(self):
        corrected_df = self.data.copy()
        for sensor in corrected_df.columns[1:]:
            data = corrected_df[sensor].values
            baseline = self.baseline_corrector.baseline_median(data)
            corrected_df[sensor] = data - baseline
        return corrected_df

    def run(self):
        analyzer = MeanPeakAnalyzer(self.df_corrected)
        analyzer.analyze()

if __name__ == "__main__":
    file_path = 'resources/клин 600 701.txt'
    pipeline = SignalPipeline(file_path)
    pipeline.run()