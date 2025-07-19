import pandas as pd


class BaselineCorrector:
    def __init__(self, fs, win_sec=0.5):
        self.fs = fs
        self.win_sec = win_sec

    def baseline_median(self, data):
        w = int(self.win_sec * self.fs)
        if w % 2 == 0: w += 1
        return pd.Series(data).rolling(window=w, center=True, min_periods=1).median().values

