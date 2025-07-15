import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, contamination=0.01):
        self.contamination = contamination

    def detect_anomalies(self, time_values, sensor_data, sensor_name, plot=False):
        # Создаем копию данных для очищенной версии
        cleaned_data = sensor_data.copy()

        # Обнаружение аномалий
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        anomalies = iso_forest.fit_predict(sensor_data.values.reshape(-1, 1))
        point_anomalies = sensor_data[anomalies == -1]

        if plot:
            self._plot_anomalies(time_values, sensor_data, sensor_name, point_anomalies)

        # Удаление аномалий (заменяем на NaN)
        cleaned_data[point_anomalies.index] = np.nan

        # Интерполяция пропущенных значений
        cleaned_data = cleaned_data.interpolate(method='linear', limit_direction='both')

        return cleaned_data, point_anomalies

    def _plot_anomalies(self, time_values, sensor_data, sensor_name, point_anomalies):
        plt.figure(figsize=(16, 6))

        # График ДО обработки (с аномалиями)
        plt.subplot(1, 2, 1)
        plt.plot(time_values, sensor_data, label='Исходные данные', color='blue', linewidth=1)

        if not point_anomalies.empty:
            plt.scatter(time_values[point_anomalies.index], point_anomalies,
                        color='red', s=40, zorder=5, label=f'Аномалии ({len(point_anomalies)} шт.)')

        plt.title(f'ДО обработки: {sensor_name}')
        plt.xlabel('Время, с')
        plt.ylabel('Значение, мВ')
        plt.legend()
        plt.grid(True)

        # График ПОСЛЕ обработки (без аномалий)
        plt.subplot(1, 2, 2)
        cleaned_series = sensor_data.copy()
        cleaned_series[point_anomalies.index] = np.nan
        cleaned_series = cleaned_series.interpolate()

        plt.plot(time_values, cleaned_series, label='Очищенные данные', color='green', linewidth=1)
        plt.title(f'ПОСЛЕ обработки: {sensor_name}')
        plt.xlabel('Время, с')
        plt.ylabel('Значение, мВ')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()