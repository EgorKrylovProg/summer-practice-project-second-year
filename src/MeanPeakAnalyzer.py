import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class MeanPeakAnalyzer:
    def __init__(self, df):
        self.df = df
        self.time = df.iloc[:, 0].values
        self.sensor_columns = df.columns[1:]

    def _find_peaks(self, data):
        std = np.std(data)
        height = np.percentile(np.abs(data), 70)
        max_peaks, _ = find_peaks(data, height=height, prominence=std*0.5, distance=5)
        min_peaks, _ = find_peaks(-data, height=height, prominence=std*0.5, distance=5)
        return max_peaks, min_peaks

    def _classify_signal(self, data, max_peaks, min_peaks):
        mean_max = np.mean(data[max_peaks]) if len(max_peaks) > 0 else 0
        mean_min = np.mean(np.abs(data[min_peaks])) if len(min_peaks) > 0 else 0
        if mean_max == 0 and mean_min == 0:
            return 'abnormal'
        elif mean_max > mean_min * 1.5:
            return 'max_dominated'
        elif mean_min > mean_max * 1.5:
            return 'min_dominated'
        else:
            return 'balanced'

    def _create_zones(self, peaks, time_values, signal_type):
        zones = []
        if signal_type in ['max_dominated', 'min_dominated']:
            peak_count_per_zone = 4
            num_zones = len(peaks) // peak_count_per_zone
            for i in range(num_zones):
                start_idx = i * peak_count_per_zone
                end_idx = (i + 1) * peak_count_per_zone
                zone_peaks = peaks[start_idx:end_idx]
                if len(zone_peaks) == 0:
                    continue
                zones.append({
                    'start': time_values[zone_peaks[0]],
                    'end': time_values[zone_peaks[-1]],
                    'peaks': zone_peaks
                })
        return zones

    def _plot_signal(self, sensor_name, time, data, max_peaks, min_peaks, signal_type):
        plt.figure(figsize=(14, 6))
        plt.plot(time, data, label='Сигнал')

        color_map = {
            'max_dominated': 'red',
            'min_dominated': 'green',
            'balanced': 'blue',
            'abnormal': 'gray'
        }

        if signal_type == 'max_dominated':
            active_peaks = max_peaks
            peak_color = 'red'
        elif signal_type == 'min_dominated':
            active_peaks = min_peaks
            peak_color = 'green'
        else:
            active_peaks = np.array([])
            peak_color = 'gray'

        zones = self._create_zones(active_peaks, time, signal_type)
        for i, zone in enumerate(zones):
            plt.axvspan(zone['start'], zone['end'], facecolor=peak_color, alpha=0.1)

        plt.scatter(time[max_peaks], data[max_peaks], color='red', marker='^', s=100, label='Максимумы')
        plt.scatter(time[min_peaks], data[min_peaks], color='green', marker='v', s=100, label='Минимумы')

        mean_max = np.mean(data[max_peaks]) if len(max_peaks) > 0 else 0
        mean_min = np.mean(np.abs(data[min_peaks])) if len(min_peaks) > 0 else 0

        plt.title(f'{sensor_name} | Тип: {signal_type}', color=color_map.get(signal_type, 'black'))
        if len(max_peaks) > 0:
            plt.axhline(mean_max, color='red', linestyle='--', alpha=0.5, label=f'Средний максимум: {mean_max:.2f}')
        if len(min_peaks) > 0:
            plt.axhline(-mean_min, color='green', linestyle='--', alpha=0.5, label=f'Средний минимум: {-mean_min:.2f}')

        plt.xlabel('Время, с')
        plt.ylabel('Напряжение, мВ')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze(self):
        print(f"Найдено датчиков: {len(self.sensor_columns)}")
        print("Закрывайте график для перехода к следующему датчику...")

        results = []
        for sensor in self.sensor_columns:
            print(f"\nАнализ датчика: {sensor}")
            data = self.df[sensor].values
            max_peaks, min_peaks = self._find_peaks(data)
            signal_type = self._classify_signal(data, max_peaks, min_peaks)
            mean_max = np.mean(data[max_peaks]) if len(max_peaks) > 0 else 0
            mean_min = np.mean(np.abs(data[min_peaks])) if len(min_peaks) > 0 else 0
            ratio = mean_max / mean_min if mean_min != 0 else float('inf')

            print(f"Тип: {signal_type}, Средний максимум: {mean_max:.2f}, Средний минимум: {mean_min:.2f}, Соотношение: {ratio:.2f}:1")
            self._plot_signal(sensor, self.time, data, max_peaks, min_peaks, signal_type)

            results.append({
                'sensor': sensor,
                'type': signal_type,
                'mean_max': mean_max,
                'mean_min': mean_min,
                'ratio': ratio
            })

        self._print_summary(results)
        return results

    def _print_summary(self, results):
        print("\n=== ИТОГОВАЯ СТАТИСТИКА ===")
        type_counts = {'max_dominated': 0, 'min_dominated': 0, 'balanced': 0, 'abnormal': 0}
        for res in results:
            type_counts[res['type']] += 1
        for typ, count in type_counts.items():
            if count:
                print(f"{typ}: {count}")
        for typ in ['max_dominated', 'min_dominated']:
            filtered = [r for r in results if r['type'] == typ]
            if filtered:
                ratios = [r['ratio'] if typ == 'max_dominated' else 1 / r['ratio'] for r in filtered]
                avg_ratio = np.mean(ratios)
                print(f"Среднее соотношение для {typ}: {avg_ratio:.2f}:1")
