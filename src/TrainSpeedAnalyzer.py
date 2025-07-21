class TrainSpeedAnalyzer:
    def __init__(self, axle_distance=1.850, graph_type='max_dominated'):
        self.axle_distance = axle_distance  # Расстояние между 3-й и 4-й осями в метрах
        self.graph_type = graph_type.lower()  # Тип графика: 'max_dominated' или 'min_dominated'

    def calculate_speeds(self, timestamps, peaks):
        """
        Возвращает список кортежей (время, скорость_кмч, delta_t)
        """
        speed_data = []
        for i in range(3, len(peaks), 4):  # Обрабатываем каждую 4-ю ось (начиная с 3-й)
            if i < len(peaks):
                # Выбираем пики в зависимости от типа графика
                if self.graph_type == 'min_dominated':
                    # Для min_dominated берём минимальные значения в зоне (провалы)
                    zone_peaks = peaks[i - 3:i + 1]  # 4 оси текущей зоны
                    min_idx = min(zone_peaks, key=lambda x: timestamps[x])  # Индекс min пика
                    max_idx = max(zone_peaks, key=lambda x: timestamps[x])  # Индекс max пика
                    delta_t = timestamps[max_idx] - timestamps[min_idx]  # Разница времени
                else:  # max_dominated (по умолчанию)
                    delta_t = timestamps[peaks[i]] - timestamps[peaks[i - 1]]  # Разница между 3-й и 4-й осями

                if delta_t > 0:
                    speed = (self.axle_distance / delta_t) * 3.6  # Переводим в км/ч
                    time = timestamps[peaks[i]]  # Время прохождения 4-й оси
                    speed_data.append((time, speed, delta_t))
        return speed_data

    def get_speed_table(self, timestamps, peaks):
        """
        Возвращает список словарей с данными о скоростях для отображения в таблице
        """
        speed_data = self.calculate_speeds(timestamps, peaks)
        table_data = []

        for i, (time, speed, delta_t) in enumerate(speed_data, 1):
            table_data.append({
                'zone': i,
                'time': f"{time:.3f}",
                'speed_kmh': f"{speed:.2f}",
                'speed_ms': f"{(speed / 3.6):.2f}",
                'delta_t': f"{delta_t:.3f}",
                'distance': f"{self.axle_distance}",
                'graph_type': self.graph_type
            })

        return table_data