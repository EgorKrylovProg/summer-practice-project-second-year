class TrainSpeedAnalyzer:
    def __init__(self, axle_distance=1.850, graph_type='max_dominated'):
        self.axle_distance = axle_distance  # Расстояние между 3-й и 4-й осями в метрах
        self.graph_type = graph_type.lower()  # Тип графика: 'max_dominated' или 'min_dominated'

    def calculate_speeds(self, timestamps, peaks):
        speed_data = []
        for i in range(3, len(peaks), 4):  # Обрабатываем каждую 4-ю ось (начиная с 3-й)
            if i < len(peaks):
                # Выбираем пики в зависимости от типа графика
                if self.graph_type == 'min_dominated':
                    # Для min_dominated берём минимальные значения в зоне (провалы)
                    zone_peaks = peaks[i-3:i+1]  # 4 оси текущей зоны
                    min_idx = min(zone_peaks, key=lambda x: timestamps[x])  # Индекс min пика
                    max_idx = max(zone_peaks, key=lambda x: timestamps[x])  # Индекс max пика
                    delta_t = timestamps[max_idx] - timestamps[min_idx]  # Разница времени
                else:  # max_dominated (по умолчанию)
                    delta_t = timestamps[peaks[i]] - timestamps[peaks[i-1]]  # Разница между 3-й и 4-й осями

                if delta_t > 0:
                    speed = (self.axle_distance / delta_t) * 3.6  # Переводим в км/ч
                    time = timestamps[peaks[i]]  # Время прохождения 4-й оси
                    speed_data.append((time, speed, delta_t))
        return speed_data

    def print_speeds(self, speed_data):
        if not speed_data:
            print("  [i] Недостаточно пиков для расчёта скорости")
            return

        # Вывод вычислений для первой зоны
        time, speed, delta_t = speed_data[0]
        print("\n[Вычисления для первой зоны]")
        if self.graph_type == 'min_dominated':
            print("  (Тип графика: min_dominated, расчёт по минимальным пикам)")
        else:
            print("  (Тип графика: max_dominated, расчёт по 3-й и 4-й осям)")
        print(f"  - Разница времени (Δt): {delta_t:.3f} с")
        print(f"  - Расстояние между осями: {self.axle_distance} м")
        print(f"  - Скорость (м/с): {self.axle_distance / delta_t:.2f} м/с")
        print(f"  - Скорость (км/ч): {speed:.2f} км/ч")

        # Вывод скоростей для всех зон
        print("\n[Результаты для всех зон]")
        for i, (time, speed, _) in enumerate(speed_data, 1):
            print(f"  Зона {i}: {speed:.2f} км/ч (время: {time:.3f} с)")