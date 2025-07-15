import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# Загрузка данных
data = pd.read_csv('клин 600 701.txt', delimiter='\t', decimal=',', encoding='windows-1251')
data.columns = data.columns.str.strip()

# Создаем копию данных для очищенной версии
cleaned_data = data.copy()


def detect_and_remove_anomalies(sensor_name, contamination=0.005):
    # Создаем фигуру с двумя графиками
    plt.figure(figsize=(16, 12))

    # 1. График ДО обработки (с аномалиями)
    plt.subplot(2, 1, 1)
    plt.plot(data['Время, s'], data[sensor_name],
             label=f'Исходные данные ({sensor_name})',
             color='blue', linewidth=1)

    # Обнаружение аномалий
    sensor_data = data[[sensor_name]].copy()
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(sensor_data)
    point_anomalies = sensor_data[anomalies == -1]

    # Выделение аномалий на графике
    if not point_anomalies.empty:
        plt.scatter(data['Время, s'][point_anomalies.index],
                    point_anomalies[sensor_name],
                    color='red', s=40, zorder=5,
                    label=f'Аномалии ({len(point_anomalies)} шт.)')

    plt.title(f'ДО обработки: {sensor_name} (Обнаружено {len(point_anomalies)} аномалий)')
    plt.xlabel('Время, с')
    plt.ylabel('Значение, mV')
    plt.legend()
    plt.grid(True)

    # 2. График ПОСЛЕ обработки (без аномалий)
    plt.subplot(2, 1, 2)

    # Удаление аномалий (заменяем на NaN)
    cleaned_series = data[sensor_name].copy()
    cleaned_series[point_anomalies.index] = np.nan

    # Просто график без точек
    plt.plot(data['Время, s'], cleaned_series,
             label=f'Очищенные данные ({sensor_name})',
             color='green', linewidth=1)

    plt.title(f'ПОСЛЕ обработки: {sensor_name}')
    plt.xlabel('Время, с')
    plt.ylabel('Значение, mV')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Обновляем очищенные данные
    cleaned_data[sensor_name] = cleaned_series

    # Вывод информации
    if not point_anomalies.empty:
        print(f'\n{sensor_name}:')
        print(f'Удалено аномалий: {len(point_anomalies)}')
        print(f'Процент удаленных точек: {len(point_anomalies) / len(data) * 100:.2f}%')

        # Статистика по аномалиям
        anomalies_report = pd.DataFrame({
            'Время': data.loc[point_anomalies.index, 'Время, s'],
            'Значение': point_anomalies[sensor_name],
            'Отклонение от медианы (%)': ((point_anomalies[sensor_name] - data[sensor_name].median()) /
                                          data[sensor_name].median() * 100).round(1)
        })
        print("\nДетали аномалий:")
        print(anomalies_report.to_string(index=False))
    else:
        print(f'\n{sensor_name}: аномалии не обнаружены')

    print('═' * 80)


# Анализ выбранных датчиков
important_sensors = ['Датчик 1, mV', 'Датчик 2, mV', 'VertG_1_L, mV', 'Long_1_L, mV', 'T3_1_L, mV']

for sensor in important_sensors:
    detect_and_remove_anomalies(sensor, contamination=0.005)

# Сохранение очищенных данных (аномалии заменены на NaN)
cleaned_data.to_csv('cleaned_sensor_data.csv', index=False, encoding='utf-8')
print("\nОчищенные данные сохранены в файл 'cleaned_sensor_data.csv'")