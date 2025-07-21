import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('tkagg')
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')

sys.path.append(str(Path(__file__).parent))
from SignalPipeline import SignalPipeline
from src.MeanPeakAnalyzer import MeanPeakAnalyzer
from TrainSpeedAnalyzer import TrainSpeedAnalyzer


class SensorAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор сигналов")
        self.root.geometry("1200x800")

        # Data variables
        self.current_sensor = 0
        self.sensors = []
        self.results = []
        self.pipeline = None
        self.current_filename = "Файл не выбран"

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Top panel with buttons
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, fill=tk.X)

        self.load_btn = tk.Button(top_frame, text="Загрузить файл", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(top_frame, text=self.current_filename)
        self.file_label.pack(side=tk.LEFT, padx=5)

        # Navigation buttons
        nav_frame = tk.Frame(top_frame)
        nav_frame.pack(side=tk.RIGHT, padx=10)

        self.speed_btn = tk.Button(nav_frame, text="Скорости поезда", command=self.show_speeds, state=tk.DISABLED)
        self.speed_btn.pack(side=tk.LEFT, padx=2)

        self.prev_btn = tk.Button(nav_frame, text="← Предыдущий", command=self.prev_sensor, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=2)

        self.next_btn = tk.Button(nav_frame, text="Следующий →", command=self.next_sensor, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=2)

        self.close_btn = tk.Button(nav_frame, text="Закрыть все", command=self.close_all, state=tk.DISABLED)
        self.close_btn.pack(side=tk.LEFT, padx=2)
        # Main content area
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel with table
        left_frame = tk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Table
        table_frame = tk.LabelFrame(left_frame, text="Датчики")
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.table = ttk.Treeview(table_frame, columns=('sensor', 'type'), show='headings')
        self.table.heading('sensor', text='Датчик')
        self.table.heading('type', text='Тип сигнала')
        self.table.column('sensor', width=120)
        self.table.column('type', width=150)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)

        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.table.bind('<<TreeviewSelect>>', self.on_table_select)

        # Right panel with graph and info
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Info frame
        info_frame = tk.LabelFrame(right_frame, text="Информация о датчике")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.sensor_name_label = tk.Label(info_frame, text="Датчик: -", font=('Arial', 10, 'bold'))
        self.sensor_name_label.pack(anchor='w')

        self.sensor_type_label = tk.Label(info_frame, text="Тип сигнала: -")
        self.sensor_type_label.pack(anchor='w')

        self.max_label = tk.Label(info_frame, text="Средний максимум: -")
        self.max_label.pack(anchor='w')

        self.min_label = tk.Label(info_frame, text="Средний минимум: -")
        self.min_label.pack(anchor='w')

        self.ratio_label = tk.Label(info_frame, text="Соотношение: -")
        self.ratio_label.pack(anchor='w')

        # Graph frame
        self.graph_frame = tk.LabelFrame(right_frame, text="График сигнала")
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Текстовые файлы", "*.txt")])
        if not file_path:
            return

        try:
            self.file_path = file_path
            self.current_filename = file_path.split('/')[-1]
            self.file_label.config(text=self.current_filename)

            # Initialize pipeline and get results
            self.pipeline = SignalPipeline(file_path)

            # Получаем частоту дискретизации из pipeline или устанавливаем фиксированное значение
            fs = getattr(self.pipeline, 'fs', 1000)  # 1000 Гц по умолчанию, если fs не определено

            # Передаем fs в MeanPeakAnalyzer
            analyzer = MeanPeakAnalyzer(self.pipeline.df_corrected, fs=fs)

            self.results = []
            for sensor in analyzer.sensor_columns:
                data = analyzer.df[sensor].values
                max_peaks, min_peaks = analyzer._find_peaks(data)
                signal_type = analyzer._classify_signal(data, max_peaks, min_peaks)

                mean_max = np.mean(data[max_peaks]) if len(max_peaks) > 0 else 0
                mean_min = np.mean(np.abs(data[min_peaks])) if len(min_peaks) > 0 else 0
                ratio = mean_max / mean_min if mean_min != 0 else float('inf')

                self.results.append({
                    'sensor': sensor,
                    'type': signal_type,
                    'mean_max': mean_max,
                    'mean_min': mean_min,
                    'ratio': ratio,
                    'data': data,
                    'time': analyzer.time,
                    'max_peaks': max_peaks,
                    'min_peaks': min_peaks
                })

            self.sensors = [r['sensor'] for r in self.results]
            self.current_sensor = 0

            self.update_table()
            self.show_sensor()
            self.update_nav_buttons()
            #self.update_summary()
            self.speed_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке файла:\n{str(e)}")
            self.close_all()

    def process_file(self, file_path):
        try:
            from SignalPipeline import SignalPipeline
            self.pipeline = SignalPipeline(file_path)
            analyzer = MeanPeakAnalyzer(self.pipeline.df_corrected, self.pipeline.fs)

            self.results = []
            for sensor in analyzer.sensor_columns:
                data = analyzer.df[sensor].values
                max_peaks, min_peaks = analyzer._find_peaks(data)
                signal_type = analyzer._classify_signal(data, max_peaks, min_peaks)

                mean_max = np.mean(data[max_peaks]) if len(max_peaks) > 0 else 0
                mean_min = np.mean(np.abs(data[min_peaks])) if len(min_peaks) > 0 else 0
                ratio = mean_max / mean_min if mean_min != 0 else float('inf')

                self.results.append({
                    'sensor': sensor,
                    'type': signal_type,
                    'mean_max': mean_max,
                    'mean_min': mean_min,
                    'ratio': ratio,
                    'data': data,
                    'time': analyzer.time,
                    'max_peaks': max_peaks,
                    'min_peaks': min_peaks
                })

            self.sensors = [r['sensor'] for r in self.results]
            self.current_sensor = 0

            self.root.after(0, self.finish_loading)

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def finish_loading(self):
        self.update_table()
        self.show_sensor()
        self.load_btn.config(state=tk.NORMAL)
        self.update_nav_buttons()
        self.file_label.config(text=self.file_path.split('/')[-1])

    def show_error(self, message):
        messagebox.showerror("Ошибка", message)
        self.load_btn.config(state=tk.NORMAL)
        self.close_all()

    def update_table(self):
        for item in self.table.get_children():
            self.table.delete(item)

        for result in self.results:
            self.table.insert('', 'end', values=(result['sensor'], result['type']))

    def show_speeds(self):
        if not self.results or not self.pipeline:
            return

        # Создаем новое окно для отображения скоростей
        speed_window = tk.Toplevel(self.root)
        speed_window.title(f"Скорости поезда - {self.current_filename}")
        speed_window.geometry("800x600")

        # Создаем таблицу для отображения скоростей
        tree = ttk.Treeview(speed_window, columns=('zone', 'time', 'speed', 'delta_t'), show='headings')
        tree.heading('zone', text='Зона')
        tree.heading('time', text='Время (с)')
        tree.heading('speed', text='Скорость (км/ч)')
        tree.heading('delta_t', text='Δt (с)')
        tree.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(speed_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Анализируем каждый датчик и получаем скорости
        for result in self.results:
            analyzer = TrainSpeedAnalyzer(graph_type=result['type'])
            speed_data = analyzer.calculate_speeds(result['time'],
                                                   result['max_peaks'] if result['type'] == 'max_dominated' else result[
                                                       'min_peaks'])

            for i, (time, speed, delta_t) in enumerate(speed_data, 1):
                tree.insert('', 'end', values=(
                    f"{result['sensor']} зона {i}",
                    f"{time:.3f}",
                    f"{speed:.2f}",
                    f"{delta_t:.3f}"
                ))

    def show_sensor(self):
        if not self.results or self.current_sensor >= len(self.sensors):
            return

        result = self.results[self.current_sensor]

        # Update info labels
        self.sensor_name_label.config(text=f"Датчик: {result['sensor']}")
        self.sensor_type_label.config(text=f"Тип сигнала: {result['type']}")
        self.max_label.config(text=f"Средний максимум: {result['mean_max']:.2f}")
        self.min_label.config(text=f"Средний минимум: {result['mean_min']:.2f}")
        ratio_text = f"{result['ratio']:.2f}:1" if result['ratio'] != float('inf') else "∞"
        self.ratio_label.config(text=f"Соотношение: {ratio_text}")

        # Update graph
        self.ax.clear()
        self.ax.plot(result['time'], result['data'], 'b-', label='Сигнал')

        if len(result['max_peaks']) > 0:
            self.ax.plot(result['time'][result['max_peaks']],
                         result['data'][result['max_peaks']],
                         'r^', label='Максимумы')

        if len(result['min_peaks']) > 0:
            self.ax.plot(result['time'][result['min_peaks']],
                         result['data'][result['min_peaks']],
                         'gv', label='Минимумы')

        self.ax.set_title(f"Датчик: {result['sensor']} (Тип: {result['type']})")
        self.ax.set_xlabel('Время, с')
        self.ax.set_ylabel('Напряжение, мВ')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def on_table_select(self, event):
        selected = self.table.selection()
        if selected:
            sensor = self.table.item(selected[0], 'values')[0]
            self.current_sensor = self.sensors.index(sensor)
            self.show_sensor()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        has_sensors = len(self.sensors) > 0
        self.prev_btn.config(state=tk.NORMAL if has_sensors and self.current_sensor > 0 else tk.DISABLED)
        self.next_btn.config(
            state=tk.NORMAL if has_sensors and self.current_sensor < len(self.sensors) - 1 else tk.DISABLED)

    def next_sensor(self):
        if self.current_sensor < len(self.sensors) - 1:
            self.current_sensor += 1
            self.show_sensor()
            self.update_nav_buttons()

    def prev_sensor(self):
        if self.current_sensor > 0:
            self.current_sensor -= 1
            self.show_sensor()
            self.update_nav_buttons()

    def close_all(self):
        self.current_sensor = 0
        self.sensors = []
        self.results = []
        self.pipeline = None
        self.file_path = ""

        self.file_label.config(text="Файл не выбран")
        self.sensor_name_label.config(text="Датчик: -")
        self.sensor_type_label.config(text="Тип сигнала: -")
        self.max_label.config(text="Средний максимум: -")
        self.min_label.config(text="Средний минимум: -")
        self.ratio_label.config(text="Соотношение: -")

        for item in self.table.get_children():
            self.table.delete(item)

        self.ax.clear()
        self.canvas.draw()
        self.update_nav_buttons()


if __name__ == "__main__":
    root = tk.Tk()
    app = SensorAnalyzerApp(root)
    root.mainloop()