import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).parent))
from SignalPipeline import SignalPipeline
from MeanPeakAnalyzer import MeanPeakAnalyzer
from TrainSpeedAnalyzer import TrainSpeedAnalyzer

class SensorAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор сигналов датчиков")
        self.root.geometry("1400x900")
        
        # Data variables
        self.current_sensor = 0
        self.sensors = []
        self.results = []
        self.pipeline = None
        self.current_filename = "Файл не выбран"
        self.analyzer = None
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        # Main frames
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel with controls and table
        left_panel = tk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # File controls
        file_frame = tk.LabelFrame(left_panel, text="Управление файлом")
        file_frame.pack(fill=tk.X, pady=5)
        
        self.load_btn = tk.Button(file_frame, text="Загрузить файл", command=self.load_file)
        self.load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_label = tk.Label(file_frame, text=self.current_filename, wraplength=280)
        self.file_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Navigation controls
        nav_frame = tk.LabelFrame(left_panel, text="Навигация")
        nav_frame.pack(fill=tk.X, pady=5)
        
        nav_buttons = tk.Frame(nav_frame)
        nav_buttons.pack(fill=tk.X, padx=5, pady=5)
        
        self.prev_btn = tk.Button(nav_buttons, text="← Предыдущий", command=self.prev_sensor, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, expand=True)
        
        self.next_btn = tk.Button(nav_buttons, text="Следующий →", command=self.next_sensor, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, expand=True)
        
        self.show_raw_btn = tk.Button(nav_frame, text="Показать сырой график", command=self.show_raw_graph, state=tk.DISABLED)
        self.show_raw_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.speed_btn = tk.Button(nav_frame, text="Показать скорости", command=self.show_speeds, state=tk.DISABLED)
        self.speed_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Sensors table
        table_frame = tk.LabelFrame(left_panel, text="Датчики")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
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
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Sensor info
        info_frame = tk.LabelFrame(right_panel, text="Информация о датчике")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        info_grid = tk.Frame(info_frame)
        info_grid.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(info_grid, text="Датчик:", anchor='w').grid(row=0, column=0, sticky='w')
        self.sensor_name = tk.Label(info_grid, text="-", anchor='w')
        self.sensor_name.grid(row=0, column=1, sticky='w')
        
        tk.Label(info_grid, text="Тип сигнала:", anchor='w').grid(row=1, column=0, sticky='w')
        self.sensor_type = tk.Label(info_grid, text="-", anchor='w')
        self.sensor_type.grid(row=1, column=1, sticky='w')
        
        tk.Label(info_grid, text="Средний максимум:", anchor='w').grid(row=2, column=0, sticky='w')
        self.mean_max = tk.Label(info_grid, text="-", anchor='w')
        self.mean_max.grid(row=2, column=1, sticky='w')
        
        tk.Label(info_grid, text="Средний минимум:", anchor='w').grid(row=3, column=0, sticky='w')
        self.mean_min = tk.Label(info_grid, text="-", anchor='w')
        self.mean_min.grid(row=3, column=1, sticky='w')
        
        tk.Label(info_grid, text="Соотношение:", anchor='w').grid(row=4, column=0, sticky='w')
        self.ratio = tk.Label(info_grid, text="-", anchor='w')
        self.ratio.grid(row=4, column=1, sticky='w')
        
        # Graph
        self.graph_frame = tk.LabelFrame(right_panel, text="График сигнала")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        self.toolbar.update()
    
    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return
        
        try:
            self.current_filename = Path(file_path).name
            self.file_label.config(text=self.current_filename)
            
            # Process file
            self.pipeline = SignalPipeline(file_path)
            self.analyzer = MeanPeakAnalyzer(self.pipeline.df_corrected, self.pipeline.fs)
            
            # Get results for all sensors
            self.results = []
            for sensor in self.analyzer.sensor_columns:
                data = self.analyzer.df[sensor].values
                
                # Быстрая классификация сигнала
                signal_type = self.analyzer._classify_signal_preliminary(data)
                
                # Для balanced сигналов не ищем пики
                if signal_type == 'balanced':
                    self.results.append({
                        'sensor': sensor,
                        'type': signal_type,
                        'mean_max': 0,
                        'mean_min': 0,
                        'ratio': 0,
                        'data': data,
                        'time': self.analyzer.time,
                        'max_peaks': np.array([], dtype=int),
                        'min_peaks': np.array([], dtype=int),
                        'signal_type': signal_type,
                        'analyzed': False
                    })
                    continue
                
                # Для других типов сигналов проводим полный анализ
                max_peaks, min_peaks = self.analyzer._find_peaks(data)
                signal_type = self.analyzer._classify_signal(data, max_peaks, min_peaks)
                
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
                    'time': self.analyzer.time,
                    'max_peaks': max_peaks,
                    'min_peaks': min_peaks,
                    'signal_type': signal_type,
                    'analyzed': True
                })
            
            self.sensors = [r['sensor'] for r in self.results]
            self.current_sensor = 0
            
            self.update_table()
            self.show_sensor()
            self.update_nav_buttons()
            
            # Enable buttons
            self.speed_btn.config(state=tk.NORMAL)
            self.show_raw_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
            self.close_all()
    
    def show_speeds(self):
        if not self.results or not self.pipeline:
            return
        
        # Create speed window
        speed_window = tk.Toplevel(self.root)
        speed_window.title(f"Скорости поезда - {self.current_filename}")
        speed_window.geometry("800x600")
        
        # Create notebook for different sensors
        notebook = ttk.Notebook(speed_window)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add tab for each sensor with speed data
        for result in self.results:
            if result['signal_type'] not in ['max_dominated', 'min_dominated']:
                continue
                
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=result['sensor'])
            
            # Create table
            tree = ttk.Treeview(frame, columns=('zone', 'time', 'speed', 'delta_t'), show='headings')
            tree.heading('zone', text='Зона')
            tree.heading('time', text='Время (с)')
            tree.heading('speed', text='Скорость (км/ч)')
            tree.heading('delta_t', text='Δt (с)')
            tree.pack(fill=tk.BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Calculate and add speed data
            analyzer = TrainSpeedAnalyzer(graph_type=result['signal_type'])
            peaks = result['max_peaks'] if result['signal_type'] == 'max_dominated' else result['min_peaks']
            speed_data = analyzer.calculate_speeds(result['time'], peaks)
            
            for i, (time, speed, delta_t) in enumerate(speed_data, 1):
                tree.insert('', 'end', values=(
                    f"Зона {i}",
                    f"{time:.3f}",
                    f"{speed:.2f}",
                    f"{delta_t:.3f}"
                ))
    
    def show_raw_graph(self):
        if not self.results or self.current_sensor >= len(self.results):
            return
        
        result = self.results[self.current_sensor]
        
        # Create raw graph window
        raw_window = tk.Toplevel(self.root)
        raw_window.title(f"Сырой график - {result['sensor']}")
        raw_window.geometry("1000x600")
        
        # Create figure
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot raw data
        ax.plot(result['time'], result['data'], 'b-', label='Сигнал', linewidth=1)
        
        # Format graph
        ax.set_title(f"Сырой график: {result['sensor']} (Тип: {result['signal_type']})")
        ax.set_xlabel('Время, с')
        ax.set_ylabel('Напряжение, мВ')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Embed in window
        canvas = FigureCanvasTkAgg(fig, master=raw_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, raw_window)
        toolbar.update()
    
    def show_sensor(self):
        if not self.results or self.current_sensor >= len(self.results):
            return
        
        result = self.results[self.current_sensor]
        
        # Update info
        self.sensor_name.config(text=result['sensor'])
        self.sensor_type.config(text=result['signal_type'])
        self.mean_max.config(text=f"{result['mean_max']:.2f}")
        self.mean_min.config(text=f"{result['mean_min']:.2f}")
        
        ratio_text = f"{result['ratio']:.2f}:1" if result['ratio'] != float('inf') else "∞"
        self.ratio.config(text=ratio_text)
        
        # Update graph
        self.ax.clear()
        
        if not result['analyzed'] or result['signal_type'] == 'balanced':
            # Для неанализированных или balanced сигналов показываем только сырой график
            self.ax.plot(result['time'], result['data'], 'b-', label='Сигнал', linewidth=1)
            self.ax.set_title(f"{result['sensor']} | Тип: {result['signal_type']} (не анализировался)")
        else:
            # Для анализированных сигналов показываем полный график с пиками и зонами
            self.plot_analyzed_signal(result)
        
        self.ax.set_xlabel('Время, с')
        self.ax.set_ylabel('Напряжение, мВ')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def plot_analyzed_signal(self, result):
        """Отрисовка анализированного сигнала с пиками и зонами"""
        # Plot signal
        self.ax.plot(result['time'], result['data'], 'b-', label='Сигнал', linewidth=1)
        
        # Color mapping for signal types
        color_map = {
            'max_dominated': 'red',
            'min_dominated': 'green',
            'balanced': 'blue',
            'abnormal': 'gray'
        }
        
        # Determine active peaks and color based on signal type
        if result['signal_type'] == 'max_dominated':
            active_peaks = result['max_peaks']
            peak_color = 'red'
        elif result['signal_type'] == 'min_dominated':
            active_peaks = result['min_peaks']
            peak_color = 'green'
        else:
            active_peaks = np.array([])
            peak_color = 'gray'
        
        # Create zones for active peaks
        zones = []
        if result['signal_type'] in ['max_dominated', 'min_dominated']:
            peak_count_per_zone = 4
            num_zones = len(active_peaks) // peak_count_per_zone
            for i in range(num_zones):
                start_idx = i * peak_count_per_zone
                end_idx = (i + 1) * peak_count_per_zone
                zone_peaks = active_peaks[start_idx:end_idx]
                if len(zone_peaks) == 0:
                    continue
                zones.append({
                    'start': result['time'][zone_peaks[0]],
                    'end': result['time'][zone_peaks[-1]],
                    'peaks': zone_peaks
                })
        
        # Draw zones
        for i, zone in enumerate(zones):
            self.ax.axvspan(zone['start'], zone['end'], facecolor=peak_color, alpha=0.1)
        
        # Plot all peaks
        self.ax.scatter(result['time'][result['max_peaks']], 
                       result['data'][result['max_peaks']], 
                       color='red', marker='^', s=100, label='Максимумы')
        
        self.ax.scatter(result['time'][result['min_peaks']], 
                       result['data'][result['min_peaks']], 
                       color='green', marker='v', s=100, label='Минимумы')
        
        # Add mean lines
        if len(result['max_peaks']) > 0:
            mean_max = np.mean(result['data'][result['max_peaks']])
            self.ax.axhline(mean_max, color='red', linestyle='--', alpha=0.5, 
                           label=f'Средний максимум: {mean_max:.2f}')
        
        if len(result['min_peaks']) > 0:
            mean_min = np.mean(np.abs(result['data'][result['min_peaks']]))
            self.ax.axhline(-mean_min, color='green', linestyle='--', alpha=0.5, 
                           label=f'Средний минимум: {-mean_min:.2f}')
        
        # Format title
        title_color = color_map.get(result['signal_type'], 'black')
        self.ax.set_title(f"{result['sensor']} | Тип: {result['signal_type']}", color=title_color)
    
    def update_table(self):
        for item in self.table.get_children():
            self.table.delete(item)
        
        for result in self.results:
            self.table.insert('', 'end', values=(result['sensor'], result['signal_type']))
    
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
        self.next_btn.config(state=tk.NORMAL if has_sensors and self.current_sensor < len(self.sensors)-1 else tk.DISABLED)
        
        # Enable raw graph button only if there are sensors
        self.show_raw_btn.config(state=tk.NORMAL if has_sensors else tk.DISABLED)
        
        # Disable speed button for balanced signals
        if has_sensors:
            current_result = self.results[self.current_sensor]
            self.speed_btn.config(state=tk.NORMAL if current_result['signal_type'] in ['max_dominated', 'min_dominated'] else tk.DISABLED)
    
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
        self.analyzer = None
        self.current_filename = "Файл не выбран"
        
        # Reset UI
        self.file_label.config(text=self.current_filename)
        self.sensor_name.config(text="-")
        self.sensor_type.config(text="-")
        self.mean_max.config(text="-")
        self.mean_min.config(text="-")
        self.ratio.config(text="-")
        
        self.update_table()
        
        self.ax.clear()
        self.canvas.draw()
        
        self.update_nav_buttons()
        self.speed_btn.config(state=tk.DISABLED)
        self.show_raw_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorAnalyzerApp(root)
    root.mainloop()