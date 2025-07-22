import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import StringVar, Radiobutton, Toplevel
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import pdfplumber
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import re

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
        self.results = []  # Результаты анализа датчиков
        self.nature_list_data = None  # Данные натурного листа
        self.pipeline = None
        self.current_filename = "Файл не выбран"
        self.analyzer = None
        self.export_data_df = None
        
        # Create UI
        self.create_widgets()
        self.show_analysis_mode()
    
    def create_widgets(self):
        # Main container
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Content area (will be filled by mode functions)
        self.content_frame = tk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bottom control panel
        self.bottom_panel = tk.Frame(self.root, height=50, bg='#f0f0f0')
        self.bottom_panel.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        # Mode switch buttons
        self.analysis_btn = tk.Button(
            self.bottom_panel, 
            text="Анализ графиков", 
            command=self.show_analysis_mode,
            width=20,
            height=2
        )
        self.analysis_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.pdf_btn = tk.Button(
            self.bottom_panel, 
            text="Работа с PDF", 
            command=self.show_pdf_mode,
            width=20,
            height=2
        )
        self.pdf_btn.pack(side=tk.LEFT, padx=10, pady=5)

    def _update_preview_table_columns(self):
        """Обновляет колонки таблицы предпросмотра с настройками для горизонтального скролла"""
        if not hasattr(self, 'export_data_df') or self.export_data_df.empty:
            return
        
        # Очищаем текущие колонки
        for col in self.preview_table['columns']:
            self.preview_table.heading(col, text='')
            self.preview_table.column(col, width=0)
        
        # Устанавливаем новые колонки
        columns = list(self.export_data_df.columns)
        self.preview_table['columns'] = columns
        
        # Настраиваем заголовки и ширину колонок
        for col in columns:
            self.preview_table.heading(col, text=col, anchor='center')
            # Устанавливаем минимальную ширину и возможность растягивания
            self.preview_table.column(col, width=100, minwidth=50, stretch=tk.YES, anchor='center')
        
        # Первые две колонки делаем фиксированной ширины
        if len(columns) >= 1:
            self.preview_table.column(columns[0], width=80, stretch=tk.NO)
        if len(columns) >= 2:
            self.preview_table.column(columns[1], width=100, stretch=tk.NO)

    def _fill_preview_table(self):
        """Заполняет таблицу предпросмотра данными"""
        if not hasattr(self, 'export_data_df') or self.export_data_df.empty:
            return
        
        self.preview_table.delete(*self.preview_table.get_children())
        
        for _, row in self.export_data_df.iterrows():
            values = []
            for col in self.export_data_df.columns:
                # Форматируем числа с двумя знаками после запятой
                if isinstance(row[col], (int, float)):
                    values.append(f"{row[col]:.2f}")
                else:
                    values.append(str(row[col]))
            
            self.preview_table.insert('', 'end', values=values)

    def create_side_panel(self):
        # Left panel with buttons
        self.side_panel = tk.Frame(self.root, width=150, bg='#f0f0f0')
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Analysis mode button
        self.analyze_btn = tk.Button(
            self.side_panel, 
            text="Анализ графиков", 
            command=self.show_analysis_mode,
            width=20,
            height=2
        )
        self.analyze_btn.pack(pady=10, padx=5, fill=tk.X)
        
        # Import mode button
        self.import_btn = tk.Button(
            self.side_panel, 
            text="Импорт натурного листа", 
            command=self.show_import_mode,
            width=20,
            height=2
        )
        self.import_btn.pack(pady=10, padx=5, fill=tk.X)
        
        # Start with analysis mode
        self.show_analysis_mode()
    def clear_content(self):
        """Очищает область контента, не затрагивая данные"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    def show_pdf_mode(self):
        """Показывает интерфейс работы с PDF с сохранением состояния"""
        self._save_analysis_mode_state()
        self.clear_content()
        
        # Title
        title_label = tk.Label(self.content_frame, text="Работа с натурным листом (PDF)", font=('Arial', 14))
        title_label.pack(pady=10)
        
        # Load PDF button
        self.load_pdf_btn = tk.Button(
            self.content_frame, 
            text="Загрузить PDF", 
            command=self.load_pdf,
            width=20,
            height=2
        )
        self.load_pdf_btn.pack(pady=10)
        
        # Preview frame with both scrollbars
        preview_frame = tk.LabelFrame(self.content_frame, text="Предпросмотр данных")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create horizontal scrollbar
        xscrollbar = ttk.Scrollbar(preview_frame, orient="horizontal")
        xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create vertical scrollbar
        yscrollbar = ttk.Scrollbar(preview_frame, orient="vertical")
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create table with both scrollbars
        self.preview_table = ttk.Treeview(
            preview_frame, 
            show='headings',
            xscrollcommand=xscrollbar.set,
            yscrollcommand=yscrollbar.set
        )
        self.preview_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        xscrollbar.config(command=self.preview_table.xview)
        yscrollbar.config(command=self.preview_table.yview)
        
        # Make columns resizable and enable horizontal scrolling
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        
        # Check for existing export data
        export_data_exists = (hasattr(self, 'export_data_df') and 
                            self.export_data_df is not None and 
                            not self.export_data_df.empty)
        
        if export_data_exists:
            self._update_preview_table_columns()
            self._fill_preview_table()
        
        # Export options
        export_frame = tk.Frame(self.content_frame)
        export_frame.pack(pady=10)
        
        tk.Label(export_frame, text="Формат экспорта:").pack(side=tk.LEFT)
        
        self.export_format = StringVar(value='csv')
        
        Radiobutton(
            export_frame, 
            text="CSV (для Excel)", 
            variable=self.export_format, 
            value='csv'
        ).pack(side=tk.LEFT, padx=10)
        
        Radiobutton(
            export_frame, 
            text="TXT", 
            variable=self.export_format, 
            value='txt'
        ).pack(side=tk.LEFT, padx=10)
        
        # Export button
        export_btn_state = tk.NORMAL if export_data_exists else tk.DISABLED
        self.export_btn = tk.Button(
            self.content_frame, 
            text="Экспорт данных", 
            command=self.export_data, 
            state=export_btn_state,
            width=20,
            height=2
        )
        self.export_btn.pack(pady=10)

    def load_nature_list(self):
        """Загружает натурный лист из PDF и корректно извлекает данные 'Т на ось'"""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        
        try:
            with pdfplumber.open(file_path) as pdf:
                self.nature_list_data = []
                
                for page in pdf.pages:
                    # Извлекаем таблицу с текущей страницы
                    table = page.extract_table()
                    if not table or len(table) < 2:  # Пропускаем пустые таблицы или без данных
                        continue
                    
                    # Находим индекс столбца "Т на ось"
                    headers = [col.strip().lower() for col in table[0]]
                    try:
                        load_col_index = headers.index("т на ось")
                    except ValueError:
                        # Альтернативные варианты написания заголовка
                        alt_names = ["t на ось", "нагрузка", "на ось"]
                        for name in alt_names:
                            if name in headers:
                                load_col_index = headers.index(name)
                                break
                        else:
                            continue  # Пропускаем страницу, если не нашли нужный столбец
                    
                    # Обрабатываем строки данных
                    for row in table[1:]:
                        if len(row) <= load_col_index:
                            continue
                        
                        # Извлекаем номер и нагрузку
                        num = row[0].strip() if row[0] else ""
                        load = row[load_col_index].strip().replace(",", ".").replace("°", "") if row[load_col_index] else "0"
                        
                        # Преобразуем нагрузку в число
                        try:
                            load_val = float(load) if load else 0.0
                        except ValueError:
                            load_val = 0.0
                        
                        self.nature_list_data.append({
                            "№ п/п": num,
                            "Т на ось": load_val
                        })
                
                if not self.nature_list_data:
                    messagebox.showerror("Ошибка", "Не удалось найти данные 'Т на ось' в PDF")
                    return
                
                # Синхронизация между вкладками
                if hasattr(self, '_analysis_mode_state'):
                    self._analysis_mode_state['nature_list_data'] = self.nature_list_data.copy()
                if hasattr(self, '_pdf_mode_state'):
                    self._pdf_mode_state['nature_list_data'] = self.nature_list_data.copy()
                
                # Обновление интерфейса
                nature_filename = Path(file_path).name
                current_text = self.file_label.cget("text")
                self.file_label.config(text=f"{current_text}\nНатурный лист: {nature_filename}")
                
                # Подготовка данных для экспорта
                if hasattr(self, 'results') and self.results:
                    self.prepare_export_data()
                    if hasattr(self, 'preview_table'):
                        self._update_preview_table_columns()
                        self._fill_preview_table()
                        self.export_btn.config(state=tk.NORMAL)
                
                messagebox.showinfo("Успех", 
                    f"Натурный лист загружен. Записей: {len(self.nature_list_data)}\n"
                    f"Пример данных: №{self.nature_list_data[0]['№ п/п']} - Т={self.nature_list_data[0]['Т на ось']}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке PDF:\n{str(e)}")
    
    def _save_pdf_mode_state(self):
        """Сохраняет состояние PDF режима перед переключением"""
        self._pdf_mode_state = {
            'export_data_df': self.export_data_df.copy() if hasattr(self, 'export_data_df') and self.export_data_df is not None else None,
            'nature_list_data': self.nature_list_data.copy() if self.nature_list_data else None
        }

    def _save_analysis_mode_state(self):
        """Сохраняет состояние анализа графиков перед переключением"""
        self._analysis_mode_state = {
            'current_sensor': self.current_sensor,
            'results': self.results.copy() if self.results else None,
            'sensors': self.sensors.copy() if self.sensors else None,
            'pipeline': self.pipeline,
            'analyzer': self.analyzer,
            'current_filename': self.current_filename,
            'nature_list_data': self.nature_list_data.copy() if self.nature_list_data else None,
            'table_items': [self.table.item(item) for item in self.table.get_children()] if hasattr(self, 'table') else []
        }

    def load_sensor_data(self):
        """Загружает данные с датчиков из текстового файла и готовит данные для экспорта"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if not file_path:
            return
        
        try:
            self.current_filename = Path(file_path).name
            self.file_label.config(text=f"Датчики: {self.current_filename}")
            
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
            
            # Если есть натурный лист, готовим данные для экспорта
            if hasattr(self, 'nature_list_data') and self.nature_list_data:
                self.prepare_export_data()
            
            # Enable buttons
            self.speed_btn.config(state=tk.NORMAL)
            self.show_raw_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
            self.close_all()
    
    def show_analysis_mode(self):
        """Показывает интерфейс анализа графиков"""
        # Сохраняем состояние PDF режима перед переключением
        if hasattr(self, '_pdf_mode_state'):
            state = self._pdf_mode_state
            self.export_data_df = state['export_data_df'].copy() if state['export_data_df'] is not None else None
            self.nature_list_data = state['nature_list_data'].copy() if state['nature_list_data'] else None
        """Показывает интерфейс анализа графиков"""
        # Восстанавливаем состояние, если оно было сохранено
        if hasattr(self, '_analysis_mode_state'):
            state = self._analysis_mode_state
            self.current_sensor = state['current_sensor']
            self.results = state['results']
            self.sensors = state['sensors']
            self.pipeline = state['pipeline']
            self.analyzer = state['analyzer']
            self.current_filename = state['current_filename']
            self.nature_list_data = state['nature_list_data']
        
        self.clear_content()
        
        # Left panel with controls and table
        left_panel = tk.Frame(self.content_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # File controls frame
        file_frame = tk.LabelFrame(left_panel, text="Управление данными")
        file_frame.pack(fill=tk.X, pady=5)
        
        # Button to load sensor data
        self.load_sensors_btn = tk.Button(
            file_frame, 
            text="Загрузить данные датчиков", 
            command=self.load_sensor_data,
            width=20,
            height=2
        )
        self.load_sensors_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Button to load nature list
        self.load_nature_btn = tk.Button(
            file_frame, 
            text="Загрузить натурный лист", 
            command=self.load_nature_list,
            width=20,
            height=2
        )
        self.load_nature_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # File info label
        self.file_label = tk.Label(file_frame, text=self.current_filename if self.current_filename else "Файл не выбран", wraplength=280)
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
        right_panel = tk.Frame(self.content_frame)
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
        
        # Восстанавливаем данные после создания интерфейса
        if hasattr(self, '_analysis_mode_state'):
            # Обновляем таблицу датчиков
            self.update_table()
            
            # Показываем текущий датчик
            if self.sensors and 0 <= self.current_sensor < len(self.sensors):
                self.show_sensor()
            
            # Обновляем кнопки навигации
            self.update_nav_buttons()
            
            # Обновляем информацию о файле
            file_text = f"Датчики: {self.current_filename}" if self.current_filename else "Файл не выбран"
            if self.nature_list_data:
                file_text += "\nНатурный лист: загружен"
            self.file_label.config(text=file_text)
    

    def _parse_complex_page(self, page):
        """Парсинг страниц со сложной структурой"""
        text = page.extract_text()
        if not text:
            return
        
        # Улучшенный поиск данных в тексте
        lines = text.split('\n')
        for line in lines:
            parts = re.split(r'\s{2,}|\|', line.strip())
            if len(parts) >= 2:
                num_part = parts[0].strip()
                load_part = parts[1].replace(",", ".").strip()
                
                if (num_part.isdigit() and 
                    load_part.replace(".", "").isdigit()):
                    try:
                        self.nature_list_data.append({
                            "№ п/п": num_part,
                            "Т на ось": float(load_part)
                        })
                    except ValueError:
                        continue
    
    def _validate_and_save_results(self, file_path):
        """Проверка и сохранение результатов"""
        if not self.nature_list_data:
            messagebox.showerror("Ошибка", "Не удалось извлечь данные")
            return
        
        # Удаление дубликатов и сортировка
        df = pd.DataFrame(self.nature_list_data)
        df = df.drop_duplicates(subset=["№ п/п"])
        df = df.sort_values(by=["№ п/п"])
        self.nature_list_data = df.to_dict('records')
        
        # Обновление интерфейса
        nature_filename = Path(file_path).name
        self.file_label.config(text=f"Натурный лист: {nature_filename} (записей: {len(self.nature_list_data)})")
        
        # Автоматическая подготовка к экспорту
        if hasattr(self, 'results'):
            self.prepare_export_data()
            if hasattr(self, 'preview_table'):
                self._update_preview_table_columns()
                self._fill_preview_table()
        
        messagebox.showinfo("Успех", f"Загружено {len(self.nature_list_data)} записей")

    def load_pdf(self):
        """Улучшенный парсинг сложных PDF таблиц"""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        
        try:
            with pdfplumber.open(file_path) as pdf:
                self.nature_list_data = []
                num_pattern = re.compile(r'^\d+$')  # Для номера
                load_pattern = re.compile(r'^\d+[,.]?\d*$')  # Для нагрузки
                
                for page in pdf.pages:
                    # Улучшенное извлечение таблиц с настройками
                    table = page.extract_table({
                        "vertical_strategy": "text", 
                        "horizontal_strategy": "text",
                        "intersection_y_tolerance": 10
                    })
                    
                    if not table:
                        # Альтернативный метод для сложных страниц
                        self._parse_complex_page(page)
                        continue
                    
                    # Автоматическое определение структуры
                    for row in table:
                        if len(row) < 2:
                            continue
                            
                        # Поиск данных в строке
                        num, load = None, None
                        for cell in row:
                            if not cell:
                                continue
                            if not num and num_pattern.match(str(cell).strip()):
                                num = str(cell).strip()
                            elif not load and load_pattern.match(str(cell).replace(",", ".").strip()):
                                load = float(str(cell).replace(",", ".").strip())
                        
                        if num and load is not None:
                            self.nature_list_data.append({
                                "№ п/п": num,
                                "Т на ось": load
                            })
                
                # Проверка и сохранение результатов
                self._validate_and_save_results(file_path)
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка парсинга: {str(e)}")

    def _update_after_pdf_load(self, file_path):
        """Обновление интерфейса после загрузки PDF"""
        nature_filename = Path(file_path).name
        current_text = self.file_label.cget("text")
        self.file_label.config(text=f"{current_text}\nНатурный лист: {nature_filename}")
        
        # Подготовка данных для экспорта
        if hasattr(self, 'results') and self.results:
            self.prepare_export_data()
            if hasattr(self, 'preview_table'):
                self._update_preview_table_columns()
                self._fill_preview_table()
            self.export_btn.config(state=tk.NORMAL)
        
        messagebox.showinfo("Успех", 
            f"Загружено записей: {len(self.nature_list_data)}\n"
            f"Первая запись: №{self.nature_list_data[0]['№ п/п']} - Т={self.nature_list_data[0]['Т на ось']}\n"
            f"Последняя запись: №{self.nature_list_data[-1]['№ п/п']} - Т={self.nature_list_data[-1]['Т на ось']}")

    def _alternative_pdf_parsing(self, pdf):
        """Альтернативный метод для сложных PDF с проверкой структуры данных"""
        self.nature_list_data = []
        record_count = 0
        
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            for line in lines:
                # Улучшенное определение структуры строки
                parts = [p.strip() for p in re.split(r'\s{2,}|\|', line) if p.strip()]
                
                if len(parts) >= 2:
                    # Проверяем формат номера и нагрузки
                    num_part = parts[0].replace(" ", "")
                    load_part = parts[1].replace(" ", "").replace(",", ".")
                    
                    if (num_part.isdigit() and 
                        load_part.replace(".", "").isdigit() and 
                        "." in load_part):
                        
                        try:
                            self.nature_list_data.append({
                                "№ п/п": num_part,
                                "Т на ось": float(load_part)
                            })
                            record_count += 1
                        except ValueError:
                            continue
        
        # Если нашли хотя бы 5 корректных записей, считаем успешным
        if record_count >= 5:
            return True
        return False

    def _alternative_pdf_parsing(self, pdf):
        """Альтернативный метод парсинга для сложных PDF"""
        self.nature_list_data = []
        
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            
            lines = text.split('\n')
            for line in lines:
                # Ищем строки вида "1|614 40 574|23,4|93,6|P|текст"
                if re.match(r'^\d+\|.+\|\d+,\d+\|', line):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            num = parts[0].strip()
                            load = parts[2].strip().replace(",", ".")
                            
                            try:
                                load_val = float(load) if load else 0.0
                            except ValueError:
                                load_val = 0.0
                            
                            self.nature_list_data.append({
                                "№ п/п": num,
                                "Т на ось": load_val
                            })
                        except (IndexError, ValueError):
                            continue
    
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
    
    def prepare_export_data(self):
        """Подготавливает данные для экспорта, объединяя натурный лист и данные датчиков"""
        if not self.nature_list_data or not self.results:
            self.export_data_df = None
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return None
        
        try:
            # Создаем DataFrame из натурного листа
            export_df = pd.DataFrame(self.nature_list_data)
            
            # Добавляем колонки для каждого подходящего датчика
            for result in self.results:
                if result['signal_type'] not in ['max_dominated', 'min_dominated']:
                    continue
                    
                sensor_name = result['sensor']
                values = []
                
                # Определяем активные пики
                active_peaks = (result['max_peaks'] if result['signal_type'] == 'max_dominated' 
                            else result['min_peaks'])
                
                # Разбиваем на зоны по 4 пика
                num_zones = len(active_peaks) // 4
                for i in range(num_zones):
                    start_idx = i * 4
                    end_idx = (i + 1) * 4
                    zone_peaks = active_peaks[start_idx:end_idx]
                    
                    # Вычисляем среднее значение в зоне
                    zone_mean = np.mean(np.abs(result['data'][zone_peaks]))
                    values.append(zone_mean)
                
                # Добавляем данные датчика (обрезаем до количества строк натурного листа)
                if len(values) >= len(export_df):
                    export_df[sensor_name] = values[:len(export_df)]
                else:
                    # Если зон меньше, чем строк в натурном листе, заполняем оставшиеся NaN
                    values.extend([np.nan] * (len(export_df) - len(values)))
                    export_df[sensor_name] = values
            
            self.export_data_df = export_df
            return export_df
            
        except Exception as e:
            self.export_data_df = None
            messagebox.showerror("Ошибка", f"Ошибка при подготовке данных: {str(e)}")
            return None

    def export_data(self):
        """Экспортирует данные в выбранном формате"""
        if not hasattr(self, 'export_data_df') or self.export_data_df.empty:
            # Если данных нет, попробуем подготовить их
            if not self.prepare_export_data():
                return
        
        file_types = {
            'csv': ("CSV файлы", "*.csv"),
            'txt': ("Текстовые файлы", "*.txt")
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{self.export_format.get()}",
            filetypes=[file_types[self.export_format.get()]],
            title="Сохранить файл"
        )
        
        if not file_path:
            return
        
        try:
            if self.export_format.get() == 'csv':
                self.export_data_df.to_csv(file_path, index=False, sep=';', decimal=',', encoding='utf-8-sig')
            else:
                self.export_data_df.to_string(file_path, index=False)
            
            messagebox.showinfo("Успех", f"Данные успешно экспортированы в {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать данные:\n{str(e)}")

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
    
    def clear_main_area(self):
        # Очищаем основную область
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
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