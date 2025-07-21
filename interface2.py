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
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    import matplotlib
    matplotlib.use('TkAgg')

sys.path.append(str(Path(__file__).parent))
from src.SignalPipeline import SignalPipeline
from src.MeanPeakAnalyzer import MeanPeakAnalyzer

from PIL import Image, ImageTk


class TrainAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор данных поезда")
        self.root.geometry("1000x700")

        self.current_sensor_index = 0
        self.sensor_names = []
        self.figures = []
        self.canvases = []

        self.setup_ui()

    def setup_ui(self):
        # Создаем фреймы для организации интерфейса
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Кнопки в верхнем фрейме
        self.load_btn = tk.Button(top_frame, text="Загрузить файл", command=self.load_file)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.prev_btn = tk.Button(top_frame, text="← Предыдущий", command=self.prev_sensor, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(top_frame, text="Следующий →", command=self.next_sensor, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.close_btn = tk.Button(top_frame, text="Закрыть все графики", command=self.close_all_figures)
        self.close_btn.pack(side=tk.LEFT, padx=5)

        self.speed_table_btn = tk.Button(top_frame, text="Таблица скоростей", command=self.show_speed_table)
        self.speed_table_btn.pack(side=tk.LEFT, padx=5)

        # Область для графика
        self.graph_frame = tk.Frame(middle_frame)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        # Информационная панель внизу
        self.info_label = tk.Label(bottom_frame, text="Загрузите файл данных для анализа", anchor=tk.W)
        self.info_label.pack(fill=tk.X)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов")
        self.status_bar = tk.Label(bottom_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
        )

        if not file_path:
            return

        self.status_var.set("Обработка файла...")
        self.root.update()

        try:
            self.pipeline = SignalPipeline(file_path)
            self.pipeline.run()

            # Получаем результаты анализа
            self.sensor_names = self.pipeline.df_corrected.columns[1:]
            self.current_sensor_index = 0

            # Активируем кнопки навигации
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)

            # Показываем первый график
            self.show_current_sensor()

            self.info_label.config(text=f"Файл: {file_path.split('/')[-1]} | Всего датчиков: {len(self.sensor_names)}")
            self.status_var.set("Готов")

        except Exception as e:
            self.status_var.set(f"Ошибка: {str(e)}")
            tk.messagebox.showerror("Ошибка", f"Не удалось обработать файл:\n{str(e)}")

    def show_current_sensor(self):
        # Очищаем предыдущий график
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        if not hasattr(self, 'pipeline') or not self.sensor_names:
            return

        sensor_name = self.sensor_names[self.current_sensor_index]
        data = self.pipeline.df_corrected[sensor_name].values

        # Создаем фигуру matplotlib
        fig = plt.Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)

        # Рисуем график
        ax.plot(self.pipeline.time, data, label='Сигнал')
        ax.set_title(f"Датчик: {sensor_name}")
        ax.set_xlabel('Время, с')
        ax.set_ylabel('Напряжение, мВ')
        ax.grid(True)

        # Добавляем график в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Добавляем панель инструментов
        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Сохраняем ссылки для последующего управления
        self.figures.append(fig)
        self.canvases.append(canvas)

        # Обновляем статус
        self.status_var.set(f"Датчик {self.current_sensor_index + 1} из {len(self.sensor_names)}: {sensor_name}")

    def prev_sensor(self):
        if self.current_sensor_index > 0:
            self.current_sensor_index -= 1
            self.show_current_sensor()

    def next_sensor(self):
        if self.current_sensor_index < len(self.sensor_names) - 1:
            self.current_sensor_index += 1
            self.show_current_sensor()

    def close_all_figures(self):
        plt.close('all')
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        self.figures = []
        self.canvases = []
        self.status_var.set("Все графики закрыты")

    def show_speed_table(self):
        if not hasattr(self, 'pipeline'):
            tk.messagebox.showinfo("Информация", "Сначала загрузите файл данных")
            return

        # Создаем новое окно для таблицы
        table_window = tk.Toplevel(self.root)
        table_window.title("Таблица скоростей")
        table_window.geometry("800x600")

        # Создаем Treeview для отображения таблицы
        tree = ttk.Treeview(table_window)

        # Добавляем вертикальную прокрутку
        vsb = ttk.Scrollbar(table_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        # Упаковываем
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Заголовки столбцов
        tree["columns"] = ("zone", "time", "speed")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("zone", anchor=tk.CENTER, width=100)
        tree.column("time", anchor=tk.CENTER, width=200)
        tree.column("speed", anchor=tk.CENTER, width=100)

        tree.heading("zone", text="Зона", anchor=tk.CENTER)
        tree.heading("time", text="Время (с)", anchor=tk.CENTER)
        tree.heading("speed", text="Скорость (км/ч)", anchor=tk.CENTER)

        # Здесь должна быть логика получения данных о скоростях
        # Временно заполняем тестовыми данными
        for i in range(1, 21):
            tree.insert("", tk.END, values=(f"Зона {i}", f"{i * 0.5:.2f}", f"{50 + i}"))

        # Кнопка экспорта
        export_btn = tk.Button(table_window, text="Экспорт в CSV", command=lambda: self.export_to_csv(tree))
        export_btn.pack(side=tk.BOTTOM, pady=5)

    def export_to_csv(self, tree):
        # Получаем данные из Treeview
        items = tree.get_children()
        data = []
        for item in items:
            data.append(tree.item(item)['values'])

        # Создаем DataFrame
        df = pd.DataFrame(data, columns=["Зона", "Время (с)", "Скорость (км/ч)"])

        # Сохраняем в файл
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV файлы", "*.csv"), ("Все файлы", "*.*"))
        )

        if file_path:
            df.to_csv(file_path, index=False, encoding='utf-8')
            self.status_var.set(f"Данные экспортированы в {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainAnalysisApp(root)
    root.mainloop()
