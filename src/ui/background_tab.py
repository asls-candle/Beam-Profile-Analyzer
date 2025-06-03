from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QSplitter, QGridLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.ui.constants import TOP_PANEL_HEIGHT

class BackgroundTab(QWidget):
    """
    Вкладка для отображения фонового изображения
    """
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window
        self.last_data_hash = None  # Для отслеживания изменений
        
        # Таймер для обновления UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(1000)  # Обновление каждую секунду
        
        # Инициализация UI
        self.init_ui()
        
    def init_ui(self):
        """
        Инициализирует пользовательский интерфейс вкладки
        """
        # Основной макет
        main_layout = QVBoxLayout(self)
        
        # Верхняя панель с информацией о камере
        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)  # Увеличиваем расстояние между элементами
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)
        
        # Панель информации о камере
        camera_info_panel = QGroupBox("Информация о камере")
        camera_info_panel.setMinimumWidth(150)  # Устанавливаем минимальную ширину
        camera_info_layout = QVBoxLayout(camera_info_panel)
        camera_info_layout.setSpacing(5)  # Уменьшаем расстояние между элементами внутри панели
        
        # Метки с информацией
        self.camera_name_label = QLabel("Название: -")
        self.camera_resolution_label = QLabel("Разрешение: -")
        self.camera_pixel_size_label = QLabel("Размер пикселя: -")
        
        # Добавляем в макет панели информации
        camera_info_layout.addWidget(self.camera_name_label)
        camera_info_layout.addWidget(self.camera_resolution_label)
        camera_info_layout.addWidget(self.camera_pixel_size_label)
        
        # Панель с информацией о центроиде и RMS
        beam_info_panel = QGroupBox("Информация о фоне")
        beam_info_panel.setMinimumWidth(140)  # Устанавливаем минимальную ширину
        beam_info_layout = QVBoxLayout(beam_info_panel)
        beam_info_layout.setSpacing(5)  # Уменьшаем расстояние между элементами внутри панели
        
        # Метки с информацией
        self.centroid_x_label = QLabel("Центроид X: -")
        self.centroid_y_label = QLabel("Центроид Y: -")
        self.rms_x_label = QLabel("RMS X: -")
        self.rms_y_label = QLabel("RMS Y: -")
        
        # Уменьшаем размер шрифта для меток
        font = QFont()
        font.setPointSize(9)
        self.centroid_x_label.setFont(font)
        self.centroid_y_label.setFont(font)
        self.rms_x_label.setFont(font)
        self.rms_y_label.setFont(font)
        
        # Добавляем в макет панели информации
        beam_info_layout.addWidget(self.centroid_x_label)
        beam_info_layout.addWidget(self.centroid_y_label)
        beam_info_layout.addWidget(self.rms_x_label)
        beam_info_layout.addWidget(self.rms_y_label)
        
        # Добавляем панели в верхнюю панель
        top_panel.addWidget(camera_info_panel)
        
        # Создаем невидимые пустые панели для имитации остальных панелей в camera_tab
        for i in range(5):  # 5 других панелей в camera_tab
            empty_panel = QGroupBox()
            empty_panel.setMinimumWidth(120)
            empty_panel.setStyleSheet("border: none; background-color: transparent;")
            empty_layout = QVBoxLayout(empty_panel)
            top_panel.addWidget(empty_panel)
            
        # Добавляем панель информации о пучке в конце
        top_panel.addWidget(beam_info_panel)
        
        # Центральный контейнер для графиков - один виджет вместо трех
        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(400, 400)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_canvas = None
        
        # Добавляем все в главный макет
        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.plot_widget, 1)
        
        self.setLayout(main_layout)
        
    def update_camera_info(self):
        """
        Обновляет информацию о камере
        """
        camera_info = self.main_window.get_camera_info()
        
        if camera_info:
            self.camera_name_label.setText("Название: {}".format(camera_info.get('camera_name', '-')))
            resolution = camera_info.get("resolution", (0, 0))
            self.camera_resolution_label.setText("Разрешение: {} x {}".format(resolution[0], resolution[1]))
            
            pixel_size_x = camera_info.get("pixel_size_x", 0)
            pixel_size_y = camera_info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText("Размер пикселя: {:.8f} x {:.8f} мм".format(pixel_size_x, pixel_size_y))
        else:
            self.camera_name_label.setText("Название: -")
            self.camera_resolution_label.setText("Разрешение: -")
            self.camera_pixel_size_label.setText("Размер пикселя: -")
            
    def update_plots(self):
        """
        Обновляет все графики только если данные изменились
        """
        # Получаем текущие данные
        data = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()
        
        # Если нет данных фона или информации о камере
        if data["background"] is None or camera_info is None:
            return
        
        # Проверяем изменились ли данные
        current_hash = hash(str(data["background"].data.tobytes()))
        if self.last_data_hash == current_hash and self.plot_canvas is not None:
            return  # Данные не изменились, выходим
        
        self.last_data_hash = current_hash
        
        # Получаем информацию о камере
        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        
        # Создаем новую фигуру с тремя областями
        fig = Figure(figsize=(10, 8))
        
        # Изменяем GridSpec для добавления места под colorbar
        gs = GridSpec(2, 3, width_ratios=[0.5, 5, 0.3], height_ratios=[5, 0.8], figure=fig)
        
        # Область для проекции Y (слева)
        ax_y_proj = fig.add_subplot(gs[0, 0])
        # Область для тепловой карты (справа вверху)
        ax_heatmap = fig.add_subplot(gs[0, 1])
        # Область для проекции X (справа внизу)
        ax_x_proj = fig.add_subplot(gs[1, 1])
        # Область для colorbar
        cax = fig.add_subplot(gs[0, 2])
        
        # Получаем данные изображения
        img_data = data["background"].data
        
        # Вычисляем координаты в миллиметрах (центрированные относительно нуля)
        height, width = img_data.shape
        x_mm = (np.arange(width) - width / 2) * pixel_size_x
        y_mm = (np.arange(height) - height / 2) * pixel_size_y
        
        # Тепловая карта
        im = ax_heatmap.imshow(
            img_data, 
            extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
            origin='lower', 
            aspect='auto',
            cmap='jet'
        )
        ax_heatmap.set_xlabel('X (мм)')
        ax_heatmap.set_ylabel('Y (мм)')
        ax_heatmap.set_title('Профиль пучка')
        
        # Добавляем colorbar
        fig.colorbar(im, cax=cax, label='Интенсивность')
        
        # Проекции
        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)
        
        # Аппроксимация гауссианой - отключена
        # _, gauss_x = self.main_window.image_analyzer.fit_gaussian(x_mm, x_proj)
        # _, gauss_y = self.main_window.image_analyzer.fit_gaussian(y_mm, y_proj)
        
        # Рисуем проекцию X
        ax_x_proj.plot(x_mm, x_proj, 'b-', linewidth=1, label='Данные')
        # if gauss_x is not None:
        #     ax_x_proj.plot(x_mm, gauss_x, 'r--', label='Гаусс')
        ax_x_proj.set_xlabel('X (мм)')
        ax_x_proj.set_ylabel('Интенсивность')
        ax_x_proj.grid(True, linestyle='--', alpha=0.7)
        
        # Рисуем проекцию Y
        ax_y_proj.plot(y_proj, y_mm, 'b-', linewidth=1, label='Данные')
        # if gauss_y is not None:
        #     ax_y_proj.plot(gauss_y, y_mm, 'r--', label='Гаусс')
        ax_y_proj.set_ylabel('Y (мм)')
        ax_y_proj.set_xlabel('Интенсивность')
        ax_y_proj.grid(True, linestyle='--', alpha=0.7)
        
        # Выравниваем оси
        ax_y_proj.set_ylim(ax_heatmap.get_ylim())
        ax_x_proj.set_xlim(ax_heatmap.get_xlim())
        
        # Убираем пустое место
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
        
        # Очищаем текущий холст
        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            # Правильное освобождение ресурсов matplotlib
            plt_figure = self.plot_canvas.figure
            plt_figure.clear()
            self.plot_canvas.close()
        
        # Создаем новый холст
        self.plot_canvas = FigureCanvasQTAgg(fig)
        
        # Добавляем холст на макет
        self.plot_layout.addWidget(self.plot_canvas)
        
        # Вычисляем или берем предварительно рассчитанные центроид и RMS фона
        if self.main_window.current_mode == "file" and "background_centroid" in self.main_window.file_data:
            # Используем предварительно вычисленные значения
            centroid_x, centroid_y = self.main_window.file_data["background_centroid"]
            rms_x, rms_y = self.main_window.file_data["background_rms"]
        else:
            # Вычисляем на месте
            centroid_x, centroid_y = self.main_window.image_analyzer.calculate_centroid(
                data["background"], pixel_size_x, pixel_size_y
            )
            
            rms_x, rms_y = self.main_window.image_analyzer.calculate_rms(
                data["background"], pixel_size_x, pixel_size_y
            )
        
        # Обновляем информацию о центроиде и RMS
        self.centroid_x_label.setText("Центроид X: {:.6f} мм".format(centroid_x))
        self.centroid_y_label.setText("Центроид Y: {:.6f} мм".format(centroid_y))
        self.rms_x_label.setText("RMS X: {:.6f} мм".format(rms_x))
        self.rms_y_label.setText("RMS Y: {:.6f} мм".format(rms_y))
        
    def update_tab(self):
        """
        Обновляет содержимое вкладки
        """
        self.update_camera_info()
        self.update_plots()
