from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QSplitter
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np

class BackgroundTab(QWidget):
    """
    Вкладка для отображения фонового изображения
    """
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window
        
        # Таймер для обновления UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(500)  # Обновление каждые 500 мс
        
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
        
        # Панель информации о камере
        camera_info_panel = QGroupBox("Информация о камере")
        camera_info_layout = QVBoxLayout(camera_info_panel)
        
        # Метки с информацией
        self.camera_name_label = QLabel("Название: -")
        self.camera_resolution_label = QLabel("Разрешение: -")
        self.camera_pixel_size_label = QLabel("Размер пикселя: -")
        
        # Добавляем в макет панели информации
        camera_info_layout.addWidget(self.camera_name_label)
        camera_info_layout.addWidget(self.camera_resolution_label)
        camera_info_layout.addWidget(self.camera_pixel_size_label)
        
        # Добавляем в верхнюю панель
        top_panel.addWidget(camera_info_panel)
        top_panel.addStretch(1)  # Растягиваем пустое пространство
        
        # Центральная панель с графиками и данными
        central_panel = QSplitter(Qt.Vertical)
        
        # Панель с тепловой картой и проекциями
        plot_panel = QSplitter(Qt.Horizontal)
        
        # Левая панель с проекцией на ось Y
        self.y_proj_widget = QWidget()
        self.y_proj_layout = QVBoxLayout(self.y_proj_widget)
        self.y_proj_canvas = None
        
        # Центральная панель с тепловой картой
        self.heatmap_widget = QWidget()
        self.heatmap_layout = QVBoxLayout(self.heatmap_widget)
        self.heatmap_canvas = None
        
        # Нижняя панель с проекцией на ось X
        self.x_proj_widget = QWidget()
        self.x_proj_layout = QVBoxLayout(self.x_proj_widget)
        self.x_proj_canvas = None
        
        # Добавляем виджеты в панель с графиками
        plot_panel.addWidget(self.y_proj_widget)
        plot_panel.addWidget(self.heatmap_widget)
        
        # Формируем панель с проекцией X под тепловой картой
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout(bottom_panel)
        
        # Пустой виджет для выравнивания с панелью Y
        empty_widget = QWidget()
        empty_widget.setFixedWidth(self.y_proj_widget.sizeHint().width())
        
        bottom_layout.addWidget(empty_widget)
        bottom_layout.addWidget(self.x_proj_widget)
        
        # Панель с информацией о центроиде и RMS
        info_panel = QGroupBox("Информация о фоне")
        info_layout = QVBoxLayout(info_panel)
        
        # Метки с информацией
        self.centroid_x_label = QLabel("Центроид X: -")
        self.centroid_y_label = QLabel("Центроид Y: -")
        self.rms_x_label = QLabel("RMS X: -")
        self.rms_y_label = QLabel("RMS Y: -")
        
        # Размер шрифта
        font = QFont()
        font.setPointSize(12)
        self.centroid_x_label.setFont(font)
        self.centroid_y_label.setFont(font)
        self.rms_x_label.setFont(font)
        self.rms_y_label.setFont(font)
        
        # Добавляем в макет панели информации
        info_layout.addWidget(self.centroid_x_label)
        info_layout.addWidget(self.centroid_y_label)
        info_layout.addWidget(self.rms_x_label)
        info_layout.addWidget(self.rms_y_label)
        
        # Добавляем все в центральную панель
        central_panel.addWidget(plot_panel)
        central_panel.addWidget(bottom_panel)
        central_panel.addWidget(info_panel)
        
        # Устанавливаем размеры сплиттеров
        central_panel.setSizes([600, 200, 100])
        plot_panel.setSizes([150, 600])
        
        # Добавляем все в главный макет
        main_layout.addLayout(top_panel)
        main_layout.addWidget(central_panel, 1)
        
        self.setLayout(main_layout)
        
    def update_camera_info(self):
        """
        Обновляет информацию о камере
        """
        camera_info = self.main_window.get_camera_info()
        
        if camera_info:
            self.camera_name_label.setText(f"Название: {camera_info.get('camera_name', '-')}")
            resolution = camera_info.get("resolution", (0, 0))
            self.camera_resolution_label.setText(f"Разрешение: {resolution[0]} x {resolution[1]}")
            
            pixel_size_x = camera_info.get("pixel_size_x", 0)
            pixel_size_y = camera_info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText(f"Размер пикселя: {pixel_size_x:.8f} x {pixel_size_y:.8f} мм")
        else:
            self.camera_name_label.setText("Название: -")
            self.camera_resolution_label.setText("Разрешение: -")
            self.camera_pixel_size_label.setText("Размер пикселя: -")
            
    def update_plots(self):
        """
        Обновляет все графики
        """
        # Получаем текущие данные
        data = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()
        
        # Если нет данных фона или информации о камере
        if data["background"] is None or camera_info is None:
            return
            
        # Получаем информацию о камере
        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        
        # Тепловая карта фона
        heatmap_fig = self.main_window.plot_manager.create_heatmap_figure(
            data["background"],
            pixel_size_x,
            pixel_size_y
        )
        
        # Проекции фона
        x_coords, y_coords, x_proj, y_proj = self.main_window.image_analyzer.calculate_projections(
            data["background"],
            pixel_size_x,
            pixel_size_y
        )
        
        # Аппроксимация гауссианой
        if x_coords is not None and x_proj is not None:
            _, gauss_x = self.main_window.image_analyzer.fit_gaussian(x_coords, x_proj)
        else:
            gauss_x = None
            
        if y_coords is not None and y_proj is not None:
            _, gauss_y = self.main_window.image_analyzer.fit_gaussian(y_coords, y_proj)
        else:
            gauss_y = None
            
        x_proj_fig, y_proj_fig = self.main_window.plot_manager.create_projection_figure(
            x_coords, x_proj, y_coords, y_proj, gauss_x, gauss_y
        )
        
        # Очищаем текущие графики
        if self.heatmap_canvas is not None:
            self.heatmap_layout.removeWidget(self.heatmap_canvas)
            self.heatmap_canvas.close()
            
        if self.x_proj_canvas is not None:
            self.x_proj_layout.removeWidget(self.x_proj_canvas)
            self.x_proj_canvas.close()
            
        if self.y_proj_canvas is not None:
            self.y_proj_layout.removeWidget(self.y_proj_canvas)
            self.y_proj_canvas.close()
            
        # Создаем новые canvas
        self.heatmap_canvas = self.main_window.plot_manager.create_canvas_from_figure(heatmap_fig)
        self.x_proj_canvas = self.main_window.plot_manager.create_canvas_from_figure(x_proj_fig)
        self.y_proj_canvas = self.main_window.plot_manager.create_canvas_from_figure(y_proj_fig)
        
        # Добавляем canvas на макеты
        self.heatmap_layout.addWidget(self.heatmap_canvas)
        self.x_proj_layout.addWidget(self.x_proj_canvas)
        self.y_proj_layout.addWidget(self.y_proj_canvas)
        
        # Вычисляем центроид и RMS фона
        centroid_x, centroid_y = self.main_window.image_analyzer.calculate_centroid(
            data["background"],
            pixel_size_x,
            pixel_size_y
        )
        
        rms_x, rms_y = self.main_window.image_analyzer.calculate_rms(
            data["background"],
            pixel_size_x,
            pixel_size_y
        )
        
        # Обновляем информацию о центроиде и RMS
        self.centroid_x_label.setText(f"Центроид X: {centroid_x:.6f} мм")
        self.centroid_y_label.setText(f"Центроид Y: {centroid_y:.6f} мм")
        self.rms_x_label.setText(f"RMS X: {rms_x:.6f} мм")
        self.rms_y_label.setText(f"RMS Y: {rms_y:.6f} мм")
        
    def update_tab(self):
        """
        Обновляет содержимое вкладки
        """
        self.update_camera_info()
        self.update_plots()
