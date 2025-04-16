from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QRadioButton, QButtonGroup,
                           QSpinBox, QGroupBox, QSplitter, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np

class CameraTab(QWidget):
    """
    Вкладка для основной съемки и управления камерой
    """
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window
        
        # Флаги состояния
        self.is_capturing = False
        
        # Таймер для обновления UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(500)  # Обновление каждые 500 мс
        
        # Инициализация UI
        self.init_ui()
        
        # Соединяем сигналы
        self.connect_signals()

    def init_ui(self):
        """
        Инициализирует пользовательский интерфейс вкладки
        """
        # Основной макет
        main_layout = QVBoxLayout(self)
        
        # Верхняя панель (режим работы, управление камерой, фоном и т.д.)
        top_panel = QHBoxLayout()
        
        # Панель выбора режима
        mode_panel = QGroupBox("Режим работы")
        mode_layout = QVBoxLayout(mode_panel)
        
        # Радиокнопки выбора режима
        self.camera_radio = QRadioButton("Камера")
        self.file_radio = QRadioButton("Чтение файлов")
        
        # Группа радиокнопок
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.camera_radio, 0)
        self.mode_group.addButton(self.file_radio, 1)
        
        # Кнопка открытия файла
        self.open_file_btn = QPushButton("Открыть файл")
        
        # Добавляем в макет панели режима
        mode_layout.addWidget(self.camera_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addWidget(self.open_file_btn)
        
        # Панель управления камерой
        camera_panel = QGroupBox("Управление камерой")
        camera_layout = QVBoxLayout(camera_panel)
        
        # Выпадающий список камер
        self.camera_label = QLabel("Камера:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Не выбрано")
        self.camera_combo.addItems(self.main_window.camera_manager.get_camera_list())
        
        # Кнопки управления камерой
        self.launch_camera_btn = QPushButton("Launch the camera")
        self.stop_camera_btn = QPushButton("Завершить работу с камерой")
        
        # Добавляем в макет панели камеры
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.launch_camera_btn)
        camera_layout.addWidget(self.stop_camera_btn)
        
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
        
        # Панель управления фоном
        background_panel = QGroupBox("Сбор фона")
        background_layout = QVBoxLayout(background_panel)
        
        # Количество кадров для фона
        bg_frames_layout = QHBoxLayout()
        bg_frames_layout.addWidget(QLabel("Количество кадров:"))
        self.bg_frames_spinbox = QSpinBox()
        self.bg_frames_spinbox.setRange(1, 100)
        self.bg_frames_spinbox.setValue(40)
        bg_frames_layout.addWidget(self.bg_frames_spinbox)
        
        # Кнопки управления сбором фона
        self.get_background_btn = QPushButton("Get background")
        self.stop_bg_collection_btn = QPushButton("Stop")
        
        # Добавляем в макет панели фона
        background_layout.addLayout(bg_frames_layout)
        background_layout.addWidget(self.get_background_btn)
        background_layout.addWidget(self.stop_bg_collection_btn)
        
        # Панель управления съемкой
        capture_panel = QGroupBox("Управление съемкой")
        capture_layout = QVBoxLayout(capture_panel)
        
        # Кнопки управления съемкой
        self.start_capture_btn = QPushButton("Start")
        self.stop_capture_btn = QPushButton("Stop")
        self.export_data_btn = QPushButton("Export Data")
        
        # Добавляем в макет панели съемки
        capture_layout.addWidget(self.start_capture_btn)
        capture_layout.addWidget(self.stop_capture_btn)
        capture_layout.addWidget(self.export_data_btn)
        
        # Панель состояния
        status_panel = QGroupBox("Статус")
        status_layout = QVBoxLayout(status_panel)
        
        # Метка с текущим режимом
        self.mode_status_label = QLabel("Режим: Камера")
        self.camera_status_label = QLabel("Камера: Не подключена")
        self.capture_status_label = QLabel("Сбор данных: Остановлен")
        
        # Добавляем в макет панели состояния
        status_layout.addWidget(self.mode_status_label)
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.capture_status_label)
        
        # Добавляем все панели в верхнюю панель
        top_panel.addWidget(mode_panel)
        top_panel.addWidget(camera_panel)
        top_panel.addWidget(camera_info_panel)
        top_panel.addWidget(background_panel)
        top_panel.addWidget(capture_panel)
        top_panel.addWidget(status_panel)
        
        # Центральная панель с графиками и данными
        central_panel = QSplitter(Qt.Vertical)
        
        # Панель с тепловой картой и проекциями
        plot_panel = QSplitter(Qt.Horizontal)
        plot_panel.setChildrenCollapsible(False)  # Запрещаем сворачивание графиков
        
        # Левая панель с проекцией на ось Y
        self.y_proj_widget = QWidget()
        self.y_proj_widget.setMinimumWidth(120)  # Устанавливаем минимальную ширину
        self.y_proj_layout = QVBoxLayout(self.y_proj_widget)
        self.y_proj_layout.setContentsMargins(0, 0, 0, 0)  # Убираем отступы
        self.y_proj_canvas = None
        
        # Центральная панель с тепловой картой
        self.heatmap_widget = QWidget()
        self.heatmap_widget.setMinimumSize(300, 300)  # Устанавливаем минимальные размеры
        self.heatmap_layout = QVBoxLayout(self.heatmap_widget)
        self.heatmap_layout.setContentsMargins(0, 0, 0, 0)  # Убираем отступы
        self.heatmap_canvas = None
        
        # Нижняя панель с проекцией на ось X
        self.x_proj_widget = QWidget()
        self.x_proj_widget.setMinimumHeight(120)  # Устанавливаем минимальную высоту
        self.x_proj_layout = QVBoxLayout(self.x_proj_widget)
        self.x_proj_layout.setContentsMargins(0, 0, 0, 0)  # Убираем отступы
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
        info_panel = QGroupBox("Информация о пучке")
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
        
        # Инициализируем состояние виджетов
        self.update_ui_state()
        
    def connect_signals(self):
        """
        Соединяет сигналы с обработчиками
        """
        # Режим работы
        self.camera_radio.toggled.connect(self.on_mode_changed)
        self.file_radio.toggled.connect(self.on_mode_changed)
        self.open_file_btn.clicked.connect(self.on_open_file)
        
        # Управление камерой
        self.launch_camera_btn.clicked.connect(self.on_launch_camera)
        self.stop_camera_btn.clicked.connect(self.on_stop_camera)
        
        # Управление фоном
        self.get_background_btn.clicked.connect(self.on_get_background)
        self.stop_bg_collection_btn.clicked.connect(self.on_stop_bg_collection)
        
        # Управление съемкой
        self.start_capture_btn.clicked.connect(self.on_start_capture)
        self.stop_capture_btn.clicked.connect(self.on_stop_capture)
        self.export_data_btn.clicked.connect(self.on_export_data)
        
        # Устанавливаем начальный режим
        self.camera_radio.setChecked(True)
        
    def update_ui_state(self):
        """
        Обновляет состояние элементов интерфейса в зависимости от текущего режима
        """
        # Получаем текущий режим
        is_camera_mode = self.main_window.current_mode == "camera"
        
        # Управление режимами
        self.camera_radio.setChecked(is_camera_mode)
        self.file_radio.setChecked(not is_camera_mode)
        
        # Обновление метки режима
        self.mode_status_label.setText(f"Режим: {'Камера' if is_camera_mode else 'Чтение файлов'}")
        
        # Управление кнопкой "Открыть файл" - активна только в режиме чтения файла
        self.open_file_btn.setEnabled(not is_camera_mode)
        
        # В режиме камеры
        if is_camera_mode:
            # Доступность элементов управления камерой
            camera_connected = self.main_window.camera_manager.is_connected
            
            self.camera_combo.setEnabled(not camera_connected)
            self.launch_camera_btn.setEnabled(not camera_connected)
            self.stop_camera_btn.setEnabled(camera_connected)
            
            # Доступность элементов управления фоном
            has_background = self.main_window.image_reader.background is not None
            is_collecting_bg = self.main_window.image_reader.is_collecting_background
            
            self.bg_frames_spinbox.setEnabled(camera_connected and not is_collecting_bg)
            self.get_background_btn.setEnabled(camera_connected and not is_collecting_bg)
            self.stop_bg_collection_btn.setEnabled(camera_connected and is_collecting_bg)
            
            # Доступность элементов управления съемкой
            self.start_capture_btn.setEnabled(camera_connected and has_background and not self.is_capturing)
            self.stop_capture_btn.setEnabled(camera_connected and self.is_capturing)
            self.export_data_btn.setEnabled(camera_connected and not self.is_capturing)
            
            # Обновление статуса камеры
            if camera_connected:
                camera_info = self.main_window.camera_manager.get_camera_info()
                if camera_info:
                    self.camera_status_label.setText(f"Камера: {camera_info.get('name', 'Подключена')}")
                else:
                    self.camera_status_label.setText("Камера: Подключена")
            else:
                self.camera_status_label.setText("Камера: Не подключена")
                
            # Обновление статуса съемки
            if self.is_capturing:
                self.capture_status_label.setText("Сбор данных: Идет сбор")
            elif is_collecting_bg:
                self.capture_status_label.setText("Сбор данных: Сбор фона")
            else:
                self.capture_status_label.setText("Сбор данных: Остановлен")
        else:
            # В режиме чтения файлов
            # Отключаем все элементы управления камерой
            self.camera_combo.setEnabled(False)
            self.launch_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(False)
            
            # Отключаем управление фоном
            self.bg_frames_spinbox.setEnabled(False)
            self.get_background_btn.setEnabled(False)
            self.stop_bg_collection_btn.setEnabled(False)
            
            # Отключаем управление съемкой
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(False)
            
            # Включаем экспорт, если есть данные
            self.export_data_btn.setEnabled(self.main_window.file_data["current_frame"] is not None)
            
            # Обновление статуса
            self.camera_status_label.setText("Камера: -")
            self.capture_status_label.setText("Сбор данных: -")
            
        # Обновляем информацию о камере
        self.update_camera_info()
        
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
        
        if data["current_frame"] is None or camera_info is None:
            return
            
        # Получаем информацию о камере
        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        
        # Тепловая карта
        heatmap_fig = self.main_window.plot_manager.create_heatmap_figure(
            data["current_frame"],
            pixel_size_x,
            pixel_size_y
        )
        
        # Проекции
        x_coords, x_proj, y_coords, y_proj = self.main_window.image_analyzer.calculate_projections(
            data["current_frame"],
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
            self.heatmap_canvas.deleteLater()
            self.heatmap_canvas = None
            
        if self.x_proj_canvas is not None:
            self.x_proj_layout.removeWidget(self.x_proj_canvas)
            self.x_proj_canvas.deleteLater()
            self.x_proj_canvas = None
            
        if self.y_proj_canvas is not None:
            self.y_proj_layout.removeWidget(self.y_proj_canvas)
            self.y_proj_canvas.deleteLater()
            self.y_proj_canvas = None
            
        # Создаем новые canvas
        self.heatmap_canvas = self.main_window.plot_manager.create_canvas_from_figure(heatmap_fig)
        self.x_proj_canvas = self.main_window.plot_manager.create_canvas_from_figure(x_proj_fig)
        self.y_proj_canvas = self.main_window.plot_manager.create_canvas_from_figure(y_proj_fig)
        
        # Добавляем canvas на макеты
        self.heatmap_layout.addWidget(self.heatmap_canvas)
        self.x_proj_layout.addWidget(self.x_proj_canvas)
        self.y_proj_layout.addWidget(self.y_proj_canvas)
        
        # Обновляем информацию о центроиде и RMS
        centroid_x, centroid_y = data["centroid"]
        rms_x, rms_y = data["rms"]
        
        self.centroid_x_label.setText(f"Центроид X: {centroid_x:.6f} мм")
        self.centroid_y_label.setText(f"Центроид Y: {centroid_y:.6f} мм")
        self.rms_x_label.setText(f"RMS X: {rms_x:.6f} мм")
        self.rms_y_label.setText(f"RMS Y: {rms_y:.6f} мм")
        
    def update_tab(self):
        """
        Обновляет содержимое вкладки
        """
        self.update_ui_state()
        self.update_plots()
        
    def on_mode_changed(self, checked):
        """
        Обработчик изменения режима работы
        
        Args:
            checked: Флаг нажатия радиокнопки
        """
        if checked:
            if self.camera_radio.isChecked():
                self.main_window.switch_mode("camera")
            else:
                self.main_window.switch_mode("file")
                
    def on_open_file(self):
        """
        Обработчик нажатия кнопки открытия файла
        """
        if self.main_window.open_file_dialog():
            self.file_radio.setChecked(True)
            
    def on_launch_camera(self):
        """
        Обработчик нажатия кнопки запуска камеры
        """
        camera_name = self.camera_combo.currentText()
        
        if camera_name == "Не выбрано":
            QMessageBox.warning(self, "Предупреждение", "Выберите камеру")
            return
            
        if not self.main_window.connect_to_camera(camera_name):
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключиться к камере {camera_name}")
            
    def on_stop_camera(self):
        """
        Обработчик нажатия кнопки остановки камеры
        """
        if not self.main_window.disconnect_camera():
            QMessageBox.critical(self, "Ошибка", "Не удалось отключиться от камеры")
            
    def on_get_background(self):
        """
        Обработчик нажатия кнопки сбора фона
        """
        # Получаем количество кадров
        frames_count = self.bg_frames_spinbox.value()
        
        # Показываем предупреждение
        reply = QMessageBox.question(
            self,
            "Сбор фона",
            "Убедитесь, что вы закрыли затвор лазера. Начать?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if not self.main_window.start_background_collection(frames_count):
                QMessageBox.critical(self, "Ошибка", "Не удалось запустить сбор фона")
                
    def on_stop_bg_collection(self):
        """
        Обработчик нажатия кнопки остановки сбора фона
        """
        if not self.main_window.stop_background_collection():
            QMessageBox.critical(self, "Ошибка", "Не удалось остановить сбор фона")
            
    def on_start_capture(self):
        """
        Обработчик нажатия кнопки запуска съемки
        """
        # Проверяем, что фон собран
        if self.main_window.image_reader.background is None:
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Необходимо собрать фон перед началом съемки"
            )
            return
            
        self.is_capturing = True
        self.update_ui_state()
        
    def on_stop_capture(self):
        """
        Обработчик нажатия кнопки остановки съемки
        """
        self.is_capturing = False
        self.update_ui_state()
        
    def on_export_data(self):
        """
        Обработчик нажатия кнопки экспорта данных
        """
        self.main_window.save_file_dialog()
