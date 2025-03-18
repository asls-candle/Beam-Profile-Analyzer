import os
import sys
import logging
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (QMainWindow, QWidget, QTabWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QComboBox, QLabel, 
                            QFileDialog, QMessageBox, QGroupBox, QRadioButton,
                            QButtonGroup, QSpinBox, QLineEdit)
from PyQt6.QtCore import Qt, QTimer

from src.ui.beam_profile_widget import BeamProfileWidget
from src.core.camera_manager import CameraManager
from src.core.data_processor import DataProcessor

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        
        # Настройка логирования
        self.setup_logging()
        
        # Инициализация компонентов
        self.init_ui()
        self.init_camera_manager()
        self.init_data_processor()
        
        # Таймер для обновления изображения
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_frame)
        
        # Установка состояния по умолчанию
        self.set_default_state()
        
        logging.info("Приложение инициализировано")
        
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = "logs"
        
        # Создаем директорию для логов, если она не существует
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Формат имени файла лога с текущей датой и временем
        log_filename = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Настройка логгера
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle("Анализатор профиля пучка")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Вкладка с основными изображениями
        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Основные изображения")
        
        # Вкладка с фоновыми изображениями
        self.background_tab = QWidget()
        self.tabs.addTab(self.background_tab, "Фоновые изображения")
        
        # Вкладка с разностными изображениями
        self.diff_tab = QWidget()
        self.tabs.addTab(self.diff_tab, "Разностные изображения")
        
        # Настройка вкладок
        self.setup_main_tab()
        self.setup_background_tab()
        self.setup_diff_tab()
        
        # Панель управления
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        
        # Группировка режимов работы
        mode_group = QGroupBox("Режим работы")
        mode_layout = QHBoxLayout(mode_group)
        
        self.camera_mode_radio = QRadioButton("Работа с камерой")
        self.file_mode_radio = QRadioButton("Чтение из файла")
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.camera_mode_radio, 1)
        self.mode_group.addButton(self.file_mode_radio, 2)
        self.mode_group.buttonClicked.connect(self.mode_changed)
        
        mode_layout.addWidget(self.camera_mode_radio)
        mode_layout.addWidget(self.file_mode_radio)
        
        control_layout.addWidget(mode_group)
        
        # Настройки камеры
        camera_group = QGroupBox("Настройки камеры")
        camera_layout = QHBoxLayout(camera_group)
        
        camera_layout.addWidget(QLabel("Выбрать камеру:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["GUN_YAG1", "GUN_YAG2"])
        camera_layout.addWidget(self.camera_combo)
        
        camera_layout.addWidget(QLabel("Кол-во кадров для усреднения:"))
        self.avg_frames_spin = QSpinBox()
        self.avg_frames_spin.setRange(1, 10)
        self.avg_frames_spin.setValue(1)
        camera_layout.addWidget(self.avg_frames_spin)
        
        control_layout.addWidget(camera_group)
        
        # Настройки файлов
        file_group = QGroupBox("Работа с файлами")
        file_layout = QHBoxLayout(file_group)
        
        self.load_btn = QPushButton("Загрузить файл...")
        self.load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Сохранить данные...")
        self.save_btn.clicked.connect(self.save_data)
        file_layout.addWidget(self.save_btn)
        
        control_layout.addWidget(file_group)
        
        # Управление камерой
        camera_control_group = QGroupBox("Управление камерой")
        camera_control_layout = QHBoxLayout(camera_control_group)
        
        self.connect_btn = QPushButton("Подключить камеру")
        self.connect_btn.clicked.connect(self.connect_camera)
        camera_control_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("Отключить камеру")
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        camera_control_layout.addWidget(self.disconnect_btn)
        
        self.capture_btn = QPushButton("Захват")
        self.capture_btn.clicked.connect(self.toggle_capture)
        camera_control_layout.addWidget(self.capture_btn)
        
        self.collect_bg_btn = QPushButton("Собрать фон")
        self.collect_bg_btn.clicked.connect(self.collect_background)
        camera_control_layout.addWidget(self.collect_bg_btn)
        
        control_layout.addWidget(camera_control_group)
        
    def setup_main_tab(self):
        """Настройка вкладки с основными изображениями"""
        layout = QVBoxLayout(self.main_tab)
        self.main_widget = BeamProfileWidget()
        layout.addWidget(self.main_widget)
        
    def setup_background_tab(self):
        """Настройка вкладки с фоновыми изображениями"""
        layout = QVBoxLayout(self.background_tab)
        self.background_widget = BeamProfileWidget()
        layout.addWidget(self.background_widget)
        
    def setup_diff_tab(self):
        """Настройка вкладки с разностными изображениями"""
        layout = QVBoxLayout(self.diff_tab)
        self.diff_widget = BeamProfileWidget()
        layout.addWidget(self.diff_widget)
        
    def init_camera_manager(self):
        """Инициализация менеджера камеры"""
        self.camera_manager = CameraManager()
        self.camera_connected = False
        self.is_capturing = False
        
    def init_data_processor(self):
        """Инициализация обработчика данных"""
        self.data_processor = DataProcessor()
        
    def set_default_state(self):
        """Установка состояния по умолчанию"""
        # Устанавливаем режим работы с камерой по умолчанию
        self.camera_mode_radio.setChecked(True)
        self.mode_changed()
        
        # Отключаем кнопки управления камерой
        self.disconnect_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.collect_bg_btn.setEnabled(False)
        
    def mode_changed(self):
        """Обработка изменения режима работы"""
        camera_mode = self.camera_mode_radio.isChecked()
        
        # Включаем/отключаем соответствующие элементы управления
        self.camera_combo.setEnabled(camera_mode)
        self.avg_frames_spin.setEnabled(camera_mode)
        self.connect_btn.setEnabled(camera_mode)
        self.disconnect_btn.setEnabled(camera_mode and self.camera_connected)
        self.capture_btn.setEnabled(camera_mode and self.camera_connected)
        self.collect_bg_btn.setEnabled(camera_mode and self.camera_connected)
        
        self.load_btn.setEnabled(not camera_mode)
        self.save_btn.setEnabled(self.data_processor.has_data())
        
        # Если переключаемся из режима камеры, то останавливаем захват
        if not camera_mode and self.is_capturing:
            self.toggle_capture()
            
        logging.info(f"Режим работы изменен: {'Камера' if camera_mode else 'Файл'}")
        
    def connect_camera(self):
        """Подключение к камере"""
        camera_model = self.camera_combo.currentText()
        
        try:
            logging.info(f"Подключение к камере {camera_model}...")
            self.camera_manager.init_camera(camera_model)
            self.camera_connected = True
            
            # Обновляем состояние кнопок
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            self.collect_bg_btn.setEnabled(True)
            
            QMessageBox.information(self, "Подключение", f"Камера {camera_model} успешно подключена")
            logging.info(f"Камера {camera_model} подключена успешно")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключиться к камере: {str(e)}")
            logging.error(f"Ошибка подключения к камере: {str(e)}")
            
    def disconnect_camera(self):
        """Отключение от камеры"""
        try:
            # Если идет захват, останавливаем его
            if self.is_capturing:
                self.toggle_capture()
                
            logging.info("Отключение от камеры...")
            self.camera_manager.close_camera()
            self.camera_connected = False
            
            # Обновляем состояние кнопок
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.collect_bg_btn.setEnabled(False)
            
            QMessageBox.information(self, "Отключение", "Камера отключена")
            logging.info("Камера отключена успешно")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при отключении камеры: {str(e)}")
            logging.error(f"Ошибка при отключении камеры: {str(e)}")
            
    def toggle_capture(self):
        """Переключение режима захвата кадров"""
        if not self.is_capturing:
            # Начинаем захват
            try:
                logging.info("Начало захвата кадров...")
                self.camera_manager.start_capture()
                self.is_capturing = True
                self.capture_btn.setText("Остановить захват")
                
                # Запускаем таймер для обновления изображения
                self.capture_timer.start(100)  # Каждые 100 мс (10 кадров в секунду)
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось начать захват: {str(e)}")
                logging.error(f"Ошибка при начале захвата: {str(e)}")
                
        else:
            # Останавливаем захват
            try:
                logging.info("Остановка захвата кадров...")
                self.capture_timer.stop()
                self.camera_manager.stop_capture()
                self.is_capturing = False
                self.capture_btn.setText("Захват")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось остановить захват: {str(e)}")
                logging.error(f"Ошибка при остановке захвата: {str(e)}")
                
    def capture_frame(self):
        """Захват кадра и обновление данных"""
        try:
            # Получаем количество кадров для усреднения
            num_frames = self.avg_frames_spin.value()
            
            # Захватываем кадр
            frame = self.camera_manager.capture_frame(num_frames)
            
            # Если есть кадр, обрабатываем его
            if frame is not None:
                # Устанавливаем кадр в обработчик данных
                self.data_processor.set_data(frame)
                
                # Получаем проекции и статистики
                x_proj, y_proj = self.data_processor.calculate_projections()
                centroid_x, centroid_y = self.data_processor.calculate_centroids()
                rms_x, rms_y = self.data_processor.calculate_rms()
                
                # Получаем координатные сетки
                x_mm, y_mm = self.data_processor.get_coordinate_grids()
                
                # Обновляем виджет с данными
                self.main_widget.update_data(
                    frame, x_mm, y_mm, centroid_x, centroid_y, 
                    rms_x, rms_y, x_proj, y_proj
                )
                
                # Если есть фоновые данные, то вычисляем и обновляем разностное изображение
                if self.data_processor.has_background():
                    diff_data = self.data_processor.calculate_difference()
                    
                    # Пересчитываем статистики для разностного изображения
                    self.data_processor.set_data(diff_data)
                    x_proj, y_proj = self.data_processor.calculate_projections()
                    centroid_x, centroid_y = self.data_processor.calculate_centroids()
                    rms_x, rms_y = self.data_processor.calculate_rms()
                    
                    # Обновляем виджет с разностными данными
                    self.diff_widget.update_data(
                        diff_data, x_mm, y_mm, centroid_x, centroid_y, 
                        rms_x, rms_y, x_proj, y_proj
                    )
                    
                    # Возвращаем исходные данные в обработчик
                    self.data_processor.set_data(frame)
                
        except Exception as e:
            logging.error(f"Ошибка при захвате кадра: {str(e)}")
            # Останавливаем захват при ошибке
            self.toggle_capture()
            QMessageBox.critical(self, "Ошибка", f"Ошибка при захвате кадра: {str(e)}")
            
    def collect_background(self):
        """Сбор фонового изображения"""
        try:
            # Получаем количество кадров для усреднения
            num_frames = self.avg_frames_spin.value()
            
            # Сбор фонового изображения
            logging.info(f"Сбор фонового изображения (усреднение {num_frames} кадров)...")
            background = self.camera_manager.collect_background(num_frames)
            
            # Если есть фоновый кадр, обрабатываем его
            if background is not None:
                # Устанавливаем фоновый кадр в обработчик данных
                self.data_processor.set_background(background)
                
                # Получаем проекции и статистики
                self.data_processor.set_data(background)
                x_proj, y_proj = self.data_processor.calculate_projections()
                centroid_x, centroid_y = self.data_processor.calculate_centroids()
                rms_x, rms_y = self.data_processor.calculate_rms()
                
                # Получаем координатные сетки
                x_mm, y_mm = self.data_processor.get_coordinate_grids()
                
                # Обновляем виджет с фоновыми данными
                self.background_widget.update_data(
                    background, x_mm, y_mm, centroid_x, centroid_y, 
                    rms_x, rms_y, x_proj, y_proj
                )
                
                QMessageBox.information(self, "Фон", "Фоновое изображение успешно собрано")
                logging.info("Фоновое изображение успешно собрано")
                
                # Активируем вкладку с фоновым изображением
                self.tabs.setCurrentWidget(self.background_tab)
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сборе фона: {str(e)}")
            logging.error(f"Ошибка при сборе фона: {str(e)}")
            
    def load_file(self):
        """Загрузка данных из файла"""
        try:
            # Открываем диалог выбора файла
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите файл данных", "", 
                "Файлы NumPy (*.npy);;Файлы MATLAB (*.mat);;Все файлы (*.*)"
            )
            
            if file_path:
                logging.info(f"Загрузка данных из файла: {file_path}")
                
                # Загружаем данные из файла
                if file_path.lower().endswith('.npy'):
                    data = self.data_processor.load_from_npy(file_path)
                elif file_path.lower().endswith('.mat'):
                    data = self.data_processor.load_from_mat(file_path)
                else:
                    raise ValueError("Неподдерживаемый формат файла")
                
                # Если данные загружены, обрабатываем их
                if data is not None:
                    # Устанавливаем загруженные данные
                    self.data_processor.set_data(data)
                    
                    # Пытаемся найти фоновый файл
                    bg_file = self.data_processor.find_background_file(file_path)
                    if bg_file:
                        # Загружаем фоновые данные
                        if bg_file.lower().endswith('.npy'):
                            bg_data = self.data_processor.load_from_npy(bg_file)
                        elif bg_file.lower().endswith('.mat'):
                            bg_data = self.data_processor.load_from_mat(bg_file)
                            
                        if bg_data is not None:
                            self.data_processor.set_background(bg_data)
                            logging.info(f"Загружен фоновый файл: {bg_file}")
                    
                    # Получаем проекции и статистики
                    x_proj, y_proj = self.data_processor.calculate_projections()
                    centroid_x, centroid_y = self.data_processor.calculate_centroids()
                    rms_x, rms_y = self.data_processor.calculate_rms()
                    
                    # Получаем координатные сетки
                    x_mm, y_mm = self.data_processor.get_coordinate_grids()
                    
                    # Обновляем виджет с данными
                    self.main_widget.update_data(
                        data, x_mm, y_mm, centroid_x, centroid_y, 
                        rms_x, rms_y, x_proj, y_proj
                    )
                    
                    # Активируем вкладку с основным изображением
                    self.tabs.setCurrentWidget(self.main_tab)
                    
                    # Если есть фоновые данные, обновляем соответствующие виджеты
                    if self.data_processor.has_background():
                        bg_data = self.data_processor.get_background()
                        
                        # Обновляем виджет с фоновыми данными
                        self.data_processor.set_data(bg_data)
                        x_proj, y_proj = self.data_processor.calculate_projections()
                        centroid_x, centroid_y = self.data_processor.calculate_centroids()
                        rms_x, rms_y = self.data_processor.calculate_rms()
                        
                        self.background_widget.update_data(
                            bg_data, x_mm, y_mm, centroid_x, centroid_y, 
                            rms_x, rms_y, x_proj, y_proj
                        )
                        
                        # Обновляем виджет с разностными данными
                        self.data_processor.set_data(data)
                        diff_data = self.data_processor.calculate_difference()
                        
                        self.data_processor.set_data(diff_data)
                        x_proj, y_proj = self.data_processor.calculate_projections()
                        centroid_x, centroid_y = self.data_processor.calculate_centroids()
                        rms_x, rms_y = self.data_processor.calculate_rms()
                        
                        self.diff_widget.update_data(
                            diff_data, x_mm, y_mm, centroid_x, centroid_y, 
                            rms_x, rms_y, x_proj, y_proj
                        )
                        
                        # Возвращаем исходные данные в обработчик
                        self.data_processor.set_data(data)
                    
                    # Обновляем состояние кнопки сохранения
                    self.save_btn.setEnabled(True)
                    
                    QMessageBox.information(self, "Загрузка", "Данные успешно загружены")
                    logging.info("Данные успешно загружены")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке файла: {str(e)}")
            logging.error(f"Ошибка при загрузке файла: {str(e)}")
            
    def save_data(self):
        """Сохранение данных в файл"""
        try:
            if not self.data_processor.has_data():
                QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения")
                return
                
            # Открываем диалог сохранения файла
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить данные", "", 
                "Файлы NumPy (*.npy);;Файлы MATLAB (*.mat)"
            )
            
            if file_path:
                # Добавляем расширение, если не указано
                if not file_path.lower().endswith(('.npy', '.mat')):
                    file_path += '.npy'
                    
                logging.info(f"Сохранение данных в файл: {file_path}")
                
                # Сохраняем данные в файл
                if file_path.lower().endswith('.npy'):
                    self.data_processor.save_to_npy(file_path)
                elif file_path.lower().endswith('.mat'):
                    self.data_processor.save_to_mat(file_path)
                    
                # Если есть фоновые данные, сохраняем их также
                if self.data_processor.has_background():
                    # Формируем имя для фонового файла
                    base_name, ext = os.path.splitext(file_path)
                    bg_file = f"{base_name}_bg{ext}"
                    
                    # Сохраняем фоновые данные
                    if file_path.lower().endswith('.npy'):
                        self.data_processor.save_background_to_npy(bg_file)
                    elif file_path.lower().endswith('.mat'):
                        self.data_processor.save_background_to_mat(bg_file)
                        
                    logging.info(f"Фоновые данные сохранены в файл: {bg_file}")
                
                QMessageBox.information(self, "Сохранение", "Данные успешно сохранены")
                logging.info("Данные успешно сохранены")
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении файла: {str(e)}")
            logging.error(f"Ошибка при сохранении файла: {str(e)}")
            
    def closeEvent(self, event):
        """Обработка закрытия окна"""
        try:
            # Если камера подключена, отключаем её
            if self.camera_connected:
                if self.is_capturing:
                    self.toggle_capture()
                self.camera_manager.close_camera()
                
            logging.info("Приложение закрывается")
            event.accept()
            
        except Exception as e:
            logging.error(f"Ошибка при закрытии приложения: {str(e)}")
            event.accept()  # Всё равно закрываем приложение 