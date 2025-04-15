from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QMessageBox, 
                           QFileDialog, QVBoxLayout, QWidget)
from PyQt5.QtCore import QSettings

from src.ui.camera_tab import CameraTab
from src.ui.background_tab import BackgroundTab
from src.ui.difference_tab import DifferenceTab

from src.camera.camera_manager import CameraManager
from src.camera.image_reader import ImageReader
from src.analysis.image_analyzer import ImageAnalyzer
from src.visualizer.plot_manager import PlotManager
from src.data.exporter import DataExporter
from src.data.importer import DataImporter

class MainWindow(QMainWindow):
    """
    Главное окно приложения Beam Profile Analyzer
    """
    def __init__(self):
        super().__init__()
        
        # Инициализация менеджеров и анализаторов
        self.camera_manager = CameraManager()
        self.image_reader = ImageReader(self.camera_manager)
        self.image_analyzer = ImageAnalyzer()
        self.plot_manager = PlotManager()
        
        # Настройки приложения
        self.settings = QSettings("BeamProfileAnalyzer", "BeamProfileAnalyzer")
        
        # Буферы данных для разных режимов
        self.camera_data = {
            "current_frame": None,
            "background": None,
            "difference": None,
            "centroid": (0, 0),
            "rms": (0, 0)
        }
        
        self.file_data = {
            "current_frame": None,
            "background": None,
            "difference": None,
            "centroid": (0, 0),
            "rms": (0, 0),
            "filepath": None
        }
        
        # Режим работы (camera или file)
        self.current_mode = "camera"
        
        # Настройка UI
        self.setWindowTitle("Анализатор профиля пучка")
        self.resize(1200, 800)
        
        # Создание вкладок
        self.tabs = QTabWidget()
        self.camera_tab = CameraTab(self)
        self.background_tab = BackgroundTab(self)
        self.difference_tab = DifferenceTab(self)
        
        # Добавление вкладок в контейнер
        self.tabs.addTab(self.camera_tab, "Основная съемка")
        self.tabs.addTab(self.background_tab, "Фон")
        self.tabs.addTab(self.difference_tab, "Разность")
        
        # Установка центрального виджета
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Соединение сигналов
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
    def on_tab_changed(self, index):
        """
        Обработчик смены вкладки
        
        Args:
            index: Индекс новой вкладки
        """
        # Обновляем только активную вкладку
        active_tab = self.tabs.widget(index)
        active_tab.update_tab()
        
        # Останавливаем таймеры обновления в неактивных вкладках
        for i in range(self.tabs.count()):
            if i != index:
                tab = self.tabs.widget(i)
                if hasattr(tab, 'update_timer'):
                    tab.update_timer.stop()
        
        # Запускаем таймер обновления только активной вкладки
        if hasattr(active_tab, 'update_timer'):
            active_tab.update_timer.start(1000)  # Увеличиваем интервал до 1 секунды
        
    def switch_mode(self, mode):
        """
        Переключает режим работы приложения
        
        Args:
            mode: Режим работы ("camera" или "file")
        """
        if mode not in ["camera", "file"]:
            return
            
        self.current_mode = mode
        
        # Обновляем все вкладки
        self.camera_tab.update_tab()
        self.background_tab.update_tab()
        self.difference_tab.update_tab()
        
    def connect_to_camera(self, camera_name):
        """
        Подключается к выбранной камере
        
        Args:
            camera_name: Имя камеры
            
        Returns:
            bool: True если подключение успешно, иначе False
        """
        # Отключаемся от текущей камеры если есть
        self.disconnect_camera()
        
        # Подключаемся к выбранной камере
        result = self.camera_manager.connect_to_camera(camera_name)
        
        if result:
            # Запускаем предварительный просмотр
            self.image_reader.start_preview()
            
            # Переключаемся в режим работы с камерой
            self.switch_mode("camera")
            
        return result
        
    def disconnect_camera(self):
        """
        Отключается от камеры
        
        Returns:
            bool: True если отключение успешно, иначе False
        """
        # Останавливаем предварительный просмотр
        self.image_reader.stop_preview()
        
        # Отключаемся от камеры
        return self.camera_manager.disconnect_camera()
        
    def start_background_collection(self, frames_count=40):
        """
        Запускает сбор кадров для фона
        
        Args:
            frames_count: Количество кадров для сбора
            
        Returns:
            bool: True если запуск успешен, иначе False
        """
        return self.image_reader.start_background_collection(frames_count)
        
    def stop_background_collection(self):
        """
        Останавливает сбор кадров для фона
        
        Returns:
            bool: True если остановка успешна, иначе False
        """
        return self.image_reader.stop_background_collection()
        
    def get_current_data(self):
        """
        Возвращает текущие данные в зависимости от режима работы
        
        Returns:
            dict: Словарь с текущими данными
        """
        # Обновляем данные если в режиме камеры
        if self.current_mode == "camera":
            # Получаем текущий кадр
            self.camera_data["current_frame"] = self.image_reader.get_current_frame()
            self.camera_data["background"] = self.image_reader.get_background()
            self.camera_data["difference"] = self.image_reader.get_difference()
            
            # Если есть текущий кадр, вычисляем центроид и RMS
            if self.camera_data["current_frame"] is not None:
                camera_info = self.camera_manager.get_camera_info()
                if camera_info:
                    pixel_size_x = camera_info["pixel_size_x"]
                    pixel_size_y = camera_info["pixel_size_y"]
                    
                    self.camera_data["centroid"] = self.image_analyzer.calculate_centroid(
                        self.camera_data["current_frame"],
                        pixel_size_x,
                        pixel_size_y
                    )
                    
                    self.camera_data["rms"] = self.image_analyzer.calculate_rms(
                        self.camera_data["current_frame"],
                        pixel_size_x,
                        pixel_size_y
                    )
            
            return self.camera_data
        else:
            return self.file_data
            
    def get_camera_info(self):
        """
        Возвращает информацию о текущей камере
        
        Returns:
            dict: Информация о камере или None если камера не выбрана
        """
        if self.current_mode == "camera":
            return self.camera_manager.get_camera_info()
        else:
            # В режиме чтения файла берем информацию из данных файла
            if self.file_data["current_frame"] is not None:
                return {
                    "resolution": self.file_data.get("resolution", (0, 0)),
                    "pixel_size_x": self.file_data.get("pixel_size_x", 0),
                    "pixel_size_y": self.file_data.get("pixel_size_y", 0),
                    "camera_name": self.file_data.get("camera_name", "Неизвестная камера")
                }
            return None
            
    def export_data(self, filepath, export_format="npy"):
        """
        Экспортирует данные в указанном формате
        
        Args:
            filepath: Путь для сохранения файла/директории
            export_format: Формат экспорта ("npy", "csv", "png")
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        # Получаем текущие данные
        data = self.get_current_data()
        camera_info = self.get_camera_info()
        
        # Если нет данных или информации о камере
        if data["current_frame"] is None or camera_info is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
            return False
            
        # Формируем словарь для экспорта
        export_data = {
            "shot": data["current_frame"],
            "background": data["background"],
            "difference": data["difference"],
            "raw_shot": self.image_reader.get_raw_current_frame() if self.current_mode == "camera" 
                       else data.get("raw_shot"),
            "raw_background": self.image_reader.get_raw_background() if self.current_mode == "camera"
                            else data.get("raw_background"),
            "resolution": camera_info["resolution"],
            "pixel_size_x": camera_info["pixel_size_x"],
            "pixel_size_y": camera_info["pixel_size_y"],
            "camera_name": camera_info.get("camera_name", "Неизвестная камера")
        }
        
        # Экспортируем данные
        result = DataExporter.export_data(filepath, export_data, export_format, self.plot_manager)
        
        if result:
            QMessageBox.information(self, "Информация", f"Данные успешно экспортированы в {filepath}")
        else:
            QMessageBox.warning(self, "Предупреждение", "Ошибка при экспорте данных")
            
        return result
        
    def import_data(self, filepath):
        """
        Импортирует данные из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            bool: True если импорт успешен, иначе False
        """
        try:
            # Останавливаем таймеры обновления во всех вкладках
            if hasattr(self, 'camera_tab') and self.camera_tab:
                self.camera_tab.update_timer.stop()
            if hasattr(self, 'background_tab') and self.background_tab:
                self.background_tab.update_timer.stop()
            if hasattr(self, 'difference_tab') and self.difference_tab:
                self.difference_tab.update_timer.stop()
            
            # Импортируем данные
            data = DataImporter.import_data(filepath)
            
            if data is None:
                return False
            
            # Переключаемся в режим чтения файла
            self.current_mode = "file"
            
            # Инициализируем file_data с базовой структурой во избежание KeyError
            self.file_data = {
                "shot": None,
                "background": None,
                "difference": None,
                "current_frame": None,
                "raw_shot": None,
                "raw_background": None,
                "resolution": (0, 0),
                "pixel_size_x": 1.0,
                "pixel_size_y": 1.0,
                "camera_name": "Неизвестная камера",
                "centroid": (0, 0),
                "rms": (0, 0),
                "filepath": filepath
            }
            
            # Обновляем данные из импортированного словаря
            self.file_data.update(data)
            
            # Вычисляем центроид и RMS, если они не были вычислены при импорте
            if "shot" in data and data["shot"] is not None and "centroid" not in data:
                pixel_size_x = self.file_data.get("pixel_size_x", 1)
                pixel_size_y = self.file_data.get("pixel_size_y", 1)
                
                self.file_data["centroid"] = self.image_analyzer.calculate_centroid(
                    self.file_data["shot"],
                    pixel_size_x,
                    pixel_size_y
                )
                
                self.file_data["rms"] = self.image_analyzer.calculate_rms(
                    self.file_data["shot"],
                    pixel_size_x,
                    pixel_size_y
                )
            
            # Обновляем все вкладки
            self.camera_tab.update_tab()
            self.background_tab.update_tab()
            self.difference_tab.update_tab()
            
            # Запускаем таймеры обновления
            if hasattr(self, 'camera_tab') and self.camera_tab:
                self.camera_tab.update_timer.start(500)
            if hasattr(self, 'background_tab') and self.background_tab:
                self.background_tab.update_timer.start(500)
            if hasattr(self, 'difference_tab') and self.difference_tab:
                self.difference_tab.update_timer.start(500)
            
            return True
        
        except Exception as e:
            # В случае ошибки также восстанавливаем таймеры
            if hasattr(self, 'camera_tab') and self.camera_tab:
                self.camera_tab.update_timer.start(500)
            if hasattr(self, 'background_tab') and self.background_tab:
                self.background_tab.update_timer.start(500)
            if hasattr(self, 'difference_tab') and self.difference_tab:
                self.difference_tab.update_timer.start(500)
            
            return False
        
    def open_file_dialog(self):
        """
        Открывает диалог выбора файла для импорта
        
        Returns:
            bool: True если файл выбран и импортирован, иначе False
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть файл",
            "",
            "Файлы данных (*.npy *.mat);;Все файлы (*)"
        )
        
        if filepath:
            return self.import_data(filepath)
            
        return False
        
    def save_file_dialog(self, default_format="npy"):
        """
        Открывает диалог сохранения файла для экспорта
        
        Args:
            default_format: Формат экспорта по умолчанию ("npy", "csv", "png")
            
        Returns:
            bool: True если данные успешно экспортированы, иначе False
        """
        # Форматы и фильтры
        formats = {
            "npy": "Файлы NumPy (*.npy)",
            "csv": "CSV файлы (*.csv)",
            "png": "PNG изображения (папка)"
        }
        
        # Строка фильтров
        filter_str = ";;".join(formats.values())
        
        # Выбираем формат по умолчанию
        default_filter = formats.get(default_format, formats["npy"])
        
        # Открываем диалог сохранения
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить данные",
            "",
            filter_str,
            default_filter
        )
        
        if not filepath:
            return False
            
        # Определяем выбранный формат
        export_format = None
        for fmt, filter_name in formats.items():
            if filter_name == selected_filter:
                export_format = fmt
                break
                
        if export_format is None:
            export_format = default_format
            
        # Добавляем расширение если его нет
        if export_format != "png" and not filepath.endswith(f".{export_format}"):
            filepath += f".{export_format}"
            
        # Экспортируем данные
        return self.export_data(filepath, export_format)
        
    def closeEvent(self, event):
        """
        Обработчик закрытия окна
        
        Args:
            event: Объект события
        """
        # Отключаемся от камеры
        self.disconnect_camera()
        
        # Сохраняем настройки
        self.settings.sync()
        
        event.accept()
