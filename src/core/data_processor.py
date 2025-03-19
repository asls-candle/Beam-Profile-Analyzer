import numpy as np
import logging
from scipy import io

class DataProcessor:
    """Класс для обработки данных пучка и расчета параметров"""
    
    def __init__(self):
        self.data = None
        self.background = None
        self.difference = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        
    def set_data(self, data, background=None, pixel_size_x=None, pixel_size_y=None):
        """Установка данных для обработки"""
        self.data = data
        self.background = background
        
        if background is not None and data.shape == background.shape:
            # Вычисляем разницу и обрезаем отрицательные значения
            self.difference = np.maximum(data.astype(np.float32) - background.astype(np.float32), 0).astype(data.dtype)
        else:
            self.difference = data
            
        # Устанавливаем размеры пикселей
        if pixel_size_x is not None:
            self.pixel_size_x = pixel_size_x
        elif self.pixel_size_x is None:
            self.pixel_size_x = 0.0068  # Размер пикселя Flea2 по умолчанию
            
        if pixel_size_y is not None:
            self.pixel_size_y = pixel_size_y
        elif self.pixel_size_y is None:
            self.pixel_size_y = 0.0068  # Размер пикселя Flea2 по умолчанию
            
    def has_data(self):
        """Проверка наличия данных"""
        return self.data is not None
        
    def has_background(self):
        """Проверка наличия фоновых данных"""
        return self.background is not None
        
    def get_background(self):
        """Получение фоновых данных"""
        return self.background
        
    def set_background(self, background):
        """Установка фоновых данных"""
        self.background = background
        
        # Пересчитываем разницу, если есть данные и фон
        if self.data is not None and background.shape == self.data.shape:
            self.difference = np.maximum(self.data - background, 0)
        
    def calculate_difference(self):
        """Расчет разницы между основными данными и фоном"""
        if not self.has_data() or not self.has_background():
            logging.warning("Нет данных или фона для расчета разности")
            return None
            
        try:
            # Проверяем, что данные и фон имеют одинаковый размер
            if self.data.shape != self.background.shape:
                logging.error(f"Разные размеры данных ({self.data.shape}) и фона ({self.background.shape})")
                return None
                
            # Преобразуем данные в float32 для безопасного вычитания
            data_float = self.data.astype(np.float32)
            bg_float = self.background.astype(np.float32)
            
            # Вычитаем фон, отрицательные значения заменяем на ноль
            difference = np.maximum(data_float - bg_float, 0)
            
            # Преобразуем обратно в исходный тип данных
            self.difference = difference.astype(self.data.dtype)
            
            logging.info(f"Вычислена разность, диапазон значений: [{np.min(self.difference)}-{np.max(self.difference)}]")
            
            return self.difference
        except Exception as e:
            logging.error(f"Ошибка при расчете разницы данных и фона: {str(e)}")
            logging.exception(e)
            return None
        
    def load_from_npy(self, filename):
        """Загрузка данных из файла .npy"""
        try:
            data = np.load(filename, allow_pickle=True).item()
            
            self.data = data['raw']
            self.background = data.get('background', None)
            self.difference = data.get('difference', None)
            
            # Если разницы нет, но есть фон, вычисляем её
            if self.difference is None and self.background is not None:
                self.calculate_difference()
            
            # Извлекаем размеры пикселей
            if 'pixel_size' in data:
                self.pixel_size_x, self.pixel_size_y = data['pixel_size']
            else:
                self.pixel_size_x = self.pixel_size_y = 0.0068  # Размер пикселя Flea2 по умолчанию
            
            # Создаем тестовые данные, если файл не загружен корректно или данные пустые
            if self.data is None or self.data.size == 0:
                logging.warning("Не удалось загрузить данные из файла, генерируем тестовые данные")
                self.generate_test_data()
            
            return self.data
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла .npy: {str(e)}")
            logging.info("Генерируем тестовые данные")
            return self.generate_test_data()
            
    def load_from_mat(self, filename):
        """Загрузка данных из файла формата MATLAB (.mat)"""
        import scipy.io as io
        
        try:
            # Загрузка основных данных
            mat_data = io.loadmat(filename)
            
            # Ищем переменную с данными, исключая служебные переменные
            data_var_name = None
            for var in mat_data.keys():
                if not var.startswith('__'):
                    # Если это массив нужного размера, используем его
                    if isinstance(mat_data[var], np.ndarray) and len(mat_data[var].shape) == 2:
                        data_var_name = var
                        break
            
            if data_var_name is None:
                logging.error(f"Не удалось найти данные изображения в .mat файле: {filename}")
                return None
                
            logging.info(f"Найдена переменная с данными: {data_var_name}")
            self.data = mat_data[data_var_name]
            
            # Определяем размер пикселя по размерности матрицы
            height, width = self.data.shape
            
            if width == 1032 and height == 776:
                # GUN_YAG1
                self.pixel_size_x = 0.02840909
                self.pixel_size_y = 0.02840909
                logging.info("Определена камера: GUN_YAG1")
            elif width == 1624 and height == 1224:
                # GUN_YAG2
                self.pixel_size_x = 0.01875468
                self.pixel_size_y = 0.01875468
                logging.info("Определена камера: GUN_YAG2")
            else:
                logging.warning(f"Неизвестный размер изображения: {width}x{height}, будут использованы стандартные размеры пикселей")
                self.pixel_size_x = 0.0068  # Размер пикселя Flea2 по умолчанию
                self.pixel_size_y = 0.0068
            
            logging.info(f"Загружены данные размером {width}x{height}, диапазон [{np.min(self.data)}-{np.max(self.data)}]")
            return self.data
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла .mat: {str(e)}")
            logging.exception(e)
            return None
    
    def find_background_file(self, filename):
        """Поиск соответствующего файла фона для данного файла данных"""
        try:
            import os
            import glob
            
            # Получаем директорию и базовое имя файла
            dir_path = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            
            # Убираем расширение
            name_without_ext, ext = os.path.splitext(base_name)
            
            # Список возможных вариантов суффиксов для фонового файла
            bg_suffixes = ["_bg", "_background", "_фон", "-bg", "-background"]
            
            # Ищем файл с различными суффиксами
            for suffix in bg_suffixes:
                bg_name = f"{name_without_ext}{suffix}{ext}"
                bg_path = os.path.join(dir_path, bg_name)
                if os.path.exists(bg_path):
                    logging.info(f"Найден файл фона с суффиксом {suffix}: {bg_name}")
                    return bg_path
            
            # Если не нашли по суффиксу, пробуем другие форматы именования
            parts = name_without_ext.split('_')
            
            # Формат вида "префикс_bg_суффикс"
            if len(parts) >= 2:
                for i in range(1, len(parts)):
                    # Пробуем вставить "bg" в разные позиции имени
                    bg_parts = parts[:i] + ['bg'] + parts[i:]
                    bg_name = '_'.join(bg_parts) + ext
                    bg_path = os.path.join(dir_path, bg_name)
                    
                    if os.path.exists(bg_path):
                        logging.info(f"Найден файл фона с bg внутри имени: {bg_name}")
                        return bg_path
            
            # Если не нашли точное соответствие, ищем любой bg-файл с похожим именем
            # в той же директории с тем же расширением
            similar_files = []
            pattern = os.path.join(dir_path, f"*bg*{ext}")
            for file in glob.glob(pattern):
                if os.path.isfile(file):
                    similar_files.append(file)
            
            if similar_files:
                # Берем первый найденный файл
                bg_file = similar_files[0]
                logging.info(f"Найден похожий файл фона: {os.path.basename(bg_file)}")
                return bg_file
            
            logging.warning(f"Не удалось найти файл фона для {base_name}")
            return None
        except Exception as e:
            logging.error(f"Ошибка при поиске файла фона: {str(e)}")
            logging.exception(e)
            return None
    
    def calculate_projections(self, data=None):
        """Расчет проекций на оси X и Y"""
        if data is None:
            data = self.data
            
        if data is None:
            return None, None
        
        try:
            # Суммируем по осям для получения проекций
            # Для получения проекции по X суммируем все строки (axis=0)
            # Для получения проекции по Y суммируем все столбцы (axis=1)
            proj_x = np.sum(data, axis=0)  # Суммирование по оси Y
            proj_y = np.sum(data, axis=1)  # Суммирование по оси X
            
            # Нормализуем проекции для отображения на графике
            # Нормализация не изменяет форму распределения, только его масштаб
            if np.max(proj_x) > 0:
                proj_x = proj_x / np.max(proj_x)
            
            if np.max(proj_y) > 0:
                proj_y = proj_y / np.max(proj_y)
                
            return proj_x, proj_y
        except Exception as e:
            logging.error(f"Ошибка при расчете проекций: {str(e)}")
            return None, None
    
    def calculate_centroids(self, data=None):
        """Расчет центроидов по осям X и Y"""
        if data is None:
            data = self.data
            
        if data is None:
            return None, None
            
        try:
            height, width = data.shape
            
            # Создаем сетку координат, центрированную относительно центра изображения
            x = np.linspace(-width/2, width/2, width)
            y = np.linspace(-height/2, height/2, height)
            
            # Получаем проекции
            proj_x, proj_y = self.calculate_projections(data)
            
            # Вычисляем центроиды
            centroid_x = np.sum(x * proj_x) / np.sum(proj_x) if np.sum(proj_x) > 0 else 0
            centroid_y = np.sum(y * proj_y) / np.sum(proj_y) if np.sum(proj_y) > 0 else 0
            
            # Применяем коэффициенты для перевода в мм, если они установлены
            if self.pixel_size_x:
                centroid_x *= self.pixel_size_x
            
            if self.pixel_size_y:
                centroid_y *= self.pixel_size_y
                
            return centroid_x, centroid_y
        except Exception as e:
            logging.error(f"Ошибка при расчете центроидов: {str(e)}")
            return None, None
    
    def calculate_rms(self, data=None):
        """Расчет RMS по осям X и Y"""
        if data is None:
            data = self.data
            
        if data is None:
            return None, None
            
        try:
            height, width = data.shape
            
            # Создаем сетку координат, центрированную относительно центра изображения
            x = np.linspace(-width/2, width/2, width)
            y = np.linspace(-height/2, height/2, height)
            
            # Получаем проекции
            proj_x, proj_y = self.calculate_projections(data)
            
            # Вычисляем центроиды (в пикселях)
            centroid_x = np.sum(x * proj_x) / np.sum(proj_x) if np.sum(proj_x) > 0 else 0
            centroid_y = np.sum(y * proj_y) / np.sum(proj_y) if np.sum(proj_y) > 0 else 0
            
            # Вычисляем RMS
            rms_x = np.sqrt(np.sum(proj_x * (x - centroid_x)**2) / np.sum(proj_x)) if np.sum(proj_x) > 0 else 0
            rms_y = np.sqrt(np.sum(proj_y * (y - centroid_y)**2) / np.sum(proj_y)) if np.sum(proj_y) > 0 else 0
            
            # Применяем коэффициенты для перевода в мм, если они установлены
            if self.pixel_size_x:
                rms_x *= self.pixel_size_x
                
            if self.pixel_size_y:
                rms_y *= self.pixel_size_y
                
            return rms_x, rms_y
        except Exception as e:
            logging.error(f"Ошибка при расчете RMS: {str(e)}")
            return None, None
    
    def get_coordinate_grids(self):
        """Получение сеток координат в мм для отображения на графиках"""
        if self.data is None:
            return None, None
            
        try:
            height, width = self.data.shape
            
            # Создаем сетки координат в пикселях, центрированные в центре изображения
            x_pixels = np.arange(width) - width/2
            y_pixels = np.arange(height) - height/2
            
            # Преобразуем в мм, используя размер пикселя
            x_mm = x_pixels * (self.pixel_size_x if self.pixel_size_x else 0.0068)
            y_mm = y_pixels * (self.pixel_size_y if self.pixel_size_y else 0.0068)
            
            return x_mm, y_mm
        except Exception as e:
            logging.error(f"Ошибка при создании сеток координат: {str(e)}")
            return None, None
            
    def save_to_npy(self, filename):
        """Сохранение данных в файл .npy"""
        if not self.has_data():
            logging.error("Нет данных для сохранения")
            return False
            
        try:
            data_dict = {
                'raw': self.data,
                'background': self.background,
                'difference': self.difference,
                'pixel_size': (self.pixel_size_x, self.pixel_size_y)
            }
            
            np.save(filename, data_dict)
            return True
        except Exception as e:
            logging.error(f"Ошибка при сохранении в файл .npy: {str(e)}")
            return False
            
    def save_to_mat(self, filename):
        """Сохранение данных в файл .mat"""
        if not self.has_data():
            logging.error("Нет данных для сохранения")
            return False
            
        try:
            data_dict = {
                'raw_data': self.data,
                'pixel_size_x': self.pixel_size_x,
                'pixel_size_y': self.pixel_size_y
            }
            
            if self.has_background():
                data_dict['background'] = self.background
                
            if self.difference is not None:
                data_dict['difference'] = self.difference
                
            io.savemat(filename, data_dict)
            return True
        except Exception as e:
            logging.error(f"Ошибка при сохранении в файл .mat: {str(e)}")
            return False
            
    def save_background_to_npy(self, filename):
        """Сохранение фоновых данных в файл .npy"""
        if not self.has_background():
            logging.error("Нет фоновых данных для сохранения")
            return False
            
        try:
            data_dict = {
                'raw': self.background,
                'pixel_size': (self.pixel_size_x, self.pixel_size_y)
            }
            
            np.save(filename, data_dict)
            return True
        except Exception as e:
            logging.error(f"Ошибка при сохранении фона в файл .npy: {str(e)}")
            return False
            
    def save_background_to_mat(self, filename):
        """Сохранение фоновых данных в файл .mat"""
        if not self.has_background():
            logging.error("Нет фоновых данных для сохранения")
            return False
            
        try:
            data_dict = {
                'raw_data': self.background,
                'pixel_size_x': self.pixel_size_x,
                'pixel_size_y': self.pixel_size_y
            }
            
            io.savemat(filename, data_dict)
            return True
        except Exception as e:
            logging.error(f"Ошибка при сохранении фона в файл .mat: {str(e)}")
            return False

    def get_camera_info(self):
        """Определение информации о камере на основе размеров данных"""
        if self.data is None:
            return None
            
        height, width = self.data.shape
        
        if width == 1032 and height == 776:
            return {
                "name": "GUN_YAG1",
                "resolution": (width, height),
                "pixel_size": (self.pixel_size_x, self.pixel_size_y)
            }
        elif width == 1624 and height == 1224:
            return {
                "name": "GUN_YAG2",
                "resolution": (width, height),
                "pixel_size": (self.pixel_size_x, self.pixel_size_y)
            }
        else:
            return {
                "name": "Unknown",
                "resolution": (width, height),
                "pixel_size": (self.pixel_size_x, self.pixel_size_y)
            }

    def generate_test_data(self):
        """Генерация тестовых данных для демонстрации"""
        # Увеличиваем размер матрицы для лучшего отображения
        height, width = 400, 600
        
        # Создаем сетки координат
        x = np.linspace(-15, 15, width)
        y = np.linspace(-10, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Создаем основной гауссов пучок с более выраженным сигналом
        sigma_x = 1.5  # Размер пучка по Х
        sigma_y = 1.2  # Размер пучка по Y
        x0, y0 = 1.0, 0.0  # Центр пучка
        
        # Формула гауссова пучка (более интенсивный)
        Z1 = np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2)))
        
        # Добавляем второй яркий пучок
        x1, y1 = -4.0, 2.0  # Центр второго пучка (дальше от первого)
        sigma_x1, sigma_y1 = 0.8, 0.8
        Z2 = 0.9 * np.exp(-((X - x1)**2 / (2 * sigma_x1**2) + (Y - y1)**2 / (2 * sigma_y1**2)))
        
        # Добавляем третий пучок меньшей интенсивности
        x2, y2 = 3.0, -3.0  # Центр третьего пучка
        sigma_x2, sigma_y2 = 0.6, 0.6
        Z3 = 0.5 * np.exp(-((X - x2)**2 / (2 * sigma_x2**2) + (Y - y2)**2 / (2 * sigma_y2**2)))
        
        # Комбинируем все пучки
        Z = Z1 + Z2 + Z3
        
        # Усиливаем контраст и добавляем меньше шума
        Z = Z / np.max(Z)  # Нормализуем к максимуму
        
        # Добавляем минимальный фоновый шум
        noise = np.random.normal(0, 0.002, Z.shape)
        Z = Z + noise
        Z = np.clip(Z, 0, 1)  # Обрезаем значения до диапазона [0, 1]
        
        # Увеличиваем яркость с помощью гамма-коррекции
        Z = Z ** 0.4  # Сильная гамма-коррекция для увеличения яркости
        
        # Масштабируем в диапазон [0, 255] для типа uint8
        Z = (Z * 255).astype(np.uint8)
        
        # Устанавливаем сгенерированные данные
        self.set_data(Z, pixel_size_x=0.05, pixel_size_y=0.05)
        
        # Генерируем фоновые данные
        background = np.random.randint(0, 15, Z.shape, dtype=np.uint8)
        self.set_background(background)
        
        logging.info(f"Сгенерированы тестовые данные размером {Z.shape}, диапазон значений: [{np.min(Z)}-{np.max(Z)}]")
        
        return Z 