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
            self.difference = np.maximum(data - background, 0)
        else:
            self.difference = data
            
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        
    def load_npy_file(self, filename):
        """Загрузка данных из файла .npy"""
        try:
            data = np.load(filename, allow_pickle=True).item()
            
            self.data = data['raw']
            self.background = data['background']
            self.difference = data['difference']
            
            # Извлекаем размеры пикселей
            self.pixel_size_x, self.pixel_size_y = data['pixel_size']
            
            return True
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла .npy: {str(e)}")
            return False
            
    def load_mat_file(self, filename, bg_filename=None):
        """Загрузка данных из файла .mat"""
        try:
            # Загрузка основных данных
            mat_data = io.loadmat(filename)
            
            # Ищем переменную с данными, обычно это первая переменная, не начинающаяся с '__'
            data_var_name = next(var for var in mat_data.keys() if not var.startswith('__'))
            self.data = mat_data[data_var_name]
            
            # Определяем размер пикселя по размерности матрицы
            height, width = self.data.shape
            
            if width == 1032 and height == 776:
                # GUN_YAG1
                self.pixel_size_x = 0.02840909
                self.pixel_size_y = 0.02840909
            elif width == 1624 and height == 1224:
                # GUN_YAG2
                self.pixel_size_x = 0.01875468
                self.pixel_size_y = 0.01875468
            else:
                logging.warning(f"Неизвестный размер изображения: {width}x{height}")
                self.pixel_size_x = 1.0
                self.pixel_size_y = 1.0
            
            # Если предоставлен файл фона, загружаем его
            if bg_filename:
                bg_mat_data = io.loadmat(bg_filename)
                bg_var_name = next(var for var in bg_mat_data.keys() if not var.startswith('__'))
                self.background = bg_mat_data[bg_var_name]
                
                # Проверяем соответствие размеров
                if self.data.shape == self.background.shape:
                    self.difference = np.maximum(self.data - self.background, 0)
                else:
                    logging.warning("Размеры данных и фона не совпадают")
                    self.background = None
                    self.difference = self.data
            else:
                self.background = None
                self.difference = self.data
                
            return True
        except Exception as e:
            logging.error(f"Ошибка при загрузке файла .mat: {str(e)}")
            return False
    
    def find_matching_bg_file(self, filename):
        """Поиск соответствующего файла фона для данного файла данных"""
        try:
            import os
            
            # Получаем директорию и базовое имя файла
            dir_path = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            
            # Убираем расширение
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Ищем подходящий файл фона
            # Например, 039_4_14_57_12.mat -> 039_4_bg_14_57_12.mat
            parts = name_without_ext.split('_')
            
            if len(parts) >= 2:
                # Вставляем "bg" после первых двух сегментов
                bg_parts = parts[:2] + ['bg'] + parts[2:]
                bg_name = '_'.join(bg_parts) + os.path.splitext(base_name)[1]
                bg_path = os.path.join(dir_path, bg_name)
                
                if os.path.exists(bg_path):
                    return bg_path
            
            # Если не нашли точное соответствие, ищем любой bg-файл с похожим именем
            for file in os.listdir(dir_path):
                if 'bg' in file and file.endswith(os.path.splitext(base_name)[1]):
                    return os.path.join(dir_path, file)
            
            return None
        except Exception as e:
            logging.error(f"Ошибка при поиске файла фона: {str(e)}")
            return None
    
    def calculate_projections(self, data=None):
        """Расчет проекций на оси X и Y"""
        if data is None:
            data = self.data
            
        if data is None:
            return None, None
        
        try:
            # Суммируем по осям для получения проекций
            proj_x = np.sum(data, axis=0)
            proj_y = np.sum(data, axis=1)
            
            # Нормализуем проекции
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
    
    def get_coordinate_grid(self, data=None):
        """Получение сетки координат в мм для отображения графиков"""
        if data is None:
            data = self.data
            
        if data is None:
            return None, None, None, None
            
        try:
            height, width = data.shape
            
            # Создаем сетку координат в пикселях
            x_pixels = np.linspace(-width/2, width/2, width)
            y_pixels = np.linspace(-height/2, height/2, height)
            
            # Преобразуем в мм, если установлены коэффициенты
            x_mm = x_pixels * self.pixel_size_x if self.pixel_size_x else x_pixels
            y_mm = y_pixels * self.pixel_size_y if self.pixel_size_y else y_pixels
            
            return x_pixels, y_pixels, x_mm, y_mm
        except Exception as e:
            logging.error(f"Ошибка при создании сетки координат: {str(e)}")
            return None, None, None, None
    
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