import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

from src.ui.constants import GAUSSIAN_FILTER_SIGMA_THRESHOLD

class ImageAnalyzer:
    """
    Класс для анализа изображений с профилем лазерного пучка.
    
    Этот класс предоставляет инструменты для всестороннего анализа профиля пучка, 
    включая расчет геометрических характеристик, статистических параметров 
    и аппроксимацию распределения интенсивности функцией Гаусса.
    
    Основные функциональные возможности:
    - Задание и применение области интереса (ROI) для фокусировки анализа
    - Расчет проекций интенсивности пучка на оси X и Y
    - Вычисление центроидов (центров масс) пучка по обеим осям
    - Расчет среднеквадратичных (RMS) размеров пучка
    - Аппроксимация профилей пучка гауссовой функцией с определением 
      параметров распределения (амплитуда, центр, ширина)
    - Гауссовская фильтрация для удаления артефактов за пределами 3σ
    
    Все пространственные расчеты могут быть выражены в физических единицах (мм)
    при условии, что предоставлены размеры пикселей по каждой из осей.
    
    Attributes:
        roi (tuple): Кортеж (x_min, y_min, x_max, y_max), определяющий текущую 
                    область интереса. None, если ROI не задан.
    """
    def __init__(self):
        self.roi = None  # Region of Interest
        
    def set_roi(self, x_min, y_min, x_max, y_max):
        """
        Устанавливает область интереса (ROI)
        
        Args:
            x_min, y_min: Координаты верхнего левого угла ROI
            x_max, y_max: Координаты нижнего правого угла ROI
        """
        self.roi = (x_min, y_min, x_max, y_max)
        
    def reset_roi(self):
        """Сбрасывает ROI"""
        self.roi = None
        
    def apply_roi(self, image):
        """
        Применяет ROI к изображению
        
        Args:
            image: Двумерный массив значений светимости
            
        Returns:
            Обрезанное изображение с применением ROI
        """
        if self.roi is None or image is None:
            return image
            
        x_min, y_min, x_max, y_max = self.roi
        return image[y_min:y_max, x_min:x_max]
    
    def apply_gaussian_filter(self, image, pixel_size_x, pixel_size_y, sigma_threshold=GAUSSIAN_FILTER_SIGMA_THRESHOLD):
        """
        Применяет гауссовскую фильтрацию для удаления артефактов за пределами заданного количества сигм.
        
        Метод использует итеративный подход: сначала находит главный максимум интенсивности,
        затем рассчитывает центроид и RMS размеры только в окрестности этого максимума,
        и создает эллиптическую маску для удаления отдаленных артефактов.
        
        Args:
            image: Двумерный массив значений светимости
            pixel_size_x: Размер пикселя по оси X (мм)
            pixel_size_y: Размер пикселя по оси Y (мм)
            sigma_threshold: Порог в единицах сигма (по умолчанию 2.0 для более строгой фильтрации)
            
        Returns:
            Отфильтрованное изображение с удаленными артефактами
        """
        if image is None or image.size == 0:
            return image
            
        # Применяем ROI если задан
        processed_image = self.apply_roi(image)
        if processed_image is None:
            return image
            
        y_size, x_size = processed_image.shape
        
        # Этап 1: Найдем позицию главного максимума интенсивности
        max_pos = np.unravel_index(np.argmax(processed_image), processed_image.shape)
        max_y, max_x = max_pos
        
        # Этап 2: Создаем предварительную маску вокруг максимума (более широкую область)
        # Используем примерную оценку размера пучка как 1/4 от размера изображения
        rough_size_x = x_size // 5
        rough_size_y = y_size // 5
        
        # Определяем границы предварительной области вокруг максимума
        x_start = max(0, max_x - rough_size_x)
        x_end = min(x_size, max_x + rough_size_x)
        y_start = max(0, max_y - rough_size_y)
        y_end = min(y_size, max_y + rough_size_y)
        
        # Создаем предварительную маску
        preliminary_mask = np.zeros((y_size, x_size), dtype=bool)
        preliminary_mask[y_start:y_end, x_start:x_end] = True
        
        # Применяем предварительную маску для расчета центроида основного пучка
        masked_image = processed_image * preliminary_mask
        
        # Этап 3: Рассчитываем центроид и RMS только для основного пучка
        x_proj = np.sum(masked_image, axis=0)
        y_proj = np.sum(masked_image, axis=1)
        
        # Избегаем деления на ноль
        x_sum = np.sum(x_proj)
        y_sum = np.sum(y_proj)
        
        if x_sum == 0 or y_sum == 0:
            return processed_image
        
        # Вычисляем взвешенное среднее положение (центроид) в пикселях для основного пучка
        centroid_x_px = np.sum(np.arange(x_size) * x_proj) / x_sum
        centroid_y_px = np.sum(np.arange(y_size) * y_proj) / y_sum
        
        # Рассчитываем RMS в пикселях для основного пучка
        rms_x_px = np.sqrt(np.sum(x_proj * (np.arange(x_size) - centroid_x_px)**2) / x_sum)
        rms_y_px = np.sqrt(np.sum(y_proj * (np.arange(y_size) - centroid_y_px)**2) / y_sum)
        
        # Этап 4: Создаем финальную эллиптическую маску на основе рассчитанных параметров
        x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
        
        # Рассчитываем эллиптическое расстояние от центроида основного пучка
        if rms_x_px > 0 and rms_y_px > 0:
            distance_x = (x_grid - centroid_x_px) / rms_x_px
            distance_y = (y_grid - centroid_y_px) / rms_y_px
            elliptical_distance = np.sqrt(distance_x**2 + distance_y**2)
            
            # Создаем финальную маску: True для пикселей в пределах sigma_threshold
            final_mask = elliptical_distance <= sigma_threshold
            
            # Применяем финальную маску к исходному изображению
            filtered_image = processed_image.copy()
            filtered_image[~final_mask] = 0
            
            # Отладочная информация
            removed_pixels = np.sum(~final_mask)
            total_pixels = final_mask.size
            print("Гауссовская фильтрация: удалено {} из {} пикселей ({:.1f}%) с порогом {}σ".format(
                removed_pixels, total_pixels, 100.0 * removed_pixels / total_pixels, sigma_threshold))
            
            return filtered_image
        else:
            # Если RMS равен нулю, возвращаем исходное изображение
            return processed_image
    
    def calculate_projections(self, image, pixel_size_x, pixel_size_y):
        """
        Рассчитывает проекции изображения на оси X и Y
        
        Args:
            image: Двумерный массив значений светимости
            pixel_size_x: Размер пикселя по оси X (мм)
            pixel_size_y: Размер пикселя по оси Y (мм)
            
        Returns:
            x_coords: Координаты по оси X (мм)
            x_proj: Проекция на ось X
            y_coords: Координаты по оси Y (мм)
            y_proj: Проекция на ось Y
        """
        if image is None or image.size == 0:
            return None, None, None, None
            
        # Применяем ROI если задан
        processed_image = self.apply_roi(image)
        
        # Создаем координатную сетку (в пикселях)
        y_size, x_size = processed_image.shape
        x_grid = np.arange(x_size)
        y_grid = np.arange(y_size)
        
        # Проекция на оси X и Y (сумма интенсивности по соответствующим осям)
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)
        
        # Нормализация проекций относительно максимального значения
        if np.max(x_proj) > 0:
            x_proj = x_proj / np.max(x_proj)
        if np.max(y_proj) > 0:
            y_proj = y_proj / np.max(y_proj)
        
        # Преобразование в милиметры
        x_coords = x_grid * pixel_size_x
        y_coords = y_grid * pixel_size_y
        
        return x_coords, x_proj, y_coords, y_proj
    
    def calculate_centroid(self, image, pixel_size_x, pixel_size_y):
        """
        Рассчитывает координаты центроида изображения в центрированной системе координат
        
        Args:
            image: Двумерный массив значений светимости
            pixel_size_x: Размер пикселя по оси X (мм)
            pixel_size_y: Размер пикселя по оси Y (мм)
            
        Returns:
            (centroid_x, centroid_y): Координаты центроида (мм) в центрированной системе координат
        """
        if image is None or image.size == 0:
            return 0, 0
            
        # Применяем ROI если задан
        processed_image = self.apply_roi(image)
        
        # Создаем координатную сетку (в пикселях)
        y_size, x_size = processed_image.shape
        x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
        
        # Вычисляем суммы интенсивности по осям
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)
        
        # Избегаем деления на ноль
        x_sum = np.sum(x_proj)
        y_sum = np.sum(y_proj)
        
        if x_sum == 0 or y_sum == 0:
            return 0, 0
        
        # Вычисляем взвешенное среднее положение (центроид) в пикселях
        centroid_x_px = np.sum(np.arange(x_size) * x_proj) / x_sum
        centroid_y_px = np.sum(np.arange(y_size) * y_proj) / y_sum
        
        # Преобразование в милиметры и центрирование относительно середины изображения
        centroid_x = (centroid_x_px - x_size / 2) * pixel_size_x
        centroid_y = (centroid_y_px - y_size / 2) * pixel_size_y
        
        return centroid_x, centroid_y
    
    def calculate_rms(self, image, pixel_size_x, pixel_size_y):
        """
        Рассчитывает RMS (Root Mean Square) отклонения пучка по осям X и Y
        в центрированной системе координат
        
        Args:
            image: Двумерный массив значений светимости
            pixel_size_x: Размер пикселя по оси X (мм)
            pixel_size_y: Размер пикселя по оси Y (мм)
            
        Returns:
            (rms_x, rms_y): RMS отклонения по осям X и Y (мм)
        """
        if image is None or image.size == 0:
            return 0, 0
            
        # Применяем ROI если задан
        processed_image = self.apply_roi(image)
        
        # Получаем центроид в пикселях относительно начала координат
        y_size, x_size = processed_image.shape
        
        # Вычисляем суммы интенсивности по осям
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)
        
        # Избегаем деления на ноль
        x_sum = np.sum(x_proj)
        y_sum = np.sum(y_proj)
        
        if x_sum == 0 or y_sum == 0:
            return 0, 0
        
        # Вычисляем взвешенное среднее положение (центроид) в пикселях
        centroid_x_px = np.sum(np.arange(x_size) * x_proj) / x_sum
        centroid_y_px = np.sum(np.arange(y_size) * y_proj) / y_sum
        
        # Вычисляем RMS
        rms_x_px = np.sqrt(np.sum(x_proj * (np.arange(x_size) - centroid_x_px)**2) / x_sum)
        rms_y_px = np.sqrt(np.sum(y_proj * (np.arange(y_size) - centroid_y_px)**2) / y_sum)
        
        # Преобразование в милиметры
        rms_x = rms_x_px * pixel_size_x
        rms_y = rms_y_px * pixel_size_y
        
        return rms_x, rms_y
        
    @staticmethod
    def gaussian(x, a, mu, sigma):
        """
        Функция Гаусса для аппроксимации распределения
        
        Args:
            x: Массив координат
            a: Амплитуда
            mu: Среднее значение
            sigma: Стандартное отклонение
            
        Returns:
            Значения функции Гаусса в точках x
        """
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    def fit_gaussian(self, coords, proj):
        """
        Аппроксимирует проекцию распределения функцией Гаусса
        
        Args:
            coords: Массив координат по оси
            proj: Проекция на ось
            
        Returns:
            (a, mu, sigma): Параметры функции Гаусса
            fitted_curve: Значения аппроксимирующей функции
        """
        if coords is None or proj is None or len(coords) == 0 or len(proj) == 0:
            return (0, 0, 0), np.zeros_like(coords) if coords is not None else np.array([])
            
        # Начальное приближение параметров
        a_init = np.max(proj)
        mu_init = coords[np.argmax(proj)]
        sigma_init = (np.max(coords) - np.min(coords)) / 6  # Примерно 3-sigma
        
        try:
            # Аппроксимация функцией Гаусса
            popt, _ = curve_fit(self.gaussian, coords, proj, p0=[a_init, mu_init, sigma_init])
            a, mu, sigma = popt
            
            # Вычисление аппроксимирующей кривой
            fitted_curve = self.gaussian(coords, a, mu, sigma)
            
            return (a, mu, sigma), fitted_curve
        except:
            # В случае ошибки аппроксимации возвращаем начальное приближение
            fitted_curve = self.gaussian(coords, a_init, mu_init, sigma_init)
            return (a_init, mu_init, sigma_init), fitted_curve
