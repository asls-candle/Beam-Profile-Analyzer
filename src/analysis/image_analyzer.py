import numpy as np
from scipy.optimize import curve_fit

class ImageAnalyzer:
    """
    Класс для анализа изображений с профилем пучка
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
