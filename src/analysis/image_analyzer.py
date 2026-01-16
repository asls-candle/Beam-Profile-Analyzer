import numpy as np
from scipy.optimize import curve_fit

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

        # Создаем координатную сетку (в пикселях), центрированную относительно 0
        y_size, x_size = processed_image.shape
        x_grid = np.arange(x_size) - (x_size - 1) / 2
        y_grid = np.arange(y_size) - (y_size - 1) / 2

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

        Метод реализован аналогично MATLAB версии (AVG_STD.m):
        - Используется система координат, центрированная относительно 0
        - Для размера N координаты от -(N-1)/2 до N/2
        - Центроид вычисляется как взвешенное среднее: sum(coords * projection) / sum(projection)

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

        # Получаем размеры изображения
        y_size, x_size = processed_image.shape

        # Создаем координаты, центрированные относительно 0 (как в MATLAB)
        # xdata = (-(x_size-1)/2):(x_size/2) с шагом 1
        x_coords = np.arange(x_size) - (x_size - 1) / 2
        y_coords = np.arange(y_size) - (y_size - 1) / 2

        # Вычисляем проекции (суммы интенсивности по осям)
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)

        # Нормализуем проекции (как в MATLAB: sxn = sx/max(sx))
        if np.max(x_proj) > 0:
            x_proj_norm = x_proj / np.max(x_proj)
        else:
            x_proj_norm = x_proj

        if np.max(y_proj) > 0:
            y_proj_norm = y_proj / np.max(y_proj)
        else:
            y_proj_norm = y_proj

        # Избегаем деления на ноль
        x_sum = np.sum(x_proj_norm)
        y_sum = np.sum(y_proj_norm)

        if x_sum == 0 or y_sum == 0:
            return 0, 0

        # Вычисляем взвешенное среднее положение (центроид) в пикселях
        # averagex = sum(xdata * sxn) / sum(sxn)
        centroid_x_px = np.sum(x_coords * x_proj_norm) / x_sum
        centroid_y_px = np.sum(y_coords * y_proj_norm) / y_sum

        # Преобразование в милиметры (координаты уже центрированные)
        centroid_x = centroid_x_px * pixel_size_x
        centroid_y = centroid_y_px * pixel_size_y

        return centroid_x, centroid_y

    def calculate_centroid_unnormalized(self, image, pixel_size_x, pixel_size_y):
        """
        Рассчитывает координаты центроида изображения в центрированной системе координат
        используя НЕ нормализованные значения интенсивности

        Метод аналогичен calculate_centroid, но использует исходные значения интенсивности
        без нормализации на максимум.

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

        # Получаем размеры изображения
        y_size, x_size = processed_image.shape

        # Создаем координаты, центрированные относительно 0
        x_coords = np.arange(x_size) - (x_size - 1) / 2
        y_coords = np.arange(y_size) - (y_size - 1) / 2

        # Вычисляем проекции (суммы интенсивности по осям)
        # НЕ НОРМАЛИЗУЕМ - используем исходные значения
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)

        # Избегаем деления на ноль
        x_sum = np.sum(x_proj)
        y_sum = np.sum(y_proj)

        if x_sum == 0 or y_sum == 0:
            return 0, 0

        # Вычисляем взвешенное среднее положение (центроид) в пикселях
        # используя НЕ нормализованные проекции
        centroid_x_px = np.sum(x_coords * x_proj) / x_sum
        centroid_y_px = np.sum(y_coords * y_proj) / y_sum

        # Преобразование в милиметры
        centroid_x = centroid_x_px * pixel_size_x
        centroid_y = centroid_y_px * pixel_size_y

        return centroid_x, centroid_y

    def calculate_rms(self, image, pixel_size_x, pixel_size_y):
        """
        Рассчитывает RMS (Root Mean Square) отклонения пучка по осям X и Y
        в центрированной системе координат

        Метод реализован аналогично MATLAB версии (AVG_STD.m):
        - Используется система координат, центрированная относительно 0
        - RMS вычисляется как: sqrt(sum(projection * (coords - centroid)^2) / sum(projection))

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

        # Получаем размеры изображения
        y_size, x_size = processed_image.shape

        # Создаем координаты, центрированные относительно 0 (как в MATLAB)
        x_coords = np.arange(x_size) - (x_size - 1) / 2
        y_coords = np.arange(y_size) - (y_size - 1) / 2

        # Вычисляем проекции (суммы интенсивности по осям)
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)

        # Нормализуем проекции (как в MATLAB: sxn = sx/max(sx))
        if np.max(x_proj) > 0:
            x_proj_norm = x_proj / np.max(x_proj)
        else:
            x_proj_norm = x_proj

        if np.max(y_proj) > 0:
            y_proj_norm = y_proj / np.max(y_proj)
        else:
            y_proj_norm = y_proj

        # Избегаем деления на ноль
        x_sum = np.sum(x_proj_norm)
        y_sum = np.sum(y_proj_norm)

        if x_sum == 0 or y_sum == 0:
            return 0, 0

        # Вычисляем взвешенное среднее положение (центроид) в пикселях
        centroid_x_px = np.sum(x_coords * x_proj_norm) / x_sum
        centroid_y_px = np.sum(y_coords * y_proj_norm) / y_sum

        # Вычисляем RMS
        # sigmax = sqrt(sum(sxn * (xdata - averagex)^2) / sum(sxn))
        rms_x_px = np.sqrt(np.sum(x_proj_norm * (x_coords - centroid_x_px)**2) / x_sum)
        rms_y_px = np.sqrt(np.sum(y_proj_norm * (y_coords - centroid_y_px)**2) / y_sum)

        # Преобразование в милиметры
        rms_x = rms_x_px * pixel_size_x
        rms_y = rms_y_px * pixel_size_y

        return rms_x, rms_y

    def calculate_rms_unnormalized(self, image, pixel_size_x, pixel_size_y):
        """
        Рассчитывает RMS (Root Mean Square) отклонения пучка по осям X и Y
        в центрированной системе координат используя НЕ нормализованные значения интенсивности

        Метод аналогичен calculate_rms, но использует исходные значения интенсивности
        без нормализации на максимум.

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

        # Получаем размеры изображения
        y_size, x_size = processed_image.shape

        # Создаем координаты, центрированные относительно 0
        x_coords = np.arange(x_size) - (x_size - 1) / 2
        y_coords = np.arange(y_size) - (y_size - 1) / 2

        # Вычисляем проекции (суммы интенсивности по осям)
        # НЕ НОРМАЛИЗУЕМ - используем исходные значения
        x_proj = np.sum(processed_image, axis=0)
        y_proj = np.sum(processed_image, axis=1)

        # Избегаем деления на ноль
        x_sum = np.sum(x_proj)
        y_sum = np.sum(y_proj)

        if x_sum == 0 or y_sum == 0:
            return 0, 0

        # Вычисляем взвешенное среднее положение (центроид) в пикселях
        # используя НЕ нормализованные проекции
        centroid_x_px = np.sum(x_coords * x_proj) / x_sum
        centroid_y_px = np.sum(y_coords * y_proj) / y_sum

        # Вычисляем RMS используя НЕ нормализованные проекции
        rms_x_px = np.sqrt(np.sum(x_proj * (x_coords - centroid_x_px)**2) / x_sum)
        rms_y_px = np.sqrt(np.sum(y_proj * (y_coords - centroid_y_px)**2) / y_sum)

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

    @staticmethod
    def denormalize(normalized_data, max_value):
        """
        Денормализует матрицу интенсивностей, приводя её к изначальному виду

        Эта функция выполняет обратную операцию нормализации, которая была
        произведена делением на максимальное значение.

        Args:
            normalized_data: Нормализованная матрица/массив интенсивностей
                           (значения от 0 до 1)
            max_value: Максимальное значение интенсивности до нормализации
                      (используется как множитель для восстановления)

        Returns:
            Денормализованная матрица с исходными значениями интенсивности

        Example:
            >>> # Нормализация
            >>> original = np.array([100, 200, 300, 400])
            >>> normalized = original / np.max(original)  # [0.25, 0.5, 0.75, 1.0]
            >>> # Денормализация
            >>> restored = ImageAnalyzer.denormalize(normalized, 400)  # [100, 200, 300, 400]
        """
        if normalized_data is None:
            return None

        if max_value == 0:
            return normalized_data

        return normalized_data * max_value
