import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImageNormalizer:
    """
    Класс для нормализации изображений и обработки фона при анализе профиля пучка.
    
    Предоставляет функциональность для стандартизации и предварительной обработки 
    данных изображений перед их анализом. Включает методы для нормализации интенсивности 
    изображений в диапазон [0, 1] и обработки кадров фона для последующего вычитания.
    
    Основные функциональные возможности:
    - Нормализация изображений для приведения значений в стандартный диапазон [0, 1]
    - Обработка множества кадров фона для создания усредненной модели фона
    
    Все методы класса реализованы как статические, так как не требуют сохранения 
    состояния между вызовами и могут быть использованы без создания экземпляра класса.
    
    Примечание:
        Класс обрабатывает угловые случаи, такие как пустые массивы или 
        массивы с одинаковыми значениями, чтобы избежать ошибок вычисления.
    """
    @staticmethod
    def normalize(image):
        """
        Нормализует изображение в диапазоне [0, 1]
        
        Args:
            image: Двумерный массив значений светимости
            
        Returns:
            Нормализованный двумерный массив значений светимости
        """
        logger.debug("Нормализация массива размером {}".format(image.shape if hasattr(image, 'shape') else 'неизвестно'))
        if image is None or image.size == 0:
            logger.warning("Пустой массив на входе. Возвращается нулевой массив.")
            return np.zeros((1, 1), dtype=np.float64)
            
        # Преобразование в тип float64 для большей точности
        image_float = image.astype(np.float64)
        
        # Находим минимальное и максимальное значение
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        logger.debug("Диапазон значений в массиве: [{}, {}]".format(min_val, max_val))
        
        # Избегаем деления на ноль
        if max_val == min_val:
            logger.warning("Массив содержит одинаковые значения. Возвращается нулевой массив.")
            return np.zeros_like(image_float, dtype=np.float64)
        
        # Нормализация в диапазон [0, 1]
        normalized = (image_float - min_val) / (max_val - min_val)
        logger.debug("Нормализация массива завершена успешно")
        
        return normalized
        
    @staticmethod
    def process_background(background_frames):
        """
        Обрабатывает кадры фона для получения усредненного фона
        
        Args:
            background_frames: Трехмерный массив кадров фона
            
        Returns:
            Усредненный двумерный массив фона
        """
        if background_frames is None or len(background_frames) == 0:
            return None
            
        # Усреднение кадров фона по оси Z
        avg_background = np.mean(background_frames, axis=0)
        
        # Замена отрицательных значений на ноль
        avg_background[avg_background < 0] = 0
        
        return avg_background
