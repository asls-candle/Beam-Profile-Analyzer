import numpy as np

class ImageNormalizer:
    """
    Класс для нормализации изображений (матриц светимости)
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
        if image is None or image.size == 0:
            return np.zeros((1, 1), dtype=np.float64)
            
        # Преобразование в тип float64 для большей точности
        image_float = image.astype(np.float64)
        
        # Находим минимальное и максимальное значение
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        
        # Избегаем деления на ноль
        if max_val == min_val:
            return np.zeros_like(image_float, dtype=np.float64)
        
        # Нормализация в диапазон [0, 1]
        normalized = (image_float - min_val) / (max_val - min_val)
        
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
