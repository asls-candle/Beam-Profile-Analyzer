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
    - Нормализация с использованием референсного изображения для сохранения относительных интенсивностей
    - Обработка множества кадров фона для создания усредненной модели фона

    Все методы класса реализованы как статические, так как не требуют сохранения
    состояния между вызовами и могут быть использованы без создания экземпляра класса.

    Примечание:
        Класс обрабатывает угловые случаи, такие как пустые массивы или
        массивы с одинаковыми значениями, чтобы избежать ошибок вычисления.
    """
    @staticmethod
    def normalize(image, reference_image=None):
        """
        Нормализует изображение в диапазоне [0, 1]

        Args:
            image: Двумерный массив значений светимости для нормализации
            reference_image: Опциональное референсное изображение для определения min/max значений.
                           Если None, используются min/max самого изображения.

        Returns:
            Нормализованный двумерный массив значений светимости
        """
        logger.debug("Нормализация массива размером {}".format(image.shape if hasattr(image, 'shape') else 'неизвестно'))
        if image is None or image.size == 0:
            logger.warning("Пустой массив на входе. Возвращается нулевой массив.")
            return np.zeros((1, 1), dtype=np.float64)

        # Преобразование в тип float64 для большей точности
        image_float = image.astype(np.float64)

        # Определяем источник для min/max значений
        if reference_image is not None and reference_image.size > 0:
            reference_float = reference_image.astype(np.float64)
            min_val = np.min(reference_float)
            max_val = np.max(reference_float)
            logger.debug("Используются min/max из референсного изображения: [{}, {}]".format(min_val, max_val))
        else:
            min_val = np.min(image_float)
            max_val = np.max(image_float)
            logger.debug("Используются min/max из самого изображения: [{}, {}]".format(min_val, max_val))

        # Избегаем деления на ноль
        if max_val == min_val:
            logger.warning("Референсный диапазон содержит одинаковые значения. Возвращается нулевой массив.")
            return np.zeros_like(image_float, dtype=np.float64)

        # Нормализация в диапазон [0, 1] с использованием референсных min/max
        normalized = (image_float - min_val) / (max_val - min_val)

        # Ограничиваем значения в диапазоне [0, 1] на случай, если image выходит за пределы reference
        normalized = np.clip(normalized, 0, 1)

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
        logger.debug("Averaged background:")
        logger.debug(avg_background)
        return avg_background
