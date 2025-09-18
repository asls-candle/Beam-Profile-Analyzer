import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a 2D median filter to an image.

    Args:
        image: Двумерный массив значений интенсивности (numpy.ndarray).
        kernel_size: Нечетный размер окна медианного фильтра. Если чётный, будет увеличен на 1.

    Returns:
        numpy.ndarray: Отфильтрованное изображение с той же формой и dtype, что у входного массива.

    Notes:
        - Функция безопасно обрабатывает значения `None`, возвращая `None` без ошибок.
        - Для краёв используется режим 'nearest', что сохраняет размеры изображения.
    """
    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        # Пытаемся привести вход к numpy массиву (например, для masked array берём .data)
        try:
            image = np.asarray(getattr(image, "data", image))
        except Exception:
            return None

    if kernel_size is None or kernel_size < 1:
        return image

    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    filtered = scipy_median_filter(image, size=int(kernel_size), mode="nearest")
    # Возвращаем в исходном dtype при необходимости
    if filtered.dtype != image.dtype:
        try:
            filtered = filtered.astype(image.dtype, copy=False)
        except Exception:
            pass
    return filtered
