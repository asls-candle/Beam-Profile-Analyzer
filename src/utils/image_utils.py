"""
Утилиты для обработки изображений
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Нормализация изображения в диапазон [0, 1]
    
    Args:
        image: Входное изображение
        
    Returns:
        Нормализованное изображение
    """
    if image.size == 0:
        return image
    
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max > img_min:
        return (image - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(image, dtype=np.float32)

def clip_and_scale(image: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    Обрезка и масштабирование изображения в диапазон [0, 2^bit_depth - 1]
    
    Args:
        image: Входное изображение
        bit_depth: Битовая глубина (по умолчанию 8)
        
    Returns:
        Изображение в диапазоне [0, 2^bit_depth - 1]
    """
    if image.size == 0:
        return image
    
    # Нормализуем в диапазон [0, 1]
    normalized = normalize_image(image)
    
    # Масштабируем в диапазон [0, 2^bit_depth - 1]
    max_val = (2 ** bit_depth) - 1
    return (normalized * max_val).astype(np.uint8 if bit_depth <= 8 else np.uint16)

def apply_threshold(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Применение порога к изображению
    
    Args:
        image: Входное изображение
        threshold: Пороговое значение (от 0 до 1)
        
    Returns:
        Изображение после порога
    """
    if image.size == 0:
        return image
    
    # Нормализуем в диапазон [0, 1]
    normalized = normalize_image(image)
    
    # Применяем порог
    return np.where(normalized > threshold, normalized, 0)

def remove_background(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Удаление фона из изображения
    
    Args:
        image: Входное изображение
        background: Фоновое изображение
        
    Returns:
        Изображение без фона
    """
    if image.size == 0 or background.size == 0:
        return image
    
    if image.shape != background.shape:
        logging.warning(f"Несоответствие размеров: изображение {image.shape}, фон {background.shape}")
        return image
    
    # Преобразуем в float32 для предотвращения переполнения
    img_float = image.astype(np.float32)
    bg_float = background.astype(np.float32)
    
    # Вычитаем фон и обрезаем отрицательные значения
    result = np.maximum(img_float - bg_float, 0)
    
    # Возвращаем результат в исходном типе данных
    return result.astype(image.dtype)

def find_centroid(image: np.ndarray) -> Tuple[float, float]:
    """
    Нахождение центроида изображения
    
    Args:
        image: Входное изображение
        
    Returns:
        Координаты центроида (x, y)
    """
    if image.size == 0 or np.sum(image) == 0:
        return (0.0, 0.0)
    
    # Создаем координатные сетки
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    
    # Вычисляем взвешенные суммы
    sum_intensity = np.sum(image)
    
    if sum_intensity == 0:
        return (w / 2, h / 2)  # Возвращаем центр изображения, если нет интенсивности
    
    center_x = np.sum(x * image) / sum_intensity
    center_y = np.sum(y * image) / sum_intensity
    
    return (center_x, center_y)

def calculate_rms(image: np.ndarray, centroid: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """
    Вычисление RMS (среднеквадратичного разброса) изображения
    
    Args:
        image: Входное изображение
        centroid: Координаты центроида (x, y), если не указаны, будут вычислены
        
    Returns:
        RMS по координатам (rms_x, rms_y)
    """
    if image.size == 0 or np.sum(image) == 0:
        return (0.0, 0.0)
    
    # Вычисляем центроид, если не указан
    if centroid is None:
        centroid = find_centroid(image)
    
    center_x, center_y = centroid
    
    # Создаем координатные сетки
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    
    # Вычисляем квадраты отклонений от центроида
    dx2 = (x - center_x) ** 2
    dy2 = (y - center_y) ** 2
    
    # Вычисляем взвешенную сумму
    sum_intensity = np.sum(image)
    
    if sum_intensity == 0:
        return (0.0, 0.0)
    
    rms_x = np.sqrt(np.sum(dx2 * image) / sum_intensity)
    rms_y = np.sqrt(np.sum(dy2 * image) / sum_intensity)
    
    return (rms_x, rms_y) 