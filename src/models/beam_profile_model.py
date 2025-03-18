"""
Модель данных для профиля пучка
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class BeamProfileData:
    """Класс для хранения данных о профиле пучка"""
    
    # Данные изображения
    image_data: np.ndarray
    
    # Фоновое изображение (может отсутствовать)
    background_data: Optional[np.ndarray] = None
    
    # Разностное изображение (data - background)
    difference_data: Optional[np.ndarray] = None
    
    # Координатные сетки
    x_grid_mm: Optional[np.ndarray] = None
    y_grid_mm: Optional[np.ndarray] = None
    
    # Проекции по осям
    x_projection: Optional[np.ndarray] = None
    y_projection: Optional[np.ndarray] = None
    
    # Статистики пучка
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    rms_x: float = 0.0
    rms_y: float = 0.0
    
    # Метаданные
    filename: Optional[str] = None
    camera_model: Optional[str] = None
    timestamp: Optional[str] = None
    
    # Настройки обработки
    pixel_size_mm: float = 0.0068  # Размер пикселя в мм (для Flea2)
    
    def calculate_difference(self) -> None:
        """Вычисление разностного изображения"""
        if self.background_data is not None:
            if self.background_data.shape == self.image_data.shape:
                # Вычитаем фон и обрезаем отрицательные значения
                self.difference_data = np.maximum(
                    self.image_data.astype(np.float32) - 
                    self.background_data.astype(np.float32), 
                    0
                ).astype(self.image_data.dtype)
            else:
                raise ValueError("Размеры изображения и фона не совпадают")
        else:
            raise ValueError("Фоновые данные отсутствуют")
    
    def calculate_coordinate_grids(self) -> None:
        """Вычисление координатных сеток в мм"""
        if self.image_data is not None:
            # Получаем размеры изображения
            height, width = self.image_data.shape
            
            # Создаем координатные сетки в пикселях
            x_pixels = np.arange(width)
            y_pixels = np.arange(height)
            
            # Преобразуем в мм, центрируя изображение
            x_center = width / 2
            y_center = height / 2
            
            self.x_grid_mm = (x_pixels - x_center) * self.pixel_size_mm
            self.y_grid_mm = (y_pixels - y_center) * self.pixel_size_mm
        else:
            raise ValueError("Данные изображения отсутствуют")
    
    def calculate_projections(self) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление проекций профиля пучка по осям"""
        if self.image_data is not None:
            # Получаем данные для обработки
            data = self.difference_data if self.difference_data is not None else self.image_data
            
            # Вычисляем проекции
            self.x_projection = np.sum(data, axis=0)
            self.y_projection = np.sum(data, axis=1)
            
            # Нормализуем проекции
            if np.max(self.x_projection) > 0:
                self.x_projection = self.x_projection / np.max(self.x_projection)
                
            if np.max(self.y_projection) > 0:
                self.y_projection = self.y_projection / np.max(self.y_projection)
                
            return self.x_projection, self.y_projection
        else:
            raise ValueError("Данные изображения отсутствуют")
    
    def calculate_centroids(self) -> Tuple[float, float]:
        """Вычисление центроидов профиля пучка"""
        if self.image_data is not None and self.x_grid_mm is not None and self.y_grid_mm is not None:
            # Получаем данные для обработки
            data = self.difference_data if self.difference_data is not None else self.image_data
            
            # Вычисляем проекции, если они еще не вычислены
            if self.x_projection is None or self.y_projection is None:
                self.calculate_projections()
            
            # Вычисляем центроиды
            if np.sum(self.x_projection) > 0:
                self.centroid_x = np.sum(self.x_grid_mm * self.x_projection) / np.sum(self.x_projection)
            else:
                self.centroid_x = 0.0
                
            if np.sum(self.y_projection) > 0:
                self.centroid_y = np.sum(self.y_grid_mm * self.y_projection) / np.sum(self.y_projection)
            else:
                self.centroid_y = 0.0
                
            return self.centroid_x, self.centroid_y
        else:
            raise ValueError("Данные или координатные сетки отсутствуют")
    
    def calculate_rms(self) -> Tuple[float, float]:
        """Вычисление RMS (среднеквадратичного разброса) профиля пучка"""
        if self.image_data is not None and self.x_grid_mm is not None and self.y_grid_mm is not None:
            # Получаем данные для обработки
            data = self.difference_data if self.difference_data is not None else self.image_data
            
            # Вычисляем проекции, если они еще не вычислены
            if self.x_projection is None or self.y_projection is None:
                self.calculate_projections()
                
            # Вычисляем центроиды, если они еще не вычислены
            if self.centroid_x == 0.0 or self.centroid_y == 0.0:
                self.calculate_centroids()
            
            # Вычисляем RMS
            if np.sum(self.x_projection) > 0:
                self.rms_x = np.sqrt(
                    np.sum(((self.x_grid_mm - self.centroid_x) ** 2) * self.x_projection) / 
                    np.sum(self.x_projection)
                )
            else:
                self.rms_x = 0.0
                
            if np.sum(self.y_projection) > 0:
                self.rms_y = np.sqrt(
                    np.sum(((self.y_grid_mm - self.centroid_y) ** 2) * self.y_projection) / 
                    np.sum(self.y_projection)
                )
            else:
                self.rms_y = 0.0
                
            return self.rms_x, self.rms_y
        else:
            raise ValueError("Данные или координатные сетки отсутствуют") 