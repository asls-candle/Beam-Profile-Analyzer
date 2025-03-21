import numpy as np
from threading import Thread, Event, Lock
import time

from src.analysis.normalizer import ImageNormalizer

class ImageReader:
    """
    Класс для считывания и обработки изображений с камеры
    """
    def __init__(self, camera_manager):
        """
        Инициализирует объект для считывания изображений
        
        Args:
            camera_manager: Объект управления камерой
        """
        self.camera_manager = camera_manager
        self.current_frame = None
        self.raw_current_frame = None
        self.background = None
        self.raw_background = None
        self.difference = None
        
        # Для работы в режиме непрерывного чтения
        self.reading_thread = None
        self.stop_reading = Event()
        self.frame_lock = Lock()
        self.background_lock = Lock()
        self.is_collecting_background = False
        
        # Для сбора фона
        self.background_frames = []
        self.background_frames_count = 40
        self.background_collection_thread = None
        
    def start_preview(self):
        """
        Запускает предварительный просмотр в режиме реального времени
        
        Returns:
            True если запуск успешен, иначе False
        """
        if self.reading_thread is not None and self.reading_thread.is_alive():
            return False
            
        self.stop_reading.clear()
        self.reading_thread = Thread(target=self._reading_worker)
        self.reading_thread.daemon = True
        self.reading_thread.start()
        return True
        
    def stop_preview(self):
        """
        Останавливает предварительный просмотр
        
        Returns:
            True если остановка успешна, иначе False
        """
        if self.reading_thread is None or not self.reading_thread.is_alive():
            return True
            
        self.stop_reading.set()
        self.reading_thread.join(timeout=1.0)
        return not self.reading_thread.is_alive()
        
    def _reading_worker(self):
        """
        Рабочая функция для потока чтения изображений
        """
        while not self.stop_reading.is_set():
            frame = self.camera_manager.capture_single_frame()
            if frame is not None:
                with self.frame_lock:
                    self.raw_current_frame = frame
                    self.current_frame = ImageNormalizer.normalize(frame)
                    
                    # Вычисляем разницу если есть фон
                    if self.background is not None:
                        # Проверяем совпадение размеров
                        if self.current_frame.shape == self.background.shape:
                            # Вычитаем фон и обрезаем отрицательные значения
                            diff = self.current_frame - self.background
                            diff[diff < 0] = 0
                            self.difference = diff
                        else:
                            print("Размеры текущего кадра и фона не совпадают")
                
            # Небольшая задержка
            time.sleep(0.1)
            
    def get_current_frame(self):
        """
        Возвращает текущий кадр
        
        Returns:
            Нормализованный текущий кадр
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def get_raw_current_frame(self):
        """
        Возвращает текущий необработанный кадр
        
        Returns:
            Необработанный текущий кадр
        """
        with self.frame_lock:
            return self.raw_current_frame.copy() if self.raw_current_frame is not None else None
            
    def get_background(self):
        """
        Возвращает фоновое изображение
        
        Returns:
            Нормализованное фоновое изображение
        """
        with self.background_lock:
            return self.background.copy() if self.background is not None else None
            
    def get_raw_background(self):
        """
        Возвращает необработанное фоновое изображение
        
        Returns:
            Необработанное фоновое изображение
        """
        with self.background_lock:
            return self.raw_background.copy() if self.raw_background is not None else None
            
    def get_difference(self):
        """
        Возвращает разницу между текущим кадром и фоном
        
        Returns:
            Разница между текущим кадром и фоном
        """
        with self.frame_lock, self.background_lock:
            return self.difference.copy() if self.difference is not None else None
    
    def start_background_collection(self, frames_count=40):
        """
        Запускает сбор кадров для фона
        
        Args:
            frames_count: Количество кадров для сбора
            
        Returns:
            True если запуск успешен, иначе False
        """
        if self.is_collecting_background:
            return False
            
        self.background_frames = []
        self.background_frames_count = frames_count
        self.is_collecting_background = True
        
        self.background_collection_thread = Thread(target=self._background_collection_worker)
        self.background_collection_thread.daemon = True
        self.background_collection_thread.start()
        
        return True
        
    def stop_background_collection(self):
        """
        Останавливает сбор кадров для фона
        
        Returns:
            True если остановка успешна, иначе False
        """
        if not self.is_collecting_background:
            return True
            
        self.is_collecting_background = False
        if self.background_collection_thread is not None:
            self.background_collection_thread.join(timeout=1.0)
            
        # Обрабатываем собранные кадры
        self._process_background_frames()
        
        return True
        
    def _background_collection_worker(self):
        """
        Рабочая функция для потока сбора фоновых кадров
        """
        frames = self.camera_manager.capture_background_frames(self.background_frames_count)
        if frames is not None and len(frames) > 0:
            self.background_frames = frames
            
        self.is_collecting_background = False
        self._process_background_frames()
        
    def _process_background_frames(self):
        """
        Обрабатывает собранные кадры фона
        """
        if not self.background_frames or len(self.background_frames) == 0:
            return
            
        # Усредняем кадры фона
        raw_bg = ImageNormalizer.process_background(self.background_frames)
        
        with self.background_lock:
            self.raw_background = raw_bg
            self.background = ImageNormalizer.normalize(raw_bg)
            
        # Вычисляем разницу с текущим кадром
        with self.frame_lock, self.background_lock:
            if self.current_frame is not None and self.background is not None:
                if self.current_frame.shape == self.background.shape:
                    diff = self.current_frame - self.background
                    diff[diff < 0] = 0
                    self.difference = diff
