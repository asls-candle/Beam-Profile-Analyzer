import numpy as np
from threading import Thread, Event, Lock
import time

from src.analysis.normalizer import ImageNormalizer

class ImageReader:
    """
    Класс для считывания и обработки изображений с камеры при анализе профиля пучка.
    
    Предоставляет интерфейс для работы с видеопотоком камеры, управления предварительным 
    просмотром, сбора и обработки фоновых кадров и вычисления разницы между текущим 
    изображением и фоном. Реализует многопоточную архитектуру для эффективной 
    обработки кадров в режиме реального времени.
    
    Основные функциональные возможности:
    - Захват и обработка кадров с камеры в режиме реального времени
    - Сбор и усреднение фоновых кадров для компенсации фона
    - Автоматическое вычитание фона из текущего кадра
    - Нормализация изображений для единообразного отображения и анализа
    - Многопоточная архитектура для одновременной обработки и отображения
    
    Attributes:
        camera_manager: Объект для управления камерой
        current_frame: Текущий нормализованный кадр
        raw_current_frame: Текущий необработанный кадр
        background: Нормализованное фоновое изображение
        raw_background: Необработанное фоновое изображение
        difference: Разница между текущим кадром и фоном
        reading_thread: Поток для непрерывного считывания кадров
        stop_reading: Событие для остановки потока чтения
        frame_lock: Блокировка для синхронизации доступа к кадрам
        background_lock: Блокировка для синхронизации доступа к фону
        is_collecting_background: Флаг состояния сбора фона
        background_frames: Список собранных фоновых кадров
        background_frames_count: Количество кадров для сбора фона
        background_collection_thread: Поток для сбора фона
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
        Рабочая функция для потока непрерывного чтения изображений с камеры.
        
        Выполняется в отдельном потоке и циклически захватывает кадры с камеры.
        Для каждого полученного кадра:
        1. Захватывает кадр с камеры
        2. Сохраняет исходный (необработанный) кадр
        3. Нормализует кадр в диапазон [0,1]
        4. Вычисляет разницу между текущим кадром и фоном (если фон доступен)
        5. Отрицательные значения разницы обрезаются до нуля
        
        Метод содержит блокировку потока для безопасного доступа к общим ресурсам
        и небольшую задержку между итерациями для снижения нагрузки на систему.
        
        Примечание:
            Этот метод выполняется в фоновом потоке и останавливается
            при установке флага self.stop_reading.
        """
        while not self.stop_reading.is_set():
            frame = self.camera_manager.capture_single_frame()
            if frame is not None:
                with self.frame_lock:
                    self.raw_current_frame = frame
                    self.current_frame = ImageNormalizer.normalize(frame)
                    
                    # Вычисляем разницу если есть фон
                    if self.background is not None:
                        with self.background_lock:
                            self._calculate_difference()
                
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
        
        # Явное оповещение о наличии фона для UI
        has_background = self.background is not None
        print("pydc1394: Фон собран: {}".format(has_background))
        
        # Явно обновляем UI после сбора фона
        print("pydc1394: Сигнализируем об обновлении UI после сбора фона")
        
    def _process_background_frames(self):
        """
        Обрабатывает собранные кадры фона для вычитания из текущего изображения.
        
        Этот метод выполняет следующие операции:
        1. Проверяет наличие собранных кадров фона
        2. Усредняет все собранные кадры с помощью ImageNormalizer.process_background
        3. Создает нормализованную версию фонового изображения
        4. Вычитает фоновое изображение из текущего кадра с защитой от отрицательных значений
        5. Сохраняет результат вычитания в self.difference
        
        Содержит синхронизацию доступа через блокировки потоков для исключения 
        состояния гонки при многопоточном доступе к общим ресурсам. 
        
        Примечание:
            Метод выполняет проверку совместимости размеров текущего кадра и фона 
            для корректного вычитания. Если размеры не совпадают, выводится сообщение 
            об ошибке и вычитание не производится.
        
        Raises:
            None: Ошибки обрабатываются внутри метода и выводятся в консоль.
        """
        if self.background_frames is None or (isinstance(self.background_frames, list) and len(self.background_frames) == 0):
            print("pydc1394: Нет кадров для обработки фона")
            return
            
        print("pydc1394: Обработка {} кадров фона".format(len(self.background_frames)))
        # Усредняем кадры фона
        raw_bg = ImageNormalizer.process_background(self.background_frames)
        
        with self.background_lock:
            self.raw_background = raw_bg
            # Используем текущий кадр как референс для нормализации фона, если он есть
            reference_frame = self.raw_current_frame if self.raw_current_frame is not None else None
            self.background = ImageNormalizer.normalize(raw_bg, reference_image=reference_frame)
            print("pydc1394: Фон успешно создан, размер: {}".format(self.background.shape))
            
        # Вычисляем разницу с текущим кадром
        with self.frame_lock, self.background_lock:
            if self._calculate_difference():
                print("pydc1394: Разница с текущим кадром вычислена")

    def _calculate_difference(self):
        """
        Вычисляет разницу между текущим кадром и фоном.
        
        Сначала выполняет вычитание необработанного фонового изображения из 
        необработанного текущего кадра, затем нормализует результат, используя
        текущий кадр как референсное изображение для сохранения правильного
        масштаба интенсивности. Все отрицательные значения заменяются на ноль.
        Результат сохраняется в атрибуте self.difference.
        
        Примечание:
            Метод предполагает, что блокировки self.frame_lock и 
            self.background_lock уже установлены вызывающим кодом.
        
        Returns:
            bool: True если вычитание успешно, False в случае ошибки
        """
        if self.raw_current_frame is None or self.raw_background is None:
            return False
            
        if self.raw_current_frame.shape != self.raw_background.shape:
            print("Размеры текущего кадра и фона не совпадают: текущий кадр {}, фон {}".format(
                self.raw_current_frame.shape, self.raw_background.shape))
            return False
            
        # Вычитаем необработанный фон из необработанного кадра
        raw_diff = self.raw_current_frame - self.raw_background
        
        # Обрезаем отрицательные значения
        raw_diff[raw_diff < 0] = 0
        
        # Нормализуем результат, используя текущий кадр как референс
        self.difference = ImageNormalizer.normalize(raw_diff, reference_image=self.raw_current_frame)
        return True
