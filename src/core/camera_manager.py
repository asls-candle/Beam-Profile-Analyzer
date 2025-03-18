import numpy as np
import time
import logging
from datetime import datetime

try:
    import PyCapture2
except ImportError:
    logging.warning("PyCapture2 не установлен. Работа с камерами недоступна.")

class CameraManager:
    """Класс для управления камерами Flea2 через PyCapture2"""
    
    # Константы для камер
    CAMERA_MODELS = {
        "GUN_YAG1": {
            "name": "GUN_YAG1",
            "resolution": (1032, 776),
            "pixel_size": (0.02840909, 0.02840909),
            "linux_name": "fw1"
        },
        "GUN_YAG2": {
            "name": "GUN_YAG2",
            "resolution": (1624, 1224),
            "pixel_size": (0.01875468, 0.01875468),
            "linux_name": "fw2"
        }
    }
    
    def __init__(self):
        self.camera = None
        self.current_camera_model = None
        self.bus_manager = None
        self.background_data = None
        self.capturing = False
        self.last_frame = None
        
    def init_camera(self, camera_model):
        """Инициализация камеры по выбранной модели"""
        if camera_model not in self.CAMERA_MODELS:
            raise ValueError(f"Неизвестная модель камеры: {camera_model}")
        
        try:
            self.bus_manager = PyCapture2.BusManager()
            num_cameras = self.bus_manager.getNumOfCameras()
            
            if num_cameras == 0:
                raise RuntimeError("Камеры не обнаружены")
            
            # Для простоты выбираем первую камеру
            # В реальности нужно идентифицировать камеру по серийному номеру или другим параметрам
            camera_index = 0
            
            if camera_model == "GUN_YAG1":
                # Логика поиска камеры GUN_YAG1
                pass
            elif camera_model == "GUN_YAG2":
                # Логика поиска камеры GUN_YAG2
                pass
            
            # Создаем объект камеры и подключаемся
            self.camera = PyCapture2.Camera()
            self.camera.connect(self.bus_manager.getCameraFromIndex(camera_index))
            
            # Настраиваем камеру на работу по триггеру
            trigger_mode = PyCapture2.TRIGGER_MODE.MODE_0
            self.camera.setTriggerMode(trigger_mode)
            self.camera.setTriggerDelay(0)
            
            # Сохраняем модель текущей камеры
            self.current_camera_model = camera_model
            
            return True
        except Exception as e:
            logging.error(f"Ошибка инициализации камеры: {str(e)}")
            return False
            
    def close_camera(self):
        """Закрытие камеры"""
        if self.camera:
            try:
                self.camera.disconnect()
                self.camera = None
                self.current_camera_model = None
                return True
            except Exception as e:
                logging.error(f"Ошибка при закрытии камеры: {str(e)}")
                return False
        return True
    
    def get_camera_info(self):
        """Получение информации о текущей камере"""
        if not self.current_camera_model:
            return None
        
        return self.CAMERA_MODELS[self.current_camera_model]
    
    def start_capture(self):
        """Начало захвата изображений по триггеру"""
        if not self.camera:
            return False
        
        try:
            self.camera.startCapture()
            self.capturing = True
            return True
        except Exception as e:
            logging.error(f"Ошибка при запуске захвата: {str(e)}")
            return False
    
    def stop_capture(self):
        """Остановка захвата изображений"""
        if not self.camera or not self.capturing:
            return False
        
        try:
            self.camera.stopCapture()
            self.capturing = False
            return True
        except Exception as e:
            logging.error(f"Ошибка при остановке захвата: {str(e)}")
            return False
    
    def capture_frame(self):
        """Захват одного кадра по триггеру"""
        if not self.camera or not self.capturing:
            return None
        
        try:
            image = self.camera.retrieveBuffer()
            
            # Преобразуем формат Bayer в оттенки серого и нормализуем
            gray_image = self._bayer_to_gray(image)
            normalized_image = gray_image / np.max(gray_image)  # Нормализация от 0 до 1
            
            self.last_frame = normalized_image
            return normalized_image
        except Exception as e:
            logging.error(f"Ошибка при захвате кадра: {str(e)}")
            return None
    
    def _bayer_to_gray(self, bayer_image):
        """Преобразование изображения из формата Bayer в оттенки серого"""
        # Это упрощенная версия, фактическая реализация будет зависеть от API PyCapture2
        try:
            # Преобразование объекта PyCapture2.Image в массив NumPy
            image_data = bayer_image.getData()
            
            # Получаем разрешение камеры
            width = bayer_image.getCols()
            height = bayer_image.getRows()
            
            # Создаем массив NumPy и преобразуем в double
            img_array = np.array(image_data, dtype=np.uint8).reshape(height, width)
            gray_image = img_array.astype(np.float64)
            
            return gray_image
        except Exception as e:
            logging.error(f"Ошибка при преобразовании Bayer в оттенки серого: {str(e)}")
            return None
    
    def collect_background(self, num_frames=40, callback=None):
        """Сбор кадров для усреднения фона"""
        if not self.camera or not self.capturing:
            return False
        
        try:
            frames = []
            
            for i in range(num_frames):
                frame = self.capture_frame()
                if frame is None:
                    raise RuntimeError(f"Не удалось получить кадр {i+1}")
                
                frames.append(frame)
                
                # Вызываем колбэк для обновления UI, если он предоставлен
                if callback:
                    callback(i+1, num_frames)
                
                # Небольшая задержка между кадрами
                time.sleep(0.25)
            
            # Усредняем собранные кадры
            if frames:
                self.background_data = np.mean(np.array(frames), axis=0)
                return True
            
            return False
        except Exception as e:
            logging.error(f"Ошибка при сборе фона: {str(e)}")
            return False
    
    def get_background(self):
        """Получение усредненного фона"""
        return self.background_data
    
    def get_difference(self):
        """Получение разницы между текущим кадром и фоном"""
        if self.last_frame is not None and self.background_data is not None:
            # Вычитаем и обрезаем отрицательные значения
            diff = np.maximum(self.last_frame - self.background_data, 0)
            return diff
        return None
    
    def export_data(self, filename):
        """Экспорт данных в формате NumPy (.npy)"""
        if self.last_frame is None:
            return False
        
        try:
            camera_info = self.get_camera_info()
            
            # Создаем словарь с данными для экспорта
            export_data = {
                'resolution': camera_info['resolution'],
                'pixel_size': camera_info['pixel_size'],
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'raw': self.last_frame,
                'background': self.background_data if self.background_data is not None else np.zeros_like(self.last_frame),
                'difference': self.get_difference() if self.background_data is not None else self.last_frame
            }
            
            # Сохраняем в формате NumPy
            np.save(filename, export_data)
            return True
        except Exception as e:
            logging.error(f"Ошибка при экспорте данных: {str(e)}")
            return False 