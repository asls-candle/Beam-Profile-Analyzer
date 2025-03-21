import time
import numpy as np
from threading import Lock
try:
    import PyCapture2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("PyCapture2 не найден. Функциональность камеры не будет доступна.")

class CameraManager:
    """
    Класс для управления камерой и получения данных с нее
    """
    # Информация о камерах
    CAMERAS = {
        "GUN_YAG1": {
            "resolution": (1032, 776),
            "pixel_size_x": 0.02840909,
            "pixel_size_y": 0.02840909,
            "linux_name": "fw1"
        },
        "GUN_YAG2": {
            "resolution": (1624, 1224),
            "pixel_size_x": 0.01875468,
            "pixel_size_y": 0.01875468,
            "linux_name": "fw2"
        }
    }
    
    def __init__(self):
        self.camera = None
        self.camera_name = None
        self.bus_manager = None
        self.is_connected = False
        self.lock = Lock()  # Для потокобезопасности
        
    def get_camera_list(self):
        """
        Возвращает список доступных камер
        
        Returns:
            Список имен камер
        """
        return list(self.CAMERAS.keys())
    
    def get_camera_info(self, camera_name=None):
        """
        Возвращает информацию о камере
        
        Args:
            camera_name: Имя камеры. Если None, используется текущая выбранная камера.
            
        Returns:
            Словарь с информацией о камере или None, если камера не найдена
        """
        if camera_name is None:
            camera_name = self.camera_name
            
        if camera_name in self.CAMERAS:
            return self.CAMERAS[camera_name]
        return None
        
    def connect_to_camera(self, camera_name):
        """
        Подключается к камере по имени
        
        Args:
            camera_name: Имя камеры
            
        Returns:
            True если подключение успешно, иначе False
        """
        if not CAMERA_AVAILABLE:
            print("PyCapture2 не установлен. Невозможно подключиться к камере.")
            return False
            
        if camera_name not in self.CAMERAS:
            print(f"Камера {camera_name} не найдена")
            return False
            
        try:
            with self.lock:
                # Закрываем текущее соединение если есть
                if self.camera is not None:
                    self.disconnect_camera()
                
                # Создаем новый менеджер шины
                self.bus_manager = PyCapture2.BusManager()
                
                # Получаем количество камер
                num_cameras = self.bus_manager.getNumOfCameras()
                if num_cameras == 0:
                    print("Камеры не найдены")
                    return False
                
                # Ищем камеру по имени в Linux
                linux_name = self.CAMERAS[camera_name]["linux_name"]
                camera_index = None
                
                for i in range(num_cameras):
                    camera_guid = self.bus_manager.getCameraFromIndex(i)
                    camera = PyCapture2.Camera()
                    camera.connect(camera_guid)
                    camera_info = camera.getCameraInfo()
                    
                    # Проверяем совпадает ли имя камеры с заданным
                    if linux_name in camera_info.interfaceName:
                        camera_index = i
                        self.camera = camera
                        break
                    else:
                        # Освобождаем камеру, если это не та что нам нужна
                        camera.disconnect()
                
                if camera_index is None:
                    print(f"Камера {camera_name} не найдена в системе")
                    return False
                
                # Настраиваем камеру
                self._configure_camera()
                
                self.camera_name = camera_name
                self.is_connected = True
                
                print(f"Подключено к камере {camera_name}")
                return True
                
        except Exception as e:
            print(f"Ошибка при подключении к камере: {e}")
            return False
    
    def disconnect_camera(self):
        """
        Отключается от камеры
        
        Returns:
            True если отключение успешно, иначе False
        """
        if not self.is_connected or self.camera is None:
            return True
            
        try:
            with self.lock:
                self.camera.stopCapture()
                self.camera.disconnect()
                self.camera = None
                self.camera_name = None
                self.bus_manager = None
                self.is_connected = False
                return True
        except Exception as e:
            print(f"Ошибка при отключении от камеры: {e}")
            return False
    
    def _configure_camera(self):
        """
        Настраивает параметры камеры
        """
        if not self.is_connected or self.camera is None:
            return
            
        try:
            # Останавливаем захват если он был включен
            try:
                self.camera.stopCapture()
            except:
                pass
                
            # Настраиваем триггер
            trigger_mode = PyCapture2.TRIGGER_MODE.TRIGGER_MODE_0
            
            trigger = PyCapture2.TriggerMode()
            trigger.onOff = True
            trigger.mode = trigger_mode
            trigger.parameter = 0
            trigger.source = 0  # Внешний триггер
            
            self.camera.setTriggerMode(trigger)
            
            # Запускаем захват
            self.camera.startCapture()
            
        except Exception as e:
            print(f"Ошибка при настройке камеры: {e}")
    
    def capture_single_frame(self):
        """
        Захватывает один кадр с камеры
        
        Returns:
            Двумерный массив значений светимости или None в случае ошибки
        """
        if not self.is_connected or self.camera is None:
            return None
            
        try:
            with self.lock:
                # Ожидаем кадр
                image = self.camera.retrieveBuffer()
                
                # Преобразуем данные изображения в numpy массив
                rows, cols = image.getRows(), image.getCols()
                image_data = image.getData()
                
                # Преобразуем данные в 16-битный массив
                frame = np.array(image_data, dtype=np.uint16).reshape((rows, cols))
                
                return frame
        except PyCapture2.Fc2error as e:
            print(f"Ошибка при захвате кадра: {e}")
            return None
        except Exception as e:
            print(f"Неизвестная ошибка при захвате кадра: {e}")
            return None
    
    def capture_background_frames(self, num_frames=40):
        """
        Захватывает несколько кадров для фона
        
        Args:
            num_frames: Количество кадров для захвата
            
        Returns:
            Трехмерный массив кадров фона или None в случае ошибки
        """
        if not self.is_connected or self.camera is None:
            return None
            
        frames = []
        
        try:
            for i in range(num_frames):
                # Захватываем кадр
                frame = self.capture_single_frame()
                if frame is not None:
                    frames.append(frame)
                
                # Задержка между кадрами
                time.sleep(0.25)
                
                # Если собрали достаточно кадров или процесс был прерван
                if len(frames) >= num_frames:
                    break
                    
            # Преобразуем список кадров в трехмерный массив numpy
            if frames:
                return np.array(frames)
            return None
            
        except Exception as e:
            print(f"Ошибка при захвате кадров фона: {e}")
            return None
