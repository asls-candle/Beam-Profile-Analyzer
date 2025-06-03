import time
import numpy as np
from threading import Lock
try:
    from pydc1394 import Camera, Context
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("pydc1394 не найден. Функциональность камеры не будет доступна.")

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
        """
        Инициализирует менеджер камеры
        """
        self.camera = None
        self.camera_name = None
        self.context = None
        self.is_connected = False
        self.lock = Lock()  # Для потокобезопасности
        
    def get_camera_list(self):
        """
        Возвращает список доступных камер
        
        Returns:
            Список имен камер
        """
        if not CAMERA_AVAILABLE:
            return list(self.CAMERAS.keys())
            
        # Получаем список физически подключенных камер
        try:
            context = Context()
            cameras = context.cameras
            
            # Возвращаем список имен камер, которые мы можем обнаружить
            available_cameras = []
            for camera_name, camera_info in self.CAMERAS.items():
                linux_name = camera_info.get("linux_name", "")
                
                for cam in cameras:
                    if linux_name in str(cam):
                        available_cameras.append(camera_name)
                        break
                        
            return available_cameras
        except Exception as e:
            print("Ошибка при получении списка камер: {}".format(e))
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
            print("pydc1394 не установлен. Невозможно подключиться к камере.")
            return False
            
        if camera_name not in self.CAMERAS:
            print("Камера {} не найдена в конфигурации".format(camera_name))
            return False
            
        try:
            with self.lock:
                # Закрываем текущее соединение если есть
                if self.camera is not None:
                    self.disconnect_camera()
                
                # Инициализируем контекст pydc1394
                self.context = Context()
                
                # Получаем список камер
                cameras = self.context.cameras
                if not cameras:
                    print("Камеры не найдены в системе")
                    return False
                
                # Ищем камеру по имени в Linux
                linux_name = self.CAMERAS[camera_name]["linux_name"]
                camera_found = False
                cam_guid = None
                
                for cam in cameras:
                    # Проверяем соответствие имени камеры
                    cam_info = str(cam)
                    if linux_name in cam_info:
                        cam_guid = cam[0]  # GUID находится в первом элементе кортежа
                        camera_found = True
                        break
                
                if not camera_found:
                    print("Камера {} не найдена в системе".format(camera_name))
                    return False
                
                # Создаем объект камеры с указанным GUID
                self.camera = Camera(guid=cam_guid)
                
                # Выводим информацию о камере
                print("Подключено к камере: {}".format(camera_name))
                if hasattr(self.camera, 'vendor') and hasattr(self.camera, 'model'):
                    vendor = self.camera.vendor.decode('utf-8', errors='replace') if isinstance(self.camera.vendor, bytes) else self.camera.vendor
                    model = self.camera.model.decode('utf-8', errors='replace') if isinstance(self.camera.model, bytes) else self.camera.model
                    print("Производитель: {}".format(vendor))
                    print("Модель: {}".format(model))
                
                # Инициализируем камеру
                self.camera.start_capture()
                
                # Настраиваем камеру
                self._configure_camera()
                
                self.camera_name = camera_name
                self.is_connected = True
                
                return True
                
        except Exception as e:
            print("Ошибка при подключении к камере: {}".format(e))
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
                self.camera.stop_capture()
                del self.camera
                self.camera = None
                self.camera_name = None
                if self.context is not None:
                    del self.context
                self.context = None
                self.is_connected = False
                print("Камера успешно отключена")
                return True
        except Exception as e:
            print("Ошибка при отключении от камеры: {}".format(e))
            return False
    
    def _configure_camera(self):
        """
        Настраивает параметры камеры
        """
        if not self.is_connected or self.camera is None:
            return
            
        try:
            # Настраиваем параметры камеры с помощью pydc1394
            
            # Устанавливаем режим триггера
            self.camera.trigger_mode = 'external'
            
            # Устанавливаем источник триггера (обычно 0 для Line0)
            self.camera.trigger_source = 0
            
            # Применяем настройки
            self.camera.apply_settings()
            
        except Exception as e:
            print("Ошибка при настройке камеры: {}".format(e))
    
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
                # Захватываем кадр
                self.camera.start_one_shot()
                frame = self.camera.dequeue()
                
                # Копируем данные кадра
                frame_data = frame.copy()
                
                # Преобразуем в numpy массив, если необходимо
                if not isinstance(frame_data, np.ndarray):
                    frame_data = np.array(frame_data)
                
                # Возвращаем кадр в очередь
                frame.enqueue()
                self.camera.stop_one_shot()
                
                return frame_data
        except Exception as e:
            print("Ошибка при захвате кадра: {}".format(e))
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
                
                # Вывод прогресса
                print("Захвачено кадров: {}/{}".format(i+1, num_frames), end="\r")
                
                # Если собрали достаточно кадров или процесс был прерван
                if len(frames) >= num_frames:
                    break
            
            print()  # Новая строка после прогресса
            
            # Преобразуем список кадров в трехмерный массив numpy
            if frames:
                return np.array(frames)
            return None
            
        except Exception as e:
            print("Ошибка при захвате кадров фона: {}".format(e))
            return None
