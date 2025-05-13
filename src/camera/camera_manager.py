import time
import numpy as np
from threading import Lock
try:
    import PySpin
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("Spinnaker не найден. Функциональность камеры не будет доступна.")

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
        self.system = None
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
            print("Spinnaker не установлен. Невозможно подключиться к камере.")
            return False
            
        if camera_name not in self.CAMERAS:
            print(f"Камера {camera_name} не найдена")
            return False
            
        try:
            with self.lock:
                # Закрываем текущее соединение если есть
                if self.camera is not None:
                    self.disconnect_camera()
                
                # Инициализируем систему
                self.system = PySpin.System.GetInstance()
                
                # Получаем список камер
                cam_list = self.system.GetCameras()
                if cam_list.GetSize() == 0:
                    print("Камеры не найдены")
                    return False
                
                # Ищем камеру по имени в Linux
                linux_name = self.CAMERAS[camera_name]["linux_name"]
                camera_found = False
                
                for i in range(cam_list.GetSize()):
                    camera = cam_list.GetByIndex(i)
                    camera_info = camera.GetTLDeviceNodeMap()
                    device_name = PySpin.CStringPtr(camera_info.GetNode("DeviceModelName")).GetValue()
                    
                    if linux_name in device_name:
                        self.camera = camera
                        camera_found = True
                        break
                
                if not camera_found:
                    print(f"Камера {camera_name} не найдена в системе")
                    return False
                
                # Инициализируем камеру
                self.camera.Init()
                
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
                self.camera.EndAcquisition()
                self.camera.DeInit()
                del self.camera
                self.camera = None
                self.camera_name = None
                if self.system is not None:
                    del self.system
                self.system = None
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
                self.camera.EndAcquisition()
            except:
                pass
                
            # Настраиваем триггер
            node_map = self.camera.GetNodeMap()
            
            # Включаем триггер
            trigger_mode = PySpin.CEnumerationPtr(node_map.GetNode("TriggerMode"))
            trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetCurrentEntry())
            trigger_mode_on.SetIntValue(1)
            
            # Устанавливаем источник триггера
            trigger_source = PySpin.CEnumerationPtr(node_map.GetNode("TriggerSource"))
            trigger_source_line0 = PySpin.CEnumEntryPtr(trigger_source.GetEntryByName("Line0"))
            trigger_source.SetIntValue(trigger_source_line0.GetValue())
            
            # Запускаем захват
            self.camera.BeginAcquisition()
            
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
                image = self.camera.GetNextImage()
                
                # Получаем данные изображения как numpy массив
                frame = image.GetNDArray()
                
                # Освобождаем изображение
                image.Release()
                
                return frame
        except PySpin.SpinnakerException as e:
            print(f"Ошибка при захвате кадра: {e.message}")
            print(f"Полное сообщение об ошибке: {e.fullmessage}")
            print(f"Код ошибки: {e.errorcode}")
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
