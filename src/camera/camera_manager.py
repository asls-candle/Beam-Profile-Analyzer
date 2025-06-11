import time
import numpy as np
import logging
from threading import Lock
try:
    from pydc1394 import Camera, Context
    CAMERA_AVAILABLE = True
except (ImportError, TypeError, OSError) as e:
    CAMERA_AVAILABLE = False
    print("pydc1394 не найден или не может быть загружен. Функциональность камеры не будет доступна.")
    print("Ошибка: {}".format(e))

# Получаем логгер для модуля camera
logger = logging.getLogger("camera")

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
    
    def __init__(self, use_trigger=False):
        """
        Инициализирует менеджер камеры
        
        Args:
            use_trigger: Использовать ли внешний триггер для камеры
        """
        self.camera = None
        self.camera_name = None
        self.context = None
        self.is_connected = False
        self.use_trigger = use_trigger  # Флаг использования триггера
        self.lock = Lock()  # Для потокобезопасности
        logger.info("Инициализация CameraManager, use_trigger=%s", use_trigger)
        print("Инициализация CameraManager, режим триггера: {}".format("Внешний" if use_trigger else "Без триггера"))
        
    def set_trigger_mode(self, use_trigger):
        """
        Устанавливает режим триггера камеры
        
        Args:
            use_trigger: True для использования внешнего триггера, False для работы без триггера
            
        Returns:
            True если режим успешно изменен, иначе False
        """
        self.use_trigger = use_trigger
        logger.info("Установка режима триггера: %s", "Внешний" if use_trigger else "Без триггера")
        print("Установка режима триггера: {}".format("Внешний" if use_trigger else "Без триггера"))
        
        # Если камера подключена, перенастраиваем ее
        if self.is_connected and self.camera is not None:
            return self._configure_camera()
        return True
        
    def get_camera_list(self):
        """
        Возвращает список доступных камер
        
        Returns:
            Список имен камер
        """
        if not CAMERA_AVAILABLE:
            logger.warning("pydc1394 не доступен, возвращаем список камер из конфигурации")
            print("pydc1394 не доступен, возвращаем список камер из конфигурации")
            return list(self.CAMERAS.keys())
            
        # Получаем список физически подключенных камер
        try:
            logger.info("Получение списка физически подключенных камер")
            print("Получение списка физически подключенных камер")
            context = Context()
            cameras = context.cameras
            
            if not cameras:
                logger.warning("Физические камеры не обнаружены, возвращаем список из конфигурации")
                print("Физические камеры не обнаружены, возвращаем список из конфигурации")
                return list(self.CAMERAS.keys())
            
            # Создаем список имен камер на основе реальных устройств
            available_cameras = []
            for i, cam in enumerate(cameras):
                try:
                    # Пытаемся получить информацию о камере
                    logger.debug("Получение информации о камере %d: %s", i, cam)
                    camera = Camera(guid=cam[0])
                    vendor = camera.vendor.decode('utf-8', errors='replace') if isinstance(camera.vendor, bytes) else camera.vendor
                    model = camera.model.decode('utf-8', errors='replace') if isinstance(camera.model, bytes) else camera.model
                    
                    # Создаем имя на основе информации о камере
                    camera_name = "Camera_{}_{}_{}"
                    camera_name = camera_name.format(i, vendor, model)
                    logger.info("Обнаружена камера: %s", camera_name)
                    
                    # Добавляем информацию о камере в словарь CAMERAS, если её там еще нет
                    if camera_name not in self.CAMERAS:
                        try:
                            width = camera.width
                            height = camera.height
                            self.CAMERAS[camera_name] = {
                                "resolution": (width, height),
                                "pixel_size_x": 0.01, # приблизительно, нужно уточнить для реальной камеры
                                "pixel_size_y": 0.01, # приблизительно, нужно уточнить для реальной камеры
                                "linux_name": str(cam)
                            }
                            logger.info("Добавлена новая камера в конфигурацию: %s, разрешение %dx%d", 
                                       camera_name, width, height)
                            print("Добавлена новая камера в конфигурацию: {}, разрешение {}x{}".format(
                                camera_name, width, height))
                        except AttributeError:
                            # Если не удалось получить разрешение
                            self.CAMERAS[camera_name] = {
                                "resolution": (800, 600),  # стандартное разрешение
                                "pixel_size_x": 0.01,
                                "pixel_size_y": 0.01,
                                "linux_name": str(cam)
                            }
                            logger.warning("Не удалось получить разрешение камеры %s, установлено стандартное значение", camera_name)
                            print("Не удалось получить разрешение камеры {}, установлено стандартное значение".format(camera_name))
                    
                    available_cameras.append(camera_name)
                    
                    # Закрываем соединение с камерой
                    if hasattr(camera, 'stop_capture'):
                        camera.stop_capture()
                    del camera
                    
                except Exception as e:
                    logger.error("Ошибка при получении информации о камере %d: %s", i, e, exc_info=True)
                    print("Ошибка при получении информации о камере {}: {}".format(i, e))
                    # Добавляем камеру с общим именем
                    camera_name = "Camera_{}".format(i)
                    if camera_name not in self.CAMERAS:
                        self.CAMERAS[camera_name] = {
                            "resolution": (800, 600),
                            "pixel_size_x": 0.01,
                            "pixel_size_y": 0.01,
                            "linux_name": str(cam)
                        }
                        logger.info("Добавлена камера с ограниченной информацией: %s", camera_name)
                        print("Добавлена камера с ограниченной информацией: {}".format(camera_name))
                    available_cameras.append(camera_name)
            
            # Если не найдено ни одной камеры, вернем весь список из конфигурации
            if not available_cameras:
                logger.warning("Физические камеры не обнаружены, возвращаем список из конфигурации")
                print("Физические камеры не обнаружены, возвращаем список из конфигурации")
                return list(self.CAMERAS.keys())
            
            logger.info("Найдено %d камер: %s", len(available_cameras), available_cameras)
            print("Найдено {} камер: {}".format(len(available_cameras), available_cameras))
            return available_cameras
            
        except Exception as e:
            logger.error("Ошибка при получении списка камер: %s", e, exc_info=True)
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
            
        if camera_name is None:
            return None
            
        if camera_name in self.CAMERAS:
            return self.CAMERAS[camera_name]
        
        logger.warning("Камера %s не найдена в конфигурации", camera_name)
        print("Камера {} не найдена в конфигурации".format(camera_name))
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
            logger.error("pydc1394 не установлен. Невозможно подключиться к камере.")
            print("pydc1394 не установлен. Невозможно подключиться к камере.")
            return False
            
        if camera_name not in self.CAMERAS:
            logger.error("Камера %s не найдена в конфигурации", camera_name)
            print("Камера {} не найдена в конфигурации".format(camera_name))
            return False
            
        try:
            with self.lock:
                logger.info("Подключение к камере %s", camera_name)
                
                # Закрываем текущее соединение если есть
                if self.camera is not None:
                    self.disconnect_camera()
                
                # Инициализируем контекст pydc1394
                self.context = Context()
                
                # Получаем список камер
                cameras = self.context.cameras
                if not cameras:
                    logger.error("Камеры не найдены в системе")
                    print("Камеры не найдены в системе")
                    return False
                
                # Ищем камеру по имени в Linux
                linux_name = self.CAMERAS[camera_name]["linux_name"]
                logger.debug("Поиск камеры с linux_name: %s", linux_name)
                camera_found = False
                cam_guid = None
                
                for cam in cameras:
                    # Проверяем соответствие имени камеры
                    cam_info = str(cam)
                    logger.debug("Проверка камеры: %s", cam_info)
                    if linux_name in cam_info:
                        cam_guid = cam[0]  # GUID находится в первом элементе кортежа
                        camera_found = True
                        logger.info("Найдена камера %s с GUID: %s", camera_name, cam_guid)
                        break
                
                if not camera_found:
                    print("Камера {} не найдена в системе".format(camera_name))
                    logger.error("Камера {} не найдена в системе".format(camera_name))
                    return False
                
                # Создаем объект камеры с указанным GUID
                self.camera = Camera(guid=cam_guid)
                
                # Выводим информацию о камере
                print("Подключено к камере: {}".format(camera_name))
                logger.info("Подключено к камере: %s", camera_name)
                if hasattr(self.camera, 'vendor') and hasattr(self.camera, 'model'):
                    vendor = self.camera.vendor.decode('utf-8', errors='replace') if isinstance(self.camera.vendor, bytes) else self.camera.vendor
                    model = self.camera.model.decode('utf-8', errors='replace') if isinstance(self.camera.model, bytes) else self.camera.model
                    print("Производитель: {}".format(vendor))
                    print("Модель: {}".format(model))
                    logger.info("Производитель: %s", vendor)
                    logger.info("Модель: %s", model)
                
                # Инициализируем камеру
                self.camera.start_capture()
                
                # Настраиваем камеру
                self._configure_camera()
                
                self.camera_name = camera_name
                self.is_connected = True
                
                # Выводим информацию о режиме триггера
                print("Режим триггера: {}".format("Внешний" if self.use_trigger else "Без триггера"))
                logger.info("Режим триггера: {}".format("Внешний" if self.use_trigger else "Без триггера"))
                
                return True
                
        except Exception as e:
            print("Ошибка при подключении к камере: {}".format(e))
            logger.error("Ошибка при подключении к камере: {}".format(e))
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
                logger.info("Отключение от камеры %s", self.camera_name)
                print("Отключение от камеры {}".format(self.camera_name))
                self.camera.stop_capture()
                del self.camera
                self.camera = None
                self.camera_name = None
                if self.context is not None:
                    del self.context
                self.context = None
                self.is_connected = False
                logger.info("Камера успешно отключена")
                print("Камера успешно отключена")
                return True
        except Exception as e:
            logger.error("Ошибка при отключении от камеры: %s", e, exc_info=True)
            print("Ошибка при отключении от камеры: {}".format(e))
            return False
    
    def _configure_camera(self):
        """
        Настраивает параметры камеры
        
        Returns:
            True если настройка успешна, иначе False
        """
        if not self.is_connected or self.camera is None:
            return False
            
        try:
            logger.info("Настройка параметров камеры %s", self.camera_name)
            print("Настройка параметров камеры {}".format(self.camera_name))
            
            # Настраиваем общие параметры камеры
            try:
                # Настраиваем яркость
                if hasattr(self.camera, 'brightness'):
                    self.camera.brightness.mode = 'auto'
                    if self.camera.brightness.mode != 'auto':
                        self.camera.brightness.value = 200  # Увеличиваем яркость
                        logger.info("Установлена ручная яркость: %s", self.camera.brightness.value)
                        print("Установлена ручная яркость: {}".format(self.camera.brightness.value))
                
                # Настраиваем экспозицию
                if hasattr(self.camera, 'exposure'):
                    self.camera.exposure.mode = 'auto'
                    if self.camera.exposure.mode != 'auto':
                        self.camera.exposure.value = 400  # Увеличиваем экспозицию
                        logger.info("Установлена ручная экспозиция: %s", self.camera.exposure.value)
                        print("Установлена ручная экспозиция: {}".format(self.camera.exposure.value))
                
                # Настраиваем баланс белого
                if hasattr(self.camera, 'white_balance'):
                    self.camera.white_balance.mode = 'auto'
                    logger.info("Установлен автоматический баланс белого")
                    print("Установлен автоматический баланс белого")
                
                # Устанавливаем максимальное усиление для камеры
                if hasattr(self.camera, 'gain'):
                    self.camera.gain.mode = 'auto'
                    if self.camera.gain.mode != 'auto':
                        self.camera.gain.value = self.camera.gain.max
                        logger.info("Установлено максимальное усиление: %s", self.camera.gain.value)
                        print("Установлено максимальное усиление: {}".format(self.camera.gain.value))
            except AttributeError:
                logger.warning("Внимание: некоторые автоматические настройки не поддерживаются камерой")
                print("Внимание: некоторые автоматические настройки не поддерживаются камерой")
            
            # Настраиваем режим триггера, если нужно
            if self.use_trigger:
                logger.info("Настройка камеры для работы с внешним триггером")
                print("Настройка камеры для работы с внешним триггером")
                # Устанавливаем режим триггера
                self.camera.trigger_mode = 'external'
                
                # Устанавливаем источник триггера (обычно 0 для Line0)
                self.camera.trigger_source = 0
                logger.info("Включен внешний триггер, источник: %s", self.camera.trigger_source)
                print("Включен внешний триггер, источник: {}".format(self.camera.trigger_source))
            else:
                # Отключаем режим триггера, если он был включен
                if hasattr(self.camera, 'trigger_mode') and self.camera.trigger_mode != 'internal':
                    logger.info("Отключение внешнего триггера")
                    print("Отключение внешнего триггера")
                    self.camera.trigger_mode = 'internal'
            
            # Применяем настройки
            self.camera.apply_settings()
            logger.info("Настройки камеры успешно применены")
            print("Настройки камеры успешно применены")
            return True
            
        except Exception as e:
            logger.error("Ошибка при настройке камеры: %s", e, exc_info=True)
            print("Ошибка при настройке камеры: {}".format(e))
            return False
    
    def capture_single_frame(self):
        """
        Захватывает один кадр с камеры
        
        Returns:
            Двумерный массив значений светимости или None в случае ошибки
        """
        if not self.is_connected or self.camera is None:
            logger.warning("Попытка захвата кадра при отсутствии подключения к камере")
            print("Попытка захвата кадра при отсутствии подключения к камере")
            return None
            
        try:
            with self.lock:
                if self.use_trigger:
                    # В режиме внешнего триггера ожидаем кадр
                    logger.info("Ожидание кадра по внешнему триггеру...")
                    print("Ожидание кадра по внешнему триггеру...")
                    frame = self.camera.dequeue(timeout=5000)  # таймаут 5 секунд
                else:
                    # В режиме без триггера явно запускаем захват
                    logger.info("Запуск захвата одиночного кадра")
                    print("Запуск захвата одиночного кадра")
                    self.camera.start_one_shot()
                    frame = self.camera.dequeue(timeout=2000)  # таймаут 2 секунды
                
                # Копируем данные кадра
                frame_data = frame.copy()
                
                # Преобразуем в numpy массив, если необходимо
                if not isinstance(frame_data, np.ndarray):
                    frame_data = np.array(frame_data)
                
                # Возвращаем кадр в очередь
                frame.enqueue()
                
                # Останавливаем режим захвата одного кадра, если не используем триггер
                if not self.use_trigger:
                    self.camera.stop_one_shot()
                
                logger.info("Кадр успешно захвачен, размер: %s", frame_data.shape)
                print("Кадр успешно захвачен, размер: {}".format(frame_data.shape))
                return frame_data
        except Exception as e:
            logger.error("Ошибка при захвате кадра: %s", e, exc_info=True)
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
            logger.warning("Попытка захвата фоновых кадров при отсутствии подключения к камере")
            print("Попытка захвата фоновых кадров при отсутствии подключения к камере")
            return None
            
        frames = []
        
        try:
            logger.info("Начало сбора фоновых кадров, количество: %d", num_frames)
            print("Начало сбора фоновых кадров, количество: {}".format(num_frames))
            for i in range(num_frames):
                # Захватываем кадр
                frame = self.capture_single_frame()
                if frame is not None:
                    frames.append(frame)
                
                # Задержка между кадрами
                time.sleep(0.25)
                
                # Вывод прогресса
                logger.info("Захвачено кадров: %d/%d", i+1, num_frames)
                print("Захвачено кадров: {}/{}".format(i+1, num_frames), end="\r")
                
                # Если собрали достаточно кадров или процесс был прерван
                if len(frames) >= num_frames:
                    break
            
            print()  # Новая строка после прогресса
            
            # Преобразуем список кадров в трехмерный массив numpy
            if frames:
                logger.info("Сбор фоновых кадров завершен, всего кадров: %d", len(frames))
                print("Сбор фоновых кадров завершен, всего кадров: {}".format(len(frames)))
                return np.array(frames)
            
            logger.warning("Не удалось собрать фоновые кадры")
            print("Не удалось собрать фоновые кадры")
            return None
            
        except Exception as e:
            logger.error("Ошибка при захвате кадров фона: %s", e, exc_info=True)
            print("Ошибка при захвате кадров фона: {}".format(e))
            return None
