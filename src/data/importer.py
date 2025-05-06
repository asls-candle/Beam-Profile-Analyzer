import os
import numpy as np
import re
import scipy.io as sio

import logging
logger = logging.getLogger('data')

class DataImporter:
    """
    Класс для импорта данных профиля пучка
    """
    
    @staticmethod
    def import_mat(filepath, background_filepath=None):
        """
        Импортирует данные из MAT файла
        
        Args:
            filepath: Путь к файлу
            background_filepath: Путь к файлу с фоном (если None, пытается найти автоматически)
            
        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info(f"Начало импорта MAT файла: {filepath}")
            # Загружаем основной файл
            logger.debug(f"Загрузка данных из основного файла: {filepath}")
            shot_data = sio.loadmat(filepath)
            
            # Проверяем наличие данных в файле
            if not shot_data:
                print(f"Файл {filepath} не содержит данных")
                logger.error(f"Файл {filepath} не содержит данных")
                return None
                
            # Получаем имя переменной с данными (обычно первая переменная, не начинающаяся с "__")
            logger.debug("Поиск переменных с данными в файле")
            shot_var_names = [key for key in shot_data.keys() if not key.startswith("__")]
            
            if not shot_var_names:
                print(f"Не удалось найти данные в файле {filepath}")
                logger.error(f"Не удалось найти данные в файле {filepath}")
                return None
            
            # Извлекаем первую переменную, которая не начинается с "__"
            shot_var_name = shot_var_names[0]
            logger.debug(f"Найдены переменные в файле: {shot_var_names}, выбрана первая: {shot_var_name}")
                
            # Получаем данные снимка
            logger.debug(f"Получение данных снимка с использованием переменной: {shot_var_name}")
            shot = shot_data[shot_var_name]
            
            # Если путь к файлу с фоном не указан, пытаемся найти его автоматически
            if background_filepath is None:
                logger.info("Путь к файлу с фоном не указан, выполняется автоматический поиск")
                background_filepath = DataImporter._find_background_file(filepath)
                
            # Если файл с фоном не найден
            if background_filepath is None:
                # Выводим сообщение один раз, а не в каждой итерации поиска файла
                print(f"Файл с фоном для {filepath} не найден")
                logger.error(f"Файл с фоном для {filepath} не найден")
                return None
                
            # Загружаем файл с фоном
            logger.debug(f"Загрузка файла с фоном: {background_filepath}")
            bg_data = sio.loadmat(background_filepath)
            
            # Получаем имя переменной с данными фона
            logger.debug("Поиск переменных с данными фона в файле")
            bg_var_names = [key for key in bg_data.keys() if not key.startswith("__")]
            
            if not bg_var_names:
                print(f"Не удалось найти данные в файле {background_filepath}")
                logger.error(f"Не удалось найти данные в файле {background_filepath}")
                return None
                
            # Извлекаем первую переменную фона, которая не начинается с "__"
            bg_var_name = bg_var_names[0]
            logger.debug(f"Найдены переменные в файле фона: {bg_var_names}, выбрана первая: {bg_var_name}")
                
            # Получаем данные фона
            logger.debug(f"Получение данных фона с использованием переменной: {bg_var_name}")
            background = bg_data[bg_var_name]
            
            # Проверяем совпадение размеров
            if shot.shape != background.shape:
                print(f"Размеры снимка ({shot.shape}) и фона ({background.shape}) не совпадают")
                logger.error(f"Размеры снимка ({shot.shape}) и фона ({background.shape}) не совпадают")
                return None
                
            # Определяем камеру по размеру массива
            logger.debug(f"Определение камеры по размеру массива: {shot.shape}")
            camera_info = DataImporter._determine_camera_by_shape(shot.shape)
            logger.info(f"Определена камера: {camera_info['name']}")
            
            # Нормализуем данные
            logger.debug("Начало нормализации данных снимка")
            normalized_shot = DataImporter._normalize_array(shot)
            logger.debug("Начало нормализации данных фона")
            normalized_background = DataImporter._normalize_array(background)
            
            # Вычисляем разницу
            logger.debug("Вычисление разницы между снимком и фоном")
            difference = normalized_shot - normalized_background
            difference[difference < 0] = 0
            
            # Формируем результирующий словарь
            logger.debug("Формирование результирующего словаря с данными")
            result = {
                "shot": normalized_shot,
                "background": normalized_background,
                "difference": difference,
                "raw_shot": shot,
                "raw_background": background,
                "resolution": camera_info["resolution"],
                "pixel_size_x": camera_info["pixel_size_x"],
                "pixel_size_y": camera_info["pixel_size_y"],
                "camera_name": camera_info["name"],
                # Добавляем эти поля, чтобы избежать KeyError в UI
                "current_frame": normalized_shot,
                "centroid": (0, 0),
                "rms": (0, 0),
                "filepath": filepath
            }
            
            logger.info(f"Успешно импортирован MAT файл: {filepath}")
            return result
        except Exception as e:
            print(f"Ошибка при импорте MAT файла: {e}")
            logger.error(f"Ошибка при импорте MAT файла: {e}", exc_info=True)
            return None
            
    @staticmethod
    def _find_background_file(filepath):
        """
        Находит соответствующий файл с фоном
        
        Args:
            filepath: Путь к основному файлу
            
        Returns:
            str: Путь к файлу с фоном или None, если не найден
        """
        try:
            logger.debug(f"Поиск файла с фоном для: {filepath}")
            # Получаем директорию и имя файла
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            base_name, ext = os.path.splitext(filename)
            
            logger.debug(f"Исходный файл: директория={directory}, имя={filename}, базовое имя={base_name}, расширение={ext}")
            
            # Шаблоны для поиска фонового файла
            logger.debug("Применение шаблонов для поиска фонового файла")
            # Для случая 039_1_14_56_41.mat -> 039_1_bg_14_56_41.mat
            patterns = [
                # Новый паттерн 1: Явно ищем bg после первого числа (для формата 039_1_bg_14_56_41.mat)
                lambda f: re.sub(r'(\d+_\d+)_', r'\1_bg_', f),
                
                # Новый паттерн 2: Вставка "_bg" перед временем (для других форматов)
                lambda f: re.sub(r'_(\d{2}_\d{2}_\d{2})', r'_bg_\1', f),
                
                # Старые паттерны оставляем как запасные варианты
                lambda f: re.sub(r'_(\d+)_', r'_\1_bg_', f),  # Замена "_X_" на "_X_bg_"
                lambda f: f.replace(ext, f"_bg{ext}"),  # Замена расширения на "_bg.mat"
                
                # Новый общий паттерн: вставка "bg" перед расширением
                lambda f: base_name + "_bg" + ext,
                
                # Самый простой вариант - просто добавление "bg" после базового имени
                lambda f: os.path.join(directory, base_name + "_bg" + ext)
            ]
            
            # Прямая проверка наиболее вероятных имен файлов с фоном
            logger.debug("Проверка наиболее вероятных имен файлов с фоном:")
            
            # Проверяем возможные имена файлов
            for i, pattern in enumerate(patterns):
                try:
                    # Применяем паттерн и получаем имя потенциального файла с фоном
                    if callable(pattern):
                        bg_filename = pattern(filename)
                    else:
                        bg_filename = pattern  # Если это готовый путь
                        
                    # Если результат - полный путь, используем его
                    if os.path.isabs(bg_filename):
                        bg_filepath = bg_filename
                    else:
                        bg_filepath = os.path.join(directory, bg_filename)
                        
                    logger.debug(f"Проверка шаблона {i+1}: {bg_filepath}")
                    
                    # Если файл существует, возвращаем его путь
                    if os.path.exists(bg_filepath) and os.path.isfile(bg_filepath):
                        logger.info(f"Найден файл с фоном: {bg_filepath}")
                        return bg_filepath
                except Exception as e:
                    logger.debug(f"Ошибка при применении шаблона {i+1}: {e}")
            
            # Ищем в директории файлы с похожими именами
            logger.debug("Поиск файлов с 'bg' в названии в той же директории:")
            for file in os.listdir(directory):
                if file.endswith(ext) and "bg" in file.lower() and base_name.split('_')[0] in file:
                    bg_filepath = os.path.join(directory, file)
                    logger.debug(f"Найден возможный файл с фоном: {bg_filepath}")
                    logger.info(f"Найден файл с фоном: {bg_filepath}")
                    return bg_filepath
            
            # Печатаем сообщение только в debug и warning, но не в консоль,
            # чтобы избежать дублирования в методе import_mat
            logger.warning(f"Файл с фоном для {filepath} не найден")
            return None
        except Exception as e:
            print(f"Ошибка при поиске файла с фоном: {e}")
            logger.error(f"Ошибка при поиске файла с фоном: {e}", exc_info=True)
            return None
            
    @staticmethod
    def _determine_camera_by_shape(shape):
        """
        Определяет камеру по размеру массива
        
        Args:
            shape: Размер массива
            
        Returns:
            dict: Информация о камере
        """
        logger.debug(f"Определение камеры по размеру массива: {shape}")
        # Информация о известных камерах
        cameras = {
            "GUN_YAG1": {
                "resolution": (1032, 776),
                "pixel_size_x": 0.02840909,
                "pixel_size_y": 0.02840909,
                "name": "GUN_YAG1"
            },
            "GUN_YAG2": {
                "resolution": (1624, 1224),
                "pixel_size_x": 0.01875468,
                "pixel_size_y": 0.01875468,
                "name": "GUN_YAG2"
            }
        }
        
        # Проверяем соответствие размеров
        for camera_name, info in cameras.items():
            if shape == info["resolution"] or shape == (info["resolution"][1], info["resolution"][0]):
                logger.debug(f"Определена известная камера: {camera_name}")
                return info
                
        # Если не нашли соответствие, возвращаем значения по умолчанию
        logger.warning(f"Не удалось определить камеру по размеру {shape}. Используются значения по умолчанию.")
        return {
            "resolution": shape,
            "pixel_size_x": 1.0,
            "pixel_size_y": 1.0,
            "name": "Неизвестная камера"
        }

    @staticmethod
    def _normalize_array(array):
        """
        Нормализует массив в диапазон [0, 1]
        
        Args:
            array: Массив для нормализации
            
        Returns:
            numpy.ndarray: Нормализованный массив
        """
        logger.debug(f"Нормализация массива размером {array.shape if hasattr(array, 'shape') else 'неизвестно'}")
        if array is None or array.size == 0:
            logger.warning("Пустой массив на входе. Возвращается нулевой массив.")
            return np.zeros((1, 1), dtype=np.float64)
            
        # Преобразование в тип float64 для большей точности
        array_float = array.astype(np.float64)
        
        # Находим минимальное и максимальное значение
        min_val = np.min(array_float)
        max_val = np.max(array_float)
        logger.debug(f"Диапазон значений в массиве: [{min_val}, {max_val}]")
        
        # Избегаем деления на ноль
        if max_val == min_val:
            logger.warning("Массив содержит одинаковые значения. Возвращается нулевой массив.")
            return np.zeros_like(array_float, dtype=np.float64)
        
        # Нормализация в диапазон [0, 1]
        normalized = (array_float - min_val) / (max_val - min_val)
        logger.debug("Нормализация массива завершена успешно")
        
        return normalized

    @staticmethod
    def import_data(filepath):
        """
        Импортирует данные из файла на основе его расширения
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info(f"Импорт данных из файла: {filepath}")
            # Определяем формат по расширению
            _, ext = os.path.splitext(filepath)
            logger.debug(f"Определено расширение файла: {ext}")
            
            if ext.lower() == ".mat":
                logger.debug("Выбран импорт MAT файла")
                return DataImporter.import_mat(filepath)
            else:
                print(f"Неподдерживаемый формат файла: {ext}")
                logger.error(f"Неподдерживаемый формат файла: {ext}")
                return None
        except Exception as e:
            print(f"Ошибка при импорте данных: {e}")
            logger.error(f"Ошибка при импорте данных: {e}", exc_info=True)
            return None
