import os
import numpy as np
import re
import scipy.io as sio
import pandas as pd
import json

import logging
logger = logging.getLogger('data')

from src.analysis.normalizer import ImageNormalizer

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
            logger.info("Начало импорта MAT файла: {}".format(filepath))
            # Загружаем основной файл
            logger.debug("Загрузка данных из основного файла: {}".format(filepath))
            shot_data = sio.loadmat(filepath)
            
            # Проверяем наличие данных в файле
            if not shot_data:
                print("Файл {} не содержит данных".format(filepath))
                logger.error("Файл {} не содержит данных".format(filepath))
                return None
                
            # Получаем имя переменной с данными (обычно первая переменная, не начинающаяся с "__")
            logger.debug("Поиск переменных с данными в файле")
            shot_var_names = [key for key in shot_data.keys() if not key.startswith("__")]
            
            if not shot_var_names:
                print("Не удалось найти данные в файле {}".format(filepath))
                logger.error("Не удалось найти данные в файле {}".format(filepath))
                return None
            
            # Извлекаем первую переменную, которая не начинается с "__"
            shot_var_name = shot_var_names[0]
            logger.debug("Найдены переменные в файле: {}, выбрана первая: {}".format(shot_var_names, shot_var_name))
                
            # Получаем данные снимка
            logger.debug("Получение данных снимка с использованием переменной: {}".format(shot_var_name))
            shot = shot_data[shot_var_name]
            
            # Если путь к файлу с фоном не указан, пытаемся найти его автоматически
            if background_filepath is None:
                logger.info("Путь к файлу с фоном не указан, выполняется автоматический поиск")
                background_filepath = DataImporter._find_background_file(filepath)
                
            # Если файл с фоном не найден
            if background_filepath is None:
                # Выводим сообщение один раз, а не в каждой итерации поиска файла
                print("Файл с фоном для {} не найден".format(filepath))
                logger.error("Файл с фоном для {} не найден".format(filepath))
                return None
                
            # Загружаем файл с фоном
            logger.debug("Загрузка файла с фоном: {}".format(background_filepath))
            bg_data = sio.loadmat(background_filepath)
            
            # Получаем имя переменной с данными фона
            logger.debug("Поиск переменных с данными фона в файле")
            bg_var_names = [key for key in bg_data.keys() if not key.startswith("__")]
            
            if not bg_var_names:
                print("Не удалось найти данные в файле {}".format(background_filepath))
                logger.error("Не удалось найти данные в файле {}".format(background_filepath))
                return None
                
            # Извлекаем первую переменную фона, которая не начинается с "__"
            bg_var_name = bg_var_names[0]
            logger.debug("Найдены переменные в файле фона: {}, выбрана первая: {}".format(bg_var_names, bg_var_name))
                
            # Получаем данные фона
            logger.debug("Получение данных фона с использованием переменной: {}".format(bg_var_name))
            background = bg_data[bg_var_name]
            
            # Проверяем совпадение размеров
            if shot.shape != background.shape:
                print("Размеры снимка ({}) и фона ({}) не совпадают".format(shot.shape, background.shape))
                logger.error("Размеры снимка ({}) и фона ({}) не совпадают".format(shot.shape, background.shape))
                return None
                
            # Определяем камеру по размеру массива
            logger.debug("Определение камеры по размеру массива: {}".format(shot.shape))
            camera_info = DataImporter._determine_camera_by_shape(shot.shape)
            logger.info("Определена камера: {}".format(camera_info['name']))
            
            # Вычитаем фон если он есть
            if background is not None:
                # Проверяем совпадение размеров
                if shot.shape == background.shape:
                    # Сначала вычитаем необработанный фон, затем нормализуем
                    raw_difference = shot - background
                    raw_difference[raw_difference < 0] = 0
                    difference = ImageNormalizer.normalize(raw_difference)
                else:
                    print("Размеры снимка и фона не совпадают")
                    difference = None
            
            # Нормализуем снимок и фон для возврата
            normalized_shot = ImageNormalizer.normalize(shot)
            normalized_background = ImageNormalizer.normalize(background) if background is not None else None
            
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
            
            logger.info("Успешно импортирован MAT файл: {}".format(filepath))
            return result
        except Exception as e:
            print("Ошибка при импорте MAT файла: {}".format(e))
            logger.error("Ошибка при импорте MAT файла: {}".format(e), exc_info=True)
            return None
    
    @staticmethod
    def import_csv(filepath, background_filepath=None):
        """
        Импортирует данные из CSV файла
        
        Args:
            filepath: Путь к файлу
            background_filepath: Путь к файлу с фоном (если None, пытается найти автоматически)
            
        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info("Начало импорта CSV файла: {}".format(filepath))
            
            # Читаем файл построчно для разбора метаданных и данных
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Словарь для хранения метаданных и данных
            result_data = {}
            
            # Извлекаем метаданные
            metadata_mode = False
            data_mode = False
            current_data_key = None
            current_data_rows = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Проверяем маркеры секций
                if line.startswith("# Metadata"):
                    metadata_mode = True
                    data_mode = False
                    continue
                elif line.startswith("# Data"):
                    metadata_mode = False
                    data_mode = True
                    continue
                elif line.startswith("# End of"):
                    # Закончилась секция данных
                    if current_data_key and current_data_rows:
                        try:
                            # Преобразуем строки в numpy массив
                            data_array = np.array([[float(val) for val in row.split(',')] 
                                                for row in current_data_rows])
                            result_data[current_data_key] = data_array
                        except Exception as e:
                            logger.warning("Ошибка при преобразовании данных {}: {}".format(current_data_key, e))
                    
                    current_data_key = None
                    current_data_rows = []
                    continue
                
                # Обрабатываем метаданные
                if metadata_mode and line.startswith("# "):
                    parts = line[2:].split(',', 1)  # Разделяем только по первой запятой
                    if len(parts) == 2:
                        key, value = parts
                        key = key.strip()
                        value = value.strip()
                        
                        # Если значение в кавычках, удаляем их
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        
                        # Пробуем преобразовать строковые значения в соответствующие типы
                        try:
                            if value.startswith('(') and value.endswith(')'):
                                # Это кортеж
                                value = eval(value)
                            elif value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            else:
                                try:
                                    value = float(value)
                                    if value.is_integer():
                                        value = int(value)
                                except ValueError:
                                    pass  # Оставляем как строку
                        except Exception as e:
                            logger.warning("Ошибка при преобразовании значения {}: {}".format(value, e))
                        
                        result_data[key] = value
                
                # Обрабатываем заголовки данных
                elif data_mode and line.startswith("# "):
                    parts = line[2:].split(',', 1)
                    if len(parts) == 2:
                        current_data_key = parts[0].strip()
                        current_data_rows = []  # Сбрасываем буфер строк
                
                # Обрабатываем строки данных (не начинающиеся с #)
                elif not line.startswith("#") and current_data_key:
                    current_data_rows.append(line)
            
            # Обрабатываем последний блок данных, если он есть и не был завершен
            if current_data_key and current_data_rows:
                try:
                    data_array = np.array([[float(val) for val in row.split(',')] 
                                        for row in current_data_rows])
                    result_data[current_data_key] = data_array
                except Exception as e:
                    logger.warning("Ошибка при преобразовании данных {}: {}".format(current_data_key, e))
            
            # Проверяем наличие необходимых данных
            if 'shot' not in result_data or not isinstance(result_data['shot'], np.ndarray):
                # Если нет ключа shot, но есть данные, пробуем использовать их
                data_keys = [k for k, v in result_data.items() if isinstance(v, np.ndarray)]
                if data_keys:
                    result_data['shot'] = result_data[data_keys[0]]
                    logger.info("Использование {} в качестве shot".format(data_keys[0]))
                else:
                    # Если не удалось найти данные, пробуем прочитать как обычный CSV
                    logger.warning("Не найдены структурированные данные, пробуем прочитать как обычный CSV")
                    try:
                        data = pd.read_csv(filepath, header=None, comment='#').values
                        result_data['shot'] = data
                    except Exception as e:
                        logger.error("Ошибка при чтении CSV как обычного файла: {}".format(e))
                        return None
            
            # Если в результате нет фона, но указан путь к файлу с фоном, пробуем загрузить его
            if 'background' not in result_data and background_filepath:
                try:
                    bg_data = DataImporter.import_csv(background_filepath)
                    if bg_data and 'shot' in bg_data:
                        result_data['background'] = bg_data['shot']
                except Exception as e:
                    logger.error("Ошибка при загрузке фона: {}".format(e))
            
            # Если всё еще нет фона, но есть shot, создаем нулевой фон
            if 'background' not in result_data and 'shot' in result_data:
                result_data['background'] = np.zeros_like(result_data['shot'])
            
            # Проверяем формат и размеры данных
            if 'shot' in result_data and 'background' in result_data:
                shot = result_data['shot']
                background = result_data['background']
                
                # Проверяем совпадение размеров
                if shot.shape != background.shape:
                    logger.warning("Размеры снимка ({}) и фона ({}) не совпадают".format(shot.shape, background.shape))
                    background = np.zeros_like(shot)
                    result_data['background'] = background
                
                # Определяем камеру по размеру массива если не указана
                if 'camera_name' not in result_data or 'resolution' not in result_data:
                    logger.debug("Определение камеры по размеру массива: {}".format(shot.shape))
                    camera_info = DataImporter._determine_camera_by_shape(shot.shape)
                    result_data.update(camera_info)
                
                # Вычитаем фон если он есть
                if background is not None:
                    # Проверяем совпадение размеров
                    if shot.shape == background.shape:
                        # Сначала вычитаем необработанный фон, затем нормализуем
                        raw_difference = shot - background
                        raw_difference[raw_difference < 0] = 0
                        difference = ImageNormalizer.normalize(raw_difference)
                    else:
                        print("Размеры снимка и фона не совпадают")
                        difference = None
                
                # Нормализуем снимок и фон для возврата
                normalized_shot = ImageNormalizer.normalize(shot)
                normalized_background = ImageNormalizer.normalize(background) if background is not None else None
                
                # Добавляем нормализованные данные и разницу
                result_data['shot'] = normalized_shot
                result_data['background'] = normalized_background
                result_data['difference'] = difference
                result_data['raw_shot'] = shot
                result_data['raw_background'] = background
                
                # Добавляем эти поля, чтобы избежать KeyError в UI
                result_data['current_frame'] = normalized_shot
                result_data['centroid'] = (0, 0)
                result_data['rms'] = (0, 0)
                result_data['filepath'] = filepath
                
                logger.info("Успешно импортирован CSV файл: {}".format(filepath))
                return result_data
            else:
                logger.error("Не удалось извлечь необходимые данные из CSV файла")
                return None
                
        except Exception as e:
            print("Ошибка при импорте CSV файла: {}".format(e))
            logger.error("Ошибка при импорте CSV файла: {}".format(e), exc_info=True)
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
            logger.debug("Поиск файла с фоном для: {}".format(filepath))
            # Получаем директорию и имя файла
            directory = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            base_name, ext = os.path.splitext(filename)
            
            logger.debug("Исходный файл: директория={}, имя={}, базовое имя={}, расширение={}".format(directory, filename, base_name, ext))
            
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
                lambda f: f.replace(ext, "_bg{}".format(ext)),  # Замена расширения на "_bg.mat"
                
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
                        
                    logger.debug("Проверка шаблона {}: {}".format(i+1, bg_filepath))
                    
                    # Если файл существует, возвращаем его путь
                    if os.path.exists(bg_filepath) and os.path.isfile(bg_filepath):
                        logger.info("Найден файл с фоном: {}".format(bg_filepath))
                        return bg_filepath
                except Exception as e:
                    logger.debug("Ошибка при применении шаблона {}: {}".format(i+1, e))
            
            # Ищем в директории файлы с похожими именами
            logger.debug("Поиск файлов с 'bg' в названии в той же директории:")
            for file in os.listdir(directory):
                if file.endswith(ext) and "bg" in file.lower() and base_name.split('_')[0] in file:
                    bg_filepath = os.path.join(directory, file)
                    logger.debug("Найден возможный файл с фоном: {}".format(bg_filepath))
                    logger.info("Найден файл с фоном: {}".format(bg_filepath))
                    return bg_filepath
            
            # Печатаем сообщение только в debug и warning, но не в консоль,
            # чтобы избежать дублирования в методе import_mat
            logger.warning("Файл с фоном для {} не найден".format(filepath))
            return None
        except Exception as e:
            print("Ошибка при поиске файла с фоном: {}".format(e))
            logger.error("Ошибка при поиске файла с фоном: {}".format(e), exc_info=True)
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
        logger.debug("Определение камеры по размеру массива: {}".format(shape))
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
                logger.debug("Определена известная камера: {}".format(camera_name))
                return info
                
        # Если не нашли соответствие, возвращаем значения по умолчанию
        logger.warning("Не удалось определить камеру по размеру {}. Используются значения по умолчанию.".format(shape))
        return {
            "resolution": shape,
            "pixel_size_x": 1.0,
            "pixel_size_y": 1.0,
            "name": "Неизвестная камера"
        }

    @staticmethod
    def is_data_folder(path):
        """
        Проверяет, является ли указанный путь папкой с данными анализатора профиля пучка
        
        Args:
            path: Путь к папке или файлу
            
        Returns:
            bool: True если это папка с данными, иначе False
        """
        # Если путь указывает на файл, проверяем его папку
        if os.path.isfile(path) and os.path.basename(path) == 'metadata.json':
            path = os.path.dirname(path)
        
        # Проверяем, является ли путь папкой
        if not os.path.isdir(path):
            return False
            
        # Проверяем наличие метаданных
        metadata_path = os.path.join(path, 'metadata.json')
        if not os.path.isfile(metadata_path):
            return False
            
        # Проверяем наличие хотя бы одного из основных CSV файлов
        expected_files = ['shot.csv', 'background.csv', 'difference.csv']
        found = False
        for filename in expected_files:
            if os.path.isfile(os.path.join(path, filename)):
                found = True
                break
                
        return found
    
    @staticmethod
    def import_folder(folder_path):
        """
        Импортирует данные из папки с метаданными и CSV файлами
        
        Args:
            folder_path: Путь к папке с данными
            
        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        # Если путь указывает на файл metadata.json, берем его директорию
        if os.path.isfile(folder_path) and os.path.basename(folder_path) == 'metadata.json':
            metadata_filepath = folder_path
            folder_path = os.path.dirname(folder_path)
        else:
            # Проверяем, что это папка с данными
            if not DataImporter.is_data_folder(folder_path):
                print(f"Указанный путь не является папкой с данными: {folder_path}")
                logger.error(f"Указанный путь не является папкой с данными: {folder_path}")
                return None
                
            metadata_filepath = os.path.join(folder_path, 'metadata.json')
        
        return DataImporter.import_multifile_csv(metadata_filepath)
    
    @staticmethod
    def import_multifile_csv(metadata_filepath):
        """
        Импортирует данные из набора файлов в папке: JSON метаданные и чистые CSV файлы для массивов
        
        Args:
            metadata_filepath: Путь к JSON файлу с метаданными
            
        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info("Начало импорта данных из папки: {}".format(metadata_filepath))
            
            if not os.path.exists(metadata_filepath):
                print(f"Файл метаданных не найден: {metadata_filepath}")
                logger.error(f"Файл метаданных не найден: {metadata_filepath}")
                return None
                
            # Загружаем метаданные из JSON
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
                
            # Определяем папку, в которой находятся файлы
            folder_path = os.path.dirname(metadata_filepath)
            
            # Если в метаданных есть информация о массивах, используем её
            expected_arrays = []
            if 'arrays' in result_data:
                expected_arrays = list(result_data['arrays'].keys())
            else:
                # Иначе используем стандартный набор
                expected_arrays = ['shot', 'background', 'difference']
            
            for array_name in expected_arrays:
                array_path = os.path.join(folder_path, f"{array_name}.csv")
                
                if not os.path.exists(array_path):
                    logger.warning(f"Файл массива не найден: {array_path}")
                    continue
                    
                # Импортируем массив из CSV (только числа, без заголовков)
                try:
                    # Читаем файл построчно
                    with open(array_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Получаем форму массива из метаданных, если есть
                    shape = None
                    if 'arrays' in result_data and array_name in result_data['arrays']:
                        shape_str = result_data['arrays'][array_name]['shape']
                        if isinstance(shape_str, list):
                            shape = tuple(shape_str)
                        elif isinstance(shape_str, str):
                            # Преобразуем строку '(rows, cols)' в кортеж
                            shape_match = re.search(r'\((\d+), (\d+)\)', shape_str)
                            if shape_match:
                                shape = (int(shape_match.group(1)), int(shape_match.group(2)))
                    
                    # Убираем пустые строки
                    data_rows = [line.strip() for line in lines if line.strip()]
                    
                    # Преобразуем строки в numpy массив
                    if data_rows:
                        try:
                            # Сначала пытаемся разделить по запятой (CSV)
                            data_array = np.array([[float(val) for val in row.split(',')] 
                                                for row in data_rows])
                            
                            # Если указана форма и текущая форма не соответствует, пробуем изменить
                            if shape and data_array.shape != shape:
                                try:
                                    # Попробуем преобразовать массив к правильной форме
                                    data_array = data_array.flatten().reshape(shape)
                                except:
                                    logger.warning(f"Не удалось преобразовать массив {array_name} к форме {shape}")
                            
                            result_data[array_name] = data_array
                            logger.info(f"Успешно импортирован массив: {array_name}, форма: {data_array.shape}")
                        except ValueError:
                            # Если не получилось, пробуем по пробелам
                            try:
                                data_array = np.array([[float(val) for val in row.split()] 
                                                    for row in data_rows])
                                
                                # Проверяем форму
                                if shape and data_array.shape != shape:
                                    try:
                                        data_array = data_array.flatten().reshape(shape)
                                    except:
                                        logger.warning(f"Не удалось преобразовать массив {array_name} к форме {shape}")
                                
                                result_data[array_name] = data_array
                                logger.info(f"Успешно импортирован массив: {array_name}, форма: {data_array.shape}")
                            except Exception as e:
                                logger.error(f"Не удалось разобрать данные в файле {array_path}: {e}")
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла {array_path}: {e}")
            
            # Проверяем наличие необходимых массивов
            if 'shot' not in result_data or 'background' not in result_data:
                logger.error("Не найдены необходимые массивы данных (shot и/или background)")
                return None
                
            # Проверяем совпадение размеров массивов
            if result_data['shot'].shape != result_data['background'].shape:
                logger.error(f"Размеры массивов shot и background не совпадают: "
                           f"{result_data['shot'].shape} vs {result_data['background'].shape}")
                return None
                
            # Добавляем разность, если её нет
            if 'difference' not in result_data:
                difference = result_data['shot'] - result_data['background']
                difference[difference < 0] = 0
                result_data['difference'] = difference
                logger.info("Автоматически рассчитана разность shot - background")
            
            # Добавляем дополнительные поля, если их нет
            if 'current_frame' not in result_data:
                result_data['current_frame'] = result_data['shot']
            if 'centroid' not in result_data:
                result_data['centroid'] = (0, 0)
            if 'rms' not in result_data:
                result_data['rms'] = (0, 0)
            if 'filepath' not in result_data:
                result_data['filepath'] = folder_path
            
            logger.info(f"Успешно импортированы данные из папки: {folder_path}")
            return result_data
            
        except Exception as e:
            print(f"Ошибка при импорте данных из JSON и CSV файлов: {e}")
            logger.error(f"Ошибка при импорте данных из JSON и CSV файлов: {e}", exc_info=True)
            return None
            
    @staticmethod
    def import_data(filepath):
        """
        Импортирует данные из MAT-файла или папки с данными
        
        Args:
            filepath: Путь к файлу MAT или папке с данными
            
        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        if not os.path.exists(filepath):
            print("Файл или папка не найдены: {}".format(filepath))
            logger.error("Файл или папка не найдены: {}".format(filepath))
            return None
        
        # Проверяем, является ли путь папкой с данными
        if os.path.isdir(filepath) and DataImporter.is_data_folder(filepath):
            print(f"Импорт данных из папки: {filepath}")
            logger.info(f"Импорт данных из папки: {filepath}")
            return DataImporter.import_folder(filepath)
            
        # Проверяем, указывает ли путь на metadata.json в папке с данными
        if os.path.isfile(filepath) and os.path.basename(filepath) == "metadata.json":
            print(f"Импорт данных из метафайла: {filepath}")
            logger.info(f"Импорт данных из метафайла: {filepath}")
            return DataImporter.import_multifile_csv(filepath)
            
        # Если это MAT-файл
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext == '.mat':
            print(f"Импорт данных из MAT файла: {filepath}")
            logger.info(f"Импорт данных из MAT файла: {filepath}")
            return DataImporter.import_mat(filepath)
        else:
            print(f"Неподдерживаемый формат файла: {filepath}")
            logger.error(f"Неподдерживаемый формат файла: {filepath}")
            return None
