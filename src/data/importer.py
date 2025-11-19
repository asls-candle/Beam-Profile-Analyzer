import os
import numpy as np
import re
import scipy.io as sio
import pandas as pd
import json

from src.analysis.normalizer import ImageNormalizer

import logging
logger = logging.getLogger('data')


# from src.analysis.filters import apply_median_filter

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
            logger.debug("Variavle names: {}".format(shot_data.keys()))
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

            # Максимальное значение для 16-битной камеры
            bit_depth_max = 65535.0

            # Вычитаем фон если он есть
            if background is not None:
                # Проверяем совпадение размеров
                if shot.shape == background.shape:
                    # Сначала вычитаем необработанный фон
                    raw_difference = shot - background
                    raw_difference[raw_difference < 0] = 0
                    # Нормализуем difference относительно максимума разрядности
                    difference = raw_difference / bit_depth_max
                    # Применяем медианный фильтр к разностному изображению
                    # difference = apply_median_filter(difference, kernel_size=3)
                else:
                    print("Размеры снимка и фона не совпадают")
                    difference = None

            # Нормализуем снимок и фон от 0 до максимального значения разрядности
            normalized_shot = shot / bit_depth_max
            normalized_background = background / bit_depth_max if background is not None else None

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

                # Максимальное значение для 16-битной камеры
                bit_depth_max = 65535.0

                # Вычитаем фон если он есть
                if background is not None:
                    # Проверяем совпадение размеров
                    if shot.shape == background.shape:
                        # Сначала вычитаем необработанный фон
                        raw_difference = shot - background
                        raw_difference[raw_difference < 0] = 0
                        # Нормализуем difference относительно максимума разрядности
                        difference = raw_difference / bit_depth_max
                        # Применяем медианный фильтр к разностному изображению
                        # difference = apply_median_filter(difference, kernel_size=3)
                    else:
                        print("Размеры снимка и фона не совпадают")
                        difference = None

                # Нормализуем снимок и фон от 0 до максимального значения разрядности
                normalized_shot = shot / bit_depth_max
                normalized_background = background / bit_depth_max if background is not None else None

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

            logger.debug("Исходный файл: директория={}, имя={}, базовое имя={}, расширение={}".format(
                directory, filename, base_name, ext))

            # Шаблоны для поиска фонового файла
            logger.debug("Применение шаблонов для поиска фонового файла")

            # Список возможных имен файлов с фоном (в порядке приоритета)
            possible_bg_names = [
                # 1. Самый простой и распространенный: base_name + "_bg" + ext
                # Например: Q_0.03A_S_8.4A_E_3.6MeV_Q_220PC_1.mat -> Q_0.03A_S_8.4A_E_3.6MeV_Q_220PC_1_bg.mat
                base_name + "_bg" + ext,

                # 2. Для формата 039_1_14_56_41.mat -> 039_1_bg_14_56_41.mat
                re.sub(r'(\d+_\d+)_(\d{2}_\d{2}_\d{2})', r'\1_bg_\2', filename),

                # 3. Вставка "_bg" перед временем
                re.sub(r'_(\d{2}_\d{2}_\d{2})', r'_bg_\1', filename),

                # 4. Замена "_X_" на "_X_bg_"
                re.sub(r'_(\d+)_', r'_\1_bg_', filename),

                # 5. Замена расширения на "_bg.ext"
                filename.replace(ext, "_bg{}".format(ext)),
            ]

            # Проверяем каждое возможное имя файла
            for i, bg_filename in enumerate(possible_bg_names, 1):
                # Пропускаем если это то же имя, что и исходный файл
                if bg_filename == filename:
                    continue

                bg_filepath = os.path.join(directory, bg_filename)
                logger.debug("Проверка варианта {}: {}".format(i, bg_filepath))

                # Если файл существует, возвращаем его путь
                if os.path.exists(bg_filepath) and os.path.isfile(bg_filepath):
                    logger.info("Найден файл с фоном: {}".format(bg_filepath))
                    return bg_filepath

            # Ищем в директории файлы с похожими именами
            logger.debug("Поиск файлов с 'bg' в названии в той же директории:")
            try:
                for file in os.listdir(directory):
                    if file == filename:  # Пропускаем сам исходный файл
                        continue

                    if file.endswith(ext) and "bg" in file.lower():
                        # Проверяем, что это похожий файл (имеет общую часть имени)
                        # Берем первую часть имени до первого "_" или весь base_name
                        file_base = os.path.splitext(file)[0]

                        # Простая эвристика: если в имени bg-файла есть существенная часть
                        # исходного имени, считаем его подходящим
                        common_part = base_name.split('_')[0] if '_' in base_name else base_name[:10]

                        if common_part and common_part in file_base:
                            bg_filepath = os.path.join(directory, file)
                            logger.debug("Найден возможный файл с фоном: {}".format(bg_filepath))
                            logger.info("Найден файл с фоном: {}".format(bg_filepath))
                            return bg_filepath
            except Exception as e:
                logger.debug("Ошибка при поиске в директории: {}".format(e))

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
                print("Указанный путь не является папкой с данными: {}".format(folder_path))
                logger.error("Указанный путь не является папкой с данными: {}".format(folder_path))
                return None

            metadata_filepath = os.path.join(folder_path, 'metadata.json')

        return DataImporter.import_multifile_csv(metadata_filepath)

    @staticmethod
    def _load_csv_array(array_path, expected_shape=None):
        """
        Загружает массив из CSV файла

        Args:
            array_path: Путь к CSV файлу
            expected_shape: Ожидаемая форма массива (опционально)

        Returns:
            numpy.ndarray или None в случае ошибки
        """
        try:
            # Читаем файл построчно
            with open(array_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Убираем пустые строки
            data_rows = [line.strip() for line in lines if line.strip()]

            if not data_rows:
                logger.warning("Файл {} пуст".format(array_path))
                return None

            # Преобразуем строки в numpy массив
            try:
                # Пытаемся разделить по запятой (CSV)
                data_array = np.array([[float(val) for val in row.split(',')]
                                    for row in data_rows])
            except ValueError:
                # Если не получилось, пробуем по пробелам
                try:
                    data_array = np.array([[float(val) for val in row.split()]
                                        for row in data_rows])
                except Exception as e:
                    logger.error("Не удалось разобрать данные в файле {}: {}".format(array_path, e))
                    return None

            # Если указана ожидаемая форма и текущая форма не соответствует, пробуем изменить
            if expected_shape and data_array.shape != expected_shape:
                try:
                    data_array = data_array.flatten().reshape(expected_shape)
                except Exception as e:
                    logger.warning("Не удалось преобразовать массив к форме {}: {}".format(expected_shape, e))

            logger.info("Успешно загружен массив из {}, форма: {}".format(array_path, data_array.shape))
            return data_array

        except Exception as e:
            logger.error("Ошибка при чтении файла {}: {}".format(array_path, e))
            return None

    @staticmethod
    def _compute_processed_data(raw_shot, raw_background):
        """
        Вычисляет обработанные данные из сырых массивов

        Args:
            raw_shot: Сырой массив снимка
            raw_background: Сырой массив фона

        Returns:
            dict: Словарь с обработанными данными (shot, background, difference)
        """
        # Вычисляем разность
        raw_difference = raw_shot - raw_background
        raw_difference[raw_difference < 0] = 0

        # Максимальное значение для 16-битной камеры
        bit_depth_max = 65535.0

        # Нормализуем данные от 0 до максимального значения разрядности
        # Это сохраняет абсолютные значения светимости: 0 света = 0, 65535 = 1
        shot = raw_shot / bit_depth_max
        background = raw_background / bit_depth_max

        # difference нормализуется относительно максимума разрядности
        difference = raw_difference / bit_depth_max

        return {
            'shot': shot,
            'background': background,
            'difference': difference
        }

    @staticmethod
    def import_multifile_csv(metadata_filepath):
        """
        Импортирует данные из набора файлов в папке: JSON метаданные и чистые CSV файлы для массивов

        Поддерживает два варианта данных:
        1. Полные данные: shot.csv, background.csv, difference.csv, raw_shot.csv, raw_background.csv
        2. Неполные данные: только raw_shot.csv и raw_background.csv (обработанные вычисляются автоматически)

        Args:
            metadata_filepath: Путь к JSON файлу с метаданными

        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info("Начало импорта данных из папки: {}".format(metadata_filepath))

            if not os.path.exists(metadata_filepath):
                print("Файл метаданных не найден: {}".format(metadata_filepath))
                logger.error("Файл метаданных не найден: {}".format(metadata_filepath))
                return None

            # Загружаем метаданные из JSON
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Определяем папку, в которой находятся файлы
            folder_path = os.path.dirname(metadata_filepath)

            # Определяем, какие массивы нужно загрузить из метаданных
            arrays_info = metadata.get('arrays', {})
            available_arrays = list(arrays_info.keys())

            logger.info("Доступные массивы по метаданным: {}".format(available_arrays))

            # Загружаем все доступные массивы
            loaded_arrays = {}
            for array_name in available_arrays:
                array_path = os.path.join(folder_path, "{}.csv".format(array_name))

                if not os.path.exists(array_path):
                    logger.warning("Файл массива не найден: {}".format(array_path))
                    continue

                # Получаем ожидаемую форму из метаданных
                expected_shape = None
                if 'shape' in arrays_info[array_name]:
                    shape_data = arrays_info[array_name]['shape']
                    if isinstance(shape_data, list):
                        expected_shape = tuple(shape_data)

                # Загружаем массив
                array_data = DataImporter._load_csv_array(array_path, expected_shape)
                if array_data is not None:
                    loaded_arrays[array_name] = array_data

            # Определяем, какой набор данных у нас есть
            has_full_data = ('shot' in loaded_arrays and 'background' in loaded_arrays)
            has_raw_data = ('raw_shot' in loaded_arrays and 'raw_background' in loaded_arrays)

            logger.info("Проверка наличия данных: полные={}, сырые={}".format(has_full_data, has_raw_data))

            # Подготавливаем результат с метаданными
            result_data = {
                'resolution': metadata.get('resolution'),
                'pixel_size_x': metadata.get('pixel_size_x'),
                'pixel_size_y': metadata.get('pixel_size_y'),
                'camera_name': metadata.get('camera_name'),
                'date': metadata.get('date')
            }

            if has_full_data:
                # Вариант 1: Есть полные обработанные данные
                logger.info("Обнаружены полные данные, используем их напрямую")

                result_data['shot'] = loaded_arrays['shot']
                result_data['background'] = loaded_arrays['background']
                result_data['difference'] = loaded_arrays.get('difference')
                result_data['raw_shot'] = loaded_arrays.get('raw_shot')
                result_data['raw_background'] = loaded_arrays.get('raw_background')

                # Если нет difference, но есть shot и background, вычисляем
                if result_data['difference'] is None:
                    logger.info("Файл difference.csv не найден, вычисляем разность")
                    raw_diff = result_data['shot'] - result_data['background']
                    raw_diff[raw_diff < 0] = 0
                    result_data['difference'] = ImageNormalizer.normalize(raw_diff, reference_image=result_data['shot'])

            elif has_raw_data:
                # Вариант 2: Есть только сырые данные, вычисляем обработанные
                logger.info("Обнаружены только сырые данные, вычисляем обработанные")

                raw_shot = loaded_arrays['raw_shot']
                raw_background = loaded_arrays['raw_background']

                # Проверяем совпадение размеров
                if raw_shot.shape != raw_background.shape:
                    logger.error("Размеры raw_shot ({}) и raw_background ({}) не совпадают".format(
                        raw_shot.shape, raw_background.shape))
                    return None

                # Вычисляем обработанные данные
                processed = DataImporter._compute_processed_data(raw_shot, raw_background)

                result_data['raw_shot'] = raw_shot
                result_data['raw_background'] = raw_background
                result_data['shot'] = processed['shot']
                result_data['background'] = processed['background']
                result_data['difference'] = processed['difference']

            else:
                # Нет ни полных, ни сырых данных
                logger.error("Не найдены необходимые массивы данных. "
                           "Требуются либо (shot и background), либо (raw_shot и raw_background)")
                return None

            # Добавляем дополнительные поля для совместимости с UI
            result_data['current_frame'] = result_data['shot']
            result_data['centroid'] = (0, 0)
            result_data['rms'] = (0, 0)
            result_data['filepath'] = folder_path

            logger.info("Успешно импортированы данные из папки: {}".format(folder_path))
            return result_data

        except Exception as e:
            print("Ошибка при импорте данных из JSON и CSV файлов: {}".format(e))
            logger.error("Ошибка при импорте данных из JSON и CSV файлов: {}".format(e), exc_info=True)
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
            print("Импорт данных из папки: {}".format(filepath))
            logger.info("Импорт данных из папки: {}".format(filepath))
            return DataImporter.import_folder(filepath)

        # Проверяем, указывает ли путь на metadata.json в папке с данными
        if os.path.isfile(filepath) and os.path.basename(filepath) == "metadata.json":
            print("Импорт данных из метафайла: {}".format(filepath))
            logger.info("Импорт данных из метафайла: {}".format(filepath))
            return DataImporter.import_multifile_csv(filepath)

        # Если это MAT-файл
        file_ext = os.path.splitext(filepath)[1].lower()
        if file_ext == '.mat':
            print("Импорт данных из MAT файла: {}".format(filepath))
            logger.info("Импорт данных из MAT файла: {}".format(filepath))
            return DataImporter.import_mat(filepath)
        else:
            print("Неподдерживаемый формат файла: {}".format(filepath))
            logger.error("Неподдерживаемый формат файла: {}".format(filepath))
            return None
