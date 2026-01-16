import os
import numpy as np
import re
import scipy.io as sio
import pandas as pd
import json
from typing import Dict, Tuple, Optional, List, Any

from src.negative_numbers_evaluation import plot_negative_density, evaluate_zeros_in_shot
from src.analysis.normalizer import ImageNormalizer

import logging
logger = logging.getLogger('data')


# Константы для имен файлов и конфигурации
class ImporterConfig:
    """Конфигурация импортера данных"""

    # Имена CSV файлов
    FILENAME_SHOT = 'shot.csv'
    FILENAME_BACKGROUND = 'background.csv'
    FILENAME_DIFFERENCE = 'difference.csv'
    FILENAME_RAW_SHOT = 'raw_shot.csv'
    FILENAME_RAW_BACKGROUND = 'raw_background.csv'
    FILENAME_METADATA = 'metadata.json'

    # Суффиксы для поиска фоновых файлов
    BACKGROUND_SUFFIXES = ['_bg', '_background']

    # Информация о камерах
    CAMERAS = {
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

    # Камера по умолчанию
    DEFAULT_CAMERA = {
        "pixel_size_x": 1.0,
        "pixel_size_y": 1.0,
        "name": "Неизвестная камера"
    }


# from src.analysis.filters import apply_median_filter

class DataImporter:
    """
    Класс для импорта данных профиля пучка
    """

    @staticmethod
    def _validate_array_shape(array, expected_shape=None, array_name="array"):
        # type: (np.ndarray, Optional[Tuple[int, int]], str) -> bool
        """
        Проверяет корректность размера массива

        Args:
            array: Проверяемый массив
            expected_shape: Ожидаемая форма (опционально)
            array_name: Имя массива для логирования

        Returns:
            bool: True если размер корректен
        """
        if array is None:
            logger.error("Массив {} является None".format(array_name))
            return False

        if expected_shape is not None and array.shape != expected_shape:
            logger.error("Размер массива {} ({}) не соответствует ожидаемому ({})".format(
                array_name, array.shape, expected_shape))
            return False

        return True

    @staticmethod
    def _validate_shapes_match(array1, array2, name1="array1", name2="array2"):
        # type: (np.ndarray, np.ndarray, str, str) -> bool
        """
        Проверяет совпадение размеров двух массивов

        Args:
            array1: Первый массив
            array2: Второй массив
            name1: Имя первого массива
            name2: Имя второго массива

        Returns:
            bool: True если размеры совпадают
        """
        if array1.shape != array2.shape:
            logger.error("Размеры {} ({}) и {} ({}) не совпадают".format(
                name1, array1.shape, name2, array2.shape))
            return False
        return True

    @staticmethod
    def _load_csv_with_pandas(csv_path, expected_shape=None):
        # type: (str, Optional[Tuple[int, int]]) -> Optional[np.ndarray]
        """
        Загружает массив из CSV файла используя pandas

        ВАЖНО: CSV файлы содержат сырые значения яркости (8-бит, 0-255)

        Args:
            csv_path: Путь к CSV файлу
            expected_shape: Ожидаемая форма массива (опционально)

        Returns:
            numpy.ndarray или None в случае ошибки
        """
        try:
            if not os.path.exists(csv_path):
                logger.warning("Файл не найден: {}".format(csv_path))
                return None

            logger.debug("Загрузка CSV файла: {}".format(csv_path))

            # Используем pandas для быстрого чтения CSV
            # header=None - нет заголовков, данные начинаются с первой строки
            df = pd.read_csv(csv_path, header=None, dtype=np.float64)

            # Преобразуем DataFrame в numpy массив
            data_array = df.values

            # Проверяем, что получили данные
            if data_array.size == 0:
                logger.warning("Файл {} не содержит данных".format(csv_path))
                return None

            # Если указана ожидаемая форма, пытаемся преобразовать
            if expected_shape is not None:
                if data_array.shape != expected_shape:
                    logger.debug("Форма массива {} не соответствует ожидаемой {}. Попытка reshape.".format(
                        data_array.shape, expected_shape))
                    try:
                        data_array = data_array.flatten().reshape(expected_shape)
                        logger.debug("Успешно преобразован к форме {}".format(expected_shape))
                    except ValueError as e:
                        logger.error("Не удалось преобразовать массив к форме {}: {}".format(
                            expected_shape, e))
                        return None

            logger.info("Успешно загружен CSV массив из {}, форма: {}, диапазон значений: [{:.2f}, {:.2f}]".format(
                csv_path, data_array.shape, data_array.min(), data_array.max()))

            return data_array

        except pd.errors.EmptyDataError:
            logger.error("Файл {} пуст".format(csv_path))
            return None
        except pd.errors.ParserError as e:
            logger.error("Ошибка парсинга CSV файла {}: {}".format(csv_path, e))
            return None
        except Exception as e:
            logger.error("Ошибка при чтении CSV файла {}: {}".format(csv_path, e), exc_info=True)
            return None

    @staticmethod
    def _process_image_data(shot, background):
        # type: (np.ndarray, np.ndarray) -> Dict[str, np.ndarray]
        """
        Обрабатывает данные изображения: нормализует и вычитает фон

        Args:
            shot: Сырой массив снимка (8-бит или нормализованный)
            background: Сырой массив фона (8-бит или нормализованный)

        Returns:
            dict: Словарь с обработанными данными (shot, background, difference)
        """
        logger.debug("Обработка данных изображения")

        # Проверяем совпадение размеров
        if not DataImporter._validate_shapes_match(shot, background, "shot", "background"):
            raise ValueError("Размеры shot и background не совпадают")

        # Нормализуем данные используя ImageNormalizer (как MATLAB im2double)
        normalized_shot = ImageNormalizer.normalize(shot)
        normalized_background = ImageNormalizer.normalize(background)

        # Вычитаем нормализованный фон из нормализованного shot
        difference = normalized_shot - normalized_background

        # Логируем статистику отрицательных значений
        negative_count = np.sum(difference < 0)
        negative_percent = 100.0 * negative_count / difference.size
        logger.info("Количество отрицательных значений в difference: {} из {} ({:.2f}%)".format(
            negative_count, difference.size, negative_percent))

        # ЗАКОММЕНТИРОВАНО: обнуление отрицательных значений (под вопросом)
        # difference[difference < 0] = 0

        return {
            'shot': normalized_shot,
            'background': normalized_background,
            'difference': difference
        }

    @staticmethod
    def _create_result_dict(shot, background, difference, raw_shot, raw_background,
                           camera_info, filepath):
        # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], str) -> Dict[str, Any]
        """
        Формирует результирующий словарь с данными

        Args:
            shot: Нормализованный снимок
            background: Нормализованный фон
            difference: Разностное изображение
            raw_shot: Сырой снимок
            raw_background: Сырой фон
            camera_info: Информация о камере
            filepath: Путь к файлу

        Returns:
            dict: Результирующий словарь
        """
        return {
            "shot": shot,
            "background": background,
            "difference": difference,
            "raw_shot": raw_shot,
            "raw_background": raw_background,
            "resolution": camera_info.get("resolution", shot.shape),
            "pixel_size_x": camera_info["pixel_size_x"],
            "pixel_size_y": camera_info["pixel_size_y"],
            "camera_name": camera_info["name"],
            # Добавляем эти поля, чтобы избежать KeyError в UI
            "current_frame": shot,
            "centroid": (0, 0),
            "rms": (0, 0),
            "filepath": filepath
        }

    @staticmethod
    def _load_mat_variable(mat_filepath, var_type="shot"):
        # type: (str, str) -> Optional[np.ndarray]
        """
        Загружает переменную из MAT файла

        Args:
            mat_filepath: Путь к MAT файлу
            var_type: Тип переменной для логирования ('shot' или 'background')

        Returns:
            numpy.ndarray или None в случае ошибки
        """
        try:
            logger.debug("Загрузка {} из MAT файла: {}".format(var_type, mat_filepath))
            mat_data = sio.loadmat(mat_filepath)

            # Проверяем наличие данных
            if not mat_data:
                logger.error("MAT файл {} не содержит данных".format(mat_filepath))
                return None

            # Получаем имена переменных (исключая системные с "__")
            var_names = [key for key in mat_data.keys() if not key.startswith("__")]

            if not var_names:
                logger.error("Не найдено переменных в MAT файле {}".format(mat_filepath))
                return None

            # Используем первую найденную переменную
            var_name = var_names[0]
            logger.debug("Найдены переменные: {}, выбрана: {}".format(var_names, var_name))

            array = mat_data[var_name]
            logger.info("Загружен {} из MAT, форма: {}, диапазон: [{:.3f}, {:.3f}]".format(
                var_type, array.shape, array.min(), array.max()))

            return array

        except Exception as e:
            logger.error("Ошибка при загрузке {} из MAT файла {}: {}".format(
                var_type, mat_filepath, e))
            return None

    @staticmethod
    def import_mat(filepath, background_filepath=None):
        # type: (str, Optional[str]) -> Optional[Dict[str, Any]]
        """
        Импортирует данные из MAT файла

        ВАЖНО: MAT файлы уже содержат нормализованные данные (из MATLAB im2double)

        Args:
            filepath: Путь к файлу
            background_filepath: Путь к файлу с фоном (если None, пытается найти автоматически)

        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info("Начало импорта MAT файла: {}".format(filepath))

            # Загружаем основной файл
            shot = DataImporter._load_mat_variable(filepath, "shot")
            if shot is None:
                print("Не удалось загрузить данные из файла {}".format(filepath))
                return None

            # Оцениваем количество нулей (диагностика)
            evaluate_zeros_in_shot(shot)

            # Ищем файл с фоном если не указан
            if background_filepath is None:
                logger.info("Автоматический поиск файла с фоном")
                background_filepath = DataImporter._find_background_file(filepath)

            if background_filepath is None:
                print("Файл с фоном для {} не найден".format(filepath))
                logger.error("Файл с фоном не найден")
                return None

            # Загружаем файл с фоном
            background = DataImporter._load_mat_variable(background_filepath, "background")
            if background is None:
                print("Не удалось загрузить фон из файла {}".format(background_filepath))
                return None

            # Валидация размеров
            if not DataImporter._validate_shapes_match(shot, background, "shot", "background"):
                print("Размеры снимка ({}) и фона ({}) не совпадают".format(
                    shot.shape, background.shape))
                return None

            # Определяем камеру по размеру
            camera_info = DataImporter._determine_camera_by_shape(shot.shape)
            logger.info("Определена камера: {}".format(camera_info['name']))

            # MAT файлы уже содержат нормализованные данные (из MATLAB im2double)
            # НЕ нормализуем повторно!
            normalized_shot = shot
            normalized_background = background

            # Вычитаем фон
            difference = normalized_shot - normalized_background

            # Диагностика отрицательных значений
            plot_negative_density(difference, "negative_density", "negative_density.png")
            negative_count = np.sum(difference < 0)
            negative_percent = 100.0 * negative_count / difference.size
            logger.info("Отрицательных значений в difference: {} ({:.2f}%)".format(
                negative_count, negative_percent))

            # ЗАКОММЕНТИРОВАНО: обнуление отрицательных значений (под вопросом)
            difference[difference < 0] = 0

            # Формируем результат используя общий метод
            result = DataImporter._create_result_dict(
                shot=normalized_shot,
                background=normalized_background,
                difference=difference,
                raw_shot=shot,
                raw_background=background,
                camera_info=camera_info,
                filepath=filepath
            )

            logger.info("Успешно импортирован MAT файл: {}".format(filepath))
            return result

        except Exception as e:
            print("Ошибка при импорте MAT файла: {}".format(e))
            logger.error("Ошибка при импорте MAT файла: {}".format(e), exc_info=True)
            return None


    @staticmethod
    def _find_background_file(filepath):
        # type: (str) -> Optional[str]
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
        # type: (Tuple[int, int]) -> Dict[str, Any]
        """
        Определяет камеру по размеру массива

        Args:
            shape: Размер массива

        Returns:
            dict: Информация о камере
        """
        logger.debug("Определение камеры по размеру массива: {}".format(shape))

        # Проверяем соответствие размеров с известными камерами
        for camera_name, info in ImporterConfig.CAMERAS.items():
            resolution = info["resolution"]
            # Проверяем прямое и обратное совпадение (на случай транспонированных данных)
            if shape == resolution or shape == (resolution[1], resolution[0]):
                logger.debug("Определена известная камера: {}".format(camera_name))
                return info

        # Если не нашли соответствие, возвращаем значения по умолчанию
        logger.warning("Не удалось определить камеру по размеру {}. Используются значения по умолчанию.".format(shape))
        default_camera = ImporterConfig.DEFAULT_CAMERA.copy()
        default_camera["resolution"] = shape
        return default_camera

    @staticmethod
    def is_data_folder(path):
        # type: (str) -> bool
        """
        Проверяет, является ли указанный путь папкой с данными анализатора профиля пучка

        Args:
            path: Путь к папке или файлу

        Returns:
            bool: True если это папка с данными, иначе False
        """
        # Если путь указывает на файл metadata.json, проверяем его папку
        if os.path.isfile(path) and os.path.basename(path) == ImporterConfig.FILENAME_METADATA:
            path = os.path.dirname(path)

        # Проверяем, является ли путь папкой
        if not os.path.isdir(path):
            return False

        # Проверяем наличие метаданных
        metadata_path = os.path.join(path, ImporterConfig.FILENAME_METADATA)
        if not os.path.isfile(metadata_path):
            return False

        # Проверяем наличие хотя бы одного из основных CSV файлов
        # Поддерживаем как обработанные (shot, background, difference),
        # так и сырые данные (raw_shot, raw_background)
        expected_files = [
            ImporterConfig.FILENAME_SHOT,
            ImporterConfig.FILENAME_BACKGROUND,
            ImporterConfig.FILENAME_DIFFERENCE,
            ImporterConfig.FILENAME_RAW_SHOT,
            ImporterConfig.FILENAME_RAW_BACKGROUND
        ]

        for filename in expected_files:
            if os.path.isfile(os.path.join(path, filename)):
                return True

        return False

    @staticmethod
    def import_folder(folder_path):
        # type: (str) -> Optional[Dict[str, Any]]
        """
        Импортирует данные из папки с метаданными и CSV файлами

        Args:
            folder_path: Путь к папке с данными

        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        # Если путь указывает на файл metadata.json, берем его директорию
        if os.path.isfile(folder_path) and os.path.basename(folder_path) == ImporterConfig.FILENAME_METADATA:
            metadata_filepath = folder_path
            folder_path = os.path.dirname(folder_path)
        else:
            # Проверяем, что это папка с данными
            if not DataImporter.is_data_folder(folder_path):
                print("Указанный путь не является папкой с данными: {}".format(folder_path))
                logger.error("Указанный путь не является папкой с данными: {}".format(folder_path))
                return None

            metadata_filepath = os.path.join(folder_path, ImporterConfig.FILENAME_METADATA)

        return DataImporter.import_multifile_csv(metadata_filepath)

    @staticmethod
    def _load_metadata(metadata_filepath):
        # type: (str) -> Optional[Dict[str, Any]]
        """
        Загружает метаданные из JSON файла

        Args:
            metadata_filepath: Путь к JSON файлу с метаданными

        Returns:
            dict: Метаданные или None в случае ошибки
        """
        try:
            if not os.path.exists(metadata_filepath):
                logger.error("Файл метаданных не найден: {}".format(metadata_filepath))
                return None

            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            logger.debug("Метаданные успешно загружены из {}".format(metadata_filepath))
            return metadata

        except json.JSONDecodeError as e:
            logger.error("Ошибка парсинга JSON файла {}: {}".format(metadata_filepath, e))
            return None
        except Exception as e:
            logger.error("Ошибка при чтении метаданных {}: {}".format(metadata_filepath, e))
            return None

    @staticmethod
    def _load_csv_arrays(folder_path, arrays_info):
        # type: (str, Dict[str, Any]) -> Dict[str, np.ndarray]
        """
        Загружает все CSV массивы из папки согласно метаданным

        Args:
            folder_path: Путь к папке с CSV файлами
            arrays_info: Информация о массивах из метаданных

        Returns:
            dict: Словарь загруженных массивов
        """
        loaded_arrays = {}

        for array_name in arrays_info.keys():
            csv_filename = "{}.csv".format(array_name)
            csv_path = os.path.join(folder_path, csv_filename)

            if not os.path.exists(csv_path):
                logger.warning("Файл массива не найден: {}".format(csv_path))
                continue

            # Получаем ожидаемую форму из метаданных
            expected_shape = None
            if 'shape' in arrays_info[array_name]:
                shape_data = arrays_info[array_name]['shape']
                if isinstance(shape_data, list):
                    expected_shape = tuple(shape_data)

            # Загружаем массив используя pandas
            array_data = DataImporter._load_csv_with_pandas(csv_path, expected_shape)
            if array_data is not None:
                loaded_arrays[array_name] = array_data

        logger.info("Загружено {} массивов из {}".format(len(loaded_arrays), len(arrays_info)))
        return loaded_arrays

    @staticmethod
    def import_multifile_csv(metadata_filepath):
        # type: (str) -> Optional[Dict[str, Any]]
        """
        Импортирует данные из набора файлов в папке: JSON метаданные и чистые CSV файлы для массивов

        Поддерживает два варианта данных:
        1. Полные данные: shot.csv, background.csv, difference.csv, raw_shot.csv, raw_background.csv
        2. Неполные данные: только raw_shot.csv и raw_background.csv (обработанные вычисляются автоматически)

        ВАЖНО: CSV файлы содержат сырые значения яркости (8-бит, 0-255)

        Args:
            metadata_filepath: Путь к JSON файлу с метаданными

        Returns:
            dict: Словарь с импортированными данными или None в случае ошибки
        """
        try:
            logger.info("Начало импорта данных из папки: {}".format(metadata_filepath))

            # Загружаем метаданные
            metadata = DataImporter._load_metadata(metadata_filepath)
            if metadata is None:
                print("Не удалось загрузить метаданные: {}".format(metadata_filepath))
                return None

            # Определяем папку с файлами
            folder_path = os.path.dirname(metadata_filepath)

            # Загружаем все CSV массивы
            arrays_info = metadata.get('arrays', {})
            logger.info("Доступные массивы по метаданным: {}".format(list(arrays_info.keys())))

            loaded_arrays = DataImporter._load_csv_arrays(folder_path, arrays_info)

            if not loaded_arrays:
                logger.error("Не удалось загрузить ни одного массива")
                return None

            # Определяем тип данных (полные или только сырые)
            has_full_data = ('shot' in loaded_arrays and 'background' in loaded_arrays)
            has_raw_data = ('raw_shot' in loaded_arrays and 'raw_background' in loaded_arrays)

            logger.info("Проверка наличия данных: полные={}, сырые={}".format(has_full_data, has_raw_data))

            # Подготавливаем базовую структуру результата
            result_data = {
                'resolution': metadata.get('resolution'),
                'pixel_size_x': metadata.get('pixel_size_x'),
                'pixel_size_y': metadata.get('pixel_size_y'),
                'camera_name': metadata.get('camera_name'),
                'date': metadata.get('date')
            }

            # Обрабатываем данные в зависимости от их типа
            if has_full_data:
                # Вариант 1: Есть полные обработанные данные (уже нормализованные)
                logger.info("Обнаружены полные нормализованные данные")

                result_data['shot'] = loaded_arrays['shot']
                result_data['background'] = loaded_arrays['background']
                result_data['raw_shot'] = loaded_arrays.get('raw_shot')
                result_data['raw_background'] = loaded_arrays.get('raw_background')

                # Вычисляем difference если отсутствует
                if 'difference' in loaded_arrays:
                    result_data['difference'] = loaded_arrays['difference']
                else:
                    logger.info("Файл difference.csv не найден, вычисляем разность")
                    difference = result_data['shot'] - result_data['background']

                    # Логируем статистику отрицательных значений
                    negative_count = np.sum(difference < 0)
                    negative_percent = 100.0 * negative_count / difference.size
                    logger.info("Отрицательных значений: {} из {} ({:.2f}%)".format(
                        negative_count, difference.size, negative_percent))

                    result_data['difference'] = difference

            elif has_raw_data:
                # Вариант 2: Есть только сырые данные (8-бит) - нужна нормализация
                logger.info("Обнаружены только сырые данные, выполняется нормализация")

                raw_shot = loaded_arrays['raw_shot']
                raw_background = loaded_arrays['raw_background']

                # Валидация размеров
                if not DataImporter._validate_shapes_match(raw_shot, raw_background,
                                                          'raw_shot', 'raw_background'):
                    return None

                # Обрабатываем сырые данные (нормализация + вычитание фона)
                processed = DataImporter._process_image_data(raw_shot, raw_background)

                result_data['raw_shot'] = raw_shot
                result_data['raw_background'] = raw_background
                result_data['shot'] = processed['shot']
                result_data['background'] = processed['background']
                result_data['difference'] = processed['difference']

            else:
                # Нет необходимых данных
                logger.error("Не найдены необходимые массивы. "
                           "Требуются либо (shot и background), либо (raw_shot и raw_background)")
                print("Ошибка: не найдены необходимые данные в папке {}".format(folder_path))
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
        # type: (str) -> Optional[Dict[str, Any]]
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
        if os.path.isfile(filepath) and os.path.basename(filepath) == ImporterConfig.FILENAME_METADATA:
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
