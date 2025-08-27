#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для подключения к монохромной камере и создания снимков через интерактивное меню.
Использует библиотеку pydc1394 для работы с камерами IEEE1394 (Firewire).
"""

import os
import sys
from datetime import datetime
try:
    from PIL import Image
except ImportError:
    print("Библиотека PIL не установлена. Устанавливаем: pip install Pillow")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image
import numpy as np
from pydc1394 import Camera
from pydc1394.camera2 import Context

def list_cameras():
    """
    Перечисляет все доступные камеры в системе с детальной информацией.
    
    Returns:
        list: Список доступных камер или пустой список, если камеры не обнаружены
    """
    context = Context()
    cameras = context.cameras
    
    if not cameras:
        print("Камеры не обнаружены.")
        return []
    
    print("Найдено камер: {}".format(len(cameras)))
    
    # Показываем детальную информацию по каждой камере
    for i, cam in enumerate(cameras):
        print("{}: {} ({})".format(i, cam, "инициализация..."))
        
        # Пытаемся получить детальную информацию о камере
        try:
            # Временно подключаемся к камере для получения информации
            camera = Camera(guid=cam[0])
            
            # Получаем название и модель
            vendor = camera.vendor.decode('utf-8', errors='replace') if hasattr(camera, 'vendor') else "Неизвестно"
            model = camera.model.decode('utf-8', errors='replace') if hasattr(camera, 'model') else "Неизвестно"
            
            # Получаем информацию о разрешении
            try:
                width = camera.width
                height = camera.height
                resolution_info = "{}x{}".format(width, height)
            except AttributeError:
                resolution_info = "Неизвестно"
            
            # Получаем доступные режимы камеры
            try:
                modes_count = len(camera.modes) if hasattr(camera, 'modes') else 0
                modes_info = "{} режимов".format(modes_count) if modes_count > 0 else "Режимы неизвестны"
            except AttributeError:
                modes_info = "Режимы неизвестны"
            
            # Выводим детальную информацию
            print("   └─ Модель: {} {}".format(vendor, model))
            print("   └─ Разрешение: {}".format(resolution_info))
            print("   └─ Доступно: {}".format(modes_info))
            
            # Закрываем подключение к камере
            camera.stop_capture()
            del camera
            
        except Exception as e:
            print("   └─ Ошибка при получении информации: {}".format(e))
    
    return cameras

def capture_image(camera_index=None, output_dir="images", filename=None, bit_depth=16, file_format="tiff"):
    """
    Подключается к монохромной камере и делает снимок.
    
    Args:
        camera_index: Индекс камеры (None для первой доступной)
        output_dir: Папка для сохранения изображений
        filename: Имя файла (None для автоматической генерации)
        bit_depth: Глубина цвета (8 или 16 бит)
        file_format: Формат файла (png, tiff)
        
    Returns:
        str: Путь к сохраненному изображению или None в случае ошибки
    """
    # Создаем папку для изображений, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Если имя файла не указано, генерируем его из текущей даты и времени
    if filename is None:
        if file_format.lower() == "tiff" or file_format.lower() == "tif":
            extension = ".tiff"
        else:
            extension = ".png"
            
        filename = "image_{}{}".format(datetime.now().strftime('%Y%m%d_%H%M%S'), extension)
    elif not (filename.lower().endswith('.png') or filename.lower().endswith('.tiff') or filename.lower().endswith('.tif')):
        if file_format.lower() == "tiff" or file_format.lower() == "tif":
            filename += ".tiff"
        else:
            filename += ".png"
    
    # Полный путь к файлу
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Получаем список доступных камер
        context = Context()
        cameras = context.cameras
        
        if not cameras:
            print("Камеры не обнаружены.")
            return None
        
        # Выбираем камеру
        if camera_index is None:
            camera = Camera()  # Подключаемся к первой доступной камере
        else:
            if camera_index >= len(cameras):
                print("Ошибка: Камера с индексом {} не найдена.".format(camera_index))
                return None
            # Получаем GUID камеры из первого элемента кортежа
            cam_guid = cameras[camera_index][0]
            camera = Camera(guid=cam_guid)
        
        # Выводим информацию о камере
        print("Подключено к монохромной камере:")
        print("  Производитель: {}".format(camera.vendor))
        print("  Модель: {}".format(camera.model))
        print("  GUID: {}".format(camera.guid))
        
        # Записываем настройки камеры в лог
        log_camera_settings(camera)
        
        # Выводим дополнительную информацию о камере
        try:
            print("  Доступные режимы: {}".format(camera.modes))
            current_mode = camera.mode
            print("  Текущий режим: {}".format(current_mode))
            print("  Текущее разрешение: {}x{}".format(camera.width, camera.height))
            print("  Частота кадров: {}".format(camera.framerate))
        except AttributeError as e:
            print("  Невозможно получить все параметры камеры: {}".format(e))
        
        # Пробуем настроить режим с максимальной битностью, если требуется 16 бит
        if bit_depth == 16:
            try:
                # Ищем режим с 16-битной глубиной, подходящий для монохромной камеры
                high_bit_modes = [mode for mode in camera.modes if 'format7' in str(mode).lower() or '16' in str(mode).lower() or 'mono' in str(mode).lower()]
                if high_bit_modes:
                    print("Найден режим с высокой битностью для монохромной камеры: {}".format(high_bit_modes[0]))
                    camera.mode = high_bit_modes[0]
                    print("Установлен режим: {}".format(camera.mode))
                else:
                    print("Предупреждение: режим с 16-битной глубиной не найден, используем текущий режим")
            except Exception as e:
                print("Не удалось установить 16-битный режим: {}".format(e))
        
        # Пробуем установить автоматические параметры
        try:
            camera.brightness.mode = 'auto'
            camera.exposure.mode = 'auto'
            # Убираем настройку баланса белого для монохромной камеры
            # camera.white_balance.mode = 'auto'
            
            # Устанавливаем минимальные значения яркости и экспозиции
            # если изображение получается черным
            if camera.exposure.mode != 'auto':
                print("Установка ручных настроек экспозиции...")
                camera.exposure.value = 400  # Увеличиваем экспозицию
            
            if camera.brightness.mode != 'auto':
                print("Установка ручных настроек яркости...")
                camera.brightness.value = 200  # Увеличиваем яркость
            
            # Устанавливаем максимальное усиление для камеры
            try:
                camera.gain.mode = 'auto'
                if camera.gain.mode != 'auto':
                    camera.gain.value = camera.gain.max
                    print("Установлено ручное усиление: {}".format(camera.gain.value))
            except AttributeError:
                pass
        except AttributeError:
            # Некоторые камеры могут не поддерживать эти функции
            print("Внимание: некоторые автоматические настройки не поддерживаются камерой")
            pass
        
        # Захватываем изображение
        print("Получение снимка с монохромной камеры...")
        camera.start_capture()
        camera.start_one_shot()
        frame = camera.dequeue()
        
        # Получаем данные из кадра
        frame_data = frame.copy()
        print("Формат кадра: {}, форма: {}, тип данных: {}".format(
            type(frame_data), frame_data.shape, frame_data.dtype))
        
        # Конвертируем в нужный формат в зависимости от требуемой битности
        if bit_depth == 16:
            # Убедимся, что у нас есть numpy array
            if not isinstance(frame_data, np.ndarray):
                frame_data = np.array(frame_data)
            
            # Если данные уже в 16-битном формате, используем их напрямую
            if frame_data.dtype == np.uint16:
                print("Данные монохромной камеры уже в 16-битном формате")
                image_data = frame_data
            # Если данные в 8-битном формате, конвертируем в 16-битный
            elif frame_data.dtype == np.uint8:
                print("Конвертация из 8-бит в 16-бит")
                # Масштабируем от 0-255 до 0-65535
                image_data = (frame_data.astype(np.uint16) * 257)
            else:
                print("Неожиданный формат данных: {}, преобразуем в 16-бит".format(frame_data.dtype))
                # Пытаемся преобразовать в 16 бит независимо от исходного формата
                image_data = frame_data.astype(np.uint16)
            
            # Создаем 16-битное монохромное изображение
            # Для монохромной камеры всегда используем монохромный режим
            print("Создание 16-битного монохромного изображения")
            image = Image.fromarray(image_data, mode='I;16')
        else:
            # Стандартное 8-битное монохромное изображение
            # Для монохромной камеры всегда преобразуем в режим 'L'
            frame_data_array = np.array(frame_data)
            if len(frame_data_array.shape) == 3:
                # Если по какой-то причине данные имеют 3 канала, конвертируем в оттенки серого
                if frame_data_array.shape[2] == 3:
                    print("Конвертация RGB данных в монохромные")
                    gray_data = np.dot(frame_data_array[...,:3], [0.2989, 0.5870, 0.1140])
                    image = Image.fromarray(gray_data.astype(np.uint8), mode='L')
                else:
                    image = Image.fromarray(frame_data_array, mode='L')
            else:
                image = Image.fromarray(frame_data_array, mode='L')
            print("Создано 8-битное монохромное изображение")
        
        # Проверяем, не черное ли изображение
        is_black = check_black_image(image)
        if is_black:
            print("Предупреждение: Изображение слишком темное, попытка скорректировать...")
            # Пробуем увеличить яркость программно
            try:
                # Создаем более яркую версию изображения для проверки
                img_array = np.array(image)
                # Увеличиваем яркость
                bright_img = Image.fromarray(np.clip(img_array * 2, 0, 65535 if bit_depth == 16 else 255).astype(np.uint16 if bit_depth == 16 else np.uint8))
                # Используем более яркую версию
                image = bright_img
                print("Яркость изображения увеличена программно")
            except Exception as e:
                print("Не удалось скорректировать яркость: {}".format(e))
        
        # Освобождаем буфер и останавливаем захват
        frame.enqueue()
        camera.stop_one_shot()
        camera.stop_capture()
        
        # Сохраняем изображение в соответствующем формате
        if file_format.lower() == "tiff" or file_format.lower() == "tif":
            print("Сохранение 16-битного TIFF монохромного изображения")
            image.save(output_path, format='TIFF')
        else:
            print("Сохранение монохромного изображения в формате PNG")
            image.save(output_path, format='PNG')
        
        print("Изображение сохранено: {}".format(output_path))
        print("Битность: {} бит, формат: {}".format(bit_depth, file_format.upper()))
        
        return output_path
    
    except Exception as e:
        print("Ошибка при захвате изображения: {}".format(e))
        return None

def clear_screen():
    """
    Очищает экран терминала.
    """
    # Для Windows
    if os.name == 'nt':
        os.system('cls')
    # Для Mac и Linux
    else:
        os.system('clear')

def print_menu(output_dir, filename, bit_depth, file_format):
    """
    Выводит меню программы.
    
    Args:
        output_dir: Текущая папка для сохранения изображений
        filename: Текущее имя файла (или None для автоматической генерации)
        bit_depth: Текущая битность изображения
        file_format: Текущий формат файла
    """
    clear_screen()
    print("=" * 50)
    print("ПРОГРАММА ДЛЯ РАБОТЫ С МОНОХРОМНЫМИ КАМЕРАМИ DC1394")
    print("=" * 50)
    print("Текущие настройки:")
    print("  Папка для сохранения: {}".format(output_dir))
    print("  Имя файла: {}".format(filename if filename else "Автоматическое"))
    print("  Битность: {} бит".format(bit_depth))
    print("  Формат: {}".format(file_format.upper()))
    print("-" * 50)
    print("Меню:")
    print("  1. Показать список доступных камер")
    print("  2. Сделать снимок с первой доступной камеры")
    print("  3. Сделать снимок с выбранной камеры")
    print("  4. Изменить папку для сохранения")
    print("  5. Изменить имя файла")
    print("  6. Показать подробную информацию о камерах")
    print("  7. Настройка битности изображения")
    print("  8. Настройка формата файла")
    print("  9. Настройка ROI для полного разрешения")
    print("  0. Выход")
    print("-" * 50)

def get_integer_input(prompt, min_value=0, max_value=None):
    """
    Получает целочисленный ввод от пользователя с проверкой диапазона.
    
    Args:
        prompt: Текст приглашения для ввода
        min_value: Минимальное допустимое значение
        max_value: Максимальное допустимое значение (или None, если нет верхнего предела)
        
    Returns:
        int: Введенное пользователем число
    """
    while True:
        try:
            value = int(input(prompt))
            if value < min_value:
                print("Значение должно быть не меньше {}.".format(min_value))
                continue
            if max_value is not None and value > max_value:
                print("Значение должно быть не больше {}.".format(max_value))
                continue
            return value
        except ValueError:
            print("Пожалуйста, введите целое число.")

def get_format7_full_info(camera, mode):
    """
    Получает полную информацию о режиме Format 7, включая максимальные разрешения.
    
    Args:
        camera: Объект камеры
        mode: Режим Format 7
        
    Returns:
        dict: Словарь с информацией о режиме
    """
    info = {}
    
    try:
        # Временно переключаемся на режим
        current_mode = camera.mode
        camera.mode = mode
        
        # Получаем максимальные размеры изображения
        try:
            # Пробуем получить максимальный размер через разные способы
            if hasattr(mode, 'max_image_size'):
                max_width, max_height = mode.max_image_size
                info['max_resolution'] = "{}x{}".format(max_width, max_height)
            elif hasattr(mode, 'image_size'):
                width, height = mode.image_size
                info['max_resolution'] = "{}x{}".format(width, height)
            
            # Пробуем установить максимально возможное разрешение
            try:
                # Пробуем получить информацию о ROI разными способами
                try:
                    # Способ 1: через mode.roi
                    if hasattr(mode, 'roi'):
                        current_roi = mode.roi
                        info['current_roi'] = "left={}, top={}, width={}, height={}".format(*current_roi)
                    # Способ 2: через позицию и размер отдельно
                    elif hasattr(mode, 'image_position') and hasattr(mode, 'image_size'):
                        pos = mode.image_position
                        size = mode.image_size
                        info['current_roi'] = "left={}, top={}, width={}, height={}".format(pos[0], pos[1], size[0], size[1])
                    
                    # Пробуем установить максимальную ROI для проверки
                    if hasattr(mode, 'roi'):
                        try:
                            original_roi = mode.roi
                            # Пробуем установить максимальное разрешение
                            mode.roi = (0, 0, 1624, 1224)
                            new_roi = mode.roi
                            info['max_possible_roi'] = "left={}, top={}, width={}, height={}".format(*new_roi)
                            # Возвращаем исходную ROI
                            mode.roi = original_roi
                        except Exception as e:
                            info['roi_test_error'] = str(e)
                    elif hasattr(mode, 'image_position') and hasattr(mode, 'image_size'):
                        try:
                            original_pos = mode.image_position
                            original_size = mode.image_size
                            # Пробуем установить максимальное разрешение
                            mode.image_position = (0, 0)
                            mode.image_size = (1624, 1224)
                            new_pos = mode.image_position
                            new_size = mode.image_size
                            info['max_possible_roi'] = "left={}, top={}, width={}, height={}".format(new_pos[0], new_pos[1], new_size[0], new_size[1])
                            # Возвращаем исходные значения
                            mode.image_position = original_pos
                            mode.image_size = original_size
                        except Exception as e:
                            info['roi_test_error'] = str(e)
                            
                except Exception as e:
                    info['roi_error'] = str(e)
                        
            except Exception as e:
                info['general_error'] = str(e)
                
        except Exception as e:
            info['size_error'] = str(e)
            
        # Возвращаем исходный режим
        camera.mode = current_mode
        
    except Exception as e:
        info['mode_error'] = str(e)
    
    return info

def debug_format7_attributes(camera, mode):
    """
    Выводит все доступные атрибуты режима Format 7 для отладки.
    
    Args:
        camera: Объект камеры
        mode: Режим Format 7
    """
    try:
        current_mode = camera.mode
        camera.mode = mode
        
        print("Отладка атрибутов режима Format 7: {}".format(mode))
        print("Все доступные атрибуты:")
        
        for attr in sorted(dir(mode)):
            if not attr.startswith('_'):
                try:
                    value = getattr(mode, attr)
                    if callable(value):
                        print("  - {} (метод)".format(attr))
                    else:
                        print("  - {} = {}".format(attr, value))
                except Exception as e:
                    print("  - {} (ошибка: {})".format(attr, e))
        
        camera.mode = current_mode
        
    except Exception as e:
        print("Ошибка при отладке атрибутов: {}".format(e))

def set_custom_roi(camera_index=None, left=0, top=0, width=1624, height=1224):
    """
    Устанавливает пользовательскую ROI для камеры.
    
    Args:
        camera_index: Индекс камеры (None для первой доступной)
        left: Левая граница ROI
        top: Верхняя граница ROI  
        width: Ширина ROI
        height: Высота ROI
        
    Returns:
        bool: True если ROI установлена успешно, False в противном случае
    """
    try:
        # Получаем список доступных камер
        context = Context()
        cameras = context.cameras
        
        if not cameras:
            print("Камеры не обнаружены.")
            return False
        
        # Выбираем камеру
        if camera_index is None:
            camera = Camera()
        else:
            if camera_index >= len(cameras):
                print("Ошибка: Камера с индексом {} не найдена.".format(camera_index))
                return False
            cam_guid = cameras[camera_index][0]
            camera = Camera(guid=cam_guid)
        
        print("Подключение к камере для настройки ROI...")
        print("Модель: {}".format(camera.model.decode('utf-8', errors='replace')))
        
        # Переводим камеру в режим Format 7, если она не в нём
        current_mode = camera.mode
        print("Текущий режим: {}".format(current_mode))
        
        # Ищем подходящий режим Format 7
        format7_modes = [mode for mode in camera.modes if 'format7' in str(mode).lower()]
        
        if not format7_modes:
            print("Режимы Format 7 недоступны для данной камеры")
            return False
            
        # Выбираем первый режим Format 7
        selected_mode = format7_modes[0]
        print("Переключение на режим: {}".format(selected_mode))
        camera.mode = selected_mode
        
        # Выводим отладочную информацию об атрибутах режима
        print("\n--- ОТЛАДОЧНАЯ ИНФОРМАЦИЯ ---")
        debug_format7_attributes(camera, selected_mode)
        print("--- КОНЕЦ ОТЛАДОЧНОЙ ИНФОРМАЦИИ ---\n")
        
        # Получаем информацию о текущих параметрах
        try:
            if hasattr(selected_mode, 'roi'):
                current_roi = selected_mode.roi
                # ROI имеет формат: ((width, height), (left, top), color_coding, packet_size)
                current_size = current_roi[0]  # (width, height)
                current_pos = current_roi[1]   # (left, top)
                current_coding = current_roi[2] # color_coding
                current_packet = current_roi[3] # packet_size
                
                print("Текущая ROI:")
                print("  Размер: {}x{}".format(current_size[0], current_size[1]))
                print("  Позиция: left={}, top={}".format(current_pos[0], current_pos[1]))
                print("  Цветовой режим: {}".format(current_coding))
                print("  Размер пакета: {}".format(current_packet))
                
            # Также выводим информацию через отдельные атрибуты
            if hasattr(selected_mode, 'image_position') and hasattr(selected_mode, 'image_size'):
                pos = selected_mode.image_position
                size = selected_mode.image_size
                max_size = selected_mode.max_image_size
                print("Через отдельные атрибуты:")
                print("  Текущая позиция: left={}, top={}".format(pos[0], pos[1]))
                print("  Текущий размер: {}x{}".format(size[0], size[1]))
                print("  Максимальный размер: {}x{}".format(max_size[0], max_size[1]))
                
        except Exception as e:
            print("Ошибка при получении текущих параметров: {}".format(e))
        
        # Проверяем, не превышает ли запрашиваемый размер максимальный
        try:
            max_size = selected_mode.max_image_size
            max_width, max_height = max_size
            
            if width > max_width or height > max_height:
                print("ВНИМАНИЕ: Запрашиваемое разрешение {}x{} превышает максимальное {}x{}".format(
                    width, height, max_width, max_height))
                print("Устанавливаем максимально возможное разрешение: {}x{}".format(max_width, max_height))
                width, height = max_width, max_height
                
        except Exception as e:
            print("Не удалось получить максимальный размер: {}".format(e))
        
        # Устанавливаем новую ROI
        print("\nПопытка установить ROI: left={}, top={}, width={}, height={}".format(left, top, width, height))
        
        try:
            # Используем отдельные атрибуты для установки позиции и размера
            if hasattr(selected_mode, 'image_position') and hasattr(selected_mode, 'image_size'):
                print("Установка через отдельные атрибуты...")
                
                # Устанавливаем позицию
                selected_mode.image_position = (left, top)
                new_pos = selected_mode.image_position
                print("  Позиция установлена: left={}, top={}".format(new_pos[0], new_pos[1]))
                
                # Устанавливаем размер
                selected_mode.image_size = (width, height)
                new_size = selected_mode.image_size
                print("  Размер установлен: {}x{}".format(new_size[0], new_size[1]))
                
                # Проверяем итоговую ROI
                if hasattr(selected_mode, 'roi'):
                    final_roi = selected_mode.roi
                    final_size = final_roi[0]
                    final_pos = final_roi[1]
                    print("  Итоговая ROI: размер={}x{}, позиция=left={}, top={}".format(
                        final_size[0], final_size[1], final_pos[0], final_pos[1]))
                
                print("ROI успешно установлена!")
                return True
                
            else:
                print("Не удалось найти атрибуты image_position и image_size")
                return False
                
        except Exception as e:
            print("Ошибка при установке ROI: {}".format(e))
            print("Тип ошибки: {}".format(type(e).__name__))
            return False
            
    except Exception as e:
        print("Ошибка при работе с камерой: {}".format(e))
        return False

def log_camera_settings(camera, log_file="camera_settings.log"):
    """
    Записывает настройки камеры в лог файл.
    
    Args:
        camera: Объект камеры pydc1394.Camera
        log_file: Путь к файлу лога (по умолчанию 'camera_settings.log')
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n{}\n".format('='*50))
            f.write("Настройки камеры - {}\n".format(timestamp))
            f.write("{}\n".format('='*50))
            
            # Основная информация о камере
            vendor_str = camera.vendor.decode('utf-8', errors='replace') if hasattr(camera, 'vendor') else 'Неизвестно'
            model_str = camera.model.decode('utf-8', errors='replace') if hasattr(camera, 'model') else 'Неизвестно'
            f.write("Производитель: {}\n".format(vendor_str))
            f.write("Модель: {}\n".format(model_str))
            f.write("GUID: {}\n".format(camera.guid))
            
            # Пытаемся инициализировать камеру для получения дополнительных параметров
            try:
                # Запускаем захват для получения параметров
                camera.start_capture()
                
                # Получаем информацию о режиме и разрешении
                if hasattr(camera, 'modes') and camera.modes:
                    f.write("Доступные режимы:\n")
                    for mode in camera.modes:
                        f.write("  - {}\n".format(mode))
                        
                        # Проверяем, является ли режим Format 7
                        if 'format7' in str(mode).lower():
                            try:
                                # Получаем полную информацию о режиме Format 7
                                format7_info = get_format7_full_info(camera, mode)
                                
                                f.write("    Детали Format 7:\n")
                                
                                # Выводим максимальное разрешение
                                if 'max_resolution' in format7_info:
                                    f.write("    - Максимальное разрешение: {}\n".format(format7_info['max_resolution']))
                                
                                # Выводим текущую ROI
                                if 'current_roi' in format7_info:
                                    f.write("    - Текущая ROI: {}\n".format(format7_info['current_roi']))
                                
                                # Пробуем получить максимально возможную ROI
                                if 'max_possible_roi' in format7_info:
                                    f.write("    - Максимально возможная ROI: {}\n".format(format7_info['max_possible_roi']))
                                
                                # Выводим ошибки, если они есть
                                for error_key in ['roi_error', 'general_error', 'size_error', 'mode_error']:
                                    if error_key in format7_info:
                                        f.write("    - Ошибка ({}): {}\n".format(error_key, format7_info[error_key]))
                                
                                # Получаем дополнительную информацию о режиме
                                current_mode = camera.mode
                                camera.mode = mode
                                
                                # Получаем информацию о пикселях
                                if hasattr(mode, 'color_coding'):
                                    f.write("    - Цветовой режим: {}\n".format(mode.color_coding))
                                
                                # Получаем информацию о пакете данных
                                try:
                                    if hasattr(mode, 'packet_size'):
                                        packet_size = mode.packet_size
                                        # Пробуем получить максимальный размер пакета через свойство mode
                                        try:
                                            max_packet = mode.packet_parameters['max_size']
                                            f.write("    - Размер пакета: текущий={}, макс={}\n".format(
                                                packet_size, max_packet))
                                        except (AttributeError, KeyError, TypeError):
                                            f.write("    - Текущий размер пакета: {}\n".format(packet_size))
                                except Exception as e:
                                    f.write("    - Ошибка при получении размера пакета: {}\n".format(e))
                                
                                # Получаем поддерживаемые цветовые режимы
                                try:
                                    if hasattr(mode, 'color_codings'):
                                        f.write("    - Поддерживаемые цветовые режимы:\n")
                                        for coding in mode.color_codings:
                                            f.write("      * {}\n".format(coding))
                                except Exception as e:
                                    f.write("    - Ошибка при получении цветовых режимов: {}\n".format(e))
                                
                                # Возвращаем исходный режим
                                camera.mode = current_mode
                                
                            except AttributeError as e:
                                f.write("    Ошибка при получении деталей Format 7: {}\n".format(e))
                            except Exception as e:
                                f.write("    Ошибка при работе с режимом Format 7: {}\n".format(e))
                    
                    # Если есть текущий режим, записываем его
                    if hasattr(camera, 'mode'):
                        f.write("Текущий режим: {}\n".format(camera.mode))
                        # Если текущий режим - Format 7, выводим его текущие настройки
                        if 'format7' in str(camera.mode).lower():
                            try:
                                f.write("Текущие настройки Format 7:\n")
                                if hasattr(camera, 'roi'):
                                    left, top, width, height = camera.roi
                                    f.write("  - ROI: left={}, top={}, width={}, height={}\n".format(
                                        left, top, width, height))
                                if hasattr(camera.mode, 'color_coding'):
                                    f.write("  - Текущий цветовой режим: {}\n".format(camera.mode.color_coding))
                                if hasattr(camera.mode, 'packet_size'):
                                    f.write("  - Текущий размер пакета: {}\n".format(camera.mode.packet_size))
                            except Exception as e:
                                f.write("  Ошибка при получении текущих настроек Format 7: {}\n".format(e))
                
                # Получаем размеры кадра из текущего режима
                try:
                    mode = camera.mode
                    if hasattr(mode, 'image_size'):
                        width, height = mode.image_size
                        f.write("Разрешение: {}x{}\n".format(width, height))
                except AttributeError:
                    pass
                
                # Получаем частоту кадров
                try:
                    if hasattr(camera, 'framerate'):
                        framerate_feature = camera.framerate
                        if hasattr(framerate_feature, 'value'):
                            f.write("Частота кадров: {:.2f} fps\n".format(float(framerate_feature.value)))
                        else:
                            f.write("Частота кадров недоступна\n")
                except Exception as e:
                    f.write("Ошибка при получении частоты кадров: {}\n".format(e))
                
                # Функция для получения параметров feature
                def get_feature_params(feature, name):
                    try:
                        if not hasattr(feature, 'value'):
                            return None
                            
                        result = ["{}:".format(name)]
                        
                        # Получаем режим
                        if hasattr(feature, 'mode'):
                            result.append("  Режим: {}".format(feature.mode))
                        
                        # Получаем значение
                        result.append("  Значение: {}".format(feature.value))
                        
                        # Пробуем получить границы
                        try:
                            if hasattr(feature, 'raw_info'):
                                info = feature.raw_info
                                if hasattr(info, 'min') and hasattr(info, 'max'):
                                    result.append("  Мин/Макс: {}/{}".format(info.min, info.max))
                        except Exception:
                            # Если не удалось получить границы через raw_info,
                            # пробуем получить через абсолютные значения
                            try:
                                if hasattr(feature, 'absolute_value'):
                                    abs_val = feature.absolute_value
                                    if hasattr(feature, 'absolute_boundaries'):
                                        min_val, max_val = feature.absolute_boundaries
                                        result.append("  Мин/Макс (абс.): {:.2f}/{:.2f}".format(min_val, max_val))
                            except Exception:
                                pass
                        
                        return "\n".join(result)
                    except Exception as e:
                        return "Ошибка при получении параметров {}: {}".format(name, e)
                
                # Получаем параметры для каждой функции
                for feature_name in ['exposure', 'brightness', 'gain']:
                    if hasattr(camera, feature_name):
                        feature = getattr(camera, feature_name)
                        params = get_feature_params(feature, feature_name.capitalize())
                        if params:
                            f.write(params + "\n")
                
                # Останавливаем захват
                camera.stop_capture()
                
            except AttributeError as e:
                f.write("Ошибка при получении параметров камеры: {}\n".format(e))
            except Exception as e:
                f.write("Ошибка при инициализации камеры: {}\n".format(e))
            
            f.write("\n")
            
    except Exception as e:
        print("Ошибка при записи лога настроек камеры: {}".format(e))

def check_black_image(image):
    """
    Проверяет, является ли изображение полностью или почти полностью черным.
    
    Args:
        image: Объект PIL.Image для проверки
        
    Returns:
        bool: True если изображение черное, False в противном случае
    """
    # Для монохромной камеры изображение уже должно быть в оттенках серого,
    # но для уверенности конвертируем, если это не так
    # if image.mode != 'L' and image.mode != 'I;16':
    #     image = image.convert('L')
    
    # Получаем статистику по изображению
    min_val, max_val, mean, std_dev = image.getextrema()[0], image.getextrema()[1], 0, 0
    
    # Подсчитываем среднюю яркость
    pixels = list(image.getdata())
    if pixels:
        mean = sum(pixels) / len(pixels)
        
        # Для 16-битных изображений нормализуем к шкале 0-255
        if image.mode == 'I;16':
            mean = mean / 256
    
    print("Информация о монохромном изображении - Мин: {}, Макс: {}, Среднее: {:.2f}".format(min_val, max_val, mean))
    
    # Считаем изображение черным, если среднее значение яркости менее 10 (по шкале 0-255)
    # или если максимальное значение меньше 30 (для 16-бит соответственно умножаем на 256)
    if image.mode == 'I;16':
        return mean < (10 * 256) or max_val < (30 * 256)
    else:
        return mean < 10 or max_val < 30

def main():
    """
    Основная функция скрипта с интерактивным меню.
    
    Returns:
        None
    """
    output_dir = "images"
    filename = None
    cameras = []
    bit_depth = 16  # По умолчанию используем 16-битный режим
    file_format = "tiff"  # По умолчанию сохраняем в TIFF
    
    while True:
        print_menu(output_dir, filename, bit_depth, file_format)
        choice = get_integer_input("Выберите пункт меню: ", 0, 9)
        
        if choice == 0:
            print("Выход из программы...")
            break
            
        elif choice == 1:
            print("\nСписок доступных камер:")
            cameras = list_cameras()
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 2:
            print("\nСоздание снимка с первой доступной камеры...")
            capture_image(None, output_dir, filename, bit_depth, file_format)
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 3:
            print("\nСоздание снимка с выбранной камеры...")
            cameras = list_cameras()
            if not cameras:
                print("Нет доступных камер для выбора.")
                input("\nНажмите Enter для продолжения...")
                continue
            
            try:    
                camera_index = get_integer_input("\nВыберите номер камеры: ", 0, len(cameras)-1)
                print("\nПодготовка к съемке с камеры #{}...".format(camera_index))
                print("GUID: {}".format(cameras[camera_index][0]))
                
                # Запрашиваем подтверждение перед съемкой
                confirm = input("Продолжить съемку? (y/n): ").strip().lower()
                if confirm == 'y' or confirm == '':
                    capture_image(camera_index, output_dir, filename, bit_depth, file_format)
                else:
                    print("Съемка отменена.")
            except Exception as e:
                print("Ошибка при выборе камеры: {}".format(e))
                
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 4:
            new_dir = input("\nВведите путь к папке для сохранения [{}]: ".format(output_dir))
            if new_dir.strip():
                output_dir = new_dir.strip()
            print("Установлена папка для сохранения: {}".format(output_dir))
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 5:
            print("\nТекущее имя файла: {}".format(filename if filename else "Автоматическое"))
            new_filename = input("Введите новое имя файла (пустая строка для автоматической генерации): ")
            filename = new_filename.strip() if new_filename.strip() else None
            print("Установлено имя файла: {}".format(filename if filename else "Автоматическое"))
            input("\nНажмите Enter для продолжения...")
        
        elif choice == 6:
            print("\nПодробная информация о камерах:")
            # Обновляем список камер, если он пуст
            if not cameras:
                print("Получение списка камер...")
                cameras = list_cameras()
                
            if not cameras:
                print("Камеры не обнаружены.")
            else:
                # Показываем подробную информацию для каждой камеры
                for i, cam in enumerate(cameras):
                    try:
                        print("\nКамера #{}:".format(i))
                        print("GUID: {}".format(cam[0]))
                        
                        # Подключаемся к камере для получения дополнительной информации
                        camera = Camera(guid=cam[0])
                        
                        # Записываем настройки в лог
                        log_camera_settings(camera)
                        
                        # Получаем все доступные характеристики
                        vendor_str = camera.vendor.decode('utf-8', errors='replace') if hasattr(camera, 'vendor') else 'Неизвестно'
                        model_str = camera.model.decode('utf-8', errors='replace') if hasattr(camera, 'model') else 'Неизвестно'
                        print("  Производитель: {}".format(vendor_str))
                        print("  Модель: {}".format(model_str))
                        
                        # Показываем настройки изображения
                        try:
                            print("  Разрешение: {}x{}".format(camera.width, camera.height))
                            framerate_str = camera.framerate if hasattr(camera, 'framerate') else 'Неизвестно'
                            print("  Частота кадров: {}".format(framerate_str))
                            
                            # Показываем доступные режимы
                            if hasattr(camera, 'modes') and camera.modes:
                                print("  Доступные режимы: {}".format(len(camera.modes)))
                                for j, mode in enumerate(camera.modes[:5]):  # Ограничиваем до 5 режимов
                                    print("    {}: {}".format(j, mode))
                                if len(camera.modes) > 5:
                                    print("    ... и еще {} режимов".format(len(camera.modes) - 5))
                            
                            # Показываем настройки экспозиции и яркости
                            if hasattr(camera, 'exposure'):
                                print("  Экспозиция: режим={}, мин={}, макс={}".format(
                                    camera.exposure.mode, camera.exposure.min, camera.exposure.max))
                            
                            if hasattr(camera, 'brightness'):
                                print("  Яркость: режим={}, мин={}, макс={}".format(
                                    camera.brightness.mode, camera.brightness.min, camera.brightness.max))
                            
                        except AttributeError as e:
                            print("  Ошибка при получении параметров: {}".format(e))
                        
                        # Закрываем подключение к камере
                        camera.stop_capture()
                        del camera
                        
                    except Exception as e:
                        print("  Ошибка при получении информации о камере: {}".format(e))
            
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 7:
            print("\nНастройка битности изображения:")
            print("Текущая битность: {} бит".format(bit_depth))
            print("Доступные варианты:")
            print("  1. 8 бит (стандартное качество)")
            print("  2. 16 бит (высокое качество)")
            
            bit_choice = get_integer_input("Выберите битность: ", 1, 2)
            if bit_choice == 1:
                bit_depth = 8
            else:
                bit_depth = 16
                
            print("Установлена битность: {} бит".format(bit_depth))
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 8:
            print("\nНастройка формата файла:")
            print("Текущий формат: {}".format(file_format.upper()))
            print("Доступные форматы:")
            print("  1. PNG (стандартный формат)")
            print("  2. TIFF (высокое качество, сохраняет 16 бит)")
            
            format_choice = get_integer_input("Выберите формат: ", 1, 2)
            if format_choice == 1:
                file_format = "png"
            else:
                file_format = "tiff"
                
            print("Установлен формат файла: {}".format(file_format.upper()))
            input("\nНажмите Enter для продолжения...")
            
        elif choice == 9:
            print("\nНастройка ROI для полного разрешения:")
            cameras = list_cameras()
            if not cameras:
                print("Нет доступных камер для настройки.")
                input("\nНажмите Enter для продолжения...")
                continue
            
            try:
                camera_index = get_integer_input("\nВыберите номер камеры: ", 0, len(cameras)-1)
                print("\nВыбрана камера #{}".format(camera_index))
                print("GUID: {}".format(cameras[camera_index][0]))
                
                # Определяем максимальное разрешение для выбранной камеры
                camera_model = ""
                max_res_width, max_res_height = 1624, 1224  # по умолчанию
                
                try:
                    # Временно подключаемся для получения модели камеры
                    temp_camera = Camera(guid=cameras[camera_index][0])
                    camera_model = temp_camera.model.decode('utf-8', errors='replace')
                    temp_camera.stop_capture()
                    del temp_camera
                    
                    # Устанавливаем максимальные разрешения для разных моделей
                    if "FL2-08S2M" in camera_model:
                        max_res_width, max_res_height = 1032, 776
                    elif "FL2-20S4M" in camera_model:
                        max_res_width, max_res_height = 1624, 1224
                        
                except Exception:
                    pass
                
                # Спрашиваем пользователя о разрешении
                print("\nВыберите разрешение:")
                print("  1. {}x{} (максимальное разрешение для {})".format(max_res_width, max_res_height, camera_model))
                print("  2. Пользовательское разрешение")
                
                res_choice = get_integer_input("Выберите вариант: ", 1, 2)
                
                if res_choice == 1:
                    width, height = max_res_width, max_res_height
                else:
                    width = get_integer_input("Введите ширину: ", 1, max_res_width)
                    height = get_integer_input("Введите высоту: ", 1, max_res_height)
                
                left = get_integer_input("Введите левую границу (left): ", 0, max_res_width-width)
                top = get_integer_input("Введите верхнюю границу (top): ", 0, max_res_height-height)
                
                print("\nПопытка установить ROI:")
                print("  left={}, top={}, width={}, height={}".format(left, top, width, height))
                
                if set_custom_roi(camera_index, left, top, width, height):
                    print("ROI успешно установлена!")
                    print("Теперь вы можете сделать снимок с новыми параметрами.")
                else:
                    print("Не удалось установить ROI.")
                    
            except Exception as e:
                print("Ошибка при настройке ROI: {}".format(e))
                
            input("\nНажмите Enter для продолжения...")

if __name__ == "__main__":
    print("Version 3 - Монохромная камера")
    main() 
