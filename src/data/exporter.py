import os
import json
import numpy as np
import csv
import datetime
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class DataExporter:
    """
    Класс для экспорта данных профиля пучка
    """
    
    @staticmethod
    def export_npy(filepath, data_dict):
        """
        Экспортирует данные в формате NPY
        
        Args:
            filepath: Путь для сохранения файла
            data_dict: Словарь с данными для экспорта
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Убеждаемся, что все массивы numpy в словаре
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    data_dict[key] = value
                    
            # Сохраняем словарь
            np.save(filepath, data_dict)
            return True
        except Exception as e:
            print(f"Ошибка при экспорте NPY файла: {e}")
            return False
            
    @staticmethod
    def export_csv(filepath, data_dict):
        """
        Экспортирует данные в формате CSV
        
        Args:
            filepath: Путь для сохранения файла
            data_dict: Словарь с данными для экспорта
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Создаем директорию, если ее нет
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Информационная часть (метаданные)
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Записываем метаданные
                writer.writerow(["# Metadata"])
                for key, value in data_dict.items():
                    if not isinstance(value, np.ndarray):
                        writer.writerow([f"# {key}", value])
                
                writer.writerow(["# Data"])
                
                # Записываем массивы данных
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        # Добавляем заголовок для массива
                        writer.writerow([f"# {key}", value.shape])
                        
                        # Записываем данные массива с 7 знаками после запятой
                        if value.ndim == 2:  # Для двумерных массивов
                            for row in value:
                                writer.writerow([f"{x:.7f}" for x in row])
                        else:  # Для других размерностей
                            writer.writerow([f"{x:.7f}" for x in value.flatten()])
                        
                        writer.writerow(["# End of", key])
                        
            return True
        except Exception as e:
            print(f"Ошибка при экспорте CSV файла: {e}")
            return False
    
    @staticmethod
    def apply_colormap(data):
        """
        Применяет цветовую карту jet к данным
        
        Args:
            data: Двумерный массив значений светимости
            
        Returns:
            np.ndarray: RGB изображение с применённой цветовой картой
        """
        normalized = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else np.zeros_like(data)
        
        # Преобразуем в uint8 для OpenCV
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        
        # Применяем цветовую карту jet
        colored = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
        
        # Конвертируем из BGR в RGB (OpenCV использует BGR)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored_rgb
            
    @staticmethod
    def export_png(folder_path, data_dict, plot_manager):
        """
        Экспортирует данные в виде PNG изображений и JSON с метаданными
        
        Args:
            folder_path: Путь к папке для сохранения
            data_dict: Словарь с данными для экспорта
            plot_manager: Экземпляр PlotManager для построения графиков
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Создаем папку если ее нет
            os.makedirs(folder_path, exist_ok=True)
            
            # Сохраняем метаданные в JSON
            metadata = {
                "resolution": data_dict.get("resolution", (0, 0)),
                "pixel_size": {
                    "x": data_dict.get("pixel_size_x", 0),
                    "y": data_dict.get("pixel_size_y", 0)
                },
                "date": data_dict.get("date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }
            
            with open(os.path.join(folder_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Сохраняем изображения
            images_data = [
                ("shot", data_dict.get("shot")),
                ("background", data_dict.get("background")),
                ("difference", data_dict.get("difference"))
            ]
            
            for name, data in images_data:
                if data is not None:
                    # Всегда используем PIL для сохранения изображений
                    # Получаем размеры изображения
                    height, width = data.shape
                    
                    # Создаем цветное изображение из данных
                    colored_data = DataExporter.apply_colormap(data)
                    
                    # Создаем изображение PIL с исходными размерами пикселей
                    img = Image.fromarray(colored_data)
                    
                    # Сохраняем изображение без изменения пропорций
                    img.save(os.path.join(folder_path, f"{name}.png"))
                    
                    # Выводим размеры для диагностики
                    print(f"Экспортировано изображение {name}.png с размерами {width}x{height} пикселей")
            
            return True
        except Exception as e:
            print(f"Ошибка при экспорте PNG файлов: {e}")
            return False
            
    @staticmethod
    def export_data(filepath, data, export_format="npy", plot_manager=None):
        """
        Экспортирует данные в указанном формате
        
        Args:
            filepath: Путь для сохранения файла/директории
            data: Словарь с данными для экспорта
            export_format: Формат экспорта ("npy", "csv", "png")
            plot_manager: Экземпляр PlotManager для построения графиков (нужен для PNG)
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Добавляем текущую дату и время
            data["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Экспортируем в зависимости от формата
            if export_format.lower() == "npy":
                return DataExporter.export_npy(filepath, data)
            elif export_format.lower() == "csv":
                return DataExporter.export_csv(filepath, data)
            elif export_format.lower() == "png":
                # Создаем папку с датой и временем в названии
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"{os.path.basename(filepath)}_{timestamp}"
                folder_path = os.path.join(os.path.dirname(filepath), folder_name)
                
                return DataExporter.export_png(folder_path, data, plot_manager)
            else:
                print(f"Неизвестный формат экспорта: {export_format}")
                return False
        except Exception as e:
            print(f"Ошибка при экспорте данных: {e}")
            return False
