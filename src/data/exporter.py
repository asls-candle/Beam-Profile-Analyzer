import os
import json
import numpy as np
import csv
import datetime
import cv2
from matplotlib import pyplot as plt

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
                    # Сохраняем тепловую карту
                    if plot_manager:
                        fig = plot_manager.create_heatmap_figure(
                            data, 
                            data_dict.get("pixel_size_x", 1), 
                            data_dict.get("pixel_size_y", 1)
                        )
                        plot_manager.save_figure(
                            fig, 
                            os.path.join(folder_path, f"{name}.png")
                        )
                    else:
                        # Если plot_manager не доступен, сохраняем с помощью OpenCV
                        # Нормализуем изображение к формату 8-bit
                        normalized = (data * 255).astype(np.uint8)
                        
                        # Применяем цветовую карту
                        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                        
                        # Сохраняем изображение
                        cv2.imwrite(os.path.join(folder_path, f"{name}.png"), colored)
            
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
