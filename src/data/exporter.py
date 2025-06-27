import os
import json
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

class DataExporter:
    """
    Класс для экспорта данных профиля пучка
    """
    
    @staticmethod
    def export_csv(filepath, data_dict):
        """
        Экспортирует данные в формате JSON для метаданных и отдельных CSV файлов для каждого массива
        
        Args:
            filepath: Базовый путь для сохранения файлов
            data_dict: Словарь с данными для экспорта
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Создаем директорию для файлов если ее нет
            base_dir = os.path.dirname(filepath)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            
            # Получаем базовое имя файла без расширения
            base_filename = os.path.splitext(os.path.basename(filepath))[0]
            
            # Создаем словарь с метаданными (все, кроме массивов)
            metadata = {}
            for key, value in data_dict.items():
                if not isinstance(value, np.ndarray):
                    metadata[key] = value
            
            # Экспортируем метаданные в JSON
            json_path = os.path.join(base_dir, f"{base_filename}_metadata.json")
            with open(json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(metadata, jsonfile, indent=4, default=str)
            
            # Экспортируем массивы в отдельные CSV файлы
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    array_path = os.path.join(base_dir, f"{base_filename}_{key}.csv")
                    with open(array_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        # Записываем информацию о массиве
                        writer.writerow(["# Array", key])
                        writer.writerow(["# Shape", str(value.shape)])
                        
                        # Записываем данные массива с 7 знаками после запятой
                        if value.ndim == 2:  # Для двумерных массивов
                            for row in value:
                                writer.writerow(["{:.7f}".format(x) for x in row])
                        else:  # Для других размерностей
                            writer.writerow(["{:.7f}".format(x) for x in value.flatten()])
            
            print(f"Экспортированы: метаданные в {json_path} и массивы в отдельные CSV файлы.")
            return True
            
        except Exception as e:
            print("Ошибка при экспорте файлов: {}".format(e))
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
        # Нормализуем данные к диапазону [0, 1]
        if data.max() > data.min():
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            normalized = np.zeros_like(data)
            
        # Применяем цветовую карту jet через matplotlib
        colored = cm.jet(normalized)
        
        # Преобразуем в RGB формат (убираем альфа-канал)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return colored_rgb
            
    @staticmethod
    def export_png(folder_path, data_dict, plot_manager):
        """
        Экспортирует данные в виде PNG изображений с точным соответствием разрешению камеры
        
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
                    # Получаем точные размеры данных с камеры
                    height, width = data.shape
                    
                    # Важно: отключаем все автоматические настройки matplotlib
                    plt.ioff()  # Отключаем интерактивный режим
                    
                    # Создаем фигуру с точным размером в пикселях
                    dpi = 100  # фиксированное значение DPI для расчета
                    # Размер фигуры в дюймах = размер в пикселях / DPI
                    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False)
                    
                    # Добавляем ось, занимающую всю область фигуры
                    ax = plt.Axes(fig, [0, 0, 1, 1])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    
                    # Отображаем данные с отключенной интерполяцией для сохранения пиксельной точности
                    ax.imshow(data, cmap='jet', interpolation='nearest', origin='lower')
                    
                    # Сохраняем изображение с точными настройками
                    filepath = os.path.join(folder_path, "{}.png".format(name))
                    fig.savefig(filepath, dpi=dpi, bbox_inches=None, pad_inches=0)
                    plt.close(fig)
                    
                    # Проверяем размеры созданного файла для подтверждения
                    with Image.open(filepath) as img:
                        actual_width, actual_height = img.size
                        print("Экспортировано изображение {}.png размером {}x{} ".format(name, actual_width, actual_height) 
                              + "(ожидаемый размер: {}x{})".format(width, height))
                        
                        # Если размеры не совпадают, выводим предупреждение
                        if actual_width != width or actual_height != height:
                            print("ВНИМАНИЕ: Размеры экспортированного изображения {}.png не соответствуют ".format(name) 
                                  + "размерам данных с камеры!")
            
            return True
        except Exception as e:
            print("Ошибка при экспорте PNG файлов: {}".format(e))
            return False
            
    @staticmethod
    def export_data(filepath, data, export_format="csv", plot_manager=None):
        """
        Экспортирует данные в указанном формате
        
        Args:
            filepath: Путь для сохранения файла/директории
            data: Словарь с данными для экспорта
            export_format: Формат экспорта ("csv", "png")
            plot_manager: Экземпляр PlotManager для построения графиков (нужен для PNG)
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Добавляем текущую дату и время
            data["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Экспортируем в зависимости от формата
            if export_format.lower() == "csv":
                return DataExporter.export_csv(filepath, data)
            elif export_format.lower() == "png":
                # Берем базовое имя файла без расширения
                base_name = os.path.basename(filepath)
                # Удаляем расширение файла, если оно есть
                base_name_without_ext = os.path.splitext(base_name)[0]
                
                # Создаем папку с датой и временем в названии, используя разделитель между датой и временем
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                folder_name = "{}_{}".format(base_name_without_ext, timestamp)
                folder_path = os.path.join(os.path.dirname(filepath), folder_name)
                
                return DataExporter.export_png(folder_path, data, plot_manager)
            else:
                print("Неизвестный формат экспорта: {}".format(export_format))
                return False
        except Exception as e:
            print("Ошибка при экспорте данных: {}".format(e))
            return False
