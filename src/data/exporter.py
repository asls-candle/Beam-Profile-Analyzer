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
        Экспортирует данные в формате CSV
        
        Args:
            filepath: Путь для сохранения файла
            data_dict: Словарь с данными для экспорта
            
        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Создаем директорию, если ее нет
            dir_path = os.path.dirname(filepath)
            if dir_path:
                try:
                    os.makedirs(dir_path)
                except OSError:
                    if not os.path.isdir(dir_path):
                        raise
            
            # Информационная часть (метаданные)
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Записываем метаданные
                writer.writerow(["# Metadata"])
                for key, value in data_dict.items():
                    if not isinstance(value, np.ndarray):
                        writer.writerow(["# {}".format(key), value])
                
                writer.writerow(["# Data"])
                
                # Записываем массивы данных
                for key, value in data_dict.items():
                    if isinstance(value, np.ndarray):
                        # Добавляем заголовок для массива
                        writer.writerow(["# {}".format(key), value.shape])
                        
                        # Записываем данные массива с 7 знаками после запятой
                        if value.ndim == 2:  # Для двумерных массивов
                            for row in value:
                                writer.writerow(["{:.7f}".format(x) for x in row])
                        else:  # Для других размерностей
                            writer.writerow(["{:.7f}".format(x) for x in value.flatten()])
                        
                        writer.writerow(["# End of", key])
                        
            return True
        except Exception as e:
            print("Ошибка при экспорте CSV файла: {}".format(e))
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
                    filepath = os.path.join(folder_path, f"{name}.png")
                    fig.savefig(filepath, dpi=dpi, bbox_inches=None, pad_inches=0)
                    plt.close(fig)
                    
                    # Проверяем размеры созданного файла для подтверждения
                    with Image.open(filepath) as img:
                        actual_width, actual_height = img.size
                        print(f"Экспортировано изображение {name}.png размером {actual_width}x{actual_height} " 
                              f"(ожидаемый размер: {width}x{height})")
                        
                        # Если размеры не совпадают, выводим предупреждение
                        if actual_width != width or actual_height != height:
                            print(f"ВНИМАНИЕ: Размеры экспортированного изображения {name}.png не соответствуют " 
                                  f"размерам данных с камеры!")
            
            return True
        except Exception as e:
            print(f"Ошибка при экспорте PNG файлов: {e}")
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
