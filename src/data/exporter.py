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
    def export_csv(folder_path, data_dict):
        """
        Экспортирует данные в отдельную папку: JSON для метаданных и чистые CSV файлы для массивов

        Args:
            folder_path: Путь к папке для сохранения
            data_dict: Словарь с данными для экспорта

        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            # Создаем директорию для файлов
            os.makedirs(folder_path, exist_ok=True)

            # Создаем словарь с метаданными (все, кроме массивов)
            metadata = {}
            for key, value in data_dict.items():
                if not isinstance(value, np.ndarray):
                    metadata[key] = value

            # Добавляем информацию о размерах массивов в метаданные
            metadata['arrays'] = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    metadata['arrays'][key] = {
                        'shape': value.shape,
                        'dtype': str(value.dtype)
                    }

            # Экспортируем метаданные в JSON
            json_path = os.path.join(folder_path, "metadata.json")
            with open(json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(metadata, jsonfile, indent=4, default=str)

            # Экспортируем массивы в отдельные CSV файлы без метаинформации
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    array_path = os.path.join(folder_path, "{}.csv".format(key))
                    with open(array_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        # Записываем только числа без заголовков
                        if value.ndim == 2:  # Для двумерных массивов
                            for row in value:
                                writer.writerow(["{:.17g}".format(x) for x in row])
                        else:  # Для других размерностей
                            writer.writerow(["{:.17g}".format(x) for x in value.flatten()])

            print("Экспортированы данные в папку {}: метаданные и массивы в отдельных CSV файлах.".format(folder_path))
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
    def export_data(folder_path, data, export_format="folder", plot_manager=None):
        """
        Экспортирует данные в папку выбранного формата

        Args:
            folder_path: Базовый путь для создания папки с данными
            data: Словарь с данными для экспорта
            export_format: Формат экспорта ("folder" для CSV+JSON, "png" для PNG изображений)
            plot_manager: Экземпляр PlotManager для построения графиков (нужен для PNG)

        Returns:
            str: Путь к созданной папке с данными или None в случае ошибки
        """
        try:
            # Добавляем текущую дату и время
            data["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Формируем название папки с датой и временем
            base_name = os.path.basename(folder_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = "{}_{}".format(base_name_without_ext, timestamp)
            target_folder = os.path.join(os.path.dirname(folder_path), folder_name)

            # Экспортируем в зависимости от формата
            if export_format.lower() in ["folder", "csv"]:
                success = DataExporter.export_csv(target_folder, data)
                if success:
                    print("Данные успешно сохранены в папку: {}".format(target_folder))
                    return target_folder
                return None
            elif export_format.lower() == "png":
                success = DataExporter.export_png(target_folder, data, plot_manager)
                if success:
                    print("Изображения успешно сохранены в папку: {}".format(target_folder))
                    return target_folder
                return None
            else:
                print("Неизвестный формат экспорта: {}".format(export_format))
                return None
        except Exception as e:
            print("Ошибка при экспорте данных: {}".format(e))
            return None
