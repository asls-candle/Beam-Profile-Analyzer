# src/data/exporter.py

import os
import json
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def _to_plain_array(arr):
    """
    Возвращает plain numpy ndarray из любого массивоподобного объекта.

    MaskedArray.filled(0) заменяет маскированные элементы нулями и возвращает
    обычный ndarray.  Для plain ndarray возвращает arr без копирования.
    Это гарантирует корректную итерацию в export_csv и корректный imshow в
    export_png независимо от того, что вернул ImageNormalizer.
    """
    if isinstance(arr, np.ma.MaskedArray):
        return arr.filled(0)
    return np.asarray(arr)


class DataExporter:
    """
    Класс для экспорта данных профиля пучка
    """

    @staticmethod
    def export_csv(folder_path, data_dict):
        """
        Экспортирует данные в отдельную папку: JSON для метаданных и чистые CSV файлы для массивов.

        Скалярные значения (в том числе rms_x, rms_y) автоматически попадают
        в metadata.json, так как они не являются numpy-массивами.

        Args:
            folder_path: Путь к папке для сохранения
            data_dict: Словарь с данными для экспорта

        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            os.makedirs(folder_path, exist_ok=True)

            # Метаданные — всё кроме массивов
            metadata = {}
            for key, value in data_dict.items():
                if not isinstance(value, np.ndarray):
                    metadata[key] = value

            # Размеры массивов добавляем в метаданные
            metadata['arrays'] = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    metadata['arrays'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(np.asarray(value).dtype)
                    }

            json_path = os.path.join(folder_path, "metadata.json")
            with open(json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(metadata, jsonfile, indent=4, default=str)

            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    # Конвертируем в plain ndarray, чтобы итерация по строкам
                    # всегда давала обычные float, а не ma.masked.
                    plain = np.asarray(value)
                    array_path = os.path.join(folder_path, "{}.csv".format(key))
                    with open(array_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if plain.ndim == 2:
                            for row in plain:
                                writer.writerow(["{:.17f}".format(float(x)) for x in row])
                        else:
                            writer.writerow(["{:.17f}".format(float(x)) for x in plain.flatten()])

            print("Экспортированы данные в папку {}: метаданные и массивы в отдельных CSV файлах.".format(folder_path))
            return True

        except Exception as e:
            print("Ошибка при экспорте файлов: {}".format(e))
            return False

    @staticmethod
    def apply_colormap(data):
        """
        Применяет цветовую карту jet к данным.

        Args:
            data: Двумерный массив значений светимости

        Returns:
            np.ndarray: RGB изображение с применённой цветовой картой
        """
        plain = np.asarray(data)
        if plain.max() > plain.min():
            normalized = (plain - plain.min()) / (plain.max() - plain.min())
        else:
            normalized = np.zeros_like(plain)

        colored = cm.jet(normalized)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    @staticmethod
    def export_png(folder_path, data_dict, plot_manager):
        """
        Экспортирует данные в виде PNG изображений с точным соответствием разрешению камеры.

        Все три изображения (shot, background, difference) сохраняются с одинаковой
        цветовой шкалой vmin=0, vmax=1 — той же, что использует PlotManager в интерфейсе.
        Это гарантирует визуальное соответствие экспортированных файлов тому, что видно
        на экране.

        Args:
            folder_path: Путь к папке для сохранения
            data_dict: Словарь с данными для экспорта
            plot_manager: Экземпляр PlotManager для построения графиков

        Returns:
            bool: True если экспорт успешен, иначе False
        """
        try:
            os.makedirs(folder_path, exist_ok=True)

            metadata = {
                "resolution": data_dict.get("resolution", (0, 0)),
                "pixel_size": {
                    "x": data_dict.get("pixel_size_x", 0),
                    "y": data_dict.get("pixel_size_y", 0)
                },
                "date": data_dict.get("date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "beam": {
                    "rms_x_mm": data_dict.get("rms_x", None),
                    "rms_y_mm": data_dict.get("rms_y", None),
                },
            }
            with open(os.path.join(folder_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=4)

            images_data = [
                ("shot",       data_dict.get("shot")),
                ("background", data_dict.get("background")),
                ("difference", data_dict.get("difference")),
            ]

            for name, data in images_data:
                if data is None:
                    continue

                # Конвертируем в plain ndarray (MaskedArray → filled(0))
                # чтобы imshow не получал masked-значения и не менял отображение
                plain = np.asarray(data)

                height, width = plain.shape

                plt.ioff()
                dpi = 100
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
                ax = plt.Axes(fig, [0, 0, 1, 1])
                ax.set_axis_off()
                fig.add_axes(ax)

                # vmin=0, vmax=1 — единая шкала для всех трёх изображений,
                # совпадает с тем, что показывает PlotManager в интерфейсе.
                # Без этого matplotlib авто-растягивает каждое изображение
                # на собственный диапазон, что даёт другую картинку для фона
                # (его значения обычно << 1.0 из-за нормализации с reference_image).
                ax.imshow(plain, cmap='jet', interpolation='nearest',
                          origin='lower', vmin=0.0, vmax=1.0)

                filepath = os.path.join(folder_path, "{}.png".format(name))
                fig.savefig(filepath, dpi=dpi, bbox_inches=None, pad_inches=0)
                plt.close(fig)

                with Image.open(filepath) as img:
                    actual_width, actual_height = img.size
                    print("Экспортировано изображение {}.png размером {}x{} "
                          "(ожидаемый размер: {}x{})".format(
                              name, actual_width, actual_height, width, height))
                    if actual_width != width or actual_height != height:
                        print("ВНИМАНИЕ: Размеры {}.png не соответствуют данным камеры!".format(name))

            return True

        except Exception as e:
            print("Ошибка при экспорте PNG файлов: {}".format(e))
            return False

    @staticmethod
    def export_data(folder_path, data, export_format="folder", plot_manager=None):
        """
        Экспортирует данные в папку выбранного формата.

        Args:
            folder_path: Базовый путь для создания папки с данными
            data: Словарь с данными для экспорта
            export_format: Формат экспорта ("folder"/"csv" для CSV+JSON, "png" для PNG)
            plot_manager: Экземпляр PlotManager (нужен для PNG)

        Returns:
            str: Путь к созданной папке с данными или None в случае ошибки
        """
        try:
            data["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            base_name = os.path.basename(folder_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = "{}_{}".format(base_name_without_ext, timestamp)
            target_folder = os.path.join(os.path.dirname(folder_path), folder_name)

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