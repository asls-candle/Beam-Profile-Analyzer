"""
Тест корректности расчёта центроида и RMS размеров пучка.
Проверяет вычисления на симметричных и смещённых распределениях.
"""

import numpy as np
from src.analysis.image_analyzer import ImageAnalyzer


def test_centroid_and_rms_42():
    """Test centroid and RMS calculation for a simple symmetric 5x5 distribution."""

    analyzer = ImageAnalyzer()

    # Create a simple 5x5 vertical stripe distribution
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=float)

    pixel_size = 1  # mm

    # Compute centroid
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)

    # Compute RMS
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    print("=== Test: 5x5 Distribution ===")
    print(f"Image shape: {image.shape}")
    print(f"Pixel size: {pixel_size} mm")

    print("\nCentroid:")
    print(f"  X: {centroid_x:.17f} mm")
    print(f"  Y: {centroid_y:.17f} mm")

    print("\nRMS:")
    print(f"  X: {rms_x:.17f} mm")
    print(f"  Y: {rms_y:.17f} mm")


def test_centroid_and_rms():
    """Тест расчёта центроида и RMS для простого симметричного распределения"""

    analyzer = ImageAnalyzer()

    # Создаём простое симметричное распределение 5x5 с пиком в центре
    # Интенсивность: центральный пиксель - максимум, остальные - спад
    image = np.array([
        [0, 1, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [2, 3, 4, 3, 2],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 1, 0]
    ], dtype=float)

    pixel_size = 0.0284  # мм (типичный размер пикселя камеры)

    # Рассчитываем центроид
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)

    # Рассчитываем RMS
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    print("=== Тест симметричного распределения 5x5 ===")
    print(f"Размер изображения: {image.shape}")
    print(f"Размер пикселя: {pixel_size} мм")
    print(f"\nЦентроид:")
    print(f"  X: {centroid_x:.6f} мм")
    print(f"  Y: {centroid_y:.6f} мм")
    print(f"\nRMS размеры:")
    print(f"  X: {rms_x:.6f} мм")
    print(f"  Y: {rms_y:.6f} мм")

    # Для симметричного распределения с пиком в центре, центроид должен быть близок к 0
    assert abs(centroid_x) < 0.001, f"Центроид X должен быть ~0, получено {centroid_x}"
    assert abs(centroid_y) < 0.001, f"Центроид Y должен быть ~0, получено {centroid_y}"

    # RMS должны быть одинаковыми для симметричного распределения
    assert abs(rms_x - rms_y) < 0.001, f"RMS X и Y должны быть одинаковыми, X={rms_x}, Y={rms_y}"

    print("\n[OK] Test passed!")

def test_centroid_offset():
    """Тест расчёта центроида для смещённого распределения"""

    analyzer = ImageAnalyzer()

    # Создаём распределение 10x10 со смещением вправо
    image = np.zeros((10, 10), dtype=float)
    # Пик смещён вправо (индекс 7, что соответствует x_coord = 7 - 4.5 = 2.5 пикселя от центра)
    image[5, 7] = 10  # Центр по Y, смещение по X
    image[4:7, 6:9] = 5  # Окружение

    pixel_size = 1.0  # мм

    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)

    print("\n=== Тест смещённого распределения 10x10 ===")
    print(f"Пик в пикселе [5, 7]")
    print(f"Координата в центрированной системе: x = 7 - (10-1)/2 = {7 - (10-1)/2}")
    print(f"\nЦентроид:")
    print(f"  X: {centroid_x:.17f} мм (должен быть положительным)")
    print(f"  Y: {centroid_y:.17f} мм (должен быть ~0)")

    # Центроид X должен быть положительным (пик справа от центра)
    assert centroid_x > 1.0, f"Центроид X должен быть > 1.0, получено {centroid_x}"
    # Центроид Y должен быть близок к 0.5 (пик в пикселе 5 = координата 0.5)
    assert abs(centroid_y - 0.5) < 0.2, f"Центроид Y должен быть ~0.5, получено {centroid_y}"

    print("\n[OK] Test passed!")

def test_coordinate_system():
    """Тест центрированной системы координат"""

    print("\n=== Тест системы координат ===")

    # Проверяем для разных размеров
    sizes = [5, 10, 11]

    for size in sizes:
        # Создаём центрированную систему координат: (-(N-1)/2):(N/2)
        x_coords = np.arange(size) - (size - 1) / 2

        # Проверяем ожидаемые значения
        expected_start = -(size - 1) / 2
        expected_end = size / 2
        expected_coords = np.arange(expected_start, expected_end, 1)

        print(f"\nРазмер {size}:")
        print(f"  Координаты: {x_coords}")
        print(f"  Диапазон:   [{x_coords[0]:.1f}, {x_coords[-1]:.1f}]")
        print(f"  Центр:      {np.mean([x_coords[0], x_coords[-1]]):.1f}")

        # Проверяем правильность центрирования
        np.testing.assert_array_almost_equal(x_coords, expected_coords,
                                             err_msg=f"Система координат некорректна для размера {size}")

        # Центр должен быть в нуле (с учётом округления)
        center = np.mean([x_coords[0], x_coords[-1]])
        assert abs(center) < 0.01, f"Центр должен быть в 0, получено {center}"

    print("\n[OK] Test passed!")

if __name__ == "__main__":
    test_centroid_and_rms_42()
    # test_coordinate_system()
    # test_centroid_and_rms()
    # test_centroid_offset()

    # print("\n" + "="*50)
    # print("Все тесты успешно пройдены!")
    # print("Вычисления центроида и RMS корректны")
    # print("="*50)
