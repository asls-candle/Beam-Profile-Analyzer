"""
Расширенный тест для проверки корректности вычислений центроида и RMS
на аналитически известном гауссовом распределении.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from src.analysis.image_analyzer import ImageAnalyzer
import matplotlib.pyplot as plt


def create_2d_gaussian(size, sigma, center=None, amplitude=1.0):
    """
    Создаёт 2D гауссово распределение с известными параметрами.

    Args:
        size: Размер изображения (size x size)
        sigma: Стандартное отклонение (в пикселях)
        center: Центр распределения (x, y) в координатах от центра.
                По умолчанию (0, 0) - центр изображения
        amplitude: Амплитуда распределения

    Returns:
        2D массив с гауссовым распределением
    """
    if center is None:
        center = (0, 0)

    # Создаем координатную сетку (центрированную, как в MATLAB)
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    # 2D гауссово распределение
    # I(x,y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    gaussian = amplitude * np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))

    return gaussian


def test_gaussian_centered():
    """
    Тест 1: Гауссово распределение с центром в (0, 0).
    Для идеального гауссиана центроид должен совпадать с центром распределения.
    RMS должен быть равен sigma (стандартному отклонению гауссиана).
    """
    print("\n" + "="*70)
    print("Тест 1: Центрированное гауссово распределение")
    print("="*70)

    analyzer = ImageAnalyzer()

    size = 101  # Нечетный размер для четкого центра
    sigma_pixels = 10.0  # Стандартное отклонение в пикселях
    pixel_size = 0.0284  # мм (как в реальной камере)

    # Создаем идеальный гауссиан с центром в (0, 0)
    image = create_2d_gaussian(size, sigma_pixels, center=(0, 0))

    # Вычисляем центроид и RMS
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Ожидаемые значения
    expected_centroid = 0.0  # В центре
    expected_rms = sigma_pixels * pixel_size  # RMS = sigma для гауссиана

    print(f"\nПараметры теста:")
    print(f"  Размер изображения: {size}x{size} пикселей")
    print(f"  Sigma гауссиана: {sigma_pixels} пикселей = {sigma_pixels * pixel_size:.4f} мм")
    print(f"  Размер пикселя: {pixel_size} мм")

    print(f"\nРезультаты вычислений:")
    print(f"  Центроид X: {centroid_x:.6f} мм (ожидается: {expected_centroid:.6f} мм)")
    print(f"  Центроид Y: {centroid_y:.6f} мм (ожидается: {expected_centroid:.6f} мм)")
    print(f"  RMS X: {rms_x:.6f} мм (ожидается: {expected_rms:.6f} мм)")
    print(f"  RMS Y: {rms_y:.6f} мм (ожидается: {expected_rms:.6f} мм)")

    print(f"\nОтклонения от теоретических значений:")
    print(f"  Центроид X: {abs(centroid_x - expected_centroid):.8f} мм")
    print(f"  Центроид Y: {abs(centroid_y - expected_centroid):.8f} мм")
    print(f"  RMS X: {abs(rms_x - expected_rms):.8f} мм ({abs(rms_x - expected_rms)/expected_rms*100:.4f}%)")
    print(f"  RMS Y: {abs(rms_y - expected_rms):.8f} мм ({abs(rms_y - expected_rms)/expected_rms*100:.4f}%)")

    # Проверяем точность (допустимая погрешность 0.1%)
    assert abs(centroid_x) < 0.0001, f"Центроид X должен быть ~0, получено {centroid_x}"
    assert abs(centroid_y) < 0.0001, f"Центроид Y должен быть ~0, получено {centroid_y}"
    assert abs(rms_x - expected_rms) / expected_rms < 0.001, \
        f"RMS X должен быть ~{expected_rms}, получено {rms_x}"
    assert abs(rms_y - expected_rms) / expected_rms < 0.001, \
        f"RMS Y должен быть ~{expected_rms}, получено {rms_y}"

    print("\n✓ Тест пройден! Центроид и RMS совпадают с теоретическими значениями.")

    return image, centroid_x, centroid_y, rms_x, rms_y


def test_gaussian_offset():
    """
    Тест 2: Гауссово распределение со смещением.
    Центроид должен совпадать с центром распределения.
    RMS не зависит от смещения и должен быть равен sigma.
    """
    print("\n" + "="*70)
    print("Тест 2: Смещённое гауссово распределение")
    print("="*70)

    analyzer = ImageAnalyzer()

    size = 101
    sigma_pixels = 8.0
    pixel_size = 0.0284
    offset_x = 5.0  # Смещение в пикселях
    offset_y = -3.0

    # Создаем гауссиан со смещением
    image = create_2d_gaussian(size, sigma_pixels, center=(offset_x, offset_y))

    # Вычисляем центроид и RMS
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Ожидаемые значения
    expected_centroid_x = offset_x * pixel_size
    expected_centroid_y = offset_y * pixel_size
    expected_rms = sigma_pixels * pixel_size

    print(f"\nПараметры теста:")
    print(f"  Размер изображения: {size}x{size} пикселей")
    print(f"  Sigma гауссиана: {sigma_pixels} пикселей = {sigma_pixels * pixel_size:.4f} мм")
    print(f"  Смещение: ({offset_x}, {offset_y}) пикселей = ({expected_centroid_x:.4f}, {expected_centroid_y:.4f}) мм")

    print(f"\nРезультаты вычислений:")
    print(f"  Центроид X: {centroid_x:.6f} мм (ожидается: {expected_centroid_x:.6f} мм)")
    print(f"  Центроид Y: {centroid_y:.6f} мм (ожидается: {expected_centroid_y:.6f} мм)")
    print(f"  RMS X: {rms_x:.6f} мм (ожидается: {expected_rms:.6f} мм)")
    print(f"  RMS Y: {rms_y:.6f} мм (ожидается: {expected_rms:.6f} мм)")

    print(f"\nОтклонения от теоретических значений:")
    print(f"  Центроид X: {abs(centroid_x - expected_centroid_x):.8f} мм")
    print(f"  Центроид Y: {abs(centroid_y - expected_centroid_y):.8f} мм")
    print(f"  RMS X: {abs(rms_x - expected_rms):.8f} мм ({abs(rms_x - expected_rms)/expected_rms*100:.4f}%)")
    print(f"  RMS Y: {abs(rms_y - expected_rms):.8f} мм ({abs(rms_y - expected_rms)/expected_rms*100:.4f}%)")

    # Проверяем точность
    assert abs(centroid_x - expected_centroid_x) < 0.001, \
        f"Центроид X должен быть ~{expected_centroid_x}, получено {centroid_x}"
    assert abs(centroid_y - expected_centroid_y) < 0.001, \
        f"Центроид Y должен быть ~{expected_centroid_y}, получено {centroid_y}"
    assert abs(rms_x - expected_rms) / expected_rms < 0.001, \
        f"RMS X должен быть ~{expected_rms}, получено {rms_x}"
    assert abs(rms_y - expected_rms) / expected_rms < 0.001, \
        f"RMS Y должен быть ~{expected_rms}, получено {rms_y}"

    print("\n✓ Тест пройден! Центроид соответствует смещению, RMS не зависит от смещения.")

    return image


def test_gaussian_elliptical():
    """
    Тест 3: Эллиптическое гауссово распределение.
    Разные sigma по осям X и Y.
    """
    print("\n" + "="*70)
    print("Тест 3: Эллиптическое гауссово распределение")
    print("="*70)

    analyzer = ImageAnalyzer()

    size = 101
    sigma_x = 12.0  # Разные sigma по осям
    sigma_y = 6.0
    pixel_size = 0.0284

    # Создаем эллиптический гауссиан вручную
    x = np.arange(size) - (size - 1) / 2
    y = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    image = np.exp(-(xx**2 / (2*sigma_x**2) + yy**2 / (2*sigma_y**2)))

    # Вычисляем центроид и RMS
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Ожидаемые значения
    expected_centroid = 0.0
    expected_rms_x = sigma_x * pixel_size
    expected_rms_y = sigma_y * pixel_size

    print(f"\nПараметры теста:")
    print(f"  Размер изображения: {size}x{size} пикселей")
    print(f"  Sigma X: {sigma_x} пикселей = {sigma_x * pixel_size:.4f} мм")
    print(f"  Sigma Y: {sigma_y} пикселей = {sigma_y * pixel_size:.4f} мм")
    print(f"  Соотношение осей: {sigma_x/sigma_y:.2f}:1")

    print(f"\nРезультаты вычислений:")
    print(f"  Центроид X: {centroid_x:.6f} мм (ожидается: {expected_centroid:.6f} мм)")
    print(f"  Центроид Y: {centroid_y:.6f} мм (ожидается: {expected_centroid:.6f} мм)")
    print(f"  RMS X: {rms_x:.6f} мм (ожидается: {expected_rms_x:.6f} мм)")
    print(f"  RMS Y: {rms_y:.6f} мм (ожидается: {expected_rms_y:.6f} мм)")

    print(f"\nОтклонения от теоретических значений:")
    print(f"  Центроид X: {abs(centroid_x - expected_centroid):.8f} мм")
    print(f"  Центроид Y: {abs(centroid_y - expected_centroid):.8f} мм")
    print(f"  RMS X: {abs(rms_x - expected_rms_x):.8f} мм ({abs(rms_x - expected_rms_x)/expected_rms_x*100:.4f}%)")
    print(f"  RMS Y: {abs(rms_y - expected_rms_y):.8f} мм ({abs(rms_y - expected_rms_y)/expected_rms_y*100:.4f}%)")

    # Проверяем точность
    assert abs(centroid_x) < 0.0001, f"Центроид X должен быть ~0, получено {centroid_x}"
    assert abs(centroid_y) < 0.0001, f"Центроид Y должен быть ~0, получено {centroid_y}"
    assert abs(rms_x - expected_rms_x) / expected_rms_x < 0.001, \
        f"RMS X должен быть ~{expected_rms_x}, получено {rms_x}"
    assert abs(rms_y - expected_rms_y) / expected_rms_y < 0.001, \
        f"RMS Y должен быть ~{expected_rms_y}, получено {rms_y}"

    print("\n✓ Тест пройден! Эллиптическое распределение обработано корректно.")

    return image


def visualize_gaussian_test():
    """
    Визуализация результатов для визуальной проверки корректности.
    """
    print("\n" + "="*70)
    print("Визуализация результатов")
    print("="*70)

    analyzer = ImageAnalyzer()

    # Параметры
    size = 101
    sigma_pixels = 10.0
    pixel_size = 0.0284

    # Создаем гауссиан
    image = create_2d_gaussian(size, sigma_pixels, center=(0, 0))

    # Вычисляем параметры
    centroid_x, centroid_y = analyzer.calculate_centroid(image, pixel_size, pixel_size)
    rms_x, rms_y = analyzer.calculate_rms(image, pixel_size, pixel_size)

    # Получаем проекции
    x_coords, x_proj, y_coords, y_proj = analyzer.calculate_projections(
        image, pixel_size, pixel_size
    )

    # Создаем график
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 2D изображение
    ax = axes[0, 0]
    extent = [
        -(size-1)/2 * pixel_size,
        size/2 * pixel_size,
        -(size-1)/2 * pixel_size,
        size/2 * pixel_size
    ]
    im = ax.imshow(image, extent=extent, origin='lower', cmap='hot')
    ax.axhline(y=centroid_y, color='cyan', linestyle='--', label=f'Centroid Y={centroid_y:.4f}')
    ax.axvline(x=centroid_x, color='cyan', linestyle='--', label=f'Centroid X={centroid_x:.4f}')

    # RMS границы (1 sigma)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((centroid_x, centroid_y), 2*rms_x, 2*rms_y,
                     fill=False, edgecolor='lime', linestyle='--', linewidth=2,
                     label=f'1σ: ({rms_x:.4f}, {rms_y:.4f})')
    ax.add_patch(ellipse)

    ax.set_xlabel('X (мм)')
    ax.set_ylabel('Y (мм)')
    ax.set_title('2D Гауссово распределение')
    ax.legend()
    plt.colorbar(im, ax=ax)

    # Проекция X
    ax = axes[0, 1]
    ax.plot(x_coords, x_proj, 'b-', linewidth=2, label='X проекция')
    ax.axvline(x=centroid_x, color='red', linestyle='--', label=f'Centroid={centroid_x:.4f}')
    ax.axvline(x=centroid_x - rms_x, color='orange', linestyle=':', label=f'±RMS={rms_x:.4f}')
    ax.axvline(x=centroid_x + rms_x, color='orange', linestyle=':')
    ax.set_xlabel('X (мм)')
    ax.set_ylabel('Нормализованная интенсивность')
    ax.set_title('Проекция на ось X')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Проекция Y
    ax = axes[1, 0]
    ax.plot(y_coords, y_proj, 'g-', linewidth=2, label='Y проекция')
    ax.axvline(x=centroid_y, color='red', linestyle='--', label=f'Centroid={centroid_y:.4f}')
    ax.axvline(x=centroid_y - rms_y, color='orange', linestyle=':', label=f'±RMS={rms_y:.4f}')
    ax.axvline(x=centroid_y + rms_y, color='orange', linestyle=':')
    ax.set_xlabel('Y (мм)')
    ax.set_ylabel('Нормализованная интенсивность')
    ax.set_title('Проекция на ось Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Сводная таблица
    ax = axes[1, 1]
    ax.axis('off')

    info_text = f"""
    РЕЗУЛЬТАТЫ АНАЛИЗА
    {'='*40}

    Параметры изображения:
    • Размер: {size}×{size} пикселей
    • Размер пикселя: {pixel_size} мм
    • Теоретическая sigma: {sigma_pixels * pixel_size:.4f} мм

    Центроид (центр масс):
    • X: {centroid_x:.6f} мм
    • Y: {centroid_y:.6f} мм

    RMS размеры (ширина пучка):
    • X: {rms_x:.6f} мм
    • Y: {rms_y:.6f} мм

    Точность (отклонение от теории):
    • Центроид: {max(abs(centroid_x), abs(centroid_y)):.8f} мм
    • RMS: {abs(rms_x - sigma_pixels*pixel_size)/( sigma_pixels*pixel_size)*100:.4f}%

    ✓ Вычисления корректны!
    """

    ax.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
           verticalalignment='center')

    plt.tight_layout()
    plt.savefig('d:/DEV/CANDLE/Beam-Profile-Analyzer/gaussian_validation.png', dpi=150)
    print("\n✓ График сохранён в gaussian_validation.png")
    print("\nВизуальная проверка:")
    print("  - Центроид должен быть в центре распределения (синие линии)")
    print("  - RMS контур (зелёный) должен охватывать ~68% энергии пучка")

    return fig


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  РАСШИРЕННАЯ ВАЛИДАЦИЯ ВЫЧИСЛЕНИЙ ЦЕНТРОИДА И RMS  ".center(70))
    print("="*70)

    try:
        # Запускаем все тесты
        test_gaussian_centered()
        test_gaussian_offset()
        test_gaussian_elliptical()

        # Визуализация
        try:
            visualize_gaussian_test()
        except ImportError:
            print("\n⚠ matplotlib не установлен - визуализация пропущена")
            print("  Установите: pip install matplotlib")

        print("\n" + "="*70)
        print("  ✓✓✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ✓✓✓  ".center(70))
        print("  Вычисления центроида и RMS полностью корректны  ".center(70))
        print("  и соответствуют теоретическим значениям!  ".center(70))
        print("="*70)

    except AssertionError as e:
        print(f"\n✗ ОШИБКА: {e}")
        print("\nПроверьте реализацию вычислений!")
        raise
