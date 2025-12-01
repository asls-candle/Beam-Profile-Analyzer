import matplotlib.pyplot as plt
import numpy as np
import os
import logging
logger = logging.getLogger('data')


def plot_negative_density(matrix, output_dir, output_filename):
    # Убеждаемся, что директория существует
    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Directory exist")

    output_file = os.path.join(output_dir, output_filename)

    # Извлекаем отрицательные значения из матрицы
    negative_values = matrix[matrix < 0]

    # Подсчитываем количество (не сумму!) отрицательных элементов
    negative_count = np.sum(matrix < 0)  # или len(negative_values) или negative_values.size
    total_count = matrix.size

    logger.info("Количество отрицательных значений в difference после вычитания фона: {} из {} ({:.2f}%)".format(
        negative_count, total_count, 100.0 * negative_count / total_count
        )
    )

    # Строим гистограмму плотности
    plt.figure(figsize=(8, 6))
    plt.hist(negative_values, bins=30, density=True, alpha=0.7, color='blue')
    plt.title('Density of Negative Values')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Сохраняем график в файл
    plt.savefig(output_file)
    plt.close()


def evaluate_zeros_in_shot(matrix):
    """
    Оценивает количество нулевых значений в матрице shot при импорте из MATLAB

    Args:
        matrix: Матрица данных для анализа (обычно shot из MATLAB)

    Returns:
        dict: Словарь со статистикой нулевых значений
    """
    # Подсчитываем количество нулевых элементов
    zero_count = np.sum(matrix == 0)
    total_count = matrix.size
    zero_percentage = 100.0 * zero_count / total_count if total_count > 0 else 0.0

    logger.info("Количество нулевых значений в shot при импорте из MATLAB: {} из {} ({:.2f}%)".format(
        zero_count, total_count, zero_percentage
        )
    )

    return {
        'zero_count': zero_count,
        'total_count': total_count,
        'zero_percentage': zero_percentage
    }
