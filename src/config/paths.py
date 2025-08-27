"""В данном модуле собраны все пути, используемые в программе.
"""
import os
from typing import Callable, Tuple, Dict, Union

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_file_path(folders: Tuple[str, ...], filename: str, extension: str) -> str:
    """
    Возвращает путь к файлу в указанных подпапках с заданным именем и расширением.
    
    Args:
        folders: Кортеж директорий для формирования пути
        filename: Имя файла без расширения
        extension: Расширение файла
        
    Returns:
        Строка с полным путем к файлу
    """
    nested_path = os.path.join(BASE_DIR, *folders)
    return os.path.join(nested_path, "{}.{}".format(filename, extension))


def get_folder_path(folders: Tuple[str, ...]) -> str:
    """
    Возвращает путь к папке в указанных подпапках.
    
    Args:
        folders: Кортеж директорий для формирования пути
        
    Returns:
        Строка с полным путем к директории
    """
    return os.path.join(BASE_DIR, *folders)


def get_file_log_path(module_name):
    """
    Возвращает путь к файлу лога для конкретного модуля.
    
    Args:
        module_name: Имя модуля (файла) без расширения
        
    Returns:
        Строка с полным путем к файлу лога
    """
    file_logs_dir = os.path.join(BASE_DIR, "logs", "files")
    try:
        os.makedirs(file_logs_dir)
    except OSError:
        if not os.path.isdir(file_logs_dir):
            raise
    return os.path.join(file_logs_dir, "{}.log".format(module_name))


# Тип: Dict[str, Union[str, Callable[..., str]]]
PATH = {
    # Директории для логов
    "logs_dir": os.path.join(BASE_DIR, "logs"),
    "file_logs_dir": os.path.join(BASE_DIR, "logs", "files"),
    "app_log": os.path.join(BASE_DIR, "logs", "app.log"),
    "error_log": os.path.join(BASE_DIR, "logs", "error.log"),
    
    # Директории для данных
    "data_dir": os.path.join(BASE_DIR, "src", "data"),
    "temp_dir": os.path.join(BASE_DIR, "temp"),
    
    # Пути к файлам конфигурации
    "config_dir": os.path.join(BASE_DIR, "src", "config"),
    "settings": os.path.join(BASE_DIR, "src", "config", "settings.json"),
    
    # Пути к результатам анализа
    "results_dir": os.path.join(BASE_DIR, "results"),
    
    # Динамические пути
    "dynamic_result_file": lambda filename: os.path.join(
        BASE_DIR, "results", "{}.txt".format(filename)
    ),
    "dynamic_data_file": lambda filename, ext="csv": os.path.join(
        BASE_DIR, "src", "data", "{}.{}".format(filename, ext)
    ),
    "dynamic_file_log": lambda module_name: get_file_log_path(module_name),
}
