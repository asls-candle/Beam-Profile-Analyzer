#!/usr/bin/env python3
"""
Файл запуска Анализатора профиля пучка из корневой директории
"""

import sys
import os

# Добавляем текущую директорию в путь поиска модулей
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Импортируем главную функцию из модуля src
from src.main import main

# Запускаем приложение
if __name__ == "__main__":
    main() 