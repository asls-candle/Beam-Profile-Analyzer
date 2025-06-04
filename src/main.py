#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication
import matplotlib
matplotlib.use('Qt5Agg')  # Задаем Qt5 как бэкенд для matplotlib

import logging
import logging.config
from src.config.log_config import log_config
from src.config.paths import PATH
from src.ui.main_window import MainWindow

# Создаем директорию для логов, если её нет
try:
    os.makedirs(PATH["logs_dir"])
except OSError:
    if not os.path.isdir(PATH["logs_dir"]):
        raise

# Инициализируем систему логирования
logging.config.dictConfig(log_config)

logger = logging.getLogger('app')

def main():
    """
    Главная функция запуска приложения
    """
    logger.info("Запуск приложения")
    
    # Создаем приложение Qt
    app = QApplication(sys.argv)
    
    # Устанавливаем путь к текущей директории
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Создаем и отображаем главное окно
    main_window = MainWindow()
    main_window.show()
    
    # Запускаем основной цикл приложения
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
