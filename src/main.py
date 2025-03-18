#!/usr/bin/env python3
"""
Главный модуль приложения Анализатор профиля пучка
"""

import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    """Главная функция запуска приложения"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 