import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QFrame
from PyQt6.QtCore import Qt
import pyqtgraph as pg

class BeamProfileWidget(QWidget):
    """Базовый класс виджета для отображения профиля пучка"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Настраиваем внешний вид графиков
        pg.setConfigOption('background', 'w')  # Белый фон
        pg.setConfigOption('foreground', 'k')  # Черные линии
        
        # Создаем основной layout
        self.main_layout = QGridLayout(self)
        
        # Создаем виджет для тепловой карты
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        
        # Создаем графики для профилей
        self.x_profile_plot = pg.PlotWidget()
        self.y_profile_plot = pg.PlotWidget()
        
        # Настраиваем графики
        self.x_profile_plot.getPlotItem().hideAxis('right')
        self.y_profile_plot.getPlotItem().hideAxis('top')
        
        # Переворачиваем Y-профиль и меняем оси местами
        self.y_profile_plot.getPlotItem().setLabel('bottom', 'Интенсивность')
        self.y_profile_plot.getPlotItem().setLabel('left', 'Y (мм)')
        self.y_profile_plot.setYRange(-10, 10)  # Будет обновлено при загрузке данных
        
        self.x_profile_plot.getPlotItem().setLabel('bottom', 'X (мм)')
        self.x_profile_plot.getPlotItem().setLabel('left', 'Интенсивность')
        self.x_profile_plot.setXRange(-10, 10)  # Будет обновлено при загрузке данных
        
        # Создаем виджет информации
        self.info_widget = QFrame()
        self.info_widget.setFrameShape(QFrame.Shape.StyledPanel)
        self.info_widget.setFrameShadow(QFrame.Shadow.Raised)
        
        self.info_layout = QVBoxLayout(self.info_widget)
        
        # Добавляем заголовок
        self.title_label = QLabel("Информация о пучке")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.info_layout.addWidget(self.title_label)
        
        # Добавляем Labels для информации
        self.centroid_x_label = QLabel("Центроид X: N/A мм")
        self.centroid_y_label = QLabel("Центроид Y: N/A мм")
        self.rms_x_label = QLabel("RMS X: N/A мм")
        self.rms_y_label = QLabel("RMS Y: N/A мм")
        
        self.info_layout.addWidget(self.centroid_x_label)
        self.info_layout.addWidget(self.centroid_y_label)
        self.info_layout.addWidget(self.rms_x_label)
        self.info_layout.addWidget(self.rms_y_label)
        
        # Добавляем все в основной layout
        # Основная тепловая карта в центре
        self.main_layout.addWidget(self.image_view, 0, 1)
        
        # График X-профиля внизу
        self.main_layout.addWidget(self.x_profile_plot, 1, 1)
        
        # График Y-профиля слева
        self.main_layout.addWidget(self.y_profile_plot, 0, 0)
        
        # Информационный виджет внизу слева
        self.main_layout.addWidget(self.info_widget, 1, 0)
        
        # Настраиваем растяжение виджетов
        self.main_layout.setColumnStretch(0, 1)  # Y-профиль
        self.main_layout.setColumnStretch(1, 4)  # Тепловая карта и X-профиль
        
        self.main_layout.setRowStretch(0, 4)  # Тепловая карта и Y-профиль
        self.main_layout.setRowStretch(1, 1)  # X-профиль и информация
        
        # Инициализируем переменные для данных
        self.data = None
        self.x_profile = None
        self.y_profile = None
        self.x_mm = None
        self.y_mm = None
        
    def update_data(self, data, x_mm, y_mm, centroid_x, centroid_y, rms_x, rms_y, x_profile, y_profile):
        """Обновление данных и графиков"""
        self.data = data
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.x_profile = x_profile
        self.y_profile = y_profile
        
        # Обновляем тепловую карту
        self.image_view.setImage(data)
        
        # Настраиваем оси тепловой карты
        self.image_view.getView().setAspectLocked(False)
        
        # Настраиваем диапазоны осей
        x_min, x_max = min(x_mm), max(x_mm)
        y_min, y_max = min(y_mm), max(y_mm)
        
        # Устанавливаем пределы для тепловой карты
        self.image_view.setRect(pg.QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
        
        # Обновляем графики профилей
        self.x_profile_plot.clear()
        self.y_profile_plot.clear()
        
        # X-профиль
        self.x_profile_plot.plot(x_mm, x_profile, pen='g')
        self.x_profile_plot.setXRange(x_min, x_max)
        
        # Y-профиль (инвертируем оси для соответствия с тепловой картой)
        self.y_profile_plot.plot(y_profile, y_mm, pen='g')
        self.y_profile_plot.setYRange(y_min, y_max)
        
        # Обновляем информацию
        self.centroid_x_label.setText(f"Центроид X: {centroid_x:.3f} мм")
        self.centroid_y_label.setText(f"Центроид Y: {centroid_y:.3f} мм")
        self.rms_x_label.setText(f"RMS X: {rms_x:.3f} мм")
        self.rms_y_label.setText(f"RMS Y: {rms_y:.3f} мм")
        
    def clear(self):
        """Очистка данных и графиков"""
        self.image_view.clear()
        self.x_profile_plot.clear()
        self.y_profile_plot.clear()
        
        self.centroid_x_label.setText("Центроид X: N/A мм")
        self.centroid_y_label.setText("Центроид Y: N/A мм")
        self.rms_x_label.setText("RMS X: N/A мм")
        self.rms_y_label.setText("RMS Y: N/A мм")
        
        self.data = None
        self.x_profile = None
        self.y_profile = None 