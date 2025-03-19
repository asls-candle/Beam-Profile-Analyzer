import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QFrame
from PyQt6.QtCore import Qt, QRectF
import pyqtgraph as pg
import logging

class BeamProfileWidget(QWidget):
    """Базовый класс виджета для отображения профиля пучка"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Настраиваем внешний вид графиков
        pg.setConfigOption('background', 'w')  # Белый фон
        pg.setConfigOption('foreground', 'k')  # Черные линии
        
        # Создаем основной layout без отступов и пространства
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Создаем виджеты для графиков
        self.y_profile_plot = pg.PlotWidget()  # Профиль по Y (слева)
        self.image_view = pg.ImageView()       # Тепловая карта (центр)
        self.x_profile_plot = pg.PlotWidget()  # Профиль по X (внизу)
        self.info_widget = QFrame()            # Информация (внизу слева)
        
        # Отключаем ненужные элементы управления в ImageView
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        
        # Перемещаем гистограмму в правую часть макета
        # Создаем виджет для гистограммы отдельно, чтобы переместить его вправо 
        self.histogram_widget = QWidget()
        histogram_layout = QVBoxLayout(self.histogram_widget)
        histogram_layout.setContentsMargins(0, 0, 0, 0)
        # Извлекаем гистограмму из ImageView и помещаем её в наш виджет
        self.image_view.ui.histogram.setParent(self.histogram_widget)
        histogram_layout.addWidget(self.image_view.ui.histogram)
        
        # Настраиваем стили для графиков
        for plot in [self.y_profile_plot, self.x_profile_plot]:
            plot.setBackground('w')
            plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Настраиваем представления осей
        self.x_profile_plot.showAxis('right', False)
        self.y_profile_plot.showAxis('top', False)
        
        # Настраиваем Y-профиль: по X - интенсивность, по Y - координаты Y
        self.y_profile_plot.setLabel('bottom', 'Интенсивность')
        self.y_profile_plot.setLabel('left', 'Y (мм)')
        
        # Настраиваем X-профиль: по X - координаты X, по Y - интенсивность
        self.x_profile_plot.setLabel('bottom', 'X (мм)')
        self.x_profile_plot.setLabel('left', 'Интенсивность')
        
        # Настраиваем информационную панель
        self.info_widget.setFrameShape(QFrame.Shape.StyledPanel)
        self.info_widget.setStyleSheet("background-color: white;")
        
        self.info_layout = QVBoxLayout(self.info_widget)
        self.info_layout.setContentsMargins(5, 5, 5, 5)
        
        # Добавляем заголовок информационной панели
        self.title_label = QLabel("Информация о пучке")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.info_layout.addWidget(self.title_label)
        
        # Добавляем метки для информации о пучке
        self.centroid_x_label = QLabel("Центроид X: N/A мм")
        self.centroid_y_label = QLabel("Центроид Y: N/A мм")
        self.rms_x_label = QLabel("RMS X: N/A мм")
        self.rms_y_label = QLabel("RMS Y: N/A мм")
        
        for label in [self.centroid_x_label, self.centroid_y_label, self.rms_x_label, self.rms_y_label]:
            self.info_layout.addWidget(label)
        
        # Размещаем компоненты в сетке с правильной привязкой
        self.main_layout.addWidget(self.y_profile_plot, 0, 0)  # Y-профиль в левой колонке, верхняя строка
        self.main_layout.addWidget(self.image_view, 0, 1)     # Тепловая карта в центральной колонке, верхняя строка
        self.main_layout.addWidget(self.histogram_widget, 0, 2)  # Гистограмма в правой колонке
        self.main_layout.addWidget(self.x_profile_plot, 1, 1)  # X-профиль в центральной колонке, нижняя строка
        self.main_layout.addWidget(self.info_widget, 1, 0)    # Информация в левой колонке, нижняя строка
        
        # Настраиваем соотношения сетки, чтобы тепловая карта точно соответствовала профилям
        self.main_layout.setColumnStretch(0, 1)   # Y-профиль и инфо - узкие
        self.main_layout.setColumnStretch(1, 5)   # Тепловая карта и X-профиль - широкие
        self.main_layout.setColumnStretch(2, 1)   # Гистограмма - узкая
        
        self.main_layout.setRowStretch(0, 5)      # Тепловая карта и Y-профиль - высокие
        self.main_layout.setRowStretch(1, 1)      # X-профиль и информация - низкие
        
        # Связываем оси для синхронизации масштабирования
        self.x_profile_plot.getPlotItem().setXLink(self.image_view.getView())
        self.y_profile_plot.getPlotItem().setYLink(self.image_view.getView())
        
        # Инициализируем переменные данных
        self.data = None
        self.x_mm = None
        self.y_mm = None
        self.x_profile = None
        self.y_profile = None
        
    def update_data(self, data, x_mm, y_mm, centroid_x, centroid_y, rms_x, rms_y, x_profile, y_profile):
        """Обновление данных и графиков с точным выравниванием областей"""
        try:
            # Сохраняем данные
            self.data = data
            self.x_mm = x_mm
            self.y_mm = y_mm
            self.x_profile = x_profile
            self.y_profile = y_profile
            
            # Получаем границы осей в миллиметрах
            x_min, x_max = min(x_mm), max(x_mm)
            y_min, y_max = min(y_mm), max(y_mm)
            
            # Предварительная обработка данных изображения
            data_float = data.astype(np.float32)
            
            # Вычисляем минимум и максимум для нормализации
            min_val = np.min(data_float)
            max_val = np.max(data_float)
            
            # Проверка на отсутствие вариации в данных
            if max_val <= min_val:
                logging.warning("Данные не содержат вариаций (min=max). Невозможно нормализовать.")
                norm_data = data_float
            else:
                # Нормализуем данные для лучшего отображения
                norm_data = (data_float - min_val) / (max_val - min_val)
            
            # Создаем трансформацию для правильного отображения координат в мм
            height, width = data.shape
            
            # Устанавливаем изображение 
            self.image_view.setImage(
                norm_data, 
                autoLevels=False,
                levels=(0.0, 1.0)
            )
            
            # Отключаем автоматическое масштабирование тепловой карты
            self.image_view.view.setAspectLocked(False)
            self.image_view.view.invertY(True)
            
            # Точно устанавливаем диапазон координат - ОЧЕНЬ ВАЖНО!
            self.image_view.view.setRange(
                QRectF(x_min, y_min, x_max - x_min, y_max - y_min),
                padding=0
            )
            
            # Настраиваем цветовую карту
            try:
                # Пробуем установить высококонтрастную карту
                for cmap_name in ['hot', 'thermal', 'flame', 'inferno']:
                    try:
                        colormap = pg.colormap.get(cmap_name)
                        self.image_view.setColorMap(colormap)
                        logging.info(f"Установлена цветовая карта: {cmap_name}")
                        break
                    except Exception:
                        continue
                else:
                    # Создаем собственную карту, если встроенные недоступны
                    cm = pg.ColorMap(
                        pos=[0.0, 0.33, 0.66, 1.0],
                        color=[[0, 0, 0, 255], [128, 0, 0, 255], 
                               [255, 128, 0, 255], [255, 255, 255, 255]]
                    )
                    self.image_view.setColorMap(cm)
            except Exception as e:
                logging.warning(f"Не удалось установить цветовую карту: {e}")
            
            # Обновляем профиль по оси X
            self.x_profile_plot.clear()
            self.x_profile_plot.plot(
                x_mm, x_profile,
                pen=pg.mkPen('g', width=2)
            )
            # Важно: устанавливаем ТОЧНЫЙ диапазон для оси X без отступов
            self.x_profile_plot.setXRange(x_min, x_max, padding=0)
            
            # Обновляем профиль по оси Y (инвертированный - значения по X, координаты по Y)
            self.y_profile_plot.clear()
            self.y_profile_plot.plot(
                y_profile, y_mm,
                pen=pg.mkPen('g', width=2)
            )
            # Важно: устанавливаем ТОЧНЫЙ диапазон для оси Y без отступов
            self.y_profile_plot.setYRange(y_min, y_max, padding=0)
            
            # Обновляем информацию о пучке
            self.centroid_x_label.setText(f"Центроид X: {centroid_x:.3f} мм")
            self.centroid_y_label.setText(f"Центроид Y: {centroid_y:.3f} мм")
            self.rms_x_label.setText(f"RMS X: {rms_x:.3f} мм")
            self.rms_y_label.setText(f"RMS Y: {rms_y:.3f} мм")
            
            logging.info(f"Обновлены данные: {data.shape}, значения [{min_val:.2f}-{max_val:.2f}], X: [{x_min:.2f}-{x_max:.2f}] мм, Y: [{y_min:.2f}-{y_max:.2f}] мм")
            
        except Exception as e:
            logging.error(f"Ошибка при обновлении данных: {str(e)}")
            logging.exception(e)
        
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