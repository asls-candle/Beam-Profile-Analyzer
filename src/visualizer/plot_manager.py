import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class PlotManager:
    """
    Класс для построения графиков профиля пучка
    """
    def __init__(self):
        """
        Инициализирует менеджер графиков.
        
        Создает фигуры и оси для тепловой карты и проекций.
        """
        # Настройка цветовой карты для тепловой карты
        self.colormap = cm.get_cmap('jet')  # Используем jet для лучшей визуализации
        
        # Создаем и сохраняем фигуры для переиспользования
        self.heatmap_fig = plt.figure(figsize=(8, 8))
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        
        # Создание фигуры для проекции по X
        self.projection_x_fig = plt.figure(figsize=(8, 3))
        self.projection_x_ax = self.projection_x_fig.add_subplot(111)
        
        # Создание фигуры для проекции по Y
        self.projection_y_fig = plt.figure(figsize=(3, 8))
        self.projection_y_ax = self.projection_y_fig.add_subplot(111)
        
        # Устанавливаем отступы для фигур
        self.heatmap_fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        self.projection_x_fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.25)
        self.projection_y_fig.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
        
        # Инициализируем объекты для тепловой карты и графиков
        self.heatmap_im = None
        self.x_proj_line = None
        self.y_proj_line = None
        self.x_gauss_line = None
        self.y_gauss_line = None
        
        # Инициализация colorbar
        self.colorbar = None
        
        # Сохраняем текущие пределы для синхронизации
        self.current_xlim = None
        self.current_ylim = None
        
    def create_heatmap_figure(self, data, pixel_size_x, pixel_size_y):
        """
        Обновляет тепловую карту
        
        Args:
            data: Двумерный массив значений светимости
            pixel_size_x: Размер пикселя по оси X (мм)
            pixel_size_y: Размер пикселя по оси Y (мм)
            
        Returns:
            Figure: Объект фигуры Matplotlib
        """
        # Очищаем текущие графики
        self.heatmap_ax.clear()
        
        # Если данных нет, показываем пустую фигуру
        if data is None or data.size == 0:
            self.heatmap_ax.set_title("Нет данных")
            return self.heatmap_fig
            
        # Получаем размеры изображения
        height, width = data.shape
        
        # Вычисляем координаты для центрированного изображения
        x_min = -(width - 1) / 2 * pixel_size_x
        x_max = (width - 1) / 2 * pixel_size_x
        y_min = -(height - 1) / 2 * pixel_size_y
        y_max = (height - 1) / 2 * pixel_size_y
        
        # Сохраняем пределы для синхронизации с проекциями
        self.current_xlim = (x_min, x_max)
        self.current_ylim = (y_min, y_max)
        
        # Строим тепловую карту
        if self.heatmap_im is None:
            self.heatmap_im = self.heatmap_ax.imshow(
                data, 
                cmap=self.colormap,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                aspect='equal'  # Сохраняем пропорции
            )
            self.colorbar = self.heatmap_fig.colorbar(self.heatmap_im, ax=self.heatmap_ax, label="Интенсивность")
        else:
            self.heatmap_im.set_data(data)
            self.heatmap_im.set_extent([x_min, x_max, y_min, y_max])
        
        # Обновляем заголовок и подписи осей
        self.heatmap_ax.set_title("Профиль пучка")
        self.heatmap_ax.set_xlabel("X (мм)")
        self.heatmap_ax.set_ylabel("Y (мм)")
        self.heatmap_ax.grid(True, linestyle='--', alpha=0.7)
        
        return self.heatmap_fig
    
    def create_projection_figure(self, x_coords, x_proj, y_coords, y_proj, gauss_x=None, gauss_y=None):
        """
        Создает фигуры для проекций на оси X и Y, масштабированные под основное изображение.

        Args:
            x_coords: Координаты по оси X (мм)
            x_proj: Проекция на ось X
            y_coords: Координаты по оси Y (мм)
            y_proj: Проекция на ось Y
            gauss_x: Гауссово приближение для X проекции
            gauss_y: Гауссово приближение для Y проекции
            
        Returns:
            fig_x: Фигура с проекцией на ось X
            fig_y: Фигура с проекцией на ось Y
        """
        # Очищаем текущие графики
        self.projection_x_ax.clear()
        self.projection_y_ax.clear()
        
        if self.current_xlim is None or self.current_ylim is None:
            return self.projection_x_fig, self.projection_y_fig
            
        # Отображаем проекцию на X
        if x_coords is not None and x_proj is not None:
            self.projection_x_ax.plot(x_coords, x_proj, 'b-', label='Data')
            if gauss_x is not None:
                self.projection_x_ax.plot(x_coords, gauss_x, 'r--', label='Gaussian fit')
            
            # Устанавливаем пределы по X такие же, как у основного изображения
            self.projection_x_ax.set_xlim(self.current_xlim)
            
            # Добавляем подписи и сетку
            self.projection_x_ax.set_xlabel('X (мм)')
            self.projection_x_ax.set_ylabel('Интенсивность (отн. ед.)')
            self.projection_x_ax.grid(True, linestyle='--', alpha=0.7)
            self.projection_x_ax.legend()
        
        # Отображаем проекцию на Y
        if y_coords is not None and y_proj is not None:
            # Для Y-проекции меняем местами координаты и значения
            self.projection_y_ax.plot(y_proj, y_coords, 'b-', label='Data')
            if gauss_y is not None:
                self.projection_y_ax.plot(gauss_y, y_coords, 'r--', label='Gaussian fit')
            
            # Устанавливаем пределы по Y такие же, как у основного изображения
            self.projection_y_ax.set_ylim(self.current_ylim)
            
            # Добавляем подписи и сетку
            self.projection_y_ax.set_ylabel('Y (мм)')
            self.projection_y_ax.set_xlabel('Интенсивность (отн. ед.)')
            self.projection_y_ax.grid(True, linestyle='--', alpha=0.7)
            self.projection_y_ax.legend()
            
            # Инвертируем ось Y для соответствия изображению
            self.projection_y_ax.invert_yaxis()
        
        return self.projection_x_fig, self.projection_y_fig
    
    @staticmethod
    def create_canvas_from_figure(figure):
        """
        Создает виджет канваса Qt из фигуры Matplotlib
        
        Args:
            figure: Объект фигуры Matplotlib
            
        Returns:
            FigureCanvasQTAgg: Виджет канваса Qt
        """
        return FigureCanvasQTAgg(figure)
        
    @staticmethod
    def save_figure(figure, filepath, dpi=300):
        """
        Сохраняет фигуру в файл
        
        Args:
            figure: Объект фигуры Matplotlib
            filepath: Путь для сохранения
            dpi: Разрешение изображения
            
        Returns:
            bool: True если сохранение успешно, иначе False
        """
        try:
            # Устанавливаем DPI через свойство фигуры
            figure.set_dpi(dpi)
            figure.savefig(filepath, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Ошибка при сохранении фигуры: {e}")
            return False
