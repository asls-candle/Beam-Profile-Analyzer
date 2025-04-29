import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.cm as cm

class PlotManager:
    """
    Класс для построения графиков профиля пучка
    """
    def __init__(self):
        # Настройка цветовой карты для тепловой карты
        self.colormap = cm.get_cmap('jet')  # Используем jet для лучшей визуализации
        
    def create_heatmap_figure(self, data, pixel_size_x, pixel_size_y):
        """
        Создает фигуру с тепловой картой
        
        Args:
            data: Двумерный массив значений светимости
            pixel_size_x: Размер пикселя по оси X (мм)
            pixel_size_y: Размер пикселя по оси Y (мм)
            
        Returns:
            Figure: Объект фигуры Matplotlib
        """
        # Если данных нет, возвращаем пустую фигуру
        if data is None or data.size == 0:
            fig = Figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.set_title("Нет данных")
            return fig
            
        # Получаем размеры изображения
        height, width = data.shape
        
        # Создаем массивы координат в миллиметрах (центрированные относительно нуля)
        x_mm = (np.arange(width) - width / 2) * pixel_size_x
        y_mm = (np.arange(height) - height / 2) * pixel_size_y
        
        # Создаем фигуру с жестко заданными полями
        fig = Figure(figsize=(5, 5))
        fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)
        ax = fig.add_subplot(111)
        
        # Уменьшаем разрешение тепловой карты для производительности
        if data.shape[0] > 300 or data.shape[1] > 300:
            step_y = max(1, data.shape[0] // 300)
            step_x = max(1, data.shape[1] // 300)
            data_downsampled = data[::step_y, ::step_x]
            x_downsampled = x_mm[::step_x]
            y_downsampled = y_mm[::step_y]
            im = ax.imshow(data_downsampled, cmap=self.colormap, origin='lower',
                           extent=[x_downsampled[0], x_downsampled[-1], y_downsampled[0], y_downsampled[-1]],
                           aspect='auto', interpolation='bilinear')
        else:
            im = ax.imshow(data, cmap=self.colormap, origin='lower',
                           extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
                           aspect='auto', interpolation='bilinear')
        
        # Добавляем заголовок и подписи осей
        ax.set_title("Профиль пучка", fontsize=10)
        ax.set_xlabel("X (мм)", fontsize=9)
        ax.set_ylabel("Y (мм)", fontsize=9)
        
        # Добавляем шкалу цветов c точными настройками положения
        fig.colorbar(
            im,
            ax=ax,
            label="Интенсивность", 
            fraction=0.046,
            pad=0.04,
            orientation='vertical'
        )
                
        return fig
    
    def create_projection_figure(self, x_coords, x_proj, y_coords, y_proj, gauss_x=None, gauss_y=None):
        """
        Создает фигуры с графиками проекций на оси X и Y
        
        Args:
            x_coords: Координаты по оси X (мм)
            x_proj: Проекция на ось X
            y_coords: Координаты по оси Y (мм)
            y_proj: Проекция на ось Y
            gauss_x: Гауссово приближение для X-проекции
            gauss_y: Гауссово приближение для Y-проекции
            
        Returns:
            tuple: (Figure_X, Figure_Y) - Объекты фигур с проекциями
        """
        # Проверяем наличие данных
        if (x_coords is None or x_proj is None or 
            y_coords is None or y_proj is None or 
            len(x_coords) == 0 or len(x_proj) == 0 or
            len(y_coords) == 0 or len(y_proj) == 0):
            
            # Создаем пустые фигуры
            fig_x = Figure(figsize=(5, 1.5))
            ax_x = fig_x.add_subplot(111)
            ax_x.set_title("Нет данных")
            
            fig_y = Figure(figsize=(1.5, 5))
            ax_y = fig_y.add_subplot(111)
            ax_y.set_title("Нет данных")
            
            return fig_x, fig_y
        
        # Создаем фигуру для X-проекции с явно заданными полями
        fig_x = Figure(figsize=(5, 1.5))
        fig_x.subplots_adjust(left=0.15, right=0.85, bottom=0.25, top=0.9)
        ax_x = fig_x.add_subplot(111)
        
        # Строим проекцию на ось X
        ax_x.plot(x_coords, x_proj, 'b-', linewidth=2, label='Данные')
        
        # Если есть Гауссово приближение, добавляем его
        if gauss_x is not None:
            ax_x.plot(x_coords, gauss_x, 'r--', linewidth=1, label='Гаусс')
            ax_x.legend(loc='upper right', fontsize='x-small')
            
        ax_x.set_xlabel("X (мм)", fontsize=9)
        ax_x.set_ylabel("Интенсивность", fontsize=9)
        ax_x.grid(True, linestyle='--', alpha=0.5)
        ax_x.tick_params(labelsize=8)
        
        # Настраиваем оси
        ax_x.set_xlim(x_coords[0], x_coords[-1])
        ax_x.set_ylim(0, 1.05)
        
        # Создаем фигуру для Y-проекции с явно заданными полями
        fig_y = Figure(figsize=(1.5, 5))
        fig_y.subplots_adjust(left=0.25, right=0.85, bottom=0.15, top=0.9)
        ax_y = fig_y.add_subplot(111)
        
        # Строим проекцию на ось Y (инвертируем оси!)
        ax_y.plot(y_proj, y_coords, 'b-', linewidth=2, label='Данные')
        
        # Если есть Гауссово приближение, добавляем его
        if gauss_y is not None:
            ax_y.plot(gauss_y, y_coords, 'r--', linewidth=1, label='Гаусс')
            ax_y.legend(loc='upper right', fontsize='x-small')
            
        ax_y.set_ylabel("Y (мм)", fontsize=9)
        ax_y.set_xlabel("Интенсивность", fontsize=9)
        ax_y.grid(True, linestyle='--', alpha=0.5)
        ax_y.tick_params(labelsize=8)
        
        # Настраиваем оси
        ax_y.set_ylim(y_coords[0], y_coords[-1])
        ax_y.set_xlim(0, 1.05)
        
        return fig_x, fig_y
    
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
