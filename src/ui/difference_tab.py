# src.ui.difference_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QSplitter, QGridLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.ui.constants import TOP_PANEL_HEIGHT
from src.analysis.filters import apply_median_filter

class DifferenceTab(QWidget):
    """
    Tab for displaying and analyzing the difference between current image and background.
    
    This component provides an interactive interface for visualizing the result
    of subtracting the background image from the current frame. It includes displaying the beam
    profile as a heatmap with projections along X and Y axes, as well as calculating and
    displaying key beam parameters: centroid coordinates and RMS sizes.
    
    Main functionality:
    - Visualization of difference between current image and background via heatmap
    - Display of intensity projections along X and Y axes
    - Calculation and display of centroid and RMS beam sizes
    - Display of camera information (name, resolution, pixel size)
    - Automatic data updates at regular time intervals
    
    Visualization includes centered coordinates in physical units (mm),
    which allows quantitative analysis of beam spatial characteristics.
    
    Attributes:
        main_window: Reference to the main application window
        last_data_hash: Hash of last displayed data for optimization of redrawing
        update_timer: Timer for regular tab content updates
        plot_canvas: Canvas for displaying matplotlib plots
        camera_name_label: Label with camera name
        camera_resolution_label: Label with camera resolution
        camera_pixel_size_label: Label with camera pixel size
        centroid_x_label: Label with X centroid coordinate
        centroid_y_label: Label with Y centroid coordinate
        rms_x_label: Label with RMS size along X
        rms_y_label: Label with RMS size along Y
    """
    def __init__(self, main_window):
        super().__init__()
        
        self.main_window = main_window
        
        # For tracking data changes
        self.last_data_hash = None
        
        # Timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(500)  # Update every 500 ms
        
        # UI initialization
        self.init_ui()
        
    def init_ui(self):
        """
        Initializes the tab's user interface
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Top panel with camera information
        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)  # Increase spacing between elements
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)
        
        # Camera information panel
        camera_info_panel = QGroupBox("Camera Information")
        camera_info_panel.setMinimumWidth(150)  # Set minimum width
        camera_info_layout = QVBoxLayout(camera_info_panel)
        camera_info_layout.setSpacing(5)  # Reduce spacing between elements inside the panel
        
        # Information labels
        self.camera_name_label = QLabel("Name: -")
        self.camera_resolution_label = QLabel("Resolution: -")
        self.camera_pixel_size_label = QLabel("Pixel size: -")
        
        # Add to camera info panel layout
        camera_info_layout.addWidget(self.camera_name_label)
        camera_info_layout.addWidget(self.camera_resolution_label)
        camera_info_layout.addWidget(self.camera_pixel_size_label)
        
        # Panel with centroid and RMS information
        beam_info_panel = QGroupBox("Beam Information (difference)")
        beam_info_panel.setMinimumWidth(140)  # Set minimum width
        beam_info_layout = QVBoxLayout(beam_info_panel)
        beam_info_layout.setSpacing(5)  # Reduce spacing between elements inside the panel
        
        # Information labels
        self.centroid_x_label = QLabel("Centroid X: -")
        self.centroid_y_label = QLabel("Centroid Y: -")
        self.rms_x_label = QLabel("RMS X: -")
        self.rms_y_label = QLabel("RMS Y: -")
        
        # Reduce font size for labels
        font = QFont()
        font.setPointSize(9)
        self.centroid_x_label.setFont(font)
        self.centroid_y_label.setFont(font)
        self.rms_x_label.setFont(font)
        self.rms_y_label.setFont(font)
        
        # Add to information panel layout
        beam_info_layout.addWidget(self.centroid_x_label)
        beam_info_layout.addWidget(self.centroid_y_label)
        beam_info_layout.addWidget(self.rms_x_label)
        beam_info_layout.addWidget(self.rms_y_label)
        
        # Add panels to top panel
        top_panel.addWidget(camera_info_panel)
        
        # Create invisible empty panels to imitate other panels in camera_tab
        for i in range(5):  # 5 other panels in camera_tab
            empty_panel = QGroupBox()
            empty_panel.setMinimumWidth(120)
            empty_panel.setStyleSheet("border: none; background-color: transparent;")
            empty_layout = QVBoxLayout(empty_panel)
            top_panel.addWidget(empty_panel)
            
        # Add beam information panel at the end
        top_panel.addWidget(beam_info_panel)
        
        # Central container for plots - one widget instead of three
        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(400, 400)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_canvas = None
        
        # Add all to main layout
        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.plot_widget, 1)
        
        self.setLayout(main_layout)
        
    def update_camera_info(self):
        """
        Updates camera information
        """
        camera_info = self.main_window.get_camera_info()
        
        if camera_info:
            self.camera_name_label.setText("Name: {}".format(camera_info.get('camera_name', '-')))
            resolution = camera_info.get("resolution", (0, 0))
            self.camera_resolution_label.setText("Resolution: {} x {}".format(resolution[0], resolution[1]))
            
            pixel_size_x = camera_info.get("pixel_size_x", 0)
            pixel_size_y = camera_info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText("Pixel size: {:.8f} x {:.8f} mm".format(pixel_size_x, pixel_size_y))
        else:
            self.camera_name_label.setText("Name: -")
            self.camera_resolution_label.setText("Resolution: -")
            self.camera_pixel_size_label.setText("Pixel size: -")
            
    def update_plots(self):
        """
        Updates all plots
        """
        # Get current data
        data = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()
        
        # If there's no difference data or camera information
        if data["difference"] is None or camera_info is None:
            return
        
        # Check if data changed
        if data["difference"] is not None:
            current_hash = hash(str(data["difference"].data.tobytes()))
            if self.last_data_hash == current_hash and self.plot_canvas is not None:
                # Data didn't change, exit
                return
            self.last_data_hash = current_hash
            
        # Get camera information
        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        
        # Create new figure with three areas
        fig = Figure(figsize=(10, 8))
        
        # Change GridSpec to add space for colorbar
        gs = GridSpec(2, 3, width_ratios=[0.5, 5, 0.3], height_ratios=[5, 0.8], figure=fig)
        
        # Area for Y projection (left)
        ax_y_proj = fig.add_subplot(gs[0, 0])
        # Area for heatmap (right above)
        ax_heatmap = fig.add_subplot(gs[0, 1])
        # Area for X projection (right below)
        ax_x_proj = fig.add_subplot(gs[1, 1])
        # Area for colorbar
        cax = fig.add_subplot(gs[0, 2])
        
        # Get image data and apply median filtering for visualization
        img_data = data["difference"].data
        img_data = apply_median_filter(img_data, kernel_size=3)
        
        # Calculate coordinates in millimeters (centered relative to zero)
        height, width = img_data.shape
        x_mm = (np.arange(width) - width / 2) * pixel_size_x
        y_mm = (np.arange(height) - height / 2) * pixel_size_y
        
        # Heatmap
        im = ax_heatmap.imshow(
            img_data, 
            extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
            origin='lower', 
            aspect='auto',
            cmap='jet'
        )
        ax_heatmap.set_xlabel('X (mm)')
        ax_heatmap.set_ylabel('Y (mm)')
        ax_heatmap.set_title('Beam Profile')
        
        # Add colorbar
        fig.colorbar(im, cax=cax, label='Intensity')
        
        # --- Fix: use centered coordinates for projections ---
        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)

        # Gaussian approximation - disabled
        # _, gauss_x = self.main_window.image_analyzer.fit_gaussian(x_mm, x_proj)
        # _, gauss_y = self.main_window.image_analyzer.fit_gaussian(y_mm, y_proj)

        # Draw X projection
        ax_x_proj.plot(x_mm, x_proj, 'b-', label='Data')
        # if gauss_x is not None:
        #     ax_x_proj.plot(x_mm, gauss_x, 'r--', label='Gauss')
        ax_x_proj.set_xlabel('X (mm)')
        ax_x_proj.set_ylabel('Intensity')
        ax_x_proj.grid(True, linestyle='--', alpha=0.7)
        
        # Draw Y projection
        ax_y_proj.plot(y_proj, y_mm, 'b-', label='Data')
        # if gauss_y is not None:
        #     ax_y_proj.plot(gauss_y, y_mm, 'r--', label='Gauss')
        ax_y_proj.set_ylabel('Y (mm)')
        ax_y_proj.set_xlabel('Intensity')
        ax_y_proj.grid(True, linestyle='--', alpha=0.7)
        
        # Equalize axes
        ax_y_proj.set_ylim(ax_heatmap.get_ylim())
        ax_x_proj.set_xlim(ax_heatmap.get_xlim())
        
        # Remove empty space
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
        
        # Clear current canvas
        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            # Proper matplotlib resource release
            plt_figure = self.plot_canvas.figure
            plt_figure.clear()
            self.plot_canvas.close()
        
        # Create new canvas
        self.plot_canvas = FigureCanvasQTAgg(fig)
        
        # Add canvas to layout
        self.plot_layout.addWidget(self.plot_canvas)
        
        # Calculate or use pre-calculated centroid and RMS of difference
        if self.main_window.current_mode == "file" and "difference_centroid" in self.main_window.file_data:
            # Use pre-calculated values
            centroid_x, centroid_y = self.main_window.file_data["difference_centroid"]
            rms_x, rms_y = self.main_window.file_data["difference_rms"]
        else:
            # Calculate on the spot
            centroid_x, centroid_y = self.main_window.image_analyzer.calculate_centroid(
                data["difference"], pixel_size_x, pixel_size_y
            )
            
            rms_x, rms_y = self.main_window.image_analyzer.calculate_rms(
                data["difference"], pixel_size_x, pixel_size_y
            )
        
        # Update centroid and RMS information
        self.centroid_x_label.setText("Centroid X: {:.6f} mm".format(centroid_x))
        self.centroid_y_label.setText("Centroid Y: {:.6f} mm".format(centroid_y))
        self.rms_x_label.setText("RMS X: {:.6f} mm".format(rms_x))
        self.rms_y_label.setText("RMS Y: {:.6f} mm".format(rms_y))
        
    def update_tab(self):
        """
        Updates tab content
        """
        self.update_camera_info()
        self.update_plots()
