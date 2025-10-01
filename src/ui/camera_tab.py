from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QComboBox, QRadioButton, QButtonGroup,
                           QSpinBox, QGroupBox, QSplitter, QFrame, QMessageBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT

class CameraTab(QWidget):
    """
    Tab for main capture and camera control
    """
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        # Status flags
        self.is_capturing = False

        # For tracking data changes
        self.last_data_hash = None

        # Timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(500)  # Update every 500 ms

        # UI initialization
        self.init_ui()

        # Connect signals
        self.connect_signals()

    def init_ui(self):
        """
        Initializes the tab's user interface
        """
        # Main layout
        main_layout = QVBoxLayout(self)

        # Top panel (operation mode, camera control, background, etc.)
        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)  # Increase spacing between elements
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)

        # Mode selection panel
        mode_panel = QGroupBox("Operation Mode")
        mode_panel.setMinimumWidth(120)  # Reduce minimum width
        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setSpacing(5)  # Reduce spacing between elements inside the panel

        # Mode selection radio buttons
        self.camera_radio = QRadioButton("Camera")
        self.file_radio = QRadioButton("File Reading")

        # Radio button group
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.camera_radio, 0)
        self.mode_group.addButton(self.file_radio, 1)

        # Open file button
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.open_file_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.open_file_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        # Add to mode panel layout
        mode_layout.addWidget(self.camera_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addWidget(self.open_file_btn)

        # Camera control panel
        camera_panel = QGroupBox("Camera Control")
        camera_panel.setMinimumWidth(120)  # Reduce minimum width
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setSpacing(5)  # Reduce spacing between elements inside the panel

        # Camera dropdown
        self.camera_label = QLabel("Camera:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Not selected")
        self.camera_combo.addItems(self.main_window.camera_manager.get_camera_list())

        # Camera control buttons
        self.launch_camera_btn = QPushButton("Launch the camera")
        self.launch_camera_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.launch_camera_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.launch_camera_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.stop_camera_btn = QPushButton("Stop camera")
        self.stop_camera_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.stop_camera_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.stop_camera_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        # Add to camera panel layout
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.launch_camera_btn)
        camera_layout.addWidget(self.stop_camera_btn)

        # Camera information panel
        camera_info_panel = QGroupBox("Camera Information")
        camera_info_panel.setMinimumWidth(120)  # Reduce minimum width
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

        # Background collection panel
        background_panel = QGroupBox("Background Collection")
        background_panel.setMinimumWidth(120)  # Reduce minimum width
        background_layout = QVBoxLayout(background_panel)
        background_layout.setSpacing(5)  # Reduce spacing between elements inside the panel

        # Number of frames for background
        bg_frames_layout = QHBoxLayout()
        bg_frames_layout.addWidget(QLabel("Number of frames:"))
        self.bg_frames_spinbox = QSpinBox()
        self.bg_frames_spinbox.setRange(1, 100)
        self.bg_frames_spinbox.setValue(40)
        bg_frames_layout.addWidget(self.bg_frames_spinbox)

        # Background collection control buttons
        self.get_background_btn = QPushButton("Get background")
        self.get_background_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.get_background_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.get_background_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.stop_bg_collection_btn = QPushButton("Stop")
        self.stop_bg_collection_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.stop_bg_collection_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.stop_bg_collection_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        # Add to background panel layout
        background_layout.addLayout(bg_frames_layout)
        background_layout.addWidget(self.get_background_btn)
        background_layout.addWidget(self.stop_bg_collection_btn)

        # Capture control panel
        capture_panel = QGroupBox("Capture Control")
        capture_panel.setMinimumWidth(120)  # Reduce minimum width
        capture_layout = QVBoxLayout(capture_panel)
        capture_layout.setSpacing(5)  # Reduce spacing between elements inside the panel

        # Capture control buttons
        self.start_capture_btn = QPushButton("Start")
        self.start_capture_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.start_capture_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.start_capture_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.stop_capture_btn = QPushButton("Stop")
        self.stop_capture_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.stop_capture_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.stop_capture_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.export_data_btn.setMinimumHeight(BUTTON_MIN_HEIGHT)
        self.export_data_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        # Add to capture panel layout
        capture_layout.addWidget(self.start_capture_btn)
        capture_layout.addWidget(self.stop_capture_btn)
        capture_layout.addWidget(self.export_data_btn)

        # Status panel
        status_panel = QGroupBox("Status")
        status_panel.setMinimumWidth(120)  # Reduce minimum width
        status_layout = QVBoxLayout(status_panel)
        status_layout.setSpacing(5)  # Reduce spacing between elements inside the panel

        # Status labels
        self.mode_status_label = QLabel("Mode: Camera")
        self.camera_status_label = QLabel("Camera: Not connected")
        self.capture_status_label = QLabel("Data collection: Stopped")

        # Add to status panel layout
        status_layout.addWidget(self.mode_status_label)
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.capture_status_label)

        # Beam information panel
        beam_info_panel = QGroupBox("Beam Information")
        beam_info_panel.setMinimumWidth(140)  # Increase minimum width
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

        # Add to beam info panel layout
        beam_info_layout.addWidget(self.centroid_x_label)
        beam_info_layout.addWidget(self.centroid_y_label)
        beam_info_layout.addWidget(self.rms_x_label)
        beam_info_layout.addWidget(self.rms_y_label)

        # Add all panels to top panel with equal stretch
        top_panel.addWidget(mode_panel, 1)
        top_panel.addWidget(camera_panel, 1)
        top_panel.addWidget(camera_info_panel, 1)
        top_panel.addWidget(background_panel, 1)
        top_panel.addWidget(capture_panel, 1)
        top_panel.addWidget(status_panel, 1)
        top_panel.addWidget(beam_info_panel, 1)

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

        # Initialize widget states
        self.update_ui_state()

    def connect_signals(self):
        """
        Connects signals to handlers
        """
        # Operation mode
        self.camera_radio.toggled.connect(self.on_mode_changed)
        self.file_radio.toggled.connect(self.on_mode_changed)
        self.open_file_btn.clicked.connect(self.on_open_file)

        # Camera control
        self.launch_camera_btn.clicked.connect(self.on_launch_camera)
        self.stop_camera_btn.clicked.connect(self.on_stop_camera)

        # Background collection
        self.get_background_btn.clicked.connect(self.on_get_background)
        self.stop_bg_collection_btn.clicked.connect(self.on_stop_bg_collection)

        # Capture control
        self.start_capture_btn.clicked.connect(self.on_start_capture)
        self.stop_capture_btn.clicked.connect(self.on_stop_capture)
        self.export_data_btn.clicked.connect(self.on_export_data)

        # Set initial mode
        self.camera_radio.setChecked(True)

    def update_ui_state(self):
        """
        Updates the interface state depending on the current mode
        """
        # Get current mode
        is_camera_mode = self.main_window.current_mode == "camera"

        # Mode control
        self.camera_radio.setChecked(is_camera_mode)
        self.file_radio.setChecked(not is_camera_mode)

        # Update mode label
        self.mode_status_label.setText("Mode: {}".format('Camera' if is_camera_mode else 'File Reading'))

        # Open file button - active only in file reading mode
        self.open_file_btn.setEnabled(not is_camera_mode)

        # In camera mode
        if is_camera_mode:
            # Camera control availability
            camera_connected = self.main_window.camera_manager.is_connected

            self.camera_combo.setEnabled(not camera_connected)
            self.launch_camera_btn.setEnabled(not camera_connected)
            self.stop_camera_btn.setEnabled(camera_connected)

            # Background control availability
            # Explicitly check for background presence in image_reader
            has_background = self.main_window.image_reader.background is not None
            is_collecting_bg = self.main_window.image_reader.is_collecting_background

            # Background state logging for debugging
            print("CameraTab: has_background={}, is_collecting_bg={}".format(has_background, is_collecting_bg))

            self.bg_frames_spinbox.setEnabled(camera_connected and not is_collecting_bg)
            self.get_background_btn.setEnabled(camera_connected and not is_collecting_bg)
            self.stop_bg_collection_btn.setEnabled(camera_connected and is_collecting_bg)

            # Capture control availability
            can_start_capture = camera_connected and has_background and not self.is_capturing
            self.start_capture_btn.setEnabled(can_start_capture)
            self.stop_capture_btn.setEnabled(camera_connected and self.is_capturing)
            self.export_data_btn.setEnabled(camera_connected and not self.is_capturing)

            # Start button logging
            print("CameraTab: start_capture_btn.isEnabled={}".format(can_start_capture))

            # Camera status update
            if camera_connected:
                camera_info = self.main_window.camera_manager.get_camera_info()
                if camera_info:
                    self.camera_status_label.setText("Camera: {}".format(camera_info.get('name', 'Connected')))
                else:
                    self.camera_status_label.setText("Camera: Connected")
            else:
                self.camera_status_label.setText("Camera: Not connected")

            # Capture status update
            if self.is_capturing:
                self.capture_status_label.setText("Data collection: Collecting")
            elif is_collecting_bg:
                self.capture_status_label.setText("Data collection: Background collection")
            else:
                self.capture_status_label.setText("Data collection: Stopped")
        else:
            # In file reading mode
            # Disable all camera control elements
            self.camera_combo.setEnabled(False)
            self.launch_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(False)

            # Disable background control
            self.bg_frames_spinbox.setEnabled(False)
            self.get_background_btn.setEnabled(False)
            self.stop_bg_collection_btn.setEnabled(False)

            # Disable capture control
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(False)

            # Enable export if there are data
            self.export_data_btn.setEnabled(self.main_window.file_data["current_frame"] is not None)

            # Update status
            self.camera_status_label.setText("Camera: -")
            self.capture_status_label.setText("Data collection: -")

        # Update camera information
        self.update_camera_info()

    def update_camera_info(self):
        """
        Updates camera information
        """
        # Get current mode
        is_camera_mode = self.main_window.current_mode == "camera"

        # In camera mode, check if camera is connected
        if is_camera_mode and not self.main_window.camera_manager.is_connected:
            self.camera_name_label.setText("Name: -")
            self.camera_resolution_label.setText("Resolution: -")
            self.camera_pixel_size_label.setText("Pixel size: -")
            return

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
        # In camera mode, check if camera is connected
        if self.main_window.current_mode == "camera" and not self.main_window.camera_manager.is_connected:
            return

        # Get current data
        data = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()

        if data["current_frame"] is None or camera_info is None:
            return

        # Check if data changed
        if data["current_frame"] is not None:
            current_hash = hash(str(data["current_frame"].data.tobytes()))
            if self.last_data_hash == current_hash and self.plot_canvas is not None:
                # Data didn't change, exit
                return
            self.last_data_hash = current_hash

        # Get camera information
        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)

        # Create new figure with three areas
        fig = Figure(figsize=(10, 8))

        # Change GridSpec for adding colorbar space
        gs = GridSpec(2, 3, width_ratios=[0.5, 5, 0.3], height_ratios=[5, 0.8], figure=fig)

        # Area for Y projection (left)
        ax_y_proj = fig.add_subplot(gs[0, 0])
        # Area for heatmap (right above)
        ax_heatmap = fig.add_subplot(gs[0, 1])
        # Area for X projection (right below)
        ax_x_proj = fig.add_subplot(gs[1, 1])
        # Area for colorbar
        cax = fig.add_subplot(gs[0, 2])

        # Get image data
        img_data = data["current_frame"].data

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

        # Projections
        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)

        # Gaussian approximation - disabled
        # _, gauss_x = self.main_window.image_analyzer.fit_gaussian(x_mm, x_proj)
        # _, gauss_y = self.main_window.image_analyzer.fit_gaussian(y_mm, y_proj)

        # Draw X projection
        ax_x_proj.plot(x_mm, x_proj, 'b-', linewidth=1, label='Data')
        # if gauss_x is not None:
        #     ax_x_proj.plot(x_mm, gauss_x, 'r--', label='Gauss')
        ax_x_proj.set_xlabel('X (mm)')
        ax_x_proj.set_ylabel('Intensity')
        ax_x_proj.grid(True, linestyle='--', alpha=0.7)

        # Draw Y projection
        ax_y_proj.plot(y_proj, y_mm, 'b-', linewidth=1, label='Data')
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
            # Correct matplotlib resource release
            plt_figure = self.plot_canvas.figure
            plt_figure.clear()
            self.plot_canvas.close()

        # Create new canvas
        self.plot_canvas = FigureCanvasQTAgg(fig)

        # Add canvas to layout
        self.plot_layout.addWidget(self.plot_canvas)

        # Update beam information and RMS
        centroid_x, centroid_y = data["centroid"]
        rms_x, rms_y = data["rms"]

        self.centroid_x_label.setText("Centroid X: {:.6f} mm".format(centroid_x))
        self.centroid_y_label.setText("Centroid Y: {:.6f} mm".format(centroid_y))
        self.rms_x_label.setText("RMS X: {:.6f} mm".format(rms_x))
        self.rms_y_label.setText("RMS Y: {:.6f} mm".format(rms_y))

    def update_tab(self):
        """
        Updates tab content
        """
        self.update_ui_state()
        self.update_plots()

    def on_mode_changed(self, checked):
        """
        Handler for mode change

        Args:
            checked: Radio button press flag
        """
        if checked:
            if self.camera_radio.isChecked():
                self.main_window.switch_mode("camera")
            else:
                self.main_window.switch_mode("file")

    def on_open_file(self):
        """
        Handler for open file button press
        """
        if self.main_window.open_file_dialog():
            self.file_radio.setChecked(True)

    def on_launch_camera(self):
        """
        Handler for camera launch button press
        """
        camera_name = self.camera_combo.currentText()

        if camera_name == "Not selected":
            QMessageBox.warning(self, "Warning", "Select camera")
            return

        if not self.main_window.connect_to_camera(camera_name):
            QMessageBox.critical(self, "Error", "Failed to connect to camera {}".format(camera_name))

    def on_stop_camera(self):
        """
        Handler for camera stop button press
        """
        if not self.main_window.disconnect_camera():
            QMessageBox.critical(self, "Error", "Failed to disconnect from camera")

    def on_get_background(self):
        """
        Handler for background get button press
        """
        # Get number of frames
        frames_count = self.bg_frames_spinbox.value()

        # Show warning
        reply = QMessageBox.question(
            self,
            "Background Collection",
            "Ensure that you closed the laser shutter. Start?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if not self.main_window.start_background_collection(frames_count):
                QMessageBox.critical(self, "Error", "Failed to start background collection")

    def on_stop_bg_collection(self):
        """
        Handler for background stop button press
        """
        if not self.main_window.stop_background_collection():
            QMessageBox.critical(self, "Error", "Failed to stop background collection")

    def on_start_capture(self):
        """
        Handler for capture start button press
        """
        # Check if background is collected
        has_background = self.main_window.image_reader.background is not None
        print("CameraTab.on_start_capture: has_background = {}".format(has_background))

        if not has_background:
            QMessageBox.warning(
                self,
                "Warning",
                "Background must be collected before starting capture"
            )
            return

        self.is_capturing = True
        self.update_ui_state()

    def on_stop_capture(self):
        """
        Handler for capture stop button press
        """
        self.is_capturing = False
        self.update_ui_state()

    def on_export_data(self):
        """
        Handler for export data button press
        """
        export_path = self.main_window.save_file_dialog()
        if export_path:
            # Показываем сообщение о том, где сохранены данные
            self.main_window.statusBar().showMessage("Data exported to: {}".format(export_path), 5000)
