# src/ui/camera_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QRadioButton, QButtonGroup,
                             QSpinBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT


class CameraTab(QWidget):
    """
    Tab for main capture and camera control.
    Camera names and parameters are discovered dynamically — no defaults.
    """

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.last_data_hash = None

        # Persistent plot artists — created once, updated in-place afterwards
        self.plot_canvas = None
        self._im         = None   # AxesImage from imshow
        self._line_x     = None   # Line2D for X projection
        self._line_y     = None   # Line2D for Y projection
        self._ax_heatmap = None
        self._ax_x_proj  = None
        self._ax_y_proj  = None
        # Track image shape so we can recreate when resolution changes
        self._last_img_shape = None

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(500)

        self.init_ui()
        self.connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # ── Top panel ────────────────────────────────────────────────────────
        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)

        # Mode selection
        mode_panel = QGroupBox("Operation Mode")
        mode_panel.setMinimumWidth(120)
        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setSpacing(5)

        self.camera_radio = QRadioButton("Camera")
        self.file_radio = QRadioButton("File Reading")

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.camera_radio, 0)
        self.mode_group.addButton(self.file_radio, 1)

        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.open_file_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        mode_layout.addWidget(self.camera_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addWidget(self.open_file_btn)

        # Camera control
        camera_panel = QGroupBox("Camera Control")
        camera_panel.setMinimumWidth(120)
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setSpacing(5)

        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Not selected")
        self._populate_camera_combo(
            self.main_window.camera_manager.get_camera_list())

        self.refresh_cameras_btn = QPushButton("Refresh")
        self.refresh_cameras_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.refresh_cameras_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.launch_camera_btn = QPushButton("Launch camera")
        self.launch_camera_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.launch_camera_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.stop_camera_btn = QPushButton("Stop camera")
        self.stop_camera_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.stop_camera_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.refresh_cameras_btn)
        camera_layout.addWidget(self.launch_camera_btn)
        camera_layout.addWidget(self.stop_camera_btn)

        # Camera information + pixel size input
        camera_info_panel = QGroupBox("Camera Information")
        camera_info_panel.setMinimumWidth(120)
        camera_info_layout = QVBoxLayout(camera_info_panel)
        camera_info_layout.setSpacing(4)

        self.camera_name_label = QLabel("Name: -")
        self.camera_id_label = QLabel("ID: -")
        self.camera_resolution_label = QLabel("Resolution: -")
        self.camera_pixel_size_label = QLabel("Pixel size: -")

        camera_info_layout.addWidget(self.camera_name_label)
        camera_info_layout.addWidget(self.camera_id_label)
        camera_info_layout.addWidget(self.camera_resolution_label)
        camera_info_layout.addWidget(self.camera_pixel_size_label)

        # Background collection
        background_panel = QGroupBox("Background Collection")
        background_panel.setMinimumWidth(120)
        background_layout = QVBoxLayout(background_panel)
        background_layout.setSpacing(5)

        bg_frames_row = QHBoxLayout()
        bg_frames_row.addWidget(QLabel("Frames:"))
        self.bg_frames_spinbox = QSpinBox()
        self.bg_frames_spinbox.setRange(1, 100)
        self.bg_frames_spinbox.setValue(40)
        bg_frames_row.addWidget(self.bg_frames_spinbox)

        self.get_background_btn = QPushButton("Get background")
        self.get_background_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.get_background_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.stop_bg_collection_btn = QPushButton("Stop")
        self.stop_bg_collection_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.stop_bg_collection_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        background_layout.addLayout(bg_frames_row)
        background_layout.addWidget(self.get_background_btn)
        background_layout.addWidget(self.stop_bg_collection_btn)

        # Beam information
        beam_info_panel = QGroupBox("Beam Information")
        beam_info_panel.setMinimumWidth(140)
        beam_info_layout = QVBoxLayout(beam_info_panel)
        beam_info_layout.setSpacing(5)

        self.centroid_x_label = QLabel("Centroid X: -")
        self.centroid_y_label = QLabel("Centroid Y: -")
        self.rms_x_label = QLabel("RMS X: -")
        self.rms_y_label = QLabel("RMS Y: -")

        font = QFont()
        font.setPointSize(9)
        for lbl in (self.centroid_x_label, self.centroid_y_label,
                    self.rms_x_label, self.rms_y_label):
            lbl.setFont(font)

        beam_info_layout.addWidget(self.centroid_x_label)
        beam_info_layout.addWidget(self.centroid_y_label)
        beam_info_layout.addWidget(self.rms_x_label)
        beam_info_layout.addWidget(self.rms_y_label)

        # Assemble top panel
        top_panel.addWidget(mode_panel, 1)
        top_panel.addWidget(camera_panel, 1)
        top_panel.addWidget(camera_info_panel, 1)
        top_panel.addWidget(background_panel, 1)
        empty_panel = QGroupBox()
        empty_panel.setMinimumWidth(120)
        empty_panel.setStyleSheet("border: none; background-color: transparent;")
        QVBoxLayout(empty_panel)
        top_panel.addWidget(empty_panel, 1)
        top_panel.addWidget(beam_info_panel, 1)

        # ── Plot area ─────────────────────────────────────────────────────────
        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(400, 400)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.plot_widget, 1)

        self.setLayout(main_layout)
        self.update_ui_state()

    # ── Signal connections ────────────────────────────────────────────────────

    def connect_signals(self):
        self.camera_radio.toggled.connect(self.on_mode_changed)
        self.file_radio.toggled.connect(self.on_mode_changed)
        self.open_file_btn.clicked.connect(self.on_open_file)

        self.refresh_cameras_btn.clicked.connect(self.on_refresh_cameras)
        self.launch_camera_btn.clicked.connect(self.on_launch_camera)
        self.stop_camera_btn.clicked.connect(self.on_stop_camera)

        self.get_background_btn.clicked.connect(self.on_get_background)
        self.stop_bg_collection_btn.clicked.connect(self.on_stop_bg_collection)

        self.camera_radio.setChecked(True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _populate_camera_combo(self, camera_list):
        for name in camera_list:
            self.camera_combo.addItem(name)

    # ── UI state ──────────────────────────────────────────────────────────────

    def update_ui_state(self):
        is_camera_mode = self.main_window.current_mode == "camera"

        self.camera_radio.setChecked(is_camera_mode)
        self.file_radio.setChecked(not is_camera_mode)

        self.open_file_btn.setEnabled(not is_camera_mode)

        if is_camera_mode:
            camera_connected = self.main_window.camera_manager.is_connected

            self.camera_combo.setEnabled(not camera_connected)
            self.refresh_cameras_btn.setEnabled(not camera_connected)
            self.launch_camera_btn.setEnabled(not camera_connected)
            self.stop_camera_btn.setEnabled(camera_connected)

            has_background = self.main_window.image_reader.background is not None
            is_collecting_bg = self.main_window.image_reader.is_collecting_background

            self.bg_frames_spinbox.setEnabled(camera_connected and not is_collecting_bg)
            self.get_background_btn.setEnabled(camera_connected and not is_collecting_bg)
            self.stop_bg_collection_btn.setEnabled(camera_connected and is_collecting_bg)

        else:
            self.camera_combo.setEnabled(False)
            self.refresh_cameras_btn.setEnabled(False)
            self.launch_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(False)
            self.bg_frames_spinbox.setEnabled(False)
            self.get_background_btn.setEnabled(False)
            self.stop_bg_collection_btn.setEnabled(False)

        self.update_camera_info()

    def update_camera_info(self):
        is_camera_mode = self.main_window.current_mode == "camera"

        if is_camera_mode and not self.main_window.camera_manager.is_connected:
            self.camera_name_label.setText("Name: -")
            self.camera_id_label.setText("ID: -")
            self.camera_resolution_label.setText("Resolution: -")
            self.camera_pixel_size_label.setText("Pixel size: -")
            return

        camera_info = self.main_window.get_camera_info()
        if camera_info:
            self.camera_name_label.setText(
                "Name: {}".format(camera_info.get('model', '-')))
            self.camera_id_label.setText(
                "ID: {}".format(camera_info.get('guid_str', '-')))
            resolution = camera_info.get("resolution", (0, 0))
            self.camera_resolution_label.setText(
                "Resolution: {} x {}".format(resolution[0], resolution[1]))
            px = camera_info.get("pixel_size_x", 0)
            py = camera_info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText(
                "Pixel size: {:.8f} x {:.8f} mm".format(px, py))
        else:
            self.camera_name_label.setText("Name: -")
            self.camera_id_label.setText("ID: -")
            self.camera_resolution_label.setText("Resolution: -")
            self.camera_pixel_size_label.setText("Pixel size: -")

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _build_axes(self, img_data, x_mm, y_mm):
        """
        Create the Figure / axes / artists from scratch.
        Called once on first render and whenever the image resolution changes.
        """
        # Tear down previous canvas if any
        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.figure.clear()
            self.plot_canvas.close()
            self.plot_canvas = None

        fig = Figure(figsize=(10, 8))
        gs = GridSpec(2, 3,
                      width_ratios=[0.5, 5, 0.3],
                      height_ratios=[5, 0.8],
                      figure=fig)

        self._ax_y_proj  = fig.add_subplot(gs[0, 0])
        self._ax_heatmap = fig.add_subplot(gs[0, 1])
        self._ax_x_proj  = fig.add_subplot(gs[1, 1])
        cax              = fig.add_subplot(gs[0, 2])

        self._im = self._ax_heatmap.imshow(
            img_data,
            extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
            origin='lower', aspect='auto', cmap='jet')
        self._ax_heatmap.set_xlabel('X (mm)')
        self._ax_heatmap.set_ylabel('Y (mm)')
        self._ax_heatmap.set_title('Beam Profile')
        fig.colorbar(self._im, cax=cax, label='Intensity')

        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)

        self._line_x, = self._ax_x_proj.plot(x_mm, x_proj, 'b-', linewidth=1)
        self._ax_x_proj.set_xlabel('X (mm)')
        self._ax_x_proj.set_ylabel('Intensity')
        self._ax_x_proj.grid(True, linestyle='--', alpha=0.7)

        self._line_y, = self._ax_y_proj.plot(y_proj, y_mm, 'b-', linewidth=1)
        self._ax_y_proj.set_ylabel('Y (mm)')
        self._ax_y_proj.set_xlabel('Intensity')
        self._ax_y_proj.grid(True, linestyle='--', alpha=0.7)

        self._ax_y_proj.set_ylim(self._ax_heatmap.get_ylim())
        self._ax_x_proj.set_xlim(self._ax_heatmap.get_xlim())

        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                            wspace=0.3, hspace=0.3)

        self.plot_canvas = FigureCanvasQTAgg(fig)
        self.plot_layout.addWidget(self.plot_canvas)

    def _update_artists(self, img_data, x_mm, y_mm):
        """
        Update existing artists in-place — no widget recreation.
        """
        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)

        # Heatmap
        self._im.set_data(img_data)
        self._im.set_extent([x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]])
        self._im.set_clim(vmin=img_data.min(), vmax=img_data.max())

        # X projection
        self._line_x.set_data(x_mm, x_proj)
        self._ax_x_proj.set_xlim(x_mm[0], x_mm[-1])
        self._ax_x_proj.set_ylim(0, x_proj.max() * 1.05 if x_proj.max() > 0 else 1)

        # Y projection
        self._line_y.set_data(y_proj, y_mm)
        self._ax_y_proj.set_ylim(y_mm[0], y_mm[-1])
        self._ax_y_proj.set_xlim(0, y_proj.max() * 1.05 if y_proj.max() > 0 else 1)

        self.plot_canvas.draw_idle()

    # ── Plot update ───────────────────────────────────────────────────────────

    def update_plots(self):
        if self.main_window.current_mode == "camera" \
                and not self.main_window.camera_manager.is_connected:
            return

        data = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()

        if data["current_frame"] is None or camera_info is None:
            return

        current_hash = hash(str(data["current_frame"].data.tobytes()))
        if self.last_data_hash == current_hash and self.plot_canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)

        img_data = np.asarray(data["current_frame"].data)
        height, width = img_data.shape
        x_mm = (np.arange(width)  - width  / 2) * pixel_size_x
        y_mm = (np.arange(height) - height / 2) * pixel_size_y

        # Recreate axes if this is the first render or resolution changed
        if self.plot_canvas is None or self._last_img_shape != img_data.shape:
            self._build_axes(img_data, x_mm, y_mm)
            self._last_img_shape = img_data.shape
        else:
            self._update_artists(img_data, x_mm, y_mm)

        centroid_x, centroid_y = data["centroid"]
        rms_x, rms_y = data["rms"]

        self.centroid_x_label.setText("Centroid X: {:.6f} mm".format(centroid_x))
        self.centroid_y_label.setText("Centroid Y: {:.6f} mm".format(centroid_y))
        self.rms_x_label.setText("RMS X: {:.6f} mm".format(rms_x))
        self.rms_y_label.setText("RMS Y: {:.6f} mm".format(rms_y))

    def update_tab(self):
        self.update_ui_state()
        self.update_plots()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def on_mode_changed(self, checked):
        if checked:
            if self.camera_radio.isChecked():
                self.main_window.switch_mode("camera")
            else:
                self.main_window.switch_mode("file")

    def on_open_file(self):
        if self.main_window.open_file_dialog():
            self.file_radio.setChecked(True)

    def on_refresh_cameras(self):
        current_text = self.camera_combo.currentText()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("Not selected")
        camera_list = self.main_window.camera_manager.get_camera_list()
        self._populate_camera_combo(camera_list)
        idx = self.camera_combo.findText(current_text)
        self.camera_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.camera_combo.blockSignals(False)

    def on_launch_camera(self):
        camera_name = self.camera_combo.currentText()
        if camera_name == "Not selected":
            QMessageBox.warning(self, "Warning", "Select a camera first.")
            return
        if not self.main_window.connect_to_camera(camera_name):
            QMessageBox.critical(self, "Error",
                                 "Failed to connect to camera:\n{}".format(camera_name))

    def on_stop_camera(self):
        if not self.main_window.disconnect_camera():
            QMessageBox.critical(self, "Error", "Failed to disconnect from camera.")

    def on_get_background(self):
        frames_count = self.bg_frames_spinbox.value()
        reply = QMessageBox.question(
            self, "Background Collection",
            "Ensure the laser shutter is closed. Start background collection?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if not self.main_window.start_background_collection(frames_count):
                QMessageBox.critical(self, "Error", "Failed to start background collection.")

    def on_stop_bg_collection(self):
        if not self.main_window.stop_background_collection():
            QMessageBox.critical(self, "Error", "Failed to stop background collection.")