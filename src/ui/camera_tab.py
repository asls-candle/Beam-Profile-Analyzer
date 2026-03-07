# src/ui/camera_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QRadioButton, QButtonGroup,
                             QSpinBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT
from src.ui.roi_selector import RoiSelector

_ROI_LINE_KW = dict(color='red', linewidth=1.2, linestyle='--', alpha=0.9, zorder=6)


class CameraTab(QWidget):
    """
    Tab for main capture and camera control.

    ROI: left-click+drag on heatmap → sets ROI, recalculates centroid/RMS,
    draws yellow rectangle + dashed gold lines on the projection plots.
    Double-click or "Reset ROI" button → clears ROI.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.last_data_hash = None

        # Persistent plot artists
        self.plot_canvas     = None
        self._im             = None
        self._line_x         = None
        self._line_y         = None
        self._ax_heatmap     = None
        self._ax_x_proj      = None
        self._ax_y_proj      = None
        self._last_img_shape = None

        # ROI boundary lines on projections
        self._roi_vline_x0 = None   # left  vertical line on ax_x_proj
        self._roi_vline_x1 = None   # right vertical line on ax_x_proj
        self._roi_hline_y0 = None   # bottom horizontal line on ax_y_proj
        self._roi_hline_y1 = None   # top    horizontal line on ax_y_proj

        # ROI state
        self._roi_selector = None
        self._roi_patch    = None
        self._img_width    = 0
        self._img_height   = 0
        self._pixel_size_x = 1.0
        self._pixel_size_y = 1.0

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(500)

        self.init_ui()
        self.connect_signals()

    # ── UI construction ───────────────────────────────────────────────────────

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)

        mode_panel = QGroupBox("Operation Mode")
        mode_panel.setMinimumWidth(120)
        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setSpacing(5)
        self.camera_radio = QRadioButton("Camera")
        self.file_radio   = QRadioButton("File Reading")
        self.mode_group   = QButtonGroup()
        self.mode_group.addButton(self.camera_radio, 0)
        self.mode_group.addButton(self.file_radio,   1)
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.open_file_btn.setFixedHeight(BUTTON_MIN_HEIGHT)
        mode_layout.addWidget(self.camera_radio)
        mode_layout.addWidget(self.file_radio)
        mode_layout.addWidget(self.open_file_btn)

        camera_panel = QGroupBox("Camera Control")
        camera_panel.setMinimumWidth(120)
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setSpacing(5)
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Not selected")
        self._populate_camera_combo(self.main_window.camera_manager.get_camera_list())
        self.refresh_cameras_btn  = QPushButton("Refresh")
        self.launch_camera_btn    = QPushButton("Launch camera")
        self.stop_camera_btn      = QPushButton("Stop camera")
        for btn in (self.refresh_cameras_btn, self.launch_camera_btn, self.stop_camera_btn):
            btn.setMinimumWidth(BUTTON_MIN_WIDTH)
            btn.setFixedHeight(BUTTON_MIN_HEIGHT)
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.refresh_cameras_btn)
        camera_layout.addWidget(self.launch_camera_btn)
        camera_layout.addWidget(self.stop_camera_btn)

        camera_info_panel = QGroupBox("Camera Information")
        camera_info_panel.setMinimumWidth(120)
        ci_layout = QVBoxLayout(camera_info_panel)
        ci_layout.setSpacing(4)
        self.camera_name_label       = QLabel("Name: -")
        self.camera_id_label         = QLabel("ID: -")
        self.camera_resolution_label = QLabel("Resolution: -")
        self.camera_pixel_size_label = QLabel("Pixel size: -")
        for lbl in (self.camera_name_label, self.camera_id_label,
                    self.camera_resolution_label, self.camera_pixel_size_label):
            ci_layout.addWidget(lbl)

        background_panel = QGroupBox("Background Collection")
        background_panel.setMinimumWidth(120)
        bg_layout = QVBoxLayout(background_panel)
        bg_layout.setSpacing(5)
        bg_frames_row = QHBoxLayout()
        bg_frames_row.addWidget(QLabel("Frames:"))
        self.bg_frames_spinbox = QSpinBox()
        self.bg_frames_spinbox.setRange(1, 100)
        self.bg_frames_spinbox.setValue(40)
        bg_frames_row.addWidget(self.bg_frames_spinbox)
        self.get_background_btn      = QPushButton("Get background")
        self.stop_bg_collection_btn  = QPushButton("Stop")
        for btn in (self.get_background_btn, self.stop_bg_collection_btn):
            btn.setMinimumWidth(BUTTON_MIN_WIDTH)
            btn.setFixedHeight(BUTTON_MIN_HEIGHT)
        bg_layout.addLayout(bg_frames_row)
        bg_layout.addWidget(self.get_background_btn)
        bg_layout.addWidget(self.stop_bg_collection_btn)

        beam_info_panel = QGroupBox("Beam Information")
        beam_info_panel.setMinimumWidth(140)
        bi_layout = QVBoxLayout(beam_info_panel)
        bi_layout.setSpacing(5)
        self.centroid_x_label  = QLabel("Centroid X: -")
        self.centroid_y_label  = QLabel("Centroid Y: -")
        self.rms_x_label       = QLabel("RMS X: -")
        self.rms_y_label       = QLabel("RMS Y: -")
        self.roi_status_label  = QLabel("ROI: None")
        self.roi_status_label.setStyleSheet("color: gray;")
        self.reset_roi_btn = QPushButton("Reset ROI")
        self.reset_roi_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.reset_roi_btn.setFixedHeight(BUTTON_MIN_HEIGHT)
        self.reset_roi_btn.setEnabled(False)
        self.reset_roi_btn.clicked.connect(self._on_roi_reset)
        font = QFont()
        font.setPointSize(9)
        for lbl in (self.centroid_x_label, self.centroid_y_label,
                    self.rms_x_label, self.rms_y_label, self.roi_status_label):
            lbl.setFont(font)
        for w in (self.centroid_x_label, self.centroid_y_label,
                  self.rms_x_label, self.rms_y_label,
                  self.roi_status_label, self.reset_roi_btn):
            bi_layout.addWidget(w)

        top_panel.addWidget(mode_panel, 1)
        top_panel.addWidget(camera_panel, 1)
        top_panel.addWidget(camera_info_panel, 1)
        top_panel.addWidget(background_panel, 1)
        empty = QGroupBox()
        empty.setMinimumWidth(120)
        empty.setStyleSheet("border: none; background-color: transparent;")
        QVBoxLayout(empty)
        top_panel.addWidget(empty, 1)
        top_panel.addWidget(beam_info_panel, 1)

        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(400, 400)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.plot_widget, 1)
        self.setLayout(main_layout)
        self.update_ui_state()

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

    def _populate_camera_combo(self, camera_list):
        for name in camera_list:
            self.camera_combo.addItem(name)

    # ── UI state ──────────────────────────────────────────────────────────────

    def update_ui_state(self):
        is_camera = self.main_window.current_mode == "camera"
        self.camera_radio.setChecked(is_camera)
        self.file_radio.setChecked(not is_camera)
        self.open_file_btn.setEnabled(not is_camera)
        if is_camera:
            connected     = self.main_window.camera_manager.is_connected
            has_bg        = self.main_window.image_reader.background is not None
            collecting_bg = self.main_window.image_reader.is_collecting_background
            self.camera_combo.setEnabled(not connected)
            self.refresh_cameras_btn.setEnabled(not connected)
            self.launch_camera_btn.setEnabled(not connected)
            self.stop_camera_btn.setEnabled(connected)
            self.bg_frames_spinbox.setEnabled(connected and not collecting_bg)
            self.get_background_btn.setEnabled(connected and not collecting_bg)
            self.stop_bg_collection_btn.setEnabled(connected and collecting_bg)
        else:
            for w in (self.camera_combo, self.refresh_cameras_btn,
                      self.launch_camera_btn, self.stop_camera_btn,
                      self.bg_frames_spinbox, self.get_background_btn,
                      self.stop_bg_collection_btn):
                w.setEnabled(False)
        self.update_camera_info()

    def update_camera_info(self):
        is_camera = self.main_window.current_mode == "camera"
        if is_camera and not self.main_window.camera_manager.is_connected:
            for lbl in (self.camera_name_label, self.camera_id_label,
                        self.camera_resolution_label, self.camera_pixel_size_label):
                lbl.setText(lbl.text().split(':')[0] + ': -')
            return
        info = self.main_window.get_camera_info()
        if info:
            self.camera_name_label.setText("Name: {}".format(info.get('model', '-')))
            self.camera_id_label.setText("ID: {}".format(info.get('guid_str', '-')))
            res = info.get("resolution", (0, 0))
            self.camera_resolution_label.setText("Resolution: {} x {}".format(*res))
            px, py = info.get("pixel_size_x", 0), info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText(
                "Pixel size: {:.8f} x {:.8f} mm".format(px, py))
        else:
            self.camera_name_label.setText("Name: -")
            self.camera_id_label.setText("ID: -")
            self.camera_resolution_label.setText("Resolution: -")
            self.camera_pixel_size_label.setText("Pixel size: -")

    # ── ROI helpers ───────────────────────────────────────────────────────────

    def _on_roi_selected(self, x0, y0, x1, y1):
        if self._img_width == 0 or self._img_height == 0:
            return
        self.main_window.set_roi_mm(x0, y0, x1, y1,
                                    self._img_width, self._img_height,
                                    self._pixel_size_x, self._pixel_size_y)
        self._draw_roi_patch(x0, y0, x1, y1)
        self._show_roi_lines(x0, y0, x1, y1)
        self._update_roi_status(x0, y0, x1, y1)
        self.reset_roi_btn.setEnabled(True)
        self.last_data_hash = None          # force recalculation
        if self.plot_canvas:
            self.plot_canvas.draw_idle()

    def _on_roi_reset(self):
        self.main_window.reset_roi()
        self._remove_roi_patch()
        self._hide_roi_lines()
        self.roi_status_label.setText("ROI: None")
        self.roi_status_label.setStyleSheet("color: gray;")
        self.reset_roi_btn.setEnabled(False)
        self.last_data_hash = None          # force recalculation on full image
        if self.plot_canvas:
            self.plot_canvas.draw_idle()

    def _draw_roi_patch(self, x0, y0, x1, y1):
        if self._ax_heatmap is None:
            return
        self._remove_roi_patch()
        self._roi_patch = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.0, edgecolor='yellow',
            facecolor=(1.0, 1.0, 0.0, 0.06),
            linestyle='-', zorder=8)
        self._ax_heatmap.add_patch(self._roi_patch)

    def _remove_roi_patch(self):
        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except ValueError:
                pass
            self._roi_patch = None

    def _show_roi_lines(self, x0, y0, x1, y1):
        """Create or move the four boundary lines on the projection axes."""
        if self._ax_x_proj is None or self._ax_y_proj is None:
            return
        # Vertical lines on X-projection (bottom plot)
        if self._roi_vline_x0 is None:
            self._roi_vline_x0 = self._ax_x_proj.axvline(x0, **_ROI_LINE_KW)
            self._roi_vline_x1 = self._ax_x_proj.axvline(x1, **_ROI_LINE_KW)
        else:
            self._roi_vline_x0.set_xdata([x0, x0])
            self._roi_vline_x1.set_xdata([x1, x1])
            self._roi_vline_x0.set_visible(True)
            self._roi_vline_x1.set_visible(True)
        # Horizontal lines on Y-projection (left plot)
        if self._roi_hline_y0 is None:
            self._roi_hline_y0 = self._ax_y_proj.axhline(y0, **_ROI_LINE_KW)
            self._roi_hline_y1 = self._ax_y_proj.axhline(y1, **_ROI_LINE_KW)
        else:
            self._roi_hline_y0.set_ydata([y0, y0])
            self._roi_hline_y1.set_ydata([y1, y1])
            self._roi_hline_y0.set_visible(True)
            self._roi_hline_y1.set_visible(True)

    def _hide_roi_lines(self):
        for line in (self._roi_vline_x0, self._roi_vline_x1,
                     self._roi_hline_y0, self._roi_hline_y1):
            if line is not None:
                line.set_visible(False)

    def _update_roi_status(self, x0, y0, x1, y1):
        self.roi_status_label.setText(
            "ROI: [{:.2f},{:.2f}]–[{:.2f},{:.2f}] mm".format(x0, y0, x1, y1))
        self.roi_status_label.setStyleSheet("color: #e6a817;")

    def _restore_roi_from_shared_state(self):
        roi = self.main_window.current_roi_mm
        if roi is not None:
            self._draw_roi_patch(*roi)
            self._show_roi_lines(*roi)
            self._update_roi_status(*roi)
            self.reset_roi_btn.setEnabled(True)

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _build_axes(self, img_data, x_mm, y_mm):
        if self._roi_selector is not None:
            self._roi_selector.disconnect()
            self._roi_selector = None
        self._roi_patch = None
        self._roi_vline_x0 = self._roi_vline_x1 = None
        self._roi_hline_y0 = self._roi_hline_y1 = None

        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.figure.clear()
            self.plot_canvas.close()
            self.plot_canvas = None

        fig = Figure(figsize=(10, 8))
        gs  = GridSpec(2, 3, width_ratios=[0.5, 5, 0.3],
                       height_ratios=[5, 0.8], figure=fig)

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
        self._ax_heatmap.set_title(
            'Beam Profile — drag to select ROI, dbl-click to reset')
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
        self._roi_selector = RoiSelector(
            self._ax_heatmap, self._on_roi_selected, self._on_roi_reset)
        self._restore_roi_from_shared_state()

    def _update_artists(self, img_data, x_mm, y_mm):
        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)

        self._im.set_data(img_data)
        self._im.set_extent([x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]])
        self._im.set_clim(vmin=img_data.min(), vmax=img_data.max())

        self._line_x.set_data(x_mm, x_proj)
        self._ax_x_proj.set_xlim(x_mm[0], x_mm[-1])
        self._ax_x_proj.set_ylim(0, x_proj.max() * 1.05 if x_proj.max() > 0 else 1)

        self._line_y.set_data(y_proj, y_mm)
        self._ax_y_proj.set_ylim(y_mm[0], y_mm[-1])
        self._ax_y_proj.set_xlim(0, y_proj.max() * 1.05 if y_proj.max() > 0 else 1)

        self.plot_canvas.draw_idle()

    # ── Plot update ───────────────────────────────────────────────────────────

    def update_plots(self):
        if (self.main_window.current_mode == "camera"
                and not self.main_window.camera_manager.is_connected):
            return
        data        = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()
        if data["current_frame"] is None or camera_info is None:
            return

        current_hash = hash(str(data["current_frame"].data.tobytes()))
        if self.last_data_hash == current_hash and self.plot_canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y

        img_data = np.asarray(data["current_frame"].data)
        self._img_height, self._img_width = img_data.shape
        h, w = self._img_height, self._img_width
        x_mm = (np.arange(w) - w / 2) * pixel_size_x
        y_mm = (np.arange(h) - h / 2) * pixel_size_y

        if self.plot_canvas is None or self._last_img_shape != img_data.shape:
            self._build_axes(img_data, x_mm, y_mm)
            self._last_img_shape = img_data.shape
        else:
            self._update_artists(img_data, x_mm, y_mm)

        centroid_x, centroid_y = data["centroid"]
        rms_x, rms_y           = data["rms"]
        roi_sfx = " (ROI)" if self.main_window.image_analyzer.roi is not None else ""

        self.centroid_x_label.setText(
            "Centroid X{}: {:.6f} mm".format(roi_sfx, centroid_x))
        self.centroid_y_label.setText(
            "Centroid Y{}: {:.6f} mm".format(roi_sfx, centroid_y))
        self.rms_x_label.setText(
            "RMS X{}: {:.6f} mm".format(roi_sfx, rms_x))
        self.rms_y_label.setText(
            "RMS Y{}: {:.6f} mm".format(roi_sfx, rms_y))

    def update_tab(self):
        self.update_ui_state()
        self.update_plots()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def on_mode_changed(self, checked):
        if checked:
            self.main_window.switch_mode(
                "camera" if self.camera_radio.isChecked() else "file")

    def on_open_file(self):
        if self.main_window.open_file_dialog():
            self.file_radio.setChecked(True)

    def on_refresh_cameras(self):
        cur = self.camera_combo.currentText()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("Not selected")
        self._populate_camera_combo(self.main_window.camera_manager.get_camera_list())
        idx = self.camera_combo.findText(cur)
        self.camera_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.camera_combo.blockSignals(False)

    def on_launch_camera(self):
        name = self.camera_combo.currentText()
        if name == "Not selected":
            QMessageBox.warning(self, "Warning", "Select a camera first.")
            return
        if not self.main_window.connect_to_camera(name):
            QMessageBox.critical(self, "Error",
                                 "Failed to connect to camera:\n{}".format(name))

    def on_stop_camera(self):
        if not self.main_window.disconnect_camera():
            QMessageBox.critical(self, "Error", "Failed to disconnect from camera.")

    def on_get_background(self):
        n = self.bg_frames_spinbox.value()
        reply = QMessageBox.question(
            self, "Background Collection",
            "Ensure the laser shutter is closed. Start background collection?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if not self.main_window.start_background_collection(n):
                QMessageBox.critical(self, "Error",
                                     "Failed to start background collection.")

    def on_stop_bg_collection(self):
        if not self.main_window.stop_background_collection():
            QMessageBox.critical(self, "Error",
                                 "Failed to stop background collection.")