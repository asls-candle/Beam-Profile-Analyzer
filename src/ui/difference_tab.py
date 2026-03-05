# src/ui/difference_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QMessageBox, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT
from src.analysis.filters import apply_median_filter
from src.data.exporter import DataExporter


class DifferenceTab(QWidget):
    """
    Tab for displaying and analyzing the difference between current image and background.

    Video is shown only while actively capturing (Start → Stop).
    Stop freezes the last frame; Export saves that frozen frame.
    """

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.last_data_hash = None

        # ── Capture state ─────────────────────────────────────────────────────
        self.is_capturing = False

        # Frozen data — populated on Stop, used for Export
        self.frozen_difference = None   # Image object for display
        self.frozen_snapshot   = None   # Full dict for DataExporter

        # Persistent plot artists — created once, updated in-place afterwards
        self.plot_canvas     = None
        self._im             = None   # AxesImage from imshow
        self._line_x         = None   # Line2D for X projection
        self._line_y         = None   # Line2D for Y projection
        self._ax_heatmap     = None
        self._ax_x_proj      = None
        self._ax_y_proj      = None
        # Track image shape so we can recreate when resolution changes
        self._last_img_shape = None

        # Timer — drives live video while capturing; starts only on Start
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)

        self.init_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # ── Top panel ────────────────────────────────────────────────────────
        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)

        # Camera information
        camera_info_panel = QGroupBox("Camera Information")
        camera_info_panel.setMinimumWidth(150)
        camera_info_layout = QVBoxLayout(camera_info_panel)
        camera_info_layout.setSpacing(5)

        self.camera_name_label = QLabel("Name: -")
        self.camera_resolution_label = QLabel("Resolution: -")
        self.camera_pixel_size_label = QLabel("Pixel size: -")

        camera_info_layout.addWidget(self.camera_name_label)
        camera_info_layout.addWidget(self.camera_resolution_label)
        camera_info_layout.addWidget(self.camera_pixel_size_label)

        # Capture Control
        capture_panel = QGroupBox("Capture Control")
        capture_panel.setMinimumWidth(120)
        capture_layout = QVBoxLayout(capture_panel)
        capture_layout.setSpacing(5)

        self.start_capture_btn = QPushButton("Start")
        self.start_capture_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.start_capture_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.stop_capture_btn = QPushButton("Stop")
        self.stop_capture_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.stop_capture_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.setMinimumWidth(BUTTON_MIN_WIDTH)
        self.export_data_btn.setFixedHeight(BUTTON_MIN_HEIGHT)

        capture_layout.addWidget(self.start_capture_btn)
        capture_layout.addWidget(self.stop_capture_btn)
        capture_layout.addWidget(self.export_data_btn)

        self.start_capture_btn.clicked.connect(self.on_start_capture)
        self.stop_capture_btn.clicked.connect(self.on_stop_capture)
        self.export_data_btn.clicked.connect(self.on_export_data)

        # Status
        status_panel = QGroupBox("Status")
        status_panel.setMinimumWidth(120)
        status_layout = QVBoxLayout(status_panel)
        status_layout.setSpacing(5)

        self.mode_status_label = QLabel("Mode: Camera")
        self.camera_status_label = QLabel("Camera: Not connected")
        self.capture_status_label = QLabel("Data collection: Stopped")

        status_layout.addWidget(self.mode_status_label)
        status_layout.addWidget(self.camera_status_label)
        status_layout.addWidget(self.capture_status_label)

        # Beam information
        beam_info_panel = QGroupBox("Beam Information (difference)")
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

        # Assemble top panel — empty panels keep layout consistent with other tabs
        top_panel.addWidget(camera_info_panel)
        for _ in range(3):
            empty_panel = QGroupBox()
            empty_panel.setMinimumWidth(120)
            empty_panel.setStyleSheet(
                "border: none; background-color: transparent;")
            QVBoxLayout(empty_panel)
            top_panel.addWidget(empty_panel)
        top_panel.addWidget(status_panel)
        top_panel.addWidget(capture_panel)
        top_panel.addWidget(beam_info_panel)

        # ── Plot area ─────────────────────────────────────────────────────────
        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(400, 400)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.plot_widget, 1)

        self.setLayout(main_layout)
        self.update_ui_state()

    # ── UI state ──────────────────────────────────────────────────────────────

    def update_ui_state(self):
        is_camera_mode = self.main_window.current_mode == "camera"

        self.mode_status_label.setText(
            "Mode: {}".format('Camera' if is_camera_mode else 'File Reading'))

        if is_camera_mode:
            camera_connected = self.main_window.camera_manager.is_connected
            has_background = self.main_window.image_reader.background is not None
            is_collecting_bg = self.main_window.image_reader.is_collecting_background

            self.start_capture_btn.setEnabled(
                camera_connected and has_background and not self.is_capturing)
            self.stop_capture_btn.setEnabled(self.is_capturing)
            self.export_data_btn.setEnabled(
                self.frozen_snapshot is not None and not self.is_capturing)

            if camera_connected:
                camera_info = self.main_window.camera_manager.get_camera_info()
                short_name = camera_info.get('camera_name', 'Connected') if camera_info else 'Connected'
                self.camera_status_label.setText("Camera: {}".format(short_name))
            else:
                self.camera_status_label.setText("Camera: Not connected")

            if self.is_capturing:
                self.capture_status_label.setText("Data collection: Capturing")
            elif is_collecting_bg:
                self.capture_status_label.setText("Data collection: Background collection")
            else:
                self.capture_status_label.setText("Data collection: Stopped")

        else:
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(False)
            has_data = (self.frozen_snapshot is not None
                        or self.main_window.file_data.get("current_frame") is not None)
            self.export_data_btn.setEnabled(has_data)
            self.camera_status_label.setText("Camera: -")
            self.capture_status_label.setText("Data collection: -")

    # ── Camera info ───────────────────────────────────────────────────────────

    def update_camera_info(self):
        camera_info = self.main_window.get_camera_info()
        if camera_info:
            self.camera_name_label.setText(
                "Name: {}".format(camera_info.get('camera_name', '-')))
            resolution = camera_info.get("resolution", (0, 0))
            self.camera_resolution_label.setText(
                "Resolution: {} x {}".format(resolution[0], resolution[1]))
            px = camera_info.get("pixel_size_x", 0)
            py = camera_info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText(
                "Pixel size: {:.8f} x {:.8f} mm".format(px, py))
        else:
            self.camera_name_label.setText("Name: -")
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

    # ── Plot rendering ────────────────────────────────────────────────────────

    def _render_difference(self, difference_image, camera_info):
        """
        Renders a difference image onto the plot canvas.
        First call (or resolution change) builds the full Figure;
        subsequent calls only update artists in-place.
        """
        current_hash = hash(str(difference_image.data.tobytes()))
        if self.last_data_hash == current_hash and self.plot_canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)

        img_data = apply_median_filter(np.asarray(difference_image.data), kernel_size=3)
        height, width = img_data.shape
        x_mm = (np.arange(width)  - width  / 2) * pixel_size_x
        y_mm = (np.arange(height) - height / 2) * pixel_size_y

        # Recreate axes if this is the first render or resolution changed
        if self.plot_canvas is None or self._last_img_shape != img_data.shape:
            self._build_axes(img_data, x_mm, y_mm)
            self._last_img_shape = img_data.shape
        else:
            self._update_artists(img_data, x_mm, y_mm)

        centroid_x, centroid_y = self.main_window.image_analyzer.calculate_centroid(
            difference_image, pixel_size_x, pixel_size_y)
        rms_x, rms_y = self.main_window.image_analyzer.calculate_rms(
            difference_image, pixel_size_x, pixel_size_y)

        self.centroid_x_label.setText("Centroid X: {:.6f} mm".format(centroid_x))
        self.centroid_y_label.setText("Centroid Y: {:.6f} mm".format(centroid_y))
        self.rms_x_label.setText("RMS X: {:.6f} mm".format(rms_x))
        self.rms_y_label.setText("RMS Y: {:.6f} mm".format(rms_y))

    def update_plots(self):
        """
        Live path   — runs on every timer tick while is_capturing=True.
        Frozen path — runs once after Stop to display the last captured frame.
        Idle        — does nothing (startup, or background collected but Start not pressed).
        """
        if self.is_capturing:
            # ── Live ──────────────────────────────────────────────────────────
            data = self.main_window.get_current_data()
            camera_info = self.main_window.get_camera_info()
            if data["difference"] is None or camera_info is None:
                return
            self._render_difference(data["difference"], camera_info)

        elif self.frozen_difference is not None and self.frozen_snapshot is not None:
            # ── Frozen ────────────────────────────────────────────────────────
            camera_info = {
                "pixel_size_x": self.frozen_snapshot.get("pixel_size_x", 1),
                "pixel_size_y": self.frozen_snapshot.get("pixel_size_y", 1),
                "resolution":   self.frozen_snapshot.get("resolution", (0, 0)),
                "camera_name":  self.frozen_snapshot.get("camera_name", "-"),
            }
            self._render_difference(self.frozen_difference, camera_info)

        # else: idle — nothing to render

    def update_tab(self):
        """Called by the timer (active only during live capture)."""
        self.update_camera_info()
        self.update_ui_state()
        self.update_plots()

    # ── Capture Control handlers ──────────────────────────────────────────────

    def on_start_capture(self):
        """Start live streaming of the difference image."""
        if self.main_window.image_reader.background is None:
            QMessageBox.warning(self, "Warning",
                                "Collect background before starting capture.")
            return

        self.is_capturing = True
        self.last_data_hash = None  # force redraw on first live frame
        self.update_ui_state()
        self.update_timer.start(500)

    def on_stop_capture(self):
        """
        Stop live streaming and freeze the last frame on screen.
        The frozen frame is what Export will save.
        """
        self.update_timer.stop()
        self.is_capturing = False

        data = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()

        if data["difference"] is not None and camera_info is not None:
            self.frozen_difference = data["difference"]

            self.frozen_snapshot = {
                "shot":           data.get("current_frame"),
                "background":     data.get("background"),
                "difference":     data.get("difference"),
                "raw_shot":       (self.main_window.image_reader.get_raw_current_frame()
                                   if self.main_window.current_mode == "camera"
                                   else data.get("raw_shot")),
                "raw_background": (self.main_window.image_reader.get_raw_background()
                                   if self.main_window.current_mode == "camera"
                                   else data.get("raw_background")),
                "resolution":     camera_info.get("resolution", (0, 0)),
                "pixel_size_x":   camera_info.get("pixel_size_x", 1.0),
                "pixel_size_y":   camera_info.get("pixel_size_y", 1.0),
                "camera_name":    camera_info.get("camera_name", "Unknown camera"),
            }

            # Force one final render of the frozen frame
            self.last_data_hash = None
            self.update_plots()

        self.update_ui_state()

    def on_export_data(self):
        """Export the frame that was frozen on Stop."""
        if self.frozen_snapshot is None:
            QMessageBox.warning(self, "Warning",
                                "No frozen frame to export. "
                                "Press Start then Stop first.")
            return

        formats = {
            "folder": "Data folder (CSV+JSON)",
            "png":    "PNG images folder",
        }
        filter_str = ";;".join(formats.values())

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export frozen frame",
            "",
            filter_str,
            formats["folder"],
        )
        if not filepath:
            return

        export_format = next(
            (fmt for fmt, name in formats.items() if name == selected_filter),
            "folder",
        )

        result = DataExporter.export_data(
            filepath,
            self.frozen_snapshot,
            export_format,
            self.main_window.plot_manager,
        )

        if result:
            self.main_window.statusBar().showMessage(
                "Data exported to: {}".format(result), 5000)
            QMessageBox.information(self, "Information",
                                    "Data successfully exported to {}".format(result))
        else:
            QMessageBox.warning(self, "Warning", "Error exporting data.")