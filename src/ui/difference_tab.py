# src/ui/difference_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGroupBox, QMessageBox, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT
from src.analysis.filters import apply_median_filter
from src.data.exporter import DataExporter
from src.ui.roi_selector import RoiSelector

_ROI_LINE_KW = dict(color='red', linewidth=1.2, linestyle='--', alpha=0.9, zorder=6)


class DifferenceTab(QWidget):
    """
    Tab for displaying and analyzing the difference between current image and background.

    Video is shown only while actively capturing (Start → Stop).
    Stop freezes the last frame; Export saves that frozen frame.

    ROI: left-click+drag on heatmap → sets ROI, recalculates centroid/RMS,
    draws yellow rectangle + dashed gold lines on projection plots.
    If an ROI is active when Export is pressed, only the ROI region is saved.
    Double-click or "Reset ROI" button → clears ROI.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window    = main_window
        self.last_data_hash = None

        # Capture state
        self.is_capturing      = False
        self.frozen_difference = None
        self.frozen_snapshot   = None

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
        self._roi_vline_x0 = None
        self._roi_vline_x1 = None
        self._roi_hline_y0 = None
        self._roi_hline_y1 = None

        # ROI state
        self._roi_selector = None
        self._roi_patch    = None
        self._img_width    = 0
        self._img_height   = 0
        self._pixel_size_x = 1.0
        self._pixel_size_y = 1.0

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)

        self.init_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_panel = QHBoxLayout()
        top_panel.setSpacing(10)
        top_panel_widget = QWidget()
        top_panel_widget.setLayout(top_panel)
        top_panel_widget.setFixedHeight(TOP_PANEL_HEIGHT)

        camera_info_panel = QGroupBox("Camera Information")
        camera_info_panel.setMinimumWidth(150)
        ci_layout = QVBoxLayout(camera_info_panel)
        ci_layout.setSpacing(5)
        self.camera_name_label       = QLabel("Name: -")
        self.camera_resolution_label = QLabel("Resolution: -")
        self.camera_pixel_size_label = QLabel("Pixel size: -")
        for lbl in (self.camera_name_label, self.camera_resolution_label,
                    self.camera_pixel_size_label):
            ci_layout.addWidget(lbl)

        capture_panel = QGroupBox("Capture Control")
        capture_panel.setMinimumWidth(120)
        cap_layout = QVBoxLayout(capture_panel)
        cap_layout.setSpacing(5)
        self.start_capture_btn = QPushButton("Start")
        self.stop_capture_btn  = QPushButton("Stop")
        self.export_data_btn   = QPushButton("Export Data")
        for btn in (self.start_capture_btn, self.stop_capture_btn, self.export_data_btn):
            btn.setMinimumWidth(BUTTON_MIN_WIDTH)
            btn.setFixedHeight(BUTTON_MIN_HEIGHT)
            cap_layout.addWidget(btn)
        self.start_capture_btn.clicked.connect(self.on_start_capture)
        self.stop_capture_btn.clicked.connect(self.on_stop_capture)
        self.export_data_btn.clicked.connect(self.on_export_data)

        status_panel = QGroupBox("Status")
        status_panel.setMinimumWidth(120)
        st_layout = QVBoxLayout(status_panel)
        st_layout.setSpacing(5)
        self.mode_status_label    = QLabel("Mode: Camera")
        self.camera_status_label  = QLabel("Camera: Not connected")
        self.capture_status_label = QLabel("Data collection: Stopped")
        for lbl in (self.mode_status_label, self.camera_status_label,
                    self.capture_status_label):
            st_layout.addWidget(lbl)

        beam_info_panel = QGroupBox("Beam Information (difference)")
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

        top_panel.addWidget(camera_info_panel)
        for _ in range(3):
            ep = QGroupBox()
            ep.setMinimumWidth(120)
            ep.setStyleSheet("border: none; background-color: transparent;")
            QVBoxLayout(ep)
            top_panel.addWidget(ep)
        top_panel.addWidget(status_panel)
        top_panel.addWidget(capture_panel)
        top_panel.addWidget(beam_info_panel)

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
        is_camera = self.main_window.current_mode == "camera"
        self.mode_status_label.setText(
            "Mode: {}".format('Camera' if is_camera else 'File Reading'))
        if is_camera:
            connected     = self.main_window.camera_manager.is_connected
            has_bg        = self.main_window.image_reader.background is not None
            collecting_bg = self.main_window.image_reader.is_collecting_background
            self.start_capture_btn.setEnabled(
                connected and has_bg and not self.is_capturing)
            self.stop_capture_btn.setEnabled(self.is_capturing)
            self.export_data_btn.setEnabled(
                self.frozen_snapshot is not None and not self.is_capturing)
            if connected:
                info = self.main_window.camera_manager.get_camera_info()
                name = info.get('camera_name', 'Connected') if info else 'Connected'
                self.camera_status_label.setText("Camera: {}".format(name))
            else:
                self.camera_status_label.setText("Camera: Not connected")
            if self.is_capturing:
                self.capture_status_label.setText("Data collection: Capturing")
            elif collecting_bg:
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

    def update_camera_info(self):
        info = self.main_window.get_camera_info()
        if info:
            self.camera_name_label.setText(
                "Name: {}".format(info.get('camera_name', '-')))
            res = info.get("resolution", (0, 0))
            self.camera_resolution_label.setText(
                "Resolution: {} x {}".format(*res))
            px, py = info.get("pixel_size_x", 0), info.get("pixel_size_y", 0)
            self.camera_pixel_size_label.setText(
                "Pixel size: {:.8f} x {:.8f} mm".format(px, py))
        else:
            self.camera_name_label.setText("Name: -")
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
        self.update_plots()  # recalc immediately (works for frozen frame too)

    def _on_roi_reset(self):
        self.main_window.reset_roi()
        self._remove_roi_patch()
        self._hide_roi_lines()
        self.roi_status_label.setText("ROI: None")
        self.roi_status_label.setStyleSheet("color: gray;")
        self.reset_roi_btn.setEnabled(False)
        self.last_data_hash = None
        self.update_plots()  # recalc on full image immediately

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
        if self._ax_x_proj is None or self._ax_y_proj is None:
            return
        if self._roi_vline_x0 is None:
            self._roi_vline_x0 = self._ax_x_proj.axvline(x0, **_ROI_LINE_KW)
            self._roi_vline_x1 = self._ax_x_proj.axvline(x1, **_ROI_LINE_KW)
        else:
            self._roi_vline_x0.set_xdata([x0, x0]); self._roi_vline_x0.set_visible(True)
            self._roi_vline_x1.set_xdata([x1, x1]); self._roi_vline_x1.set_visible(True)
        if self._roi_hline_y0 is None:
            self._roi_hline_y0 = self._ax_y_proj.axhline(y0, **_ROI_LINE_KW)
            self._roi_hline_y1 = self._ax_y_proj.axhline(y1, **_ROI_LINE_KW)
        else:
            self._roi_hline_y0.set_ydata([y0, y0]); self._roi_hline_y0.set_visible(True)
            self._roi_hline_y1.set_ydata([y1, y1]); self._roi_hline_y1.set_visible(True)

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
        self._roi_patch    = None
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

    # ── Plot rendering ────────────────────────────────────────────────────────

    def _render_difference(self, difference_image, camera_info):
        current_hash = hash(str(difference_image.data.tobytes()))
        if self.last_data_hash == current_hash and self.plot_canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y

        img_data = apply_median_filter(np.asarray(difference_image.data), kernel_size=3)
        self._img_height, self._img_width = img_data.shape
        h, w = self._img_height, self._img_width
        x_mm = (np.arange(w) - w / 2) * pixel_size_x
        y_mm = (np.arange(h) - h / 2) * pixel_size_y

        if self.plot_canvas is None or self._last_img_shape != img_data.shape:
            self._build_axes(img_data, x_mm, y_mm)
            self._last_img_shape = img_data.shape
        else:
            self._update_artists(img_data, x_mm, y_mm)

        # Centroid / RMS — computed on ROI region if ROI is active
        cx, cy = self.main_window.image_analyzer.calculate_centroid(
            difference_image, pixel_size_x, pixel_size_y)
        rx, ry = self.main_window.image_analyzer.calculate_rms(
            difference_image, pixel_size_x, pixel_size_y)

        roi_sfx = " (ROI)" if self.main_window.image_analyzer.roi is not None else ""
        self.centroid_x_label.setText(
            "Centroid X{}: {:.6f} mm".format(roi_sfx, cx))
        self.centroid_y_label.setText(
            "Centroid Y{}: {:.6f} mm".format(roi_sfx, cy))
        self.rms_x_label.setText("RMS X{}: {:.6f} mm".format(roi_sfx, rx))
        self.rms_y_label.setText("RMS Y{}: {:.6f} mm".format(roi_sfx, ry))

    def update_plots(self):
        if self.is_capturing:
            data        = self.main_window.get_current_data()
            camera_info = self.main_window.get_camera_info()
            if data["difference"] is None or camera_info is None:
                return
            self._render_difference(data["difference"], camera_info)
        elif self.frozen_difference is not None and self.frozen_snapshot is not None:
            camera_info = {
                "pixel_size_x": self.frozen_snapshot.get("pixel_size_x", 1),
                "pixel_size_y": self.frozen_snapshot.get("pixel_size_y", 1),
                "resolution":   self.frozen_snapshot.get("resolution", (0, 0)),
                "camera_name":  self.frozen_snapshot.get("camera_name", "-"),
            }
            self._render_difference(self.frozen_difference, camera_info)

    def update_tab(self):
        self.update_camera_info()
        self.update_ui_state()
        self.update_plots()

    # ── Capture Control handlers ──────────────────────────────────────────────

    def on_start_capture(self):
        if self.main_window.image_reader.background is None:
            QMessageBox.warning(self, "Warning",
                                "Collect background before starting capture.")
            return
        self.is_capturing = True
        self.last_data_hash = None
        self.update_ui_state()
        self.update_timer.start(500)

    def on_stop_capture(self):
        self.update_timer.stop()
        self.is_capturing = False
        data        = self.main_window.get_current_data()
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
            self.last_data_hash = None
            self.update_plots()
        self.update_ui_state()

    # ── Export ────────────────────────────────────────────────────────────────

    def on_export_data(self):
        """Export frame (cropped to ROI if active)."""
        if self.main_window.current_mode == "file":
            snapshot = dict(self.main_window.file_data)
            if "current_frame" in snapshot and "shot" not in snapshot:
                snapshot["shot"] = snapshot["current_frame"]
            info = self.main_window.get_camera_info()
            if info:
                snapshot.setdefault("resolution",   info.get("resolution", (0, 0)))
                snapshot.setdefault("pixel_size_x", info.get("pixel_size_x", 1.0))
                snapshot.setdefault("pixel_size_y", info.get("pixel_size_y", 1.0))
                snapshot.setdefault("camera_name",  info.get("camera_name", "Unknown camera"))
        else:
            if self.frozen_snapshot is None:
                QMessageBox.warning(self, "Warning",
                                    "No frozen frame to export. "
                                    "Press Start then Stop first.")
                return
            snapshot = self.frozen_snapshot

        # Apply ROI crop if active
        snapshot = self.main_window.apply_roi_to_data(snapshot)

        roi_active = self.main_window.image_analyzer.roi is not None
        if roi_active:
            reply = QMessageBox.question(
                self, "ROI active",
                "An ROI is currently selected.\n"
                "The export will contain only the ROI region.\n\nProceed?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return

        formats = {"folder": "Data folder (CSV+JSON)", "png": "PNG images folder"}
        filter_str = ";;".join(formats.values())
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export{}frame".format(" ROI " if roi_active else " frozen "),
            "", filter_str, formats["folder"])
        if not filepath:
            return

        export_format = next(
            (f for f, n in formats.items() if n == selected_filter), "folder")

        result = DataExporter.export_data(
            filepath, snapshot, export_format, self.main_window.plot_manager)

        if result:
            self.main_window.statusBar().showMessage(
                "Data exported to: {}".format(result), 5000)
            QMessageBox.information(self, "Information",
                                    "Data successfully exported to {}".format(result))
        else:
            QMessageBox.warning(self, "Warning", "Error exporting data.")