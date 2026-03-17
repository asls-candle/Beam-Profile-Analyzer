# src/ui/background_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import numpy as np

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT
from src.ui.roi_selector import RoiSelector
from src.visualizer.beam_profile import BeamProfilePlot, compute_mm_axes


class BackgroundTab(QWidget):
    """
    Tab for displaying the background image.

    ROI: left-click+drag on heatmap → sets ROI, recalculates centroid/RMS,
    draws yellow rectangle + dashed gold lines on the projection plots.
    Double-click or "Reset ROI" button → clears ROI.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window    = main_window
        self.last_data_hash = None

        # Shared beam-profile plot (figure + canvas + ROI overlays)
        self._beam_plot = BeamProfilePlot()

        # ROI selector (tab-owned; re-created on every full figure rebuild)
        self._roi_selector = None

        # Image geometry — kept here so ROI callbacks can convert coordinates
        self._img_width    = 0
        self._img_height   = 0
        self._pixel_size_x = 1.0
        self._pixel_size_y = 1.0

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_tab)
        self.update_timer.start(1000)

        self.init_ui()

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

        beam_info_panel = QGroupBox("Background Information")
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
        for _ in range(5):
            ep = QGroupBox()
            ep.setMinimumWidth(120)
            ep.setStyleSheet("border: none; background-color: transparent;")
            QVBoxLayout(ep)
            top_panel.addWidget(ep)
        top_panel.addWidget(beam_info_panel)

        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(400, 400)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(top_panel_widget)
        main_layout.addWidget(self.plot_widget, 1)
        self.setLayout(main_layout)

    # ── Camera info ───────────────────────────────────────────────────────────

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
        self._beam_plot.draw_roi_patch(x0, y0, x1, y1)
        self._beam_plot.show_roi_lines(x0, y0, x1, y1)
        self._update_roi_status(x0, y0, x1, y1)
        self.reset_roi_btn.setEnabled(True)
        self.last_data_hash = None          # force recalculation
        if self._beam_plot.canvas:
            self._beam_plot.canvas.draw_idle()
        # Immediately refresh centroid/RMS labels
        self._refresh_beam_info()

    def _on_roi_reset(self):
        self.main_window.reset_roi()
        self._beam_plot.remove_roi_patch()
        self._beam_plot.hide_roi_lines()
        self.roi_status_label.setText("ROI: None")
        self.roi_status_label.setStyleSheet("color: gray;")
        self.reset_roi_btn.setEnabled(False)
        self.last_data_hash = None
        if self._beam_plot.canvas:
            self._beam_plot.canvas.draw_idle()
        self._refresh_beam_info()

    def _update_roi_status(self, x0, y0, x1, y1):
        self.roi_status_label.setText(
            "ROI: [{:.2f},{:.2f}]–[{:.2f},{:.2f}] mm".format(x0, y0, x1, y1))
        self.roi_status_label.setStyleSheet("color: #e6a817;")

    def _refresh_beam_info(self):
        """Recalculate and display centroid/RMS with current ROI setting."""
        data        = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()
        if data["background"] is None or camera_info is None:
            return
        px = camera_info.get("pixel_size_x", 1)
        py = camera_info.get("pixel_size_y", 1)
        cx, cy = self.main_window.image_analyzer.calculate_centroid(
            data["background"], px, py)
        rx, ry = self.main_window.image_analyzer.calculate_rms(
            data["background"], px, py)
        roi_sfx = " (ROI)" if self.main_window.image_analyzer.roi is not None else ""
        self.centroid_x_label.setText(
            "Centroid X{}: {:.6f} mm".format(roi_sfx, cx))
        self.centroid_y_label.setText(
            "Centroid Y{}: {:.6f} mm".format(roi_sfx, cy))
        self.rms_x_label.setText("RMS X{}: {:.6f} mm".format(roi_sfx, rx))
        self.rms_y_label.setText("RMS Y{}: {:.6f} mm".format(roi_sfx, ry))

    # ── Plot update ───────────────────────────────────────────────────────────

    def update_plots(self):
        data        = self.main_window.get_current_data()
        camera_info = self.main_window.get_camera_info()
        if data["background"] is None or camera_info is None:
            return

        current_hash = hash(str(data["background"].data.tobytes()))
        if self.last_data_hash == current_hash and self._beam_plot.canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y

        img_data = data["background"].data
        self._img_height, self._img_width = img_data.shape
        x_mm, y_mm = compute_mm_axes(img_data, pixel_size_x, pixel_size_y)

        # ── Full rebuild (BackgroundTab always rebuilds on new data) ──────────
        if self._roi_selector is not None:
            self._roi_selector.disconnect()
            self._roi_selector = None

        if self._beam_plot.canvas is not None:
            self.plot_layout.removeWidget(self._beam_plot.canvas)

        canvas = self._beam_plot.build(img_data, x_mm, y_mm)
        self.plot_layout.addWidget(canvas)

        self._roi_selector = RoiSelector(
            self._beam_plot.ax_heatmap, self._on_roi_selected, self._on_roi_reset)

        # Restore ROI patch + lines from shared state
        roi_mm = self.main_window.current_roi_mm
        if roi_mm is not None:
            self._beam_plot.draw_roi_patch(*roi_mm)
            self._beam_plot.show_roi_lines(*roi_mm)
            self._update_roi_status(*roi_mm)
            self.reset_roi_btn.setEnabled(True)

        # ── Centroid / RMS (respects active ROI) ─────────────────────────────
        if (self.main_window.current_mode == "file"
                and "background_centroid" in self.main_window.file_data
                and self.main_window.image_analyzer.roi is None):
            cx, cy = self.main_window.file_data["background_centroid"]
            rx, ry = self.main_window.file_data["background_rms"]
        else:
            cx, cy = self.main_window.image_analyzer.calculate_centroid(
                data["background"], pixel_size_x, pixel_size_y)
            rx, ry = self.main_window.image_analyzer.calculate_rms(
                data["background"], pixel_size_x, pixel_size_y)

        roi_sfx = " (ROI)" if self.main_window.image_analyzer.roi is not None else ""
        self.centroid_x_label.setText(
            "Centroid X{}: {:.6f} mm".format(roi_sfx, cx))
        self.centroid_y_label.setText(
            "Centroid Y{}: {:.6f} mm".format(roi_sfx, cy))
        self.rms_x_label.setText("RMS X{}: {:.6f} mm".format(roi_sfx, rx))
        self.rms_y_label.setText("RMS Y{}: {:.6f} mm".format(roi_sfx, ry))

    # ── Tab update ────────────────────────────────────────────────────────────

    def update_tab(self):
        if self.main_window.current_mode == "camera":
            has_bg = self.main_window.image_reader.background is not None
        else:
            has_bg = self.main_window.file_data["background"] is not None

        self.update_camera_info()
        if has_bg:
            self.update_plots()