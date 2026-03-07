# src/ui/background_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT
from src.ui.roi_selector import RoiSelector

_ROI_LINE_KW = dict(color='red', linewidth=1.2, linestyle='--', alpha=0.9, zorder=6)


class BackgroundTab(QWidget):
    """
    Tab for displaying the background image.

    ROI: left-click+drag on heatmap → sets ROI, recalculates centroid/RMS,
    draws yellow rectangle + dashed gold lines on the projection plots.
    Double-click or "Reset ROI" button → clears ROI.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window   = main_window
        self.last_data_hash = None

        # ROI state
        self._roi_selector = None
        self._roi_patch    = None
        self._ax_heatmap   = None
        self._ax_x_proj    = None
        self._ax_y_proj    = None
        self._img_width    = 0
        self._img_height   = 0
        self._pixel_size_x = 1.0
        self._pixel_size_y = 1.0

        # ROI boundary lines on projections
        self._roi_vline_x0 = None
        self._roi_vline_x1 = None
        self._roi_hline_y0 = None
        self._roi_hline_y1 = None

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
        self.plot_canvas = None

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
        self._draw_roi_patch(x0, y0, x1, y1)
        self._show_roi_lines(x0, y0, x1, y1)
        self._update_roi_status(x0, y0, x1, y1)
        self.reset_roi_btn.setEnabled(True)
        self.last_data_hash = None          # force recalculation
        if self.plot_canvas:
            self.plot_canvas.draw_idle()
        # Immediately refresh centroid/RMS labels
        self._refresh_beam_info()

    def _on_roi_reset(self):
        self.main_window.reset_roi()
        self._remove_roi_patch()
        self._hide_roi_lines()
        self.roi_status_label.setText("ROI: None")
        self.roi_status_label.setStyleSheet("color: gray;")
        self.reset_roi_btn.setEnabled(False)
        self.last_data_hash = None
        if self.plot_canvas:
            self.plot_canvas.draw_idle()
        self._refresh_beam_info()

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
        # Rebuild figure only when image data actually changes
        if self.last_data_hash == current_hash and self.plot_canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y

        # ── Disconnect old ROI selector ───────────────────────────────────────
        if self._roi_selector is not None:
            self._roi_selector.disconnect()
            self._roi_selector = None
        self._roi_patch    = None
        self._ax_heatmap   = None
        self._ax_x_proj    = None
        self._ax_y_proj    = None
        self._roi_vline_x0 = self._roi_vline_x1 = None
        self._roi_hline_y0 = self._roi_hline_y1 = None

        # ── Build figure ──────────────────────────────────────────────────────
        fig = Figure(figsize=(10, 8))
        gs  = GridSpec(2, 3, width_ratios=[0.5, 5, 0.3],
                       height_ratios=[5, 0.8], figure=fig)

        ax_y_proj  = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[0, 1])
        ax_x_proj  = fig.add_subplot(gs[1, 1])
        cax        = fig.add_subplot(gs[0, 2])

        self._ax_heatmap = ax_heatmap
        self._ax_x_proj  = ax_x_proj
        self._ax_y_proj  = ax_y_proj

        img_data = data["background"].data
        h, w     = img_data.shape
        self._img_width  = w
        self._img_height = h

        x_mm = (np.arange(w) - w / 2) * pixel_size_x
        y_mm = (np.arange(h) - h / 2) * pixel_size_y

        im = ax_heatmap.imshow(
            img_data,
            extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
            origin='lower', aspect='auto', cmap='jet')
        ax_heatmap.set_xlabel('X (mm)')
        ax_heatmap.set_ylabel('Y (mm)')
        ax_heatmap.set_title(
            'Beam Profile — drag to select ROI, dbl-click to reset')
        fig.colorbar(im, cax=cax, label='Intensity')

        x_proj = np.sum(img_data, axis=0)
        y_proj = np.sum(img_data, axis=1)

        ax_x_proj.plot(x_mm, x_proj, 'b-', linewidth=1)
        ax_x_proj.set_xlabel('X (mm)')
        ax_x_proj.set_ylabel('Intensity')
        ax_x_proj.grid(True, linestyle='--', alpha=0.7)

        ax_y_proj.plot(y_proj, y_mm, 'b-', linewidth=1)
        ax_y_proj.set_ylabel('Y (mm)')
        ax_y_proj.set_xlabel('Intensity')
        ax_y_proj.grid(True, linestyle='--', alpha=0.7)

        ax_y_proj.set_ylim(ax_heatmap.get_ylim())
        ax_x_proj.set_xlim(ax_heatmap.get_xlim())

        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                            wspace=0.3, hspace=0.3)

        # Replace canvas
        if self.plot_canvas is not None:
            self.plot_layout.removeWidget(self.plot_canvas)
            self.plot_canvas.figure.clear()
            self.plot_canvas.close()

        self.plot_canvas = FigureCanvasQTAgg(fig)
        self.plot_layout.addWidget(self.plot_canvas)

        # Attach ROI selector
        self._roi_selector = RoiSelector(
            ax_heatmap, self._on_roi_selected, self._on_roi_reset)

        # Restore ROI patch + lines from shared state
        roi_mm = self.main_window.current_roi_mm
        if roi_mm is not None:
            self._draw_roi_patch(*roi_mm)
            self._show_roi_lines(*roi_mm)
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