# src/ui/camera_tab.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QRadioButton, QButtonGroup,
                             QSpinBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

import numpy as np

from src.ui.constants import TOP_PANEL_HEIGHT, BUTTON_MIN_WIDTH, BUTTON_MIN_HEIGHT
from src.ui.roi_selector import RoiSelector
from src.ui.camera_controls_widget import CameraControlsWidget
from src.visualizer.beam_profile import BeamProfilePlot, compute_mm_axes


class CameraTab(QWidget):
    """
    Tab for main capture and camera control.

    ROI: left-click+drag on heatmap → sets ROI, recalculates centroid/RMS,
    draws yellow rectangle + dashed gold lines on the projection plots.
    Double-click or "Reset ROI" button → clears ROI.

    Camera Controls: Shutter / Gain / Brightness sliders are kept in sync
    with the twin widget on DifferenceTab through
    ``main_window.sync_camera_controls()``.  They are disabled while
    background collection is running.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.last_data_hash = None

        # Shared beam-profile plot (figure + canvas + ROI overlays)
        self._beam_plot      = BeamProfilePlot()
        self._last_img_shape = None

        # ROI selector (tab-owned; re-created on every full figure rebuild)
        self._roi_selector = None

        # Image geometry — kept here so ROI callbacks can convert coordinates
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

        # ── Operation Mode ──
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

        # ── Camera Control ──
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

        # ── Camera Information ──
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

        # ── Background Collection ──
        background_panel = QGroupBox("Background Collection")
        background_panel.setMinimumWidth(120)
        bg_layout = QVBoxLayout(background_panel)
        bg_layout.setSpacing(5)
        bg_frames_row = QHBoxLayout()
        bg_frames_row.addWidget(QLabel("Frames:"))
        self.bg_frames_spinbox = QSpinBox()
        self.bg_frames_spinbox.setRange(1, 400)
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

        # ── Camera Controls (Shutter / Gain / Brightness) ──
        # This widget is the master; its twin in DifferenceTab is the slave.
        self.camera_controls = CameraControlsWidget("Exposure Controls")
        self.camera_controls.setMinimumWidth(240)

        # ── Beam Information ──
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

        # ── Assemble top panel ──
        top_panel.addWidget(mode_panel,          1)
        top_panel.addWidget(camera_panel,        1)
        top_panel.addWidget(camera_info_panel,   1)
        top_panel.addWidget(background_panel,    1)
        top_panel.addWidget(self.camera_controls, 2)   # wider to fit sliders
        top_panel.addWidget(beam_info_panel,     1)

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
        # Exposure controls
        self.camera_controls.exposure_changed.connect(self.on_exposure_changed)
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
            # Exposure controls: enabled only when connected and not collecting background
            self.camera_controls.set_controls_enabled(connected and not collecting_bg)
        else:
            for w in (self.camera_combo, self.refresh_cameras_btn,
                      self.launch_camera_btn, self.stop_camera_btn,
                      self.bg_frames_spinbox, self.get_background_btn,
                      self.stop_bg_collection_btn):
                w.setEnabled(False)
            self.camera_controls.set_controls_enabled(False)
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

    # ── Exposure controls ─────────────────────────────────────────────────────

    def refresh_exposure_ranges(self):
        """
        Read hardware limits from the connected camera and update both the
        local widget and the twin widget on DifferenceTab.

        Called once after a successful camera connection.
        Skips any feature that the camera does not expose (info[name] is None).
        """
        info = self.main_window.camera_manager.get_exposure_info()
        if info is None:
            return

        shutter_info    = info.get('shutter')
        gain_info       = info.get('gain')
        brightness_info = info.get('brightness')

        self.camera_controls.set_ranges(
            shutter_range=shutter_info['range']       if shutter_info    else None,
            gain_range=gain_info['range']             if gain_info       else None,
            brightness_range=brightness_info['range'] if brightness_info else None,
        )
        self.camera_controls.set_values(
            shutter=shutter_info['value']       if shutter_info    else None,
            gain=gain_info['value']             if gain_info       else None,
            brightness=brightness_info['value'] if brightness_info else None,
        )
        # Propagate to DifferenceTab twin (without triggering camera write)
        self._sync_twin_widget(
            shutter_info['value']       if shutter_info    else None,
            gain_info['value']          if gain_info       else None,
            brightness_info['value']    if brightness_info else None,
        )

    def _sync_twin_widget(self, shutter, gain, brightness):
        """
        Push the current exposure values to the DifferenceTab twin widget
        (silent update — does not loop back to the camera).
        """
        try:
            diff_tab = self.main_window.difference_tab
            diff_tab.camera_controls.set_ranges(
                shutter_range=self.camera_controls._shutter_slider.minimum(),
                gain_range=self.camera_controls._gain_slider.minimum(),
                brightness_range=self.camera_controls._brightness_slider.minimum(),
            )
        except Exception:
            pass
        try:
            diff_tab = self.main_window.difference_tab
            # Also sync ranges
            diff_tab.camera_controls.set_ranges(
                shutter_range=(
                    self.camera_controls._shutter_slider.minimum(),
                    self.camera_controls._shutter_slider.maximum()),
                gain_range=(
                    self.camera_controls._gain_slider.minimum(),
                    self.camera_controls._gain_slider.maximum()),
                brightness_range=(
                    self.camera_controls._brightness_slider.minimum(),
                    self.camera_controls._brightness_slider.maximum()),
            )
            diff_tab.camera_controls.set_values(
                shutter=shutter,
                gain=gain,
                brightness=brightness,
            )
        except AttributeError:
            # DifferenceTab not yet created — harmless, it will read values
            # from the camera when it first needs them.
            pass

    def on_exposure_changed(self, shutter, gain, brightness):
        """
        User moved a slider -> write to camera, propagate to twin widget.
        """
        import logging
        _log = logging.getLogger("camera")
        _log.info("[DIAG] Slider moved (CameraTab) -> shutter=%s  gain=%s  brightness=%s",
                  shutter, gain, brightness)
        self.main_window.camera_manager.set_exposure(
            shutter=shutter,
            gain=gain,
            brightness=brightness,
        )
        self._sync_twin_widget(shutter, gain, brightness)

    def sync_controls_from_twin(self, shutter, gain, brightness):
        """
        Called by DifferenceTab when the user changes values there so this
        widget mirrors them (without re-triggering the camera write).
        """
        self.camera_controls.set_values(
            shutter=shutter,
            gain=gain,
            brightness=brightness,
        )

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
        self.last_data_hash = None
        if self._beam_plot.canvas:
            self._beam_plot.canvas.draw_idle()

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

    def _update_roi_status(self, x0, y0, x1, y1):
        self.roi_status_label.setText(
            "ROI: [{:.2f},{:.2f}]–[{:.2f},{:.2f}] mm".format(x0, y0, x1, y1))
        self.roi_status_label.setStyleSheet("color: #e6a817;")

    def _restore_roi_from_shared_state(self):
        if self._beam_plot.canvas is None:
            return
        roi = self.main_window.current_roi_mm
        if roi is not None:
            self._beam_plot.draw_roi_patch(*roi)
            self._beam_plot.show_roi_lines(*roi)
            self._update_roi_status(*roi)
            self.reset_roi_btn.setEnabled(True)
        else:
            self._beam_plot.remove_roi_patch()
            self._beam_plot.hide_roi_lines()
            self.roi_status_label.setText("ROI: None")
            self.roi_status_label.setStyleSheet("color: gray;")
            self.reset_roi_btn.setEnabled(False)

    # ── Plot helpers ──────────────────────────────────────────────────────────

    def _build_axes(self, img_data, x_mm, y_mm):
        """Full figure rebuild — called on first render or when shape changes."""
        if self._roi_selector is not None:
            self._roi_selector.disconnect()
            self._roi_selector = None

        if self._beam_plot.canvas is not None:
            self.plot_layout.removeWidget(self._beam_plot.canvas)

        canvas = self._beam_plot.build(img_data, x_mm, y_mm)
        self.plot_layout.addWidget(canvas)

        self._roi_selector = RoiSelector(
            self._beam_plot.ax_heatmap, self._on_roi_selected, self._on_roi_reset)
        self._restore_roi_from_shared_state()

    def _update_artists(self, img_data, x_mm, y_mm):
        """Incremental update — same shape, just new pixel values."""
        self._beam_plot.update(img_data, x_mm, y_mm)

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
        if self.last_data_hash == current_hash and self._beam_plot.canvas is not None:
            return
        self.last_data_hash = current_hash

        pixel_size_x = camera_info.get("pixel_size_x", 1)
        pixel_size_y = camera_info.get("pixel_size_y", 1)
        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y

        img_data = np.asarray(data["current_frame"].data)
        self._img_height, self._img_width = img_data.shape
        x_mm, y_mm = compute_mm_axes(img_data, pixel_size_x, pixel_size_y)

        if self._beam_plot.canvas is None or self._last_img_shape != img_data.shape:
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
        self._restore_roi_from_shared_state()

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
            return
        # Camera connected successfully — load real hardware ranges & values
        self.refresh_exposure_ranges()

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