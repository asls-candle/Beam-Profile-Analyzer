# src/ui/camera_controls_widget.py

from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout,
                              QLabel, QSlider, QSpinBox, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal


class CameraControlsWidget(QGroupBox):
    """
    Shared widget for adjusting camera exposure parameters:
    Shutter, Gain, Brightness.

    Design rules
    ───────────
    • Emits ``exposure_changed(shutter, gain, brightness)`` only when the
      *user* moves a control (slider or spinbox).
    • Call ``set_values()`` to update controls programmatically — does NOT
      emit the signal, preventing feedback loops when syncing the twin widget
      in the other tab.
    • Call ``set_ranges()`` once after the camera connects to load the true
      hardware limits reported by pydc1394.
    • Call ``set_controls_enabled(False)`` while background is being
      collected so the user cannot disturb the camera during that period.

    Shutter minimum value
    ──────────────────────
    The Flea2 reports ``value_range = (0, 863)`` for shutter.
    Raw value 0 maps to ≈ 4 µs absolute exposure and is perfectly valid —
    the camera will produce a nearly black image, but no hardware error
    occurs.  The slider therefore starts at 0.
    """

    exposure_changed = pyqtSignal(int, int, int)   # shutter, gain, brightness

    # Fallback ranges used before a camera connects.
    _DEFAULTS = {
        'shutter':    {'range': (0, 863),  'value': 472},
        'gain':       {'range': (0, 683),  'value': 0},
        'brightness': {'range': (0, 255),  'value': 0},
    }

    def __init__(self, title="Camera Controls", parent=None):
        super().__init__(title, parent)
        self._updating = False     # re-entrancy guard
        self._init_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(7, 8, 7, 8)

        d = self._DEFAULTS
        (self._shutter_slider,
         self._shutter_spin) = self._add_row(
            layout, "Shutter",
            *d['shutter']['range'], d['shutter']['value'])

        (self._gain_slider,
         self._gain_spin) = self._add_row(
            layout, "Gain",
            *d['gain']['range'], d['gain']['value'])

        (self._brightness_slider,
         self._brightness_spin) = self._add_row(
            layout, "Brightness",
            *d['brightness']['range'], d['brightness']['value'])

        self._wire_internal_signals()

    def _add_row(self, parent_layout, label, lo, hi, default):
        """Create a labelled slider + spinbox row and return (slider, spin)."""
        row = QHBoxLayout()

        lbl = QLabel("{}:".format(label))
        lbl.setFixedWidth(70)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(default)
        spin.setFixedWidth(68)

        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return slider, spin

    # ── Internal signal wiring ────────────────────────────────────────────────

    def _wire_internal_signals(self):
        """Keep each slider ↔ spinbox pair in sync and emit on user action."""
        pairs = [
            (self._shutter_slider,    self._shutter_spin),
            (self._gain_slider,       self._gain_spin),
            (self._brightness_slider, self._brightness_spin),
        ]
        for slider, spin in pairs:
            slider.valueChanged.connect(
                lambda v, s=spin: self._on_slider_moved(v, s))
            spin.valueChanged.connect(
                lambda v, sl=slider: self._on_spin_changed(v, sl))

    def _on_slider_moved(self, value, spin):
        if self._updating:
            return
        self._updating = True
        spin.setValue(value)
        self._updating = False
        self._emit_changed()

    def _on_spin_changed(self, value, slider):
        if self._updating:
            return
        self._updating = True
        slider.setValue(value)
        self._updating = False
        self._emit_changed()

    def _emit_changed(self):
        self.exposure_changed.emit(
            self._shutter_slider.value(),
            self._gain_slider.value(),
            self._brightness_slider.value())

    # ── Public API ────────────────────────────────────────────────────────────

    def set_ranges(self, shutter_range=None, gain_range=None,
                   brightness_range=None):
        """
        Update slider/spinbox ranges from camera hardware limits.

        Each argument is a ``(min, max)`` tuple or ``None`` to leave
        the current range unchanged.
        """
        self._updating = True
        try:
            if shutter_range:
                lo, hi = int(shutter_range[0]), int(shutter_range[1])
                self._shutter_slider.setRange(lo, hi)
                self._shutter_spin.setRange(lo, hi)
            if gain_range:
                lo, hi = int(gain_range[0]), int(gain_range[1])
                self._gain_slider.setRange(lo, hi)
                self._gain_spin.setRange(lo, hi)
            if brightness_range:
                lo, hi = int(brightness_range[0]), int(brightness_range[1])
                self._brightness_slider.setRange(lo, hi)
                self._brightness_spin.setRange(lo, hi)
        finally:
            self._updating = False

    def set_values(self, shutter=None, gain=None, brightness=None):
        """
        Update controls programmatically without emitting ``exposure_changed``.
        Use this to synchronise the twin widget in the other tab.
        """
        self._updating = True
        try:
            if shutter is not None:
                self._shutter_slider.setValue(int(shutter))
                self._shutter_spin.setValue(int(shutter))
            if gain is not None:
                self._gain_slider.setValue(int(gain))
                self._gain_spin.setValue(int(gain))
            if brightness is not None:
                self._brightness_slider.setValue(int(brightness))
                self._brightness_spin.setValue(int(brightness))
        finally:
            self._updating = False

    def get_values(self):
        """Return current ``(shutter, gain, brightness)`` as a tuple of ints."""
        return (self._shutter_slider.value(),
                self._gain_slider.value(),
                self._brightness_slider.value())

    def set_controls_enabled(self, enabled: bool):
        """Enable or disable all sliders and spinboxes at once."""
        for widget in (
            self._shutter_slider,    self._shutter_spin,
            self._gain_slider,       self._gain_spin,
            self._brightness_slider, self._brightness_spin,
        ):
            widget.setEnabled(enabled)