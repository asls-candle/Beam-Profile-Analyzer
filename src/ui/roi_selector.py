# src/ui/roi_selector.py

from matplotlib.patches import Rectangle


class RoiSelector:
    """
    Attaches to a matplotlib Axes and lets the user draw a rectangular ROI
    by left-click-drag on the heatmap.

    Interaction
    -----------
    - Left-click + drag  : draw new ROI
    - Double-click       : reset ROI
    - Single click (tiny): treated as reset

    Callbacks
    ---------
    on_roi_selected(x0, y0, x1, y1)
        Called on mouse-release with the bounding box in axes data units (mm).
        x0 < x1, y0 < y1 are guaranteed.

    on_roi_reset()
        Called on double-click or when the drawn rectangle is below MIN_SIZE.
    """

    MIN_SIZE = 1e-4  # minimum selection size in data units

    def __init__(self, ax, on_roi_selected, on_roi_reset=None):
        self.ax = ax
        self.on_roi_selected = on_roi_selected
        self.on_roi_reset = on_roi_reset

        self._press_xy = None       # (x, y) at mouse-down
        self._rubber_patch = None   # Rectangle shown while dragging

        canvas = ax.figure.canvas
        self._cids = [
            canvas.mpl_connect('button_press_event',   self._on_press),
            canvas.mpl_connect('motion_notify_event',  self._on_motion),
            canvas.mpl_connect('button_release_event', self._on_release),
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def disconnect(self):
        """Detach all event listeners (call before destroying the canvas)."""
        try:
            canvas = self.ax.figure.canvas
            for cid in self._cids:
                canvas.mpl_disconnect(cid)
        except Exception:
            pass
        self._cids = []

    def clear_rubber_patch(self):
        """Remove the rubber-band rectangle (drag preview) from the axes."""
        if self._rubber_patch is not None:
            try:
                self._rubber_patch.remove()
            except ValueError:
                pass
            self._rubber_patch = None
            try:
                self.ax.figure.canvas.draw_idle()
            except Exception:
                pass

    # ── private handlers ──────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return

        if event.dblclick:
            self.clear_rubber_patch()
            if self.on_roi_reset:
                self.on_roi_reset()
            return

        self._press_xy = (event.xdata, event.ydata)
        self.clear_rubber_patch()

    def _on_motion(self, event):
        if self._press_xy is None:
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata

        if self._rubber_patch is not None:
            try:
                self._rubber_patch.remove()
            except ValueError:
                pass

        self._rubber_patch = Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(x1 - x0), abs(y1 - y0),
            linewidth=1.5,
            edgecolor='white',
            facecolor=(1.0, 1.0, 1.0, 0.07),
            linestyle='--',
            zorder=10,
        )
        self.ax.add_patch(self._rubber_patch)
        self.ax.figure.canvas.draw_idle()

    def _on_release(self, event):
        if self._press_xy is None or event.button != 1:
            return

        x0, y0 = self._press_xy
        self._press_xy = None

        if event.inaxes is not self.ax or event.xdata is None:
            self.clear_rubber_patch()
            return

        x1, y1 = event.xdata, event.ydata

        # Too small → treat as plain click → reset
        if abs(x1 - x0) < self.MIN_SIZE or abs(y1 - y0) < self.MIN_SIZE:
            self.clear_rubber_patch()
            if self.on_roi_reset:
                self.on_roi_reset()
            return

        # Successful selection — keep the rubber patch visible,
        # it will be replaced by a persistent yellow patch via the callback.
        self.clear_rubber_patch()
        if self.on_roi_selected:
            self.on_roi_selected(
                min(x0, x1), min(y0, y1),
                max(x0, x1), max(y0, y1),
            )