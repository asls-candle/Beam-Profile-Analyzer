# src/visualizer/beam_profile.py
"""
Shared beam-profile plotting utilities.

Provides
--------
compute_mm_axes(img_data, pixel_size_x, pixel_size_y) -> (x_mm, y_mm)
    Build centred coordinate arrays from image dimensions and pixel pitch.

compute_projections(img_data) -> (x_proj, y_proj)
    Column-wise and row-wise intensity sums.

BeamProfilePlot
    Reusable figure/canvas that renders:
      • heatmap with colour-bar
      • X-projection (below)
      • Y-projection (left)
      • Optional ROI rectangle on the heatmap
      • Optional ROI boundary lines on both projection axes

Used identically by CameraTab, BackgroundTab, and DifferenceTab so that
all plotting logic lives in exactly one place.
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# ── Module-level constants ────────────────────────────────────────────────────

# Style applied to the dashed ROI boundary lines on projection axes
ROI_LINE_KW = dict(color='red', linewidth=1.2, linestyle='--', alpha=0.9, zorder=6)

_DEFAULT_TITLE = 'Beam Profile — drag to select ROI, dbl-click to reset'


# ── Pure functions ────────────────────────────────────────────────────────────

def compute_mm_axes(img_data, pixel_size_x, pixel_size_y):
    """
    Return (x_mm, y_mm) coordinate arrays centred on the image.

    Parameters
    ----------
    img_data : np.ndarray  shape (H, W)
    pixel_size_x, pixel_size_y : float  — physical pixel pitch in mm

    Returns
    -------
    x_mm : np.ndarray  shape (W,)
    y_mm : np.ndarray  shape (H,)
    """
    h, w = img_data.shape
    x_mm = (np.arange(w) - w / 2) * pixel_size_x
    y_mm = (np.arange(h) - h / 2) * pixel_size_y
    return x_mm, y_mm


def compute_projections(img_data):
    """
    Return (x_proj, y_proj) intensity projection arrays.

    x_proj : column-wise sum, shape (W,)
    y_proj : row-wise sum,    shape (H,)
    """
    return np.sum(img_data, axis=0), np.sum(img_data, axis=1)


# ── BeamProfilePlot ───────────────────────────────────────────────────────────

class BeamProfilePlot:
    """
    Reusable matplotlib figure: heatmap + X/Y intensity projections.

    Lifecycle
    ---------
    1. Instantiate once per tab::

           self._beam_plot = BeamProfilePlot()

    2. Call ``build(img_data, x_mm, y_mm)`` whenever the image shape changes
       or on the very first render.  The *caller* must remove the previous
       canvas from its Qt layout before calling ``build()``:

           if self._beam_plot.canvas is not None:
               layout.removeWidget(self._beam_plot.canvas)
           canvas = self._beam_plot.build(img_data, x_mm, y_mm)
           layout.addWidget(canvas)

    3. Call ``update(img_data, x_mm, y_mm)`` for subsequent frames with the
       **same** image shape — updates artists in-place without rebuilding the
       Figure.

    4. ROI visualisation::

           self._beam_plot.draw_roi_patch(x0, y0, x1, y1)
           self._beam_plot.show_roi_lines(x0, y0, x1, y1)
           self._beam_plot.hide_roi_lines()
           self._beam_plot.remove_roi_patch()

    Public attributes
    -----------------
    canvas : FigureCanvasQTAgg | None
        The Qt widget embedding the current figure.
    ax_heatmap, ax_x_proj, ax_y_proj : Axes | None
        Expose the axes so callers can attach RoiSelector or query limits.
    """

    def __init__(self):
        self.canvas        = None   # FigureCanvasQTAgg
        self._fig          = None
        self._im           = None
        self._line_x       = None
        self._line_y       = None
        self._ax_heatmap   = None
        self._ax_x_proj    = None
        self._ax_y_proj    = None
        self._roi_patch    = None
        self._roi_vline_x0 = None
        self._roi_vline_x1 = None
        self._roi_hline_y0 = None
        self._roi_hline_y1 = None

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def ax_heatmap(self):
        return self._ax_heatmap

    @property
    def ax_x_proj(self):
        return self._ax_x_proj

    @property
    def ax_y_proj(self):
        return self._ax_y_proj

    # ── main API ──────────────────────────────────────────────────────────────

    def build(self, img_data, x_mm, y_mm, title=_DEFAULT_TITLE):
        """
        Create a brand-new Figure/Canvas for *img_data*.

        The caller is responsible for removing the previous canvas from the
        Qt layout **before** calling this method.

        Returns
        -------
        FigureCanvasQTAgg
            The newly created canvas ready to be added to the layout.
        """
        self._reset_roi_artists()

        if self.canvas is not None:
            self.canvas.figure.clear()
            self.canvas.close()
            self.canvas = None

        fig = Figure(figsize=(10, 8))
        gs  = GridSpec(2, 3,
                       width_ratios=[0.5, 5, 0.3],
                       height_ratios=[5, 0.8],
                       figure=fig)

        self._ax_y_proj  = fig.add_subplot(gs[0, 0])
        self._ax_heatmap = fig.add_subplot(gs[0, 1])
        self._ax_x_proj  = fig.add_subplot(gs[1, 1])
        cax              = fig.add_subplot(gs[0, 2])
        self._fig        = fig

        x_proj, y_proj = compute_projections(img_data)

        # ── Heatmap ──
        self._im = self._ax_heatmap.imshow(
            img_data,
            extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
            origin='lower', aspect='auto', cmap='jet')
        self._im.set_clim(vmin=0, vmax=1)
        self._ax_heatmap.set_xlabel('X (mm)')
        self._ax_heatmap.set_ylabel('Y (mm)')
        self._ax_heatmap.set_title(title)
        fig.colorbar(self._im, cax=cax, label='Intensity')

        # ── X-projection (below heatmap) ──
        self._line_x, = self._ax_x_proj.plot(x_mm, x_proj, 'b-', linewidth=1)
        self._ax_x_proj.set_xlabel('X (mm)')
        self._ax_x_proj.set_ylabel('Intensity')
        self._ax_x_proj.grid(True, linestyle='--', alpha=0.7)

        # ── Y-projection (left of heatmap) ──
        self._line_y, = self._ax_y_proj.plot(y_proj, y_mm, 'b-', linewidth=1)
        self._ax_y_proj.set_ylabel('Y (mm)')
        self._ax_y_proj.set_xlabel('Intensity')
        self._ax_y_proj.grid(True, linestyle='--', alpha=0.7)

        # Align projection axes with heatmap extents
        self._ax_y_proj.set_ylim(self._ax_heatmap.get_ylim())
        self._ax_x_proj.set_xlim(self._ax_heatmap.get_xlim())

        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                            wspace=0.3, hspace=0.3)

        self.canvas = FigureCanvasQTAgg(fig)
        return self.canvas

    def update(self, img_data, x_mm, y_mm):
        """
        Update existing artists in-place (no Figure rebuild).

        Only valid after ``build()`` has been called with data of the same
        shape.  Calls ``canvas.draw_idle()`` internally.
        """
        x_proj, y_proj = compute_projections(img_data)

        self._im.set_data(img_data)
        self._im.set_extent([x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]])
        self._im.set_clim(vmin=0, vmax=1)

        self._line_x.set_data(x_mm, x_proj)
        self._ax_x_proj.set_xlim(x_mm[0], x_mm[-1])
        self._ax_x_proj.set_ylim(
            0, x_proj.max() * 1.05 if x_proj.max() > 0 else 1)

        self._line_y.set_data(y_proj, y_mm)
        self._ax_y_proj.set_ylim(y_mm[0], y_mm[-1])
        self._ax_y_proj.set_xlim(
            0, y_proj.max() * 1.05 if y_proj.max() > 0 else 1)

        self.canvas.draw_idle()

    # ── ROI visualisation ─────────────────────────────────────────────────────

    def draw_roi_patch(self, x0, y0, x1, y1):
        """Overlay a semi-transparent yellow ROI rectangle on the heatmap."""
        if self._ax_heatmap is None:
            return
        self.remove_roi_patch()
        self._roi_patch = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.0, edgecolor='yellow',
            facecolor=(1.0, 1.0, 0.0, 0.06),
            linestyle='-', zorder=8)
        self._ax_heatmap.add_patch(self._roi_patch)

    def remove_roi_patch(self):
        """Remove the ROI rectangle from the heatmap (no-op if absent)."""
        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except ValueError:
                pass
            self._roi_patch = None

    def show_roi_lines(self, x0, y0, x1, y1):
        """Draw or update dashed boundary lines on the projection axes."""
        if self._ax_x_proj is None or self._ax_y_proj is None:
            return
        if self._roi_vline_x0 is None:
            self._roi_vline_x0 = self._ax_x_proj.axvline(x0, **ROI_LINE_KW)
            self._roi_vline_x1 = self._ax_x_proj.axvline(x1, **ROI_LINE_KW)
        else:
            self._roi_vline_x0.set_xdata([x0, x0])
            self._roi_vline_x0.set_visible(True)
            self._roi_vline_x1.set_xdata([x1, x1])
            self._roi_vline_x1.set_visible(True)
        if self._roi_hline_y0 is None:
            self._roi_hline_y0 = self._ax_y_proj.axhline(y0, **ROI_LINE_KW)
            self._roi_hline_y1 = self._ax_y_proj.axhline(y1, **ROI_LINE_KW)
        else:
            self._roi_hline_y0.set_ydata([y0, y0])
            self._roi_hline_y0.set_visible(True)
            self._roi_hline_y1.set_ydata([y1, y1])
            self._roi_hline_y1.set_visible(True)

    def hide_roi_lines(self):
        """Hide (but do not destroy) the projection boundary lines."""
        for line in (self._roi_vline_x0, self._roi_vline_x1,
                     self._roi_hline_y0, self._roi_hline_y1):
            if line is not None:
                line.set_visible(False)

    # ── internal ──────────────────────────────────────────────────────────────

    def _reset_roi_artists(self):
        """Nullify all ROI artist references (called before a full rebuild)."""
        self._roi_patch    = None
        self._roi_vline_x0 = self._roi_vline_x1 = None
        self._roi_hline_y0 = self._roi_hline_y1 = None