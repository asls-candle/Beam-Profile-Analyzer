# Data Flow and Processing Algorithms

Laser beam profile analysis system.

---

## Contents

1. [Overall Block Diagram](#1-overall-block-diagram)
2. [Camera Frame Capture](#2-camera-frame-capture)
3. [Normalization and Background Subtraction](#3-normalization-and-background-subtraction)
4. [Region of Interest (ROI)](#4-region-of-interest-roi)
5. [Beam Profile Analysis](#5-beam-profile-analysis)
6. [Data Export](#6-data-export)
7. [Data Type Summary](#7-data-type-summary)

---

## 1. Overall Block Diagram

The diagram shows which data is used **only for display** in the UI, and which data **goes to files** (CSV and PNG).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             CAMERA (IEEE 1394 / FireWire)                       │
│                  FORMAT7_0 · MONO16 · big-endian · dtype: uint16                │
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │  raw bytes (DMA buffer)
                                  ▼
                    ┌──────────────────────────┐
                    │   _decode_frame()        │
                    │   uint16 → ndarray(H,W)  │
                    └────────────┬─────────────┘
                                 │  ndarray (H,W) uint16
          ┌──────────────────────┴──────────────────────┐
          │ Single frame                                 │ Capture N background frames
          ▼                                              ▼
┌─────────────────────────┐                  ┌──────────────────────────────┐
│ normalize(frame)        │                 │ capture_background_frames(N)  │
│ uint16 → float64 [0,1]  │                  │ each frame → float64         │
│ by its own min/max      │                  │ stack shape (N,H,W) float64  │
└──────────┬──────────────┘                  └──────────────┬───────────────┘
           │ current_frame                                  │ background_frames
           │ float64 [0,1]                                  ▼
           │                                  ┌──────────────────────────────┐
           │                                  │ process_background()         │
           │                                  │ mean over axis N             │
           │                                  │ float64, values in uint16    │
           │                                  └──────────────┬───────────────┘
           │                                                 │ raw_background float64
           │                                  ┌──────────────┴──────────────┐
           │                                  │ normalize(bg, ref=raw_shot) │
           │                                  │ scaled by shot's min/max    │
           │                                  │ float64 ≈ [0, ≤1.0]         │
           │                                  └──────────────┬──────────────┘
           │                                                 │ background float64
           │◄────────────────────────────────────────────────┤
           │                                                 │
           ▼                                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  _calculate_difference()                                             │
│  raw_shot.astype(float64) − raw_background.astype(float64)           │
│  clip negatives to 0                                                 │
│  normalize(raw_diff)  [by raw_diff's own min/max]                    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ difference  float64 [0,1]
            ┌──────────────────────┼──────────────────────────────────┐
            │                      │                                  │
            ▼                      ▼                                  ▼
┌────────────────────┐  ┌──────────────────────────────┐   ┌─────────────────────┐
│  DISPLAY ONLY      │  │  ANALYSIS (centroid, RMS)     │  │  EXPORT             │
│                    │  │                               │  │                     │
│ heatmap + profiles │  │  [opt.] apply_roi(difference) │  │  [opt.] crop to ROI │
│ in UI              │  │  pixel-space slice            │  │  pixel-space slice  │
│                    │  │  ↓                            │  │  ↓                  │
│                    │  │  calculate_centroid()         │  │  export_csv()       │
│                    │  │  calculate_rms()              │  │  → .csv + .json     │
│                    │  │  result in mm                 │  │                     │
│                    │  │  ↓                            │  │  export_png()       │
│                    │  │  label in UI                  │  │  → .png (jet, [0,1])│
└────────────────────┘  └──────────────────────────────┘   └─────────────────────┘
     ↑ screen only            ↑ screen + export metadata        ↑ files on disk
```

---

## 2. Camera Frame Capture

**Module:** `camera_manager.py` · **Class:** `CameraManager`

### Hardware Interface

| Parameter | Value |
|---|---|
| Protocol | IEEE 1394 (FireWire), library `pydc1394` |
| Pixel format | MONO16 (Y16) — 16-bit monochrome |
| Byte order | big-endian (`>u2`) — fixed for the cameras used |
| Capture mode | FORMAT7_0, maximum available resolution |

### Buffer Decoding

`_decode_frame(raw_bytes)` receives the DMA byte buffer and returns a 2D array:

```
arr = np.frombuffer(raw_bytes, dtype='>u2').reshape((H, W)).copy()
```

**Result:** `ndarray shape=(H, W), dtype=uint16, range 0…65535`

### Background Frame Capture

`capture_background_frames(N)` calls `_decode_frame` in a loop and immediately converts to `float64`:

```
frames[i] = arr.astype(np.float64)   # uint16 → float64, values 0…65535
```

**Result:** `ndarray shape=(N, H, W), dtype=float64, values 0…65535`

---

## 3. Normalization and Background Subtraction

**Module:** `image_reader.py` · **Class:** `ImageReader`

### 3.1 Shot Frame Normalization

Called in `_reading_worker` for each captured frame:

```python
current_frame = ImageNormalizer.normalize(raw_frame)
```

Algorithm (`normalize` without `reference_image`):

```
min_val = min(image)
max_val = max(image)
normalized = (image.astype(float64) − min_val) / (max_val − min_val)
```

Normalization is done by the **frame's own** min/max.  
**Result:** `ndarray (H,W) float64, range [0.0, 1.0]`

### 3.2 Background Averaging

```python
raw_bg = np.mean(background_frames, axis=0)   # pixel-wise mean over N frames
raw_bg[raw_bg < 0] = 0
```

**Result:** `ndarray (H,W) float64`, values in original units (`0…65535`)

### 3.3 Background Normalization Relative to Shot

The background is normalized using `raw_current_frame` as the reference, so that the background intensity scale matches the shot scale:

```python
background = ImageNormalizer.normalize(raw_bg, reference_image=raw_current_frame)
```

Algorithm (`normalize` with `reference_image`):

```
min_val = min(raw_current_frame)   # min/max taken from shot, not bg
max_val = max(raw_current_frame)
background = (raw_bg.astype(float64) − min_val) / (max_val − min_val)
```

**Result:** `ndarray (H,W) float64, range ≈ [0.0, ≤ 1.0]`  
Background values may not reach 1.0 since they are scaled by the shot range.

### 3.4 Background Subtraction (`difference`)

Subtraction is performed in **original units** (before normalization):

```python
raw_diff = raw_current_frame.astype(float64) − raw_background.astype(float64)
raw_diff[raw_diff < 0] = 0           # pixels darker than background → 0
difference = ImageNormalizer.normalize(raw_diff)   # by raw_diff's own min/max
```

The `difference` normalization uses **its own** min/max (no reference).  
**Result:** `ndarray (H,W) float64, range [0.0, 1.0]`  
The brightest beam pixel is always 1.0.

---

## 4. Region of Interest (ROI)

### 4.1 How the User Defines the ROI

The user draws a rectangle by dragging the mouse directly on the heatmap. The heatmap is rendered in **millimeters** — the `imshow` axes are configured with `extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]`. Therefore `RoiSelector` returns coordinates in mm:

```
on_roi_selected(x0_mm, y0_mm, x1_mm, y1_mm)
```

### 4.2 Converting ROI from mm to Pixels

`MainWindow.set_roi_mm()` converts data-space (mm) coordinates to pixel array indices. Zero mm corresponds to the frame center:

```python
x_min = max(0,          round( x0_mm / pixel_size_x + W / 2 ))
x_max = min(img_width,  round( x1_mm / pixel_size_x + W / 2 ))
y_min = max(0,          round( y0_mm / pixel_size_y + H / 2 ))
y_max = min(img_height, round( y1_mm / pixel_size_y + H / 2 ))
```

The pixel coordinates are stored in `ImageAnalyzer.roi = (x_min, y_min, x_max, y_max)`.  
The mm coordinates are stored separately in `MainWindow.current_roi_mm` for re-drawing the ROI rectangle when switching tabs.

### 4.3 Applying ROI in Centroid and RMS Calculations

At the start of `calculate_centroid` and `calculate_rms` the image is cropped:

```python
processed = image[y_min:y_max, x_min:x_max]   # shape (H_roi, W_roi)
```

All subsequent calculations (projections, sums, indices) operate on the cropped array. The indices `x` and `y` start from **0 inside the ROI**, not from 0 of the full frame.

The mm conversion is relative to the **ROI center:**

```
centroid_x = (cx_px_roi − W_roi / 2) · pixel_size_x   (mm)
centroid_y = (cy_px_roi − H_roi / 2) · pixel_size_y   (mm)
```

### 4.4 Applying ROI on Export

When the user clicks "Export", `MainWindow.apply_roi_to_data()` crops all image arrays to the ROI **before** writing to disk:

```python
for key in ('shot', 'background', 'difference', 'raw_shot', 'raw_background'):
    result[key] = arr[y_min:y_max, x_min:x_max]
result['resolution'] = (x_max − x_min, y_max − y_min)
```

Only the cropped region is saved to CSV and PNG.

### 4.5 Resetting the ROI

Double-click on the heatmap or the "Reset ROI" button:

```python
ImageAnalyzer.roi = None
MainWindow.current_roi_mm = None
```

---

## 5. Beam Profile Analysis

**Module:** `image_analyzer.py` · **Class:** `ImageAnalyzer`

All calculations use the `difference` image (`ndarray float64, [0,1]`). Pixel values act as **intensity weights**.

### 5.1 1D Intensity Projections (display only)

`calculate_projections` is used exclusively for drawing profiles in the UI. It does not feed into CSV.

```
x_proj[x] = Σ_y  processed[y, x]       (sum over all rows for column x)
y_proj[y] = Σ_x  processed[y, x]       (sum over all columns for row y)

# normalization for display:
x_proj = x_proj / max(x_proj)
y_proj = y_proj / max(y_proj)

# convert pixel indices to mm for plot axes:
x_coords[x] = x · pixel_size_x   (mm)
y_coords[y] = y · pixel_size_y   (mm)
```

### 5.2 Centroid

`calculate_centroid(image, pixel_size_x, pixel_size_y)`

**Step 1** — projections (not normalized):
```
x_proj[x] = Σ_y  processed[y, x]
y_proj[y] = Σ_x  processed[y, x]
```

**Step 2** — intensity-weighted mean position **in pixels:**
```
cx_px = ( Σ_x  x · x_proj[x] ) / ( Σ_x  x_proj[x] )
cy_px = ( Σ_y  y · y_proj[y] ) / ( Σ_y  y_proj[y] )
```
Here `x` and `y` are integer indices from `0` to `W−1` (or `H−1`).

**Step 3** — convert to mm with centering:
```
centroid_x = (cx_px − W / 2) · pixel_size_x   (mm)
centroid_y = (cy_px − H / 2) · pixel_size_y   (mm)
```

Zero coordinate corresponds to the center of the frame (or ROI). The entire calculation runs in pixel indices; conversion to mm happens only on the final result.

### 5.3 RMS Beam Size

`calculate_rms(image, pixel_size_x, pixel_size_y)`

**Step 1** — projections (same as centroid):
```
x_proj[x] = Σ_y  processed[y, x]
y_proj[y] = Σ_x  processed[y, x]
```

**Step 2** — centroid in pixels (recomputed internally):
```
cx_px = ( Σ_x  x · x_proj[x] ) / ( Σ_x  x_proj[x] )
cy_px = ( Σ_y  y · y_proj[y] ) / ( Σ_y  y_proj[y] )
```

**Step 3** — RMS as intensity-weighted standard deviation **in pixels:**
```
rms_x_px = √( Σ_x  x_proj[x] · (x − cx_px)²  /  Σ_x  x_proj[x] )
rms_y_px = √( Σ_y  y_proj[y] · (y − cy_px)²  /  Σ_y  y_proj[y] )
```

**Step 4** — convert to mm:
```
rms_x = rms_x_px · pixel_size_x   (mm)
rms_y = rms_y_px · pixel_size_y   (mm)
```

RMS is not centered — it is a measure of distribution width, already computed relative to the centroid.

### 5.4 Pixel-to-mm Conversion — General Principle

`pixel_size_x` and `pixel_size_y` (mm/pixel) are determined from the `PIXEL_SIZE_BY_RESOLUTION` dictionary in `CameraManager` by the product `W·H` of the selected camera mode.

| Quantity | Conversion formula |
|---|---|
| Heatmap axes (UI) | `x_mm = (arange(W) − W/2) · pixel_size_x` |
| Centroid X | `(cx_px − W/2) · pixel_size_x` |
| Centroid Y | `(cy_px − H/2) · pixel_size_y` |
| RMS X | `rms_x_px · pixel_size_x` |
| RMS Y | `rms_y_px · pixel_size_y` |

The formula `index − size/2` shifts the origin from the top-left pixel corner (0,0) to the geometric center of the frame.

---

## 6. Data Export

**Module:** `exporter.py` · **Class:** `DataExporter`

### 6.1 CSV + JSON (`export_csv`)

Data is split into two types:

**Scalars → `metadata.json`:**
- `centroid_x_mm`, `centroid_y_mm`
- `rms_x_mm`, `rms_y_mm`
- `pixel_size_x`, `pixel_size_y`
- `resolution`, `date`
- `roi_pixels`, `roi_mm` (if ROI is active)

**Arrays → individual `<n>.csv` files:**

| File | Shape | Contents |
|---|---|---|
| `shot.csv` | H × W | normalized captured frame, float64 [0,1] |
| `background.csv` | H × W | normalized background, float64 [0,≤1] |
| `difference.csv` | H × W | difference after subtraction, float64 [0,1] |

Each number is written with 7 decimal places. 2D arrays are written row by row (H rows, W values per row, comma-separated).

`MaskedArray` is converted to a plain `ndarray` via `.filled(0)` before writing.

### 6.2 PNG (`export_png`)

- Three files are saved: `shot.png`, `background.png`, `difference.png`
- Color map: `jet`, uniform scale `vmin=0.0, vmax=1.0` for all three — matches the UI display
- File resolution exactly matches the data resolution: `dpi=100`, `figsize=(W/100, H/100)`, `interpolation='nearest'`
- Orientation: `origin='lower'` — row zero is at the bottom

---

## 7. Data Type Summary

| Variable | Shape | dtype | Range | Notes |
|---|---|---|---|---|
| `raw_current_frame` | (H, W) | uint16 | 0…65535 | raw camera frame |
| `current_frame` | (H, W) | float64 | 0.0…1.0 | normalized by itself |
| `background_frames` | (N, H, W) | float64 | 0…65535 | stack before averaging |
| `raw_background` | (H, W) | float64 | ≥0, ~0…65535 | pixel-wise mean |
| `background` | (H, W) | float64 | ≈0.0…≤1.0 | normalized by ref=shot |
| `raw_diff` | (H, W) | float64 | 0…~65535 | shot−bg, clip(0) |
| `difference` | (H, W) | float64 | 0.0…1.0 | normalized by itself |
| `cx_px`, `cy_px` | scalar | float64 | 0…W (H) | pixel indices |
| `centroid_x/y` | scalar | float64 | ±mm | centered on frame center |
| `rms_x/y` | scalar | float64 | ≥0 mm | distribution width |
| CSV cells | H×W / W | str | 0.0000000…1.0000000 | 7 decimal places |
| PNG pixels | (H, W, 3) | uint8 | 0…255 | jet colormap |
