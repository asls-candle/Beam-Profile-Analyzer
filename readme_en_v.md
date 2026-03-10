# Beam Profile Analyzer — Data Flow and Processing Algorithms

---

## 1. Overview

The application captures frames from a FireWire camera (IEEE 1394, IIDC protocol, pydc1394),
subtracts a pre-collected background, and computes beam statistics (centroid, RMS beam size)
on the resulting difference image. Results can be exported either as CSV/JSON data files or
as PNG images.

---

## 2. Complete Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    CAMERA (FireWire / IIDC)                                  ║
║                    Pixel format: MONO16 / Y16, big-endian, uint16                            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
                                           │
                        raw bytes (2 bytes per pixel, big-endian)
                                           │
                                           ▼
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                         CameraManager._decode_frame()                                        ║
║                                                                                              ║
║  arr = np.frombuffer(raw_bytes, dtype='>u2').reshape((H, W))                                 ║
║  Returns arr.copy()                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
                                           │
                          ndarray shape (H, W), dtype=uint16
                                           │
               ┌───────────────────────────┴───────────────────────────┐
               │ SHOT path                                             │ BACKGROUND path
               ▼                                                       ▼
╔══════════════════════════╗                           ╔══════════════════════════════════════╗
║  raw_current_frame       ║                           ║  capture_background_frames()         ║
║  (stored as-is)          ║                           ║                                      ║
║  dtype: uint16           ║                           ║  N=40 frames, each cast to float64,  ║
╚══════════════════════════╝                           ║  stored as ndarray (N, H, W)         ║
               │                                       ╚══════════════════════════════════════╝
               │                                                       │
               │                                   ndarray (N, H, W), dtype=float64
               │                                                       │
               │                                                       ▼
               │                              ╔══════════════════════════════════════════════╗
               │                              ║  ImageNormalizer.process_background()        ║
               │                              ║                                              ║
               │                              ║  raw_bg = np.mean(frames, axis=0)            ║
               │                              ║  raw_bg[raw_bg < 0] = 0                      ║
               │                              ╚══════════════════════════════════════════════╝
               │                                                       │
               │                                    raw_bg: ndarray (H, W), dtype=float64
               │                                    (stored as raw_background)
               │                                                       │
               ▼                                                       ▼
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                      ImageNormalizer.normalize()  ×2                                         ║
║                                                                                              ║
║  current_frame = normalize(raw_current_frame)                                                ║
║    → image_f64 = frame.astype(float64)                                                       ║
║    → min_val, max_val from image itself                                                      ║
║    → result = (image_f64 − min_val) / (max_val − min_val)    ∈ [0, 1]                        ║
║                                                                                              ║
║  background = normalize(raw_bg, reference_image=raw_current_frame)                           ║
║    → min_val, max_val taken from raw_current_frame                                           ║
║    → result = (raw_bg_f64 − min_val) / (max_val − min_val)                                   ║
║    (same scale as current_frame; may go slightly outside [0,1])                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
               │                                                       │
  current_frame: float64 ∈ [0,1]                     background: float64 (reference-scaled)
               │                                                       │
               └───────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                      ImageReader._calculate_difference()                                     ║
║                                                                                              ║
║  raw_diff = raw_current_frame.astype(float64) − raw_background.astype(float64)               ║
║  raw_diff[raw_diff < 0] = 0                     ← clip negative values                       ║
║  difference = ImageNormalizer.normalize(raw_diff)                                            ║
║    → min_val = min(raw_diff),  max_val = max(raw_diff)                                       ║
║    → difference = (raw_diff − min_val) / (max_val − min_val)    ∈ [0, 1]                     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
                                           │
                     difference: ndarray (H, W), dtype=float64 ∈ [0, 1]
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼ DISPLAY ONLY              ▼ ANALYSIS                  ▼ EXPORT
```

---

## 3. ROI — Region of Interest

### 3.1 How ROI Is Set

The user draws a rectangle on the heatmap with a left-click drag. Coordinates are captured
in **millimetres** (the axes data space). These mm coordinates are then converted to
**pixel indices**:

```
x_min = round( x0_mm / pixel_size_x  +  img_width  / 2 )
x_max = round( x1_mm / pixel_size_x  +  img_width  / 2 )
y_min = round( y0_mm / pixel_size_y  +  img_height / 2 )
y_max = round( y1_mm / pixel_size_y  +  img_height / 2 )
```

The origin of the coordinate system is the **centre of the image**. Positive X points right,
positive Y points up. Results are clamped to `[0, img_width]` and `[0, img_height]`.

The pixel ROI `(x_min, y_min, x_max, y_max)` is stored in `ImageAnalyzer.roi`.

### 3.2 How ROI Is Applied to the Image

```python
cropped = image[y_min:y_max, x_min:x_max]   # numpy array slicing
```

This cropped sub-array is used as input for **all** subsequent calculations
(projections, centroid, RMS).

### 3.3 Coordinate Grid Inside ROI

After cropping, the pixel indices restart from 0 within the sub-array:

```
i = 0, 1, …, (x_max − x_min − 1)   — column indices inside ROI (local)
j = 0, 1, …, (y_max − y_min − 1)   — row indices inside ROI (local)
```

These **local** indices are used as-is for projections and RMS.

For the **centroid**, the local index is converted back to the global
(full-frame) pixel coordinate before the final mm conversion:

```
c_x_global = c_x_local + x_min_roi
c_y_global = c_y_local + y_min_roi
```

This ensures the centroid is always expressed relative to the **centre of the
full (uncropped) frame**, regardless of where the ROI is placed.

### 3.4 ROI at Export

When the user exports data with an active ROI, all image arrays (shot, background,
difference) are physically cropped to the ROI boundaries **before** saving.
The `resolution` metadata field is updated accordingly.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Full image  (H × W)                          │
│                                                                 │
│   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                             │
│   │                                 │                           │
│   │        ROI sub-array            │  ← used for all analysis  │
│   │   image[y_min:y_max,            │                           │
│   │          x_min:x_max]           │                           │
│   │                                 │                           │
│   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Analysis Algorithms

All algorithms operate on the **difference** image (float64, ∈ [0,1]).
If an ROI is active, the image is first cropped (see §3.2) before the calculations below.

---

### 4.1 Projections

```
calculate_projections(image, pixel_size_x, pixel_size_y)
```

**Step 1 — Sum along each axis (marginal distributions):**

```
P_x[i] = Σ_j  image[j, i]       (sum over all rows for column i)
P_y[j] = Σ_i  image[j, i]       (sum over all columns for row j)
```

**Step 2 — Normalise to maximum:**

```
P_x[i] = P_x[i] / max(P_x)
P_y[j] = P_y[j] / max(P_y)
```

**Step 3 — Convert pixel index to millimetres:**

```
x_coords[i] = i · pixel_size_x       (mm, origin at left edge of ROI)
y_coords[j] = j · pixel_size_y       (mm, origin at top edge of ROI)
```

**Output:** `(x_coords [mm], P_x [a.u.], y_coords [mm], P_y [a.u.])`

---

### 4.2 Centroid

```
calculate_centroid(image, pixel_size_x, pixel_size_y)
```

The function receives the **full (uncropped) image**. The full-frame dimensions
are saved first, then the ROI crop is applied internally.

**Step 1 — Save full-frame dimensions (before cropping):**

```
full_H, full_W = image.shape
```

**Step 2 — Apply ROI crop:**

```
processed_image = image[y_min:y_max, x_min:x_max]   (H' × W' sub-array)
```

**Step 3 — Marginal projections (not normalised) on the cropped image:**

```
P_x[i] = Σ_j  processed_image[j, i]
P_y[j] = Σ_i  processed_image[j, i]
```

**Step 4 — Intensity-weighted mean position in local ROI pixels:**

```
           Σ_i  i · P_x[i]                   Σ_j  j · P_y[j]
c_x_loc = ─────────────────────    c_y_loc = ─────────────────────
               Σ_i  P_x[i]                        Σ_j  P_y[j]
```

**Step 5 — Convert local ROI pixel to global frame pixel (add ROI offset):**

```
c_x_px = c_x_loc + x_min_roi
c_y_px = c_y_loc + y_min_roi
```

This step is **skipped** when no ROI is active (`c_x_px = c_x_loc`).

**Step 6 — Convert to millimetres, centred on the full-frame centre:**

```
centroid_x = ( c_x_px  −  full_W / 2 ) · pixel_size_x   [mm]
centroid_y = ( c_y_px  −  full_H / 2 ) · pixel_size_y   [mm]
```

> **Key property:** The centroid is always measured relative to the
> **geometric centre of the full sensor frame**, both with and without an active
> ROI. Selecting an ROI does not shift the coordinate origin — it only restricts
> which pixels contribute to the intensity-weighted average.

**Output:** `(centroid_x [mm], centroid_y [mm])`

---

### 4.3 RMS Beam Size

```
calculate_rms(image, pixel_size_x, pixel_size_y)
```

**Step 1 — Same marginal projections and centroid as §4.2.**

**Step 2 — Intensity-weighted standard deviation (in pixels):**

```
            ┌  Σ_i  P_x[i] · (i − c_x_px)²    ┐½
rms_x_px =  │ ──────────────────────────────  │
            └         Σ_i  P_x[i]             ┘

            ┌  Σ_j  P_y[j] · (j − c_y_px)²    ┐½
rms_y_px =  │ ──────────────────────────────  │
            └         Σ_j  P_y[j]             ┘
```

**Step 3 — Convert to millimetres:**

```
rms_x = rms_x_px · pixel_size_x   [mm]
rms_y = rms_y_px · pixel_size_y   [mm]
```

**Output:** `(rms_x [mm], rms_y [mm])`

---

## 5. Pixel Size

Pixel size (mm/pixel) is determined automatically from the sensor resolution:

| Sensor resolution | Pixel size (mm) |
|---|---|
| 1032 × 776  | 0.02840909 |
| 1624 × 1224 | 0.01875468 |
| other       | 0.01 (default) |

The same `pixel_size_x` and `pixel_size_y` values are used for all coordinate
conversions in §4 and for the projection axes in the plots.

---

## 6. Display vs. Export

```
                difference: float64 (H, W) ∈ [0, 1]
                           │
         ┌─────────────────┴──────────────────────────┐
         │ DISPLAY ONLY                               │ DISPLAY + EXPORT
         │                                            │
         ▼                                            ▼
╔════════════════════════╗          ╔══════════════════════════════════════════════╗
║  PlotManager           ║          ║  ImageAnalyzer                               ║
║                        ║          ║                                              ║
║  imshow(difference,    ║          ║  If ROI active: crop image[y0:y1, x0:x1]     ║
║    cmap='jet',         ║          ║                                              ║
║    vmin=0, vmax=1,     ║          ║  calculate_projections() → x_coords, P_x,    ║
║    origin='lower')     ║          ║                            y_coords, P_y     ║
║                        ║          ║  calculate_centroid()    → cx, cy [mm]       ║
║  Projection plots      ║          ║  calculate_rms()         → rx, ry [mm]       ║
║  (x_coords, P_x and    ║          ╚══════════════════════════════════════════════╝
║   y_coords, P_y)       ║                           │
║  displayed on screen   ║                Beam stats shown in UI labels
╚════════════════════════╝                and written to export metadata
         │
    screen only
         │
         └─── NOT written to CSV or PNG files


                   ┌───────────────────────────┐
                   │       EXPORT PATH         │
                   └───────────────────────────┘

  On "Export Data" button press, data is (optionally ROI-cropped) and saved:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  CSV / JSON export  (DataExporter.export_csv)                               │
  │                                                                             │
  │  metadata.json ── scalar values:                                            │
  │    resolution, pixel_size_x, pixel_size_y, date,                            │
  │    centroid_x_mm, centroid_y_mm, rms_x_mm, rms_y_mm,                        │
  │    roi_active, roi_pixels, roi_mm                                           │
  │                                                                             │
  │  shot.csv        ── float64, 17 decimal places, shape (H, W)                │
  │  background.csv  ── float64, 17 decimal places, shape (H, W)                │
  │  difference.csv  ── float64, 17 decimal places, shape (H, W)                │
  │  raw_shot.csv    ── uint16 values (original camera data), shape (H, W)      │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  PNG export  (DataExporter.export_png)                                      │
  │                                                                             │
  │  For each of shot / background / difference:                                │
  │    fig size = (W px / dpi, H px / dpi) at dpi=100 → exact pixel match       │
  │    imshow(plain, cmap='jet', vmin=0, vmax=1, origin='lower')                │
  │    saves <name>.png  — same colour scale as on-screen display               │
  │                                                                             │
  │  metadata.json ── resolution, pixel_size, date, rms_x_mm, rms_y_mm          │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Type Summary Table

| Stage | Variable | dtype | Value range |
|---|---|---|---|
| Camera raw buffer | `raw_bytes` | bytes | — |
| After `_decode_frame` | `raw_current_frame` | uint16 | 0 … 65535 |
| Background frames array | `background_frames` | float64 | 0 … 65535 |
| Averaged raw background | `raw_background` | float64 | 0 … 65535 |
| Normalised current frame | `current_frame` | float64 | [0, 1] |
| Normalised background | `background` | float64 | ≈ [0, 1] |
| Difference (normalised) | `difference` | float64 | [0, 1] |
| Centroid output | `centroid_x/y` | float64 | mm, signed |
| RMS output | `rms_x/y` | float64 | mm, ≥ 0 |
| Projection coords | `x_coords`, `y_coords` | float64 | mm |
| Projection values | `P_x`, `P_y` | float64 | [0, 1] |
| CSV array cells | — | text (float64) | 17 decimal places |

---

## 8. Full Pipeline Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CAMERA  (FireWire IIDC)                               │
│  Sensor: MONO16, big-endian uint16  │  pixel_size: lookup by resolution (mm)    │
└────────────────────────────────────────────────────────┬────────────────────────┘
                                                         │ raw bytes (>u2)
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  _decode_frame()                                                                │
│  np.frombuffer(raw, dtype='>u2').reshape(H, W) → ndarray uint16                 │
└────────────────┬──────────────────────────────────────────────────┬─────────────┘
                 │ uint16 (H×W)                                     │ uint16 (H×W) × N frames
                 ▼                                                  ▼
┌────────────────────────────────────┐      ┌──────────────────────────────────────┐
│  raw_current_frame  (uint16)       │      │  capture_background_frames()         │
│  stored for difference calculation │      │  frames.astype(float64)              │
└────────────────────────────────────┘      │  → ndarray (N, H, W) float64         │
                 │                          └──────────────────┬───────────────────┘
                 │                                             │ (N,H,W) float64
                 │                                             ▼
                 │                          ┌──────────────────────────────────────┐
                 │                          │  process_background()                │
                 │                          │  raw_bg = mean(frames, axis=0)       │
                 │                          │  raw_bg[raw_bg < 0] = 0              │
                 │                          │  → raw_background  float64 (H×W)     │
                 │                          └──────────────────┬───────────────────┘
                 │                                             │ float64 (H×W)
                 ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  _calculate_difference()                                                        │
│  raw_diff = raw_current.astype(f64) − raw_background.astype(f64)                │
│  raw_diff[raw_diff < 0] = 0                                                     │
│  difference = normalize(raw_diff)  →  float64 ∈ [0, 1]                          │
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │ difference: float64 (H×W) ∈ [0,1]
                     ┌────────────┴─────────────┐
                     │                          │
                     ▼ DISPLAY                  ▼ ANALYSIS & EXPORT
          ┌──────────────────────┐    ┌──────────────────────────────────────────┐
          │  PlotManager         │    │  calculate_centroid() receives FULL frame│
          │  imshow jet,         │    │  full_W, full_H = image.shape            │
          │  vmin=0 vmax=1       │    │  crop → image[y0:y1, x0:x1]  (H'×W')     │
          │  Projection plots    │    └───────────────────┬──────────────────────┘
          │  on screen only      │                        │ full frame float64 (H×W)
          └──────────────────────┘          ┌─────────────┴────────────────┐
                                            │                              │
                                            ▼                              ▼
                                 ┌───────────────────────┐  ┌──────────────────────────┐
                                 │  calculate_centroid() │  │  calculate_rms()         │
                                 │                       │  │  P_x = sum(img, ax=0)    │
                                 │  full_W,full_H =      │  │  P_y = sum(img, ax=1)    │
                                 │    image.shape        │  │                          │
                                 │  crop → ROI sub-array │  │  c_x_loc=Σ(i·P_x)/ΣP_x   │
                                 │  P_x = sum(roi,ax=0)  │  │                          │
                                 │  P_y = sum(roi,ax=1)  │  │  σ_x_px =                │
                                 │                       │  │  √(Σ P_x·(i−c_x)²/ΣP_x)  │
                                 │  c_x_loc=Σ(i·P_x)/ΣP_x│  │                          │
                                 │  c_x_px=c_x_loc+x_min │  │  rms_x = σ_x_px          │
                                 │  c_y_px=c_y_loc+y_min │  │          · pixel_size_x  │
                                 │                       │  │  rms_y = σ_y_px          │
                                 │  cx=(c_x_px−full_W/2) │  │          · pixel_size_y  │
                                 │     · pixel_size_x    │  │  → rms_x, rms_y  [mm]    │
                                 │  cy=(c_y_px−full_H/2) │  │                          │
                                 │     · pixel_size_y    │  │  (RMS uses local ROI     │
                                 │  → cx, cy  [mm]       │  │   pixel indices only)    │
                                 └───────────────────────┘  └──────────────────────────┘
                                            │                              │
                                            └──────────────┬───────────────┘
                                                           │
                                   ┌───────────────────────┴──────────────────────────┐
                                   │  EXPORT                                          │
                                   │                                                  │
                                   │  CSV/JSON:                                       │
                                   │    metadata.json  ← scalars (cx, cy, rms, …)     │
                                   │    shot.csv       ← float64, 17 digits, (H×W)    │
                                   │    background.csv ← float64, 17 digits, (H×W)    │
                                   │    difference.csv ← float64, 17 digits, (H×W)    │
                                   │                                                  │
                                   │  PNG:                                            │
                                   │    shot.png       ← jet colormap, vmin=0,vmax=1  │
                                   │    background.png ← same scale                   │
                                   │    difference.png ← same scale                   │
                                   │    metadata.json  ← rms_x_mm, rms_y_mm, …        │
                                   └──────────────────────────────────────────────────┘
```

---

## 9. Notes

* **Byte order:** All cameras used are FireWire IIDC (Flea2 and similar). They always
  transmit data in **big-endian** order. The decoder uses `dtype='>u2'` (big-endian
  unsigned 16-bit) unconditionally.

* **Background normalisation:** The background image is normalised using the
  *current frame* as the reference. This means that `background` and `current_frame`
  share the same intensity scale, so their difference is physically meaningful.

* **Negative values:** Subtracting the raw background from the raw shot can produce
  negative values (shot noise, dark current fluctuations). These are clipped to zero
  before normalisation.

* **Centroid and RMS in mm:** All spatial statistics are computed **first in pixel
  space**, then scaled to millimetres using the look-up table pixel size. No
  intermediate mm-space grid is used for the integrals.

* **ROI effect on centroid/RMS:** When an ROI is active, the centroid is still
  expressed relative to the **centre of the full sensor frame**. The algorithm
  computes the intensity-weighted centre in local ROI pixel coordinates, then adds
  the ROI corner offset `(x_min_roi, y_min_roi)` to obtain the global pixel
  coordinate before converting to mm. This means the centroid value is directly
  comparable between ROI and non-ROI measurements.
  RMS, by contrast, is a spread measure (second central moment) and is computed
  entirely within local ROI pixel space — it reflects only the extent of the beam
  within the selected region.

* **Median filter:** The code contains a median filter (`apply_median_filter`) but
  it is **disabled** (commented out). No spatial filtering is applied to the data.

* **Gaussian fitting:** A Gaussian fitting function exists in `ImageAnalyzer` but
  is **not called** in the current processing pipeline and does not affect any
  displayed or exported values.
