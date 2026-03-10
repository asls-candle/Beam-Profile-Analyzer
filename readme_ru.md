# Поток данных и алгоритмы обработки

Система анализа профиля лазерного пучка.

---

## Содержание

1. [Общая блок-схема](#1-общая-блок-схема)
2. [Захват кадра с камеры](#2-захват-кадра-с-камеры)
3. [Нормализация и вычитание фона](#3-нормализация-и-вычитание-фона)
4. [Область интереса (ROI)](#4-область-интереса-roi)
5. [Анализ профиля пучка](#5-анализ-профиля-пучка)
6. [Экспорт данных](#6-экспорт-данных)
7. [Итоговая таблица типов данных](#7-итоговая-таблица-типов-данных)

---

## 1. Общая блок-схема

На схеме ниже показано, какие данные используются **только для отображения в интерфейсе**, а какие **попадают в файлы** (CSV и PNG).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             КАМЕРА (IEEE 1394 / FireWire)                       │
│                  FORMAT7_0 · MONO16 · big-endian · dtype: uint16                │
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │  raw bytes (DMA buffer)
                                  ▼
                    ┌─────────────────────────┐
                    │   _decode_frame()        │
                    │   uint16 → ndarray(H,W)  │
                    └────────────┬────────────┘
                                 │  ndarray (H,W) uint16
          ┌──────────────────────┴──────────────────────┐
          │ Одиночный кадр                               │ Захват N фоновых кадров
          ▼                                              ▼
┌─────────────────────────┐                  ┌──────────────────────────────┐
│ normalize(frame)        │                  │ capture_background_frames(N) │
│ uint16 → float64 [0,1]  │                  │ каждый кадр → float64        │
│ по собственным min/max  │                  │ стек shape (N,H,W) float64   │
└──────────┬──────────────┘                  └──────────────┬───────────────┘
           │ current_frame                                   │ background_frames
           │ float64 [0,1]                                   ▼
           │                                  ┌─────────────────────────────┐
           │                                  │ process_background()         │
           │                                  │ среднее по оси N             │
           │                                  │ float64, единицы uint16      │
           │                                  └──────────────┬──────────────┘
           │                                                 │ raw_background float64
           │                                  ┌─────────────┴──────────────┐
           │                                  │ normalize(bg, ref=raw_shot) │
           │                                  │ нормировка по min/max shot  │
           │                                  │ float64 ≈ [0, ≤1.0]         │
           │                                  └──────────────┬──────────────┘
           │                                                 │ background float64
           │◄────────────────────────────────────────────────┤
           │                                                 │
           ▼                                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  _calculate_difference()                                             │
│  raw_shot.astype(float64) − raw_background.astype(float64)           │
│  clip отрицательных → 0                                              │
│  normalize(raw_diff)  [по min/max самой разности]                    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ difference  float64 [0,1]
            ┌──────────────────────┼──────────────────────────────────┐
            │                      │                                   │
            ▼                      ▼                                   ▼
┌────────────────────┐  ┌──────────────────────────────┐  ┌─────────────────────┐
│  ТОЛЬКО ДИСПЛЕЙ    │  │  АНАЛИЗ (centroid, RMS)       │  │  ЭКСПОРТ            │
│                    │  │                               │  │                     │
│ apply_median_filter│  │  [опц.] apply_roi(difference) │  │  [опц.] crop to ROI │
│ kernel=3           │  │  pixel-space slice            │  │  pixel-space slice  │
│ (шум подавлен      │  │  ↓                            │  │  ↓                  │
│  только на экране) │  │  calculate_centroid()         │  │  export_csv()       │
│ ↓                  │  │  calculate_rms()              │  │  → .csv + .json     │
│ heatmap + profiles │  │  результат в мм               │  │                     │
│ в UI               │  │  ↓                            │  │  export_png()       │
│                    │  │  подпись в UI                 │  │  → .png (jet, [0,1])│
└────────────────────┘  └──────────────────────────────┘  └─────────────────────┘
     ↑ только экран           ↑ экран + экспорт-метаданные       ↑ файлы на диск
```

---

## 2. Захват кадра с камеры

**Модуль:** `camera_manager.py` · **Класс:** `CameraManager`

### Аппаратный интерфейс

| Параметр | Значение |
|---|---|
| Протокол | IEEE 1394 (FireWire), библиотека `pydc1394` |
| Формат пикселя | MONO16 (Y16) — 16-бит монохромный |
| Порядок байт | big-endian (`>u2`) — фиксирован для используемых камер |
| Режим захвата | FORMAT7_0, максимальное доступное разрешение |

### Декодирование буфера

Метод `_decode_frame(raw_bytes)` принимает байтовый буфер DMA и возвращает двумерный массив:

```
arr = np.frombuffer(raw_bytes, dtype='>u2').reshape((H, W)).copy()
```

**Результат:** `ndarray shape=(H, W), dtype=uint16, диапазон 0…65535`

### Захват фоновых кадров

`capture_background_frames(N)` в цикле вызывает `_decode_frame` и сразу конвертирует в `float64`:

```
frames[i] = arr.astype(np.float64)   # uint16 → float64, значения 0…65535
```

**Результат:** `ndarray shape=(N, H, W), dtype=float64, значения 0…65535`

---

## 3. Нормализация и вычитание фона

**Модуль:** `image_reader.py` · **Класс:** `ImageReader`

### 3.1 Нормализация кадра (shot)

Вызывается в `_reading_worker` для каждого захваченного кадра:

```python
current_frame = ImageNormalizer.normalize(raw_frame)
```

Алгоритм (`normalize` без `reference_image`):

```
min_val = min(image)
max_val = max(image)
normalized = (image.astype(float64) − min_val) / (max_val − min_val)
```

Нормировка выполняется по **собственным** min/max данного кадра.  
**Результат:** `ndarray (H,W) float64, диапазон [0.0, 1.0]`

### 3.2 Усреднение фона

```python
raw_bg = np.mean(background_frames, axis=0)   # поэлементное среднее по N кадрам
raw_bg[raw_bg < 0] = 0
```

**Результат:** `ndarray (H,W) float64`, значения в исходных единицах (`0…65535`)

### 3.3 Нормализация фона относительно shot

Фон нормализуется с использованием `raw_current_frame` как референса, чтобы шкала интенсивности фона совпадала со шкалой снятого кадра:

```python
background = ImageNormalizer.normalize(raw_bg, reference_image=raw_current_frame)
```

Алгоритм (`normalize` с `reference_image`):

```
min_val = min(raw_current_frame)   # min/max берутся из shot, не из bg
max_val = max(raw_current_frame)
background = (raw_bg.astype(float64) − min_val) / (max_val − min_val)
```

**Результат:** `ndarray (H,W) float64, диапазон ≈ [0.0, ≤ 1.0]`  
Значения фона могут не достигать 1.0, поскольку нормируются по shot.

### 3.4 Вычитание фона (`difference`)

Вычитание выполняется в **исходных единицах** (до нормализации):

```python
raw_diff = raw_current_frame.astype(float64) − raw_background.astype(float64)
raw_diff[raw_diff < 0] = 0           # пиксели темнее фона → 0
difference = ImageNormalizer.normalize(raw_diff)   # по min/max самой разности
```

Нормировка `difference` выполняется по **собственным** min/max разности (без референса).  
**Результат:** `ndarray (H,W) float64, диапазон [0.0, 1.0]`  
Максимальный пиксель пучка всегда равен 1.0.

---

## 4. Область интереса (ROI)

### 4.1 Как пользователь задаёт ROI

Пользователь рисует прямоугольник мышью прямо на тепловой карте (`heatmap`). Тепловая карта отображается в **миллиметрах** — оси `imshow` настроены через `extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]`. Поэтому `RoiSelector` возвращает координаты в мм:

```
on_roi_selected(x0_mm, y0_mm, x1_mm, y1_mm)
```

### 4.2 Перевод ROI из мм в пиксели

`MainWindow.set_roi_mm()` конвертирует координаты из пространства данных (мм) в пиксельные индексы массива. Ноль мм соответствует центру кадра:

```python
x_min = max(0,          round( x0_mm / pixel_size_x + W / 2 ))
x_max = min(img_width,  round( x1_mm / pixel_size_x + W / 2 ))
y_min = max(0,          round( y0_mm / pixel_size_y + H / 2 ))
y_max = min(img_height, round( y1_mm / pixel_size_y + H / 2 ))
```

Результирующие пиксельные координаты сохраняются в `ImageAnalyzer.roi = (x_min, y_min, x_max, y_max)`.  
Координаты в мм дублируются в `MainWindow.current_roi_mm` — для переотрисовки рамки при смене вкладки.

### 4.3 Применение ROI при вычислении центроида и RMS

В начале `calculate_centroid` и `calculate_rms` изображение обрезается:

```python
processed = image[y_min:y_max, x_min:x_max]   # shape (H_roi, W_roi)
```

Все дальнейшие вычисления (проекции, суммы, индексы) работают с обрезанным массивом. Индексы `x` и `y` при этом отсчитываются **от 0 внутри ROI**, а не от 0 всего кадра.

Перевод в мм относительно **центра ROI:**

```
centroid_x = (cx_px_roi − W_roi / 2) · pixel_size_x   (мм)
centroid_y = (cy_px_roi − H_roi / 2) · pixel_size_y   (мм)
```

### 4.4 Применение ROI при экспорте

При нажатии «Export» `MainWindow.apply_roi_to_data()` обрезает все изображения до ROI **перед** записью на диск:

```python
for key in ('shot', 'background', 'difference', 'raw_shot', 'raw_background'):
    result[key] = arr[y_min:y_max, x_min:x_max]
result['resolution'] = (x_max − x_min, y_max − y_min)
```

В CSV и PNG сохраняется только вырезанная область.

### 4.5 Сброс ROI

Двойной клик на тепловой карте или кнопка «Reset ROI»:

```python
ImageAnalyzer.roi = None
MainWindow.current_roi_mm = None
```

---

## 5. Анализ профиля пучка

**Модуль:** `image_analyzer.py` · **Класс:** `ImageAnalyzer`

Все расчёты выполняются из `difference`-изображения (`ndarray float64, [0,1]`). Значения пикселей используются как **веса интенсивности**.

### 5.1 Одномерные проекции (только для отображения в UI)

`calculate_projections` — используется исключительно для отрисовки профилей в интерфейсе. В CSV не попадает.

```
x_proj[x] = Σ_y  processed[y, x]       (сумма по всем строкам столбца x)
y_proj[y] = Σ_x  processed[y, x]       (сумма по всем столбцам строки y)

# нормализация для отображения:
x_proj = x_proj / max(x_proj)
y_proj = y_proj / max(y_proj)

# перевод индексов в мм для осей графика:
x_coords[x] = x · pixel_size_x   (мм)
y_coords[y] = y · pixel_size_y   (мм)
```

### 5.2 Центроид

`calculate_centroid(image, pixel_size_x, pixel_size_y)`

**Шаг 1** — проекции (не нормированные):
```
x_proj[x] = Σ_y  processed[y, x]
y_proj[y] = Σ_x  processed[y, x]
```

**Шаг 2** — взвешенное среднее положение **в пикселях**:
```
cx_px = ( Σ_x  x · x_proj[x] ) / ( Σ_x  x_proj[x] )
cy_px = ( Σ_y  y · y_proj[y] ) / ( Σ_y  y_proj[y] )
```
Здесь `x` и `y` — целочисленные индексы от `0` до `W−1` (или `H−1`).

**Шаг 3** — перевод в мм с центрированием:
```
centroid_x = (cx_px − W / 2) · pixel_size_x   (мм)
centroid_y = (cy_px − H / 2) · pixel_size_y   (мм)
```

Ноль координат соответствует центру кадра (или ROI). Весь расчёт ведётся в пиксельных индексах; перевод в мм — только финального результата.

### 5.3 RMS-размер пучка

`calculate_rms(image, pixel_size_x, pixel_size_y)`

**Шаг 1** — проекции (те же, что в центроиде):
```
x_proj[x] = Σ_y  processed[y, x]
y_proj[y] = Σ_x  processed[y, x]
```

**Шаг 2** — центроид в пикселях (вычисляется повторно):
```
cx_px = ( Σ_x  x · x_proj[x] ) / ( Σ_x  x_proj[x] )
cy_px = ( Σ_y  y · y_proj[y] ) / ( Σ_y  y_proj[y] )
```

**Шаг 3** — RMS как взвешенное стандартное отклонение **в пикселях**:
```
rms_x_px = √( Σ_x  x_proj[x] · (x − cx_px)²  /  Σ_x  x_proj[x] )
rms_y_px = √( Σ_y  y_proj[y] · (y − cy_px)²  /  Σ_y  y_proj[y] )
```

**Шаг 4** — перевод в мм:
```
rms_x = rms_x_px · pixel_size_x   (мм)
rms_y = rms_y_px · pixel_size_y   (мм)
```

RMS не центрируется — мера ширины распределения вычисляется уже относительно найденного центроида.

### 5.4 Перевод пикселей в мм — общий принцип

`pixel_size_x` и `pixel_size_y` (мм/пиксель) определяются из словаря `PIXEL_SIZE_BY_RESOLUTION` в `CameraManager` по произведению `W·H` выбранного режима камеры.

| Величина | Формула перевода |
|---|---|
| Оси тепловой карты (UI) | `x_mm = (arange(W) − W/2) · pixel_size_x` |
| Центроид X | `(cx_px − W/2) · pixel_size_x` |
| Центроид Y | `(cy_px − H/2) · pixel_size_y` |
| RMS X | `rms_x_px · pixel_size_x` |
| RMS Y | `rms_y_px · pixel_size_y` |

Формула `index − size/2` переносит начало координат из левого верхнего угла пикселя (0,0) в геометрический центр кадра.

---

## 6. Экспорт данных

**Модуль:** `exporter.py` · **Класс:** `DataExporter`

### 6.1 CSV + JSON (`export_csv`)

Данные разделяются на два типа:

**Скаляры → `metadata.json`:**
- `centroid_x_mm`, `centroid_y_mm`
- `rms_x_mm`, `rms_y_mm`
- `pixel_size_x`, `pixel_size_y`
- `resolution`, `date`
- `roi_pixels`, `roi_mm` (если ROI активен)

**Массивы → отдельные `<name>.csv`:**

| Файл | Shape | Содержимое |
|---|---|---|
| `shot.csv` | H × W | нормализованный снятый кадр, float64 [0,1] |
| `background.csv` | H × W | нормализованный фон, float64 [0,≤1] |
| `difference.csv` | H × W | difference после вычитания, float64 [0,1] |

Каждое число записывается с 7 знаками после запятой. Двумерные массивы — построчно (H строк, W чисел через запятую).

Перед записью `MaskedArray` конвертируется в `plain ndarray` через `.filled(0)`.

### 6.2 PNG (`export_png`)

- Сохраняются три файла: `shot.png`, `background.png`, `difference.png`
- Цветовая карта: `jet`, единая шкала `vmin=0.0, vmax=1.0` для всех трёх — соответствует отображению в интерфейсе
- Разрешение файла точно соответствует разрешению данных: `dpi=100`, `figsize=(W/100, H/100)`, `interpolation='nearest'`
- Ориентация: `origin='lower'` — нулевая строка внизу

---

## 7. Итоговая таблица типов данных

| Переменная | Shape | dtype | Диапазон | Примечание |
|---|---|---|---|---|
| `raw_current_frame` | (H, W) | uint16 | 0…65535 | сырой кадр с камеры |
| `current_frame` | (H, W) | float64 | 0.0…1.0 | нормировка по себе |
| `background_frames` | (N, H, W) | float64 | 0…65535 | стек до усреднения |
| `raw_background` | (H, W) | float64 | ≥0, ~0…65535 | поэлементное среднее |
| `background` | (H, W) | float64 | ≈0.0…≤1.0 | нормировка по ref=shot |
| `raw_diff` | (H, W) | float64 | 0…~65535 | shot−bg, clip(0) |
| `difference` | (H, W) | float64 | 0.0…1.0 | нормировка по себе |
| `cx_px`, `cy_px` | скаляр | float64 | 0…W (H) | пиксели |
| `centroid_x/y` | скаляр | float64 | ±мм | центрир. от центра кадра |
| `rms_x/y` | скаляр | float64 | ≥0 мм | ширина распределения |
| CSV-ячейки | H×W / W | str | 0.0000000…1.0000000 | 7 знаков |
| PNG-пиксели | (H, W, 3) | uint8 | 0…255 | jet colormap |
