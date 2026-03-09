# src/camera/camera_manager.py

import os
import time
import math
import numpy as np
import logging
from threading import Lock

# os.environ['DC1394_V2_STRATEGY'] = '1'

try:
    from pydc1394 import Camera, Context
    CAMERA_AVAILABLE = True
except (ImportError, TypeError, OSError) as e:
    CAMERA_AVAILABLE = False
    print("pydc1394 не найден или не может быть загружен. Функциональность камеры не будет доступна.")
    print("Ошибка: {}".format(e))

logger = logging.getLogger("camera")


# ── Вспомогательные функции ────────────────────────────────────────────────────

def _is_mono16(mode):
    """Проверяет, является ли режим 16-битным монохромным"""
    mode_str = str(mode).upper()
    return 'MONO16' in mode_str or 'GRAY16' in mode_str or 'Y16' in mode_str


def _mode_resolution(mode):
    """Возвращает количество пикселей (w*h) для данного режима"""
    try:
        w, h = mode.image_size
        return w * h
    except Exception:
        pass
    for token in str(mode).split('_'):
        if 'x' in token:
            parts = token.split('x')
            if len(parts) == 2:
                try:
                    return int(parts[0]) * int(parts[1])
                except ValueError:
                    pass
    return 0


def _get_image_size(mode):
    """Извлекает (width, height) из объекта режима"""
    try:
        return tuple(mode.image_size)
    except Exception:
        pass
    for token in str(mode).split('_'):
        if 'x' in token:
            parts = token.split('x')
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    pass
    return 0, 0


def _apply_feature(camera, name, value, mode='manual'):
    """Безопасно устанавливает параметр камеры"""
    try:
        feat = getattr(camera, name)
        try:
            feat.mode = mode
        except Exception:
            pass
        try:
            lo, hi = feat.range
            if value < lo:
                logger.debug("CLAMP %s: %s -> %s", name, value, lo)
                value = lo
            if value > hi:
                logger.debug("CLAMP %s: %s -> %s", name, value, hi)
                value = hi
        except Exception:
            pass
        feat.val = value
        try:
            actual = feat.val
            logger.info("  %s = %s (requested: %s)", name, actual, value)
        except Exception:
            logger.info("  %s set to %s (read-back unavailable)", name, value)
    except AttributeError:
        logger.debug("  %s not available on this camera", name)
    except Exception as e:
        logger.warning("  %s ERROR: %s", name, str(e))


def _decode_vendor_model(camera):
    """Декодирует vendor и model из байт или строки"""
    try:
        vendor = camera.vendor.decode('utf-8', errors='replace') \
            if isinstance(camera.vendor, bytes) else str(camera.vendor)
    except Exception:
        vendor = "Unknown"
    try:
        model = camera.model.decode('utf-8', errors='replace') \
            if isinstance(camera.model, bytes) else str(camera.model)
    except Exception:
        model = "Unknown"
    return vendor.strip(), model.strip()


def _make_camera_display_name(vendor, model, guid_str):
    """
    Формирует два варианта отображаемого имени камеры:

    - short_name: «Model (GUID_последние8)»   — для UI-виджетов
    - full_name:  «Vendor Model (GUID_последние8)»  — для тултипов

    Returns:
        (short_name, full_name)
    """
    short_guid = guid_str[-8:] if len(guid_str) >= 8 else guid_str
    short_name = "{} ({})".format(model, short_guid)
    full_name  = "{} {} ({})".format(vendor, model, short_guid)
    return short_name, full_name


class CameraManager:
    """
    Менеджер камер с полностью динамическим обнаружением.

    Никаких предустановленных камер нет. Имена формируются из
    vendor/model/GUID. Размер пикселя определяется автоматически
    по разрешению сенсора (PIXEL_SIZE_BY_RESOLUTION).

    Для каждой камеры автоматически выбирается максимальное разрешение
    в монохромном 16-битном режиме (FORMAT7_0 → Y16/MONO16).

    Поддержка внешнего триггера (IIDC 1394):
      - Триггер настраивается числовыми константами IIDC, а не строками.
      - При use_trigger=True прогрев и автоопределение byte order
        пропускаются (камера не генерирует кадры без сигнала).
      - Переключение триггера на лету выполняется с остановкой/стартом
        видеопотока, как того требует стандарт IIDC.
      - capture_single_frame() явно различает «нет сигнала» и «ошибка».
    """

    # Настройки экспозиции по умолчанию
    DEFAULT_EXPOSURE = {
        "shutter":    1,
        "gain":       0,
        "brightness": 0,
        "exposure":   0,
    }

    # Размер пикселя по умолчанию (мм)
    DEFAULT_PIXEL_SIZE = 0.01

    # Соответствие разрешения сенсора размеру пикселя (мм)
    PIXEL_SIZE_BY_RESOLUTION = {
        (1032, 776):  0.02840909,
        (1624, 1224): 0.01875468,
    }

    # Параметры внешнего триггера по умолчанию (IIDC 1394)
    TRIGGER_MODE     = 0   # Mode 0 — стандартный: экспозиция по длительности сигнала
    TRIGGER_SOURCE   = 0   # GPIO pin 0 (у Flea2 — разъём GPIO)
    TRIGGER_POLARITY = 1   # 1 = активный HIGH, 0 = активный LOW

    CAPTURE_BUFSIZE  = 8
    WARMUP_FRAMES    = 5
    DEQUEUE_ATTEMPTS = 30
    DEQUEUE_DELAY    = 0.05

    # При use_trigger=True ждём кадр дольше — сигнал может прийти позже
    TRIGGER_DEQUEUE_ATTEMPTS = 200   # 200 × 0.05 с = 10 секунд
    TRIGGER_DEQUEUE_DELAY    = 0.05

    def __init__(self, use_trigger=False, use_format7=True, manual_exposure=True):
        """
        Args:
            use_trigger:      Использовать внешний триггер
            use_format7:      Предпочитать FORMAT7_0 (полный сенсор) если доступен
            manual_exposure:  Ручная экспозиция (рекомендуется для измерений)
        """
        self.camera = None
        self.camera_name = None
        self.context = None
        self.is_connected = False
        self.use_trigger = use_trigger
        self.use_format7 = use_format7
        self.manual_exposure = manual_exposure
        self.lock = Lock()

        self._width = 0
        self._height = 0
        self._byte_order = '>u2'   # Flea2 — big-endian по умолчанию
        self._pixel_size_x = self.DEFAULT_PIXEL_SIZE
        self._pixel_size_y = self.DEFAULT_PIXEL_SIZE
        self._video_started = False

        # Словарь обнаруженных камер: short_name -> info dict
        self.cameras = {}

        logger.info("Инициализация CameraManager, use_trigger=%s, use_format7=%s",
                    use_trigger, use_format7)

    # ── Обнаружение камер ─────────────────────────────────────────────────────

    def get_camera_list(self):
        """
        Обнаруживает все подключённые FireWire-камеры.

        Returns:
            Список коротких отображаемых имён обнаруженных камер
        """
        if not CAMERA_AVAILABLE:
            logger.warning("pydc1394 не доступен")
            return []

        try:
            context = Context()
            hw_cameras = context.cameras
            if not hw_cameras:
                logger.warning("Камеры не обнаружены в системе")
                return []

            self.cameras = {}

            for i, cam_entry in enumerate(hw_cameras):
                cam_guid = cam_entry[0]
                guid_str = format(int(cam_guid), '016x')

                vendor, model = "Unknown", "Unknown"
                resolution = (0, 0)

                try:
                    tmp_cam = Camera(guid=cam_guid)
                    vendor, model = _decode_vendor_model(tmp_cam)
                    resolution = self._detect_max_resolution(tmp_cam)
                    del tmp_cam
                except Exception as e:
                    logger.warning("Не удалось получить информацию о камере %d: %s", i, e)

                short_name, full_name = _make_camera_display_name(vendor, model, guid_str)

                # Два устройства с одинаковым именем — добавляем индекс
                if short_name in self.cameras:
                    short_name = "{} [{}]".format(short_name, i)
                    full_name  = "{} [{}]".format(full_name, i)

                pixel_size = self.PIXEL_SIZE_BY_RESOLUTION.get(
                    resolution, self.DEFAULT_PIXEL_SIZE)

                self.cameras[short_name] = {
                    "guid":              cam_guid,
                    "guid_str":          guid_str,
                    "vendor":            vendor,
                    "model":             model,
                    "camera_name":       short_name,
                    "camera_full_name":  full_name,
                    "resolution":        resolution,
                    "pixel_size_x":      pixel_size,
                    "pixel_size_y":      pixel_size,
                }

                logger.info("Обнаружена камера: %s | %s, GUID=%s, разрешение=%s",
                            short_name, full_name, guid_str, resolution)

            return list(self.cameras.keys())

        except Exception as e:
            logger.error("Ошибка при поиске камер: %s", e, exc_info=True)
            return []

    def _detect_max_resolution(self, camera):
        """
        Определяет максимальное разрешение сенсора по доступным режимам.

        Приоритет: FORMAT7_0 (max_image_size) → лучший Y16/MONO16.

        Returns:
            (width, height)
        """
        best_w, best_h = 0, 0

        try:
            all_modes = list(camera.modes)
        except Exception:
            return (0, 0)

        fmt7 = [m for m in all_modes if 'FORMAT7_0' in str(m)]
        if fmt7:
            mode = fmt7[0]
            for attr in ('max_image_size', 'image_size'):
                try:
                    val = getattr(mode, attr)
                    if val and len(val) == 2 and val[0] > 0:
                        best_w, best_h = int(val[0]), int(val[1])
                        logger.debug("FORMAT7_0 %s: %dx%d", attr, best_w, best_h)
                        break
                except Exception:
                    pass

        for m in all_modes:
            if _is_mono16(m):
                w, h = _get_image_size(m)
                if w * h > best_w * best_h:
                    best_w, best_h = w, h

        return (best_w, best_h)

    # ── Информация о камере ───────────────────────────────────────────────────

    def get_camera_info(self, camera_name=None):
        """
        Возвращает информацию о камере.

        Args:
            camera_name: Короткое имя камеры. Если None — текущая подключённая.

        Returns:
            Словарь с информацией или None
        """
        if camera_name is None:
            camera_name = self.camera_name
        if camera_name is None:
            return None
        return self.cameras.get(camera_name)

    # ── Диапазоны и текущие значения параметров экспозиции ───────────────────

    def get_exposure_info(self):
        """
        Возвращает диапазоны и текущие значения для shutter, gain, brightness.

        Returns:
            dict вида::

                {
                    'shutter':    {'range': (lo, hi), 'value': current},
                    'gain':       {'range': (lo, hi), 'value': current},
                    'brightness': {'range': (lo, hi), 'value': current},
                }

            Для недоступной камеры возвращает ``None``.
        """
        if not self.is_connected or self.camera is None:
            return None

        result = {}
        for name in ('shutter', 'gain', 'brightness'):
            try:
                feat = getattr(self.camera, name)
                lo, hi = feat.value_range
                result[name] = {
                    'range': (int(lo), int(hi)),
                    'value': int(feat.val),
                }
            except Exception as e:
                logger.debug("get_exposure_info: %s not available — %s", name, e)
                # Provide safe fallback so the UI always gets usable data
                fallbacks = {
                    'shutter':    {'range': (0, 863),  'value': self.DEFAULT_EXPOSURE['shutter']},
                    'gain':       {'range': (0, 683),  'value': self.DEFAULT_EXPOSURE['gain']},
                    'brightness': {'range': (0, 255),  'value': self.DEFAULT_EXPOSURE['brightness']},
                }
                result[name] = fallbacks[name]
        return result

    def get_feature_value(self, name):
        """
        Возвращает текущее raw-значение параметра камеры или None.

        Args:
            name: имя параметра ('shutter', 'gain', 'brightness', …)
        """
        if not self.is_connected or self.camera is None:
            return None
        try:
            return int(getattr(self.camera, name).val)
        except Exception:
            return None

    # ── Подключение / отключение ──────────────────────────────────────────────

    def connect_to_camera(self, camera_name):
        """
        Подключается к камере: выбирает максимальный mono16-режим,
        настраивает экспозицию и триггер, запускает поток.

        Args:
            camera_name: Короткое имя из get_camera_list()

        Returns:
            True если успешно
        """
        if not CAMERA_AVAILABLE:
            logger.error("pydc1394 не установлен")
            return False

        if camera_name not in self.cameras:
            logger.error("Камера '%s' не найдена. Вызовите get_camera_list() сначала.",
                         camera_name)
            return False

        try:
            with self.lock:
                if self.camera is not None:
                    self._stop_and_cleanup()

                cam_info = self.cameras[camera_name]
                cam_guid = cam_info["guid"]

                logger.info("Подключение к камере '%s' (GUID: %s)", camera_name, cam_guid)

                self.context = Context()
                self.camera = Camera(guid=cam_guid)

                width, height = self._select_and_apply_mode()

                if width == 0 or height == 0:
                    logger.error("Не удалось определить разрешение камеры")
                    self._stop_and_cleanup()
                    return False

                self._width = width
                self._height = height

                cam_info["resolution"] = (width, height)
                pixel_size = self.PIXEL_SIZE_BY_RESOLUTION.get(
                    (width, height), self.DEFAULT_PIXEL_SIZE)
                cam_info["pixel_size_x"] = pixel_size
                cam_info["pixel_size_y"] = pixel_size
                self._pixel_size_x = pixel_size
                self._pixel_size_y = pixel_size

                self._configure_exposure()

                # Триггер настраивается ДО старта захвата — требование IIDC
                self._configure_trigger()

                time.sleep(0.3)

                self.camera.start_capture(bufsize=self.CAPTURE_BUFSIZE)
                self.camera.start_video()
                self._video_started = True
                time.sleep(0.5)

                # При внешнем триггере прогрев пропускается:
                # камера не генерирует кадры без входного сигнала
                if not self.use_trigger:
                    self._flush_warmup_frames()
                    self._detect_byte_order()
                else:
                    logger.info(
                        "Режим внешнего триггера: прогрев и автоопределение "
                        "byte order пропущены. Byte order: big-endian (Flea2 default)."
                    )
                    self._byte_order = '>u2'

                self.camera_name = camera_name
                self.is_connected = True

                logger.info("Камера '%s' подключена: %dx%d, trigger=%s, byte_order=%s",
                            camera_name, width, height, self.use_trigger, self._byte_order)
                return True

        except Exception as e:
            logger.error("Ошибка при подключении к '%s': %s", camera_name, e, exc_info=True)
            self._stop_and_cleanup()
            return False

    def disconnect_camera(self):
        """
        Отключается от текущей камеры.

        Returns:
            True если успешно
        """
        try:
            with self.lock:
                name = self.camera_name
                self._stop_and_cleanup()
                self.camera_name = None
                self.is_connected = False
                logger.info("Камера '%s' отключена", name)
                return True
        except Exception as e:
            logger.error("Ошибка при отключении: %s", e, exc_info=True)
            return False

    def _stop_and_cleanup(self):
        """Останавливает захват и освобождает ресурсы"""
        if self.camera is not None:
            try:
                if self._video_started:
                    self.camera.stop_video()
                    self._video_started = False
                self.camera.stop_capture()
            except Exception as e:
                logger.warning("Ошибка при остановке камеры: %s", e)
            try:
                del self.camera
            except Exception:
                pass
            self.camera = None

        if self.context is not None:
            try:
                del self.context
            except Exception:
                pass
            self.context = None

    # ── Выбор видеорежима ─────────────────────────────────────────────────────

    def _select_and_apply_mode(self):
        """
        Выбирает режим с максимальным разрешением в mono16.

        Приоритет: FORMAT7_0 → лучший Y16/MONO16.

        Returns:
            (width, height)
        """
        all_modes = list(self.camera.modes)
        logger.info("Доступные видеорежимы: %s", [str(m) for m in all_modes])

        if self.use_format7:
            fmt7 = [m for m in all_modes if 'FORMAT7_0' in str(m)]
            if fmt7:
                mode = fmt7[0]
                self.camera.mode = mode

                width, height = 0, 0
                for attr in ('max_image_size', 'image_size'):
                    try:
                        val = getattr(mode, attr)
                        if val and len(val) == 2 and val[0] > 0:
                            width, height = int(val[0]), int(val[1])
                            logger.info("FORMAT7_0 %s: %dx%d", attr, width, height)
                            break
                    except Exception:
                        pass

                if width > 0 and height > 0:
                    self._configure_format7(mode, width, height)
                    logger.info("Выбран FORMAT7_0: %dx%d (полный сенсор)", width, height)
                    return width, height
                else:
                    logger.warning("FORMAT7_0 найден, но размер не определён — fallback на Y16")

        y16_modes = [m for m in all_modes if _is_mono16(m)]
        if not y16_modes:
            logger.error("Нет доступных Y16/MONO16 режимов")
            return 0, 0

        y16_modes.sort(key=_mode_resolution, reverse=True)
        best_mode = y16_modes[0]
        self.camera.mode = best_mode
        width, height = _get_image_size(best_mode)

        logger.info("Выбран режим %s: %dx%d", best_mode, width, height)
        return width, height

    def _configure_format7(self, mode, width, height):
        """Настраивает FORMAT7: Y16, полный ROI, максимальный packet_size"""
        try:
            mode.color_coding = 'Y16'
            logger.info("  FORMAT7 color coding: Y16")
        except Exception as e:
            logger.warning("  FORMAT7 color coding: не удалось (%s)", e)

        try:
            mode.image_position = (0, 0)
            mode.image_size = (width, height)
            logger.info("  FORMAT7 ROI: 0,0 + %dx%d", width, height)
        except Exception as e:
            logger.warning("  FORMAT7 ROI: не удалось (%s)", e)

        try:
            ps_range = None
            for attr in ('packet_size_range', 'packet_size_info', 'packet_per_frame_range'):
                try:
                    ps_range = getattr(mode, attr)
                    break
                except AttributeError:
                    pass

            if ps_range is not None:
                lo, hi = ps_range
                mode.packet_size = hi
                logger.info("  FORMAT7 packet size: %s (max)", hi)
            else:
                try:
                    rec = mode.recommended_packet_size
                    mode.packet_size = rec
                    logger.info("  FORMAT7 packet size: %s (recommended)", rec)
                except Exception:
                    logger.info("  FORMAT7 packet size: default")
        except Exception as e:
            logger.warning("  FORMAT7 packet size: ошибка (%s)", e)

    # ── Экспозиция ────────────────────────────────────────────────────────────

    def _configure_exposure(self):
        if self.manual_exposure:
            logger.info("Настройка ручной экспозиции")
            _apply_feature(self.camera, 'exposure',   self.DEFAULT_EXPOSURE["exposure"],   mode='manual')
            _apply_feature(self.camera, 'shutter',    self.DEFAULT_EXPOSURE["shutter"],    mode='manual')
            _apply_feature(self.camera, 'gain',       self.DEFAULT_EXPOSURE["gain"],       mode='manual')
            _apply_feature(self.camera, 'brightness', self.DEFAULT_EXPOSURE["brightness"], mode='manual')
        else:
            logger.info("Автоматическая экспозиция")
            _apply_feature(self.camera, 'exposure', 0, mode='auto')
            _apply_feature(self.camera, 'shutter',  0, mode='auto')
            _apply_feature(self.camera, 'gain',     0, mode='auto')

    def set_exposure(self, shutter=None, gain=None, brightness=None, exposure=None):
        """
        Устанавливает параметры экспозиции камеры в реальном времени.

        Args:
            shutter:    raw-значение выдержки  (0 … 863 для Flea2)
            gain:       raw-значение усиления  (0 … 683)
            brightness: raw-значение яркости   (0 … 255)
            exposure:   raw-значение exposure (EV, авто-экспозиция)
        """
        if not self.is_connected or self.camera is None:
            return
        if shutter is not None:
            _apply_feature(self.camera, 'shutter',    shutter,    mode='manual')
        if gain is not None:
            _apply_feature(self.camera, 'gain',       gain,       mode='manual')
        if brightness is not None:
            _apply_feature(self.camera, 'brightness', brightness, mode='manual')
        if exposure is not None:
            _apply_feature(self.camera, 'exposure',   exposure,   mode='manual')

    # ── Триггер ───────────────────────────────────────────────────────────────

    def _configure_trigger(self):
        """
        Настраивает триггер по стандарту IIDC 1394 числовыми константами.

        Flea2 FL2-08S2M поддерживает:
          Mode 0  — стандартный: одиночный кадр на фронт/спад сигнала
          Mode 1  — bulk: захват N кадров за один сигнал
          Mode 3  — skip: пропуск N кадров между сигналами
          Mode 14 — overlap: разрешает перекрытие экспозиции и readout
          Mode 15 — multi-shot: захват N кадров с интервалом

        Полярность: 1 = активный HIGH (RISING), 0 = активный LOW (FALLING).
        Источник 0 соответствует GPIO pin 0 (разъём GPIO Flea2).
        """
        try:
            trig = self.camera.trigger
        except AttributeError:
            logger.warning("Объект trigger недоступен на этой камере")
            return

        if self.use_trigger:
            try:
                trig.mode     = self.TRIGGER_MODE
                trig.source   = self.TRIGGER_SOURCE
                trig.polarity = self.TRIGGER_POLARITY
                trig.active   = True
                logger.info(
                    "Внешний триггер активирован: mode=%d, source=%d, polarity=%d",
                    self.TRIGGER_MODE, self.TRIGGER_SOURCE, self.TRIGGER_POLARITY
                )
            except Exception as e:
                logger.warning("Ошибка активации триггера: %s", e)
        else:
            try:
                trig.active = False
                logger.info("Внешний триггер деактивирован")
            except Exception as e:
                logger.warning("Ошибка деактивации триггера: %s", e)

    def set_trigger_mode(self, use_trigger):
        """
        Переключает режим триггера на лету.

        Согласно требованиям IIDC 1394 изменение параметров триггера
        требует остановки и повторного запуска видеопотока.

        Args:
            use_trigger: True — внешний триггер, False — свободный захват
        """
        self.use_trigger = use_trigger

        if not (self.is_connected and self.camera is not None):
            return True

        with self.lock:
            try:
                # Останавливаем поток перед изменением параметров триггера
                if self._video_started:
                    self.camera.stop_video()
                    self._video_started = False
                    logger.info("Видеопоток остановлен для изменения триггера")

                self._configure_trigger()

                # Перезапускаем поток
                self.camera.start_video()
                self._video_started = True
                time.sleep(0.3)
                logger.info("Видеопоток перезапущен, use_trigger=%s", use_trigger)

            except Exception as e:
                logger.error("Ошибка при переключении триггера: %s", e, exc_info=True)
                return False

        return True

    def set_trigger_parameters(self, mode=None, source=None, polarity=None):
        """
        Задаёт параметры внешнего триггера и применяет их если камера подключена.

        Args:
            mode:     Режим триггера IIDC (0, 1, 3, 14, 15)
            source:   Источник сигнала — номер GPIO pin (0–3 для Flea2)
            polarity: Полярность: 1=HIGH активный, 0=LOW активный
        """
        if mode is not None:
            self.TRIGGER_MODE = mode
        if source is not None:
            self.TRIGGER_SOURCE = source
        if polarity is not None:
            self.TRIGGER_POLARITY = polarity

        logger.info(
            "Параметры триггера обновлены: mode=%d, source=%d, polarity=%d",
            self.TRIGGER_MODE, self.TRIGGER_SOURCE, self.TRIGGER_POLARITY
        )

        # Если камера работает в режиме триггера — применяем немедленно
        if self.is_connected and self.use_trigger:
            self.set_trigger_mode(use_trigger=True)

    # ── Прогрев и byte order ──────────────────────────────────────────────────

    def _flush_warmup_frames(self):
        """Сбрасывает первые кадры после старта (только в режиме свободного захвата)"""
        logger.info("Сброс %d прогревочных кадров...", self.WARMUP_FRAMES)
        for _ in range(self.WARMUP_FRAMES):
            frame = self.camera.dequeue(poll=False)
            if frame is not None:
                frame.enqueue()

    def _detect_byte_order(self):
        """
        Определяет порядок байт по первому кадру: сравнивает число
        насыщенных пикселей при BE и LE интерпретации.

        Вызывается только в режиме свободного захвата (не триггерном).
        Для Flea2 результат, как правило, всегда big-endian.
        """
        frame = self._dequeue_with_retry(use_trigger_timeout=False)
        if frame is None:
            logger.warning("Не удалось получить кадр для определения byte order — "
                           "используется big-endian по умолчанию")
            self._byte_order = '>u2'
            return

        raw = bytes(frame)
        frame.enqueue()
        n_pixels = len(raw) // 2

        if n_pixels == 0:
            return

        w, h = self._width, self._height
        if w * h != n_pixels:
            logger.warning("Размер буфера (%d пикс) != %dx%d, пересчёт", n_pixels, w, h)
            if n_pixels > 0:
                side = int(math.sqrt(n_pixels))
                w = side
                h = n_pixels // side
                self._width = w
                self._height = h

        arr_be = np.frombuffer(raw, dtype='>u2').reshape((h, w))
        arr_le = np.frombuffer(raw, dtype='<u2').reshape((h, w))
        sat_be = int(np.sum(arr_be >= 65520))
        sat_le = int(np.sum(arr_le >= 65520))

        if sat_le < sat_be:
            self._byte_order = '<u2'
            logger.info("Byte order: little-endian")
        else:
            self._byte_order = '>u2'
            logger.info("Byte order: big-endian")

    # ── Захват кадров ─────────────────────────────────────────────────────────

    def _dequeue_with_retry(self, use_trigger_timeout=None):
        """
        Пытается получить кадр из буфера камеры.

        Args:
            use_trigger_timeout: Если True — использует увеличенный таймаут
                для режима внешнего триггера. Если None — берёт из self.use_trigger.

        Returns:
            Объект кадра или None при таймауте
        """
        if use_trigger_timeout is None:
            use_trigger_timeout = self.use_trigger

        if use_trigger_timeout:
            attempts = self.TRIGGER_DEQUEUE_ATTEMPTS
            delay    = self.TRIGGER_DEQUEUE_DELAY
        else:
            attempts = self.DEQUEUE_ATTEMPTS
            delay    = self.DEQUEUE_DELAY

        for _ in range(attempts):
            frame = self.camera.dequeue(poll=False)
            if frame is not None:
                return frame
            time.sleep(delay)

        return None

    def _decode_frame(self, raw_bytes):
        n_pixels = len(raw_bytes) // 2
        w, h = self._width, self._height

        if w * h != n_pixels:
            logger.warning("Несоответствие: буфер=%d пикс, ожидается %dx%d=%d",
                           n_pixels, w, h, w * h)
            if n_pixels > 0:
                side = int(math.sqrt(n_pixels))
                w = side
                h = n_pixels // side
            else:
                return None

        try:
            arr = np.frombuffer(raw_bytes, dtype=self._byte_order).reshape((h, w))
            return arr.copy()
        except Exception as e:
            logger.error("Ошибка декодирования кадра: %s", e)
            return None

    def capture_single_frame(self):
        """
        Захватывает один кадр.

        В режиме внешнего триггера ожидает сигнала до 10 секунд
        (TRIGGER_DEQUEUE_ATTEMPTS × TRIGGER_DEQUEUE_DELAY).

        Returns:
            numpy array (height, width) dtype=uint16
            None при таймауте ожидания триггерного сигнала
            None при ошибке
        """
        if not self.is_connected or self.camera is None:
            logger.error("capture_single_frame: камера не подключена")
            return None

        try:
            with self.lock:
                frame = self._dequeue_with_retry()

                if frame is None:
                    if self.use_trigger:
                        logger.warning(
                            "Нет кадра: внешний триггерный сигнал не получен "
                            "за %.1f с (attempts=%d)",
                            self.TRIGGER_DEQUEUE_ATTEMPTS * self.TRIGGER_DEQUEUE_DELAY,
                            self.TRIGGER_DEQUEUE_ATTEMPTS
                        )
                    else:
                        logger.warning("Нет кадра: таймаут ожидания (камера не отвечает)")
                    return None

                raw = bytes(frame)
                frame.enqueue()

                arr = self._decode_frame(raw)
                if arr is not None:
                    logger.debug("Кадр захвачен: %s, min=%d, max=%d",
                                 arr.shape, arr.min(), arr.max())
                return arr

        except Exception as e:
            logger.error("Ошибка при захвате кадра: %s", e, exc_info=True)
            return None

    def capture_background_frames(self, num_frames=40):
        """
        Захватывает num_frames кадров для усреднения фона.

        В режиме внешнего триггера каждый кадр ожидается отдельно
        с увеличенным таймаутом.

        Returns:
            numpy array (num_frames, height, width) dtype=float64 или None
        """
        if not self.is_connected or self.camera is None:
            return None

        frames = []
        try:
            logger.info("Сбор %d фоновых кадров (trigger=%s)...",
                        num_frames, self.use_trigger)
            captured = 0
            misses = 0
            max_misses = num_frames * 3  # не более 3 пропусков на каждый ожидаемый кадр

            while captured < num_frames:
                with self.lock:
                    frame = self._dequeue_with_retry()
                    if frame is None:
                        misses += 1
                        if self.use_trigger:
                            logger.warning(
                                "Кадр %d/%d: триггерный сигнал не получен "
                                "(пропуск %d/%d)",
                                captured + 1, num_frames, misses, max_misses
                            )
                        else:
                            logger.warning(
                                "Пропущен кадр %d/%d (пропуск %d/%d)",
                                captured + 1, num_frames, misses, max_misses
                            )
                        if misses >= max_misses:
                            logger.error(
                                "Превышен лимит пропусков (%d) — "
                                "прерывание сбора фона. Собрано %d/%d кадров.",
                                max_misses, captured, num_frames
                            )
                            break
                        continue
                    raw = bytes(frame)
                    frame.enqueue()

                arr = self._decode_frame(raw)
                if arr is not None:
                    frames.append(arr.astype(np.float64))
                    captured += 1
                    logger.debug("Фоновый кадр %d/%d", captured, num_frames)

            if frames:
                result = np.array(frames)
                logger.info("Фон собран: %d кадров, shape=%s", len(frames), result.shape)
                return result

            logger.warning("Не удалось собрать фоновые кадры")
            return None

        except Exception as e:
            logger.error("Ошибка при сборе фона: %s", e, exc_info=True)
            if frames:
                return np.array(frames)
            return None

    # ── Свойства ──────────────────────────────────────────────────────────────

    @property
    def resolution(self):
        return (self._width, self._height)

    @property
    def pixel_size_x(self):
        return self._pixel_size_x

    @property
    def pixel_size_y(self):
        return self._pixel_size_y