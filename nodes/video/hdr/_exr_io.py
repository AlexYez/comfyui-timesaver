"""Чтение и запись EXR — единственное место в паке, которое знает про этот формат.

Живёт в ``hdr/``, но нужно и сохранятелю видео: у него появился выход
«секвенция EXR». Направление зависимости одно и то же всегда —
``video/media`` импортирует отсюда, обратно никогда.

⚠️ **Обычный загрузчик картинок для EXR не годится.** Он приводит кадр к
восьми битам и к диапазону ``[0, 1]``; именно то, что делает EXR ценным —
значения выше единицы — при этом теряется молча, без единого предупреждения.

Бэкенды и почему их три:

- **OpenImageIO** — то, чем читает официальный pipeline Lightricks. Даёт имена
  каналов и метаданные. В сборке пользователя обычно не стоит.
- **PyAV** — уже есть в любой ComfyUI (им пишет встроенная Save Image
  (Advanced)) и **не требует ничего настраивать**. Поэтому он и выбирается по
  умолчанию, когда OpenImageIO нет. Читаем через него сырые плоскости кадра —
  подробности и две причины в :func:`_try_pyav`; коротко: и half-float, и
  float32 приходят точными.
- **OpenCV** — умеет EXR только если переменная ``OPENCV_IO_ENABLE_OPENEXR=1``
  была выставлена **до** импорта ``cv2``. Выставить её позже нельзя: плагины
  чтения регистрируются один раз при импорте, а к моменту загрузки пака ``cv2``
  обычно уже импортирован кем-то из соседей (замерено: это делает и
  ``comfy_extras/nodes_sdpose.py``). ⚠️ Замерено: возвращает **BGR**.
  Держим третьим бэкендом, а не рассчитываем на него.

Сеть не читаем: путь обязан быть локальным файлом.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("comfyui_timesaver.ts_hdr.exr")
LOG_PREFIX = "[TS HDR EXR]"

BACKENDS = ("auto", "OpenImageIO", "PyAV", "OpenCV")

# Потолок на размер кадра. 8192×8192 это 67 Мпикс; берём с запасом, чтобы
# случайно выбранный гигантский файл не съел всю память молча.
DEFAULT_MAX_PIXELS = 80_000_000


class ExrError(RuntimeError):
    """Не смогли прочитать или записать EXR — с внятным объяснением."""


# ─────────────────────────────── чтение ────────────────────────────────

def _check_path(path: str | os.PathLike[str]) -> Path:
    text = str(path)
    if "://" in text:
        raise ExrError(f"{LOG_PREFIX} Remote paths are not read: {text!r}.")
    resolved = Path(text).expanduser()
    if not resolved.is_file():
        raise ExrError(f"{LOG_PREFIX} No such file: {resolved}.")
    return resolved


def _try_openimageio(path: Path) -> tuple[Any, list[str]] | None:
    try:
        import OpenImageIO as oiio                       # noqa: N813
    except Exception:                                    # noqa: BLE001 - его обычно нет
        return None
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ExrError(f"{LOG_PREFIX} OpenImageIO could not open {path.name}: {oiio.geterror()}")
    try:
        spec = source.spec()
        pixels = source.read_image(format="float")
        names = list(spec.channelnames)
    finally:
        source.close()
    if pixels is None:
        raise ExrError(f"{LOG_PREFIX} OpenImageIO read no pixels from {path.name}.")
    import numpy as np

    array = np.asarray(pixels, dtype="float32")
    if array.ndim == 2:
        array = array[..., None]
    return array, names


# Раскладки кадра, которые мы читаем плоскостями напрямую.
# Значение: (тип элемента, порядок плоскостей, имена каналов).
# ⚠️ «gbr» в названии — не украшение: плоскости лежат в порядке G, B, R, и
# индексы ниже разворачивают их в R, G, B (замерено).
_PYAV_PLANAR = {
    "gbrpf32le": ("float32", (2, 0, 1), ["R", "G", "B"]),
    "gbrpf16le": ("float16", (2, 0, 1), ["R", "G", "B"]),
    "gbrapf32le": ("float32", (2, 0, 1, 3), ["R", "G", "B", "A"]),
    "gbrapf16le": ("float16", (2, 0, 1, 3), ["R", "G", "B", "A"]),
    "grayf32le": ("float32", (0,), ["Y"]),
    "grayf16le": ("float16", (0,), ["Y"]),
}


def _try_pyav(path: Path) -> tuple[Any, list[str]] | None:
    """Прочитать EXR через PyAV, разбирая плоскости кадра руками.

    ⚠️ **Ни ``to_ndarray`` с чужим форматом, ни любая другая конверсия.**
    Замерено дважды:

    - конверсия идёт через swscale, а он зажимает float в ``[0, 1]``:
      четырёхканальный EXR со значением 4.0, прочитанный «в три канала»,
      возвращает 1.0 — тот самый скрытый clamp, который убивает HDR молча;
    - для half-float (``gbrpf16le``) PyAV вообще отвечает «Conversion to numpy
      array … is not yet supported», а half — это ровно то, в чём рендерят
      почти все.

    Поэтому берём сырые буферы плоскостей: копия байт, никаких преобразований.
    """
    try:
        import av
        import numpy as np
    except Exception:                                    # noqa: BLE001
        return None

    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        native = frame.format.name
        layout = _PYAV_PLANAR.get(native)
        if layout is None:
            raise ExrError(
                f"{LOG_PREFIX} PyAV decoded {path.name} as '{native}', which is not a "
                "known float layout. Converting it would clamp the range."
            )
        dtype_name, order, names = layout
        item = np.dtype(dtype_name).itemsize
        width, height = int(frame.width), int(frame.height)

        planes = []
        for plane in frame.planes:
            # Строка в буфере длиннее кадра: ffmpeg выравнивает её. Режем по
            # настоящей ширине, иначе справа приедет мусор выравнивания.
            stride = int(plane.line_size) // item
            raw = np.frombuffer(bytes(plane), dtype=dtype_name)
            planes.append(raw.reshape(-1, stride)[:height, :width])

    if len(planes) < len(order):
        raise ExrError(
            f"{LOG_PREFIX} {path.name} decoded to {len(planes)} planes, expected "
            f"{len(order)} for '{native}'.")
    array = np.stack([planes[index] for index in order], axis=-1).astype("float32")
    return array, list(names)


def _try_opencv(path: Path) -> tuple[Any, list[str]] | None:
    try:
        import cv2
    except Exception:                                    # noqa: BLE001
        return None
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ExrError(
            f"{LOG_PREFIX} OpenCV returned nothing for {path.name}. Its EXR reader is "
            "off unless OPENCV_IO_ENABLE_OPENEXR=1 is set BEFORE ComfyUI starts — "
            "setting it later has no effect, because the reader registers at import."
        )
    import numpy as np

    if array.dtype not in (np.float32, np.float16):
        raise ExrError(
            f"{LOG_PREFIX} OpenCV read {path.name} as {array.dtype}, not float: the "
            "value range would already be lost. Use the PyAV backend instead."
        )
    array = np.asarray(array, dtype="float32")
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] >= 3:
        # Замерено: OpenCV отдаёт BGR(A).
        rgb = array[..., 2::-1]
        array = np.concatenate([rgb, array[..., 3:]], axis=-1) if array.shape[-1] > 3 else rgb
    return array, ["R", "G", "B", "A"][: array.shape[-1]]


_READERS = {
    "OpenImageIO": _try_openimageio,
    "PyAV": _try_pyav,
    "OpenCV": _try_opencv,
}

# Порядок «auto»: сначала то, чем читает официальный pipeline, затем то, что
# работает без настройки, и только потом то, что требует переменной окружения.
_AUTO_ORDER = ("OpenImageIO", "PyAV", "OpenCV")


def available_backends() -> tuple[str, ...]:
    """Какие бэкенды вообще импортируются в этом окружении."""
    found = []
    for name, module in (("OpenImageIO", "OpenImageIO"), ("PyAV", "av"), ("OpenCV", "cv2")):
        try:
            __import__(module)
        except Exception:                                # noqa: BLE001
            continue
        found.append(name)
    return tuple(found)


def load_exr(
    path: str | os.PathLike[str],
    *,
    backend: str = "auto",
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[torch.Tensor, dict]:
    """Прочитать EXR как линейный float32 без нормализации и без зажима сверху.

    Args:
        path: локальный файл.
        backend: один из :data:`BACKENDS`.
        max_pixels: потолок на ``width * height``.

    Returns:
        Пара ``(тензор [1, H, W, 3] float32, метаданные)``. Альфа и лишние
        каналы в v1 отбрасываются — но их имена попадают в метаданные, так что
        видно, что именно отброшено.

    Raises:
        ExrError: файла нет, ни один бэкенд не смог, кадр слишком большой.
    """
    resolved = _check_path(path)
    if backend not in BACKENDS:
        raise ExrError(f"{LOG_PREFIX} Unknown backend '{backend}'. Available: {', '.join(BACKENDS)}.")

    order = _AUTO_ORDER if backend == "auto" else (backend,)
    array, names, used, failures = None, [], "", []
    for candidate in order:
        try:
            result = _READERS[candidate](resolved)
        except ExrError as error:
            failures.append(str(error))
            continue
        except Exception as error:                       # noqa: BLE001 - бэкенд не смог
            failures.append(f"{candidate}: {type(error).__name__}: {error}")
            continue
        if result is None:
            failures.append(f"{candidate}: not installed")
            continue
        array, names = result
        used = candidate
        break

    if array is None:
        detail = "; ".join(failures) or "no backend installed"
        raise ExrError(
            f"{LOG_PREFIX} Could not read {resolved.name}. Tried: {detail}. "
            "PyAV ships with ComfyUI and needs no setup — if it failed, the file is "
            "likely not a valid EXR."
        )

    height, width = int(array.shape[0]), int(array.shape[1])
    if height * width > int(max_pixels):
        raise ExrError(
            f"{LOG_PREFIX} {resolved.name} is {width}x{height} = {height * width} pixels, "
            f"over the {max_pixels} limit."
        )

    tensor = torch.from_numpy(array.copy()).to(torch.float32)
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
    dropped = list(names[3:]) if len(names) > 3 else []
    tensor = tensor[..., :3].unsqueeze(0).contiguous()

    finite = torch.isfinite(tensor)
    bad = int((~finite).sum().item())
    clean = tensor[finite]
    meta = {
        "path": str(resolved),
        "name": resolved.name,
        "backend": used,
        "width": width,
        "height": height,
        "channels": list(names),
        "dropped_channels": dropped,
        "min": float(clean.min().item()) if clean.numel() else 0.0,
        "max": float(clean.max().item()) if clean.numel() else 0.0,
        "non_finite": bad,
        "above_one": int((tensor > 1.0).sum().item()),
    }
    logger.debug("%s read %s via %s: %dx%d, max %.4g",
                 LOG_PREFIX, resolved.name, used, width, height, meta["max"])
    return tensor, meta


# ─────────────────────────────── запись ────────────────────────────────

def encode_exr_frame(frame: torch.Tensor, *, half: bool = False) -> bytes:
    """Закодировать один кадр ``[H, W, 3]`` в байты EXR.

    Через PyAV, тем же кодеком, которым пишет встроенная Save Image (Advanced).
    Проверено побитовым круговым прогоном: ``65000.0`` и отрицательные значения
    доезжают до диска и обратно без изменений — ни зажима, ни нормализации.

    Данные записываются как есть, поэтому вызывающий обязан подать **линейный
    свет в примариях Rec.709**: именно так договорено читать EXR по умолчанию.

    Args:
        frame: кадр ``[H, W, 3]``.
        half: писать 16-битными числами. Файл примерно вдвое легче, диапазон
            сохраняется (замерено: 100.0 остаётся 100.0, шаг квантования в
            районе сотни — около 0.06). Сжатия кодировщик ffmpeg не предлагает
            вовсе: список его опций пуст, а ``compression`` он отвергает.
    """
    try:
        import av
        import numpy as np
    except Exception as error:                           # noqa: BLE001
        raise ExrError(f"{LOG_PREFIX} PyAV is required to write EXR: {error}") from error

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ExrError(f"{LOG_PREFIX} EXR frame must be [H, W, 3], got {tuple(frame.shape)}.")

    array = np.ascontiguousarray(frame.detach().to(torch.float32).cpu().numpy())
    height, width = int(array.shape[0]), int(array.shape[1])

    codec = av.CodecContext.create("exr", "w")
    codec.width, codec.height = width, height
    codec.pix_fmt = "gbrpf32le"
    if half:
        codec.options = {"format": "half"}
    video_frame = av.VideoFrame.from_ndarray(array, format="gbrpf32le")
    packets = list(codec.encode(video_frame)) + list(codec.encode(None))
    if not packets:
        raise ExrError(f"{LOG_PREFIX} The EXR encoder produced no data for {width}x{height}.")
    return b"".join(bytes(packet) for packet in packets)


def write_exr(path: str | os.PathLike[str], frame: torch.Tensor, *, half: bool = False) -> int:
    """Записать один кадр в файл. Возвращает размер в байтах."""
    blob = encode_exr_frame(frame, half=half)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return len(blob)


def describe_range(tensor: torch.Tensor) -> str:
    """Короткая строка про диапазон — для логов и сообщений об ошибках."""
    finite = tensor[torch.isfinite(tensor)]
    if not finite.numel():
        return "no finite samples"
    low, high = float(finite.min().item()), float(finite.max().item())
    over = int((tensor > 1.0).sum().item())
    share = 100.0 * over / max(1, tensor.numel())
    if math.isnan(share):
        share = 0.0
    return f"min {low:.4g}, max {high:.4g}, {share:.2f}% above 1.0"
