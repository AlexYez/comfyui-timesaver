"""Общее для трёх видео-нод: пути, тип ``TS_VIDEO_INFO`` и работа с ним.

Здесь нет ни декодирования, ни кодирования — только то, что нужно всем троим и
не тянет за собой PyAV. Модуль обязан импортироваться дёшево: его читает и
загрузчик пака при обходе, и тесты без ComfyUI.

⚠️ ПУТИ. Нода читает видео из произвольного места на диске — часовые исходники
не таскают в ``input``, и это осознанное решение владельца пака. Действуют три
правила: расширение из белого списка, никакого листинга каталогов, и в лог путь
уходит только через ``safe_log_path``. Запись по-прежнему возможна только в
``output``/``temp``.

Отдельно — что показывать по HTTP: на локальном сервере (петля) ограничений нет,
на открытом наружу маршруты держатся домашней папки и каталогов ComfyUI. Правило
одно на весь пак, живёт в ``nodes/_shared.media_path_allowed``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from comfy_api.v0_0_2 import IO

from ..._shared import media_path_allowed as _media_path_allowed
from ..._shared import server_is_local_only  # noqa: F401 - переэкспорт для тестов

logger = logging.getLogger("comfyui_timesaver.ts_video")
LOG_PREFIX = "[TS Video]"

# Кастомный тип, связывающий загрузчик с нодой Video Info. Объявлен ОДИН раз:
# оба конца импортируют его отсюда (прецедент — nodes/image/studio/inpaint/_plan.py).
VIDEO_INFO_TYPE = IO.Custom("TS_VIDEO_INFO")

# Тип чужого пака VideoHelperSuite. Нужен ровно для одного: чтобы провод от его
# загрузчика подключался к нашей ноде Video Info. Свой выход мы так называть не
# станем — это был бы захват чужого контракта.
VHS_VIDEO_INFO_TYPE = IO.Custom("VHS_VIDEOINFO")

# Версия формы словаря. Ломающее изменение = +1; дописывание ключей версию не
# меняет, потому что читатель всегда спрашивает через info_get с дефолтом.
VIDEO_INFO_SCHEMA = 1

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg",
    ".m2ts", ".mts", ".ts", ".flv", ".wmv", ".ogv", ".gif",
})

_CACHE_DIRNAME = "ts_video"


def node_root() -> Path:
    """Корень пака — на четыре уровня выше этого файла."""
    return Path(__file__).resolve().parents[3]


def cache_dir() -> Path:
    """Куда складывать пробы, пики и спрайты. Каталог создаётся лениво."""
    return node_root() / ".cache" / _CACHE_DIRNAME


def safe_log_path(path: str | os.PathLike[str]) -> str:
    """Путь в виде, пригодном для лога.

    Схлопывает переводы строк (иначе имя файла подделывает строки лога) и
    оставляет только имя с родительской папкой — полный путь пользователя в лог
    не утекает (§15 CLAUDE.md).
    """
    text = str(path).replace("\r", " ").replace("\n", " ")
    tail = Path(text)
    parent = tail.parent.name
    return f"{parent}/{tail.name}" if parent else tail.name


def is_video_path(path: str | os.PathLike[str]) -> bool:
    """Похоже ли это на видеофайл по расширению."""
    return Path(str(path)).suffix.lower() in VIDEO_EXTENSIONS


# Политика «что маршруты вправе показать» — одна на весь пак,
# в `nodes/_shared`. Здесь только тонкая обёртка: у видео нет
# собственных папок сверх общих.
def http_path_allowed(path: str) -> bool:
    """Можно ли отдать ЭТОТ файл по HTTP.

    Локальный сервер (петля) — путь любой: ради этого произвольный путь и
    заведён, часовые исходники не таскают в ``input``. Открытый наружу —
    домашняя папка, каталоги ComfyUI и ``TS_MEDIA_EXTRA_ROOTS``. Подробности и
    переменные — в ``nodes/_shared.py``.
    """
    return _media_path_allowed(path)


def resolve_media_path(value: str) -> str:
    """Развернуть значение виджета в реальный путь на диске.

    Понимает три формы, потому что в поле приходят все три:
    аннотированное ``clip.mp4 [input]`` из штатной загрузки, путь относительно
    папки ``input`` и абсолютный путь куда угодно.

    Args:
        value: содержимое ``source_path``.

    Returns:
        Абсолютный путь. Существование НЕ проверяется — это дело вызывающего.
    """
    text = str(value or "").strip().strip('"')
    if not text:
        return ""

    try:
        import folder_paths

        if folder_paths.exists_annotated_filepath(text):
            return os.path.abspath(folder_paths.get_annotated_filepath(text))
        candidate = os.path.join(folder_paths.get_input_directory(), text)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    except Exception as error:              # noqa: BLE001 - вне ComfyUI и в тестах
        logger.debug("%s folder_paths unavailable: %s", LOG_PREFIX, error)

    return os.path.abspath(os.path.expanduser(text))


def file_identity(path: str | os.PathLike[str]) -> str:
    """Ключ кэша: путь + размер + время правки.

    Перезапись файла тем же именем обязана инвалидировать и кэш, и граф — иначе
    человек правит исходник и не понимает, почему ничего не изменилось.
    """
    real = os.path.abspath(str(path))
    try:
        stat = os.stat(real)
        token = f"{os.path.normcase(real)}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        token = os.path.normcase(real)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# video_info
# ──────────────────────────────────────────────────────────────────────────────

def empty_video_info() -> dict:
    """Пустой, но валидный по форме словарь.

    Возвращается вместо исключения, когда в ноду Video Info прилетело нечто
    непонятное: терять из-за этого весь прогон человек не подписывался.
    """
    return {
        "_schema": VIDEO_INFO_SCHEMA,
        "_producer": "unknown",
        "source": {},
        "loaded": {},
        "audio": None,
        "file": {},
    }


def info_get(info: Any, path: str, default: Any = None) -> Any:
    """Достать значение по пути вида ``"loaded.fps"``.

    ⚠️ ЕДИНСТВЕННЫЙ разрешённый способ читать этот словарь. Прямое
    ``info["loaded"]["fps"]`` сломается на словаре другой версии, чужой сборки
    или на обрывке; здесь недостающее просто становится дефолтом, поэтому
    читатель схемы 1 переживает словарь схемы 2 и наоборот.

    Args:
        info: словарь video_info (или что угодно).
        path: точечный путь.
        default: что вернуть, если по пути ничего нет.
    """
    current: Any = info
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return default
        if part not in current:
            return default
        current = current[part]
    return default if current is None else current


# Плоские ключи VideoHelperSuite: source_fps, loaded_frame_count и т.д.
_VHS_KEYS = ("fps", "frame_count", "duration", "width", "height")


def coerce_video_info(value: Any) -> dict:
    """Привести к нашей форме что угодно, что могло прийти на вход.

    Наш словарь возвращается как есть; ``VHS_VIDEOINFO`` переводится (у него
    ровно десять плоских ключей); всё остальное становится пустым словарём.
    """
    if isinstance(value, Mapping) and "_schema" in value:
        return dict(value)

    if isinstance(value, Mapping) and f"source_{_VHS_KEYS[0]}" in value:
        info = empty_video_info()
        info["_producer"] = "VHS"
        info["source"] = {key: value.get(f"source_{key}") for key in _VHS_KEYS}
        info["loaded"] = {key: value.get(f"loaded_{key}") for key in _VHS_KEYS}
        return info

    return empty_video_info()


def build_video_info(
    *,
    producer: str,
    source: Mapping[str, Any],
    loaded: Mapping[str, Any],
    audio: Mapping[str, Any] | None,
    filename: str,
    annotated: str,
) -> dict:
    """Собрать словарь video_info.

    Единственное место сборки: руками этот словарь не строит больше никто, иначе
    ключи начнут расходиться между нодами.

    ⚠️ Полный путь пользователя внутрь НЕ кладётся — словарь уезжает в
    сохранённый workflow, а тот ходит по рукам.
    """
    return {
        "_schema": VIDEO_INFO_SCHEMA,
        "_producer": producer,
        "source": dict(source),
        "loaded": dict(loaded),
        "audio": dict(audio) if audio else None,
        "file": {"name": filename, "annotated": annotated},
    }


def format_summary(info: Mapping[str, Any]) -> str:
    """Одна строка про ролик — для заметок, отладки и подписи в интерфейсе."""
    name = info_get(info, "file.name", "") or "video"
    width = int(info_get(info, "loaded.width", 0) or 0)
    height = int(info_get(info, "loaded.height", 0) or 0)
    fps = float(info_get(info, "loaded.fps", 0.0) or 0.0)
    frames = int(info_get(info, "loaded.frame_count", 0) or 0)
    duration = float(info_get(info, "loaded.duration", 0.0) or 0.0)
    src_w = int(info_get(info, "source.width", 0) or 0)
    src_h = int(info_get(info, "source.height", 0) or 0)
    src_fps = float(info_get(info, "source.fps", 0.0) or 0.0)

    parts = [name, f"{width}x{height}", f"{fps:g} fps", f"{frames} frames", f"{duration:.2f} s"]
    if (src_w, src_h) != (width, height) or abs(src_fps - fps) > 1e-6:
        parts.append(f"(source {src_w}x{src_h} {src_fps:g} fps)")
    return " · ".join(parts)
