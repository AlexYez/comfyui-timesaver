"""Общее для всех загрузчиков `.tsmodel`: где искать файлы и как их открывать.

Формат живёт в [`_mclock.py`](_mclock.py) и о ComfyUI не знает; здесь наоборот —
только связь с ComfyUI: папки моделей, список для выпадашки и одна точка входа
в чтение.

⚠️ `.tsmodel` НЕ регистрируется в `folder_paths.supported_pt_extensions`, и это
осознанно: штатные загрузчики не должны предлагать файл, который они всё равно
не откроют.
"""

from __future__ import annotations

import logging
import os
import time

import folder_paths

from ._mclock import EXT, LOG_PREFIX, LockedModelError, load_state_dict_comfy

logger = logging.getLogger("comfyui_timesaver.ts_locked_loaders")

#: Папка моделей ComfyUI под каждый вид загрузчика.
CATEGORIES = ("diffusion_models", "text_encoders", "checkpoints", "loras", "vae")

NOTHING_FOUND = "(no .tsmodel files found)"

#: ⚠️ Список строится на КАЖДЫЙ запрос `/object_info`, а нод шесть — без кэша это
#: тридцать обходов дерева моделей на одну загрузку страницы. Держим результат
#: пару секунд: этого хватает, чтобы обход случился один раз на запрос, и мало,
#: чтобы новый файл появился в списке практически сразу.
_CACHE_SECONDS = 2.0
_cache: dict[str, tuple[float, list[str]]] = {}


def folders(category: str) -> list[str]:
    """Все папки ComfyUI под этот вид моделей (включая extra_model_paths)."""
    try:
        return list(folder_paths.get_folder_paths(category))
    except Exception:                               # noqa: BLE001 - чужая категория
        return []


def scan(category: str) -> list[str]:
    """Относительные имена всех `.tsmodel` в папках этого вида."""
    cached = _cache.get(category)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _CACHE_SECONDS:
        return cached[1]

    names: set[str] = set()
    for directory in folders(category):
        if not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(EXT):
                    relative = os.path.relpath(os.path.join(root, name), directory)
                    names.add(relative.replace("\\", "/"))
    result = sorted(names)
    _cache[category] = (now, result)
    return result


def options(category: str) -> list[str]:
    """Что показать в выпадашке — или почему она пустая."""
    return scan(category) or [NOTHING_FOUND]


def resolve(category: str, relative: str) -> str:
    """Относительное имя -> путь на диске."""
    if not relative or relative == NOTHING_FOUND:
        raise LockedModelError(
            f"{LOG_PREFIX} No .tsmodel file is selected. Put one into "
            f"models/{category} — the stock loaders do not list these files, "
            "this node does."
        )
    for directory in folders(category):
        candidate = os.path.join(directory, relative)
        if os.path.isfile(candidate):
            return candidate
    raise LockedModelError(
        f"{LOG_PREFIX} '{relative}' was not found in any models/{category} folder."
    )


def load(category: str, relative: str):
    """Открыть запертый файл: ``(state_dict, metadata, путь)``."""
    path = resolve(category, relative)
    state, metadata = load_state_dict_comfy(path)
    logger.info("%s opened %s (%d tensors)", LOG_PREFIX, os.path.basename(path), len(state))
    return state, metadata, path
