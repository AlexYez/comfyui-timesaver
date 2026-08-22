"""Пресеты ракурсов для TS Angle Select: чтение, проверка, сборка промпта.

Пресет — обычный JSON в `nodes/text/angle_presets/`. Смысл именно в этом:
у каждой модели свой словарь, и новая модель не должна требовать правки кода.
Файл описывает, во что превращаются три положения камеры:

```json
{
  "name": "Qwen Multi-Angle",
  "template": "{trigger} {horizontal} {height} {framing}",
  "trigger": "<sks>",
  "horizontal": {"0": "front view", "45": "front-right quarter view", ...},
  "height":     {"low": "low-angle shot", ...},
  "framing":    {"wide": "wide shot", ...}
}
```

⚠️ Слова в пресете — это то, на чём модель ОБУЧАЛАСЬ, а не описание для
человека. У Qwen Multi-Angle это буквально триггерная фраза LoRA: перепишешь
формулировку «покрасивее» — обусловливание перестанет работать. Поэтому
словарь и вынесен в данные, где его видно целиком.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .._shared import make_route_registrars, resolve_prompt_server

logger = logging.getLogger("comfyui_timesaver.ts_angle_select")
LOG_PREFIX = "[TS Angle Select]"

PRESETS_DIR = Path(__file__).resolve().parent / "angle_presets"

# Положения камеры, которыми оперирует нода. Они СОЗНАТЕЛЬНО общечеловеческие,
# а не из словаря конкретной модели: сохранённый граф не должен разъезжаться
# при смене пресета, и «low-angle shot» в чужой модели может называться иначе.
AZIMUTHS: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
HEIGHTS: tuple[str, ...] = ("low", "eye-level", "elevated", "high")
FRAMINGS: tuple[str, ...] = ("wide", "medium", "close-up")

_REQUIRED = ("name", "template", "horizontal", "height", "framing")


def _validate(data: dict, source: Path) -> dict | None:
    """Пресет либо полный, либо не подключается вовсе.

    ⚠️ Половинчатый пресет хуже отсутствующего: нода собрала бы промпт с
    дырой на месте ракурса, и человек увидел бы это только по результату.
    """
    for key in _REQUIRED:
        if key not in data:
            logger.warning("%s %s has no '%s' — skipped.", LOG_PREFIX, source.name, key)
            return None
    horizontal = {str(k): str(v) for k, v in (data.get("horizontal") or {}).items()}
    missing = [str(a) for a in AZIMUTHS if str(a) not in horizontal]
    if missing:
        logger.warning("%s %s has no phrase for azimuth %s — skipped.",
                       LOG_PREFIX, source.name, ", ".join(missing))
        return None
    for key, allowed in (("height", HEIGHTS), ("framing", FRAMINGS)):
        table = {str(k): str(v) for k, v in (data.get(key) or {}).items()}
        gap = [name for name in allowed if name not in table]
        if gap:
            logger.warning("%s %s has no %s phrase for %s — skipped.",
                           LOG_PREFIX, source.name, key, ", ".join(gap))
            return None
        data[key] = table
    data["horizontal"] = horizontal
    data["trigger"] = str(data.get("trigger") or "")
    data["order"] = int(data.get("order") or 100)
    return data


def load_presets() -> dict[str, dict]:
    """Все пресеты из папки, по имени. Порядок — по полю `order`, затем имя."""
    found: list[dict] = []
    if PRESETS_DIR.is_dir():
        for path in sorted(PRESETS_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("%s Could not read %s: %s", LOG_PREFIX, path.name, exc)
                continue
            if not isinstance(data, dict):
                logger.warning("%s %s is not a JSON object — skipped.", LOG_PREFIX, path.name)
                continue
            checked = _validate(data, path)
            if checked is not None:
                found.append(checked)
    found.sort(key=lambda item: (item["order"], str(item["name"])))
    return {str(item["name"]): item for item in found}


def preset_names() -> list[str]:
    """Имена для виджета. Пустой список сломал бы схему, поэтому не бывает."""
    names = list(load_presets())
    return names or ["Qwen Multi-Angle"]


def build_prompt(preset: dict, azimuth: int, height: str, framing: str) -> str:
    """Собрать промпт по пресету и положению камеры.

    Значения приводятся к ближайшему допустимому: граф в API-формате может
    прислать что угодно, а промпт с пустым местом на месте ракурса — это тихо
    испорченный результат, а не ошибка, которую видно.
    """
    azimuth = snap_azimuth(azimuth)
    height = height if height in HEIGHTS else "eye-level"
    framing = framing if framing in FRAMINGS else "medium"
    parts = {
        "trigger": preset.get("trigger", ""),
        "horizontal": preset["horizontal"][str(azimuth)],
        "height": preset["height"][height],
        "framing": preset["framing"][framing],
    }
    text = str(preset["template"])
    for key, value in parts.items():
        text = text.replace("{" + key + "}", value)
    # Пустой триггер оставил бы двойной пробел в начале — модель этого не
    # заметит, а человек в логе заметит.
    return " ".join(text.split())


def snap_azimuth(azimuth: int) -> int:
    """Ближайший из восьми детентов, нормализованный в [0, 360)."""
    try:
        value = int(azimuth)
    except (TypeError, ValueError):
        return 0
    return int(round((value % 360) / 45.0) * 45) % 360


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
# ⚠️ Маршрута со словарём пресетов здесь БОЛЬШЕ НЕТ. Он был нужен, пока
# редактор показывал готовый промпт под сценой; строку убрали (она меняла
# высоту виджета и дёргала ноду под курсором), и словарь во фронтенде стал
# мёртвым грузом. Слова знает бэкенд — там, где они и применяются.

_PROMPT_SERVER = resolve_prompt_server(lambda message: logger.warning("%s %s", LOG_PREFIX, message))
_register_get, _register_post = make_route_registrars(
    _PROMPT_SERVER, lambda message: logger.warning("%s %s", LOG_PREFIX, message))


# ---------------------------------------------------------------------------
# HTTP: Three.js
# ---------------------------------------------------------------------------
# ⚠️ Библиотека лежит НЕ в `js/`, и это принципиально: ComfyUI импортирует
# КАЖДЫЙ `.js` из веб-папки пака при загрузке страницы (`/extensions` в
# server.py собирает их глобом). Положи 675 КБ туда — и за них платил бы каждый
# пользователь пака при каждом открытии, даже никогда не поставив эту ноду.
# Отсюда файл отдаётся отдельным маршрутом, а редактор подтягивает его
# динамическим `import()` только когда ноду действительно создали.
#
# Three.js r170, MIT. Текст лицензии лежит рядом с файлом.
_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
_THREE_FILE = _VENDOR_DIR / "three.module.min.js"


@_register_get("/ts_angle_select/three.module.js")
async def _three_endpoint(request):
    from aiohttp import web

    if not _THREE_FILE.is_file():
        return web.json_response(
            {"ok": False, "error": "three.module.min.js is missing from the pack."},
            status=404)
    return web.FileResponse(
        _THREE_FILE,
        headers={
            "Content-Type": "text/javascript; charset=utf-8",
            # Файл неизменен внутри версии пака — пусть браузер не перекачивает
            # его на каждое открытие графа.
            "Cache-Control": "public, max-age=86400",
        },
    )
