"""Пресеты стилей и сегментных тегов для TS Song Creator.

Пресет — обычный JSON в `nodes/text/song_presets/`, по файлу на модель. Смысл
тот же, что у пресетов ракурсов (`_angle_presets.py`): словарь, на котором
модель обучалась, живёт в данных, а не в коде, и новая модель добавляется
файлом.

```json
{
  "name": "YuE 2",
  "tags":   [{"tag": "[verse]", "name": {"en": "Verse", "ru": "Куплет"}}],
  "styles": [{"id": "pop-uplifting", "group": "pop",
              "name": {"en": "Uplifting Pop", "ru": "Воодушевляющий поп"},
              "hint": {"en": "...", "ru": "..."},
              "prompt": "pop, uplifting, 118 bpm, female vocal, ..."}]
}
```

⚠️ Названия стилей переводятся, `prompt` — НЕТ. Это вход модели: она обучалась
на английских дескрипторах, и «воодушевляющий поп» вместо «uplifting pop» даёт
не русскую песню, а испорченное обусловливание. Человек видит перевод, модель
получает оригинал.

⚠️ Сегментные теги (`[verse]`, `[chorus]`) тоже модель-специфичны и потому лежат
здесь же, рядом со стилями: другая модель может читать другие.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .._shared import make_route_registrars, resolve_prompt_server

logger = logging.getLogger("comfyui_timesaver.ts_song_creator")
LOG_PREFIX = "[TS Song Creator]"

PRESETS_DIR = Path(__file__).resolve().parent / "song_presets"

_LOCALES = ("en", "ru")


def _localized(value, fallback: str = "") -> dict[str, str]:
    """Пара en/ru из значения пресета.

    Строкой записанное имя считается английским: так короче для пресета, где
    перевода нет, и не заставляет писать один и тот же текст дважды.
    """
    if isinstance(value, str):
        text = value.strip() or fallback
        return {"en": text, "ru": text}
    if isinstance(value, dict):
        english = str(value.get("en") or fallback).strip()
        return {
            "en": english,
            "ru": str(value.get("ru") or english).strip(),
        }
    return {"en": fallback, "ru": fallback}


def _validate(data: dict, source: Path) -> dict | None:
    """Пресет либо годный целиком, либо не подключается.

    ⚠️ Половинчатый пресет хуже отсутствующего: в списке стилей появилась бы
    строка без промпта, и человек узнал бы об этом по молчащей модели.
    """
    name = str(data.get("name") or source.stem).strip()

    styles: list[dict] = []
    for index, raw in enumerate(data.get("styles") or []):
        if not isinstance(raw, dict):
            logger.warning("%s %s: style #%d is not an object — skipped.",
                           LOG_PREFIX, source.name, index + 1)
            continue
        prompt = str(raw.get("prompt") or "").strip()
        style_id = str(raw.get("id") or "").strip()
        if not prompt or not style_id:
            logger.warning("%s %s: style '%s' has no id or no prompt — skipped.",
                           LOG_PREFIX, source.name, style_id or index + 1)
            continue
        styles.append({
            "id": style_id,
            "group": str(raw.get("group") or "other").strip(),
            "name": _localized(raw.get("name"), style_id),
            "hint": _localized(raw.get("hint"), ""),
            "prompt": prompt,
            # Голос, под который стиль написан: интерфейс подставит его сразу,
            # чтобы поле не оставалось без вокала вовсе.
            "vocal": str(raw.get("vocal") or "").strip(),
        })

    if not styles:
        logger.warning("%s %s has no usable styles — skipped.", LOG_PREFIX, source.name)
        return None

    # ⚠️ Вокал — ОТДЕЛЬНЫЙ список, а не часть жанра. Голос человек меняет
    # независимо от стиля, и промпт жанра, в котором уже сказано «female
    # vocal», спорил бы с выбранным рядом мужским.
    vocals: list[dict] = []
    for index, raw in enumerate(data.get("vocals") or []):
        if not isinstance(raw, dict):
            continue
        prompt = str(raw.get("prompt") or "").strip()
        vocal_id = str(raw.get("id") or "").strip()
        if not prompt or not vocal_id:
            logger.warning("%s %s: vocal '%s' has no id or no prompt — skipped.",
                           LOG_PREFIX, source.name, vocal_id or index + 1)
            continue
        vocals.append({
            "id": vocal_id,
            "group": str(raw.get("group") or "other").strip(),
            "name": _localized(raw.get("name"), vocal_id),
            "hint": _localized(raw.get("hint"), ""),
            "prompt": prompt,
        })

    known_vocals = {item["id"] for item in vocals}
    for style in styles:
        # Рекомендованный голос существует или его нет вовсе: ссылка в пустоту
        # молча оставляла бы стиль без вокала, и заметно это было бы только по
        # результату.
        if style["vocal"] and style["vocal"] not in known_vocals:
            logger.warning("%s %s: style '%s' points at unknown vocal '%s'.",
                           LOG_PREFIX, source.name, style["id"], style["vocal"])
            style["vocal"] = ""

    groups: list[dict] = []
    for raw in data.get("vocal_groups") or []:
        if not isinstance(raw, dict):
            continue
        group_id = str(raw.get("id") or "").strip()
        if not group_id:
            continue
        groups.append({"id": group_id, "name": _localized(raw.get("name"), group_id)})

    tags: list[dict] = []
    for raw in data.get("tags") or []:
        if not isinstance(raw, dict):
            continue
        tag = str(raw.get("tag") or "").strip()
        if not tag:
            continue
        tags.append({"tag": tag, "name": _localized(raw.get("name"), tag)})

    return {
        "id": source.stem,
        "name": name,
        "order": int(data.get("order") or 100),
        "model": str(data.get("model") or "").strip(),
        "notes": str(data.get("notes") or "").strip(),
        "tags": tags,
        "styles": styles,
        "vocals": vocals,
        "vocal_groups": groups,
    }


def load_presets() -> list[dict]:
    """Все пресеты из `song_presets/`, в порядке `order`, затем по имени."""
    presets: list[dict] = []
    if not PRESETS_DIR.is_dir():
        logger.warning("%s No presets directory at %s.", LOG_PREFIX, PRESETS_DIR)
        return presets
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("%s %s could not be read: %s", LOG_PREFIX, path.name, error)
            continue
        if not isinstance(data, dict):
            logger.warning("%s %s is not a JSON object — skipped.", LOG_PREFIX, path.name)
            continue
        checked = _validate(data, path)
        if checked:
            presets.append(checked)
    presets.sort(key=lambda item: (item["order"], item["name"].lower()))
    return presets


def preset_names() -> list[str]:
    """Имена пресетов для выпадающего списка ноды."""
    names = [preset["name"] for preset in load_presets()]
    return names or ["YuE 2"]


# ---------------------------------------------------------------------------
# Маршрут: пресеты для интерфейса
# ---------------------------------------------------------------------------
#
# Фронтенд рисует список стилей и кнопки тегов, поэтому ему нужны те же данные,
# что и ноде. Отдаём их одним ответом, читая файлы на каждый запрос: пресетов
# три десятка строк, а вот кэш, переживающий правку файла, заставил бы
# перезапускать ComfyUI ради добавленного стиля.

_PROMPT_SERVER = resolve_prompt_server(
    lambda message: logger.warning("%s %s", LOG_PREFIX, message))
_register_get, _ = make_route_registrars(
    _PROMPT_SERVER, lambda message: logger.warning("%s %s", LOG_PREFIX, message))


@_register_get("/ts_song_creator/presets")
async def presets_route(_request):
    """Стили и сегментные теги всех пресетов."""
    from aiohttp import web

    return web.json_response({"schema": 1, "presets": load_presets()})
