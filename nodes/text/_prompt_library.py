"""Библиотека промптов для TS Prompt Library: поиск, проверка, подстановка.

Библиотека — это ДАННЫЕ в `nodes/text/prompt_library/`, код её не знает по
именам. Устройство по папкам:

```text
prompt_library/
  <section>/                  раздел: context_edit, позже video, …
    section.json              {"schema": 1, "title": {...}, "summary": {...}, "order": 10}
    <collection>/             коллекция под одну модель: qwen_image_2_1, …
      collection.json         пресеты, сгруппированные; поля подстановки
      previews/               (необязательно) картинки к пресетам
```

Новый раздел или модель — это новая папка, без правки кода. Ключ пресета —
путь `section/collection/PRESET_ID`: он и хранится в сохранённом графе, поэтому
папки и `id` пресетов после публикации не переименовывают.

Формат коллекции (schema 1):

```json
{
  "schema": 1,
  "title": "Qwen Image 2.1",
  "model": "Qwen Image 2.1 Edit",
  "order": 10,
  "references": "<image1>, <image2>",
  "summary": {"ru": "...", "en": "..."},
  "guide": {"ru": ["..."], "en": ["..."]},
  "source": {"title": "...", "url": "https://...", "edition": "2026-09-27"},
  "placeholders": {"TARGET": {"label": {"ru": "...", "en": "..."}, "example": "..."}},
  "groups": [
    {"id": "restoration", "title": {"ru": "...", "en": "..."}, "presets": [
      {"id": "R01", "title": {...}, "inputs": 1, "summary": {...},
       "prompt": "Restore ... {TARGET} ...",
       "examples": {"TARGET": "..."},
       "note": {...},
       "links": [{"title": "...", "url": "https://..."}],
       "preview": "previews/R01.webp"}
    ]}
  ]
}
```

Текстовые поля — строка либо ``{"ru": ..., "en": ...}``. Поля в фигурных
скобках (``{TARGET}``) обязаны быть объявлены в ``placeholders`` коллекции.

⚠️ Кривой пресет не подключается, а не «подключается как-нибудь»: промпт с
неизвестным полем ушёл бы в модель с буквальным ``{WHATEVER}``, и человек
увидел бы это только по испорченной картинке. В журнале — почему пропущен.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from urllib.parse import quote

from .._shared import make_route_registrars, resolve_prompt_server

logger = logging.getLogger("comfyui_timesaver.ts_prompt_library")
LOG_PREFIX = "[TS Prompt Library]"

LIBRARY_DIR = Path(__file__).resolve().parent / "prompt_library"
SCHEMA_VERSION = 1

SECTION_FILE = "section.json"
COLLECTION_FILE = "collection.json"

# Поле подстановки: {TARGET}, {TIME_OF_DAY}. Только заглавные — строчные
# фигурные скобки бывают в самих промптах (JSON-подобные примеры).
PLACEHOLDER = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
# id папок и пресетов: то, что попадает в ключ сохранённого графа.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PREVIEW_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"})
MAX_INPUTS = 8


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def localized(value) -> dict:
    """Строка или {ru, en} -> {язык: текст}; пустое выпадает."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if isinstance(v, str) and v.strip()}
    if isinstance(value, str) and value.strip():
        return {"en": value}
    return {}


def placeholders_in(prompt: str) -> list[str]:
    """Поля подстановки промпта по порядку первого появления, без повторов."""
    seen: list[str] = []
    for key in PLACEHOLDER.findall(str(prompt or "")):
        if key not in seen:
            seen.append(key)
    return seen


def is_text_value(value) -> bool:
    """Годится ли значение поля: строка или число, но не True/False.

    Так же судит интерфейс (`js/text/_prompt_library.js`): bool в Python —
    подкласс int, и без этой оговорки нода вывела бы «True» там, где кнопка
    «Копировать» оставила бы поле пустым.
    """
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def fill(prompt: str, values: dict | None) -> str:
    """Подставить значения полей. Незаполненное поле остаётся как есть.

    ⚠️ Одним проходом по ИСХОДНОМУ тексту: значение, в котором самом есть
    «{OBJECT}», не раскрывается повторно — иначе текст человека зависел бы
    от того, в каком порядке шли поля.
    """
    table = {str(k): str(v).strip() for k, v in (values or {}).items() if is_text_value(v)}

    def replace(match: re.Match) -> str:
        value = table.get(match.group(1), "")
        return value if value else match.group(0)

    return PLACEHOLDER.sub(replace, str(prompt or ""))


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("%s %s is unreadable (%s) — skipped.", LOG_PREFIX, _short(path), exc)
        return None
    if not isinstance(data, dict):
        logger.warning("%s %s is not a JSON object — skipped.", LOG_PREFIX, _short(path))
        return None
    schema = data.get("schema", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        logger.warning("%s %s has schema %r, this node reads %d — skipped.",
                       LOG_PREFIX, _short(path), schema, SCHEMA_VERSION)
        return None
    return data


def _short(path: Path) -> str:
    """Путь для журнала — от корня библиотеки, без домашней папки человека."""
    try:
        return path.relative_to(LIBRARY_DIR.parent).as_posix()
    except ValueError:
        return path.name


def _preview(collection_dir: Path, value) -> str:
    """Относительный путь к картинке пресета, либо пустая строка.

    Только файл ВНУТРИ папки коллекции и только картинка: путь пишет автор
    JSON, а отдаёт его HTTP-маршрут.
    """
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if text.startswith("/") or ":" in text or ".." in text.split("/"):
        logger.warning("%s preview %r leaves its collection — ignored.", LOG_PREFIX, text)
        return ""
    if Path(text).suffix.lower() not in PREVIEW_EXTENSIONS:
        logger.warning("%s preview %r is not an image — ignored.", LOG_PREFIX, text)
        return ""
    return text


def _parse_preset(raw, collection: dict, where: str) -> dict | None:
    if not isinstance(raw, dict):
        return None
    preset_id = str(raw.get("id") or "").strip()
    if not _ID.match(preset_id):
        logger.warning("%s %s: preset id %r is invalid — skipped.", LOG_PREFIX, where, preset_id)
        return None
    prompt = str(raw.get("prompt") or "").strip()
    if not prompt:
        logger.warning("%s %s/%s has no prompt — skipped.", LOG_PREFIX, where, preset_id)
        return None
    unknown = [key for key in placeholders_in(prompt) if key not in collection["placeholders"]]
    if unknown:
        logger.warning("%s %s/%s uses undeclared field(s) %s — skipped.",
                       LOG_PREFIX, where, preset_id, ", ".join(unknown))
        return None
    try:
        inputs = int(raw.get("inputs", 1))
    except (TypeError, ValueError):
        inputs = -1
    if not 0 <= inputs <= MAX_INPUTS:
        logger.warning("%s %s/%s: inputs must be 0..%d — skipped.", LOG_PREFIX, where, preset_id, MAX_INPUTS)
        return None
    links = [
        {"title": str(link.get("title") or link.get("url")), "url": str(link["url"])}
        for link in (raw.get("links") or [])
        if isinstance(link, dict) and str(link.get("url") or "").startswith(("https://", "http://"))
    ]
    fields = placeholders_in(prompt)
    # Свой пример поверх общего для коллекции — там, где общий сбил бы с толку
    # («сколько кружек» при примере-одной-кружке). Только для полей промпта.
    examples = {
        str(name): str(value)
        for name, value in (raw.get("examples") or {}).items()
        if str(name) in fields and is_text_value(value) and str(value).strip()
    } if isinstance(raw.get("examples"), dict) else {}
    return {
        "id": preset_id,
        "key": f"{collection['key']}/{preset_id}",
        "title": localized(raw.get("title")) or {"en": preset_id},
        "inputs": inputs,
        "summary": localized(raw.get("summary")),
        "prompt": prompt,
        "fields": fields,
        "examples": examples,
        "note": localized(raw.get("note")),
        "links": links,
        "preview": _preview(collection["dir"], raw.get("preview")),
    }


def _parse_collection(section_id: str, folder: Path) -> dict | None:
    data = _read_json(folder / COLLECTION_FILE)
    if data is None:
        return None
    key = f"{section_id}/{folder.name}"
    # ⚠️ Автор мог записать список вместо объекта. Такое поле — ошибка ЭТОЙ
    # коллекции, а не повод уронить всю библиотеку (и регистрацию ноды вместе
    # с ней: `define_schema` строит список пресетов отсюда).
    for field in ("placeholders", "guide"):
        if data.get(field) is not None and not isinstance(data.get(field), dict):
            logger.warning("%s %s: \"%s\" must be an object — collection skipped.",
                           LOG_PREFIX, key, field)
            return None
    placeholders = {}
    for name, spec in (data.get("placeholders") or {}).items():
        if not PLACEHOLDER.fullmatch("{" + str(name) + "}"):
            logger.warning("%s %s: field name %r is invalid — ignored.", LOG_PREFIX, key, name)
            continue
        spec = spec if isinstance(spec, dict) else {}
        placeholders[str(name)] = {
            "label": localized(spec.get("label")) or {"en": str(name)},
            "example": str(spec.get("example") or ""),
        }
    collection = {
        "id": folder.name,
        "key": key,
        "dir": folder,
        "title": str(data.get("title") or folder.name),
        "model": str(data.get("model") or ""),
        "order": _order(data),
        "references": str(data.get("references") or ""),
        "summary": localized(data.get("summary")),
        "guide": {lang: [str(line) for line in lines if str(line).strip()]
                  for lang, lines in (data.get("guide") or {}).items() if isinstance(lines, list)},
        "source": data.get("source") if isinstance(data.get("source"), dict) else {},
        "placeholders": placeholders,
        "groups": [],
    }
    seen: set[str] = set()
    for raw_group in data.get("groups") or []:
        if not isinstance(raw_group, dict):
            continue
        group_id = str(raw_group.get("id") or "").strip() or f"group{len(collection['groups']) + 1}"
        presets = []
        for raw in raw_group.get("presets") or []:
            preset = _parse_preset(raw, collection, key)
            if preset is None:
                continue
            if preset["id"] in seen:
                logger.warning("%s %s: preset id %s repeats — the second one skipped.",
                               LOG_PREFIX, key, preset["id"])
                continue
            seen.add(preset["id"])
            presets.append(preset)
        if presets:
            collection["groups"].append({
                "id": group_id,
                "title": localized(raw_group.get("title")) or {"en": group_id},
                "presets": presets,
            })
    if not collection["groups"]:
        logger.warning("%s %s has no usable presets — skipped.", LOG_PREFIX, key)
        return None
    return collection


def _order(data: dict) -> int:
    try:
        return int(data.get("order", 100))
    except (TypeError, ValueError):
        return 100


def _scan(root: Path) -> list[dict]:
    sections = []
    if not root.is_dir():
        return sections
    for section_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not _ID.match(section_dir.name) or not (section_dir / SECTION_FILE).is_file():
            continue
        data = _read_json(section_dir / SECTION_FILE)
        if data is None:
            continue
        collections = []
        for folder in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            if _ID.match(folder.name) and (folder / COLLECTION_FILE).is_file():
                try:
                    collection = _parse_collection(section_dir.name, folder)
                except (AttributeError, TypeError, ValueError) as exc:
                    # Форма, которую разбор не предусмотрел, — всё равно ошибка
                    # одной коллекции: остальная библиотека обязана работать.
                    logger.warning("%s %s/%s is malformed (%s) — collection skipped.",
                                   LOG_PREFIX, section_dir.name, folder.name, exc)
                    collection = None
                if collection is not None:
                    collections.append(collection)
        if not collections:
            continue
        collections.sort(key=lambda c: (c["order"], c["title"].lower()))
        sections.append({
            "id": section_dir.name,
            "title": localized(data.get("title")) or {"en": section_dir.name},
            "summary": localized(data.get("summary")),
            "order": _order(data),
            "collections": collections,
        })
    sections.sort(key=lambda s: (s["order"], s["id"]))
    return sections


# --------------------------------------------------------------------------
# Кэш: библиотеку читают /object_info (на каждую загрузку страницы), маршрут
# каталога и каждый прогон ноды. Перечитываем, только когда поменялись файлы.
# --------------------------------------------------------------------------


class _LibraryState:
    lock = threading.Lock()
    stamp: tuple = ()
    root: Path | None = None
    sections: list = []
    index: dict = {}


_state = _LibraryState()


def _stamp(root: Path) -> tuple:
    """Отпечаток библиотеки: пути и mtime всех JSON. Дёшево — файлов десятки."""
    entries = []
    if root.is_dir():
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".json"):
                    path = os.path.join(dirpath, name)
                    try:
                        entries.append((path, os.stat(path).st_mtime_ns))
                    except OSError:
                        continue
    return tuple(sorted(entries))


def load_library(root: Path | None = None) -> list[dict]:
    """Разделы с коллекциями и пресетами, по порядку. Кэшируется по mtime."""
    root = Path(root) if root is not None else LIBRARY_DIR
    stamp = _stamp(root)
    with _state.lock:
        if _state.root == root and _state.stamp == stamp:
            return _state.sections
        sections = _scan(root)
        index = {}
        for section in sections:
            for collection in section["collections"]:
                for group in collection["groups"]:
                    for preset in group["presets"]:
                        index[preset["key"]] = (preset, collection)
        _state.root, _state.stamp, _state.sections, _state.index = root, stamp, sections, index
        return sections


def library_version(root: Path | None = None) -> str:
    """Отпечаток для кэша ComfyUI: правка JSON должна перезапускать ноду."""
    stamp = _stamp(Path(root) if root is not None else LIBRARY_DIR)
    return hashlib.blake2b(repr(stamp).encode("utf-8"), digest_size=12).hexdigest()


def preset_keys(root: Path | None = None) -> list[str]:
    """Все ключи пресетов в порядке каталога — варианты выпадашки ноды."""
    return [
        preset["key"]
        for section in load_library(root)
        for collection in section["collections"]
        for group in collection["groups"]
        for preset in group["presets"]
    ]


def find_preset(key: str, root: Path | None = None):
    """(preset, collection) по ключу, либо (None, None)."""
    load_library(root)
    with _state.lock:
        return _state.index.get(str(key or ""), (None, None))


def catalog_payload(root: Path | None = None) -> dict:
    """Каталог для интерфейса ноды: всё, кроме путей на диске."""
    sections = []
    for section in load_library(root):
        collections = []
        for collection in section["collections"]:
            public = {k: v for k, v in collection.items() if k != "dir"}
            public["groups"] = [
                {**group, "presets": [
                    {**preset, "preview": (
                        f"/ts_prompt_library/preview?key={quote(preset['key'], safe='/')}"
                        if preset["preview"] else "")}
                    for preset in group["presets"]
                ]}
                for group in collection["groups"]
            ]
            collections.append(public)
        sections.append({**section, "collections": collections})
    return {"schema": SCHEMA_VERSION, "sections": sections}


def preview_path(key: str, root: Path | None = None) -> Path | None:
    """Файл превью пресета — только внутри папки его коллекции."""
    preset, collection = find_preset(key, root)
    if not preset or not preset["preview"]:
        return None
    base = collection["dir"].resolve()
    target = (base / preset["preview"]).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    if target.suffix.lower() not in PREVIEW_EXTENSIONS or not target.is_file():
        return None
    return target


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _warn_route(message: str) -> None:
    logger.warning("%s %s", LOG_PREFIX, message)


_PROMPT_SERVER = resolve_prompt_server(_warn_route)
_register_get, _ = make_route_registrars(_PROMPT_SERVER, _warn_route)


@_register_get("/ts_prompt_library/catalog")
async def catalog_route(_request):
    """Вся библиотека одним ответом: разделы, коллекции, пресеты."""
    from aiohttp import web

    return web.json_response(catalog_payload())


@_register_get("/ts_prompt_library/preview")
async def preview_route(request):
    """Картинка пресета. Ключ — тот же, что в выпадашке ноды."""
    from aiohttp import web

    path = preview_path(request.query.get("key", ""))
    if path is None:
        return web.Response(status=404)
    return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})
