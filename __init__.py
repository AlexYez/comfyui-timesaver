import ast
import importlib
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

_STANDALONE_IMPORT = __package__ in {None, ""}

if _STANDALONE_IMPORT:
    from ts_dependency_manager import TSDependencyManager
else:
    from .ts_dependency_manager import TSDependencyManager

logger = logging.getLogger("TimesaverVFX_Pack")

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_PACKAGE_DIR = Path(__file__).resolve().parent
_NODE_MODULE_DIR = _PACKAGE_DIR / "nodes"


def _discover_module_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    if not _NODE_MODULE_DIR.is_dir():
        return entries

    for py_file in sorted(_NODE_MODULE_DIR.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        # Naming convention: every public node file uses the ts_ prefix.
        # This filter skips helper packages bundled alongside nodes
        # (frame_interpolation_models/, video_depth_anything/, ...).
        if not py_file.name.startswith("ts_"):
            continue
        relative_to_node_dir = py_file.relative_to(_NODE_MODULE_DIR)
        if any(part.startswith("_") for part in relative_to_node_dir.parts):
            # Skip __pycache__ and any private/shared helpers (_shared/, _internal.py, ...).
            continue
        relative = py_file.relative_to(_PACKAGE_DIR)
        module_path = relative.with_suffix("").as_posix().replace("/", ".")
        entries.append(
            {
                "module_import": module_path,
                "module_label": relative.as_posix(),
            }
        )

    return entries


_MODULE_ENTRIES = _discover_module_entries()

_HOST_MODULE_ROOTS = {
    "comfy",
    "comfy_api",
    "folder_paths",
    "nodes",
    "server",
}

_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", set()))
_LOCAL_MODULE_ROOTS = {path.stem for path in _PACKAGE_DIR.glob("*.py")}
if _NODE_MODULE_DIR.is_dir():
    _LOCAL_MODULE_ROOTS.update(
        {
            path.stem
            for path in _NODE_MODULE_DIR.rglob("*.py")
            if path.name != "__init__.py" and "__pycache__" not in path.parts
        }
    )
    _LOCAL_MODULE_ROOTS.update(
        {
            path.name
            for path in _NODE_MODULE_DIR.rglob("*")
            if path.is_dir() and "__pycache__" not in path.parts
        }
    )
_LOCAL_MODULE_ROOTS.update(
    {
        path.name
        for path in _PACKAGE_DIR.iterdir()
        if path.is_dir()
    }
)

_MODULE_LOAD_RESULTS: list[dict] = []
_IMPORT_AUDIT_RESULTS: list[dict] = []


def _truncate(value: object, width: int) -> str:
    # Newlines from multi-line exception messages would break the ASCII
    # table layout — fold them into a visible marker first.
    text = " | ".join(str(value).splitlines()) or str(value)
    if width <= 3:
        return text[:width]
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _render_table(headers: list[str], rows: list[list[object]], max_widths: list[int]) -> str:
    if not rows:
        rows = [["-", "-", "-", "-"]] if len(headers) == 4 else [["-", "-", "-"]]

    widths = []
    for index, header in enumerate(headers):
        width = min(max_widths[index], len(header))
        for row in rows:
            width = min(max_widths[index], max(width, len(str(row[index]))))
        widths.append(width)

    def render_row(row_values: list[object]) -> str:
        cells = []
        for index, value in enumerate(row_values):
            text = _truncate(value, widths[index]).ljust(widths[index])
            cells.append(f" {text} ")
        return "|" + "|".join(cells) + "|"

    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [border, render_row(headers), border]
    for row in rows:
        lines.append(render_row(row))
    lines.append(border)
    return "\n".join(lines)


def _is_internal_module_name(module_name: str) -> bool:
    """Принадлежит ли отсутствующий модуль самому паку.

    Отличает «у человека не стоит demucs» от «мы опечатались в собственном
    импорте». Первое — штатный пропуск, второе — регрессия, которую обязано
    быть видно.
    """
    name = str(module_name or "").strip()
    if not name:
        return False
    if name.startswith(f"{__name__}.") or name == __name__:
        return True
    parts = name.split(".")
    if parts[0] == "nodes" and len(parts) > 1:
        # `nodes` — ещё и модуль ядра ComfyUI, поэтому одного имени мало:
        # внутренним считаем только то, что и правда лежит в пакете.
        candidate = _NODE_MODULE_DIR.joinpath(*parts[1:])
        return candidate.is_dir() or candidate.with_suffix(".py").is_file()
    return parts[0] in _LOCAL_MODULE_ROOTS and parts[0] not in _HOST_MODULE_ROOTS


def _is_internal_or_host_module(root: str) -> bool:
    if not root:
        return True
    if root in _HOST_MODULE_ROOTS:
        return True
    if root in _STDLIB_MODULES:
        return True
    if root in _LOCAL_MODULE_ROOTS:
        return True
    if root.startswith("ts_"):
        return True
    return False


def _extract_import_roots_from_source(source: str) -> set[str]:
    """Return the top-level module roots imported by `source`.

    Parses with the AST instead of scanning lines, so prose inside
    docstrings/comments/string literals can never be mistaken for an
    import (the old line-based scan flagged text like ``from the public
    mathematics ...`` as a missing dependency). Walking the tree also
    catches lazy imports nested in functions. Relative imports
    (``from . import x``) are skipped — they resolve inside this package.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0].strip()
                if root:
                    roots.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue  # relative or `from . import x` — internal
            root = node.module.split(".", 1)[0].strip()
            if root:
                roots.add(root)
    return roots


def _scan_external_imports() -> list[dict]:
    usage: dict[str, set[str]] = {}

    for py_file in _PACKAGE_DIR.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(_PACKAGE_DIR).as_posix()
        try:
            # utf-8-sig, а не utf-8: три файла пака несли BOM, `ast.parse`
            # падал на нём, ошибка глоталась — и импорты этих файлов не
            # попадали в аудит вовсе.
            content = py_file.read_text(encoding="utf-8-sig", errors="ignore")
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("[TS Loader] Skipping import audit for %s: %s", rel, exc)
            continue
        for root in _extract_import_roots_from_source(content):
            if _is_internal_or_host_module(root):
                continue
            usage.setdefault(root, set()).add(rel)

    results = []
    for root in sorted(usage.keys()):
        available = importlib.util.find_spec(root) is not None
        files = sorted(usage[root])
        if len(files) == 1:
            source = files[0]
        elif len(files) == 2:
            source = f"{files[0]}, {files[1]}"
        else:
            source = f"{files[0]}, {files[1]}, +{len(files) - 2} more"
        results.append(
            {
                "import": root,
                "available": "yes" if available else "no",
                "source": source,
            }
        )
    return results


def _register_module_nodes(module_name: str, module) -> int:
    if not hasattr(module, "NODE_CLASS_MAPPINGS"):
        return 0

    mappings = module.NODE_CLASS_MAPPINGS
    if not isinstance(mappings, dict) or not mappings:
        return 0

    NODE_CLASS_MAPPINGS.update(mappings)

    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        display_names = module.NODE_DISPLAY_NAME_MAPPINGS
        if isinstance(display_names, dict):
            NODE_DISPLAY_NAME_MAPPINGS.update(display_names)

    for node_name, node_cls in mappings.items():
        if isinstance(node_cls, type):
            try:
                TSDependencyManager.wrap_node_runtime(node_name=node_name, node_cls=node_cls, logger=logger)
            except Exception as exc:
                logger.exception(
                    "[TS Loader] Runtime guard attach failed for node '%s' from module '%s': %s",
                    node_name,
                    module_name,
                    exc,
                )

    return len(mappings)


def _load_module(module_import: str, module_label: str) -> None:
    result = {
        "module": module_label,
        "status": "OK",
        "nodes": 0,
        "details": "Loaded",
    }

    try:
        module = importlib.import_module(f".{module_import}", package=__name__)
        node_count = _register_module_nodes(module_label, module)
        result["nodes"] = node_count
        result["details"] = f"Loaded ({node_count} nodes)"
    except ImportError as exc:
        missing = TSDependencyManager.extract_missing_dependency(exc)
        # ⚠️ SKIPPED — только про ЧУЖОЙ отсутствующий пакет. Раньше сюда падал
        # любой ImportError, включая собственную опечатку в имени модуля пака:
        # `No module named 'nodes.image.typo'` рапортовалось как «Missing
        # dependency», CI такие модули пропускает по замыслу (у silero/demucs
        # это норма), и нода исчезала молча — до жалобы пользователя.
        if missing and not _is_internal_module_name(missing):
            result["status"] = "SKIPPED"
            result["details"] = f"Missing dependency: {missing}"
            logger.warning("[TS Loader] %s -> %s", module_label, result["details"])
        else:
            result["status"] = "ERROR"
            if missing:
                result["details"] = (
                    f"Broken internal import: {missing} "
                    f"(a module of this pack, not a third-party dependency)"
                )
            else:
                result["details"] = f"ImportError: {exc}"
            logger.exception("[TS Loader] %s -> %s", module_label, result["details"])
    except Exception as exc:
        result["status"] = "ERROR"
        result["details"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[TS Loader] Error loading module '%s': %s", module_label, exc)

    _MODULE_LOAD_RESULTS.append(result)


def _print_startup_report() -> None:
    load_issues = [r for r in _MODULE_LOAD_RESULTS if r["status"] in {"SKIPPED", "ERROR"}]
    critical_missing_roots = _collect_critical_missing_roots()

    import_rows = []
    critical_missing_imports = []
    optional_missing_imports = []
    for item in _IMPORT_AUDIT_RESULTS:
        if item["available"] == "yes":
            severity = "ok"
        else:
            severity = "critical" if item["import"] in critical_missing_roots else "optional"
            if severity == "critical":
                critical_missing_imports.append(item)
            else:
                optional_missing_imports.append(item)
        import_rows.append([item["import"], item["available"], severity, item["source"]])

    # ⚠️ Тихо, когда всё хорошо. Консоль ComfyUI — общая для десятков паков, и
    # две простыни таблиц на каждом запуске в ней только мешают: в них тонут
    # чужие настоящие ошибки. Когда всё загрузилось — одна строка. Когда нет —
    # только то, что сломалось. Таблицы целиком остаются доступны по
    # TS_VERBOSE_STARTUP=1: они писались для разбора аварий, и выбрасывать их
    # нельзя, а показывать всем каждый раз — незачем.
    #
    # Отсутствующие НЕобязательные зависимости молчат сознательно: они
    # отсутствуют у большинства и по замыслу (§14), а расскажет о них
    # TSDependencyManager в тот момент, когда нужная нода действительно
    # запустится, — там это уже не шум, а ответ на вопрос.
    verbose = os.environ.get("TS_VERBOSE_STARTUP", "").strip().lower() in {"1", "true", "yes"}
    troubled = bool(load_issues or critical_missing_imports)

    if verbose:
        # Таблицы форматируются ЗДЕСЬ, а не выше: в тихом режиме их результат
        # всё равно отбрасывался, а строится он на каждом запуске ComfyUI по
        # всем модулям и всем внешним импортам.
        module_table = _render_table(
            headers=["Module", "Status", "Nodes", "Details"],
            rows=[[r["module"], r["status"], r["nodes"], r["details"]]
                  for r in _MODULE_LOAD_RESULTS],
            max_widths=[30, 10, 8, 90],
        )
        import_table = _render_table(
            headers=["Import", "Available", "Severity", "Source"],
            rows=import_rows,
            max_widths=[28, 10, 10, 86],
        )
        loaded = sum(1 for r in _MODULE_LOAD_RESULTS if r["status"] == "OK")
        skipped = sum(1 for r in _MODULE_LOAD_RESULTS if r["status"] == "SKIPPED")
        errors = sum(1 for r in _MODULE_LOAD_RESULTS if r["status"] == "ERROR")

        logger.info("[TS Startup] comfyui-timesaver load report")
        logger.info("[TS Startup] Package path: %s", _PACKAGE_DIR)
        logger.info("[TS Startup] Modules discovered: %d", len(_MODULE_ENTRIES))
        for line in module_table.splitlines():
            logger.info("%s", line)
        logger.info("[TS Startup] External imports discovered: %d",
                    len(_IMPORT_AUDIT_RESULTS))
        for line in import_table.splitlines():
            logger.info("%s", line)
        logger.info(
            "[TS Startup] Summary: "
            "loaded=%d, skipped=%d, errors=%d, load_issues=%d, "
            "nodes_registered=%d, critical_missing_imports=%d, optional_missing_imports=%d",
            loaded, skipped, errors, len(load_issues),
            len(NODE_CLASS_MAPPINGS),
            len(critical_missing_imports),
            len(optional_missing_imports),
        )

    if not troubled:
        if not verbose:
            logger.info("[TS Timesaver] All %d nodes loaded successfully.",
                        len(NODE_CLASS_MAPPINGS))
        return

    logger.warning(
        "[TS Timesaver] %d node(s) loaded, %d module(s) did not. "
        "Set TS_VERBOSE_STARTUP=1 for the full report.",
        len(NODE_CLASS_MAPPINGS), len(load_issues))
    for item in load_issues:
        logger.warning("  - %s: %s", item["module"], item["details"])
    for item in critical_missing_imports:
        logger.warning("  - missing %s (used in: %s)", item["import"], item["source"])


def _collect_critical_missing_roots() -> set[str]:
    roots = set()
    missing_prefix = "Missing dependency:"
    import_from_pattern = re.compile(r"from '([^']+)'")

    for result in _MODULE_LOAD_RESULTS:
        if result["status"] not in {"SKIPPED", "ERROR"}:
            continue
        details = str(result.get("details", ""))
        if details.startswith(missing_prefix):
            dep = details[len(missing_prefix) :].strip()
            if dep:
                roots.add(dep.split(".", 1)[0])
            continue
        match = import_from_pattern.search(details)
        if match:
            roots.add(match.group(1).split(".", 1)[0])

    return roots


def _apply_core_patches() -> None:
    """Заплатки на чужой код: ядро ComfyUI и стандартная библиотека.

    Каждая в своём `try` — пак, который не смог что-то залатать, обязан всё
    равно загрузить свои ноды.

    1. Load Image в ядре показывает только файлы в корне `input`, поэтому
       картинка, вставленная ComfyUI в `input/pasted`, после каждой перезагрузки
       объявляется отсутствующей.
    2. asyncio на Windows роняет ConnectionResetError при закрытии websocket'а
       (баг CPython, не ComfyUI) — 12% строк лога и сокет, закрытый с задержкой.

    Обе идемпотентны: повторный вызов безвреден.
    """
    try:
        from .ts_pasted_media_fix import apply_patch

        apply_patch()
    except Exception as error:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning(
            "[TS PastedMediaFix] Not installed: %s", error,
        )

    try:
        from .compat.proactor_guard import install as _install_proactor_guard

        _install_proactor_guard()
    except Exception as error:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning(
            "[TS ProactorGuard] Not installed: %s", error,
        )


if not _STANDALONE_IMPORT:
    for _entry in _MODULE_ENTRIES:
        _load_module(_entry["module_import"], _entry["module_label"])

    _IMPORT_AUDIT_RESULTS = _scan_external_imports()
    _print_startup_report()

    _apply_core_patches()


# ── регистрация: V3, а при старом ComfyUI — прежний путь ─────────────────── #
#
# ⚠️ Ноды пака давно на V3-схемах, но регистрировался он по-старому. Загрузчик
# ComfyUI сначала смотрит `NODE_CLASS_MAPPINGS` и, найдя его, УХОДИТ в V1-ветку,
# не заглядывая в `comfy_entrypoint` (порядок ветвей — в `nodes.py` ядра).
# Поэтому корневой словарь отдаётся наружу ТОЛЬКО там, где V3-расширений нет:
# иначе он молча отменял бы всю миграцию.
try:  # pragma: no cover - зависит от версии ComfyUI
    from comfy_api.latest import ComfyExtension as _ComfyExtension
except Exception:  # pragma: no cover - старая сборка или тесты без ComfyUI
    _ComfyExtension = None


if _ComfyExtension is not None:
    # ⚠️ Список снимается ДО того, как словарь исчезнет из модуля: проверять
    # `hasattr(module, "NODE_CLASS_MAPPINGS")` будет сам ComfyUI, и пока
    # атрибут на месте, V3-ветка не начинается вовсе. Проверено импортом:
    # без удаления миграция оставалась холостой.
    _NODE_CLASSES = list(NODE_CLASS_MAPPINGS.values())

    class TimesaverExtension(_ComfyExtension):
        """Пак целиком: те же классы, те же `node_id`, другая дорога внутрь.

        Идентификаторы нод не меняются — их несёт схема каждого класса, а
        сохранённые workflow ссылаются именно на них.
        """

        async def get_node_list(self) -> list[type]:
            # ⚠️ Схемы проверяются ПОШТУЧНО. Ядро строит их в одном try/except
            # (V3-ветка `load_custom_node`) и на исключении возвращает False,
            # а регистрация идёт инкрементально — значит упавшая `define_schema`
            # уносит с собой саму ноду И ВСЕ следующие за ней по списку. Схемы
            # трёх нод читают диск при построении (LUT-ы, папки аудио, модели
            # Whisper), поэтому это не гипотетический случай.
            #
            # Отдавать наружу заведомо непригодный класс незачем: пусть пропадёт
            # одна нода, а не половина пака.
            usable = []
            for node_cls in _NODE_CLASSES:
                try:
                    node_cls.GET_SCHEMA()
                except Exception as exc:      # noqa: BLE001 - одна нода не стоит пака
                    logger.exception(
                        "[TS Loader] Schema build failed for '%s'; the node is "
                        "withheld so the rest of the pack still registers: %s",
                        getattr(node_cls, "__name__", node_cls), exc,
                    )
                    continue
                usable.append(node_cls)
            return usable

        async def on_load(self) -> None:
            # Заплатка на ядро — работа времени загрузки, а не времени импорта:
            # здесь ComfyUI уже собран, и падение видно как ошибка расширения,
            # а не как молчаливое исключение внутри `import`.
            _apply_core_patches()

    async def comfy_entrypoint() -> "_ComfyExtension":
        return TimesaverExtension()

    # Словари уходят из модуля целиком — иначе загрузчик выберет V1-ветку и
    # вернётся до `comfy_entrypoint`, как было до миграции.
    del NODE_CLASS_MAPPINGS
    del NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["comfy_entrypoint", "TimesaverExtension"]
else:
    # Сборка без V3-загрузчика: отдаём словари, как раньше.
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
