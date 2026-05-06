import importlib
import importlib.util
import logging
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
    text = str(value)
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


def _extract_import_roots_from_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []

    if stripped.startswith("import "):
        payload = stripped[len("import ") :].split("#", 1)[0].strip()
        parts = [part.strip() for part in payload.split(",") if part.strip()]
        roots = []
        for part in parts:
            module_name = part.split(" as ", 1)[0].strip()
            if module_name:
                roots.append(module_name.split(".", 1)[0].strip())
        return roots

    if stripped.startswith("from "):
        payload = stripped[len("from ") :].split(" import ", 1)[0].strip()
        if not payload or payload.startswith("."):
            return []
        return [payload.split(".", 1)[0].strip()]

    return []


def _scan_external_imports() -> list[dict]:
    usage: dict[str, set[str]] = {}

    for py_file in _PACKAGE_DIR.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(_PACKAGE_DIR).as_posix()
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("[TS Loader] Skipping import audit for %s: %s", rel, exc)
            continue
        for line in content.splitlines():
            for root in _extract_import_roots_from_line(line):
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

    mappings = getattr(module, "NODE_CLASS_MAPPINGS")
    if not isinstance(mappings, dict) or not mappings:
        return 0

    NODE_CLASS_MAPPINGS.update(mappings)

    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
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
        result["status"] = "SKIPPED"
        if missing:
            result["details"] = f"Missing dependency: {missing}"
        else:
            result["details"] = f"ImportError: {exc}"
        logger.warning("[TS Loader] %s -> %s", module_label, result["details"])
    except Exception as exc:
        result["status"] = "ERROR"
        result["details"] = f"{type(exc).__name__}: {exc}"
        logger.exception("[TS Loader] Error loading module '%s': %s", module_label, exc)

    _MODULE_LOAD_RESULTS.append(result)


def _print_startup_report() -> None:
    loaded = sum(1 for r in _MODULE_LOAD_RESULTS if r["status"] == "OK")
    skipped = sum(1 for r in _MODULE_LOAD_RESULTS if r["status"] == "SKIPPED")
    errors = sum(1 for r in _MODULE_LOAD_RESULTS if r["status"] == "ERROR")
    load_issues = [r for r in _MODULE_LOAD_RESULTS if r["status"] in {"SKIPPED", "ERROR"}]
    critical_missing_roots = _collect_critical_missing_roots()

    module_rows = [
        [r["module"], r["status"], r["nodes"], r["details"]]
        for r in _MODULE_LOAD_RESULTS
    ]
    module_table = _render_table(
        headers=["Module", "Status", "Nodes", "Details"],
        rows=module_rows,
        max_widths=[30, 10, 8, 90],
    )

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

    import_table = _render_table(
        headers=["Import", "Available", "Severity", "Source"],
        rows=import_rows,
        max_widths=[28, 10, 10, 86],
    )

    logger.info("[TS Startup] comfyui-timesaver load report")
    logger.info("[TS Startup] Package path: %s", _PACKAGE_DIR)
    logger.info("[TS Startup] Modules discovered: %d", len(_MODULE_ENTRIES))
    for line in module_table.splitlines():
        logger.info("%s", line)
    logger.info("[TS Startup] External imports discovered: %d", len(_IMPORT_AUDIT_RESULTS))
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
    if load_issues:
        logger.info("[TS Startup] Module load issues:")
        for item in load_issues:
            logger.info("  - %s: %s", item["module"], item["details"])
    else:
        logger.info("[TS Startup] Module load issues: none")
    if critical_missing_imports:
        logger.warning("[TS Startup] Critical missing imports:")
        for item in critical_missing_imports:
            logger.warning("  - %s (used in: %s)", item["import"], item["source"])
    else:
        logger.info("[TS Startup] Critical missing imports: none")
    if optional_missing_imports:
        logger.info("[TS Startup] Optional missing imports:")
        for item in optional_missing_imports:
            logger.info("  - %s (used in: %s)", item["import"], item["source"])
    else:
        logger.info("[TS Startup] Optional missing imports: none")


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


if not _STANDALONE_IMPORT:
    for _entry in _MODULE_ENTRIES:
        _load_module(_entry["module_import"], _entry["module_label"])

    _IMPORT_AUDIT_RESULTS = _scan_external_imports()
    _print_startup_report()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
