"""HTTP surface of the subscription pass — shared by every TS studio.

A side-effect module, not a node: it registers routes and exposes no class
(same pattern as the sampler/scheduler injectors). The rules themselves live
in `nodes/_pass.py`; this file only carries them to the frontend.

Every route is reached by an explicit user action — pressing Activate, opening
the studio. Nothing here runs on import beyond registration, and nothing calls
out to the network unless someone asked for it.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ._deps import TSDependencyManager  # noqa: F401  (kept for import parity)
from ._shared import make_route_registrars
from . import _pass
from . import _studio_dev
from . import _studio_packs

logger = logging.getLogger("comfyui_timesaver.pass_routes")
LOG_PREFIX = "[TS Pass]"

try:
    from server import PromptServer
    _prompt_server = PromptServer.instance
except Exception:                           # noqa: BLE001 - not inside ComfyUI
    _prompt_server = None

try:
    from aiohttp import web
except ImportError:                         # pragma: no cover - ComfyUI ships it
    web = None

register_get, register_post = make_route_registrars(
    _prompt_server, lambda message: logger.warning("%s %s", LOG_PREFIX, message))

# Where a subscriber gets this month's code. Shown by the activation dialog,
# kept here so all three live in one place.
STORE_LINKS = {
    "boosty": "https://boosty.to/timesavervfx/posts/952ea374-f917-423e-8e36-75d794afd72b",
    "vk": "https://vk.com/wall-193385259_1117",
    "patreon": "https://www.patreon.com/posts/neiroseti-dlia-i-86746437",
}

# The launcher activates the pass in its own UI; if this ComfyUI runs inside a
# launcher build, the pass is right there and the person should not be asked
# to type the code twice. Paths are relative to ComfyUI's own folder.
_LAUNCHER_PASS_CANDIDATES = (
    Path("..") / "data" / "license.json",
    Path("..") / ".." / "data" / "license.json",
)


def _adopt_launcher_pass_if_any() -> None:
    """Best-effort: take over a pass the launcher already holds."""
    try:
        import folder_paths
        base = Path(folder_paths.base_path)
    except Exception:                       # noqa: BLE001
        return
    for relative in _LAUNCHER_PASS_CANDIDATES:
        candidate = (base / relative).resolve()
        if _pass.adopt_launcher_pass(candidate):
            logger.info("%s adopted the pass activated in the launcher.", LOG_PREFIX)
            return


def _json(payload, status: int = 200):
    return web.json_response(payload, status=status)


@register_get("/api/ts_pass/status")
async def pass_status(_request):
    """Current access state — what the studio renders locks and badges from.

    Testing mode can make this report an absent or lapsed pass while the real
    one stays on disk; on a user's machine there is no such file and the state
    is simply the truth (see nodes/_studio_dev.py).
    """
    _adopt_launcher_pass_if_any()
    state = _studio_packs.current_pass()
    state["links"] = STORE_LINKS
    state["dev"] = _studio_dev.read_dev()
    return _json(state)


@register_get("/api/ts_studio/dev")
async def studio_dev_status(_request):
    """Whether testing mode is on, and what it points at."""
    return _json(_studio_dev.read_dev())


@register_post("/api/ts_studio/dev")
async def studio_dev_set(request):
    """Change testing mode from the studio.

    Only reachable in the sense that matters: with no dev file the studio never
    draws the panel that calls this. Turning it on still requires the file, so
    this route cannot switch a user's machine into testing mode by itself.
    """
    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001
        return _json({"error": "expected a JSON body"}, status=400)

    if not _studio_dev.read_dev():
        return _json({"error": "testing mode is not enabled on this machine"},
                     status=403)
    try:
        state = _studio_dev.clear_dev() if body.get("off") else _studio_dev.write_dev(body)
    except (ValueError, RuntimeError) as error:
        return _json({"error": str(error)}, status=400)
    return _json(state)


@register_post("/api/ts_pass/activate")
async def pass_activate(request):
    """Redeem a code (or a token pasted directly, for machines offline)."""
    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001
        return _json({"error": "expected a JSON body"}, status=400)

    try:
        state = _pass.activate(str(body.get("code") or ""))
    except ValueError as error:
        # A wrong code is an ordinary answer, not a server fault: the UI shows
        # the text as-is next to the field.
        return _json({"error": str(error)}, status=400)
    except Exception as error:              # noqa: BLE001
        logger.warning("%s activation failed: %s", LOG_PREFIX, error)
        return _json({"error": str(error)}, status=500)

    state["links"] = STORE_LINKS
    logger.info("%s activated until %s", LOG_PREFIX, state.get("expiresAt"))
    return _json(state)


@register_post("/api/ts_pass/clear")
async def pass_clear(_request):
    """Forget the stored pass. Installed packs keep working — see _pass.py."""
    state = _pass.clear_pass()
    state["links"] = STORE_LINKS
    return _json(state)


@register_get("/api/ts_studio/packs")
async def studio_packs(_request):
    """Catalogue joined with what is installed — the showcase reads this."""
    catalog = _studio_packs.fetch_catalog()
    return _json(_studio_packs.describe_catalog(catalog))


@register_post("/api/ts_studio/packs/install")
async def studio_pack_install(request):
    """Fetch and unpack one pack. Refuses politely without the right pass."""
    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001
        return _json({"error": "expected a JSON body"}, status=400)

    pack_id = str(body.get("id") or "")
    catalog = _studio_packs.fetch_catalog()
    entries = (catalog.get("products", {}).get(_studio_packs.PRODUCT, {})
               or {}).get("packs", [])
    entry = next((e for e in entries if str(e.get("id")) == pack_id), None)
    if not entry:
        return _json({"error": "no such pack in the catalogue"}, status=404)

    try:
        stamp = _studio_packs.install_pack(entry)
    except PermissionError as error:
        return _json({"error": str(error)}, status=403)
    except Exception as error:              # noqa: BLE001
        logger.warning("%s install failed: %s", LOG_PREFIX, error)
        return _json({"error": str(error)}, status=500)
    return _json({"installed": stamp,
                  "packs": _studio_packs.describe_catalog(catalog)})


@register_post("/api/ts_studio/packs/remove")
async def studio_pack_remove(request):
    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001
        return _json({"error": "expected a JSON body"}, status=400)
    removed = _studio_packs.remove_pack(str(body.get("id") or ""))
    return _json({"removed": removed,
                  "packs": _studio_packs.describe_catalog(
                      _studio_packs.fetch_catalog())})


# Registers routes only; the pack's loader skips files with no node mappings.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
