"""Which packs a person wants to see — and which ones this machine pretends
not to have.

TWO DIFFERENT ANSWERS, ONE QUESTION
    A studio decides what to draw by asking "is this family active?". Two
    independent things can answer no, and conflating them has already cost a
    redesign elsewhere:

        turned off   the person's own choice. Six models in the picker when
                     three are used is noise, so a pack can be hidden without
                     being deleted. Nothing is removed from disk: the graphs
                     stay, the models stay, and turning it back on is one
                     click. Deleting models is a separate act, later.

        above tier   testing only. Everything ships in one build while the
                     studio is being finished, which means the author cannot
                     see what a Free user sees. A ceiling makes families above
                     it behave exactly as undelivered ones: grey in the picker,
                     offered by the packs screen, absent from the rail.

    Both funnel into `active_packs()`, so the rest of the studio never has to
    know which of the two hid something.

WHAT THIS IS NOT
    Not a licence check. The ceiling is a VIEW: it hides, it never grants.
    Fetching a paid pack still needs a real pass with the tier secret in it
    (`nodes/_pass.py`), and nothing here can soften that — same rule as
    `_studio_dev.py`, for the same reason: a test that proves something other
    than the shipped behaviour proves nothing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import _studio_dev

logger = logging.getLogger("comfyui_timesaver.studio_pack_state")
LOG_PREFIX = "[TS Packs]"

STATE_DIR = "ts-studio"
STATE_FILE = "packs-state.json"

# Ladder shared with `_pass.TIER_NAMES`: 0 free, 2 pro, 3 ultimate. A pass of a
# higher tier opens everything below it, so the ceiling is a single number.
TIER_FREE = 0
TIER_PRO = 2
TIER_ULTIMATE = 3

TIER_BY_NAME = {
    "free": TIER_FREE,
    "pro": TIER_PRO,
    "ultimate": TIER_ULTIMATE,
}

# What the author sees: no ceiling at all.
TIER_AUTHOR = None


def state_path() -> Path | None:
    try:
        import folder_paths
    except ImportError:
        return None
    return Path(folder_paths.get_user_directory()) / "default" / STATE_DIR / STATE_FILE


def read_state() -> dict:
    """Turned-off packs, by id. An unreadable file means "nothing is off".

    Never raises: a broken preferences file must not take the studio with it.
    """
    path = state_path()
    if path is None or not path.is_file():
        return {"disabled": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001 - see the docstring
        logger.warning("%s %s is unreadable; every pack stays visible.",
                       LOG_PREFIX, path)
        return {"disabled": []}
    disabled = data.get("disabled") if isinstance(data, dict) else None
    if not isinstance(disabled, list):
        return {"disabled": []}
    return {"disabled": [str(item) for item in disabled if str(item).strip()]}


def write_state(state: dict) -> dict:
    """Store the choice. Creates the file and its folder."""
    path = state_path()
    if path is None:
        raise RuntimeError("ComfyUI's user directory is unavailable")
    disabled = sorted({str(item) for item in (state.get("disabled") or []) if str(item).strip()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"disabled": disabled}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return {"disabled": disabled}


def set_enabled(pack_id: str, enabled: bool) -> dict:
    """Show or hide one pack. Returns the state as it now stands."""
    pack_id = str(pack_id or "").strip()
    if not pack_id:
        raise ValueError("a pack id is required")
    state = read_state()
    disabled = set(state["disabled"])
    if enabled:
        disabled.discard(pack_id)
    else:
        disabled.add(pack_id)
    written = write_state({"disabled": sorted(disabled)})
    logger.info("%s %s is now %s", LOG_PREFIX, pack_id,
                "visible" if enabled else "hidden")
    return written


def view_tier() -> int | None:
    """The ceiling in force, or None when there is none.

    Lives in the testing file rather than the preferences one on purpose: it
    is not a preference. A user's machine has no `dev.json`, so this is None
    for everyone but the author, and the ceiling cannot be reached by anything
    a user can press.
    """
    dev = _studio_dev.read_dev()
    if not dev:
        return None
    raw = dev.get("viewTier")
    if raw is None or raw == "":
        return None
    try:
        tier = int(raw)
    except (TypeError, ValueError):
        return None
    return tier if tier >= 0 else None


def is_hidden(pack_id: str, tier: int, *, state: dict | None = None,
              ceiling: int | None = None) -> str:
    """Why this pack is not in the studio right now — or "" when it is.

    Answers with the REASON rather than a boolean, because the two hidings
    look different on screen: one offers a switch back, the other offers the
    pack.

    @returns "off" | "tier" | ""
    """
    known = state if state is not None else read_state()
    if str(pack_id) in set(known.get("disabled") or []):
        return "off"
    limit = ceiling if ceiling is not None else view_tier()
    if limit is not None and int(tier or 0) > limit:
        return "tier"
    return ""


def active_families(packs: list[dict]) -> list[str]:
    """Families the studio should actually offer, given both answers above.

    `packs` are catalogue entries — each carries its tier and the families it
    brings. A pack that is hidden takes its families with it.
    """
    state = read_state()
    ceiling = view_tier()
    out: list[str] = []
    for pack in packs:
        if is_hidden(pack.get("id", ""), int(pack.get("tier") or 0),
                     state=state, ceiling=ceiling):
            continue
        out.extend(str(name) for name in (pack.get("families") or []))
    return out
