"""Testing mode — so a pack can be tried before anyone publishes it.

Everything about the subscription is remote by nature: a catalogue on a host,
an archive behind a key, a pass that expires. That is right for a user and
miserable for testing, where each attempt would otherwise mean a commit, a
push and a wait.

This module makes the remote parts point wherever the author says, and lets
the studio be shown as if the pass were absent or expired without touching the
real one. Two things keep it honest:

    * it is OFF unless `user/default/ts-studio/dev.json` exists, so a user
      never meets any of it — nothing here is reachable, drawn, or read;
    * it changes only where things are read FROM and what state is REPORTED.
      The rules themselves — signature, expiry, tiers, unpacking — stay exactly
      as a user's machine runs them, or the testing would prove nothing.

Written by hand or by `tools/studio_pack.py dev`:

    {
      "localUrl": "file:///D:/…/dist/studio",   a build to read instead
      "source":   "live" | "local",             which of the two is in use
      "simulate": "off" | "none" | "expired",   how the pass should appear
      "viewTier": null | 0 | 2 | 3,             ceiling: packs above it are
                                                treated as undelivered
      "label":    "local dist"                  shown in the studio
    }

The local address is remembered even while "live" is selected, so switching
back and forth is one click rather than one path retyped.

`viewTier` exists because the whole build currently ships every family, so
without it the author cannot see what a Free user sees — the packs screen
would be honest and the studio itself would not. It HIDES and never grants:
read `_studio_pack_state.py` for why that distinction is the whole point.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("comfyui_timesaver.studio_dev")
LOG_PREFIX = "[TS Studio dev]"

DEV_DIR = "ts-studio"
DEV_FILE = "dev.json"

# How the pass may be made to look. "off" means: report the truth.
SIMULATIONS = ("off", "none", "expired")
SOURCES = ("live", "local")


def dev_path() -> Path | None:
    try:
        import folder_paths
    except ImportError:
        return None
    return Path(folder_paths.get_user_directory()) / "default" / DEV_DIR / DEV_FILE


def read_dev() -> dict:
    """The testing settings, or an empty dict when the mode is off.

    Never raises and never guesses: an unreadable file is treated as absent,
    because a broken dev file must not change how the studio behaves.
    """
    path = dev_path()
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001
        logger.warning("%s %s is unreadable; testing mode stays off.",
                       LOG_PREFIX, path)
        return {}
    if not isinstance(data, dict):
        return {}
    simulate = str(data.get("simulate") or "off")
    source = str(data.get("source") or "live")
    local_url = str(data.get("localUrl") or "").rstrip("/")
    return {
        "enabled": True,
        "localUrl": local_url,
        "source": source if source in SOURCES and (local_url or source == "live")
                  else "live",
        "simulate": simulate if simulate in SIMULATIONS else "off",
        "viewTier": _read_view_tier(data.get("viewTier")),
        "label": str(data.get("label") or ""),
        "path": str(path),
    }


def _read_view_tier(raw) -> int | None:
    """A ceiling, or None for "show everything".

    A junk value means no ceiling rather than the lowest one: a typo in a
    hand-written file must not quietly hide half the studio.
    """
    if raw is None or raw == "":
        return None
    try:
        tier = int(raw)
    except (TypeError, ValueError):
        return None
    return tier if tier >= 0 else None


def write_dev(settings: dict) -> dict:
    """Turn testing mode on, or change it. Creates the file."""
    path = dev_path()
    if path is None:
        raise RuntimeError("ComfyUI's user directory is unavailable")
    simulate = str(settings.get("simulate") or "off")
    if simulate not in SIMULATIONS:
        raise ValueError(f"simulate must be one of {SIMULATIONS}")
    source = str(settings.get("source") or "live")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    body = {
        "localUrl": str(settings.get("localUrl") or "").rstrip("/"),
        "source": source,
        "simulate": simulate,
        "viewTier": _read_view_tier(settings.get("viewTier")),
        "label": str(settings.get("label") or ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("%s testing mode: %s", LOG_PREFIX, body)
    return read_dev()


def clear_dev() -> dict:
    """Back to a user's machine exactly."""
    path = dev_path()
    if path is not None and path.is_file():
        path.unlink()
    return read_dev()


def catalog_base(default: str) -> str:
    """Where packs are read from — a local build while testing, else the host."""
    dev = read_dev()
    if dev.get("source") == "local" and dev.get("localUrl"):
        return dev["localUrl"]
    return default


def apply_pass_simulation(state: dict) -> dict:
    """Show the pass as a user without one — or with a lapsed one — would see it.

    Only the reported state changes; the stored pass is untouched, so turning
    the simulation off restores it without retyping a code. The simulated
    state carries no secrets, which is what makes the freemium path honest to
    test: a paid pack genuinely cannot be fetched while it is on.
    """
    dev = read_dev()
    simulate = dev.get("simulate", "off")
    if not dev or simulate == "off":
        return state
    simulated = dict(state)
    simulated["tier"] = 0
    simulated["secrets"] = {}
    simulated["simulated"] = simulate
    if simulate == "none":
        simulated["state"] = "none"
        simulated.pop("expiresAt", None)
        simulated.pop("daysLeft", None)
    else:
        simulated["state"] = "expired"
        simulated["daysLeft"] = 0
    return simulated
