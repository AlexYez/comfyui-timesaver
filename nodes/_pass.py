"""Subscription pass — shared access state for every TS studio.

One pass serves the whole family of studios (Image today; Video and Audio
later), so nothing here mentions a product: a studio asks "is tier 2 open?"
and gets an answer.

WHAT A PASS IS AND IS NOT
    It gates DOWNLOADS of paid packs, nothing else. Everything already
    installed keeps working without a pass and after one expires. A person who
    stops subscribing does not lose a working tool — they stop receiving new
    packs. That is the whole model, and the reason there is no check anywhere
    near the run path.

THE SAME MECHANISM AS THE LAUNCHER
    The launcher (Electron) already implements this and is the source of
    truth for the format; this module is its Python twin so the pack works
    for people who installed it from the registry without the launcher.
    Deliberately identical: same Ed25519 public key, same token layout
    (`base64url(payload).base64url(signature)`), same 35-day window measured
    from ACTIVATION, same `notAfter` ceiling, same ordered tier scale.

    Codes are not per-person: three codes a month (one per tier) are posted
    behind the paywalls of Boosty / Patreon / VK. The platform is the paywall.
    Because a code is public to subscribers, expiry IS the proof that someone
    is still subscribed — which is why nothing here ever renews itself.

WHY THE WINDOW STARTS AT ACTIVATION
    Someone subscribing on the 28th must still get a month. So the expiry is
    `min(activated + 35 days, notAfter)`: 35 rather than 30 leaves room until
    the next month's post appears, and `notAfter` stops a year-old code from
    working forever.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("comfyui_timesaver.pass")
LOG_PREFIX = "[TS Pass]"

# Verification key of the signing pair (raw Ed25519, 32 bytes, base64). The
# private half exists only on the author's machine. Changing this constant
# invalidates every issued key, so it is shared with the launcher verbatim.
PUBLIC_KEY_B64 = "FRR7ltKd0mhl/cujKYAreeWtDtCjxRoy0ROcuvl5uH0="

# Days a pass lasts from the moment it is entered. See the module docstring.
ACTIVE_DAYS = 35

TIER_NAMES = {1: "base", 2: "pro", 3: "ultimate"}

# Where key material is published. Overridable in the environment for a
# mirror or a test rig; read through base_url() rather than captured at import,
# because a default argument bound at definition time cannot be overridden at
# all — which is how it silently ignored every override until now.
BASE_URL = os.environ.get(
    "TS_PASS_URL", "https://files.timesavervfx.com/ai/comfyui/launcher").rstrip("/")
KEYS_PATH = "keys"


def base_url() -> str:
    return BASE_URL.rstrip("/")


def revoked_url() -> str:
    return f"{base_url()}/revoked.txt"

# Product-neutral on purpose: Video and Audio studios read the same file.
PASS_DIR = "ts-pass"
PASS_FILE = "pass.json"

_HTTP_TIMEOUT = 20


# ── pure helpers (unit-tested without a filesystem or network) ───────────── #

def normalize_code(raw) -> str | None:
    """A code as a person actually pastes it: spaces, newlines, odd case.

    Returns None when nothing usable is left, so callers can tell "empty" from
    "wrong" without guessing.
    """
    text = re.sub(r"[^A-Z0-9-]", "", str(raw or "").upper())
    return text if len(text) >= 8 else None


def code_to_key_file(code: str) -> str:
    """Token filename for a code: sha256 hex, so the directory is unlistable."""
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


def _b64url_to_bytes(text: str) -> bytes:
    padded = str(text).replace("-", "+").replace("_", "/")
    padded += "=" * (-len(padded) % 4)
    return base64.b64decode(padded)


def verify_token(token: str, public_key_b64: str | None = None) -> dict | None:
    """Parsed payload of a signed token, or None.

    Never raises: a corrupt file on disk must not break the studio, it must
    simply mean "no pass".
    """
    # Read now, not at definition: a default argument would freeze the key at
    # import and quietly ignore any later change (a fork, a rotation, a test).
    public_key_b64 = public_key_b64 or PUBLIC_KEY_B64
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        logger.warning("%s cryptography is unavailable; a pass cannot be checked.",
                       LOG_PREFIX)
        return None
    try:
        head, _, tail = str(token or "").strip().partition(".")
        if not head or not tail:
            return None
        payload_raw = _b64url_to_bytes(head)
        signature = _b64url_to_bytes(tail)
        if len(signature) != 64:
            return None
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
        try:
            key.verify(signature, payload_raw)
        except InvalidSignature:
            return None
        payload = json.loads(payload_raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        if not payload.get("kid") or not isinstance(payload.get("tier"), int):
            return None
        return payload
    except Exception:                       # noqa: BLE001 - never fatal
        return None


def _parse_iso(text) -> datetime | None:
    if not text:
        return None
    try:
        cleaned = str(text).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def compute_expiry(payload: dict, activated_at_iso: str,
                   active_days: int = ACTIVE_DAYS) -> datetime | None:
    """Whichever comes first: the window from activation, or the key's ceiling."""
    activated = _parse_iso(activated_at_iso)
    if activated is None:
        return None
    window = activated + timedelta(days=active_days)
    ceiling = _parse_iso((payload or {}).get("notAfter"))
    return min(window, ceiling) if ceiling else window


def describe(payload: dict | None, activated_at_iso: str, *,
             revoked: bool = False, now: datetime | None = None) -> dict:
    """The state a UI renders: tier, name, expiry, days left.

    An expired or revoked pass reports tier 0 — "nothing new to download" —
    rather than an error, because that is exactly what it means.
    """
    now = now or datetime.now(timezone.utc)
    if not payload:
        return {"state": "none", "tier": 0, "tierName": None,
                "expiresAt": None, "daysLeft": 0}
    if revoked:
        return {"state": "revoked", "tier": 0, "tierName": None,
                "code": payload.get("kid"), "expiresAt": None, "daysLeft": 0}
    expires = compute_expiry(payload, activated_at_iso)
    if expires is None or expires <= now:
        return {"state": "expired", "tier": 0,
                "tierName": TIER_NAMES.get(payload.get("tier")),
                "code": payload.get("kid"),
                "expiresAt": expires.isoformat() if expires else None,
                "daysLeft": 0}
    return {
        "state": "active",
        "tier": int(payload.get("tier") or 0),
        "tierName": TIER_NAMES.get(payload.get("tier")),
        "code": payload.get("kid"),
        "expiresAt": expires.isoformat(),
        "daysLeft": max(0, (expires - now).days),
        # Paid files live under an unguessable path named by the tier secret,
        # so a studio needs these to build a download URL.
        "secrets": {str(k): v for k, v in (payload.get("sec") or {}).items()},
    }


# ── stored pass ─────────────────────────────────────────────────────────── #

def pass_path() -> Path | None:
    """Where the pass lives: ComfyUI's user directory, shared by all studios."""
    try:
        import folder_paths
    except ImportError:
        return None
    base = Path(folder_paths.get_user_directory()) / "default" / PASS_DIR
    return base / PASS_FILE


def read_pass() -> dict:
    """Current pass state. Always answers; never raises."""
    path = pass_path()
    if path is None or not path.is_file():
        return describe(None, "")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001
        logger.warning("%s the stored pass is unreadable; treating as absent.",
                       LOG_PREFIX)
        return describe(None, "")
    payload = verify_token(stored.get("token", ""))
    return describe(payload, stored.get("activatedAt", ""),
                    revoked=bool(stored.get("revoked")))


def write_pass(token: str, activated_at_iso: str | None = None) -> dict:
    """Store a verified token and start its window now.

    Re-entering the same code refreshes the activation date. That is renewal
    within the key's own ceiling, not a way around it: `notAfter` still caps
    the result.
    """
    payload = verify_token(token)
    if not payload:
        raise ValueError("token is not signed by the expected key")
    path = pass_path()
    if path is None:
        raise RuntimeError("ComfyUI's user directory is unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    activated = activated_at_iso or datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps({
        "token": token,
        "activatedAt": activated,
        "kid": payload.get("kid"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("%s pass activated: %s until %s", LOG_PREFIX, payload.get("kid"),
                compute_expiry(payload, activated))
    return read_pass()


def clear_pass() -> dict:
    path = pass_path()
    if path is not None and path.is_file():
        path.unlink()
    return read_pass()


def has_tier(required: int) -> bool:
    """Does the current pass open a given tier? Tier 0 means free — always."""
    if not required:
        return True
    state = read_pass()
    return state.get("state") == "active" and int(state.get("tier") or 0) >= int(required)


# ── activation over the network (explicit user action only) ─────────────── #

def _fetch(url: str) -> str | None:
    """Plain GET. Returns None on any failure — offline is not an error here."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TS-Studio"})
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return response.read().decode("utf-8", "replace").strip()
    except (urllib.error.URLError, OSError, ValueError) as error:
        logger.info("%s could not fetch %s (%s)", LOG_PREFIX, url, error)
        return None


def is_revoked(kid: str) -> bool:
    """Was this key pulled after issue? A network failure means "assume fine"."""
    body = _fetch(revoked_url())
    if body is None:
        return False
    return any(line.strip() == kid for line in body.splitlines())


def activate(code_or_token: str, *, url_base: str | None = None) -> dict:
    """Turn what the user typed into a stored pass.

    Accepts either a subscription code (fetches its token) or a token pasted
    directly — the offline path for a machine that cannot reach the internet.
    """
    raw = str(code_or_token or "").strip()
    if raw.count(".") == 1 and len(raw) > 80:
        if not verify_token(raw):
            raise ValueError("this token is not valid")
        return write_pass(raw)

    code = normalize_code(raw)
    if not code:
        raise ValueError("the code is too short")
    token = _fetch(f"{(url_base or base_url()).rstrip('/')}"
                   f"/{KEYS_PATH}/{code_to_key_file(code)}.txt")
    if not token:
        raise ValueError("no key with this code was found")
    if not verify_token(token):
        raise ValueError("the key failed its signature check")
    return write_pass(token)


# ── the launcher bridge ─────────────────────────────────────────────────── #

def adopt_launcher_pass(launcher_pass: Path) -> dict | None:
    """Take over a pass the launcher already activated, if it is newer.

    The launcher writes the same shape next to its own data. Reading it means
    a person who entered their code there never types it again here.
    """
    try:
        if not launcher_pass.is_file():
            return None
        stored = json.loads(launcher_pass.read_text(encoding="utf-8"))
        token = stored.get("token", "")
        if not verify_token(token):
            return None
        mine = pass_path()
        if mine and mine.is_file():
            current = json.loads(mine.read_text(encoding="utf-8"))
            if current.get("token") == token:
                return None                 # already the same pass
        return write_pass(token, stored.get("activatedAt"))
    except Exception:                       # noqa: BLE001
        return None
