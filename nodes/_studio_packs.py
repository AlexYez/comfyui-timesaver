"""Content packs — the paid half of a studio, delivered as data.

A pack is a zip of backend graphs and presets. Nothing executable ever
travels: the studio reads these files exactly as it reads the ones shipped in
the package, so a pack cannot do anything a workflow file could not.

WHERE THINGS LIVE
    catalogue   public, unencrypted — this IS the showcase. Someone without a
                pass sees every model, its preview and what it does; that is
                the point of showing it.
    packs       under a path named by the tier secret, which arrives inside
                the pass. Public hosting, unguessable address — the same
                scheme the launcher already uses for paid workflows.
    installed   ComfyUI's user directory, where the studio's user tier
                already looks for backends and prefers them over built-ins.

Once installed, a pack keeps working forever: nothing here checks a pass when
reading local files, only when fetching new ones.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import _pass
from . import _studio_dev

logger = logging.getLogger("comfyui_timesaver.studio_packs")
LOG_PREFIX = "[TS Packs]"

# Delivery lives in its own public repository: the catalogue has to be
# readable by everyone (it is the showcase), and the paid archives beside it
# are encrypted, so a listable tree costs nothing. An install pointed at a
# different host — the author's own file storage, a mirror — only needs this
# overridden in the environment.
BASE_URL = os.environ.get(
    "TS_STUDIO_PACKS_URL",
    "https://raw.githubusercontent.com/AlexYez/ts-studio-packs/main").rstrip("/")

# Product id, so Video and Audio studios can share the catalogue later.
PRODUCT = "image"

INSTALL_DIR = "ts-studio"
PACKS_SUBDIR = "packs"
WORKFLOWS_SUBDIR = "workflows"

_HTTP_TIMEOUT = 60
_MAX_PACK_BYTES = 64 * 1024 * 1024          # a pack is graphs and text, not models


def base_url() -> str:
    """Where packs are read from right now.

    Testing mode can point this at a folder on disk (`file:///…/dist/studio`),
    which is the difference between trying a pack and publishing one to try it.
    On a user's machine there is no dev file and this is the constant above.
    """
    return _studio_dev.catalog_base(BASE_URL)


def catalog_url() -> str:
    return f"{base_url()}/index.json"


def current_pass() -> dict:
    """The pass as the studio should see it, simulation included."""
    return _studio_dev.apply_pass_simulation(_pass.read_pass())


def _user_root() -> Path | None:
    try:
        import folder_paths
    except ImportError:
        return None
    return Path(folder_paths.get_user_directory()) / "default" / INSTALL_DIR


def installed_dir() -> Path | None:
    root = _user_root()
    return None if root is None else root / WORKFLOWS_SUBDIR


def _fetch_bytes(url: str, limit: int = _MAX_PACK_BYTES) -> bytes | None:
    """GET with a ceiling. Returns None on any failure — offline is not fatal."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TS-Studio"})
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            logger.warning("%s %s is larger than the %d MB ceiling; refused.",
                           LOG_PREFIX, url, limit // (1024 * 1024))
            return None
        return data
    except (urllib.error.URLError, OSError, ValueError) as error:
        logger.info("%s could not fetch %s (%s)", LOG_PREFIX, url, error)
        return None


def fetch_catalog(url: str | None = None) -> dict:
    """The public catalogue: what exists, what it looks like, what is new.

    Readable with no pass at all — a person has to see what a subscription
    buys before buying it.
    """
    raw = _fetch_bytes(url or catalog_url(), limit=4 * 1024 * 1024)
    if raw is None:
        return {"products": {}, "offline": True}
    try:
        catalog = json.loads(raw.decode("utf-8"))
        return catalog if isinstance(catalog, dict) else {"products": {}}
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        logger.warning("%s the catalogue is unreadable: %s", LOG_PREFIX, error)
        return {"products": {}}


def read_installed() -> dict:
    """Which packs are on disk, by id, with the version and install date."""
    root = installed_dir()
    if root is None or not root.is_dir():
        return {}
    found = {}
    for stamp in sorted(root.glob("*/pack.json")):
        try:
            data = json.loads(stamp.read_text(encoding="utf-8"))
            found[str(data.get("id") or stamp.parent.name)] = {
                "id": data.get("id"),
                "version": data.get("version"),
                "installedAt": data.get("installedAt"),
                "path": stamp.parent.name,
            }
        except Exception:                   # noqa: BLE001 - a bad stamp is not fatal
            continue
    return found


def pack_url(entry: dict, secrets: dict, *, base: str | None = None,
             product: str = PRODUCT) -> str | None:
    """Address of a pack's archive, or None when the pass does not open it.

    Two shapes, because the two places a pack can be hosted leak differently:

    * plain — the tier secret IS the path segment, so a pass without it cannot
      construct the URL. Right for a file host that does not list directories.
    * encrypted — the address is public and boring, and the secret is the key
      instead. Required anywhere the file tree can be browsed (a public git
      repository), where a secret in the path would be printed on the page.
    """
    base = (base or base_url()).rstrip("/")
    tier = int(entry.get("tier") or 0)
    if not tier:
        return f"{base}/free/{product}/{entry['file']}"
    secret = (secrets or {}).get(str(tier))
    if not secret:
        return None
    if entry.get("enc"):
        return f"{base}/paid/{product}/{entry['file']}"
    return f"{base}/paid/{secret}/{product}/{entry['file']}"


ENC_MAGIC = b"TSPK1"
_NONCE_BYTES = 12


def pack_key(secret: str) -> bytes:
    """AES key for a tier secret. Domain-separated so it is this and nothing
    else the secret unlocks."""
    import hashlib
    return hashlib.sha256(f"ts-studio-pack:{secret}".encode("utf-8")).digest()


def decrypt_pack(blob: bytes, secret: str) -> bytes:
    """Undo `encrypt_pack`. Raises ValueError on a wrong key or a torn file."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not blob.startswith(ENC_MAGIC):
        raise ValueError("this is not an encrypted pack")
    nonce = blob[len(ENC_MAGIC):len(ENC_MAGIC) + _NONCE_BYTES]
    try:
        return AESGCM(pack_key(secret)).decrypt(
            nonce, blob[len(ENC_MAGIC) + _NONCE_BYTES:], ENC_MAGIC)
    except InvalidTag as error:
        raise ValueError("the pack did not open with this pass") from error


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Members that are plain files under the archive root.

    A pack is data from the internet: absolute paths, parent traversal and
    symlinks are refused rather than sanitised, because a pack has no reason
    to contain them.
    """
    safe = []
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        if member.is_dir():
            continue
        if name.startswith("/") or ".." in Path(name).parts or ":" in name:
            raise ValueError(f"pack contains an unsafe path: {member.filename}")
        if (member.external_attr >> 16) & 0o120000 == 0o120000:
            raise ValueError(f"pack contains a symlink: {member.filename}")
        if Path(name).suffix.lower() not in {".json", ".md", ".txt", ".webp", ".png"}:
            raise ValueError(f"pack contains an unexpected file: {member.filename}")
        safe.append(member)
    return safe


def install_pack(entry: dict, *, base: str | None = None,
                 product: str = PRODUCT, data: bytes | None = None) -> dict:
    """Fetch and unpack one pack. Returns its stamp.

    `data` exists for tests and for an archive a person downloaded by hand.
    """
    root = installed_dir()
    if root is None:
        raise RuntimeError("ComfyUI's user directory is unavailable")

    if data is None:
        state = current_pass()
        tier = int(entry.get("tier") or 0)
        if tier and not (state.get("state") == "active"
                         and int(state.get("tier") or 0) >= tier):
            raise PermissionError("this pack needs an active subscription")
        url = pack_url(entry, state.get("secrets") or {},
                       base=base, product=product)
        if not url:
            raise PermissionError("this pass does not carry the key for that tier")
        data = _fetch_bytes(url)
        if data is None:
            raise RuntimeError("the pack could not be downloaded")
        if entry.get("enc"):
            secret = (state.get("secrets") or {}).get(str(tier))
            if not secret:
                raise PermissionError("this pass does not carry the key for that tier")
            data = decrypt_pack(data, secret)

    target = root / str(entry["id"]).replace("/", "-")
    staging = target.with_name(target.name + ".incoming")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = _safe_members(archive)
            for member in members:
                archive.extract(member, staging)
        stamp = {
            "id": entry.get("id"),
            "version": entry.get("version"),
            "tier": entry.get("tier", 0),
            "installedAt": datetime.now(timezone.utc).isoformat(),
            "files": len(members),
        }
        (staging / "pack.json").write_text(
            json.dumps(stamp, ensure_ascii=False, indent=1), encoding="utf-8")
        # Swap only once the archive is known good: a failed install must not
        # leave a half-written pack where the studio will read it.
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("%s installed %s v%s (%d files)", LOG_PREFIX, stamp["id"],
                stamp["version"], stamp["files"])
    return stamp


def remove_pack(pack_id: str) -> bool:
    root = installed_dir()
    if root is None:
        return False
    target = root / str(pack_id).replace("/", "-")
    if not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True


def describe_catalog(catalog: dict, *, product: str = PRODUCT) -> dict:
    """Catalogue joined with what is installed and what the pass opens.

    One call gives the showcase everything it renders: entries, their state
    and whether an update is waiting.
    """
    state = current_pass()
    installed = read_installed()
    tier_held = int(state.get("tier") or 0) if state.get("state") == "active" else 0

    entries = []
    for entry in (catalog.get("products", {}).get(product, {}) or {}).get("packs", []):
        pack_id = str(entry.get("id") or "")
        here = installed.get(pack_id)
        tier = int(entry.get("tier") or 0)
        entries.append({
            **entry,
            "installed": bool(here),
            "installedVersion": (here or {}).get("version"),
            "updateAvailable": bool(here and here.get("version") != entry.get("version")),
            "open": not tier or tier_held >= tier,
        })
    return {
        "product": product,
        "pass": state,
        "packs": entries,
        "offline": bool(catalog.get("offline")),
        "updatedAt": catalog.get("updatedAt"),
    }
