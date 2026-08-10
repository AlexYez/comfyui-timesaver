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
import re
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

# Two places carry the same catalogue, and the studio reads whichever answers.
#
#   host    where the launcher's admin publishes, next to the paid workflows
#           it already sends. Primary, because that is one upload for the
#           author instead of two.
#   mirror  a public git repository. Nothing there is guessable-by-path, so
#           the packs it holds are encrypted; useful when the host is down or
#           blocked, which for this audience is not a rare event.
#
# Both are tried in order; whichever catalogue answers is also where the packs
# are fetched from, so a mirror never serves an address the other one hosts.
BASE_URLS = [
    "https://files.timesavervfx.com/ai/comfyui/studio",
    "https://raw.githubusercontent.com/AlexYez/ts-studio-packs/main",
]

# One address, set in the environment, replaces both — for a private mirror
# or an install fed from somewhere else entirely.
BASE_URL = os.environ.get("TS_STUDIO_PACKS_URL", "").rstrip("/")

# Product id, so Video and Audio studios can share the catalogue later.
PRODUCT = "image"

INSTALL_DIR = "ts-studio"
PACKS_SUBDIR = "packs"
WORKFLOWS_SUBDIR = "workflows"

_HTTP_TIMEOUT = 60
_MAX_PACK_BYTES = 64 * 1024 * 1024          # a pack is graphs and text, not models


def base_urls() -> list[str]:
    """Addresses to try, best first.

    Testing mode can put a folder on disk at the front (`file:///…/dist/studio`),
    which is the difference between trying a pack and publishing one to try it.
    On a user's machine there is no dev file and this is the list above.
    """
    forced = _studio_dev.catalog_base(BASE_URL)
    if forced:
        return [forced.rstrip("/")]
    return [url.rstrip("/") for url in BASE_URLS]


def base_url() -> str:
    """The address in use — the one a catalogue was last read from."""
    return _last_base or base_urls()[0]


def catalog_url() -> str:
    return f"{base_url()}/index.json"


# Which address answered last. Installs use it, so a pack is fetched from the
# same place its catalogue entry came from.
_last_base = ""


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
    buys before buying it. Tries each address in turn and remembers the one
    that answered, so the packs are fetched from the same place.
    """
    global _last_base

    candidates = [url] if url else [f"{base}/index.json" for base in base_urls()]
    for candidate in candidates:
        raw = _fetch_bytes(candidate, limit=4 * 1024 * 1024)
        if raw is None:
            continue
        try:
            catalog = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            logger.warning("%s %s is unreadable: %s", LOG_PREFIX, candidate, error)
            continue
        if not isinstance(catalog, dict):
            continue
        _last_base = candidate[: -len("/index.json")] if candidate.endswith("/index.json")             else _last_base
        catalog["source"] = _last_base
        return catalog
    return {"products": {}, "offline": True}


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


def pack_folder(root: Path, pack_id: str) -> Path:
    r"""Папка набора внутри `root` — и никогда снаружи.

    ⚠️ Раньше здесь стояло `root / pack_id.replace("/", "-")`. Прямой слэш это
    обезвреживало, а ОБРАТНЫЙ нет: на Windows `pathlib` считает разделителем и
    его, поэтому `..\..\..\Documents` уводил путь за пределы папки наборов —
    а по этому пути дальше шёл `rmtree`. Проверено на живых путях.

    Имя набора приходит из тела запроса и из каталога в интернете, то есть
    доверять ему нельзя ни там, ни там. Оставляем только буквы, цифры, точку,
    дефис и подчёркивание, а затем ещё раз убеждаемся, что получившийся путь
    лежит внутри корня: одно правило на установку и на удаление, чтобы они не
    разошлись.

    Args:
        root: папка установленных наборов.
        pack_id: идентификатор набора, как его назвал каталог или запрос.

    Returns:
        Путь к папке набора.

    Raises:
        ValueError: имя пустое или уводит за пределы корня.
    """
    raw = str(pack_id or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.replace("\\", "/")).strip("-")
    if not safe or set(safe) <= {"."}:
        raise ValueError(f"unsafe pack id: {pack_id!r}")
    target = (root / safe).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"unsafe pack id: {pack_id!r}") from error
    return target


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

    target = pack_folder(root, entry.get("id"))
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
    """Удалить установленный набор.

    ⚠️ Операция необратимая и приходит прямо из HTTP-запроса, поэтому имя
    проверяется (`pack_folder`) и результат записывается в журнал: молчаливое
    удаление каталога — не то, о чём человек должен узнавать по пропаже файлов.
    """
    root = installed_dir()
    if root is None:
        return False
    try:
        target = pack_folder(root, pack_id)
    except ValueError as error:
        logger.warning("%s refused to remove: %s", LOG_PREFIX, error)
        return False
    if not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    logger.info("%s removed pack %s", LOG_PREFIX, target.name)
    return True


# The catalogue that ships INSIDE the build: what a pack is, what it looks
# like and which tier it belongs to. Lives next to the graphs it describes, so
# the frontend reads it over HTTP as a static file and this module reads the
# same bytes off disk — one file, no second copy to drift.
LOCAL_CATALOG = (Path(__file__).resolve().parents[1]
                 / "js" / "image" / "studio" / "packs" / "catalog.json")

# Where the frontend reaches the same folder, for turning a cover's relative
# path into something an <img> can load.
LOCAL_CATALOG_BASE = "/extensions/comfyui-timesaver/image/studio/packs"


def local_catalog(path: Path | None = None) -> dict:
    """Packs described by the build itself.

    Absent or broken is not fatal: the studio then shows only what the remote
    catalogue offers, which is exactly what a machine with no build-time
    catalogue should see.
    """
    source = path or LOCAL_CATALOG
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"packs": [], "tiers": {}}
    except Exception as error:              # noqa: BLE001 - a bad file is not fatal
        logger.warning("%s %s is unreadable: %s", LOG_PREFIX, source, error)
        return {"packs": [], "tiers": {}}
    packs = data.get("packs")
    return {
        "packs": [p for p in packs if isinstance(p, dict)] if isinstance(packs, list) else [],
        "tiers": data.get("tiers") if isinstance(data.get("tiers"), dict) else {},
    }


def _family_names(entry: dict) -> list[str]:
    """Имена семейств пака, в каком бы виде их ни записали.

    Каталог сборки перечисляет их строками (`"krea2"`), а собранный каталог
    доставки — описаниями (`{"family": "krea2", "label": …, "modes": […]}`).
    Живут оба, и код, знающий только одну форму, тихо не находит ничего: ровно
    так устаревший набор и остался висеть в менеджере после разделения паков
    по моделям.
    """
    out = []
    for item in (entry.get("families") or []):
        name = item if isinstance(item, str) else (item or {}).get("family")
        if name:
            out.append(str(name))
    return out


def _cover_url(entry: dict, base: str = LOCAL_CATALOG_BASE) -> str:
    """A cover an <img> can load.

    The built-in catalogue stores relative paths — it sits next to its own
    covers — while a remote catalogue carries full URLs, because a relative
    one would resolve against ComfyUI instead of the host it came from.
    """
    cover = str(entry.get("cover") or entry.get("preview") or "")
    if not cover or "://" in cover or cover.startswith("/"):
        return cover
    return f"{base}/{cover}"


def merge_catalogs(local: dict, remote: dict, *, product: str = PRODUCT) -> list[dict]:
    """Every pack that exists, whether it ships here or arrives over the wire.

    The build's catalogue is the presentation (name, text, cover, tier); the
    remote one adds delivery (version, file, encryption). Where both describe
    the same id, delivery is layered onto presentation rather than replacing
    it: a machine that is offline must still know what a pack IS.
    """
    remote_packs = (remote.get("products", {}).get(product, {}) or {}).get("packs", [])
    by_id: dict[str, dict] = {}
    order: list[str] = []

    for entry in local.get("packs", []):
        pack_id = str(entry.get("id") or "")
        if not pack_id:
            continue
        resolved = {**entry, "builtin": True, "cover": _cover_url(entry)}
        # Пара «до/после» у апскейлера — такие же относительные пути, как
        # обложка: карточка вешает их на <img>, и относительный путь
        # разрешился бы относительно ComfyUI, а не папки пака.
        for key in ("before", "after"):
            if entry.get(key):
                resolved[key] = _cover_url({"cover": entry[key]})
        by_id[pack_id] = resolved
        order.append(pack_id)

    # Семейства, которые уже едут в сборке. Нужны, чтобы не показывать паки
    # прошлой раскладки: опубликованный `image/pro-2026-08` вёз krea2 вместе с
    # ideogram, и после разделения по моделям он остался висеть отдельной
    # карточкой «не установлен» в чужом уровне, предлагая то, что и так есть.
    shipped = {name for entry in local.get("packs", [])
               for name in _family_names(entry)}

    for entry in remote_packs:
        if not isinstance(entry, dict):
            continue
        pack_id = str(entry.get("id") or "")
        if not pack_id:
            continue
        known = by_id.get(pack_id)
        if known is None:
            families = set(_family_names(entry))
            if families and families <= shipped:
                logger.info("%s пропускаю пак %s: всё, что он везёт, уже в сборке",
                            LOG_PREFIX, pack_id)
                continue
            by_id[pack_id] = {**entry, "builtin": False, "cover": _cover_url(entry)}
            order.append(pack_id)
            continue
        merged = {**known}
        for key, value in entry.items():
            # Presentation the build already carries wins: it is translated,
            # it matches the graphs actually shipped, and it works offline.
            if key in ("name", "about", "cover", "preview") and known.get(key):
                continue
            merged[key] = value
        merged["builtin"] = True
        merged["cover"] = known.get("cover") or _cover_url(entry)
        by_id[pack_id] = merged

    return [by_id[pack_id] for pack_id in order]


def describe_catalog(catalog: dict, *, product: str = PRODUCT,
                     local: dict | None = None) -> dict:
    """Every pack, with its state, joined into what the manager renders.

    One call answers all four questions a card asks: does it exist, is it
    here, does the pass open it, and is the person showing it in the studio.
    """
    from . import _studio_pack_state

    state = current_pass()
    installed = read_installed()
    tier_held = int(state.get("tier") or 0) if state.get("state") == "active" else 0
    hidden = _studio_pack_state.read_state()
    ceiling = _studio_pack_state.view_tier()

    entries = []
    for entry in merge_catalogs(local if local is not None else local_catalog(),
                                catalog, product=product):
        pack_id = str(entry.get("id") or "")
        here = installed.get(pack_id)
        tier = int(entry.get("tier") or 0)
        hidden_why = _studio_pack_state.is_hidden(pack_id, tier, state=hidden,
                                                  ceiling=ceiling)
        entries.append({
            **entry,
            "installed": bool(here),
            "installedVersion": (here or {}).get("version"),
            # A built-in pack is present by definition; only a delivered one
            # can be behind on a version.
            "updateAvailable": bool(here and entry.get("version")
                                    and here.get("version") != entry.get("version")),
            "open": not tier or tier_held >= tier,
            "present": bool(here) or bool(entry.get("builtin")),
            "hidden": hidden_why,
            "enabled": not hidden_why,
        })
    return {
        "product": product,
        "pass": state,
        "packs": entries,
        "tiers": (local if local is not None else local_catalog()).get("tiers", {}),
        "viewTier": ceiling,
        "disabled": hidden["disabled"],
        "offline": bool(catalog.get("offline")),
        "updatedAt": catalog.get("updatedAt"),
    }
