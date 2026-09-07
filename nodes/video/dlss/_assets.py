"""Runtime files of TS DLSS Upscaler: where they live and how they arrive.

Everything the node runs on is downloaded on first use into ``models/DLSS`` and
never shipped with the pack: ``nvngx_dlssnr.dll`` and ``nvngx_dlss.dll`` are
NVIDIA binaries under the NVIDIA DLSS licence, the add-on and ``dxgi.dll`` are
ReShade/RenoDX. The licence texts are taken out of the same archive and kept
next to the binaries.

⚠️ The layout is dictated by the worker and the add-on and must NOT be
flattened::

    models/DLSS/
      host/  nvngx.dll  dxgi.dll  renodx-dlss5.addon64  nvngx_dlssnr.dll
             ReShade.ini (written by the worker)  ReShade.log (written by ReShade)
      dlss/  nvngx_dlss.dll

⚠️ ``host/nvngx.dll`` is an EXECUTABLE with a .dll name. That is not a mistake:
the signed NVIDIA snippet checks the image name of its caller, so the worker has
to be called that. It is launched with its working directory set to ``host/``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable

from ..._deps import TSDependencyManager

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

#: Folder under ``models/`` — the name the user asked for.
MODEL_FOLDER_NAME = "DLSS"

#: Upstream release the runtime is taken from (481 MB; only nine entries are kept).
RUNTIME_URL = (
    "https://github.com/Merserk/dlss5-visual-enhancer/releases/download/v5.0/"
    "DLSS.5.Visual.Enhancer.v5.0.zip"
)
RUNTIME_SIZE_MB = 481

#: zip entry -> path relative to the runtime root.
EXTRACT = {
    "bin/runtime/host/nvngx.dll": "host/nvngx.dll",
    "bin/runtime/host/dxgi.dll": "host/dxgi.dll",
    "bin/runtime/host/renodx-dlss5.addon64": "host/renodx-dlss5.addon64",
    "bin/runtime/host/nvngx_dlssnr.dll": "host/nvngx_dlssnr.dll",
    "bin/runtime/host/LICENSE-ReShade.txt": "host/LICENSE-ReShade.txt",
    "bin/runtime/host/LICENSE-RenoDX.txt": "host/LICENSE-RenoDX.txt",
    "bin/runtime/host/LICENSE-NVIDIA-DLSS.txt": "host/LICENSE-NVIDIA-DLSS.txt",
    "bin/runtime/dlss/nvngx_dlss.dll": "dlss/nvngx_dlss.dll",
    "bin/runtime/dlss/LICENSE-NVIDIA-DLSS.txt": "dlss/LICENSE-NVIDIA-DLSS.txt",
}

#: Without these the node cannot run; their absence triggers the download.
REQUIRED = (
    "host/nvngx.dll",
    "host/dxgi.dll",
    "host/renodx-dlss5.addon64",
    "host/nvngx_dlssnr.dll",
    "dlss/nvngx_dlss.dll",
)

# ⚠️ A hash mismatch WARNS and does not refuse: upstream rebuilds the release
# now and then, and refusing would turn a working machine into a broken one for
# a reason the user cannot fix. The worker itself is not pinned at all (its
# build varies); it is only checked for being a plausible size.
SHA256 = {
    "host/dxgi.dll": "0CEE63F9C9F13F3AC909C5B4903F4DBB4B719A7AB3B4F13B0DEAF83C814B94F7",
    "host/renodx-dlss5.addon64": "D5ADF82EB44B065F4C590AC91FE824BAB07AFEA0EB9F994BDE936710C8593952",
    "host/nvngx_dlssnr.dll": "6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927",
    "dlss/nvngx_dlss.dll": "C85F971CE023C9F3492FC7455F0B01A24BA18EA39636407A846902C4360B0B7E",
}
MIN_WORKER_BYTES = 1024 * 1024

#: Whose the files are. Printed before the download, and kept next to the
#: binaries as the licence texts the archive carries.
COMPONENT_LICENCES = (
    ("host/nvngx_dlssnr.dll, dlss/nvngx_dlss.dll",
     "NVIDIA, proprietary (NVIDIA RTX SDKs License)"),
    ("host/dxgi.dll", "ReShade, BSD-3-Clause"),
    ("host/renodx-dlss5.addon64", "RenoDX add-on, its own distribution terms"),
    ("host/nvngx.dll", "the upstream project's own worker executable"),
)

# ⚠️ Печатается ПЕРЕД первым сетевым запросом, а не после. Полгигабайта чужих
# проприетарных файлов не должны приезжать на машину молча: человек обязан
# увидеть, что именно качается, откуда и на чьих условиях. Выключатель
# `download_if_missing` — и есть его согласие; выключенный, он оставляет
# раскладку файлов на самого человека.
LICENCE_NOTICE = "\n".join([
    "",
    "  " + "-" * 74,
    "  TS DLSS Upscaler needs a runtime that is NOT part of this pack and is NOT",
    "  redistributed by it. It is about to be downloaded from a third party:",
    "",
    "      {url}",
    "      (~{size} MB, once, into {root})",
    "",
    "  What that archive contains, and under whose terms:",
] + [
    f"      {names:<44} {owner}" for names, owner in COMPONENT_LICENCES
] + [
    "",
    "  This pack hosts none of it and is not affiliated with or endorsed by",
    "  NVIDIA, ReShade, RenoDX or the upstream project. Install only components",
    "  you are authorised to use, from sources their licences permit. The licence",
    "  texts travel with the binaries into the same folders.",
    "",
    "  Not what you want? Switch 'download_if_missing' off and place the files",
    "  under models/DLSS yourself.",
    "  " + "-" * 74,
    "",
])


def licence_notice(url: str = RUNTIME_URL, root: Path | None = None) -> str:
    """The notice, filled in for this download."""
    return LICENCE_NOTICE.format(
        url=url,
        size=RUNTIME_SIZE_MB,
        root=root if root is not None else runtime_root(),
    )


def _models_dir() -> Path:
    """``models/`` of the running ComfyUI, or a local fallback under tests."""
    try:
        import folder_paths  # noqa: PLC0415 - absent outside ComfyUI

        return Path(folder_paths.models_dir)
    except Exception:  # noqa: BLE001 - tests run without ComfyUI
        return Path(__file__).resolve().parents[3] / "models"


def runtime_root() -> Path:
    """``models/DLSS`` — created on demand, honouring extra_model_paths.yaml."""
    override = os.environ.get("TS_DLSS_RUNTIME_DIR")
    if override:
        return Path(override)
    try:
        import folder_paths  # noqa: PLC0415

        known = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
        for candidate in known or ():
            if Path(candidate).is_dir():
                return Path(candidate)
    except Exception:  # noqa: BLE001 - unknown folder, or no ComfyUI at all
        pass
    return _models_dir() / MODEL_FOLDER_NAME


def register_model_folder() -> None:
    """Tell ComfyUI about ``models/DLSS`` so overrides can point elsewhere."""
    try:
        base = _models_dir() / MODEL_FOLDER_NAME
        base.mkdir(parents=True, exist_ok=True)
        import folder_paths  # noqa: PLC0415

        if hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))
    except Exception as exc:  # noqa: BLE001 - never break the import
        logger.debug("%s Could not register the '%s' folder: %s",
                     LOG_PREFIX, MODEL_FOLDER_NAME, exc)


def missing_files(root: Path | None = None) -> list[str]:
    """Which of the required files are not on disk, in layout order."""
    base = Path(root) if root is not None else runtime_root()
    return [name for name in REQUIRED if not (base / name).is_file()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify(root: Path) -> list[str]:
    """Warnings about files that are present but not what we expected."""
    warnings: list[str] = []
    worker = root / "host/nvngx.dll"
    if worker.is_file() and worker.stat().st_size < MIN_WORKER_BYTES:
        warnings.append(
            f"host/nvngx.dll is only {worker.stat().st_size} bytes — that is not the worker."
        )
    for name, expected in SHA256.items():
        path = root / name
        if not path.is_file():
            continue
        actual = _sha256(path)
        if actual != expected:
            warnings.append(
                f"{name} has SHA-256 {actual}, expected {expected} — the upstream release "
                "was probably rebuilt."
            )
    return warnings


def extract_from_zip(archive: Path, root: Path) -> list[str]:
    """Take the nine known entries out of the release zip into ``root``.

    Returns the relative paths written. Entries are addressed by NAME, never by
    index, and each target is joined to the root explicitly — a zip cannot talk
    this function into writing outside it.
    """
    written: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        for entry, relative in EXTRACT.items():
            if entry not in available:
                if relative.replace("/", os.sep) in {name.replace("/", os.sep)
                                                     for name in REQUIRED}:
                    raise RuntimeError(
                        f"{LOG_PREFIX} The release archive has no '{entry}'. "
                        "The upstream layout changed; update the node."
                    )
                continue
            target = (root / relative).resolve()
            if not str(target).startswith(str(root.resolve())):
                raise RuntimeError(f"{LOG_PREFIX} Refusing to write outside {root}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            written.append(relative)
    return written


def download_runtime(
    root: Path | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    url: str = RUNTIME_URL,
) -> Path:
    """Fetch the release archive and lay the runtime out under ``root``.

    ``progress(done_bytes, total_bytes)`` is called while the archive streams in;
    ``total_bytes`` is 0 when the server does not say how big it is.
    """
    base = Path(root) if root is not None else runtime_root()

    # ⚠️ ПЕРВЫМ делом, до проверки зависимостей и до любого запроса в сеть:
    # уведомление имеет смысл только пока ничего ещё не произошло.
    logger.warning("%s%s", LOG_PREFIX, licence_notice(url, base))

    requests = TSDependencyManager.import_optional("requests")
    if requests is None:
        raise RuntimeError(
            f"{LOG_PREFIX} The 'requests' package is required to download the DLSS runtime. "
            "Install it, or place the files under models/DLSS by hand."
        )

    base.mkdir(parents=True, exist_ok=True)
    archive = base / "_runtime_download.zip.part"
    logger.info("%s Downloading the DLSS runtime (~%d MB) from %s",
                LOG_PREFIX, RUNTIME_SIZE_MB, url)
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with archive.open("wb") as sink:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    sink.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        written = extract_from_zip(archive, base)
        logger.info("%s Runtime ready in %s (%d files).", LOG_PREFIX, base, len(written))
    finally:
        # ⚠️ Half a gigabyte does not stay on the user's disk after extraction,
        # and it does not stay there after a failure either.
        archive.unlink(missing_ok=True)

    still_missing = missing_files(base)
    if still_missing:
        raise RuntimeError(
            f"{LOG_PREFIX} The runtime is still incomplete after the download: "
            + ", ".join(still_missing)
        )
    for warning in verify(base):
        logger.warning("%s %s", LOG_PREFIX, warning)
    return base


def ensure_runtime(
    *,
    download_if_missing: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """The runtime root, downloading the files on first use.

    Called before every run: a file deleted between runs is fetched again.
    """
    root = runtime_root()
    gaps = missing_files(root)
    if not gaps:
        return root
    if not download_if_missing:
        raise RuntimeError(
            f"{LOG_PREFIX} The DLSS runtime is missing from {root}: " + ", ".join(gaps)
            + ". Nothing was downloaded because 'download_if_missing' is off — that "
            "switch is where you agree to fetch third-party components (NVIDIA, "
            "ReShade, RenoDX). Switch it on, or place the files there yourself."
        )
    return download_runtime(root, progress=progress)
