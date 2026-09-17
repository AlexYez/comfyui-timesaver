"""Runtime files of TS DLSS Upscaler: where they live and how they arrive.

Everything the node runs on is downloaded on first use into ``models/DLSS`` and
never shipped with the pack: ``nvngx_dlssnr.dll`` is an NVIDIA binary under the
NVIDIA DLSS licence, the Neuroframe Engine and its caller shim are MIT by the
upstream author. The licence texts are taken out of the same archive and kept
next to the binaries.

⚠️ The layout is dictated by the engine and must NOT be flattened::

    models/DLSS/
      dlssnr/  neuroframe_engine.dll  neuroframe_caller.dll  nvngx_dlssnr.dll
               LICENSE-Merserk.txt    LICENSE-NVIDIA-DLSS.txt

⚠️ ``neuroframe_caller.dll`` is not optional decoration: NVIDIA's signed snippet
checks the image that calls it, and it only accepts that shim. Both DLLs have to
sit in the same folder, which is the folder handed to ``dlss5nr_init``.

⚠️ Since the upstream v9 release the old v5 layout (``host/`` with ReShade and
the worker executable, ``dlss/nvngx_dlss.dll``) is dead weight — nothing here
reads it any more. It is left alone rather than deleted: those files are the
user's, not ours.
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

#: The folder the engine is told about, under the runtime root.
RUNTIME_SUBDIR = "dlssnr"

#: Upstream release the runtime is taken from (486 MB; only five entries are kept).
RUNTIME_URL = (
    "https://github.com/Merserk/dlss5-visual-enhancer/releases/download/v9.0/"
    "DLSS.5.Visual.Enhancer.v9.0.zip"
)
RUNTIME_SIZE_MB = 486

#: zip entry -> path relative to the runtime root.
EXTRACT = {
    "bin/runtime/dlssnr/neuroframe_engine.dll": "dlssnr/neuroframe_engine.dll",
    "bin/runtime/dlssnr/neuroframe_caller.dll": "dlssnr/neuroframe_caller.dll",
    "bin/runtime/dlssnr/nvngx_dlssnr.dll": "dlssnr/nvngx_dlssnr.dll",
    "bin/runtime/dlssnr/LICENSE-Merserk.txt": "dlssnr/LICENSE-Merserk.txt",
    "bin/runtime/dlssnr/LICENSE-NVIDIA-DLSS.txt": "dlssnr/LICENSE-NVIDIA-DLSS.txt",
}

#: Without these the node cannot run; their absence triggers the download.
REQUIRED = (
    "dlssnr/neuroframe_engine.dll",
    "dlssnr/neuroframe_caller.dll",
    "dlssnr/nvngx_dlssnr.dll",
)

#: The v5 runtime this node used until 17.09.2026. Nothing loads it any more.
OBSOLETE = (
    "host/nvngx.dll",
    "host/dxgi.dll",
    "host/renodx-dlss5.addon64",
    "dlss/nvngx_dlss.dll",
)

# ⚠️ Несовпадение суммы ОСТАНАВЛИВАЕТ работу, и это не перестраховка. Движок
# грузится В НАШ процесс: подменённый `neuroframe_engine.dll` — это чужой код с
# правами ComfyUI, и никакая песочница его уже не сдержит. Релиз на GitHub
# можно удалить и залить под тем же тегом что угодно, адрес этого не заметит.
# Пересборка апстрима лечится обновлением таблицы (или выключателем ниже);
# подменённый бинарник не лечится ничем.
#
# ⚠️ Суммы сверены 17.09.2026 с установленным рантаймом v9.0 и совпали со всеми
# тремя источниками: таблицей `bin/runtime/BINARIES.md` эталонного приложения,
# его `installer/manifest.json` и файлами на диске.
SHA256 = {
    "dlssnr/neuroframe_engine.dll":
        "2BDC5BFD59906DF7CB6DF98F78339D68F741B11256A26927A4C107425E7F46D4",
    "dlssnr/neuroframe_caller.dll":
        "58E2850F96FC1B81A9154E059E3F3A42239440280C79E1CE41F6142CA9F1BAD4",
    "dlssnr/nvngx_dlssnr.dll":
        "6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927",
}

#: Выключатель проверки — для того, кто СОЗНАТЕЛЬНО поставил другую сборку
#: рантайма (апстрим выпускает их часто). Живёт в окружении, а не в графе.
_SKIP_VERIFY_ENV = "TS_DLSS_SKIP_VERIFY"

#: Whose the files are. Printed before the download, and kept next to the
#: binaries as the licence texts the archive carries.
COMPONENT_LICENCES = (
    ("dlssnr/nvngx_dlssnr.dll",
     "NVIDIA, proprietary (NVIDIA RTX SDKs License)"),
    ("dlssnr/neuroframe_engine.dll", "Neuroframe Engine, MIT (upstream author)"),
    ("dlssnr/neuroframe_caller.dll", "caller shim, MIT (upstream author)"),
)

# ⚠️ Печатается ПЕРЕД первым сетевым запросом, а не после. Полгигабайта чужих
# файлов, из которых один проприетарный, не должны приезжать на машину молча:
# человек обязан увидеть, что именно качается, откуда и на чьих условиях.
# Выключатель `download_if_missing` оставляет раскладку файлов ему самому.
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
    f"      {names:<32} {owner}" for names, owner in COMPONENT_LICENCES
] + [
    "",
    "  This pack hosts none of it and is not affiliated with or endorsed by NVIDIA",
    "  or the upstream project. Install only components you are authorised to use,",
    "  from sources their licences permit. The licence texts travel with the",
    "  binaries into the same folder.",
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


def runtime_dir(root: Path | None = None) -> Path:
    """The folder handed to ``dlss5nr_init``: all three DLLs side by side."""
    base = Path(root) if root is not None else runtime_root()
    return base / RUNTIME_SUBDIR


def engine_path(root: Path | None = None) -> Path:
    """The DLL this process loads."""
    return runtime_dir(root) / "neuroframe_engine.dll"


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


def obsolete_files(root: Path | None = None) -> list[str]:
    """Leftovers of the v5 runtime, if the user still has them."""
    base = Path(root) if root is not None else runtime_root()
    return [name for name in OBSOLETE if (base / name).is_file()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


#: Что уже сверено в этом процессе: путь -> (размер, время правки).
#: ⚠️ Проверка обязана идти перед КАЖДЫМ прогоном — файл мог смениться, — но
#: пересчитывать при этом 166 МБ каждый раз незачем: замерено 130 мс на прогон,
#: то есть на коротком батче это заметная часть всей работы. Кэш держится за
#: размер и время правки файла: подменённый файл их не сохранит.
_verified: dict[str, tuple[int, int]] = {}


def _unchanged_since_check(path: Path) -> bool:
    stamp = _verified.get(str(path))
    if stamp is None:
        return False
    try:
        info = path.stat()
    except OSError:
        return False
    return stamp == (info.st_size, info.st_mtime_ns)


def _remember_check(path: Path) -> None:
    try:
        info = path.stat()
    except OSError:
        return
    _verified[str(path)] = (info.st_size, info.st_mtime_ns)


def verify_is_off() -> bool:
    """Отключил ли хозяин машины сверку сумм."""
    return str(os.environ.get(_SKIP_VERIFY_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def require_known_runtime(root: Path) -> None:
    """Отказать, если на диске лежит не то, что мы закрепили.

    ⚠️ Зовётся ПЕРЕД загрузкой движка, а не только после скачивания: файлы могли
    смениться между установкой и прогоном, а грузим мы их в свой процесс каждый
    раз заново.

    Args:
        root: корень рантайма (``models/DLSS``).

    Raises:
        RuntimeError: файл есть, но его сумма не та, что закреплена здесь.
    """
    if verify_is_off():
        return
    problems = verify(root)
    if not problems:
        return
    raise RuntimeError(
        f"{LOG_PREFIX} The DLSS runtime on disk is not the build this node pins:\n  "
        + "\n  ".join(problems)
        + f"\n  This node LOADS neuroframe_engine.dll into the ComfyUI process, so it "
        f"refuses to run an unknown build. Delete {root / RUNTIME_SUBDIR} and let the node "
        f"fetch it again, or set {_SKIP_VERIFY_ENV}=1 if you installed a different build "
        "on purpose."
    )


def verify(root: Path) -> list[str]:
    """Файлы, которые на месте, но не те, что мы закрепили."""
    warnings: list[str] = []
    for name, expected in SHA256.items():
        path = root / name
        if not path.is_file():
            continue
        if _unchanged_since_check(path):
            continue
        actual = _sha256(path)
        if actual != expected:
            warnings.append(
                f"{name} has SHA-256 {actual}, expected {expected} — the upstream release "
                "was probably rebuilt."
            )
        else:
            _remember_check(path)
    return warnings


def extract_from_zip(archive: Path, root: Path) -> list[str]:
    """Take the five known entries out of the release zip into ``root``.

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
    # ⚠️ Свежескачанное проверяется СРАЗУ: если по тому адресу лежит уже не та
    # сборка, узнать об этом надо здесь, а не в момент загрузки движка.
    require_known_runtime(base)
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
        # ⚠️ Проверяется КАЖДЫЙ прогон, а не только свежая установка: файлы на
        # диске могли смениться после неё, а движок мы грузим заново.
        require_known_runtime(root)
        return root
    if not download_if_missing:
        raise RuntimeError(
            f"{LOG_PREFIX} The DLSS runtime is missing from {root}: " + ", ".join(gaps)
            + ". Nothing was downloaded because 'download_if_missing' is off. Switch it on, "
            "or place the files there yourself — they are in the upstream release "
            f"{RUNTIME_URL}, under bin/runtime/dlssnr/."
        )
    return download_runtime(root, progress=progress)
