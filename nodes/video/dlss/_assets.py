"""Runtime files of TS DLSS Upscaler: where they live and how they arrive.

Everything the node runs on is downloaded on first use into ``models/DLSS`` and
never shipped with the pack: ``nvngx_dlssnr.dll`` is an NVIDIA binary under the
NVIDIA DLSS licence, the Neuroframe Engine and its caller shim are MIT by the
upstream author. The licence texts are taken out of the same archive and kept
next to the binaries.

⚠️ The layout is dictated by the engine and must NOT be flattened::

    models/DLSS/
      dlssnr/  neuroframe_engine.dll  neuroframe_caller.dll
               nvngx_dlssnr.dll  LICENSE-Merserk.txt  LICENSE-NVIDIA-DLSS.txt

⚠️ ``neuroframe_caller.dll`` is not optional decoration: NVIDIA's signed snippet
checks the image that calls it, and it only accepts that shim. Both DLLs have to
sit in the same folder, which is the folder handed to ``dlss5nr_init``.

⚠️ WHY v9 (decided by the pack owner, 05.10.2026). Upstream v9.0 is the last
release that is MIT as a whole: its root LICENSE and ``dlssnr/LICENSE-Merserk.txt``
both. From v10.0 the releases carry the "Merserk Source License 1.0"
(proprietary, source-available); their THIRD-PARTY-NOTICES name the engine and
the caller as Merserk components, and §6(g) forbids bundling them with another
product. v10/v11 still had the old MIT file next to the engine, but the notices
contradict it — so it is not something to rely on. The owner's own application
DLSS-Video took the same decision (its commit bed0a85).

⚠️ v11 was used from 02.10 to 05.10.2026. A folder that holds it is updated by
``ensure_runtime``: the network is the same file, so only the engine and the
caller differ, and only what does not match is fetched. v11's
``neuroframe_engine_neural_rendering.dll`` stays on disk (the user's file) and is
mentioned in the log as unused.

⚠️ The old v5 layout (``host/`` with ReShade and the worker executable,
``dlss/nvngx_dlss.dll``) is dead weight — nothing here loads it any more. It is
left alone rather than deleted: those files are the user's, not ours. Its
presence still means one thing: the user once installed the runtime from this
same upstream project, so ``ensure_runtime`` updates it even with
``download_if_missing`` off (see there why that switch is so often off without
anybody choosing it).
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import logging
import os
import shutil
import socket
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from ..._deps import TSDependencyManager

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

#: Folder under ``models/`` — the name the user asked for.
MODEL_FOLDER_NAME = "DLSS"

#: The folder the engine is told about, under the runtime root.
RUNTIME_SUBDIR = "dlssnr"

#: Upstream release the runtime is taken from — the last one that is MIT throughout.
RUNTIME_URL = (
    "https://github.com/Merserk/dlss5-visual-enhancer/releases/download/v9.0/"
    "DLSS.5.Visual.Enhancer.v9.0.zip"
)
#: The whole archive — fetched only when the server refuses partial downloads.
RUNTIME_SIZE_MB = 486
#: What a first install actually fetches from it: the five entries, compressed.
RUNTIME_FETCH_MB = 112

ENGINE = "dlssnr/neuroframe_engine.dll"
CALLER = "dlssnr/neuroframe_caller.dll"
NETWORK = "dlssnr/nvngx_dlssnr.dll"

#: zip entry -> path relative to the runtime root.
EXTRACT = {
    "bin/runtime/dlssnr/neuroframe_engine.dll": ENGINE,
    "bin/runtime/dlssnr/neuroframe_caller.dll": CALLER,
    "bin/runtime/dlssnr/nvngx_dlssnr.dll": NETWORK,
    "bin/runtime/dlssnr/LICENSE-Merserk.txt": "dlssnr/LICENSE-Merserk.txt",
    "bin/runtime/dlssnr/LICENSE-NVIDIA-DLSS.txt": "dlssnr/LICENSE-NVIDIA-DLSS.txt",
}

#: Without these the node cannot run; their absence triggers the download.
REQUIRED = (ENGINE, CALLER, NETWORK)

#: The v5 runtime this node used until 17.09.2026. Nothing loads it any more.
OBSOLETE = (
    "host/nvngx.dll",
    "host/dxgi.dll",
    "host/renodx-dlss5.addon64",
    "dlss/nvngx_dlss.dll",
)

#: The v11 engine (02.10–05.10.2026). Not loaded any more; left to the user.
UNUSED_V11_ENGINE = "dlssnr/neuroframe_engine_neural_rendering.dll"

# ⚠️ Несовпадение суммы ОСТАНАВЛИВАЕТ работу, и это не перестраховка. Движок
# грузится В НАШ процесс: подменённый движок — это чужой код с правами ComfyUI,
# и никакая песочница его уже не сдержит. Релиз на GitHub можно удалить и
# залить под тем же тегом что угодно, адрес этого не заметит. Пересборка
# апстрима лечится обновлением таблицы (или выключателем ниже); подменённый
# бинарник не лечится ничем.
#
# ⚠️ Суммы v9 сверены 17.09.2026 с установленным рантаймом и 05.10.2026 ещё раз —
# с закреплённой копией эталонного приложения (`release/pinned-runtime/v9.0`,
# вынута из официального архива и проверена там же). `nvngx_dlssnr.dll` один и
# тот же в v9, v11 и v14. ⚠️ У caller'а одно ИМЯ в v9 и v11 и разное тело:
# движок идёт только со своим caller'ом, и сверяется пара целиком.
SHA256 = {
    ENGINE: "2BDC5BFD59906DF7CB6DF98F78339D68F741B11256A26927A4C107425E7F46D4",
    CALLER: "58E2850F96FC1B81A9154E059E3F3A42239440280C79E1CE41F6142CA9F1BAD4",
    NETWORK: "6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927",
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

# ⚠️ Печатается ПЕРЕД первым сетевым запросом, а не после. Чужие файлы, из
# которых один проприетарный, не должны приезжать на машину молча: человек
# обязан увидеть, что именно качается, откуда и на чьих условиях. Выключатель
# `download_if_missing` оставляет раскладку файлов ему самому.
LICENCE_NOTICE = "\n".join([
    "",
    "  " + "-" * 74,
    "  TS DLSS Upscaler needs a runtime that is NOT part of this pack and is NOT",
    "  redistributed by it. It is about to be downloaded from a third party:",
    "",
    "      {url}",
    "      (only the files below, ~{fetch} MB at most, once, into {root};",
    "       the whole ~{size} MB archive only if the server refuses partial downloads)",
    "",
    "  What it contains, and under whose terms:",
] + [
    f"      {names:<46} {owner}" for names, owner in COMPONENT_LICENCES
] + [
    "",
    "  This pack hosts none of it and is not affiliated with or endorsed by NVIDIA",
    "  or the upstream project. Install only components you are authorised to use,",
    "  from sources their licences permit. The licence texts travel with the",
    "  binaries into the same folder.",
    "",
    "  Not what you want? Switch 'download_if_missing' off and place the files",
    "  under models/DLSS yourself. (A runtime you already installed from this",
    "  project is updated even then.)",
    "  " + "-" * 74,
    "",
])


def licence_notice(url: str = RUNTIME_URL, root: Path | None = None) -> str:
    """The notice, filled in for this download."""
    return LICENCE_NOTICE.format(
        url=url,
        size=RUNTIME_SIZE_MB,
        fetch=RUNTIME_FETCH_MB,
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
    base = Path(root) if root is not None else runtime_root()
    return base / ENGINE


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


def unused_v11_engine(root: Path | None = None) -> Path | None:
    """The v11 engine, if it is still on disk; nothing loads it any more."""
    base = Path(root) if root is not None else runtime_root()
    path = base / UNUSED_V11_ENGINE
    return path if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


#: Что уже сверено в этом процессе: путь -> (размер, время правки, сумма).
#: ⚠️ Проверка обязана идти перед КАЖДЫМ прогоном — файл мог смениться, — но
#: пересчитывать при этом 166 МБ каждый раз незачем: замерено 130 мс на прогон,
#: то есть на коротком батче это заметная часть всей работы. Кэш держится за
#: размер и время правки файла: подменённый файл их не сохранит. Сумма в ключе:
#: у caller'а одно имя в v9 и v11, и сверенный против одной сборки не годится
#: для другой.
_verified: dict[str, tuple[int, int, str]] = {}


def _unchanged_since_check(path: Path, expected: str) -> bool:
    stamp = _verified.get(str(path))
    if stamp is None:
        return False
    try:
        info = path.stat()
    except OSError:
        return False
    return stamp == (info.st_size, info.st_mtime_ns, expected)


def _remember_check(path: Path, expected: str) -> None:
    try:
        info = path.stat()
    except OSError:
        return
    _verified[str(path)] = (info.st_size, info.st_mtime_ns, expected)


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
        + f"\n  This node LOADS {engine_path(root).name} into the ComfyUI process, so it "
        f"refuses to run an unknown build. Delete {root / RUNTIME_SUBDIR} and let the node "
        f"fetch it again, or set {_SKIP_VERIFY_ENV}=1 if you installed a different build "
        "on purpose."
    )


def _file_matches(path: Path, expected: str) -> bool:
    if _unchanged_since_check(path, expected):
        return True
    if _sha256(path) != expected:
        return False
    _remember_check(path, expected)
    return True


def stale_files(root: Path) -> list[str]:
    """Закреплённые файлы, которые на месте, но не той сборки (например, от v11)."""
    return [
        name for name, expected in SHA256.items()
        if (root / name).is_file() and not _file_matches(root / name, expected)
    ]


def verify(root: Path) -> list[str]:
    """Файлы закреплённой сборки — на месте, но не те."""
    warnings: list[str] = []
    for name, expected in SHA256.items():
        path = root / name
        if not path.is_file():
            continue
        if not _file_matches(path, expected):
            warnings.append(
                f"{name} has SHA-256 {_sha256(path)}, expected {expected} — the upstream "
                "release was probably rebuilt."
            )
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
                if relative in REQUIRED:
                    raise RuntimeError(
                        f"{LOG_PREFIX} The release archive has no '{entry}'. "
                        "The upstream layout changed; update the node."
                    )
                continue
            target = _target(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            written.append(relative)
    return written


def _target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise RuntimeError(f"{LOG_PREFIX} Refusing to write outside {root}.")
    return target


# ── частичная загрузка ─────────────────────────────────────────────────────
#
# ⚠️ Зачем. Архив релиза — это целое настольное приложение на ~500 МБ (Python,
# Qt, FFmpeg), а ноде из него нужны пять файлов. Тому, у кого стоит другая
# сборка (v11), смена стоит ~240 КБ: сеть NVIDIA та же самая. Zip позволяет взять
# файл по отдельности — оглавление лежит в конце, у каждого файла известны
# смещение и размер, — а GitHub отдаёт части файла по заголовку Range.
#
# ⚠️ Сервер, не отдавший часть (ответ не 206), — не ошибка: тогда качается
# весь архив, как раньше. А вот файл, не сошедшийся по CRC или длине, —
# ошибка, и обходом она не прячется.

class PartialDownloadUnsupported(RuntimeError):
    """The server will not hand out byte ranges; the whole archive is the way."""


def _assert_public_url(url: str) -> None:
    """Block SSRF: refuse anything but an HTTPS URL resolving to a public address."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError(f"{LOG_PREFIX} Refusing to fetch a non-HTTPS URL: {url!r}")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, None)}
    except OSError as exc:
        raise RuntimeError(f"{LOG_PREFIX} Could not resolve host {parsed.hostname!r}: {exc}") from exc
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise RuntimeError(
            f"{LOG_PREFIX} Refusing to fetch from a non-public address for host {parsed.hostname!r}.")


class _RemoteArchive(io.RawIOBase):
    """A release zip read by byte ranges: enough for zipfile to list it."""

    def __init__(self, requests: Any, url: str) -> None:
        self._requests = requests
        _assert_public_url(url)
        probe = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                             timeout=(15, 60), allow_redirects=True)
        try:
            span = probe.headers.get("Content-Range", "")
            if probe.status_code != 206 or "/" not in span:
                raise PartialDownloadUnsupported(
                    f"the server answered {probe.status_code} to a range request")
            self.size = int(span.rsplit("/", 1)[1])
            self.url = probe.url
            _assert_public_url(self.url)
        finally:
            probe.close()
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._position, 2: self.size}[whence]
        self._position = base + offset
        return self._position

    def fetch(self, start: int, length: int) -> bytes:
        end = min(self.size, start + length) - 1
        if end < start:
            return b""
        response = self._requests.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                      timeout=(15, 60), allow_redirects=False)
        if response.status_code != 206:
            raise PartialDownloadUnsupported(
                f"the server answered {response.status_code} to a range request")
        return response.content

    def readinto(self, buffer) -> int:
        if self._position >= self.size:
            return 0
        data = self.fetch(self._position, len(buffer))
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)

    def stream(self, start: int, length: int):
        """The bytes [start, start+length) as one streamed response."""
        response = self._requests.get(
            self.url, headers={"Range": f"bytes={start}-{start + length - 1}"},
            stream=True, timeout=(15, 120))
        if response.status_code != 206:
            response.close()
            raise PartialDownloadUnsupported(
                f"the server answered {response.status_code} to a range request")
        return response


def _crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def _already_there(path: Path, info: zipfile.ZipInfo) -> bool:
    """Whether the file on disk is byte for byte the archive's entry."""
    try:
        return path.is_file() and path.stat().st_size == info.file_size \
            and _crc32(path) == info.CRC
    except OSError:
        return False


def _fetch_entry(remote: _RemoteArchive, info: zipfile.ZipInfo, target: Path,
                 advance: Callable[[int], None]) -> Path:
    """One entry out of the remote archive into ``<target>.part``, checked.

    Returns the ``.part`` file; putting it in place is the caller's job, done
    only once EVERY entry has arrived — see ``fetch_entries``.
    """
    header = remote.fetch(info.header_offset, 30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{LOG_PREFIX} {info.filename}: no local header in the archive.")
    name_length, extra_length = struct.unpack_from("<HH", header, 26)
    start = info.header_offset + 30 + name_length + extra_length
    if info.compress_type == zipfile.ZIP_DEFLATED:
        inflate = zlib.decompressobj(-15)
    elif info.compress_type == zipfile.ZIP_STORED:
        inflate = None
    else:
        raise PartialDownloadUnsupported(f"compression method {info.compress_type}")

    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    crc = 0
    written = 0
    try:
        with remote.stream(start, info.compress_size) as response, partial.open("wb") as sink:  # noqa: E501
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                advance(len(chunk))
                data = inflate.decompress(chunk) if inflate is not None else chunk
                crc = zlib.crc32(data, crc)
                written += len(data)
                sink.write(data)
            if inflate is not None:
                tail = inflate.flush()
                crc = zlib.crc32(tail, crc)
                written += len(tail)
                sink.write(tail)
        if written != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
            raise RuntimeError(
                f"{LOG_PREFIX} {info.filename} arrived damaged ({written} bytes, CRC "
                f"{crc & 0xFFFFFFFF:08X}; expected {info.file_size}, {info.CRC:08X}).")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return partial


def fetch_entries(requests: Any, url: str, root: Path,
                  progress: Callable[[int, int], None] | None = None) -> list[str]:
    """Take the known entries out of the remote archive, skipping what is already here.

    Returns the relative paths written. Raises ``PartialDownloadUnsupported`` when
    the server will not hand out ranges — the caller then fetches the archive.
    """
    remote = _RemoteArchive(requests, url)
    try:
        bundle = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1024 * 1024))
    except zipfile.BadZipFile as error:
        raise PartialDownloadUnsupported(f"the archive directory is unreadable: {error}") from error
    with bundle:
        listed = {info.filename: info for info in bundle.infolist()}
        wanted: list[tuple[zipfile.ZipInfo, str]] = []
        for entry, relative in EXTRACT.items():
            info = listed.get(entry)
            if info is None:
                if relative in REQUIRED:
                    raise RuntimeError(
                        f"{LOG_PREFIX} The release archive has no '{entry}'. "
                        "The upstream layout changed; update the node.")
                continue
            if not _already_there(_target(root, relative), info):
                wanted.append((info, relative))
        total = sum(info.compress_size for info, _ in wanted)
        done = 0

        def advance(count: int) -> None:
            nonlocal done
            done += count
            if progress is not None:
                progress(done, total)

        if wanted:
            logger.info("%s Fetching %d file(s), %.1f MB, out of the release archive.",
                        LOG_PREFIX, len(wanted), total / (1024 * 1024))
        # ⚠️ Всё качается рядом, в `.part`, и встаёт на место только целиком:
        # оборванное посередине обновление не должно оставить новый движок
        # рядом со старым caller'ом — такая пара не запустится, а сумма caller'а
        # укажет не на настоящую причину.
        arrived: list[tuple[Path, Path, str]] = []
        try:
            for info, relative in wanted:
                target = _target(root, relative)
                arrived.append((_fetch_entry(remote, info, target, advance), target, relative))
        except BaseException:
            for partial, _, _ in arrived:
                partial.unlink(missing_ok=True)
            raise
        for partial, target, _ in arrived:
            os.replace(partial, target)
        return [relative for _, _, relative in arrived]


def _fetch_whole_archive(requests: Any, url: str, base: Path,
                         progress: Callable[[int, int], None] | None) -> list[str]:
    archive = base / "_runtime_download.zip.part"
    logger.info("%s Downloading the whole DLSS runtime archive (~%d MB) from %s",
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
        return extract_from_zip(archive, base)
    finally:
        # ⚠️ Полгигабайта не остаются на диске после распаковки — и после
        # сбоя тоже.
        archive.unlink(missing_ok=True)


def download_runtime(
    root: Path | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
    url: str = RUNTIME_URL,
) -> Path:
    """Fetch what is missing from the release and lay the runtime out under ``root``.

    ``progress(done_bytes, total_bytes)`` is called while bytes stream in;
    ``total_bytes`` is 0 when the server does not say how much is coming.
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
    try:
        written = fetch_entries(requests, url, base, progress)
    except PartialDownloadUnsupported as reason:
        logger.info("%s Partial download unavailable (%s).", LOG_PREFIX, reason)
        written = _fetch_whole_archive(requests, url, base, progress)
    logger.info("%s Runtime ready in %s (%d file(s) written).", LOG_PREFIX, base, len(written))

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
        # ⚠️ Файлы на месте, но не той сборки — это почти всегда рантайм v11,
        # поставленный паком 12.12.3–12.12.4. Заменяется только то, что не
        # сходится (движок и caller, ~240 КБ): сеть NVIDIA в v9 та же.
        stale = [] if verify_is_off() else stale_files(root)
        if stale and download_if_missing:
            logger.info(
                "%s %s do not match the pinned v9 build (probably the v11 runtime of pack "
                "12.12.3-12.12.4); fetching the v9 files.", LOG_PREFIX, ", ".join(stale))
            return download_runtime(root, progress=progress)
        # ⚠️ Проверяется КАЖДЫЙ прогон, а не только свежая установка: файлы на
        # диске могли смениться после неё, а движок мы грузим заново.
        require_known_runtime(root)
        return root
    if download_if_missing:
        return download_runtime(root, progress=progress)
    # ⚠️ Выключатель «выключен» в графе чаще всего НЕ выбран человеком: в
    # 12.10.0-12.11.2 это было умолчание, и граф несёт его в себе. Такой
    # человек до обновления работал на v5 (файлы были — выключатель ни на что
    # не влиял), а после перехода на новый движок упирался в ошибку. v5 качался
    # из того же проекта, так что его наличие — уже данное согласие на этот
    # источник: обновляем то, что человек сам однажды поставил.
    #
    # ⚠️ Только ПЕРЕХОД, один раз: папки `dlssnr/` ещё нет вовсе. Если она уже
    # есть и в ней чего-то не хватает — это ручная раскладка или удалённый
    # файл, и выключатель снова значит ровно «не качать». Признак v5 — файлы
    # `host/`: имя `dlss/nvngx_dlss.dll` слишком общее, папку models/DLSS может
    # делить с нами другой пак.
    missing_message = (
        f"{LOG_PREFIX} The DLSS runtime is missing from {root}: " + ", ".join(gaps)
        + ". Nothing was downloaded because 'download_if_missing' is off on this node. "
        "If you did not switch it off yourself: workflows saved with pack versions "
        "12.10.0 to 12.11.2 carry 'off', the default of those versions. Switch "
        "'download_if_missing' on in the node and run again (~"
        f"{RUNTIME_FETCH_MB} MB, once). Or place the files there yourself - they are in "
        f"the upstream release {RUNTIME_URL}, under bin/runtime/dlssnr/."
    )
    v5_marks = [name for name in obsolete_files(root) if name.startswith("host/")]
    if v5_marks and not runtime_dir(root).exists():
        logger.warning(
            "%s 'download_if_missing' is off, but %s already holds the previous (v5) "
            "DLSS runtime from the same upstream project (%s). The node needs its "
            "newer files since pack 12.11.3, so the runtime you installed is updated once.",
            LOG_PREFIX, root, v5_marks[0])
        try:
            return download_runtime(root, progress=progress)
        except OSError as error:
            # Без сети — прежний понятный отказ, а не сырая ошибка requests
            # (его исключения — OSError). Несовпадение сумм сюда НЕ попадает
            # и проходит как есть.
            raise RuntimeError(
                f"{missing_message} (Updating the v5 runtime failed: {error})"
            ) from error
    raise RuntimeError(missing_message)
