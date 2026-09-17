"""The Neuroframe Engine: one DLL loaded into this process, frame ABI 6.

Since the upstream v9 runtime there is no worker process and no pipe protocol.
``neuroframe_engine.dll`` owns the D3D12 device, the NGX feature instances, the
CUDA/D3D12 shared path, NVIDIA optical flow and the GPU temporal stabiliser; we
hand it one frame and take one frame back. NVIDIA's signed snippet checks the
image that calls it, which is why ``neuroframe_caller.dll`` has to sit next to
the engine.

⚠️ ``RenderParametersV6`` is a CONTRACT with a binary we do not build. The field
order is the ABI, not a preference: a field moved by one position does not raise
— it silently renders with someone else's number. Checked byte-for-byte by
``tests/test_dlss_upscaler.py``.

⚠️ Everything here is process-wide on purpose. NGX keeps its state for the life
of the process, and the reference application never calls
``NVSDK_NGX_D3D12_Shutdown`` nor unloads the driver modules: both were observed
to wedge after a successful evaluation. A ComfyUI run therefore leaves the
engine loaded and only releases the session's surfaces.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

#: The only frame ABI this node speaks. The engine reports its own; a mismatch
#: is refused rather than guessed at.
FRAME_ABI_VERSION = 6

#: How long one native call may take before the engine is declared wedged. The
#: reference application uses the same number, multiplied by the pass count.
WATCHDOG_SECONDS = 45.0
WATCHDOG_CEILING_SECONDS = 180.0

#: ``mask_memory_type`` — this node sends no composition mask.
MEMORY_HOST = 0
MEMORY_CUDA = 1
MEMORY_NONE = 2

#: ``pixel_format`` of a frame descriptor. Only RGBA8 is used from here.
FORMAT_RGBA8 = 1

#: ``CU_CTX_SCHED_BLOCKING_SYNC`` — the one flag the engine insists on before it
#: will share memory with us. See ``enable_blocking_sync``.
CU_CTX_SCHED_BLOCKING_SYNC = 0x04

#: Выключатель для того, кто не хочет, чтобы пак трогал контекст CUDA.
KEEP_CUDA_FLAGS_ENV = "TS_DLSS_KEEP_CUDA_FLAGS"

#: Флаг ставится один раз на процесс — и говорится о нём тоже один раз.
_blocking_sync_set = False


class NeuralBridgeError(RuntimeError):
    """The engine refused the call, and the process is still healthy."""


class NeuralBridgePoisonedError(NeuralBridgeError):
    """Native state may be corrupt: nothing more can be run before a restart."""


class RenderParametersV6(ctypes.Structure):
    """ABI-6 render controls, in the engine's field order. Do not reorder."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("style", ctypes.c_int32),
        ("intensity", ctypes.c_float),
        ("tone", ctypes.c_float),
        ("structure", ctypes.c_float),
        ("skin", ctypes.c_float),
        ("automask", ctypes.c_int32),
        ("reset", ctypes.c_int32),
        ("color_strength", ctypes.c_float),
        ("tone_preservation", ctypes.c_float),
        ("mask_memory_type", ctypes.c_uint32),
        ("mask_width", ctypes.c_uint32),
        ("mask_height", ctypes.c_uint32),
        ("mask_stride", ctypes.c_uint32),
        ("mask_plane", ctypes.c_uint64),
        ("face_skin_protection", ctypes.c_float),
        ("grain_preservation", ctypes.c_float),
        ("nr_passes", ctypes.c_int32),
        ("shimmer_suppression", ctypes.c_float),
        ("prefer_nvof", ctypes.c_int32),
    ]


class FrameDescriptorV1(ctypes.Structure):
    """One frame handed to the engine by description instead of by pointer alone.

    ⚠️ This is the path that lets the GPU do the resizing: the engine accepts a
    frame at ITS OWN size and renders at the destination's size, Lanczos and all,
    without the host ever building the big picture.
    """

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("memory_type", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("planes", ctypes.c_uint64 * 3),
        ("strides", ctypes.c_uint32 * 3),
        ("color_matrix", ctypes.c_uint32),
        ("color_range", ctypes.c_uint32),
        ("rotation", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("timestamp", ctypes.c_int64),
    ]

    @classmethod
    def host_rgba8(cls, frame: np.ndarray, timestamp: int = 0) -> FrameDescriptorV1:
        """Describe a contiguous RGBA8 array that lives in this process's memory."""
        value = cls()
        value.struct_size = ctypes.sizeof(cls)
        value.abi_version = FRAME_ABI_VERSION
        value.memory_type = MEMORY_HOST
        value.pixel_format = FORMAT_RGBA8
        value.width = int(frame.shape[1])
        value.height = int(frame.shape[0])
        value.planes[0] = int(frame.ctypes.data)
        value.strides[0] = int(frame.strides[0])
        # BT.709 full range: an RGBA8 frame carries no matrix of its own, and
        # this is what the reference application passes for host frames.
        value.color_matrix = 1
        value.color_range = 1
        value.timestamp = int(timestamp)
        return value


class FrameResultV1(ctypes.Structure):
    """What the engine reports about the frame it just rendered."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("ngx_create_result", ctypes.c_int32),
        ("ngx_evaluate_result", ctypes.c_int32),
        ("cuda_result", ctypes.c_int32),
        ("scene_reset", ctypes.c_int32),
        ("scene_score", ctypes.c_float),
        ("reserved", ctypes.c_uint32),
        ("upload_bytes", ctypes.c_uint64),
        ("download_bytes", ctypes.c_uint64),
        ("timestamp", ctypes.c_int64),
    ]

    @classmethod
    def empty(cls) -> FrameResultV1:
        value = cls()
        value.struct_size = ctypes.sizeof(cls)
        value.abi_version = FRAME_ABI_VERSION
        return value


def _text(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace").strip() if value else ""


def enable_blocking_sync(ordinal: int) -> bool:
    """Ask the driver to put this process's CUDA context on blocking sync.

    ⚠️ Зачем это вообще. Движок делится памятью с нами только если первичный
    контекст CUDA создан с флагом blocking-sync — тем самым, с которым его
    создаёт FFmpeg. В отдельном процессе так и выходит: контекст поднимает сам
    движок. Но в ComfyUI первым до CUDA добирается torch, ещё на старте
    сервера, и контекст оказывается с флагами по умолчанию — после чего движок
    отвечает `CUDA interop unavailable` и кадр едет через процессор. Замерено,
    чего это стоит: 0,62 с на кадр в 4K вместо 0,26, и карта простаивает больше
    половины времени.

    ⚠️ Это изменение НА ВЕСЬ ПРОЦЕСС, и оно того стоит ровно потому, что почти
    ничего не меняет: blocking-sync значит, что поток, ждущий видеокарту,
    засыпает вместо того, чтобы крутиться в пустом цикле. Для ComfyUI это
    скорее плюс — меньше отобранного у карты процессорного времени, — но
    делается это громко, в лог, и снимается переменной окружения.

    ⚠️ Звать это надо ДО `dlss5nr_init`: движок смотрит на флаги один раз, когда
    поднимается, и потом своего мнения не меняет — переставленный позже флаг он
    уже не заметит (проверено).

    Returns True when the context now carries the flag.
    """
    global _blocking_sync_set  # noqa: PLW0603 - одно решение на процесс
    if _blocking_sync_set:
        return True
    if str(os.environ.get(KEEP_CUDA_FLAGS_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }:
        logger.info("%s Leaving the CUDA context alone (%s is set).",
                    LOG_PREFIX, KEEP_CUDA_FLAGS_ENV)
        return False
    try:
        driver = ctypes.WinDLL("nvcuda.dll")
        driver.cuInit(0)
        device = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(device), int(ordinal)) != 0:
            return False
        setter = getattr(driver, "cuDevicePrimaryCtxSetFlags_v2", None) or \
            getattr(driver, "cuDevicePrimaryCtxSetFlags", None)
        if setter is None:
            return False
        setter.argtypes = [ctypes.c_int, ctypes.c_uint]
        if setter(device, CU_CTX_SCHED_BLOCKING_SYNC) != 0:
            return False
    except OSError as exc:
        logger.debug("%s Could not reach the CUDA driver: %s", LOG_PREFIX, exc)
        return False
    _blocking_sync_set = True
    logger.info(
        "%s Put CUDA device %d on blocking sync so frames can stay in video memory "
        "(set %s=1 to leave the context untouched).", LOG_PREFIX, int(ordinal),
        KEEP_CUDA_FLAGS_ENV,
    )
    return True


def render_parameters(settings: dict[str, Any], reset: bool) -> RenderParametersV6:
    """Fill the ABI-6 structure from validated settings."""
    value = RenderParametersV6()
    value.struct_size = ctypes.sizeof(RenderParametersV6)
    value.abi_version = FRAME_ABI_VERSION
    value.style = int(settings["style"])
    value.intensity = float(settings["intensity"])
    value.tone = float(settings["local_tone"])
    value.structure = float(settings["local_structure"])
    value.skin = float(settings["skin_structure"])
    value.automask = int(bool(settings["auto_mask"]))
    value.reset = int(bool(reset))
    value.color_strength = float(settings["color_strength"])
    value.tone_preservation = float(settings["tone_preservation"])
    value.face_skin_protection = float(settings["face_skin_protection"])
    value.grain_preservation = float(settings["grain_preservation"])
    value.nr_passes = int(settings["nr_passes"])
    value.shimmer_suppression = float(settings["shimmer_suppression"])
    value.prefer_nvof = int(bool(settings["prefer_nvof"]))
    # No composition mask from this node: the picture is the whole frame.
    value.mask_memory_type = MEMORY_NONE
    return value


class NeuralBridge:
    """The engine, loaded once and shared by every run of the node."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._library: Any | None = None
        self._engine_path: Path | None = None
        self._initialized_ordinal: int | None = None
        self._poisoned_reason = ""
        # ⚠️ Buffers a timed-out call may still be writing into are never freed.
        # A use-after-free inside the driver takes the whole ComfyUI process
        # with it; a leaked frame buffer only costs memory until the restart
        # the poison already demands.
        self._timed_out_references: list[Any] = []
        self._sessions = 0
        self._version = "unloaded"
        self._gpu_name = "unknown"
        self._cuda_status = "unknown"

    # ------------------------------------------------------------------ facts
    @property
    def version(self) -> str:
        return self._version

    @property
    def gpu_name(self) -> str:
        return self._gpu_name

    @property
    def cuda_status(self) -> str:
        return self._cuda_status

    @property
    def loaded(self) -> bool:
        return self._library is not None

    def temporal_status(self) -> dict[str, Any]:
        """What the GPU stabiliser reports about itself, or an empty dict."""
        with self._lock:
            if self._library is None:
                return {}
            buffer = ctypes.create_string_buffer(2048)
            try:
                self._library.dlss5nr_temporal_status(buffer, len(buffer))
                value = json.loads(_text(buffer.value) or "{}")
            except (AttributeError, json.JSONDecodeError, OSError):
                return {}
            return value if isinstance(value, dict) else {}

    # ------------------------------------------------------------------- load
    def load(self, engine_path: Path) -> None:
        """Load the engine and bind the entry points this node uses."""
        with self._lock:
            if self._library is not None:
                return
            engine_path = Path(engine_path)
            if not engine_path.is_file():
                raise NeuralBridgeError(
                    f"{LOG_PREFIX} The Neural Rendering engine is missing: {engine_path}"
                )
            loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
            try:
                library = loader(str(engine_path))
            except OSError as exc:
                raise NeuralBridgeError(
                    f"{LOG_PREFIX} The Neural Rendering engine could not be loaded: {exc}. "
                    "It needs Windows x64, an NVIDIA RTX GPU and the Microsoft Visual C++ "
                    "2015-2022 x64 runtime."
                ) from exc

            library.dlss5nr_version.argtypes = []
            library.dlss5nr_version.restype = ctypes.c_char_p
            library.dlss5nr_frame_abi_version.argtypes = []
            library.dlss5nr_frame_abi_version.restype = ctypes.c_uint32
            version = _text(library.dlss5nr_version()) or "unknown"
            frame_abi = int(library.dlss5nr_frame_abi_version())
            if frame_abi != FRAME_ABI_VERSION:
                raise NeuralBridgeError(
                    f"{LOG_PREFIX} The engine in {engine_path.parent} is {version} and speaks "
                    f"frame ABI {frame_abi}; this node speaks ABI {FRAME_ABI_VERSION}. Delete "
                    "the folder so the node fetches the runtime it was built against."
                )

            library.dlss5nr_gpu_name.argtypes = []
            library.dlss5nr_gpu_name.restype = ctypes.c_char_p
            library.dlss5nr_adapter_luid.argtypes = []
            library.dlss5nr_adapter_luid.restype = ctypes.c_char_p
            library.dlss5nr_init.argtypes = [
                ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int,
            ]
            library.dlss5nr_init.restype = ctypes.c_int
            library.dlss5nr_rebind.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            library.dlss5nr_rebind.restype = ctypes.c_int
            library.dlss5nr_cuda_status.argtypes = [ctypes.c_char_p, ctypes.c_int]
            library.dlss5nr_cuda_status.restype = ctypes.c_int
            # ⚠️ Кадр ходит ТОЛЬКО по дескрипторам (`dlss5nr_process_frame_v6`).
            # Простой `dlss5nr_process_v6` принимает float32 на выходном размере,
            # то есть заставляет хост строить картину на 100 МБ и разбирать
            # такую же обратно — 98 мс на кадр в 4K за работу, которую карта
            # делает у себя. Второго пути в паке нет намеренно.
            library.dlss5nr_process_frame_v6.argtypes = [
                ctypes.POINTER(FrameDescriptorV1),
                ctypes.POINTER(FrameDescriptorV1),
                ctypes.POINTER(RenderParametersV6),
                ctypes.POINTER(FrameResultV1),
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            library.dlss5nr_process_frame_v6.restype = ctypes.c_int
            library.dlss5nr_process_cuda_v6.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.POINTER(RenderParametersV6),
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            library.dlss5nr_process_cuda_v6.restype = ctypes.c_int
            library.dlss5nr_cuda_supported.argtypes = []
            library.dlss5nr_cuda_supported.restype = ctypes.c_int
            library.dlss5nr_temporal_status.argtypes = [ctypes.c_char_p, ctypes.c_int]
            library.dlss5nr_temporal_status.restype = ctypes.c_int
            release = getattr(library, "dlss5nr_release_session", None)
            if release is not None:
                release.argtypes = []
                release.restype = ctypes.c_int
            self._library = library
            self._engine_path = engine_path
            self._version = version
            logger.info("%s Neuroframe Engine %s, frame ABI %d.", LOG_PREFIX, version, frame_abi)

    # ------------------------------------------------------------ bookkeeping
    def _guard_poison(self) -> None:
        if self._poisoned_reason:
            raise NeuralBridgePoisonedError(
                f"{LOG_PREFIX} The Neural Rendering runtime is poisoned: "
                f"{self._poisoned_reason}. Restart ComfyUI before rendering again."
            )

    def _call_with_watchdog(
        self,
        label: str,
        function: Callable[[], Any],
        references: tuple[Any, ...] = (),
        *,
        timeout_seconds: float = WATCHDOG_SECONDS,
    ) -> Any:
        """Run one native call under a time limit, on its own thread.

        ⚠️ A hung call inside a driver cannot be interrupted, and without this
        the whole ComfyUI server would wait for it forever. On a timeout the
        engine is declared poisoned: its state is unknown from here on.
        """
        completed = threading.Event()
        result: list[Any] = []
        failure: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(function())
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                failure.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(target=invoke, name=f"ts-dlssnr-{label}", daemon=True)
        thread.start()
        if not completed.wait(timeout_seconds):
            self._poisoned_reason = f"{label} exceeded {timeout_seconds:g} seconds"
            self._timed_out_references.extend(references)
            raise NeuralBridgePoisonedError(
                f"{LOG_PREFIX} Neural Rendering timed out during {label}; native state may be "
                "corrupt. Restart ComfyUI before rendering again."
            )
        if failure:
            self._poisoned_reason = f"native exception during {label}: {failure[0]}"
            raise NeuralBridgePoisonedError(
                f"{LOG_PREFIX} Neural Rendering raised a native exception during {label}. "
                f"Restart ComfyUI before rendering again: {failure[0]}"
            ) from failure[0]
        return result[0]

    def initialize(self, dlssnr_dir: Path, ordinal: int) -> dict[str, Any]:
        """Bring the engine up on one CUDA device, once per process."""
        with self._lock:
            self._guard_poison()
            if self._library is None:
                raise NeuralBridgeError(f"{LOG_PREFIX} The engine is not loaded.")
            ordinal = int(ordinal)
            if self._initialized_ordinal is None:
                error = ctypes.create_string_buffer(4096)
                ok = self._call_with_watchdog(
                    "initialization",
                    lambda: self._library.dlss5nr_init(
                        ordinal, str(Path(dlssnr_dir)), error, len(error)
                    ),
                    (error,),
                )
                if not ok:
                    detail = _text(error.value) or "unknown initialization failure"
                    raise NeuralBridgeError(
                        f"{LOG_PREFIX} Neural Rendering could not start: {detail}"
                    )
                self._initialized_ordinal = ordinal
                self._gpu_name = _text(self._library.dlss5nr_gpu_name()) or "unknown"
            elif ordinal != self._initialized_ordinal:
                # ⚠️ The adapter is process state: changing it while a run is in
                # flight would pull the device out from under the frames.
                if self._sessions:
                    raise NeuralBridgeError(
                        f"{LOG_PREFIX} The Neural Rendering adapter cannot change while a "
                        "render is running."
                    )
                error = ctypes.create_string_buffer(4096)
                ok = self._call_with_watchdog(
                    "adapter rebinding",
                    lambda: self._library.dlss5nr_rebind(ordinal, error, len(error)),
                    (error,),
                )
                if not ok:
                    detail = _text(error.value) or "unknown adapter rebinding failure"
                    self._poisoned_reason = detail
                    raise NeuralBridgePoisonedError(
                        f"{LOG_PREFIX} Neural Rendering could not move to CUDA device "
                        f"{ordinal}: {detail}. Restart ComfyUI before rendering again."
                    )
                self._initialized_ordinal = ordinal
                self._gpu_name = _text(self._library.dlss5nr_gpu_name()) or "unknown"

            status = ctypes.create_string_buffer(2048)
            cuda_ready = bool(self._library.dlss5nr_cuda_status(status, len(status)))
            self._cuda_status = _text(status.value) or "unavailable"
            return {
                "engine_version": self._version,
                "frame_abi_version": FRAME_ABI_VERSION,
                "gpu_name": self._gpu_name,
                "adapter_luid": _text(self._library.dlss5nr_adapter_luid()) or "unknown",
                "cuda_ordinal": ordinal,
                "cuda_supported": cuda_ready,
                "cuda_status": self._cuda_status,
            }

    def cuda_state(self) -> tuple[bool, str]:
        """Ask the engine again whether it can share video memory with us."""
        with self._lock:
            if self._library is None:
                return False, "the engine is not loaded"
            status = ctypes.create_string_buffer(2048)
            ready = bool(self._library.dlss5nr_cuda_status(status, len(status)))
            self._cuda_status = _text(status.value) or "unavailable"
            return ready, self._cuda_status

    def open_session(self) -> None:
        with self._lock:
            self._guard_poison()
            self._sessions += 1

    def close_session(self) -> None:
        """Give the feature's surfaces back; never shut NGX down.

        ⚠️ ``dlss5nr_release_session`` ends the temporal history, so the next
        frame starts a new clip. NGX itself and the driver modules stay loaded:
        unloading them after a successful evaluation is what wedges.
        """
        with self._lock:
            if self._sessions:
                self._sessions -= 1
            if self._sessions or self._library is None or self._poisoned_reason:
                return
            release = getattr(self._library, "dlss5nr_release_session", None)
            if release is not None:
                self._call_with_watchdog("session release", release)

    # --------------------------------------------------------------- the work
    def process_frame_host(
        self,
        source: np.ndarray,
        destination: np.ndarray,
        settings: dict[str, Any],
        reset: bool,
        *,
        timestamp: int = 0,
    ) -> tuple[float, dict[str, Any]]:
        """One RGBA8 frame in at its own size, one RGBA8 frame out at the wanted one.

        ⚠️ This is the fast path and the reason the card stops waiting: the
        engine takes the small frame, uploads it, resizes it (Lanczos) and
        renders — all on the GPU. The host never builds the big float picture,
        which at 4K was three copies of ~100 MB per frame.

        Returns the seconds the call took and what the engine reported.
        """
        with self._lock:
            self._guard_poison()
            if self._library is None:
                raise NeuralBridgeError(f"{LOG_PREFIX} The engine is not loaded.")
            incoming = FrameDescriptorV1.host_rgba8(source, timestamp)
            outgoing = FrameDescriptorV1.host_rgba8(destination, timestamp)
            params = render_parameters(settings, reset)
            report = FrameResultV1.empty()
            error = ctypes.create_string_buffer(4096)
            started = time.perf_counter()
            ok = self._call_with_watchdog(
                "feature-18 frame evaluation",
                lambda: self._library.dlss5nr_process_frame_v6(
                    ctypes.byref(incoming),
                    ctypes.byref(outgoing),
                    ctypes.byref(params),
                    ctypes.byref(report),
                    error,
                    len(error),
                ),
                (source, destination, incoming, outgoing, params, report, error),
                timeout_seconds=min(
                    WATCHDOG_CEILING_SECONDS, WATCHDOG_SECONDS * max(1, params.nr_passes)
                ),
            )
            elapsed = time.perf_counter() - started
            if not ok:
                self._raise_for(error, "Neural Rendering failed")
            return elapsed, {
                "ngx_create_result": f"0x{int(report.ngx_create_result) & 0xFFFFFFFF:08X}",
                "ngx_evaluate_result": f"0x{int(report.ngx_evaluate_result) & 0xFFFFFFFF:08X}",
                "cuda_result": int(report.cuda_result),
                "scene_reset": bool(report.scene_reset),
                "scene_score": float(report.scene_score),
                "upload_bytes": int(report.upload_bytes),
                "download_bytes": int(report.download_bytes),
            }

    def process_cuda(
        self,
        source_pointer: int,
        destination_pointer: int,
        width: int,
        height: int,
        settings: dict[str, Any],
        reset: bool,
        *,
        stream: int = 0,
        keep_alive: tuple[Any, ...] = (),
    ) -> float:
        """The same frame, but both buffers already live on the card.

        ⚠️ Это САМЫЙ быстрый путь, и разница не косметическая: на 3840×2048
        замерено 260 мс против 420–460 мс у хостового. Причина видна по
        `nvidia-smi` — на хостовом пути карта простаивает больше половины
        времени, пока движок перекладывает кадр через процессор.

        The pointers are plain device addresses (a torch tensor's ``data_ptr()``
        will do), float32 RGB, contiguous, at the output size. The caller is
        responsible for synchronising the stream that filled them.
        """
        with self._lock:
            self._guard_poison()
            if self._library is None:
                raise NeuralBridgeError(f"{LOG_PREFIX} The engine is not loaded.")
            params = render_parameters(settings, reset)
            error = ctypes.create_string_buffer(4096)
            started = time.perf_counter()
            ok = self._call_with_watchdog(
                "feature-18 CUDA evaluation",
                lambda: self._library.dlss5nr_process_cuda_v6(
                    int(source_pointer),
                    int(destination_pointer),
                    int(width),
                    int(height),
                    int(stream),
                    ctypes.byref(params),
                    error,
                    len(error),
                ),
                # ⚠️ Буферы видеопамяти держатся живыми вместе с остальным: если
                # вызов завис, движок всё ещё может в них писать, а освободить
                # их значило бы отдать эту память кому-то другому.
                (params, error, *keep_alive),
                timeout_seconds=min(
                    WATCHDOG_CEILING_SECONDS, WATCHDOG_SECONDS * max(1, params.nr_passes)
                ),
            )
            elapsed = time.perf_counter() - started
            if not ok:
                self._raise_for(error, "CUDA Neural Rendering failed")
            return elapsed

    def _raise_for(self, error: Any, what: str) -> None:
        """Turn the engine's error buffer into the right kind of refusal."""
        detail = _text(error.value) or "unknown feature-18 failure"
        lowered = detail.lower()
        # ⚠️ These two words mean the native heap is already damaged; anything
        # run after them is undefined, so the engine is shut out until the
        # process restarts.
        if "corrupt" in lowered or "access violation" in lowered:
            self._poisoned_reason = detail
            raise NeuralBridgePoisonedError(
                f"{LOG_PREFIX} {detail}. Restart ComfyUI before rendering again."
            )
        raise NeuralBridgeError(f"{LOG_PREFIX} {what}: {detail}")


#: One engine per process, like the runtime it wraps.
BRIDGE = NeuralBridge()
