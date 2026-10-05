"""DLSS 5 Neural Rendering: sizing, settings and one clip's worth of frames.

Since the upstream v9 runtime the feature runs **in this process**: the
Neuroframe Engine (see ``_bridge.py``) owns the D3D12 device and the NGX feature
instances, and a session here is a logical thing — the temporal history of one
clip plus the buffers its frames travel in. There is no worker process, no
ReShade and no pipe protocol any more.

⚠️ The neural pass runs at the OUTPUT size. The picture is resized to it first
(Lanczos) and the network re-renders it there; that is why the factor is a size
choice and not a separate super-resolution step.

⚠️ Motion vectors are no longer supplied by the caller. The engine estimates
motion itself when ``shimmer_suppression`` is above zero (NVIDIA optical flow,
with a bundled GPU Lucas-Kanade fallback), so the only thing a video batch still
has to tell it is where the cuts are.

⚠️ Since 05.10.2026 frames travel as float32 RGB through host memory
(``dlss5nr_process_v6``). Video memory shared with the engine is opt-in
(``TS_DLSS_CUDA_INTEROP=1``): on a card ComfyUI keeps nearly full, that road
produced corrupted 16×16 tiles and stalls — see ``NeuralBridge.process_float``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import _assets
from ._bridge import (
    BRIDGE,
    NeuralBridgeError,
    NeuralBridgePoisonedError,
    enable_blocking_sync,
)

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

#: Largest output the runtime supports.
MAX_LONG_EDGE = 7680
MAX_SHORT_EDGE = 4320
#: Both the input and the output size must reach this.
MIN_EDGE = 64

#: Включает путь через общую с движком видеопамять. По умолчанию выключен.
CUDA_INTEROP_ENV = "TS_DLSS_CUDA_INTEROP"


def cuda_interop_requested() -> bool:
    """Whether the machine owner asked for frames to stay in video memory.

    ⚠️ Выключено по умолчанию (05.10.2026) по замеру, а не из осторожности: на
    RTX 3080 Ti Laptop, пока ComfyUI держал 15 из 16 ГБ, этот путь дал битые
    плитки 16×16 в одном кадре из двенадцати и зависал — на движках v9 и v11.
    Хостовый float-путь на том же кадре: 0 из 40, все 40 прогонов побитно
    одинаковы. Цена — ~0,5 с на кадр 3264×1856.
    """
    return str(os.environ.get(CUDA_INTEROP_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }

NR_STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}

#: ⚠️ ``perf_quality`` is what the v5 worker negotiated with DLSS Super
#: Resolution. The v9 engine has no such step — the factor only chooses the
#: output size — but the number is kept in the table because the labels, and
#: therefore saved graphs, are keyed by it.
UPSCALING_MODES = {
    1.0: {"label": "1× (DLAA / native)", "name": "DLAA", "perf_quality": 5},
    1.5: {"label": "1.5× (Quality)", "name": "Quality", "perf_quality": 2},
    1.724: {"label": "1.724× (Balanced)", "name": "Balanced", "perf_quality": 1},
    2.0: {"label": "2× (Performance)", "name": "Performance", "perf_quality": 0},
    3.0: {"label": "3× (Ultra Performance)", "name": "Ultra Performance", "perf_quality": 3},
}
#: Labels in the node's combo, in factor order.
UPSCALING_LABELS = [mode["label"] for mode in UPSCALING_MODES.values()]
_LABEL_TO_FACTOR = {mode["label"]: factor for factor, mode in UPSCALING_MODES.items()}


def factor_from_label(label: str) -> float:
    """The numeric factor behind a combo label; also accepts a bare number."""
    if label in _LABEL_TO_FACTOR:
        return _LABEL_TO_FACTOR[label]
    try:
        return resolve_upscaling_mode(float(label))[0]
    except (TypeError, ValueError) as exc:
        choices = ", ".join(UPSCALING_LABELS)
        raise ValueError(
            f"{LOG_PREFIX} Unknown upscaling factor {label!r}. Choose one of: {choices}."
        ) from exc


def resolve_upscaling_mode(raw_factor: float) -> tuple[float, dict[str, str | int]]:
    try:
        factor = float(raw_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{LOG_PREFIX} The upscaling factor must be one of the supported DLSS modes."
        ) from exc
    if not math.isfinite(factor):
        raise ValueError(
            f"{LOG_PREFIX} The upscaling factor must be one of the supported DLSS modes."
        )
    for supported, mode in UPSCALING_MODES.items():
        if math.isclose(factor, supported, rel_tol=0.0, abs_tol=1e-9):
            return supported, mode
    # «x», а не «×»: текст ошибки уходит в лог ComfyUI (tests/test_log_encoding.py).
    choices = ", ".join(f"{value:g}x" for value in UPSCALING_MODES)
    raise ValueError(
        f"{LOG_PREFIX} Unsupported upscaling factor {factor:g}x. Choose one of: {choices}."
    )


def _nearest_even(value: float) -> int:
    return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)


def resolve_output_size(width: int, height: int, factor: float) -> tuple[int, int]:
    """Output size for this factor, rounded to even, refusing beyond 8K."""
    factor, _ = resolve_upscaling_mode(factor)
    output_width = _nearest_even(int(width) * factor)
    output_height = _nearest_even(int(height) * factor)
    if max(output_width, output_height) > MAX_LONG_EDGE or \
            min(output_width, output_height) > MAX_SHORT_EDGE:
        fitting = [
            candidate
            for candidate in UPSCALING_MODES
            if max(_nearest_even(width * candidate), _nearest_even(height * candidate))
            <= MAX_LONG_EDGE
            and min(_nearest_even(width * candidate), _nearest_even(height * candidate))
            <= MAX_SHORT_EDGE
        ]
        best = max(fitting) if fitting else None
        hint = (
            f" Choose {best:g}x or lower for this source."
            if best is not None
            else " The source already exceeds the supported 8K boundary."
        )
        raise ValueError(
            f"{LOG_PREFIX} The requested {output_width}x{output_height} output exceeds the "
            f"supported {MAX_LONG_EDGE}x{MAX_SHORT_EDGE} boundary.{hint}"
        )
    return output_width, output_height


def resolve_native_settings(
    *,
    nr_style: str,
    nr_intensity: float,
    nr_passes: int,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    automatic_mask: bool,
    nr_color_strength: float = 1.0,
    tone_preservation: float = 0.0,
    face_skin_protection: float = 0.0,
    grain_preservation: float = 0.0,
    shimmer_suppression: float = 0.0,
) -> dict[str, int | float | bool]:
    """Validate the public controls and translate them to the engine's fields."""
    try:
        style = NR_STYLES[nr_style]
    except KeyError as exc:
        raise ValueError(
            f"{LOG_PREFIX} Unknown NR style: {nr_style!r}. Choose one of: "
            + ", ".join(NR_STYLES)
        ) from exc

    bounds = {
        "NR intensity": (nr_intensity, 0.0, 2.0),
        "Local tone strength": (local_tone_strength, 0.0, 2.0),
        "Local structure strength": (local_structure_strength, 0.0, 2.0),
        "Skin structure strength": (skin_structure_strength, -1.0, 2.0),
        "NR colour strength": (nr_color_strength, 0.0, 1.0),
        "Tone preservation": (tone_preservation, 0.0, 1.0),
        "Face/skin protection": (face_skin_protection, 0.0, 1.0),
        "Grain preservation": (grain_preservation, 0.0, 1.0),
        "Shimmer suppression": (shimmer_suppression, 0.0, 1.0),
    }
    validated: dict[str, float] = {}
    for label, (raw, minimum, maximum) in bounds.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{LOG_PREFIX} {label} must be a number between {minimum:g} and {maximum:g}."
            ) from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{LOG_PREFIX} {label} must be between {minimum:g} and {maximum:g}."
            )
        validated[label] = value

    if isinstance(nr_passes, bool) or int(nr_passes) != nr_passes:
        raise ValueError(f"{LOG_PREFIX} NR passes must be a whole number from 1 to 4.")
    if not 1 <= int(nr_passes) <= 4:
        raise ValueError(f"{LOG_PREFIX} NR passes must be between 1 and 4.")

    return {
        "style": style,
        "auto_mask": int(bool(automatic_mask)),
        "intensity": validated["NR intensity"],
        "nr_passes": int(nr_passes),
        "local_tone": validated["Local tone strength"],
        "local_structure": validated["Local structure strength"],
        "skin_structure": validated["Skin structure strength"],
        "color_strength": validated["NR colour strength"],
        "tone_preservation": validated["Tone preservation"],
        "face_skin_protection": validated["Face/skin protection"],
        "grain_preservation": validated["Grain preservation"],
        "shimmer_suppression": validated["Shimmer suppression"],
        # ⚠️ Nothing encodes on this GPU while the node runs, so NVIDIA's own
        # optical flow is free to be used. The reference application turns it
        # off only when NVENC is busy on the same card.
        "prefer_nvof": True,
    }


def resize_fit(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a float32 RGB frame into the size the network renders at.

    ⚠️ This is the upscale. The engine does not resize on this path: the frame
    it is handed is already at the output size, and everything it adds is
    detail, not pixels.

    ⚠️ Ланцош во float, без восьмибитного шага, и с клампом после: его
    лепестки выходят за 0..1 на контрастных краях, а движку нужен кадр в 0..1.
    """
    source_height, source_width = rgb.shape[:2]
    if source_width == width and source_height == height:
        return np.ascontiguousarray(rgb, dtype=np.float32)
    try:
        import cv2  # noqa: PLC0415 - optional, see the fallback below

        scale = min(width / source_width, height / source_height)
        fit_width = max(1, min(width, int(round(source_width * scale))))
        fit_height = max(1, min(height, int(round(source_height * scale))))
        resized = cv2.resize(
            np.ascontiguousarray(rgb, dtype=np.float32), (fit_width, fit_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    except ImportError:
        fit_width, fit_height = min(width, source_width), min(height, source_height)
        resized = rgb[:fit_height, :fit_width]
    if fit_width == width and fit_height == height:
        canvas = np.ascontiguousarray(resized, dtype=np.float32)
    else:
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        x = (width - fit_width) // 2
        y = (height - fit_height) // 2
        canvas[y:y + fit_height, x:x + fit_width] = resized
    np.clip(canvas, 0.0, 1.0, out=canvas)
    return canvas


class DLSSFrameSession:
    """One clip: a temporal history plus the buffers its frames travel in.

    The engine keeps state between frames, so a batch of consecutive video
    frames goes through one session in order, and the session is released at the
    end — the next one starts a new history.
    """

    def __init__(
        self,
        root: Path,
        *,
        output_width: int,
        output_height: int,
        native_settings: dict[str, int | float | bool],
        cuda_ordinal: int = 0,
        cancelled: Callable[[], bool] | None = None,
        prefer_cuda: bool | None = None,
    ) -> None:
        if prefer_cuda is None:
            prefer_cuda = cuda_interop_requested()
        if output_width < MIN_EDGE or output_height < MIN_EDGE:
            raise ValueError(
                f"{LOG_PREFIX} The neural pass needs at least {MIN_EDGE}x{MIN_EDGE} pixels; "
                f"this run asks for {output_width}x{output_height}."
            )
        self.root = Path(root)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        # The frame is resized before the call, so these are the same size. Kept
        # under their old names: the node and the cut detector both ask for them.
        self.render_width = self.output_width
        self.render_height = self.output_height
        self.native_settings = dict(native_settings)
        self.closed = False
        self._cancelled = cancelled or (lambda: False)
        self._next_index: int | None = None
        self._logs: list[str] = []
        self._open = False

        self.frames = 0
        self.scene_resets = 0
        self.evaluate_seconds = 0.0
        self.memory_path = "host"
        self._torch = None
        self._device = None
        self._incoming = None
        self._outgoing = None

        # ⚠️ ДО того, как движок поднимется: свои возможности по CUDA он
        # определяет ровно один раз, при `dlss5nr_init`, и переставленный
        # позже флаг контекста уже не замечает (проверено).
        if prefer_cuda:
            enable_blocking_sync(int(cuda_ordinal))
        BRIDGE.load(_assets.engine_path(self.root))
        self.bridge_status = BRIDGE.initialize(_assets.runtime_dir(self.root), cuda_ordinal)
        self.bridge_status = {
            **self.bridge_status,
            "neural_dimensions": {"width": self.output_width, "height": self.output_height},
            "nr_passes": int(self.native_settings.get("nr_passes", 1)),
            "composition": {
                "color_strength": float(self.native_settings.get("color_strength", 1.0)),
                "tone_preservation": float(self.native_settings.get("tone_preservation", 0.0)),
                "face_skin_protection":
                    float(self.native_settings.get("face_skin_protection", 0.0)),
                "grain_preservation":
                    float(self.native_settings.get("grain_preservation", 0.0)),
            },
            "temporal_stabilization": {
                "shimmer_suppression":
                    float(self.native_settings.get("shimmer_suppression", 0.0)),
                "stabilizer_backend": "native_gpu_residual",
            },
        }
        if prefer_cuda:
            self._prepare_cuda(int(cuda_ordinal))
        self.bridge_status["memory_path"] = self.memory_path
        BRIDGE.open_session()
        self._open = True
        self._log("session_open", **self.bridge_status)

    # ------------------------------------------------------------------ cuda
    def _prepare_cuda(self, ordinal: int) -> None:
        """Get the frames onto the card if the engine will have them there.

        ⚠️ Отказ движка бывает ровно одного лечимого вида: «active CUDA primary
        context does not use FFmpeg blocking-sync flags». В ComfyUI это не
        случайность, а правило — контекст CUDA поднимает torch на старте
        сервера, задолго до нас. Флаг переставляется на живом контексте
        (см. `enable_blocking_sync`), после чего движок соглашается.
        """
        ready = bool(self.bridge_status.get("cuda_supported"))
        detail = str(self.bridge_status.get("cuda_status", ""))
        if not ready and "blocking-sync" in detail:
            detail += (
                " — the CUDA context was already up before this node ran; restart "
                "ComfyUI to let the engine have it"
            )
        if ready:
            self._open_cuda_buffers(ordinal)
        else:
            logger.info(
                "%s Frames go through host memory: the engine will not share video "
                "memory (%s).", LOG_PREFIX, detail or "no reason given",
            )

    def _open_cuda_buffers(self, ordinal: int) -> None:
        """Take two frame buffers on the card, if the card is there to take them.

        ⚠️ Только по явной просьбе (`TS_DLSS_CUDA_INTEROP=1`). Быстрее хоста
        (на 3840×2048 сеть 0,27 с против 0,62), но на почти полной карте давал
        битые плитки и зависания — см. `cuda_interop_requested`.

        Буферы — обычные тензоры torch, то есть та же память, которой уже
        пользуется ComfyUI: своего аллокатора CUDA пак не заводит. Не вышло —
        молча остаёмся на хостовом пути, он работает везде.
        """
        try:
            import torch  # noqa: PLC0415 - ComfyUI's own, never at import time

            if not torch.cuda.is_available():
                return
            device = torch.device(f"cuda:{ordinal}" if ordinal >= 0 else "cuda")
            incoming = torch.empty(
                (self.output_height, self.output_width, 3), dtype=torch.float32, device=device
            )
            outgoing = torch.empty_like(incoming)
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 - любая беда здесь не фатальна
            logger.info("%s Staying on the host path: %s", LOG_PREFIX, exc)
            return
        self._torch = torch
        self._device = device
        self._incoming = incoming
        self._outgoing = outgoing
        self.memory_path = "cuda"

    def _release_cuda_buffers(self) -> None:
        self._incoming = self._outgoing = None
        self._device = None

    # --------------------------------------------------------------- evidence
    def _log(self, event: str, **values: Any) -> None:
        self._logs.append(
            json.dumps({"event": event, **values}, sort_keys=True, separators=(",", ":"))
        )
        if len(self._logs) > 500:
            del self._logs[: len(self._logs) - 500]

    @property
    def logs(self) -> list[str]:
        return list(self._logs)

    def structured_status(self) -> dict[str, Any]:
        """Everything worth reporting about the run that just happened."""
        status = dict(self.bridge_status)
        temporal = dict(status.get("temporal_stabilization", {}))
        temporal.update(BRIDGE.temporal_status())
        status["temporal_stabilization"] = temporal
        status["frames"] = self.frames
        status["scene_resets"] = self.scene_resets
        status["evaluate_seconds"] = round(self.evaluate_seconds, 3)
        return status

    # ------------------------------------------------------------------ work
    def render_into(
        self, *, index: int, frame: np.ndarray, reset: bool, destination: np.ndarray
    ) -> None:
        """One float32 RGB frame at the output size in, its result written into ``destination``.

        ``destination`` is the caller's slice of the output batch: values land
        there in 0..1, without a single array being allocated per frame. Which
        road the frame takes — through video memory or through the host — is
        this session's business and nobody else's.
        """
        if self._cancelled():
            raise InterruptedError(f"{LOG_PREFIX} Cancelled.")
        if self.closed:
            raise RuntimeError(f"{LOG_PREFIX} This Neural Rendering session is closed.")
        if self._next_index is None:
            self._next_index = int(index)
        if int(index) != self._next_index:
            raise ValueError(
                f"{LOG_PREFIX} Neural Rendering frames must arrive in order; expected "
                f"{self._next_index}, got {index}."
            )
        if frame.dtype != np.float32 \
                or frame.shape != (self.output_height, self.output_width, 3):
            raise ValueError(
                f"{LOG_PREFIX} The frame handed to Neural Rendering must be float32 RGB at "
                f"{self.output_width}x{self.output_height}."
            )
        if destination.dtype != np.float32 \
                or destination.shape != (self.output_height, self.output_width, 3) \
                or not destination.flags.c_contiguous:
            raise ValueError(
                f"{LOG_PREFIX} The result buffer must be a contiguous float32 "
                f"{self.output_width}x{self.output_height}x3 array."
            )
        frame = np.ascontiguousarray(frame)
        if self._incoming is not None:
            try:
                self._render_on_card(frame, reset, destination)
            except NeuralBridgePoisonedError:
                raise
            except (NeuralBridgeError, RuntimeError) as exc:
                # ⚠️ Отказ на полпути не должен стоить человеку прогона: хостовый
                # путь работает всегда, и кадр переснимается им же.
                logger.warning(
                    "%s Video memory stopped working on frame %d (%s); the rest of the "
                    "clip goes through host memory.", LOG_PREFIX, index, exc,
                )
                self._release_cuda_buffers()
                self.memory_path = "host (after a fallback)"
                self._render_on_host(frame, reset, destination)
        else:
            self._render_on_host(frame, reset, destination)

        self.frames += 1
        self.scene_resets += int(bool(reset) and index != 0)
        self._next_index += 1
        self._log("frame", index=int(index), reset=bool(reset), path=self.memory_path)

    def _render_on_card(self, frame: np.ndarray, reset: bool,
                        destination: np.ndarray) -> None:
        torch = self._torch
        self._incoming.copy_(torch.from_numpy(frame), non_blocking=False)
        torch.cuda.synchronize(self._device)
        self.evaluate_seconds += BRIDGE.process_cuda(
            self._incoming.data_ptr(), self._outgoing.data_ptr(),
            self.output_width, self.output_height,
            self.native_settings, bool(reset),
            keep_alive=(self._incoming, self._outgoing),
        )
        torch.cuda.synchronize(self._device)
        # ⚠️ Кламп на карте, а не на процессоре: на 4K это 24 МБ чисел, и
        # выходить за 0..1 движок право имеет — IMAGE не имеет.
        self._outgoing.clamp_(0.0, 1.0)
        torch.from_numpy(destination).copy_(self._outgoing)

    def _render_on_host(self, frame: np.ndarray, reset: bool,
                        destination: np.ndarray) -> None:
        self.evaluate_seconds += BRIDGE.process_float(
            frame, destination, self.native_settings, bool(reset)
        )
        # ⚠️ Выходить за 0..1 движок право имеет — IMAGE не имеет.
        np.clip(destination, 0.0, 1.0, out=destination)

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        """End the clip: the next session starts a new temporal history."""
        if self.closed:
            return
        self.closed = True
        self._release_cuda_buffers()
        if self._open:
            self._open = False
            BRIDGE.close_session()

    def abort(self) -> None:
        """Same thing on the failure path; never raises over the real error."""
        if self.closed:
            return
        self.closed = True
        self._release_cuda_buffers()
        if not self._open:
            return
        self._open = False
        try:
            BRIDGE.close_session()
        except (NeuralBridgeError, NeuralBridgePoisonedError, OSError) as exc:
            logger.debug("%s Releasing the session after a failure: %s", LOG_PREFIX, exc)


def verify_feature_18(logs: list[str], status: dict[str, Any]) -> dict[str, Any]:
    """Prove the signed DLSSNR path actually ran, instead of assuming it did.

    ⚠️ Reported, never enforced. The pictures are already rendered by the time
    this is read, and throwing them away over a missing line would cost the user
    the whole run. What must not happen is passing silently.
    """
    evaluated = [line for line in logs if '"event":"frame"' in line]
    temporal = status.get("temporal_stabilization", {}) or {}
    return {
        "verified": bool(evaluated),
        "successful_frames": len(evaluated),
        "engine_version": status.get("engine_version", "unknown"),
        "gpu_name": status.get("gpu_name", "unknown"),
        "cuda_status": status.get("cuda_status", "unknown"),
        "motion_backend": temporal.get("motion_backend", "unreported"),
        "shimmer_suppression": temporal.get("shimmer_suppression", 0.0),
        "scene_resets": status.get("scene_resets", 0),
        "evaluate_seconds": status.get("evaluate_seconds", 0.0),
    }
