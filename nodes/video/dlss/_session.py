"""The DLSS 5 Neural Rendering worker: protocol, sizing, evidence.

DLSSNR is executed by NVIDIA's signed runtime through the RenoDX add-on, driven
by a small native worker. Python talks to that worker over stdin/stdout with a
fixed binary protocol: one 8-bit RGBA frame plus a motion field in, one 8-bit
RGBA frame out, strictly one at a time — the worker keeps temporal state between
frames.

⚠️ The header layout and the magic numbers here are a CONTRACT with a binary we
do not build. Byte-identical to the reference application (``src/core/runtime.py``,
protocol version 4, "model-preset"); changing a field silently produces garbage
or a hang, not an error.
"""

from __future__ import annotations

import collections
import logging
import math
import re
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

VIDEO_MAGIC = 0x34563544
SETUP_MAGIC = 0x34505553
FRAME_MAGIC = 0x314D5246
OUT_MAGIC = 0x3154554F
VIDEO_HEADER_FORMAT = "<14I4f"
SETUP_RESPONSE_FORMAT = "<12I"
FRAME_HEADER_FORMAT = "<4Iq"
RESULT_HEADER_FORMAT = "<5Iq"

#: Largest output the runtime supports.
MAX_LONG_EDGE = 7680
MAX_SHORT_EDGE = 4320
#: Both the input and the negotiated render size must reach this.
MIN_EDGE = 64

NR_PRESETS = {"Default": 0, "Preset #1": 1, "Preset #2": 2, "Preset #3": 3}
NR_STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}
DLSS_MODEL_PRESETS = {"Default": 0, "J": 10, "K": 11, "L": 12, "M": 13}

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
    choices = ", ".join(f"{value:g}×" for value in UPSCALING_MODES)
    raise ValueError(
        f"{LOG_PREFIX} Unsupported upscaling factor {factor:g}×. Choose one of: {choices}."
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
            f" Choose {best:g}× or lower for this source."
            if best is not None
            else " The source already exceeds the supported 8K boundary."
        )
        raise ValueError(
            f"{LOG_PREFIX} The requested {output_width}×{output_height} output exceeds the "
            f"supported {MAX_LONG_EDGE}×{MAX_SHORT_EDGE} boundary.{hint}"
        )
    return output_width, output_height


def resolve_native_settings(
    *,
    nr_preset: str,
    nr_style: str,
    dlss_model_preset: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    automatic_mask: bool,
) -> dict[str, int | float]:
    """Validate the public controls and translate them to the worker protocol."""
    def choice(table: dict[str, int], value: str, label: str) -> int:
        try:
            return table[value]
        except KeyError as exc:
            raise ValueError(
                f"{LOG_PREFIX} Unknown {label}: {value!r}. Choose one of: "
                + ", ".join(table)
            ) from exc

    preset = choice(NR_PRESETS, nr_preset, "NR preset")
    style = choice(NR_STYLES, nr_style, "NR style")
    model_preset = choice(DLSS_MODEL_PRESETS, dlss_model_preset, "DLSS model preset")

    # ⚠️ Intensity above 1 is clamped rather than refused: the app does the same,
    # and the runtime treats anything above 1 as 1.
    bounds = {
        "NR intensity": (min(float(nr_intensity), 1.0), 0.0, 1.0),
        "Local tone strength": (local_tone_strength, 0.0, 2.0),
        "Local structure strength": (local_structure_strength, 0.0, 2.0),
        "Skin structure strength": (skin_structure_strength, -1.0, 2.0),
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

    return {
        "profile": 0,
        "preset": preset,
        "style": style,
        "auto_mask": int(bool(automatic_mask)),
        "ui_correction": 0,
        "intensity": validated["NR intensity"],
        "local_tone": validated["Local tone strength"],
        "local_structure": validated["Local structure strength"],
        "skin_structure": validated["Skin structure strength"],
        "dlss_model_preset": model_preset,
    }


def resize_fit(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a frame into the size the worker negotiated.

    ⚠️ Normally never used: for every supported factor the render size equals the
    input size. It exists because the runtime is allowed to negotiate a different
    one, and sending a frame of the wrong size would not raise — it would desync
    the stream and hand back garbage.
    """
    source_height, source_width = rgba.shape[:2]
    if source_width == width and source_height == height:
        return np.ascontiguousarray(rgba)
    try:
        import cv2  # noqa: PLC0415 - optional, see the fallback below

        scale = min(width / source_width, height / source_height)
        fit_width = max(1, min(width, int(round(source_width * scale))))
        fit_height = max(1, min(height, int(round(source_height * scale))))
        resized = cv2.resize(
            rgba, (fit_width, fit_height), interpolation=cv2.INTER_LANCZOS4
        )
    except ImportError:
        fit_width, fit_height = min(width, source_width), min(height, source_height)
        resized = rgba[:fit_height, :fit_width]
    canvas = np.zeros((height, width, rgba.shape[2]), dtype=rgba.dtype)
    canvas[..., 3] = 255
    x = (width - fit_width) // 2
    y = (height - fit_height) // 2
    canvas[y:y + fit_height, x:x + fit_width] = resized
    return canvas


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError(
                f"The native worker stopped after {len(chunks)} of {size} output bytes"
            )
        chunks.extend(block)
    return bytes(chunks)


def _read_exact_into(stream, target: np.ndarray) -> None:
    view = memoryview(target).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError(
                f"The native worker stopped after {offset} of {len(view)} output bytes"
            )
        offset += count


def _array_bytes(array: np.ndarray, dtype: np.dtype) -> memoryview:
    return memoryview(np.ascontiguousarray(array, dtype=dtype)).cast("B")


def _drain(stream, sink: collections.deque) -> None:
    """Keep the worker's stderr flowing; a full pipe would deadlock the frame loop."""
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", errors="replace").rstrip())
    except (OSError, ValueError):
        pass


def classify_worker_failure(
    *,
    worker_code: int | None,
    frame_index: int,
    worker_logs: list[str],
    reshade_lines: list[str],
) -> str:
    """Turn a worker death into something the user can act on."""
    evidence = "\n".join([*worker_logs, *reshade_lines])
    access_violation = (
        "evaluate raised 0xC0000005" in evidence
        or "feature 18 evaluate raised an exception" in evidence
    )
    if access_violation:
        summary = (
            f"{LOG_PREFIX} DLSS feature 18 raised access violation 0xC0000005 before frame "
            f"{frame_index} completed, inside the neural runtime. On an RTX 30-series card "
            "this is the experimental path: update the NVIDIA driver."
        )
    else:
        summary = (
            f"{LOG_PREFIX} The native DLSS worker exited with code {worker_code} before "
            f"frame {frame_index} completed."
        )
    details = [summary]
    if worker_logs:
        details.append("Worker log:\n" + "\n".join(worker_logs[-60:]))
    if reshade_lines:
        details.append("ReShade feature-18 log:\n" + "\n".join(reshade_lines[-60:]))
    return "\n".join(details)


class DLSSFrameSession:
    """One live worker: a stream of frames at a fixed size and settings.

    A session costs about a second to start, so it is opened once per node run
    and every frame of the batch goes through it. Changing the size or any
    setting means a new worker — the runtime negotiates them at setup.
    """

    def __init__(
        self,
        root: Path,
        *,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        frame_count: int,
        mode: dict[str, str | int],
        native_settings: dict[str, int | float],
        warmup_frames: int = 0,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.root = Path(root)
        self.host = self.root / "host"
        self.worker_path = self.host / "nvngx.dll"
        self.reshade_log = self.host / "ReShade.log"
        self.mode = mode
        self.native_settings = native_settings
        self.closed = False
        self._cancelled = cancelled or (lambda: False)
        self._log: collections.deque = collections.deque(maxlen=400)

        # Remember where ReShade's log ended, so the evidence read afterwards is
        # this session's and not a previous run's.
        self._log_baseline = self.reshade_log.stat().st_size if self.reshade_log.is_file() else 0

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.worker = subprocess.Popen(  # noqa: S603 - fixed argv, no user input
            [str(self.worker_path), "--video"],
            cwd=str(self.host),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
        self._log_thread = threading.Thread(
            target=_drain, args=(self.worker.stderr, self._log), daemon=True
        )
        self._log_thread.start()

        native = native_settings
        header = struct.pack(
            VIDEO_HEADER_FORMAT,
            VIDEO_MAGIC,
            input_width,
            input_height,
            output_width,
            output_height,
            int(warmup_frames),
            int(frame_count),
            int(mode["perf_quality"]),
            int(native["dlss_model_preset"]),
            int(native["profile"]),
            int(native["preset"]),
            int(native["style"]),
            int(native["auto_mask"]),
            int(native["ui_correction"]),
            float(native["intensity"]),
            float(native["local_tone"]),
            float(native["local_structure"]),
            float(native["skin_structure"]),
        )
        try:
            self.worker.stdin.write(header)
            self.worker.stdin.flush()
            try:
                setup = _read_exact(
                    self.worker.stdout, struct.calcsize(SETUP_RESPONSE_FORMAT)
                )
            except EOFError as exc:
                code = self.worker.wait(timeout=10)
                self._log_thread.join(timeout=2)
                details = "\n".join(self.worker_logs[-60:]) or "The worker said nothing."
                raise RuntimeError(
                    f"{LOG_PREFIX} The DLSS worker failed during setup (exit {code}):\n{details}"
                ) from exc
            (
                setup_magic,
                setup_ok,
                self.setup_result,
                self.render_width,
                self.render_height,
                negotiated_width,
                negotiated_height,
                self.minimum_width,
                self.minimum_height,
                self.maximum_width,
                self.maximum_height,
                self.applied_model_preset,
            ) = struct.unpack(SETUP_RESPONSE_FORMAT, setup)

            if setup_magic != SETUP_MAGIC:
                raise RuntimeError(
                    f"{LOG_PREFIX} The DLSS worker in models/DLSS does not speak the "
                    "version-4 protocol. Delete the folder so the node fetches it again."
                )
            if not setup_ok:
                details = "\n".join(self.worker_logs[-60:])
                raise RuntimeError(
                    f"{LOG_PREFIX} DLSS {mode['name']} is unavailable for "
                    f"{output_width}×{output_height} (NGX 0x{self.setup_result:08X}). "
                    "Choose a lower factor or update the NVIDIA driver."
                    + (f"\n{details}" if details else "")
                )
            if (negotiated_width, negotiated_height) != (output_width, output_height):
                raise RuntimeError(
                    f"{LOG_PREFIX} The worker negotiated {negotiated_width}×"
                    f"{negotiated_height} instead of the requested "
                    f"{output_width}×{output_height}."
                )
            requested_preset = int(native["dlss_model_preset"])
            if self.applied_model_preset != requested_preset:
                raise RuntimeError(
                    f"{LOG_PREFIX} The worker applied DLSS model preset "
                    f"{self.applied_model_preset} instead of the requested {requested_preset}."
                )
            if self.render_width < MIN_EDGE or self.render_height < MIN_EDGE:
                raise RuntimeError(
                    f"{LOG_PREFIX} DLSS returned an unsupported render size "
                    f"{self.render_width}×{self.render_height}; both sides must be at least "
                    f"{MIN_EDGE} pixels."
                )
            self.output_width = output_width
            self.output_height = output_height
        except Exception:
            self.abort()
            raise

    @property
    def worker_logs(self) -> list[str]:
        return list(self._log)

    def reshade_log_text(self) -> str:
        """Only the ReShade output this session produced."""
        if not self.reshade_log.is_file():
            return ""
        with self.reshade_log.open("rb") as stream:
            size = self.reshade_log.stat().st_size
            stream.seek(self._log_baseline if size >= self._log_baseline else 0)
            data = stream.read()
        text = data.decode("utf-8", errors="replace")
        marker = f"[{self.worker.pid}]"
        mine = [line for line in text.splitlines() if marker in line]
        return "\n".join(mine) if mine else text

    def reshade_diagnostics(self, limit: int = 300) -> list[str]:
        lines = self.reshade_log_text().splitlines()
        relevant = [
            line
            for line in lines
            if "DLSS 5 Neural Rendering" in line
            or "DLSSNR" in line
            or "feature 18" in line
            or "exception" in line.lower()
            or "failed" in line.lower()
        ]
        return (relevant or lines)[-limit:]

    def process(
        self, *, index: int, rgba: np.ndarray, motion: np.ndarray, reset: bool, pts: int = 0
    ) -> np.ndarray:
        """One frame in, one frame out. Frames are strictly sequential."""
        if self._cancelled():
            raise InterruptedError(f"{LOG_PREFIX} Cancelled.")
        header = struct.pack(FRAME_HEADER_FORMAT, FRAME_MAGIC, index, int(reset), 0, pts)
        self.worker.stdin.write(header)
        self.worker.stdin.write(_array_bytes(rgba, np.dtype(np.uint8)))
        self.worker.stdin.write(_array_bytes(motion, np.dtype(np.float16)))
        self.worker.stdin.flush()
        try:
            result = _read_exact(self.worker.stdout, struct.calcsize(RESULT_HEADER_FORMAT))
        except EOFError as exc:
            code = self.worker.wait(timeout=10)
            self._log_thread.join(timeout=2)
            raise RuntimeError(
                classify_worker_failure(
                    worker_code=code,
                    frame_index=index,
                    worker_logs=self.worker_logs,
                    reshade_lines=self.reshade_diagnostics(),
                )
            ) from exc
        magic, out_index, ok, byte_count, ngx_result, _pts = struct.unpack(
            RESULT_HEADER_FORMAT, result
        )
        expected = self.output_width * self.output_height * 4
        if magic != OUT_MAGIC or not ok or out_index != index or byte_count != expected:
            raise RuntimeError(f"{LOG_PREFIX} Invalid worker response for frame {index}.")
        if ngx_result != 1:
            raise RuntimeError(
                f"{LOG_PREFIX} Feature-18 evaluation failed on frame {index}: "
                f"0x{ngx_result:08X}"
            )
        output = np.empty((self.output_height, self.output_width, 4), dtype=np.uint8)
        _read_exact_into(self.worker.stdout, output)
        return output

    def close(self) -> None:
        if self.closed:
            return
        if self.worker.stdin and not self.worker.stdin.closed:
            self.worker.stdin.close()
        code = self.worker.wait(timeout=60)
        self._log_thread.join(timeout=2)
        self.closed = True
        if code:
            raise RuntimeError(
                f"{LOG_PREFIX} The DLSS worker failed:\n" + "\n".join(self.worker_logs[-40:])
            )

    def abort(self) -> None:
        """Kill the worker. Called on cancellation and on every failure path."""
        if self.closed:
            return
        if self.worker.poll() is None:
            try:
                self.worker.terminate()
                self.worker.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.worker.kill()
                except OSError:
                    pass
        self._log_thread.join(timeout=2)
        self.closed = True


def verify_feature_18(worker_logs: list[str], reshade_log: str) -> dict[str, Any]:
    """Prove the signed DLSSNR path actually ran, instead of assuming it did.

    ``NR upscaling fell back to native`` in the log means the pictures came back
    resized but NOT neurally rendered — the failure this check exists to catch.
    """
    created = "feature 18 created via the signed snippet" in reshade_log
    evaluated = "inline feature 18 evaluation succeeded" in reshade_log
    initialized = "signed DLSSNR 310.8.0 D3D12 runtime initialized" in reshade_log
    evidence = [
        line
        for line in reshade_log.splitlines()
        if "signed DLSSNR" in line
        or "feature 18 created" in line
        or "feature 18 evaluation succeeded" in line
        or "NR upscaling fell back" in line
    ]
    carrier = re.findall(
        r"DLSS carrier ready:.*result=0x([0-9A-Fa-f]{8})", "\n".join(worker_logs)
    )
    return {
        "verified": bool(created and evaluated and initialized),
        "nr_upscaling_active": "[upscaling]" in reshade_log and evaluated,
        "nr_native_fallback": "NR upscaling fell back to native" in reshade_log,
        "carrier_create_result": f"0x{carrier[-1].upper()}" if carrier else "unreported",
        "evidence": evidence,
    }
