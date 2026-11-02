"""Adapter to a separately installed official Visual Enhancer v21 CLI.

No upstream executable or implementation is distributed by this package.
v9 remains in-process; this adapter never loads v21 DLLs into ComfyUI.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
PINNED_FILES = {
    "VE_CLI.exe": "c12436f670ad44492c7d360cbfc0d2a34a9e0b777bd7fcd3f64202a57b1c4da9",
    "bin/runtime/dlssnr/neuroframe_engine_neural_rendering.dll":
        "be5323256fd5b43515fae9cd3d08c81fd711875eaac746723ae0943ae2fa804b",
    "bin/runtime/dlssnr/nvngx_dlssnr.dll":
        "cdfa6c1ec784867182164c7836f324f087b5c9f374d0c9fe46f54999feabf3ed",
}


def application_root(directory: str) -> Path:
    text = directory.strip().strip('"')
    if not text:
        raise ValueError("[TS DLSS Upscaler] Choose the folder of the complete official Visual Enhancer v21 installation in v21_directory.")
    root = Path(text).expanduser().resolve()
    if root.name.lower() == "ve_cli.exe":
        root = root.parent
    for relative, expected in PINNED_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"[TS DLSS Upscaler] Missing official v21 file: {path}. Extract the complete official application; do not copy just the CLI.")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected:
            raise ValueError(f"[TS DLSS Upscaler] {path.name} does not match the tested official v21.0 build. Keep v9 selected or install that build separately.")
    return root


def make_preset(settings: dict) -> dict:
    """Only documented application settings; resize happens before NR, as in v9."""
    values = {
        "image_enabled_stages": ["neural_model"],
        "video_enabled_stages": ["neural_model"],
        "nr_style": f"Style {settings['style']}",
        "nr_intensity": settings["intensity"],
        "nr_passes": settings["nr_passes"],
        "local_tone_strength": settings["local_tone"],
        "local_structure_strength": settings["local_structure"],
        "skin_structure_strength": settings["skin_structure"],
        "automatic_mask": bool(settings["auto_mask"]),
        "upscaling_factor": 1.0,
        "hdr_mode": False,
        "image_format": "PNG",
    }
    return {"format": "dlss5-visual-enhancer-settings-preset", "schema_version": 41,
            "name": "Timesaver external v21", "settings": values}


def run_cli(root, inputs, output, preset, *, video, ordinal, cancelled, frames, work):
    command = [str(root / "VE_CLI.exe"), "render", str(inputs), "-o", str(output),
               "--preset", str(preset), "--gpu", str(ordinal), "--json"]
    if video:
        command += ["--codec", "ffv1", "--container", "mkv"]
    else:
        command += ["--format", "png", "--bit-depth", "16"]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with (work / "stdout.json").open("wb") as stdout, (work / "stderr.log").open("wb") as stderr:
        cancelled()
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL, **options)
        deadline = time.monotonic() + max(120, 60 * frames)
        try:
            while process.poll() is None:
                cancelled()
                if time.monotonic() > deadline:
                    raise TimeoutError("Visual Enhancer v21 exceeded the render timeout")
                time.sleep(0.1)
        except BaseException:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.kill()
            process.wait(timeout=15)
            raise
    raw = (work / "stdout.json").read_text(encoding="utf-8", errors="replace")
    try:
        result = json.loads(raw)
    except ValueError:
        result = {}
    if process.returncode or not result.get("ok") or result.get("failed") or result.get("cancelled"):
        detail = result.get("error") or (work / "stderr.log").read_text(encoding="utf-8", errors="replace")[-1800:]
        raise RuntimeError(f"[TS DLSS Upscaler] External v21 failed: {detail}")


def render_frames(directory, source, *, width, height, settings, temporal, ordinal,
                  prepare, cancelled, progress=None, temp_root=None):
    import av
    import cv2
    import numpy as np

    root = application_root(directory)
    count = len(source)
    with tempfile.TemporaryDirectory(prefix="ts-dlss-v21-", dir=temp_root) as temporary:
        work = Path(temporary)
        preset = work / "preset.json"
        preset.write_text(json.dumps(make_preset(settings)), encoding="utf-8")
        video = temporal and count > 1
        if video:
            inputs, output = work / "input.mkv", work / "output.mkv"
            # Lossless 16-bit RGB transport. 24 fps is an internal carrier only;
            # no frame generation or timing conversion is enabled in the preset.
            with av.open(str(inputs), "w") as container:
                stream = container.add_stream("ffv1", rate=24)
                stream.width, stream.height, stream.pix_fmt = width, height, "gbrp16le"
                stream.options = {"level": "3"}
                for index in range(count):
                    cancelled()
                    pixels = np.rint(np.clip(prepare(index), 0, 1) * 65535).astype(np.uint16)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    if progress:
                        progress(index + 1, count * 2)
                for packet in stream.encode():
                    container.mux(packet)
        else:
            inputs, output = work / "input", work / "output"
            inputs.mkdir()
            output.mkdir()
            for index in range(count):
                cancelled()
                pixels = np.rint(np.clip(prepare(index), 0, 1) * 65535).astype(np.uint16)
                if not cv2.imwrite(str(inputs / f"frame_{index:08d}.png"), pixels[..., ::-1]):
                    raise RuntimeError("Could not write the v21 input frame")
                if progress:
                    progress(index + 1, count * 2)
        started = time.perf_counter()
        run_cli(root, inputs, output, preset, video=video, ordinal=ordinal,
                cancelled=cancelled, frames=count, work=work)
        logger.info("[TS DLSS Upscaler] External v21: %d frames, %.3f s in CLI (including startup and file I/O).",
                    count, time.perf_counter() - started)
        result = np.empty((count, height, width, 3), dtype=np.float32)
        if video:
            decoded = 0
            with av.open(str(output)) as container:
                for frame in container.decode(video=0):
                    cancelled()
                    if decoded >= count:
                        raise RuntimeError("External v21 changed the frame count")
                    pixels = frame.to_ndarray(format="rgb48le")
                    if pixels.shape != (height, width, 3):
                        raise RuntimeError("External v21 changed frame dimensions")
                    result[decoded] = pixels.astype(np.float32) / 65535
                    decoded += 1
                    if progress:
                        progress(count + decoded, count * 2)
            if decoded != count:
                raise RuntimeError(f"External v21 returned {decoded} frames; expected {count}")
        else:
            for index in range(count):
                cancelled()
                matches = list(output.glob(f"frame_{index:08d}*.png"))
                if len(matches) != 1:
                    raise RuntimeError(f"External v21 did not return exactly one image for frame {index}")
                pixels = cv2.imread(str(matches[0]), cv2.IMREAD_UNCHANGED)
                if pixels is None or pixels.shape[:2] != (height, width) or pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.uint16:
                    detail = "unreadable" if pixels is None else f"{pixels.shape}, {pixels.dtype}"
                    raise RuntimeError(f"External v21 did not return the requested 16-bit RGB image ({detail})")
                # The CLI may save an opaque alpha channel; original alpha is
                # transported separately by our node, never re-generated by NR.
                result[index] = pixels[..., :3][..., ::-1].astype(np.float32) / 65535
                if progress:
                    progress(count + index + 1, count * 2)
        if not np.isfinite(result).all():
            raise RuntimeError("External v21 returned non-finite pixels")
        return result
