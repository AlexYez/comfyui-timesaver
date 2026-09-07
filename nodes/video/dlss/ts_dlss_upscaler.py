"""TS DLSS Upscaler — NVIDIA DLSS 5 Neural Rendering for images and video batches.

One node: pictures in, upscaled pictures out. On its first run it downloads the
runtime it needs into ``models/DLSS`` and lays it out the way the worker expects.

⚠️ This is a TEMPORAL upscaler. A batch of consecutive video frames is fed with
estimated motion vectors, so DLSS reuses detail across frames — that is where
the quality above a still upscaler comes from. A batch of unrelated pictures
must be run with ``temporal`` off, or each picture drags the previous one's
detail behind it.

Windows and an NVIDIA RTX card only: the work is done by NVIDIA's signed D3D12
runtime, and there is no other implementation of it.
"""

from __future__ import annotations

import logging
import sys

from comfy_api.v0_0_2 import IO

from ..._deps import TSDependencyManager
from . import _assets
from ._dither import quantise_rgba16_to_rgba8
from ._session import (
    DLSSFrameSession,
    DLSS_MODEL_PRESETS,
    NR_STYLES,
    MIN_EDGE,
    UPSCALING_LABELS,
    factor_from_label,
    resolve_native_settings,
    resize_fit,
    resolve_output_size,
    resolve_upscaling_mode,
    verify_feature_18,
)
from ._tonemap import TRANSFER_CHOICES, TRANSFER_LABELS, needs_tone_map, tone_map_for

logger = logging.getLogger("comfyui_timesaver.ts_dlss_upscaler")
LOG_PREFIX = "[TS DLSS Upscaler]"

DEFAULT_FACTOR_LABEL = "2× (Performance)"
_CURVE_LABELS = [TRANSFER_LABELS[name] for name in TRANSFER_CHOICES]
_LABEL_TO_CURVE = {TRANSFER_LABELS[name]: name for name in TRANSFER_CHOICES}

_assets.register_model_folder()


class TS_DLSSUpscaler(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_DLSSUpscaler",
            display_name="TS DLSS Upscaler",
            category="TS/Video",
            description=(
                "Upscale a picture or a video batch with NVIDIA DLSS 5 Neural Rendering. "
                "Downloads its runtime into models/DLSS on first use. Windows + RTX only."
            ),
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "One picture or a batch of frames [B,H,W,C]. Both sides must be at "
                        "least 64 pixels."
                    ),
                ),
                IO.Combo.Input(
                    "upscaling_factor",
                    options=UPSCALING_LABELS,
                    default=DEFAULT_FACTOR_LABEL,
                    tooltip=(
                        "DLSS mode. 1× is DLAA: no resize, the network re-renders at the "
                        "same size. The output is capped at 7680×4320."
                    ),
                ),
                IO.Combo.Input(
                    "dlss_model_preset",
                    options=list(DLSS_MODEL_PRESETS),
                    default="M",
                    tooltip="Network revision. M measured best; the worker confirms what it applied.",
                ),
                IO.Combo.Input(
                    "nr_style",
                    options=list(NR_STYLES),
                    default="Default",
                    tooltip="Neural rendering look: Default, Natural or Cinematic.",
                ),
                IO.Float.Input(
                    "nr_intensity",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How strongly neural rendering is applied.",
                ),
                IO.Float.Input(
                    "local_tone_strength",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Local contrast recovered by the network.",
                ),
                IO.Float.Input(
                    "local_structure_strength",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Local detail recovered by the network.",
                ),
                IO.Float.Input(
                    "skin_structure_strength",
                    default=2.0, min=-1.0, max=2.0, step=0.01,
                    tooltip="Detail on skin, kept separate so faces do not turn plastic.",
                ),
                IO.Boolean.Input(
                    "automatic_mask",
                    default=True,
                    tooltip="Let the runtime decide where neural rendering is applied.",
                ),
                IO.Boolean.Input(
                    "temporal",
                    default=True,
                    tooltip=(
                        "Treat the batch as consecutive video frames and feed motion vectors. "
                        "Switch OFF for a batch of unrelated pictures. Ignored for a single image."
                    ),
                ),
                IO.Combo.Input(
                    "source_curve",
                    options=_CURVE_LABELS,
                    default=TRANSFER_LABELS["sdr"],
                    tooltip=(
                        "What the incoming values mean. The network only understands "
                        "display-referred SDR, so log/HDR/linear pictures are converted before "
                        "it and converted back after — exactly, by the same curve."
                    ),
                ),
                IO.Boolean.Input(
                    "dither",
                    default=True,
                    tooltip=(
                        "Blue-noise dither on the way down to the worker's 8 bits, so gradients "
                        "reach the network as detail instead of steps."
                    ),
                ),
                IO.Boolean.Input(
                    "download_if_missing",
                    default=True,
                    tooltip=(
                        "Fetch the runtime into models/DLSS when it is not there "
                        "(~481 MB, once). This is your agreement to download third-party "
                        "components — NVIDIA (proprietary), ReShade (BSD-3), RenoDX — from "
                        "the upstream project. This pack hosts none of them. Off, the node "
                        "downloads nothing and you place the files yourself."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="images",
                    tooltip="The upscaled batch, same order, same channel count.",
                )
            ],
            search_aliases=["dlss", "dlss 5", "neural rendering", "upscale", "nvidia"],
        )

    # ------------------------------------------------------------------ guards
    @classmethod
    def _require_platform(cls) -> None:
        if sys.platform != "win32":
            raise RuntimeError(
                f"{LOG_PREFIX} DLSS 5 Neural Rendering runs through NVIDIA's signed D3D12 "
                "runtime, which exists for Windows only."
            )

    @classmethod
    def _require_numpy(cls):
        numpy = TSDependencyManager.import_optional("numpy")
        if numpy is None:
            raise RuntimeError(f"{LOG_PREFIX} numpy is required.")
        return numpy

    # ------------------------------------------------------------------- frames
    @classmethod
    def _to_rgba8(cls, np, frame, tone_map, dither: bool):
        """One IMAGE frame -> the RGBA8 bytes the worker takes."""
        rgb = np.asarray(frame[..., :3], dtype=np.float32)
        if tone_map is not None:
            rgb = tone_map.forward(rgb)
        rgb16 = np.clip(np.rint(np.clip(rgb, 0.0, 1.0) * 65535.0), 0, 65535).astype(np.uint16)
        if frame.shape[-1] >= 4:
            alpha = np.asarray(frame[..., 3], dtype=np.float32)
            alpha16 = np.clip(np.rint(np.clip(alpha, 0.0, 1.0) * 65535.0), 0, 65535)
        else:
            alpha16 = np.full(rgb16.shape[:2], 65535.0)
        rgba16 = np.concatenate(
            [rgb16, alpha16.astype(np.uint16)[..., None]], axis=-1
        )
        return np.ascontiguousarray(quantise_rgba16_to_rgba8(rgba16, dither=dither))

    @classmethod
    def _from_rgba8(cls, np, out8, tone_map):
        """The worker's RGBA8 result -> IMAGE floats."""
        rgb = out8[..., :3].astype(np.float32) / 255.0
        if tone_map is not None:
            rgb = tone_map.inverse(rgb)
        return rgb

    @classmethod
    def _resize_alpha(cls, np, alpha, width: int, height: int):
        """Carry the source alpha to the output size; the worker ignores alpha."""
        if alpha.shape[0] == height and alpha.shape[1] == width:
            return alpha
        cv2 = TSDependencyManager.import_optional("cv2")
        if cv2 is None:
            # Without OpenCV a nearest-neighbour lift is still better than losing it.
            ys = (np.arange(height) * alpha.shape[0] // height).clip(0, alpha.shape[0] - 1)
            xs = (np.arange(width) * alpha.shape[1] // width).clip(0, alpha.shape[1] - 1)
            return alpha[ys][:, xs]
        return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # ------------------------------------------------------------------ execute
    @classmethod
    def execute(
        cls,
        images,
        upscaling_factor: str,
        dlss_model_preset: str,
        nr_style: str,
        nr_intensity: float,
        local_tone_strength: float,
        local_structure_strength: float,
        skin_structure_strength: float,
        automatic_mask: bool,
        temporal: bool,
        source_curve: str,
        dither: bool,
        download_if_missing: bool,
    ) -> IO.NodeOutput:
        import torch  # noqa: PLC0415 - always present in ComfyUI, never at import time

        cls._require_platform()
        np = cls._require_numpy()

        if images is None or images.ndim != 4:
            raise ValueError(f"{LOG_PREFIX} 'images' must be a batch shaped [B,H,W,C].")
        batch, height, width, channels = (int(value) for value in images.shape)
        if height < MIN_EDGE or width < MIN_EDGE:
            raise ValueError(
                f"{LOG_PREFIX} DLSS needs at least {MIN_EDGE}×{MIN_EDGE} pixels; "
                f"this batch is {width}×{height}."
            )

        factor, mode = resolve_upscaling_mode(factor_from_label(upscaling_factor))
        output_width, output_height = resolve_output_size(width, height, factor)
        native = resolve_native_settings(
            nr_preset="Default",            # not exposed: the app keeps it at Default
            nr_style=nr_style,
            dlss_model_preset=dlss_model_preset,
            nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            automatic_mask=automatic_mask,
        )

        curve = _LABEL_TO_CURVE.get(source_curve, source_curve)
        tone_map = tone_map_for(curve) if needs_tone_map(curve) else None

        root = _assets.ensure_runtime(
            download_if_missing=download_if_missing,
            progress=cls._download_progress(),
        )

        # ⚠️ The source tensor is never written to: everything below reads from a
        # CPU copy and builds new arrays.
        source = images.detach().to("cpu")

        session = DLSSFrameSession(
            root,
            input_width=width,
            input_height=height,
            output_width=output_width,
            output_height=output_height,
            frame_count=batch,
            mode=mode,
            native_settings=native,
            cancelled=cls._cancelled,
        )
        logger.info(
            "%s %d frame(s) %d×%d -> %d×%d, %s, model %s%s.",
            LOG_PREFIX, batch, width, height, output_width, output_height,
            mode["name"], dlss_model_preset,
            ", temporal" if (temporal and batch > 1) else "",
        )

        guides = None
        if temporal and batch > 1:
            from ._guides import TemporalGuideGenerator  # noqa: PLC0415 - needs OpenCV

            guides = TemporalGuideGenerator(session.render_width, session.render_height)
        zero_motion = np.zeros(
            (session.render_height, session.render_width, 2), dtype=np.float16
        )

        progress = cls._progress_bar(batch)
        results = np.empty((batch, output_height, output_width, min(channels, 4)),
                           dtype=np.float32)
        try:
            for index in range(batch):
                cls._raise_if_interrupted()
                frame = source[index].numpy()
                rgba8 = cls._to_rgba8(np, frame, tone_map, dither)
                rgba8 = resize_fit(rgba8, session.render_width, session.render_height)
                if guides is None:
                    motion, reset = zero_motion, True
                else:
                    guide = guides.process(rgba8)
                    motion, reset = guide.motion, guide.reset
                out8 = session.process(
                    index=index, rgba=rgba8, motion=motion, reset=reset, pts=index
                )
                results[index, ..., :3] = cls._from_rgba8(np, out8, tone_map)
                if channels >= 4:
                    results[index, ..., 3] = cls._resize_alpha(
                        np, np.asarray(frame[..., 3], dtype=np.float32),
                        output_width, output_height,
                    )
                if progress is not None:
                    progress.update_absolute(index + 1, batch)
            session.close()
        except BaseException:
            session.abort()
            raise

        cls._report_evidence(session)
        output = torch.from_numpy(np.clip(results, 0.0, 1.0))
        return IO.NodeOutput(output.to(images.device))

    # ------------------------------------------------------------------ plumbing
    @classmethod
    def _cancelled(cls) -> bool:
        try:
            import comfy.model_management as mm  # noqa: PLC0415

            return bool(mm.processing_interrupted())
        except Exception:  # noqa: BLE001 - outside ComfyUI there is nothing to cancel
            return False

    @classmethod
    def _raise_if_interrupted(cls) -> None:
        try:
            import comfy.model_management as mm  # noqa: PLC0415

            mm.throw_exception_if_processing_interrupted()
        except ImportError:
            pass

    @classmethod
    def _progress_bar(cls, total: int):
        try:
            import comfy.utils  # noqa: PLC0415

            return comfy.utils.ProgressBar(total)
        except Exception:  # noqa: BLE001 - no server, no bar
            return None

    @classmethod
    def _download_progress(cls):
        """A progress bar for the one-off runtime download, in whole megabytes."""
        state: dict[str, object] = {"bar": None}

        def report(done: int, total: int) -> None:
            if not total:
                return
            bar = state["bar"]
            if bar is None:
                bar = cls._progress_bar(total // (1024 * 1024) or 1)
                state["bar"] = bar
            if bar is not None:
                bar.update_absolute(done // (1024 * 1024), total // (1024 * 1024) or 1)

        return report

    @classmethod
    def _report_evidence(cls, session) -> None:
        """Say once whether the signed feature-18 path really ran.

        ⚠️ Reported, never enforced: the pictures are already upscaled, and
        refusing them because a log line is missing would throw away work. A
        fallback to plain resizing is exactly what the user needs told.
        """
        try:
            evidence = verify_feature_18(session.worker_logs, session.reshade_log_text())
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail a good run
            logger.debug("%s Could not read the ReShade evidence: %s", LOG_PREFIX, exc)
            return
        if evidence["nr_native_fallback"]:
            logger.warning(
                "%s Neural rendering fell back to a plain resize — the result is NOT "
                "DLSS-enhanced. Update the NVIDIA driver.", LOG_PREFIX
            )
        elif evidence["verified"]:
            logger.info("%s Signed DLSSNR feature 18 confirmed by ReShade.", LOG_PREFIX)
        else:
            logger.warning(
                "%s The run finished but the signed feature-18 evidence is incomplete: %s",
                LOG_PREFIX, "; ".join(evidence["evidence"][-3:]) or "no DLSSNR lines logged",
            )


NODE_CLASS_MAPPINGS = {"TS_DLSSUpscaler": TS_DLSSUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_DLSSUpscaler": "TS DLSS Upscaler"}
