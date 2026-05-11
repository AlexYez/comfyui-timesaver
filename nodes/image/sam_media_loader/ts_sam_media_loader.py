"""TS SAM Media Loader - load image/video and pick SAM3 points interactively.

node_id: TS_SAM_MediaLoader

Loads an image or a video into the workflow and exposes positive/negative point
prompts compatible with the native ComfyUI ``SAM3 Detect`` node
(``comfy_extras/nodes_sam3.py``). That node consumes ``positive_coords`` and
``negative_coords`` as plain STRING JSON of the form
``[{"x": int, "y": int}, ...]`` in pixel coordinates of the input image.

The interactive UI (file picker, first-frame preview, click-to-add points)
lives in ``js/image/sam_media_loader/``. ``execute`` re-reads the media file
selected via the in-node UI and converts the persisted JSON coordinates into
the SAM3 STRING prompt format. The optional ``model`` input is preview-only
(the JS overlay calls a backend route that runs SAM3 on the first frame to
visualise the currently selected mask) and also enables the
``initial_mask`` output: when both ``model`` and at least one point are
present, ``execute`` runs SAM3 Detect on the first frame and returns the same
mask the user sees in the overlay, ready to feed into ``SAM3 Video Track``
as ``initial_mask``.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
from comfy.utils import ProgressBar

from comfy_api.v0_0_2 import IO

from ._sam_media_helpers import (
    LOG_PREFIX,
    _classify_media,
    _empty_audio,
    _extract_video_audio,
    _load_image_tensor,
    _load_video_tensor,
    _log_info,
    _log_warning,
    _normalize_path,
    _resolve_media_path,
    _video_meta,
    _working_file_signature,
)


def _parse_points(raw: str) -> list[dict[str, float]]:
    """Parse a coordinates JSON string into a list of ``{x, y}`` dicts."""
    if not raw or not isinstance(raw, str):
        return []
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    points: list[dict[str, float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        points.append({"x": x, "y": y})
    return points


def _serialize_sam3_coords(
    points: list[dict[str, float]],
    width: int,
    height: int,
) -> str:
    """Serialize pixel-space points as JSON for ``SAM3 Detect``.

    The native node parses ``positive_coords`` / ``negative_coords`` with
    ``json.loads(...)`` and expects a list of dicts ``{"x": int, "y": int}``
    in the *pixel* coordinate space of the image input.
    """
    if not points:
        return "[]"
    max_x = max(0, int(width) - 1)
    max_y = max(0, int(height) - 1)
    serialized: list[dict[str, int]] = []
    for point in points:
        try:
            x = int(round(float(point.get("x", 0.0))))
            y = int(round(float(point.get("y", 0.0))))
        except (TypeError, ValueError):
            continue
        if max_x > 0:
            x = max(0, min(max_x, x))
        if max_y > 0:
            y = max(0, min(max_y, y))
        serialized.append({"x": x, "y": y})
    return json.dumps(serialized, separators=(",", ":"))


class TS_SAM_MediaLoader(IO.ComfyNode):
    """Load an image or video and produce SAM3-compatible point prompts."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_SAM_MediaLoader",
            display_name="TS SAM Media Loader",
            category="TS/Image",
            description=(
                "Load an image or video and click positive (green) / negative "
                "(red) points directly on the first frame. Outputs IMAGE "
                "(batch=N frames for video) plus positive_points and "
                "negative_points in the SAM3 prompt format consumed by the "
                "native SAM3 Point Segmentation / Video Segmentation nodes."
            ),
            inputs=[
                IO.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Optional SAM3 model. Connect a SAM3 checkpoint loader "
                        "to enable the in-node blue mask preview overlay. "
                        "The link is preview-only: the workflow's SAM3 Detect "
                        "should be wired to the loader directly."
                    ),
                ),
                IO.String.Input(
                    "source_path",
                    default="",
                    tooltip="Internal: annotated path of the loaded image or video.",
                    socketless=True,
                ),
                IO.String.Input(
                    "media_type",
                    default="",
                    tooltip="Internal: 'image' or 'video'.",
                    socketless=True,
                ),
                IO.String.Input(
                    "coordinates",
                    default="[]",
                    tooltip="Internal: positive point JSON (pixel space of first frame).",
                    socketless=True,
                ),
                IO.String.Input(
                    "neg_coordinates",
                    default="[]",
                    tooltip="Internal: negative point JSON (pixel space of first frame).",
                    socketless=True,
                ),
                IO.String.Input(
                    "sam3_checkpoint",
                    default="",
                    tooltip=(
                        "Internal: auto-detected checkpoint filename of the "
                        "connected SAM3 model loader. Used by the preview "
                        "overlay route only."
                    ),
                    socketless=True,
                ),
                IO.Int.Input(
                    "max_frames",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Hard cap on decoded video frames (0 = no cap).",
                    socketless=True,
                ),
                IO.Int.Input(
                    "frame_stride",
                    default=1,
                    min=1,
                    max=60,
                    step=1,
                    tooltip="Decode every N-th video frame (1 = every frame).",
                    socketless=True,
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.String.Output(display_name="positive_coords"),
                IO.String.Output(display_name="negative_coords"),
                IO.Float.Output(display_name="fps"),
                IO.Audio.Output(display_name="audio"),
                IO.Mask.Output(display_name="initial_mask"),
            ],
            search_aliases=[
                "sam",
                "sam3",
                "sam media",
                "sam points",
                "video sam",
                "media loader",
                "load video",
            ],
        )

    @classmethod
    def validate_inputs(cls, **_kwargs) -> bool:
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        model=None,
        source_path: str = "",
        media_type: str = "",
        coordinates: str = "[]",
        neg_coordinates: str = "[]",
        sam3_checkpoint: str = "",
        max_frames: int = 0,
        frame_stride: int = 1,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(_normalize_path(str(source_path)).encode("utf-8"))
        hasher.update(str(media_type or "").encode("utf-8"))
        hasher.update(str(coordinates or "[]").encode("utf-8"))
        hasher.update(str(neg_coordinates or "[]").encode("utf-8"))
        hasher.update(f"{int(max_frames)}|{int(frame_stride)}".encode("utf-8"))
        resolved = _resolve_media_path(source_path)
        hasher.update(_working_file_signature(resolved).encode("utf-8"))
        # ``model`` and ``sam3_checkpoint`` only influence the in-node preview
        # overlay (a UI feature), not the execute outputs, so they are
        # deliberately excluded from the cache fingerprint.
        return hasher.hexdigest()

    @classmethod
    def execute(
        cls,
        model=None,
        source_path: str = "",
        media_type: str = "",
        coordinates: str = "[]",
        neg_coordinates: str = "[]",
        sam3_checkpoint: str = "",
        max_frames: int = 0,
        frame_stride: int = 1,
    ) -> IO.NodeOutput:
        resolved = _resolve_media_path(source_path)
        if not resolved or not os.path.isfile(resolved):
            raise RuntimeError(
                f"{LOG_PREFIX} No media loaded. Click 'Load Image/Video' inside the node "
                "before queuing the workflow."
            )
        detected = _classify_media(resolved)
        if not detected:
            raise RuntimeError(f"{LOG_PREFIX} Unsupported media type for: {resolved}")
        # Prefer the on-disk detection over the persisted media_type so the
        # node remains correct if the user replaces the file outside the UI.
        media_type = detected

        if media_type == "image":
            images = _load_image_tensor(resolved)
            fps = 0.0
            audio = _empty_audio()
        else:
            # ---- Phase 1: decode video frames (per-frame ProgressBar) ----
            # Probe first so the bar can show "kept/N" instead of being
            # indeterminate. Probe failure is non-fatal — we just lose the
            # nice frame counter.
            try:
                meta = _video_meta(resolved)
                total_frames = int(meta.get("frame_count") or 0)
                fps_raw = float(meta.get("fps") or 0.0)
            except Exception as exc:
                _log_warning(f"Could not probe video '{resolved}': {exc}")
                total_frames = 0
                fps_raw = 0.0
            stride = max(1, int(frame_stride or 1))
            limit = max(0, int(max_frames or 0))
            if total_frames > 0:
                estimated_kept = (total_frames + stride - 1) // stride
            else:
                estimated_kept = 0
            if limit > 0:
                estimated_kept = (
                    min(estimated_kept, limit) if estimated_kept > 0 else limit
                )

            decode_pbar = ProgressBar(max(1, estimated_kept))

            def _decode_progress(kept_index: int, total: int) -> None:
                try:
                    decode_pbar.update_absolute(
                        int(kept_index), total=int(max(1, total))
                    )
                except Exception:
                    pass

            images = _load_video_tensor(
                resolved,
                max_frames=limit,
                frame_stride=stride,
                progress_callback=_decode_progress,
            )
            # frame_stride subsamples the timeline, so the effective output
            # fps is the source fps divided by the stride.
            fps = fps_raw / stride if fps_raw > 0 else 0.0

            # ---- Phase 2: extract audio (single tick progress) ----
            # Audio decoding is dominated by ffmpeg subprocess startup +
            # demux; per-sample progress would mean piping stderr which
            # is overkill. A 0 -> 1 bar is enough to show the phase exists.
            audio_pbar = ProgressBar(1)
            audio_pbar.update_absolute(0, total=1)
            audio = _extract_video_audio(resolved)
            audio_pbar.update_absolute(1, total=1)

        # Image tensor is [N, H, W, 3]; first-frame H/W match all frames.
        height = int(images.shape[1])
        width = int(images.shape[2])

        positive_pts = _parse_points(coordinates)
        negative_pts = _parse_points(neg_coordinates)
        positive_coords = _serialize_sam3_coords(positive_pts, width, height)
        negative_coords = _serialize_sam3_coords(negative_pts, width, height)

        initial_mask = _compute_initial_mask(
            model=model,
            images=images,
            positive_coords=positive_coords,
            negative_coords=negative_coords,
            has_any_point=bool(positive_pts or negative_pts),
        )

        _log_info(
            f"Loaded {media_type} '{os.path.basename(resolved)}' "
            f"size={width}x{height} frames={images.shape[0]} fps={fps:.3f} "
            f"audio_samples={int(audio['waveform'].shape[-1])} "
            f"pos={len(positive_pts)} neg={len(negative_pts)} "
            f"initial_mask_pixels={int((initial_mask > 0.5).sum())}"
        )

        return IO.NodeOutput(
            images,
            positive_coords,
            negative_coords,
            float(fps),
            audio,
            initial_mask,
        )


def _compute_initial_mask(
    model,
    images: "torch.Tensor",
    positive_coords: str,
    negative_coords: str,
    has_any_point: bool,
) -> "torch.Tensor":
    """Run SAM3 Detect on the first frame to produce a seed mask.

    Returns ``[1, H, W]`` float32 ``MASK`` in ``[0, 1]``. Falls back to zeros
    when:
        - no model is connected,
        - no positive/negative points were placed,
        - SAM3 Detect raised (the workflow can still continue).

    The output dimensions match the first frame of ``images`` so that
    ``SAM3 Video Track`` can consume it directly as ``initial_mask`` for the
    rest of the batch.
    """
    height = int(images.shape[1])
    width = int(images.shape[2])
    empty = torch.zeros((1, height, width), dtype=torch.float32)

    if model is None or not has_any_point:
        return empty
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect
    except Exception as exc:
        _log_warning(f"SAM3 Detect unavailable, initial_mask = zeros: {exc}")
        return empty

    first_frame = images[0:1].contiguous()
    try:
        with torch.no_grad():
            output = SAM3_Detect.execute(
                model=model,
                image=first_frame,
                conditioning=None,
                bboxes=None,
                positive_coords=positive_coords,
                negative_coords=negative_coords,
                threshold=0.5,
                refine_iterations=2,
                individual_masks=False,
            )
    except Exception as exc:
        _log_warning(f"SAM3 Detect failed while seeding initial_mask: {exc}")
        return empty

    if output is None or output.result is None or len(output.result) == 0:
        return empty
    mask_tensor = output.result[0]
    if not torch.is_tensor(mask_tensor):
        return empty
    if mask_tensor.ndim == 3:
        return mask_tensor.detach().to(torch.float32).cpu().contiguous()
    if mask_tensor.ndim == 2:
        return mask_tensor.detach().to(torch.float32).cpu().unsqueeze(0).contiguous()
    return empty


NODE_CLASS_MAPPINGS = {"TS_SAM_MediaLoader": TS_SAM_MediaLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SAM_MediaLoader": "TS SAM Media Loader"}
