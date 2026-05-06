"""TS Audio Loader — load audio/video tracks, record from microphone, crop visually.

node_id: TS_AudioLoader
"""

import hashlib
import math
import os

import folder_paths
from comfy_api.v0_0_2 import IO

from ._audio_helpers import (
    _decode_audio_segment,
    _empty_audio,
    _get_uploadable_media_options,
    _log_info,
    _log_warning,
    _normalize_path,
    _normalize_selected_path,
    _probe_media,
    _sanitize_crop,
    _seconds_to_hms,
)


class TS_AudioLoader(IO.ComfyNode):
    _MODES = ("load", "record")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_AudioLoader",
            display_name="TS Audio Loader",
            category="TS/Audio",
            description="Load audio or video audio tracks, preview waveform, record from microphone, and crop visually.",
            inputs=[
                IO.Combo.Input("mode", options=list(cls._MODES), default="load", tooltip="Load from file or use recorded microphone input.", socketless=True),
                IO.Combo.Input(
                    "source_path",
                    display_name="audio",
                    options=_get_uploadable_media_options(),
                    upload=IO.UploadType.audio,
                    tooltip="Choose file to upload or select an audio/video file from the input directory.",
                ),
                IO.Float.Input("crop_start_seconds", default=0.0, min=0.0, step=0.01, tooltip="Crop start time in seconds.", socketless=True, advanced=True),
                IO.Float.Input("crop_end_seconds", default=-1.0, step=0.01, tooltip="Crop end time in seconds. Use -1 for full length.", socketless=True, advanced=True),
            ],
            outputs=[IO.Audio.Output(display_name="audio"), IO.Int.Output(display_name="duration")],
            search_aliases=["audio loader", "audio crop", "record audio", "video audio"],
        )

    @classmethod
    def validate_inputs(cls, mode: str, source_path: str, crop_start_seconds: float, crop_end_seconds: float) -> bool | str:
        if mode not in cls._MODES:
            return f"Unsupported mode '{mode}'."
        if crop_start_seconds < 0:
            return "crop_start_seconds must be >= 0."
        if crop_end_seconds > 0 and crop_end_seconds <= crop_start_seconds:
            return "crop_end_seconds must be greater than crop_start_seconds."
        if not source_path:
            return True
        if folder_paths.exists_annotated_filepath(source_path):
            return True
        normalized = _normalize_selected_path(source_path)
        if not os.path.isfile(normalized):
            return f"Selected file does not exist: {source_path}"
        return True

    @classmethod
    def fingerprint_inputs(cls, mode: str, source_path: str, crop_start_seconds: float, crop_end_seconds: float) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(mode).encode("utf-8"))
        hasher.update(f"{float(crop_start_seconds):.6f}".encode("utf-8"))
        hasher.update(f"{float(crop_end_seconds):.6f}".encode("utf-8"))
        normalized = _normalize_selected_path(source_path)
        hasher.update(_normalize_path(normalized).encode("utf-8"))
        if os.path.isfile(normalized):
            stat = os.stat(normalized)
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def execute(cls, mode: str = "load", source_path: str = "", crop_start_seconds: float = 0.0, crop_end_seconds: float = -1.0) -> IO.NodeOutput:
        if not source_path:
            return IO.NodeOutput(_empty_audio(), 0)
        normalized = _normalize_selected_path(source_path)
        if not os.path.isfile(normalized):
            _log_warning(f"Selected file is missing: {source_path}")
            return IO.NodeOutput(_empty_audio(), 0)
        try:
            metadata = _probe_media(normalized)
            start_seconds, end_seconds = _sanitize_crop(metadata.duration_seconds, crop_start_seconds, crop_end_seconds)
            waveform, sample_rate = _decode_audio_segment(metadata, start_seconds, end_seconds)
            clip_duration_seconds = waveform.shape[-1] / max(1, int(sample_rate))
            duration_int = int(math.ceil(clip_duration_seconds)) if clip_duration_seconds > 0 else 0
            _log_info(
                "Decoded "
                f"mode={mode} file='{metadata.filename}' kind={metadata.media_type} "
                f"range={_seconds_to_hms(start_seconds)}..{_seconds_to_hms(end_seconds)} "
                f"sample_rate={sample_rate} channels={waveform.shape[0]} samples={waveform.shape[-1]}"
            )
            return IO.NodeOutput({"waveform": waveform.unsqueeze(0).contiguous(), "sample_rate": sample_rate}, duration_int)
        except Exception as exc:
            _log_warning(f"Execution fallback activated: {exc}")
            return IO.NodeOutput(_empty_audio(), 0)



NODE_CLASS_MAPPINGS = {"TS_AudioLoader": TS_AudioLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_AudioLoader": "TS Audio Loader"}
