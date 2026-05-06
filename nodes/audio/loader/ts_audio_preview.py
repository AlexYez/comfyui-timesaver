"""TS Audio Preview — waveform-aware preview UI for an upstream AUDIO input.

node_id: TS_AudioPreview
"""

import hashlib
from typing import Any

from comfy_api.latest import IO

from ._audio_helpers import (
    _build_generated_audio_preview_payload,
    _coerce_audio_tensor,
    _hash_audio_tensor,
    _log_warning,
)


class TS_AudioPreview(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_AudioPreview",
            display_name="TS Audio Preview",
            category="TS/Audio",
            description="Preview a standard ComfyUI audio input with waveform playback and looped auditioning.",
            inputs=[
                IO.Audio.Input("audio"),
                IO.Float.Input("crop_start_seconds", default=0.0, min=0.0, step=0.01, tooltip="Crop start time in seconds.", socketless=True, advanced=True),
                IO.Float.Input("crop_end_seconds", default=-1.0, step=0.01, tooltip="Crop end time in seconds. Use -1 for full length.", socketless=True, advanced=True),
                IO.String.Input("preview_state_json", default="", socketless=True, advanced=True, tooltip="Persistent preview UI state."),
            ],
            outputs=[],
            is_output_node=True,
            search_aliases=["audio preview", "preview audio", "audio crop preview"],
        )

    @classmethod
    def validate_inputs(cls, crop_start_seconds: float, crop_end_seconds: float, **_: Any) -> bool | str:
        if crop_start_seconds < 0:
            return "crop_start_seconds must be >= 0."
        if crop_end_seconds > 0 and crop_end_seconds <= crop_start_seconds:
            return "crop_end_seconds must be greater than crop_start_seconds."
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        audio: dict[str, Any],
        crop_start_seconds: float,
        crop_end_seconds: float,
        preview_state_json: str = "",
    ) -> str:
        waveform, sample_rate = _coerce_audio_tensor(audio)
        hasher = hashlib.sha256()
        hasher.update(_hash_audio_tensor(waveform, sample_rate, "TS_AudioPreview").encode("utf-8"))
        hasher.update(f"{float(crop_start_seconds):.6f}".encode("utf-8"))
        hasher.update(f"{float(crop_end_seconds):.6f}".encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def execute(
        cls,
        audio: dict[str, Any],
        crop_start_seconds: float = 0.0,
        crop_end_seconds: float = -1.0,
        preview_state_json: str = "",
    ) -> IO.NodeOutput:
        try:
            preview_payload = _build_generated_audio_preview_payload(audio, "TS_AudioPreview", "Incoming Audio")
            return IO.NodeOutput(ui={"ts_audio_preview": [preview_payload]})
        except Exception as exc:
            _log_warning(f"Audio preview fallback activated: {exc}")
            return IO.NodeOutput(ui={"ts_audio_preview": []})


NODE_CLASS_MAPPINGS = {"TS_AudioPreview": TS_AudioPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_AudioPreview": "TS Audio Preview"}
