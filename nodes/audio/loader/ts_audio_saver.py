"""Save a ComfyUI AUDIO batch and preview the first saved track."""

import os
import subprocess
import tempfile
from pathlib import Path

import folder_paths
from comfy_api.v0_0_2 import IO

from ..._ffmpeg import require_ffmpeg
from ..._shared import expand_date_tokens
from ._audio_helpers import _build_peaks_from_audio_tensor

_CODECS = {"WAV": ["-c:a", "pcm_s16le"], "FLAC": ["-c:a", "flac", "-sample_fmt", "s32"],
           "MP3": ["-c:a", "libmp3lame"]}
_MP3_BITRATES = ("320", "256", "192", "128")


def _write_track(track, rate, fmt, prefix, output_rate, codec_args):
    folder, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory()
    )
    suffix = "." + fmt.lower()
    # Use a seekable output so even WAV and MP3 headers carry the final duration.
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".ts-audio-", suffix=suffix, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        pcm = track.transpose(0, 1).contiguous().numpy().astype("<f4", copy=False)
        command = [require_ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "f32le",
                   "-ar", str(rate), "-ac", str(track.shape[0]), "-i", "pipe:0",
                   *codec_args, "-ar", str(output_rate), str(temp)]
        result = subprocess.run(command, input=memoryview(pcm).cast("B"), stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
        while True:
            target = Path(folder) / f"{stem}_{counter:05}_{suffix}"
            try:
                # Windows rename publishes atomically and refuses an existing name.
                if os.name == "nt":
                    os.rename(temp, target)
                    break
                target.touch(exist_ok=False)
            except FileExistsError:
                counter += 1
                continue
            try:
                os.replace(temp, target)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            break
        return target, {"filename": target.name, "subfolder": subfolder, "type": "output"}
    finally:
        temp.unlink(missing_ok=True)


class TS_AudioSaver(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TS_AudioSaver", display_name="TS Audio Saver", category="TS/Audio",
            description="Save each track under output and audition the first saved track.",
            inputs=[
                IO.Audio.Input("audio", tooltip="Full audio to export. Preview selection controls playback."),
                IO.String.Input("filename_prefix", default="audio/TS", tooltip="Output subfolder and file prefix, e.g. stems/%date:yyyy-MM-dd%/vocals_%date:hh-mm-ss%."),
                IO.Combo.Input("format", options=list(_CODECS), default="WAV", tooltip="WAV: 16-bit PCM at 44.1 kHz. FLAC: 24-bit at the input rate. MP3: constant bitrate."),
                IO.Combo.Input("mp3_bitrate", options=list(_MP3_BITRATES), default="320", optional=True,
                               display_name="MP3 bitrate (kbps)", tooltip="MP3 only. 320 kbps is the highest quality."),
            ], outputs=[], is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, filename_prefix="audio/TS", format="WAV", mp3_bitrate="320"):
        batch, rate = audio["waveform"], int(audio["sample_rate"])
        if batch.ndim != 3 or min(batch.shape) < 1 or rate < 1 or format not in _CODECS:
            raise ValueError("Expected a non-empty AUDIO batch, positive sample rate and WAV, FLAC or MP3.")
        if format == "MP3" and batch.shape[1] > 2:
            raise ValueError("MP3 supports mono or stereo. Use WAV or FLAC for multichannel audio.")
        output_rate = 44100 if format == "WAV" else rate
        codec_args = _CODECS[format]
        if format == "MP3":
            if str(mp3_bitrate) not in _MP3_BITRATES:
                raise ValueError("Unsupported MP3 bitrate.")
            # Lower MPEG sample rates cannot carry 320 kbps.
            output_rate = rate if rate in (32000, 44100, 48000) else 44100
            codec_args = [*codec_args, "-b:a", f"{mp3_bitrate}k"]
        prefix = expand_date_tokens(filename_prefix)
        paths, files = [], []
        try:
            for index, item in enumerate(batch):
                track = item.detach().cpu().float()
                if index == 0:
                    peaks = _build_peaks_from_audio_tensor(track, rate)
                path, entry = _write_track(track, rate, format, prefix.replace("%batch_num%", str(index)),
                                           output_rate, codec_args)
                paths.append(path)
                files.append(entry)
        except BaseException:
            for path in paths:
                path.unlink(missing_ok=True)
            raise
        first = files[0]
        preview = {"saved_audio": first, "filename": first["filename"], "media_type": "audio",
                   "duration_seconds": batch.shape[-1] / rate, "sample_rate": output_rate,
                   "channels": batch.shape[1], "peaks": peaks}
        return IO.NodeOutput(ui={"audio": files, "ts_audio_preview": [preview]})


NODE_CLASS_MAPPINGS = {"TS_AudioSaver": TS_AudioSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_AudioSaver": "TS Audio Saver"}
