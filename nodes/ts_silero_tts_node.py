import logging
import os
import re
from typing import Any

import importlib
import inspect
import shutil

import torch

import comfy.model_management
from comfy.utils import ProgressBar
import folder_paths
from comfy_api.latest import IO


class TS_SileroTTS(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_silero_tts")
    _LOG_PREFIX = "[TS SileroTTS]"

    _MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_3_ru.pt"
    _MODEL_DIR_NAME = "silerotts"
    _MODEL_FILE_NAME = "v5_3_ru.pt"
    _MODEL_PACKAGE = "tts_models"
    _MODEL_PICKLE = "model"
    _SAMPLE_RATE = 48000
    _SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")
    _INPUT_FORMATS = ("text", "ssml")
    _RUN_DEVICES = ("gpu", "cpu")
    _DEFAULT_MAX_CHUNK_CHARS = 900
    _DEFAULT_CHUNK_PAUSE_MS = 120

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_SileroTTS",
            display_name="TS Silero TTS",
            category="TS/audio",
            description="Silero TTS v5_3_ru with ComfyUI AUDIO output.",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Text or SSML content for speech synthesis.",
                ),
                IO.Combo.Input(
                    "input_format",
                    options=list(cls._INPUT_FORMATS),
                    default="text",
                    tooltip="Input mode: plain text or SSML.",
                ),
                IO.Combo.Input(
                    "speaker",
                    options=list(cls._SPEAKERS),
                    default="xenia",
                    tooltip="Silero speaker voice.",
                ),
                IO.Combo.Input(
                    "run_device",
                    options=list(cls._RUN_DEVICES),
                    default="gpu",
                    tooltip="Execution device for Silero model.",
                ),
                IO.Boolean.Input(
                    "enable_chunking",
                    default=True,
                    tooltip="Automatically split long text into chunks to avoid Silero length limits.",
                ),
                IO.Int.Input(
                    "max_chunk_chars",
                    default=cls._DEFAULT_MAX_CHUNK_CHARS,
                    min=200,
                    max=4000,
                    step=50,
                    advanced=True,
                    tooltip="Approximate maximum characters per chunk.",
                ),
                IO.Int.Input(
                    "chunk_pause_ms",
                    default=cls._DEFAULT_CHUNK_PAUSE_MS,
                    min=0,
                    max=3000,
                    step=10,
                    advanced=True,
                    tooltip="Silence between generated chunks in milliseconds.",
                ),
                IO.Boolean.Input(
                    "put_accent",
                    default=True,
                    tooltip="Add stress marks to common words where user did not provide them.",
                ),
                IO.Boolean.Input(
                    "put_yo",
                    default=True,
                    tooltip="Replace e with yo where needed.",
                ),
                IO.Boolean.Input(
                    "put_stress_homo",
                    default=True,
                    tooltip="Add stress marks for homographs without yo.",
                ),
                IO.Boolean.Input(
                    "put_yo_homo",
                    default=True,
                    tooltip="Add stress marks for homographs with yo.",
                ),
            ],
            outputs=[IO.Audio.Output(display_name="audio")],
            search_aliases=["silero", "tts", "russian speech"],
        )

    @classmethod
    def validate_inputs(cls, text, input_format, speaker, run_device, **kwargs) -> bool | str:
        if not isinstance(text, str) or not text.strip():
            return "Text must not be empty."
        if speaker not in cls._SPEAKERS:
            return f"Unsupported speaker '{speaker}'."
        if input_format not in cls._INPUT_FORMATS:
            return f"Unsupported input_format '{input_format}'."
        if run_device not in cls._RUN_DEVICES:
            return f"Unsupported run_device '{run_device}'."
        max_chunk_chars = kwargs.get("max_chunk_chars", cls._DEFAULT_MAX_CHUNK_CHARS)
        chunk_pause_ms = kwargs.get("chunk_pause_ms", cls._DEFAULT_CHUNK_PAUSE_MS)
        try:
            if int(max_chunk_chars) < 200:
                return "max_chunk_chars must be >= 200."
            if int(chunk_pause_ms) < 0:
                return "chunk_pause_ms must be >= 0."
        except Exception:
            return "Invalid chunking parameters."
        return True

    @classmethod
    def _log_info(cls, message: str) -> None:
        cls._LOGGER.info("%s %s", cls._LOG_PREFIX, message)

    @classmethod
    def _log_warning(cls, message: str) -> None:
        cls._LOGGER.warning("%s %s", cls._LOG_PREFIX, message)

    @classmethod
    def _model_path(cls) -> str:
        model_dir = os.path.join(folder_paths.models_dir, cls._MODEL_DIR_NAME)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, cls._MODEL_FILE_NAME)

    @classmethod
    def _ensure_model_exists(cls, model_path: str) -> None:
        if os.path.isfile(model_path):
            return
        cls._log_info(f"Downloading model to: {model_path}")
        try:
            torch.hub.download_url_to_file(cls._MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Silero model v5_3_ru to models\\silerotts. Target path: {model_path}"
            ) from exc

    @classmethod
    def _resolve_device(cls, run_device: str) -> torch.device:
        if run_device == "cpu":
            return torch.device("cpu")

        target_device = comfy.model_management.get_torch_device()
        if target_device.type == "cpu":
            cls._log_warning("GPU requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        return target_device

    @classmethod
    def _load_model(cls, model_path: str, device: torch.device):
        try:
            package = torch.package.PackageImporter(model_path)
            model = package.load_pickle(cls._MODEL_PACKAGE, cls._MODEL_PICKLE)
            if hasattr(model, "to"):
                model.to(device)
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load Silero model from '{model_path}'.") from exc

    @classmethod
    def _extract_audio_path(cls, value: Any) -> str | None:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return value[0]
        return None

    @classmethod
    def _is_text_too_long_error(cls, exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "too long" in message
            or "size of tensor a" in message
            or "must match the size of tensor b" in message
            or "probably it's too long" in message
        )

    @classmethod
    def _pack_segments(cls, segments: list[str], max_chunk_chars: int) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for segment in segments:
            part = segment.strip()
            if not part:
                continue
            part_len = len(part)

            if part_len > max_chunk_chars:
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                    current_len = 0
                words = part.split()
                running_words: list[str] = []
                running_len = 0
                for word in words:
                    wlen = len(word)
                    sep = 1 if running_words else 0
                    if running_words and running_len + sep + wlen > max_chunk_chars:
                        chunks.append(" ".join(running_words).strip())
                        running_words = [word]
                        running_len = wlen
                    else:
                        if sep:
                            running_len += 1
                        running_words.append(word)
                        running_len += wlen
                if running_words:
                    chunks.append(" ".join(running_words).strip())
                continue

            sep = 1 if current else 0
            if current and current_len + sep + part_len > max_chunk_chars:
                chunks.append(" ".join(current).strip())
                current = [part]
                current_len = part_len
            else:
                if sep:
                    current_len += 1
                current.append(part)
                current_len += part_len

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if chunk]

    @classmethod
    def _split_text_into_chunks(cls, text: str, max_chunk_chars: int) -> list[str]:
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not clean_text:
            return []
        if len(clean_text) <= max_chunk_chars:
            return [clean_text]

        sentence_segments = re.split(r"(?<=[\.\!\?\u2026])\s+", clean_text)
        sentence_segments = [seg.strip() for seg in sentence_segments if seg.strip()]
        if not sentence_segments:
            sentence_segments = [clean_text]

        return cls._pack_segments(sentence_segments, max_chunk_chars)

    @classmethod
    def _split_ssml_into_chunks(cls, ssml_text: str, max_chunk_chars: int) -> list[str]:
        raw = ssml_text.strip()
        if not raw:
            return []
        if len(raw) <= max_chunk_chars:
            return [raw]

        paragraph_blocks = re.findall(
            r"<p\b[^>]*>.*?</p>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if paragraph_blocks:
            packed = cls._pack_segments(paragraph_blocks, max(200, max_chunk_chars - 30))
            return [f"<speak>{chunk}</speak>" for chunk in packed if chunk]

        sentence_blocks = re.findall(
            r"<s\b[^>]*>.*?</s>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if sentence_blocks:
            packed = cls._pack_segments(sentence_blocks, max(200, max_chunk_chars - 30))
            return [f"<speak>{chunk}</speak>" for chunk in packed if chunk]

        cls._log_warning(
            "SSML chunking fallback activated: preserving full markup is not possible for this structure."
        )
        plain_text = re.sub(r"<[^>]+>", " ", raw)
        plain_chunks = cls._split_text_into_chunks(plain_text, max_chunk_chars)
        return [f"<speak>{chunk}</speak>" for chunk in plain_chunks if chunk]

    @classmethod
    def _build_chunks(
        cls,
        input_format: str,
        text: str,
        enable_chunking: bool,
        max_chunk_chars: int,
    ) -> list[str]:
        if not enable_chunking:
            return [text]
        if input_format == "ssml":
            return cls._split_ssml_into_chunks(text, max_chunk_chars)
        return cls._split_text_into_chunks(text, max_chunk_chars)

    @classmethod
    def _synthesize_single_chunk(
        cls,
        model,
        input_format: str,
        text: str,
        speaker: str,
        put_accent: bool,
        put_yo: bool,
        put_stress_homo: bool,
        put_yo_homo: bool,
    ) -> torch.Tensor:
        is_ssml = input_format == "ssml"
        tts_kwargs = {
            "speaker": speaker,
            "sample_rate": cls._SAMPLE_RATE,
            "put_accent": put_accent,
            "put_yo": put_yo,
            "put_stress_homo": put_stress_homo,
            "put_yo_homo": put_yo_homo,
        }
        if is_ssml:
            tts_kwargs["ssml_text"] = text
        else:
            tts_kwargs["text"] = text

        if hasattr(model, "apply_tts"):
            try:
                audio = model.apply_tts(**tts_kwargs)
                return cls._normalize_generated_audio(audio)
            except TypeError:
                # Some builds may not accept all flags or SSML-specific kwargs.
                fallback_kwargs = {"speaker": speaker, "sample_rate": cls._SAMPLE_RATE}
                if is_ssml:
                    fallback_kwargs["ssml_text"] = text
                else:
                    fallback_kwargs["text"] = text
                    fallback_kwargs["put_accent"] = put_accent
                    fallback_kwargs["put_yo"] = put_yo
                audio = model.apply_tts(**fallback_kwargs)
                return cls._normalize_generated_audio(audio)

        if is_ssml:
            raise RuntimeError("SSML mode requires Silero apply_tts(ssml_text=...). save_wav fallback is not supported.")

        if hasattr(model, "save_wav"):
            save_kwargs = dict(tts_kwargs)
            try:
                path_value = model.save_wav(**save_kwargs)
            except TypeError:
                path_value = model.save_wav(
                    text=text,
                    speaker=speaker,
                    sample_rate=cls._SAMPLE_RATE,
                )

            audio_path = cls._extract_audio_path(path_value)
            if audio_path is None or not os.path.isfile(audio_path):
                raise RuntimeError("Silero save_wav returned an invalid audio path.")
            try:
                try:
                    import torchaudio
                except Exception as exc:
                    raise RuntimeError("torchaudio is required for save_wav fallback.") from exc

                waveform, loaded_sr = torchaudio.load(audio_path)
                if loaded_sr != cls._SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, loaded_sr, cls._SAMPLE_RATE)
                return waveform
            finally:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        raise RuntimeError("Silero model does not provide apply_tts or save_wav methods.")

    @classmethod
    def _harmonize_channels(cls, waveform: torch.Tensor, channels: int) -> torch.Tensor:
        if waveform.shape[0] == channels:
            return waveform
        if waveform.shape[0] == 1 and channels == 2:
            return waveform.repeat(2, 1)
        if waveform.shape[0] == 2 and channels == 1:
            return waveform[:1, :]
        return waveform[:1, :]

    @classmethod
    def _concat_chunks(cls, waveforms: list[torch.Tensor], chunk_pause_ms: int) -> torch.Tensor:
        if not waveforms:
            raise RuntimeError("Silero produced no audio chunks.")

        max_channels = max(wf.shape[0] for wf in waveforms)
        normalized = [cls._harmonize_channels(wf, max_channels).contiguous() for wf in waveforms]

        pause_samples = int(round((chunk_pause_ms / 1000.0) * cls._SAMPLE_RATE))
        if pause_samples <= 0 or len(normalized) == 1:
            return torch.cat(normalized, dim=1).contiguous()

        silence = torch.zeros((max_channels, pause_samples), dtype=torch.float32)
        pieces: list[torch.Tensor] = []
        for index, wf in enumerate(normalized):
            pieces.append(wf)
            if index < len(normalized) - 1:
                pieces.append(silence)
        return torch.cat(pieces, dim=1).contiguous()

    @classmethod
    def _synthesize_with_chunk_retry(
        cls,
        model,
        input_format: str,
        chunk_text: str,
        speaker: str,
        put_accent: bool,
        put_yo: bool,
        put_stress_homo: bool,
        put_yo_homo: bool,
        depth: int = 0,
    ) -> list[torch.Tensor]:
        try:
            waveform = cls._synthesize_single_chunk(
                model=model,
                input_format=input_format,
                text=chunk_text,
                speaker=speaker,
                put_accent=put_accent,
                put_yo=put_yo,
                put_stress_homo=put_stress_homo,
                put_yo_homo=put_yo_homo,
            )
            return [waveform]
        except Exception as exc:
            if depth >= 3 or not cls._is_text_too_long_error(exc):
                raise

            smaller_max = max(200, len(chunk_text) // 2)
            if input_format == "ssml":
                next_chunks = cls._split_ssml_into_chunks(chunk_text, smaller_max)
            else:
                next_chunks = cls._split_text_into_chunks(chunk_text, smaller_max)

            if len(next_chunks) <= 1:
                raise

            cls._log_warning(
                f"Chunk too long, splitting recursively (depth={depth + 1}, parts={len(next_chunks)})."
            )
            output_parts: list[torch.Tensor] = []
            for next_chunk in next_chunks:
                output_parts.extend(
                    cls._synthesize_with_chunk_retry(
                        model=model,
                        input_format=input_format,
                        chunk_text=next_chunk,
                        speaker=speaker,
                        put_accent=put_accent,
                        put_yo=put_yo,
                        put_stress_homo=put_stress_homo,
                        put_yo_homo=put_yo_homo,
                        depth=depth + 1,
                    )
                )
            return output_parts

    @classmethod
    def _synthesize(
        cls,
        model,
        input_format: str,
        text: str,
        speaker: str,
        put_accent: bool,
        put_yo: bool,
        put_stress_homo: bool,
        put_yo_homo: bool,
        enable_chunking: bool,
        max_chunk_chars: int,
        chunk_pause_ms: int,
        progress_bar: ProgressBar | None = None,
    ) -> torch.Tensor:
        text_chunks = cls._build_chunks(
            input_format=input_format,
            text=text,
            enable_chunking=enable_chunking,
            max_chunk_chars=max_chunk_chars,
        )
        if not text_chunks:
            raise RuntimeError("No text chunks to synthesize.")

        total_chunks = len(text_chunks)
        cls._log_info(f"Using {total_chunks} chunk(s) for synthesis.")

        synthesized_parts: list[torch.Tensor] = []
        for chunk_index, chunk in enumerate(text_chunks, start=1):
            synthesized_parts.extend(
                cls._synthesize_with_chunk_retry(
                    model=model,
                    input_format=input_format,
                    chunk_text=chunk,
                    speaker=speaker,
                    put_accent=put_accent,
                    put_yo=put_yo,
                    put_stress_homo=put_stress_homo,
                    put_yo_homo=put_yo_homo,
                )
            )
            if progress_bar is not None:
                progress_bar.update_absolute(chunk_index)

        return cls._concat_chunks(synthesized_parts, chunk_pause_ms=chunk_pause_ms)

    @classmethod
    def _normalize_generated_audio(cls, audio: Any) -> torch.Tensor:
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            audio = audio[0]

        if not isinstance(audio, torch.Tensor):
            try:
                audio = torch.as_tensor(audio)
            except Exception as exc:
                raise RuntimeError("Silero output cannot be converted to torch.Tensor.") from exc

        audio = audio.detach().float().cpu().contiguous()
        if audio.numel() == 0:
            raise RuntimeError("Silero output is empty.")

        if audio.ndim == 1:
            waveform = audio.unsqueeze(0)  # [T] -> [1, T]
        elif audio.ndim == 2:
            # Accept both [C, T] and [T, C].
            if audio.shape[0] in (1, 2):
                waveform = audio
            elif audio.shape[1] in (1, 2):
                waveform = audio.transpose(0, 1).contiguous()
            else:
                waveform = audio.reshape(1, -1)
        elif audio.ndim == 3:
            first = audio[0]
            if first.ndim == 2 and first.shape[0] not in (1, 2) and first.shape[1] in (1, 2):
                first = first.transpose(0, 1).contiguous()
            waveform = first if first.ndim == 2 else first.reshape(1, -1)
        else:
            waveform = audio.reshape(1, -1)

        if waveform.shape[0] > 2:
            waveform = waveform[:1, :]

        if waveform.shape[-1] <= 0:
            raise RuntimeError("Silero output has invalid waveform length.")

        return waveform.clamp(-1.0, 1.0).contiguous()

    @classmethod
    def execute(
        cls,
        text: str,
        input_format: str,
        speaker: str,
        run_device: str = "gpu",
        enable_chunking: bool = True,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
        chunk_pause_ms: int = _DEFAULT_CHUNK_PAUSE_MS,
        put_accent: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = True,
        put_yo_homo: bool = True,
    ) -> IO.NodeOutput:
        try:
            model_path = cls._model_path()
            cls._ensure_model_exists(model_path)
            device = cls._resolve_device(run_device)
            model = cls._load_model(model_path, device=device)
            progress_bar = ProgressBar(max(1, len(cls._build_chunks(
                input_format=input_format,
                text=text.strip(),
                enable_chunking=enable_chunking,
                max_chunk_chars=max_chunk_chars,
            ))))

            waveform = cls._synthesize(
                model=model,
                input_format=input_format,
                text=text.strip(),
                speaker=speaker,
                put_accent=put_accent,
                put_yo=put_yo,
                put_stress_homo=put_stress_homo,
                put_yo_homo=put_yo_homo,
                enable_chunking=enable_chunking,
                max_chunk_chars=max_chunk_chars,
                chunk_pause_ms=chunk_pause_ms,
                progress_bar=progress_bar,
            )

            audio_output = {
                "waveform": waveform.unsqueeze(0).contiguous(),
                "sample_rate": cls._SAMPLE_RATE,
            }
            cls._log_info(
                f"Generated audio: mode={input_format}, speaker={speaker}, device={device}, "
                f"samples={int(waveform.shape[-1])}, sample_rate={cls._SAMPLE_RATE}"
            )
            return IO.NodeOutput(audio_output)
        except Exception as exc:
            cls._log_warning(f"Execution fallback activated: {exc}")
            # Return short silence fallback to keep workflow execution stable.
            silence = torch.zeros((1, 1, 1024), dtype=torch.float32)
            return IO.NodeOutput({"waveform": silence, "sample_rate": cls._SAMPLE_RATE})


class TS_SileroStress(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_silero_stress")
    _LOG_PREFIX = "[TS SileroStress]"

    _RUN_DEVICES = ("cpu", "gpu")
    _MODEL_DIR_NAME = "silero-stress"
    _MODEL_FILE_NAME = "accentor.pt"
    _MODEL_PACKAGE = "silero_stress.data"
    _MODEL_PICKLE_PACKAGE = "accentor_models"
    _MODEL_PICKLE_NAME = "accentor"
    _COMBINING_ACUTE = "\u0301"
    _STRESS_MARKERS = ("unicode", "silero_plus")
    _STRESS_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_SileroStress",
            display_name="TS Silero Stress",
            category="TS/text",
            description=(
                "Automatic stress marks and yo restoration via silero-stress. "
                "Outputs Unicode combining acute accents."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Russian text for automatic stress and yo restoration.",
                ),
                IO.Combo.Input(
                    "run_device",
                    options=list(cls._RUN_DEVICES),
                    default="cpu",
                    advanced=True,
                    tooltip="Execution device for silero-stress.",
                ),
                IO.Combo.Input(
                    "stress_marker",
                    options=list(cls._STRESS_MARKERS),
                    default="unicode",
                    tooltip="Stress mark output format: Unicode combining acute or native Silero plus sign.",
                ),
                IO.Boolean.Input(
                    "use_accentor",
                    default=True,
                    tooltip="Run the common accentor for non-homograph stress and yo placement.",
                ),
                IO.Boolean.Input(
                    "use_homosolver",
                    default=True,
                    tooltip="Run the homograph disambiguation model.",
                ),
                IO.Boolean.Input(
                    "put_stress",
                    default=True,
                    tooltip="Place stress marks in non-homograph words.",
                ),
                IO.Boolean.Input(
                    "put_yo",
                    default=True,
                    tooltip="Restore letter yo in non-homograph words where needed.",
                ),
                IO.Boolean.Input(
                    "put_stress_homo",
                    default=True,
                    tooltip="Place stress marks in homographs.",
                ),
                IO.Boolean.Input(
                    "put_yo_homo",
                    default=True,
                    tooltip="Restore letter yo in homographs where needed.",
                ),
                IO.Boolean.Input(
                    "stress_single_vowel",
                    default=True,
                    tooltip="Place stress marks even in words with a single vowel.",
                ),
                IO.String.Input(
                    "words_to_ignore",
                    default="",
                    multiline=True,
                    advanced=True,
                    tooltip="Comma or newline separated words that should be skipped completely.",
                ),
            ],
            outputs=[IO.String.Output(display_name="text")],
            search_aliases=["silero stress", "stress", "yo", "accentor", "homograph"],
        )

    @classmethod
    def validate_inputs(cls, text, run_device, stress_marker, **kwargs) -> bool | str:
        if not isinstance(text, str):
            return "Text must be a string."
        if run_device not in cls._RUN_DEVICES:
            return f"Unsupported run_device '{run_device}'."
        if stress_marker not in cls._STRESS_MARKERS:
            return f"Unsupported stress_marker '{stress_marker}'."
        return True

    @classmethod
    def _stress_log_info(cls, message: str) -> None:
        cls._LOGGER.info("%s %s", cls._LOG_PREFIX, message)

    @classmethod
    def _stress_log_warning(cls, message: str) -> None:
        cls._LOGGER.warning("%s %s", cls._LOG_PREFIX, message)

    @classmethod
    def _resolve_stress_device(cls, run_device: str) -> str:
        if run_device == "cpu":
            return "cpu"

        target_device = comfy.model_management.get_torch_device()
        if target_device.type == "cpu":
            cls._stress_log_warning("GPU requested but not available. Falling back to CPU.")
            return "cpu"

        return str(target_device)

    @classmethod
    def _load_accentor_runtime(cls, device_name: str):
        try:
            importlib.import_module("silero_stress")
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'silero_stress'. Install package 'silero-stress' to enable TS Silero Stress."
            ) from exc

        model_path = cls._ensure_stress_model_exists()
        accentor = torch.package.PackageImporter(model_path).load_pickle(
            cls._MODEL_PICKLE_PACKAGE,
            cls._MODEL_PICKLE_NAME,
        )
        cls._restore_stress_weights(accentor)
        if hasattr(accentor, "to"):
            accentor.to(device=device_name)
        return accentor

    @classmethod
    def _stress_model_path(cls) -> str:
        model_dir = os.path.join(folder_paths.models_dir, cls._MODEL_DIR_NAME)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, cls._MODEL_FILE_NAME)

    @classmethod
    def _ensure_stress_model_exists(cls) -> str:
        model_path = cls._stress_model_path()
        if os.path.isfile(model_path):
            return model_path

        try:
            try:
                import importlib_resources as impresources
            except ImportError:
                from importlib import resources as impresources

            package_file = impresources.files(cls._MODEL_PACKAGE).joinpath(cls._MODEL_FILE_NAME)
            with impresources.as_file(package_file) as source_path:
                shutil.copyfile(str(source_path), model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare Silero Stress model at '{model_path}'."
            ) from exc

        cls._stress_log_info(f"Prepared model in ComfyUI models directory: {model_path}")
        return model_path

    @classmethod
    def _restore_stress_weights(cls, accentor) -> None:
        quantized_weight = accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data.clone()
        restored_weights = accentor.homosolver.model.bert.scale * (
            quantized_weight - accentor.homosolver.model.bert.zero_point
        )
        accentor.homosolver.model.bert.embeddings.word_embeddings.weight.data = restored_weights

    @classmethod
    def _parse_words_to_ignore(cls, words_to_ignore: str) -> list[str]:
        if not isinstance(words_to_ignore, str) or not words_to_ignore.strip():
            return []
        parts = re.split(r"[,;\r\n]+", words_to_ignore)
        return [part.strip() for part in parts if part.strip()]

    @classmethod
    def _filter_callable_kwargs(cls, callable_obj, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs

        params = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return kwargs

        return {key: value for key, value in kwargs.items() if key in params}

    @classmethod
    def _invoke_stress_processor(cls, processor, text: str, **kwargs) -> str:
        filtered_kwargs = cls._filter_callable_kwargs(processor, kwargs)
        result = processor(text, **filtered_kwargs)
        if not isinstance(result, str):
            raise RuntimeError("silero-stress returned non-string output.")
        return result

    @classmethod
    def _convert_stress_marks_to_unicode(cls, text: str) -> str:
        output: list[str] = []
        pending_stress = False

        for char in text:
            if char == "+":
                if pending_stress:
                    output.append("+")
                pending_stress = True
                continue

            if pending_stress:
                if char in cls._STRESS_VOWELS:
                    output.append(char)
                    output.append(cls._COMBINING_ACUTE)
                else:
                    output.append("+")
                    output.append(char)
                pending_stress = False
                continue

            output.append(char)

        if pending_stress:
            output.append("+")

        return "".join(output)

    @classmethod
    def execute(
        cls,
        text: str,
        run_device: str = "cpu",
        stress_marker: str = "unicode",
        use_accentor: bool = True,
        use_homosolver: bool = True,
        put_stress: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = True,
        put_yo_homo: bool = True,
        stress_single_vowel: bool = True,
        words_to_ignore: str = "",
    ) -> IO.NodeOutput:
        normalized_text = text if isinstance(text, str) else ""
        if not normalized_text.strip():
            return IO.NodeOutput("")

        try:
            if not use_accentor and not use_homosolver:
                return IO.NodeOutput(normalized_text)

            accentor = cls._load_accentor_runtime(cls._resolve_stress_device(run_device))
            ignore_words = cls._parse_words_to_ignore(words_to_ignore)

            common_kwargs = {}
            if ignore_words:
                common_kwargs["words_to_ignore"] = ignore_words

            if use_accentor and use_homosolver:
                stressed_text = cls._invoke_stress_processor(
                    accentor,
                    normalized_text,
                    put_stress=put_stress,
                    put_yo=put_yo,
                    put_stress_homo=put_stress_homo,
                    put_yo_homo=put_yo_homo,
                    stress_single_vowel=stress_single_vowel,
                    **common_kwargs,
                )
            elif use_accentor:
                accentor_processor = getattr(accentor, "accentor", None)
                if accentor_processor is None or not callable(accentor_processor):
                    raise RuntimeError("silero-stress accentor processor is not available.")
                stressed_text = cls._invoke_stress_processor(
                    accentor_processor,
                    normalized_text,
                    put_stress=put_stress,
                    put_yo=put_yo,
                    stress_single_vowel=stress_single_vowel,
                    **common_kwargs,
                )
            else:
                homosolver_processor = getattr(accentor, "homosolver", None)
                if homosolver_processor is None or not callable(homosolver_processor):
                    raise RuntimeError("silero-stress homosolver processor is not available.")
                stressed_text = cls._invoke_stress_processor(
                    homosolver_processor,
                    normalized_text,
                    put_stress_homo=put_stress_homo,
                    put_yo_homo=put_yo_homo,
                    **common_kwargs,
                )

            output_text = (
                stressed_text
                if stress_marker == "silero_plus"
                else cls._convert_stress_marks_to_unicode(stressed_text)
            )
            cls._stress_log_info(
                f"Processed text: device={run_device}, stress_marker={stress_marker}, use_accentor={use_accentor}, "
                f"use_homosolver={use_homosolver}, input_length={len(normalized_text)}, "
                f"output_length={len(output_text)}"
            )
            return IO.NodeOutput(output_text)
        except Exception as exc:
            cls._stress_log_warning(f"Execution fallback activated: {exc}")
            return IO.NodeOutput(normalized_text)


NODE_CLASS_MAPPINGS = {
    "TS_SileroTTS": TS_SileroTTS,
    "TS_SileroStress": TS_SileroStress,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_SileroTTS": "TS Silero TTS",
    "TS_SileroStress": "TS Silero Stress",
}

