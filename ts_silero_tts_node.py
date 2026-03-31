import logging
import os
import re
from typing import Any

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyAPI, IO


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
    async def _synthesize(
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
        api: ComfyAPI | None = None,
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
        if api is not None:
            await api.execution.set_progress(value=0, max_value=total_chunks)

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
            if api is not None:
                await api.execution.set_progress(value=chunk_index, max_value=total_chunks)

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
    async def execute(
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
        model_path = cls._model_path()
        cls._ensure_model_exists(model_path)
        device = cls._resolve_device(run_device)
        model = cls._load_model(model_path, device=device)
        api = ComfyAPI()

        waveform = await cls._synthesize(
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
            api=api,
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


NODE_CLASS_MAPPINGS = {
    "TS_SileroTTS": TS_SileroTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_SileroTTS": "TS Silero TTS",
}
