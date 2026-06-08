"""TS Whisper — transcription/translation on the shared native OpenAI Whisper engine.

node_id: TSWhisper

Uses the unified engine in ``nodes/_whisper_engine.py`` (native ``openai-whisper``)
so model weights and the in-memory model cache are shared with the TS Super
Prompt voice feature — load ``large-v3`` once and both nodes reuse it.

Native ``model.transcribe()`` returns ``{text, segments:[{start,end,text,words}]}``
and handles long-form audio internally (30 s windows, sequential decoding), so
the previous transformers ASR pipeline + manual chunking are gone. SRT / TTML /
plain-text are built from the returned segments (word-level when requested).
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import xml.etree.ElementTree as ET

import folder_paths
import srt

from comfy_api.v0_0_2 import IO

from .. import _whisper_engine as engine

_LOG_PREFIX = "[TS Whisper]"
_logger = logging.getLogger("comfyui_ts_whisper")


def _log_info(message: str) -> None:
    _logger.info("%s %s", _LOG_PREFIX, message)


def _log_warning(message: str) -> None:
    _logger.warning("%s %s", _LOG_PREFIX, message)


def _log_error(message: str) -> None:
    _logger.error("%s %s", _LOG_PREFIX, message)


# Friendly model labels mapped to native whisper names (shared with the engine
# and Super Prompt voice → shared weights + cache).
MODEL_OPTIONS = ["large-v3", "turbo"]


class TSWhisper(IO.ComfyNode):
    """Native-Whisper transcription node. The class stays effectively stateless;
    model weights/objects live in the shared engine cache."""

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _safe_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_temperature(temperature, fallbacks):
        """Return a temperature schedule for whisper: a tuple of floats when a
        comma-separated fallback ladder is supplied, otherwise a single float."""
        if isinstance(fallbacks, str) and fallbacks.strip():
            raw = fallbacks
            for ch in (";", "|", "\n", "\t", " "):
                raw = raw.replace(ch, ",")
            temps = []
            for part in (p.strip() for p in raw.split(",")):
                if not part:
                    continue
                try:
                    temps.append(float(part))
                except ValueError:
                    continue
            if len(temps) > 1:
                return tuple(temps)
            if len(temps) == 1:
                return temps[0]
        try:
            return float(temperature)
        except (TypeError, ValueError):
            return 0.0

    # ---- banned-phrase filtering (Russian subtitle-spam hallucinations) -----
    @staticmethod
    def _normalize_text(text):
        if text is None:
            return ""
        lowered = str(text).lower().replace("ё", "е")
        cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _tokenize_text(cls, text):
        normalized = cls._normalize_text(text)
        return normalized.split() if normalized else []

    @classmethod
    def _find_phrase_indices(cls, text_list, phrase_tokens_list):
        if not text_list or not phrase_tokens_list:
            return set()
        tokens_with_idx = []
        for idx, text in enumerate(text_list):
            for tok in cls._tokenize_text(text):
                tokens_with_idx.append((tok, idx))

        remove_indices = set()
        total = len(tokens_with_idx)
        for phrase_tokens in phrase_tokens_list:
            if not phrase_tokens:
                continue
            plen = len(phrase_tokens)
            if plen > total:
                continue
            for i in range(0, total - plen + 1):
                if all(tokens_with_idx[i + j][0] == phrase_tokens[j] for j in range(plen)):
                    for j in range(plen):
                        remove_indices.add(tokens_with_idx[i + j][1])
        return remove_indices

    @classmethod
    def _remove_unwanted_phrases(cls, segments, text_segments):
        banned_phrases = [
            "субтитры создавал dimatorzok",
            "Субтитры делал DimaTorzok",
            "редактор субтитров а.семкин корректор а.егорова",
        ]
        phrase_tokens_list = [cls._tokenize_text(p) for p in banned_phrases if p]

        filtered_segments = segments
        filtered_text_segments = text_segments

        if segments:
            remove = cls._find_phrase_indices([s.get("text", "") for s in segments], phrase_tokens_list)
            if remove:
                filtered_segments = [s for i, s in enumerate(segments) if i not in remove]
        if text_segments:
            remove = cls._find_phrase_indices(text_segments, phrase_tokens_list)
            if remove:
                filtered_text_segments = [t for i, t in enumerate(text_segments) if i not in remove]
        return filtered_segments, filtered_text_segments

    # ---- TTML ---------------------------------------------------------------
    @staticmethod
    def _seconds_to_ttml_time(seconds_float):
        td = datetime.timedelta(seconds=seconds_float)
        total = int(td.total_seconds())
        ms = int(td.microseconds / 1000)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}.{ms:03d}"

    @classmethod
    def _create_ttml_content(cls, subtitles, language_code):
        root = ET.Element(
            "tt",
            attrib={
                "xmlns": "http://www.w3.org/ns/ttml",
                "xmlns:tts": "http://www.w3.org/ns/ttml#styling",
                "xml:lang": language_code or "und",
            },
        )
        div = ET.SubElement(ET.SubElement(root, "body"), "div")
        for sub in subtitles:
            p = ET.SubElement(div, "p")
            p.set("begin", cls._seconds_to_ttml_time(sub.start.total_seconds()))
            p.set("end", cls._seconds_to_ttml_time(sub.end.total_seconds()))
            p.text = sub.content
        try:
            return ET.tostring(root, encoding="unicode")
        except Exception as exc:  # noqa: BLE001
            _log_error(f"TTML generation failed: {exc}")
            return ""

    # ----------------------------------------------------------------- schema
    @classmethod
    def define_schema(cls) -> IO.Schema:
        default_output_dir = (
            folder_paths.get_output_directory() if hasattr(folder_paths, "get_output_directory") else ""
        )
        tt = {
            "audio": "Входной аудиотензор ComfyUI (waveform + sample_rate).",
            "model": "Модель Whisper: large-v3 (Whisper 3, лучшее качество) или turbo (Whisper 3 Turbo, быстрее). Веса и кэш общие с TS Super Prompt.",
            "task": "transcribe = распознавание на исходном языке; translate_to_english = перевод в английский (turbo не умеет переводить — будет транскрипция).",
            "source_language": "Язык входного аудио. auto = автоопределение. Для русского — ru.",
            "timestamps": "segment = таймкоды по сегментам; word = по словам; none = без таймкодов (SRT/TTML отключаются).",
            "precision": "fp16 быстрее на GPU; fp32 стабильнее. На CPU всегда fp32.",
            "beam_size": "Beam search (используется при temperature=0). 1 = greedy. Больше = точнее/медленнее.",
            "temperature": "Температура декодирования. 0 = детерминированно.",
            "temperature_fallbacks": "Лестница температур через запятую (напр. 0.0,0.2,0.4,0.6,0.8,1.0). Перекрывает temperature.",
            "condition_on_previous_text": "Контекст из предыдущих сегментов (связнее, но иногда усиливает дрейф).",
            "compression_ratio_threshold": "Порог zlib-сжатия для детектора галлюцинаций (whisper default 2.4).",
            "logprob_threshold": "Порог среднего лог-проба (whisper default -1.0).",
            "no_speech_threshold": "Порог отсутствия речи (whisper default 0.6).",
            "initial_prompt": "Начальная подсказка (лексика/имена/стиль).",
            "save_srt_file": "Сохранять SRT/TTML в папку output/subtitles.",
            "output_filename_prefix": "Префикс имени файла для SRT/TTML.",
            "output_dir": "Папка вывода для SRT/TTML. Пусто = стандартная output.",
        }
        languages = ["auto", "en", "ru", "fr", "de", "es", "it", "ja", "ko", "zh", "uk", "pl"]
        return IO.Schema(
            node_id="TSWhisper",
            display_name="TS Whisper",
            category="TS/Audio",
            inputs=[
                IO.Audio.Input("audio", tooltip=tt["audio"]),
                IO.Combo.Input("model", options=MODEL_OPTIONS, default="large-v3", tooltip=tt["model"]),
                IO.Combo.Input("task", options=["transcribe", "translate_to_english"], default="transcribe", tooltip=tt["task"]),
                IO.Combo.Input("source_language", options=languages, default="ru", tooltip=tt["source_language"]),
                IO.Combo.Input("timestamps", options=["segment", "word", "none"], default="segment", tooltip=tt["timestamps"]),
                IO.Combo.Input("precision", options=["fp16", "fp32"], default="fp16", tooltip=tt["precision"]),
                IO.Int.Input("beam_size", default=5, min=1, max=10, tooltip=tt["beam_size"]),
                IO.Float.Input("temperature", default=0.0, min=0.0, max=1.0, step=0.1, tooltip=tt["temperature"]),
                IO.String.Input("temperature_fallbacks", default="0.0,0.2,0.4,0.6,0.8,1.0", multiline=False, tooltip=tt["temperature_fallbacks"]),
                IO.Boolean.Input("condition_on_previous_text", default=True, tooltip=tt["condition_on_previous_text"]),
                IO.Float.Input("compression_ratio_threshold", default=2.4, min=0.0, max=10.0, step=0.1, tooltip=tt["compression_ratio_threshold"]),
                IO.Float.Input("logprob_threshold", default=-1.0, min=-10.0, max=0.0, step=0.1, tooltip=tt["logprob_threshold"]),
                IO.Float.Input("no_speech_threshold", default=0.6, min=0.0, max=1.0, step=0.05, tooltip=tt["no_speech_threshold"]),
                IO.String.Input("initial_prompt", default="", multiline=True, tooltip=tt["initial_prompt"]),
                IO.Boolean.Input("save_srt_file", default=True, label_on="Yes (Save SRT & TTML)", label_off="No", tooltip=tt["save_srt_file"]),
                IO.String.Input("output_filename_prefix", default="transcribed_audio", tooltip=tt["output_filename_prefix"]),
                IO.String.Input("output_dir", default=default_output_dir, multiline=False, optional=True, tooltip=tt["output_dir"]),
            ],
            outputs=[
                IO.String.Output(display_name="srt_content"),
                IO.String.Output(display_name="text_content"),
                IO.String.Output(display_name="ttml_content"),
            ],
        )

    # ----------------------------------------------------------------- execute
    @classmethod
    def execute(
        cls,
        audio,
        model,
        task,
        source_language,
        timestamps,
        precision,
        beam_size,
        temperature,
        temperature_fallbacks,
        condition_on_previous_text,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        initial_prompt,
        save_srt_file,
        output_filename_prefix,
        output_dir=None,
    ) -> IO.NodeOutput:
        model = model if model in MODEL_OPTIONS else "large-v3"
        task = task or "transcribe"
        source_language = source_language or "auto"
        timestamps = timestamps if timestamps in ("segment", "word", "none") else "segment"
        precision = precision if precision in ("fp16", "fp32") else "fp16"
        beam_size = cls._safe_int(beam_size, 5)
        compression_ratio_threshold = cls._safe_float(compression_ratio_threshold, 2.4)
        logprob_threshold = cls._safe_float(logprob_threshold, -1.0)
        no_speech_threshold = cls._safe_float(no_speech_threshold, 0.6)
        initial_prompt = initial_prompt or ""
        output_filename_prefix = output_filename_prefix or "transcribed_audio"

        # 1) Decode ComfyUI audio -> 16 kHz mono float32 (shared engine helper).
        try:
            audio_np = engine.comfy_audio_to_mono16k(audio, normalize=True, device="auto", resample_quality="high")
        except Exception as exc:  # noqa: BLE001
            _log_error(f"Audio preprocessing failed: {exc}")
            return IO.NodeOutput("", "", "")
        if audio_np.size == 0:
            _log_warning("Prepared audio is empty.")
            return IO.NodeOutput("", "", "")

        # 2) Task + translate guard (turbo has no translation head).
        whisper_task = "translate" if task == "translate_to_english" else "transcribe"
        if whisper_task == "translate" and not engine.supports_translate(model):
            _log_warning(f"Model '{model}' cannot translate; falling back to transcribe.")
            whisper_task = "transcribe"

        # 3) Load model via the shared engine (shared weights + in-memory cache).
        try:
            torch, _ = engine.load_runtime()
            whisper_model, device, use_fp16 = engine.load_model(
                model, device="auto", fp16_pref=(precision == "fp16")
            )
        except Exception as exc:  # noqa: BLE001
            _log_error(f"Failed to load Whisper model '{model}': {exc}")
            return IO.NodeOutput(f"Error: {exc}", "", "")

        language = None if source_language in (None, "", "auto") else source_language
        kwargs = {
            "task": whisper_task,
            "language": language,
            "fp16": use_fp16,
            "temperature": cls._parse_temperature(temperature, temperature_fallbacks),
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": bool(condition_on_previous_text),
            "word_timestamps": timestamps == "word",
            "verbose": False,
        }
        if beam_size and beam_size > 1:
            kwargs["beam_size"] = int(beam_size)
        if initial_prompt.strip():
            kwargs["initial_prompt"] = initial_prompt

        _log_info(f"Transcribe: model={model} task={whisper_task} lang={source_language} device={device} timestamps={timestamps}")

        # 4) Run native transcription (handles long-form internally).
        try:
            with torch.inference_mode():
                result = whisper_model.transcribe(audio_np, **kwargs)
        except Exception as exc:  # noqa: BLE001
            cls._logger_error(exc)
            return IO.NodeOutput(f"Error: {exc}", "", "")

        # 5) Build segments / text from the result.
        raw_segments = result.get("segments") or []
        text_segments = [str(s.get("text", "")).strip() for s in raw_segments if str(s.get("text", "")).strip()]

        timed_segments = []
        if timestamps == "word":
            for seg in raw_segments:
                for w in seg.get("words") or []:
                    word = str(w.get("word", "")).strip()
                    if word and w.get("start") is not None and w.get("end") is not None:
                        timed_segments.append({"start": float(w["start"]), "end": float(w["end"]), "text": word})
        elif timestamps == "segment":
            for seg in raw_segments:
                text = str(seg.get("text", "")).strip()
                if text and seg.get("start") is not None and seg.get("end") is not None:
                    timed_segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})

        timed_segments, text_segments = cls._remove_unwanted_phrases(timed_segments, text_segments)

        # 6) Compose SRT / TTML / plain text.
        subtitles = []
        for idx, seg in enumerate(timed_segments, start=1):
            end = max(seg["end"], seg["start"])
            subtitles.append(
                srt.Subtitle(
                    index=idx,
                    start=datetime.timedelta(seconds=seg["start"]),
                    end=datetime.timedelta(seconds=end),
                    content=seg["text"],
                )
            )
        generated_srt = srt.compose(subtitles, reindex=True) if subtitles else ""
        ttml_lang = "en" if whisper_task == "translate" else (
            source_language if source_language != "auto" else (result.get("language") or "und")
        )
        generated_ttml = cls._create_ttml_content(subtitles, ttml_lang) if subtitles else ""
        plain_text = (str(result.get("text") or "").strip()) or " ".join(text_segments).strip()

        # 7) Optionally persist SRT/TTML.
        if save_srt_file and subtitles:
            base_dir = output_dir or (
                folder_paths.get_output_directory() if hasattr(folder_paths, "get_output_directory") else None
            )
            if base_dir and os.path.isdir(base_dir):
                try:
                    subtitles_dir = os.path.join(base_dir, "subtitles")
                    os.makedirs(subtitles_dir, exist_ok=True)
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    srt_path = os.path.join(subtitles_dir, f"{output_filename_prefix}_{stamp}.srt")
                    ttml_path = os.path.join(subtitles_dir, f"{output_filename_prefix}_{stamp}.ttml")
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(generated_srt)
                    with open(ttml_path, "w", encoding="utf-8") as f:
                        f.write(generated_ttml)
                    _log_info(f"Saved SRT: {srt_path}")
                except Exception as exc:  # noqa: BLE001
                    _log_error(f"Failed to save SRT/TTML: {exc}")

        _log_info(f"Done. segments={len(subtitles)} text_chars={len(plain_text)}")
        return IO.NodeOutput(generated_srt, plain_text, generated_ttml)

    @classmethod
    def _logger_error(cls, exc):
        _logger.error("%s Execution error: %s", _LOG_PREFIX, exc, exc_info=True)


NODE_CLASS_MAPPINGS = {"TSWhisper": TSWhisper}
NODE_DISPLAY_NAME_MAPPINGS = {"TSWhisper": "TS Whisper"}
