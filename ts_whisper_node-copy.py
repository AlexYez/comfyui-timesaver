import os
import datetime
import logging
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torchaudio.transforms import Resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import comfy.model_management
import comfy.utils
import folder_paths
import srt


class TSWhisper:
    _LOG_NAME = "comfyui_ts_whisper"
    _LOG_PREFIX = "[TS Whisper]"
    _COLOR_CYAN = "\x1b[36m"
    _COLOR_GREEN = "\x1b[32m"
    _COLOR_YELLOW = "\x1b[33m"
    _COLOR_RED = "\x1b[31m"
    _COLOR_RESET = "\x1b[0m"

    def __init__(self):
        self.logger = self._init_logger()
        self.target_device = comfy.model_management.get_torch_device()
        self.target_device_type = self.target_device.type
        self.model = None
        self.processor = None
        self.pipeline = None
        self.hf_cache_dir = None
        self._resamplers = {}
        self.current_pipeline_config = {
            "model_name": None,
            "precision": None,
            "attn_implementation": None,
        }

        try:
            base_path = (
                folder_paths.base_path
                if hasattr(folder_paths, "base_path") and folder_paths.base_path
                else os.getcwd()
            )
            comfy_models_dir = os.path.join(base_path, "models")
            if not hasattr(folder_paths, "base_path") or not folder_paths.base_path:
                self._log_warning(
                    f"folder_paths.base_path not found, using {comfy_models_dir} for models."
                )

            cache_dir = os.path.join(comfy_models_dir, "whisper")
            os.makedirs(cache_dir, exist_ok=True)
            self.hf_cache_dir = cache_dir
            self._log_info(f"Hugging Face Whisper cache dir: {self.hf_cache_dir}")
        except Exception as e:
            self._log_warning(
                f"Failed to set Hugging Face cache dir ({self.hf_cache_dir}). Using default cache. Error: {e}"
            )
            self.hf_cache_dir = None

        self._log_info(f"Initialized. Device: {self.target_device}")

    def _init_logger(self):
        logger = logging.getLogger(self._LOG_NAME)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _log_info(self, message):
        self.logger.info(f"{self._LOG_PREFIX} {message}")

    def _log_warning(self, message):
        self.logger.warning(f"{self._LOG_PREFIX} {message}")

    def _log_error(self, message):
        self.logger.error(f"{self._LOG_PREFIX} {message}")

    def _log_tensor_shape(self, label, tensor, color=None):
        if not isinstance(tensor, torch.Tensor):
            return
        color = color or self._COLOR_CYAN
        shape = tuple(tensor.shape)
        self._log_info(
            f"{color}{label} shape={shape} dtype={tensor.dtype} device={tensor.device}{self._COLOR_RESET}"
        )

    def _get_resampler(self, orig_freq, new_freq, device, quality):
        if quality not in ("fast", "balanced", "high"):
            quality = "balanced"

        key = (int(orig_freq), int(new_freq), device.type, quality)
        cached = self._resamplers.get(key)
        if cached is not None:
            return cached

        if quality == "fast":
            lowpass_filter_width = 4
            rolloff = 0.95
        elif quality == "high":
            lowpass_filter_width = 8
            rolloff = 0.99
        else:
            lowpass_filter_width = 6
            rolloff = 0.99

        resampler = Resample(
            orig_freq=int(orig_freq),
            new_freq=int(new_freq),
            resampling_method="sinc_interp_hann",
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            dtype=torch.float32,
        )
        if device.type != "cpu":
            resampler = resampler.to(device)

        self._resamplers[key] = resampler
        return resampler

    def _safe_float(self, value, default):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, value, default):
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _is_oom_error(self, exc):
        if hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError):
            return True
        if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
            exc, torch.cuda.OutOfMemoryError
        ):
            return True
        message = str(exc).lower()
        return "out of memory" in message or "allocation" in message

    def _parse_temperature_fallbacks(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            temps = []
            for item in value:
                try:
                    temps.append(float(item))
                except (TypeError, ValueError):
                    continue
            if not temps:
                return None
            return temps if len(temps) > 1 else temps[0]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            for ch in [";", "|", "\n", "\t", " "]:
                raw = raw.replace(ch, ",")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            temps = []
            for part in parts:
                try:
                    temps.append(float(part))
                except (TypeError, ValueError):
                    continue
            if not temps:
                return None
            return temps if len(temps) > 1 else temps[0]
        return None

    def _collect_output_segments(self, output_chunk, start_offset_s, all_segments, all_text_segments):
        if isinstance(output_chunk, dict) and "chunks" in output_chunk:
            for seg in output_chunk["chunks"]:
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                all_text_segments.append(text)

                ts = seg.get("timestamp")
                if ts and isinstance(ts, (tuple, list)) and len(ts) == 2:
                    s_local, e_local = ts
                    if s_local is not None and e_local is not None:
                        abs_start = start_offset_s + float(s_local)
                        abs_end = start_offset_s + float(e_local)
                        if abs_end < abs_start:
                            abs_end = abs_start
                        all_segments.append(
                            {"start": abs_start, "end": abs_end, "text": text}
                        )
        elif isinstance(output_chunk, dict) and "text" in output_chunk:
            text = str(output_chunk.get("text", "")).strip()
            if text:
                all_text_segments.append(text)
        elif isinstance(output_chunk, str):
            text = output_chunk.strip()
            if text:
                all_text_segments.append(text)

    def _run_chunked_inference(
        self,
        pipe,
        prepared_audio_numpy,
        target_sample_rate,
        return_timestamps,
        generate_kwargs,
        manual_chunk_length_s,
        manual_chunk_overlap_s,
    ):
        all_segments = []
        all_text_segments = []

        total_duration_samples = len(prepared_audio_numpy)
        chunk_length_samples = max(1, int(manual_chunk_length_s * target_sample_rate))
        overlap_samples = max(0, int(manual_chunk_overlap_s * target_sample_rate))
        if overlap_samples >= chunk_length_samples:
            overlap_samples = max(0, chunk_length_samples - 1)

        step_samples = max(1, chunk_length_samples - overlap_samples)
        num_chunks = max(
            1, (total_duration_samples - overlap_samples + step_samples - 1) // step_samples
        )
        current_sample_offset = 0
        pbar = comfy.utils.ProgressBar(num_chunks)

        for _ in range(num_chunks):
            start_sample = current_sample_offset
            end_sample = min(start_sample + chunk_length_samples, total_duration_samples)
            chunk_audio = prepared_audio_numpy[start_sample:end_sample]

            if len(chunk_audio) > 0:
                try:
                    with torch.inference_mode():
                        output_chunk = pipe(
                            chunk_audio,
                            return_timestamps=return_timestamps,
                            generate_kwargs=generate_kwargs,
                        )
                except Exception:
                    self._log_error("Pipeline execution failed inside chunk loop.")
                    self.logger.error(
                        f"{self._LOG_PREFIX} generate_kwargs={generate_kwargs}",
                        exc_info=True,
                    )
                    raise

                self._collect_output_segments(
                    output_chunk,
                    start_sample / target_sample_rate,
                    all_segments,
                    all_text_segments,
                )

            current_sample_offset += step_samples
            pbar.update(1)
            if end_sample >= total_duration_samples:
                break

        merged_segments = []
        if all_segments:
            overlap_tolerance_s = max(0.05, manual_chunk_overlap_s * 0.2)
            merged_segments = self._merge_segments(all_segments, overlap_tolerance_s)

        return merged_segments, all_text_segments

    def _should_reload_pipeline(self, new_config):
        if self.pipeline is None:
            return True

        for key, value in new_config.items():
            if self.current_pipeline_config.get(key) != value:
                self._log_info(
                    f"Pipeline config changed: {key} ('{self.current_pipeline_config.get(key)}' -> '{value}')"
                )
                return True
        return False

    def load_model_and_processor(self, model_name, precision, attn_implementation_choice):
        try:
            self._log_info(
                f"Loading model: {model_name} (precision={precision}, attn={attn_implementation_choice})"
            )

            actual_torch_dtype = torch.float32
            if self.target_device_type == "cuda":
                if precision == "fp16":
                    actual_torch_dtype = torch.float16
                elif precision == "bf16":
                    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        actual_torch_dtype = torch.bfloat16
                    else:
                        self._log_warning("bf16 not supported, falling back to fp16.")
                        actual_torch_dtype = torch.float16
            else:
                if precision != "fp32":
                    self._log_warning("Non-fp32 precision requested on CPU; using fp32.")

            model_load_kwargs = {
                "torch_dtype": actual_torch_dtype,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            }
            if self.hf_cache_dir:
                model_load_kwargs["cache_dir"] = self.hf_cache_dir

            if attn_implementation_choice == "sdpa" and hasattr(
                torch.nn.functional, "scaled_dot_product_attention"
            ):
                model_load_kwargs["attn_implementation"] = "sdpa"
                self._log_info("Using PyTorch SDPA (Scaled Dot Product Attention).")
            elif attn_implementation_choice == "sdpa":
                self._log_warning("SDPA not available. Using eager attention.")

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, **model_load_kwargs
            )
            if self.model.device != self.target_device:
                self.model.to(self.target_device)

            self._log_info(f"Model loaded on {self.model.device}")

            processor_load_kwargs = {"cache_dir": self.hf_cache_dir} if self.hf_cache_dir else {}
            self.processor = AutoProcessor.from_pretrained(model_name, **processor_load_kwargs)

            return self.model, self.processor, actual_torch_dtype

        except Exception as e:
            self._log_error(f"Failed to load model/processor: {e}")
            self.model, self.processor, self.pipeline = None, None, None
            raise

    def get_pipeline(self, model_name, precision, attn_implementation):
        new_config = {
            "model_name": model_name,
            "precision": precision,
            "attn_implementation": attn_implementation,
        }

        if self._should_reload_pipeline(new_config):
            self._log_info("(Re)loading ASR pipeline...")
            self.pipeline = None
            model, processor, actual_torch_dtype = self.load_model_and_processor(
                model_name, precision, attn_implementation
            )

            pipeline_kwargs = {
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "torch_dtype": actual_torch_dtype,
                "device": self.target_device,
            }
            self.pipeline = pipeline("automatic-speech-recognition", **pipeline_kwargs)
            self.current_pipeline_config = new_config
            self._log_info(f"ASR pipeline ready: {model_name}")

        return self.pipeline

    def _seconds_to_ttml_time(self, seconds_float):
        td = datetime.timedelta(seconds=seconds_float)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _create_ttml_content(self, subtitles, language_code):
        ttml_lang = language_code or "und"
        root = ET.Element(
            "tt",
            attrib={
                "xmlns": "http://www.w3.org/ns/ttml",
                "xmlns:tts": "http://www.w3.org/ns/ttml#styling",
                "xml:lang": ttml_lang,
            },
        )
        body = ET.SubElement(root, "body")
        div = ET.SubElement(body, "div")

        for sub in subtitles:
            p = ET.SubElement(div, "p")
            start_seconds = sub.start.total_seconds()
            end_seconds = sub.end.total_seconds()
            p.set("begin", self._seconds_to_ttml_time(start_seconds))
            p.set("end", self._seconds_to_ttml_time(end_seconds))
            p.text = sub.content

        try:
            return ET.tostring(root, encoding="unicode")
        except Exception as e:
            self._log_error(f"TTML generation failed: {e}")
            return ""

    def _prepare_audio(
        self,
        audio,
        target_sample_rate,
        normalize_audio,
        preprocess_device,
        resample_quality,
    ):
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Invalid audio input format.")

        waveform_tensor = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if isinstance(waveform_tensor, list):
            waveform_tensor = waveform_tensor[0]

        if not isinstance(waveform_tensor, torch.Tensor):
            raise ValueError("Audio waveform is not a torch.Tensor.")

        self._log_tensor_shape("Input waveform", waveform_tensor, color=self._COLOR_GREEN)

        current_waveform = waveform_tensor.detach()
        if current_waveform.ndim == 3:
            if current_waveform.shape[0] > 1:
                self._log_warning("Batch size > 1 detected; using first item.")
            current_waveform = current_waveform[0]

        if current_waveform.ndim == 2:
            channels_first = (
                current_waveform.shape[0] <= 8
                and current_waveform.shape[1] > current_waveform.shape[0]
            )
            if not channels_first:
                current_waveform = current_waveform.transpose(0, 1)
            force_mono = bool(normalize_audio)
            if force_mono:
                current_waveform = current_waveform.mean(dim=0)
            else:
                current_waveform = current_waveform[0]
        elif current_waveform.ndim == 1:
            pass
        else:
            raise ValueError(f"Unsupported waveform dims: {current_waveform.ndim}")

        current_waveform = current_waveform.to(dtype=torch.float32)

        device_choice = preprocess_device or "auto"
        if device_choice == "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")
        elif device_choice == "cpu":
            target_device = torch.device("cpu")
        else:
            target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if current_waveform.device != target_device:
            current_waveform = current_waveform.to(target_device)

        if sample_rate != target_sample_rate:
            try:
                resampler = self._get_resampler(
                    sample_rate,
                    target_sample_rate,
                    current_waveform.device,
                    resample_quality,
                )
                current_waveform = resampler(current_waveform)
            except Exception as e:
                self._log_warning(
                    f"Resample failed on device {current_waveform.device}, retrying on CPU: {e}"
                )
                current_waveform = current_waveform.cpu()
                resampler = self._get_resampler(
                    sample_rate,
                    target_sample_rate,
                    current_waveform.device,
                    resample_quality,
                )
                current_waveform = resampler(current_waveform)

        if normalize_audio:
            peak = current_waveform.abs().max().item() if current_waveform.numel() else 0.0
            if peak > 0:
                current_waveform = current_waveform / peak
            current_waveform = current_waveform.clamp(-1.0, 1.0)

        self._log_tensor_shape("Prepared waveform", current_waveform, color=self._COLOR_CYAN)

        prepared_audio_numpy = current_waveform.cpu().numpy().astype(np.float32, copy=False)
        return prepared_audio_numpy, target_sample_rate

    def _merge_segments(self, segments, overlap_tolerance_s):
        if not segments:
            return []
        segments_sorted = sorted(segments, key=lambda s: (s["start"], s["end"]))
        merged = []
        for seg in segments_sorted:
            if seg["end"] < seg["start"]:
                seg["end"] = seg["start"]
            if not merged:
                merged.append(seg)
                continue
            last = merged[-1]
            if seg["start"] <= last["end"] + overlap_tolerance_s:
                if seg["text"] == last["text"] or seg["text"] in last["text"]:
                    last["end"] = max(last["end"], seg["end"])
                    continue
                if last["text"] in seg["text"]:
                    last["text"] = seg["text"]
                    last["end"] = max(last["end"], seg["end"])
                    continue
            merged.append(seg)
        return merged

    def _build_generate_kwargs(
        self,
        task,
        language,
        num_beams,
        temperature,
        temperature_fallbacks,
        condition_on_prev_tokens,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        initial_prompt,
    ):
        generate_kwargs = {"task": task}
        if language != "auto":
            generate_kwargs["language"] = language
        if num_beams and num_beams > 1:
            generate_kwargs["num_beams"] = int(num_beams)
        if temperature_fallbacks is not None:
            generate_kwargs["temperature"] = temperature_fallbacks
        else:
            if temperature is None:
                temperature = 0.0
            generate_kwargs["temperature"] = float(temperature)
        if compression_ratio_threshold is None:
            compression_ratio_threshold = 1.35
        if logprob_threshold is None:
            logprob_threshold = -1.0
        if no_speech_threshold is None:
            no_speech_threshold = 0.6
        generate_kwargs["compression_ratio_threshold"] = float(compression_ratio_threshold)
        generate_kwargs["logprob_threshold"] = float(logprob_threshold)
        generate_kwargs["no_speech_threshold"] = float(no_speech_threshold)
        if condition_on_prev_tokens is None:
            condition_on_prev_tokens = False
        generate_kwargs["condition_on_prev_tokens"] = bool(condition_on_prev_tokens)
        if initial_prompt and self.processor and hasattr(self.processor, "get_prompt_ids"):
            try:
                generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(initial_prompt)
            except Exception as e:
                self._log_warning(f"Failed to build prompt_ids: {e}")
        return generate_kwargs

    @classmethod
    def INPUT_TYPES(s):
        default_output_dir = (
            folder_paths.get_output_directory()
            if hasattr(folder_paths, "get_output_directory")
            else ""
        )

        tooltips = {
            "audio": "Входной аудиотензор ComfyUI (waveform + sample_rate).",
            "model": "Модель Whisper из Hugging Face Hub.",
            "output_filename_prefix": "Префикс имени файла для SRT/TTML.",
            "task": "transcribe = распознавание на исходном языке; translate_to_english = перевод в английский.",
            "source_language": "Язык входного аудио. auto = автоопределение. Для русского используйте ru.",
            "timestamps": "segment = таймкоды по фразам; word = по словам; none = без таймкодов (SRT/TTML отключаются).",
            "save_srt_file": "Сохранять SRT/TTML в папку output/subtitles.",
            "precision": "Точность модели. fp16 быстрее на GPU; fp32 стабильнее.",
            "attn_implementation": "Режим attention: sdpa быстрее на PyTorch 2.x, eager совместимее.",
            "plain_text_format": "Формат текста: одной строкой или по строке на сегмент.",
            "long_form_mode": "Режим long-form: chunked = ручной чанкинг (быстро), sequential = последовательный (обычно точнее).",
            "manual_chunk_length_s": "Длина чанка в секундах. В вашем режиме оптимально ~20с.",
            "manual_chunk_overlap_s": "Перекрытие чанков в секундах для более плавной стыковки.",
            "normalize_audio": "Нормализация по пику и сведение в моно (среднее по каналам).",
            "preprocess_device": "Где выполнять препроцессинг/ресемпл: auto, cpu или cuda.",
            "resample_quality": "Качество ресемпла: fast быстрее, balanced компромисс, high качественнее (по умолчанию).",
            "num_beams": "Beam search. 1 = greedy. Больше = точнее, но медленнее.",
            "temperature": "Температура декодирования. 0 = детерминированно.",
            "temperature_fallbacks": "Список температур через запятую (пример: 0.0,0.2,0.4,0.6,0.8,1.0). Если задан, перекрывает temperature.",
            "condition_on_prev_tokens": "Использовать предыдущие токены как контекст (повышает связность, иногда усиливает дрейф).",
            "compression_ratio_threshold": "Порог zlib-сжатия для детектора галлюцинаций.",
            "logprob_threshold": "Порог среднего лог‑проба; ниже — fallback/отсев.",
            "no_speech_threshold": "Порог отсутствия речи; выше — сегмент может быть отброшен.",
            "initial_prompt": "Начальная подсказка (лексика/имена) на русском.",
            "output_dir": "Папка вывода для SRT/TTML. Пусто = стандартная output.",
        }

        model_list = [
            "openai/whisper-large-v3",
            "openai/whisper-large-v3-turbo",
        ]

        precisions = ["fp32", "fp16"]
        if (
            torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        ):
            precisions.append("bf16")

        attn_implementations = ["eager"]
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn_implementations.append("sdpa")

        default_attn = "sdpa" if "sdpa" in attn_implementations else "eager"

        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": tooltips["audio"]}),
                "model": (model_list, {"default": "openai/whisper-large-v3", "tooltip": tooltips["model"]}),
                "output_filename_prefix": (
                    "STRING",
                    {"default": "transcribed_audio", "tooltip": tooltips["output_filename_prefix"]},
                ),
                "task": (["transcribe", "translate_to_english"], {"default": "transcribe", "tooltip": tooltips["task"]}),
                "source_language": (
                    ["auto", "en", "ru", "fr", "de", "es", "it", "ja", "ko", "zh", "uk", "pl"],
                    {"default": "ru", "tooltip": tooltips["source_language"]},
                ),
                "timestamps": (
                    ["segment", "word", "none"],
                    {"default": "segment", "tooltip": tooltips["timestamps"]},
                ),
                "save_srt_file": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Yes (Save SRT & TTML)",
                        "label_off": "No",
                        "tooltip": tooltips["save_srt_file"],
                    },
                ),
                "precision": (
                    precisions,
                    {"default": "fp16" if "fp16" in precisions else "fp32", "tooltip": tooltips["precision"]},
                ),
                "attn_implementation": (
                    attn_implementations,
                    {"default": default_attn, "tooltip": tooltips["attn_implementation"]},
                ),
                "plain_text_format": (
                    ["single_block", "newline_per_segment"],
                    {"default": "single_block", "tooltip": tooltips["plain_text_format"]},
                ),
                "long_form_mode": (
                    ["chunked", "sequential"],
                    {"default": "chunked", "tooltip": tooltips["long_form_mode"]},
                ),
                "manual_chunk_length_s": (
                    "FLOAT",
                    {"default": 20.0, "min": 5.0, "max": 30.0, "step": 1.0, "tooltip": tooltips["manual_chunk_length_s"]},
                ),
                "manual_chunk_overlap_s": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.5, "tooltip": tooltips["manual_chunk_overlap_s"]},
                ),
                "normalize_audio": ("BOOLEAN", {"default": True, "tooltip": tooltips["normalize_audio"]}),
                "preprocess_device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto", "tooltip": tooltips["preprocess_device"]},
                ),
                "resample_quality": (
                    ["fast", "balanced", "high"],
                    {"default": "high", "tooltip": tooltips["resample_quality"]},
                ),
                "num_beams": ("INT", {"default": 5, "min": 1, "max": 10, "tooltip": tooltips["num_beams"]}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": tooltips["temperature"]},
                ),
                "temperature_fallbacks": (
                    "STRING",
                    {"default": "0.0,0.2,0.4,0.6,0.8,1.0", "multiline": False, "tooltip": tooltips["temperature_fallbacks"]},
                ),
                "condition_on_prev_tokens": (
                    "BOOLEAN",
                    {"default": True, "tooltip": tooltips["condition_on_prev_tokens"]},
                ),
                "compression_ratio_threshold": (
                    "FLOAT",
                    {"default": 1.35, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": tooltips["compression_ratio_threshold"]},
                ),
                "logprob_threshold": (
                    "FLOAT",
                    {"default": -1.0, "min": -5.0, "max": 0.0, "step": 0.1, "tooltip": tooltips["logprob_threshold"]},
                ),
                "no_speech_threshold": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": tooltips["no_speech_threshold"]},
                ),
                "initial_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": tooltips["initial_prompt"]}),
            },
            "optional": {
                "output_dir": (
                    "STRING",
                    {"default": default_output_dir, "multiline": False, "tooltip": tooltips["output_dir"]},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("srt_content", "text_content", "ttml_content")
    FUNCTION = "generate_srt_and_text"
    CATEGORY = "TS/Audio"
    OUTPUT_NODE = False

    def generate_srt_and_text(
        self,
        audio,
        model,
        output_filename_prefix,
        task,
        source_language,
        timestamps,
        save_srt_file,
        precision,
        attn_implementation,
        plain_text_format,
        long_form_mode,
        manual_chunk_length_s,
        manual_chunk_overlap_s,
        normalize_audio,
        preprocess_device,
        resample_quality,
        num_beams,
        temperature,
        temperature_fallbacks,
        condition_on_prev_tokens,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        initial_prompt,
        output_dir=None,
    ):
        if output_filename_prefix is None:
            output_filename_prefix = "transcribed_audio"
        if task is None:
            task = "transcribe"
        if source_language is None:
            source_language = "ru"
        if timestamps is None:
            timestamps = "segment"
        if save_srt_file is None:
            save_srt_file = True
        if plain_text_format is None:
            plain_text_format = "single_block"
        if long_form_mode is None:
            long_form_mode = "chunked"
        if long_form_mode not in ("chunked", "sequential"):
            long_form_mode = "chunked"
        if preprocess_device not in ("auto", "cpu", "cuda"):
            preprocess_device = "auto"
        if resample_quality not in ("fast", "balanced", "high"):
            resample_quality = "high"
        if manual_chunk_length_s is None:
            manual_chunk_length_s = 20.0
        if manual_chunk_overlap_s is None:
            manual_chunk_overlap_s = 1.0
        if normalize_audio is None:
            normalize_audio = True
        if num_beams is None:
            num_beams = 5
        if temperature is None:
            temperature = 0.0
        if temperature_fallbacks is None:
            temperature_fallbacks = "0.0,0.2,0.4,0.6,0.8,1.0"
        if condition_on_prev_tokens is None:
            condition_on_prev_tokens = True
        if compression_ratio_threshold is None:
            compression_ratio_threshold = 1.35
        if logprob_threshold is None:
            logprob_threshold = -1.0
        if no_speech_threshold is None:
            no_speech_threshold = 0.6
        if initial_prompt is None:
            initial_prompt = ""

        manual_chunk_length_s = self._safe_float(manual_chunk_length_s, 20.0)
        manual_chunk_overlap_s = self._safe_float(manual_chunk_overlap_s, 1.0)
        temperature = self._safe_float(temperature, 0.0)
        compression_ratio_threshold = self._safe_float(compression_ratio_threshold, 1.35)
        logprob_threshold = self._safe_float(logprob_threshold, -1.0)
        no_speech_threshold = self._safe_float(no_speech_threshold, 0.6)
        num_beams = self._safe_int(num_beams, 5)
        temperature_fallbacks_parsed = self._parse_temperature_fallbacks(temperature_fallbacks)

        manual_chunk_length_s = max(1.0, manual_chunk_length_s)
        manual_chunk_overlap_s = max(0.0, manual_chunk_overlap_s)
        if num_beams < 1:
            num_beams = 1

        self._log_info(
            f"Start. Model: '{model}', Task: '{task}', Language: '{source_language}'"
        )
        self._log_info(
            "Params: "
            f"long_form_mode={long_form_mode}, timestamps={timestamps}, num_beams={num_beams}, "
            f"temperature={temperature}, temperature_fallbacks={temperature_fallbacks_parsed}, "
            f"condition_on_prev_tokens={condition_on_prev_tokens}, "
            f"compression_ratio_threshold={compression_ratio_threshold}, "
            f"logprob_threshold={logprob_threshold}, no_speech_threshold={no_speech_threshold}, "
            f"chunk_len_s={manual_chunk_length_s}, chunk_overlap_s={manual_chunk_overlap_s}, "
            f"preprocess_device={preprocess_device}, resample_quality={resample_quality}"
        )

        try:
            prepared_audio_numpy, target_sample_rate = self._prepare_audio(
                audio,
                target_sample_rate=16000,
                normalize_audio=normalize_audio,
                preprocess_device=preprocess_device,
                resample_quality=resample_quality,
            )
        except Exception as e:
            self._log_error(f"Audio preprocessing failed: {e}")
            return ("", "", "")

        if prepared_audio_numpy.size == 0:
            self._log_warning("Prepared audio is empty.")
            return ("", "", "")

        try:
            pipe = self.get_pipeline(model, precision, attn_implementation)
            if pipe is None:
                return ("", "", "")
        except Exception as e:
            self._log_error(f"Failed to get ASR pipeline: {e}")
            return ("", "", "")

        return_timestamps = True
        if timestamps == "none":
            return_timestamps = False
            if save_srt_file:
                self._log_warning("Timestamps disabled; SRT/TTML saving turned off.")
                save_srt_file = False
        elif timestamps == "word":
            return_timestamps = "word"

        full_srt_path = ""
        full_ttml_path = ""
        if save_srt_file:
            base_output_dir = output_dir or (
                folder_paths.get_output_directory()
                if hasattr(folder_paths, "get_output_directory")
                else None
            )
            if base_output_dir and os.path.isdir(base_output_dir):
                subtitles_dir = os.path.join(base_output_dir, "subtitles")
                try:
                    os.makedirs(subtitles_dir, exist_ok=True)
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    full_srt_path = os.path.join(
                        subtitles_dir, f"{output_filename_prefix}_{timestamp_str}.srt"
                    )
                    full_ttml_path = os.path.join(
                        subtitles_dir, f"{output_filename_prefix}_{timestamp_str}.ttml"
                    )
                except Exception as e:
                    self._log_error(f"Failed to create subtitles directory: {e}")
                    save_srt_file = False
            else:
                save_srt_file = False

        all_segments = []
        all_text_segments = []
        actual_whisper_task = "translate" if task == "translate_to_english" else "transcribe"

        generate_kwargs = self._build_generate_kwargs(
            task=actual_whisper_task,
            language=source_language,
            num_beams=num_beams,
            temperature=temperature,
            temperature_fallbacks=temperature_fallbacks_parsed,
            condition_on_prev_tokens=condition_on_prev_tokens,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            initial_prompt=initial_prompt,
        )

        try:
            merged_segments = []
            if long_form_mode == "sequential":
                try:
                    with torch.inference_mode():
                        output_chunk = pipe(
                            prepared_audio_numpy,
                            return_timestamps=return_timestamps,
                            generate_kwargs=generate_kwargs,
                        )
                except Exception as exc:
                    if self._is_oom_error(exc):
                        self._log_warning(
                            "Sequential mode ran out of memory. Falling back to chunked mode."
                        )
                        try:
                            comfy.model_management.soft_empty_cache()
                        except Exception:
                            pass
                        merged_segments, all_text_segments = self._run_chunked_inference(
                            pipe=pipe,
                            prepared_audio_numpy=prepared_audio_numpy,
                            target_sample_rate=target_sample_rate,
                            return_timestamps=return_timestamps,
                            generate_kwargs=generate_kwargs,
                            manual_chunk_length_s=manual_chunk_length_s,
                            manual_chunk_overlap_s=manual_chunk_overlap_s,
                        )
                    else:
                        self._log_error("Pipeline execution failed in sequential mode.")
                        self.logger.error(
                            f"{self._LOG_PREFIX} generate_kwargs={generate_kwargs}",
                            exc_info=True,
                        )
                        raise
                else:
                    self._collect_output_segments(
                        output_chunk, 0.0, all_segments, all_text_segments
                    )
                    merged_segments = list(all_segments)
            else:
                merged_segments, all_text_segments = self._run_chunked_inference(
                    pipe=pipe,
                    prepared_audio_numpy=prepared_audio_numpy,
                    target_sample_rate=target_sample_rate,
                    return_timestamps=return_timestamps,
                    generate_kwargs=generate_kwargs,
                    manual_chunk_length_s=manual_chunk_length_s,
                    manual_chunk_overlap_s=manual_chunk_overlap_s,
                )

            subtitles = []
            if merged_segments:
                for idx, seg in enumerate(merged_segments, start=1):
                    subtitles.append(
                        srt.Subtitle(
                            index=idx,
                            start=datetime.timedelta(seconds=seg["start"]),
                            end=datetime.timedelta(seconds=seg["end"]),
                            content=seg["text"],
                        )
                    )

            generated_srt = srt.compose(subtitles, reindex=True) if subtitles else ""
            ttml_lang = "en" if actual_whisper_task == "translate" else (
                source_language if source_language != "auto" else "und"
            )
            generated_ttml = self._create_ttml_content(subtitles, ttml_lang) if subtitles else ""

            sep = "\n" if plain_text_format == "newline_per_segment" else " "
            if merged_segments:
                plain_text = sep.join([seg["text"] for seg in merged_segments]).strip()
            else:
                plain_text = sep.join(all_text_segments).strip()

            if save_srt_file:
                if full_srt_path:
                    with open(full_srt_path, "w", encoding="utf-8") as f:
                        f.write(generated_srt)
                    self._log_info(f"SRT saved: {full_srt_path}")
                if full_ttml_path:
                    with open(full_ttml_path, "w", encoding="utf-8") as f:
                        f.write(generated_ttml)
                    self._log_info(f"TTML saved: {full_ttml_path}")

            self._log_info(
                f"Done. Segments: {len(merged_segments)}, Text chars: {len(plain_text)}"
            )
            return (generated_srt, plain_text, generated_ttml)

        except Exception as e:
            self.logger.error(f"{self._LOG_PREFIX} Execution error: {e}", exc_info=True)
            return (f"Error: {e}", "", "")


NODE_CLASS_MAPPINGS = {"TSWhisper": TSWhisper}
NODE_DISPLAY_NAME_MAPPINGS = {"TSWhisper": "TS Whisper"}
