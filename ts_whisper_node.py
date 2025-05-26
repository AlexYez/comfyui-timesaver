import os
import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import srt
import datetime
import logging
import folder_paths
import numpy as np
import comfy.utils # Для ProgressBar ComfyUI

logger = logging.getLogger("comfyui_ts_whisper")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TSWhisper:
    def __init__(self):
        self.model_name = "openai/whisper-large-v3"
        self.target_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.pipeline = None
        self.hf_cache_dir = None
        self.current_pipeline_config = {}

        try:
            base_path = folder_paths.base_path if hasattr(folder_paths, 'base_path') and folder_paths.base_path else os.getcwd()
            comfy_models_dir = os.path.join(base_path, "models")
            if not hasattr(folder_paths, 'base_path') or not folder_paths.base_path:
                 logger.warning(f"folder_paths.base_path не найден, используется {comfy_models_dir} как путь к папке models.")

            self.hf_cache_dir = os.path.join(comfy_models_dir, "whisper")
            os.makedirs(self.hf_cache_dir, exist_ok=True)
            logger.info(f"Модели Hugging Face (Whisper) будут кэшироваться в: {self.hf_cache_dir}")
        except Exception as e:
            logger.warning(f"Не удалось настроить кастомную директорию для кэша Hugging Face ({self.hf_cache_dir}). Будет использован стандартный кэш. Ошибка: {e}")
            self.hf_cache_dir = None

        logger.info(f"{self.__class__.__name__} инициализирован. Целевое устройство: {self.target_device_type}")

    def _should_reload_pipeline(self, new_config):
        if self.pipeline is None: return True
        # Ключи, изменение которых требует перезагрузки модели/пайплайна
        keys_to_check = ["precision", "attn_implementation", "model_name"]
        for key in keys_to_check:
            if self.current_pipeline_config.get(key) != new_config.get(key):
                logger.info(f"Обнаружено изменение в конфигурации: {key} ('{self.current_pipeline_config.get(key)}' -> '{new_config.get(key)}')")
                return True
        return False

    def load_model_and_processor(self, precision, attn_implementation_choice):
        try:
            logger.info(f"Загрузка модели: {self.model_name} (precision={precision}, attn={attn_implementation_choice})")
            actual_torch_dtype = torch.float32
            if precision == "fp16": actual_torch_dtype = torch.float16
            elif precision == "bf16":
                if self.target_device_type == "cuda" and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    actual_torch_dtype = torch.bfloat16
                else:
                    logger.warning("bf16 не поддерживается на CUDA или текущей системе/PyTorch, используется fp16."); actual_torch_dtype = torch.float16
            
            model_load_kwargs = {"torch_dtype": actual_torch_dtype, "low_cpu_mem_usage": True, "use_safetensors": True}
            if self.hf_cache_dir: model_load_kwargs["cache_dir"] = self.hf_cache_dir
            
            # Логика выбора реализации внимания (без flash_attn)
            if attn_implementation_choice == "sdpa" and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model_load_kwargs["attn_implementation"] = "sdpa"
                logger.info("Используется PyTorch SDPA (Scaled Dot Product Attention).")
            elif attn_implementation_choice == "sdpa": # SDPA выбран, но недоступен
                 logger.warning("SDPA не доступен (PyTorch < 2.0?). Используется eager attention (по умолчанию).")
            # Если выбран "eager" или другая опция не подошла, будет использована реализация по умолчанию (eager).

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, **model_load_kwargs)
            if self.model.device.type != self.target_device_type:
                 self.model.to(self.target_device_type)
            logger.info(f"Модель {self.model_name} загружена. Финальное устройство модели: {self.model.device}")

            processor_load_kwargs = {"cache_dir": self.hf_cache_dir} if self.hf_cache_dir else {}
            self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_load_kwargs)
            logger.info(f"Процессор для {self.model_name} загружен.")
            self.current_pipeline_config = {"precision": precision, "attn_implementation": attn_implementation_choice, "model_name": self.model_name}
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки модели/процессора: {e}", exc_info=True)
            self.model, self.processor, self.pipeline = None, None, None; raise
        return self.model, self.processor, actual_torch_dtype

    def get_pipeline(self, precision, attn_implementation):
        new_config = {"precision": precision, "attn_implementation": attn_implementation, "model_name": self.model_name}
        if self._should_reload_pipeline(new_config):
            logger.info("(Пере)загрузка модели и создание пайплайна...")
            self.pipeline = None 
            model, processor, actual_torch_dtype_for_pipeline = self.load_model_and_processor(precision, attn_implementation)
            
            pipeline_kwargs = {
                "model": model, "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor, "torch_dtype": actual_torch_dtype_for_pipeline,
                "device": self.target_device_type
            }
            self.pipeline = pipeline("automatic-speech-recognition", **pipeline_kwargs)
            logger.info(f"ASR пайплайн создан/обновлен. Устройство: {self.pipeline.device}, dtype: {actual_torch_dtype_for_pipeline}")
        return self.pipeline

    @classmethod
    def INPUT_TYPES(s):
        default_output_dir = folder_paths.get_output_directory() if hasattr(folder_paths, 'get_output_directory') else ""
        precisions = ["fp32", "fp16"]
        if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported(): precisions.append("bf16")
        
        # Убираем flash_attn из доступных опций
        attn_implementations = ["eager"]
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'): attn_implementations.append("sdpa")
        
        default_attn = "eager"
        if "sdpa" in attn_implementations: default_attn = "sdpa" # SDPA теперь лучший вариант по умолчанию, если доступен

        return {
            "required": {
                "audio": ("AUDIO", ),
                "output_filename_prefix": ("STRING", {"default": "transcribed_audio"}),
                "task": (["transcribe", "translate_to_english"], {"default": "transcribe"}),
                "source_language": (["auto", "en", "ru", "fr", "de", "es", "it", "ja", "ko", "zh", "uk", "pl"], {"default": "auto"}),
                "save_srt_file": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
                "precision": (precisions, {"default": "fp16" if "fp16" in precisions else "fp32"}),
                "attn_implementation": (attn_implementations, {"default": default_attn}),
                "plain_text_format": (["single_block", "newline_per_segment"], {"default": "single_block", "tooltip":"Формат вывода простого текста"}),
                "manual_chunk_length_s": ("FLOAT", {"default": 28.0, "min": 5.0, "max": 30.0, "step": 1.0, "tooltip": "Длина ручного чанка в секундах (<=30s)"}),
                "manual_chunk_overlap_s": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5, "tooltip": "Перекрытие между ручными чанками в секундах"}),
            },
            "optional": {
                 "output_dir": ("STRING", {"default": default_output_dir, "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("srt_content_string", "plain_text_string",)
    FUNCTION = "generate_srt_and_text"
    CATEGORY = "AudioTranscription/TSNodes" 
    OUTPUT_NODE = False

    def generate_srt_and_text(self, audio, output_filename_prefix, task, source_language, save_srt_file, 
                              precision, attn_implementation, plain_text_format,
                              manual_chunk_length_s, manual_chunk_overlap_s,
                              output_dir=None):
        logger.info(f"Получен аудиовход.")
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            logger.error("Некорректный формат аудиовхода."); return ("", "",)
        waveform_tensor, sample_rate = audio["waveform"], audio["sample_rate"]
        if isinstance(waveform_tensor, list):
            if not waveform_tensor: logger.error("Аудио 'waveform' - пустой список."); return ("", "",)
            waveform_tensor = waveform_tensor[0]
        if not isinstance(waveform_tensor, torch.Tensor):
            logger.error(f"Аудио 'waveform' не тензор (тип: {type(waveform_tensor)})."); return ("", "",)
        
        logger.info(f"Параметры: Задача='{task}', Язык='{source_language}', Сохр.={save_srt_file}, Формат текста='{plain_text_format}'")
        logger.info(f"Ручной чанкинг: Длина={manual_chunk_length_s}с, Перекрытие={manual_chunk_overlap_s}с")
        logger.info(f"Оптимизации: Точность='{precision}', Внимание='{attn_implementation}'")

        try:
            pipe = self.get_pipeline(precision, attn_implementation)
            if pipe is None: return ("", "",)
        except Exception as e:
            logger.error(f"Не удалось получить ASR пайплайн: {e}", exc_info=True); return ("", "",)

        prepared_audio_numpy = None; target_sample_rate = 16000
        try:
            current_waveform = waveform_tensor.clone()
            if current_waveform.ndim == 3: current_waveform = current_waveform[0]
            if current_waveform.ndim == 2:
                current_waveform = current_waveform[0, :] if current_waveform.shape[0] > 1 else current_waveform.squeeze(0)
            if sample_rate != target_sample_rate:
                resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate).to(current_waveform.device)
                current_waveform = resampler(current_waveform)
            prepared_audio_numpy = current_waveform.cpu().numpy().astype(np.float32)
            logger.info(f"Аудио подготовлено для пайплайна: форма {prepared_audio_numpy.shape}")
        except Exception as e_proc:
            logger.error(f"Ошибка обработки аудио: {e_proc}", exc_info=True); return (f"Ошибка аудио: {str(e_proc)}", "",)

        full_srt_path_with_subdir = ""
        if save_srt_file:
            base_output_dir = output_dir or (folder_paths.get_output_directory() if hasattr(folder_paths, 'get_output_directory') else None)
            if base_output_dir and os.path.isdir(base_output_dir):
                subtitles_dir = os.path.join(base_output_dir, "subtitles")
                try:
                    os.makedirs(subtitles_dir, exist_ok=True)
                    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    srt_file_name = f"{output_filename_prefix}_{timestamp_str}.srt"
                    full_srt_path_with_subdir = os.path.join(subtitles_dir, srt_file_name)
                except Exception as e_mkdir:
                    logger.error(f"Не удалось создать/получить путь в {subtitles_dir}: {e_mkdir}"); save_srt_file = False
            else: 
                logger.warning(f"Базовая директория вывода '{base_output_dir}' некорректна или не определена. Файл не будет сохранен.")
                save_srt_file = False
        
        generated_srt_content = ""; plain_text_output = ""
        all_subtitles_segments = []; all_text_segments = []
        actual_whisper_task = "translate" if task == "translate_to_english" else "transcribe"
        
        try:
            generate_kwargs = {"task": actual_whisper_task}
            if source_language != "auto": generate_kwargs["language"] = source_language

            total_duration_samples = len(prepared_audio_numpy)
            chunk_length_samples = int(manual_chunk_length_s * target_sample_rate)
            overlap_samples = int(manual_chunk_overlap_s * target_sample_rate)
            step_samples = chunk_length_samples - overlap_samples

            if chunk_length_samples <= 0 :
                logger.error("Длина чанка должна быть больше 0."); return ("", "",)
            if step_samples <= 0 :
                logger.warning(f"Шаг чанка ({step_samples} сэмплов) <= 0. Используется длина чанка как шаг."); step_samples = chunk_length_samples

            num_chunks = (total_duration_samples - overlap_samples + step_samples - 1) // step_samples if total_duration_samples > overlap_samples else 1
            num_chunks = max(1, num_chunks)

            logger.info(f"Начало ручного чанкинга: Всего сэмплов={total_duration_samples}, Сэмплов в чанке={chunk_length_samples}, Шаг={step_samples}, Кол-во чанков={num_chunks}")

            current_sample_offset = 0
            pbar = comfy.utils.ProgressBar(num_chunks)
            
            for i in range(num_chunks):
                start_sample = current_sample_offset
                end_sample = min(start_sample + chunk_length_samples, total_duration_samples)
                chunk_audio_numpy = prepared_audio_numpy[start_sample:end_sample]

                if len(chunk_audio_numpy) == 0:
                    current_sample_offset += step_samples
                    pbar.update(1); continue
                
                # logger.info(f"Обработка чанка {i+1}/{num_chunks} (сэмплы {start_sample}-{end_sample})...") # Можно раскомментировать для детального лога
                output_chunk = pipe(chunk_audio_numpy, return_timestamps=True, generate_kwargs=generate_kwargs)

                if output_chunk and "chunks" in output_chunk and isinstance(output_chunk["chunks"], list):
                    for segment_data in output_chunk["chunks"]:
                        text = segment_data["text"].strip()
                        if not text: continue
                        all_text_segments.append(text)
                        timestamp_data = segment_data.get("timestamp")
                        if timestamp_data and isinstance(timestamp_data, tuple) and len(timestamp_data) == 2:
                            chunk_start_s, chunk_end_s = timestamp_data
                            if chunk_start_s is not None and chunk_end_s is not None:
                                absolute_start_s = (start_sample / target_sample_rate) + chunk_start_s
                                absolute_end_s = (start_sample / target_sample_rate) + chunk_end_s
                                if absolute_end_s < absolute_start_s: 
                                    absolute_end_s = absolute_start_s + max(0.1, len(text) * 0.05)
                                all_subtitles_segments.append(srt.Subtitle(
                                    index=len(all_subtitles_segments) + 1,
                                    start=datetime.timedelta(seconds=absolute_start_s),
                                    end=datetime.timedelta(seconds=absolute_end_s),
                                    content=text
                                ))
                current_sample_offset += step_samples
                pbar.update(1)
                if end_sample >= total_duration_samples: break

            logger.info("Ручной чанкинг и распознавание завершены.")

            if all_subtitles_segments: 
                generated_srt_content = srt.compose(all_subtitles_segments, reindex=True)
            if all_text_segments:
                plain_text_output = ("\n".join(all_text_segments) if plain_text_format == "newline_per_segment" else " ".join(all_text_segments)).strip()
                logger.info(f"SRT и чистый текст сгенерированы (формат: {plain_text_format}).")
            else:
                logger.warning("Текстовые сегменты не были извлечены. Чистый текст будет пустым.")

            if save_srt_file and full_srt_path_with_subdir:
                file_content_to_save = generated_srt_content or "" 
                try:
                    with open(full_srt_path_with_subdir, "w", encoding="utf-8") as f: f.write(file_content_to_save)
                    logger.info(f"SRT файл ({'пустой' if not file_content_to_save else 'с контентом'}) сохранен: {full_srt_path_with_subdir}")
                except Exception as e_save: 
                    logger.error(f"Не удалось сохранить SRT файл в {full_srt_path_with_subdir}: {e_save}")
            
            return (generated_srt_content, plain_text_output,)
        except Exception as e:
            logger.error(f"Ошибка при генерации SRT и текста: {e}", exc_info=True)
            if save_srt_file and all_subtitles_segments and full_srt_path_with_subdir:
                 try:
                    partial_srt_content = srt.compose(all_subtitles_segments, reindex=True)
                    partial_plain_text = ("\n".join(all_text_segments) if plain_text_format == "newline_per_segment" else " ".join(all_text_segments)).strip() if all_text_segments else ""
                        
                    base_name=os.path.basename(full_srt_path_with_subdir); dir_name=os.path.dirname(full_srt_path_with_subdir)
                    partial_filename = base_name.replace(".srt", "_partial_error.srt")
                    if dir_name:
                        partial_srt_path = os.path.join(dir_name, partial_filename)
                        with open(partial_srt_path, "w", encoding="utf-8") as f_partial: f_partial.write(partial_srt_content)
                        logger.info(f"Частичный SRT сохранен: {partial_srt_path}")
                        return (partial_srt_content, partial_plain_text,)
                 except Exception as e_save_partial: 
                    logger.error(f"Не удалось сохранить частичный SRT: {e_save_partial}")
            return (f"Критическая ошибка: {str(e)}","",)


NODE_CLASS_MAPPINGS = { "TSWhisper": TSWhisper }
NODE_DISPLAY_NAME_MAPPINGS = { "TSWhisper": "TS Whisper" }

logger.info("Нода TS Whisper обновлена (удален flash_attn, код очищен).")