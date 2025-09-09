import os
import logging
import gc
import torch
import folder_paths
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import numpy as np
import tempfile
import subprocess
import math
import soundfile as sf
import librosa

# Проверка наличия Transformers
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("##########")
    print("ERROR: transformers не установлены. Нода Gemma3n не будет работать.")
    print("Пожалуйста, установите их: pip install transformers torch torchvision")
    print("##########")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Gemma3nNode")

# Папка моделей ComfyUI
llm_models_dir = folder_paths.get_folder_paths("llm")[0] if folder_paths.get_folder_paths("llm") else None
if llm_models_dir and not os.path.exists(llm_models_dir):
    os.makedirs(llm_models_dir, exist_ok=True)

def round_to_32(x):
    return max(32, int(round(x / 32) * 32))

# Конвертация ComfyUI изображения в PIL с возможностью ресайза по большей стороне
def comfy_to_pil(image, resize_longest=None):
    if image is None:
        return None
    arr = image.detach().cpu().numpy() if hasattr(image, 'detach') else np.array(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype in [np.float32, np.float64]:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    pil_img = Image.fromarray(arr)

    if resize_longest is not None:
        w, h = pil_img.size
        if w >= h:
            new_w = resize_longest
            new_h = int(h * (resize_longest / w))
        else:
            new_h = resize_longest
            new_w = int(w * (resize_longest / h))

        new_w = round_to_32(new_w)
        new_h = round_to_32(new_h)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        logger.info(f"[Resize] Resized image to {new_w}x{new_h}")

    return pil_img

# Конвертация аудио для Gemma3n с сохранением временного WAV файла
def comfy_audio_to_numpy(audio):
    try:
        logger.info(f"[Audio Debug] Received audio object of type: {type(audio)}")

        if isinstance(audio, dict):
            logger.info(f"[Audio Debug] Audio dict keys: {list(audio.keys())}")
            if "waveform" in audio and "sample_rate" in audio:
                y = audio["waveform"]
                sr = audio["sample_rate"]
                logger.info(f"[Audio Debug] waveform shape: {y.shape}, sample_rate: {sr}")

                if y.ndim == 3:
                    y = y[0]
                    logger.info(f"[Audio Debug] Taking first batch. New shape: {y.shape}")

                if y.shape[0] > 1:
                    y = y.mean(dim=0)
                    logger.info(f"[Audio Debug] Converted to mono. New shape: {y.shape}")

                y = y.cpu().numpy().astype(np.float32)

                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    logger.info(f"[Audio Debug] Resampled to 16 kHz. New length: {len(y)}")
            else:
                raise TypeError(f"Audio dict missing 'waveform' or 'sample_rate'. Keys: {list(audio.keys())}")
        else:
            raise TypeError(f"Unknown audio format: {type(audio)}")

        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        logger.info(f"[Audio Debug] Normalized audio. dtype: {y.dtype}, min: {y.min()}, max: {y.max()}")
        return y

    except Exception as e:
        logger.error(f"[Audio Debug] Failed to convert audio: {e}", exc_info=True)
        raise TypeError(f"Не удалось конвертировать audio для Gemma3n: {e}")

# Конвертация видео в кадры с ресайзом по большей стороне и кратным 32
def comfy_video_to_frames(video_obj, fps=0.5, max_frames=10, resize_longest=512):
    try:
        video_path = None
        if hasattr(video_obj, "path"):
            video_path = video_obj.path
        elif hasattr(video_obj, "get_path"):
            video_path = video_obj.get_path()
        elif hasattr(video_obj, "_file_path"):
            video_path = video_obj._file_path
        elif hasattr(video_obj, "filepath"):
            video_path = video_obj.filepath
        if not video_path and hasattr(video_obj, "_VideoFromFile__file"):
            video_path = getattr(video_obj, "_VideoFromFile__file", None)

        if not video_path:
            raise TypeError(f"Unsupported video object type: {type(video_obj)} — no path attribute found")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")

        tmp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_pattern = os.path.join(frames_dir, "%04d.jpg")

        scale = f"scale={resize_longest}:-1"
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps},{scale}",
            frame_pattern
        ]
        logger.info(f"[Video Debug] Running ffmpeg: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True)

        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if not frame_files:
            raise ValueError("No frames extracted from video.")

        if len(frame_files) > max_frames:
            frame_files = frame_files[:max_frames]

        processed_files = []
        for f in frame_files:
            img = Image.open(f)
            w, h = img.size
            new_w, new_h = round_to_32(w), round_to_32(h)
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img.save(f)
                logger.info(f"[Resize] Resized frame {f} to {new_w}x{new_h}")
            processed_files.append(f)

        logger.info(f"[Video Debug] Total frames processed: {len(processed_files)}")
        return processed_files
    except Exception as e:
        logger.error(f"[Video Debug] Failed to convert video: {e}", exc_info=True)
        raise TypeError(f"Не удалось конвертировать video для Gemma3n: {e}")

class Gemma3nNode:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Describe the input"})
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "audio": ("AUDIO", {"default": None}),
                "video": ("VIDEO", {"default": None}),
                "fps": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0}),
                "resize_longest": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 8192}),
                "device": ("STRING", {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "huggingface_token": ("STRING", {"default": None}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "process"
    CATEGORY = "LLM/Gemma3n"

    def _unload_model(self):
        if self.model is not None:
            logger.info("Выгрузка модели Gemma3n...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Модель выгружена.")

    def _load_model(self, device="cuda", huggingface_token=None):
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers не установлены")
            return
        if self.model is not None:
            logger.info("Модель уже загружена")
            return

        model_name = "google/gemma-3n-e2b-it"
        model_path = os.path.join(llm_models_dir, model_name.replace("/", "_"))
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            logger.info(f"Скачивание модели {model_name} в {model_path}...")

        logger.info("Загрузка модели Gemma3n...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_auth_token=huggingface_token)
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.bfloat16, use_auth_token=huggingface_token
            ).eval()
            self.current_model_path = model_path
            logger.info("Модель Gemma3n успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def process(self, text, image=None, audio=None, video=None, fps=1.0, resize_longest=512,
                max_tokens=512, device="cuda", huggingface_token=None,
                unload_after_generation=False, enable=True):
        if not enable:
            logger.info("Обработка отключена пользователем")
            if unload_after_generation:
                self._unload_model()
            return (text.strip(),)

        if not TRANSFORMERS_AVAILABLE:
            return ("Ошибка: Transformers не установлены.",)

        if self.model is None:
            self._load_model(device=device, huggingface_token=huggingface_token)

        inputs = {"text": text}
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": []}
        ]
        messages[1]["content"].append({"type": "text", "text": text})

        # Обработка изображения
        if image is not None:
            pil_img = comfy_to_pil(image, resize_longest=resize_longest)
            inputs["image"] = pil_img
            messages[1]["content"].append({"type": "image", "image": pil_img})
            logger.info(f"[Image Debug] Image processed, size: {pil_img.size}")

        # Обработка аудио
        tmp_audio_file = None
        if audio is not None:
            try:
                np_audio = comfy_audio_to_numpy(audio)
                tmp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp_audio_file.name, np_audio, 16000)
                tmp_audio_file.flush()
                inputs["audio"] = tmp_audio_file.name
                messages[1]["content"].append({"type": "audio", "path": tmp_audio_file.name})
                logger.info(f"[Audio Debug] Temporary WAV file created: {tmp_audio_file.name}, length: {len(np_audio)}")
            except TypeError as e:
                logger.error(f"Не удалось конвертировать audio для Gemma3n: {e}")

        # Обработка видео
        if video is not None:
            try:
                frame_files = comfy_video_to_frames(video, fps=fps, resize_longest=resize_longest)
                for frame_path in frame_files:
                    messages[1]["content"].append({"type": "image", "url": frame_path})
                logger.info(f"[Video Debug] {len(frame_files)} frames processed from video")
            except TypeError as e:
                logger.error(f"Не удалось конвертировать video для Gemma3n: {e}")

        try:
            logger.info("Начинаю генерацию текста...")
            proc_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = proc_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**proc_inputs, max_new_tokens=max_tokens, do_sample=False)
                generation = generation[0][input_len:]

            generated_text = self.processor.decode(generation, skip_special_tokens=True)
            logger.info("Генерация завершена успешно.")
        finally:
            # удаляем временный WAV файл
            if tmp_audio_file is not None:
                try:
                    tmp_audio_file.close()
                    os.unlink(tmp_audio_file.name)
                    logger.info(f"[Audio Debug] Temporary WAV file deleted: {tmp_audio_file.name}")
                except Exception:
                    pass

        if unload_after_generation:
            self._unload_model()

        return (generated_text,)

NODE_CLASS_MAPPINGS = {"Gemma3nNode": Gemma3nNode}
NODE_DISPLAY_NAME_MAPPINGS = {"Gemma3nNode": "Gemma 3n Node"}
