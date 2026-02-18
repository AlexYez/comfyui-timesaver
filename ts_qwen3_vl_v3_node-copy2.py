import os
import gc
import json
import logging
import importlib.util
import inspect
from contextlib import nullcontext

import torch
import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm
from huggingface_hub import snapshot_download


class TS_Qwen3_VL_V3:
    _MODEL_LIST = [
        "hfmaster/Qwen3-VL-2B",
        "hfmaster/Qwen3-VL-4B",
        "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX",
        "prithivMLmods/Qwen3-VL-8B-Instruct-Unredacted-MAX",
        "Custom (manual)"
    ]
    _MODEL_SIZES_B = {
        "hfmaster/Qwen3-VL-2B": 2,
        "hfmaster/Qwen3-VL-4B": 4,
        "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX": 4,
        "prithivMLmods/Qwen3-VL-8B-Instruct-Unredacted-MAX": 8
    }
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self._logger = logging.getLogger("TS_Qwen3_VL_V3")
        self._cache = {}
        self._cache_order = []
        self._cache_max_items = 2
        self._snapshot_endpoint_supported = None
        self._log_versions()

    @classmethod
    def INPUT_TYPES(cls):
        presets, preset_keys = cls._load_presets()
        preset_options = preset_keys + ["Your instruction"]

        precision_options = ["auto", "bf16", "fp16", "fp32"]
        if cls._is_bitsandbytes_available():
            precision_options.extend(["int8", "int4"])

        attention_options = ["auto", "flash_attention_2", "sdpa", "eager"]

        return {
            "required": {
                "model_name": (cls._MODEL_LIST, {"default": "hfmaster/Qwen3-VL-2B"}),
                "custom_model_id": ("STRING", {"multiline": False, "default": ""}),
                "system_preset": (preset_options, {"default": preset_options[0] if preset_options else "Your instruction"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "precision": (precision_options, {"default": "auto"}),
                "attention_mode": (attention_options, {"default": "auto"}),
                "offline_mode": ("BOOLEAN", {"default": False}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "enable": ("BOOLEAN", {"default": True}),
                "hf_token": ("STRING", {"multiline": False, "default": ""}),
                "max_image_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "video_max_frames": ("INT", {"default": 16, "min": 4, "max": 256, "step": 4}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "custom_system_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generated_text", "processed_image")
    FUNCTION = "process"
    CATEGORY = "TS/LLM"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        presets_path = os.path.join(cls._CURRENT_DIR, "qwen_3_vl_presets.json")
        mtime = os.path.getmtime(presets_path) if os.path.exists(presets_path) else None
        return (mtime,)

    def process(
        self,
        model_name,
        system_preset,
        prompt,
        seed,
        max_new_tokens,
        precision,
        attention_mode,
        offline_mode,
        unload_after_generation,
        enable,
        hf_token,
        max_image_size,
        video_max_frames,
        custom_model_id,
        image=None,
        video=None,
        custom_system_prompt=None,
        hf_endpoint="huggingface.co, hf-mirror.com"
    ):
        self._logger.info("[TS Qwen3 VL V3] Start")
        if image is not None:
            self._logger.info(f"[TS Qwen3 VL V3] image shape: {tuple(image.shape)}")
        if video is not None:
            self._logger.info(f"[TS Qwen3 VL V3] video shape: {tuple(video.shape)}")

        processed_images = []

        if not enable:
            self._logger.info("[TS Qwen3 VL V3] Processing mode: bypass (CPU)")
            if image is not None:
                processed_images.extend(self._tensor_to_pil_list(image))
            if video is not None:
                processed_images.extend(self._tensor_to_pil_list(video))
            out_tensor = self._pil_to_tensor(processed_images)
            self._logger.info(f"[TS Qwen3 VL V3] output image shape: {tuple(out_tensor.shape)}")
            return (prompt.strip() if prompt else "", out_tensor)

        resolved_model_id = self._resolve_model_id(model_name, custom_model_id)
        resolved_precision = self._resolve_precision(precision, resolved_model_id)
        resolved_attention = self._resolve_attention(attention_mode, resolved_precision)
        self._logger.info(f"[TS Qwen3 VL V3] model={resolved_model_id}")
        self._logger.info(f"[TS Qwen3 VL V3] precision={resolved_precision} attention={resolved_attention}")

        try:
            model, processor = self._load_model(
                resolved_model_id,
                resolved_precision,
                resolved_attention,
                offline_mode,
                hf_token,
                hf_endpoint
            )
        except Exception as e:
            self._logger.error(f"[TS Qwen3 VL V3] Load error: {e}", exc_info=True)
            out_tensor = self._pil_to_tensor(processed_images)
            self._logger.info(f"[TS Qwen3 VL V3] output image shape: {tuple(out_tensor.shape)}")
            return (f"ERROR: {e}", out_tensor)

        if torch.cuda.is_available() and not self._model_has_cuda_device(model):
            err = "CUDA available but model is on CPU; aborting to prevent CPU generation."
            self._logger.error(f"[TS Qwen3 VL V3] {err}")
            out_tensor = self._pil_to_tensor(processed_images)
            self._logger.info(f"[TS Qwen3 VL V3] output image shape: {tuple(out_tensor.shape)}")
            return (f"ERROR: {err}", out_tensor)

        preset_configs, _ = self._load_presets()
        preset_data = preset_configs.get(system_preset)
        if system_preset == "Your instruction" and custom_system_prompt:
            system_prompt = custom_system_prompt
            gen_params = {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.0}
        elif preset_data:
            system_prompt = preset_data.get("system_prompt", "")
            gen_params = dict(preset_data.get("gen_params", {}))
        else:
            system_prompt = ""
            gen_params = {}

        if "temperature" not in gen_params:
            gen_params["temperature"] = 0.7

        user_content = []

        if video is not None:
            video_frames = self._tensor_to_pil_list(video)
            total_frames = len(video_frames)
            if total_frames > video_max_frames:
                indices = np.linspace(0, total_frames - 1, video_max_frames, dtype=int)
                video_frames = [video_frames[i] for i in indices]
            processed_video = []
            for frame in video_frames:
                frame_proc = self._resize_and_crop_image(frame, max_image_size)
                processed_images.append(frame_proc)
                processed_video.append(frame_proc)
            user_content.append({"type": "video", "video": processed_video, "fps": 1.0})

        if image is not None:
            for img in self._tensor_to_pil_list(image):
                img_proc = self._resize_and_crop_image(img, max_image_size)
                processed_images.append(img_proc)
                user_content.append({"type": "image", "image": img_proc})

        user_content.append({"type": "text", "text": prompt.strip() if prompt else ""})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
            {"role": "user", "content": user_content}
        ]

        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            target_device = self._select_input_device(model)
            inputs = self._move_inputs_to_device(inputs, target_device)
            self._log_processing_device("inputs_moved", target_device, model, inputs)

            gen_params["max_new_tokens"] = max_new_tokens
            gen_params["use_cache"] = True
            gen_params["pad_token_id"] = self._get_pad_token_id(processor, model)
            gen_params["do_sample"] = gen_params.get("temperature", 0) > 0
            rng_cuda_devices = self._cuda_indices_for_rng(model, target_device)
            if self._supports_generator(model):
                gen_device = self._select_generator_device(target_device)
                gen = torch.Generator(device=gen_device)
                gen.manual_seed(seed)
                gen_params["generator"] = gen
                rng_context = nullcontext()
            else:
                rng_context = torch.random.fork_rng(devices=rng_cuda_devices) if rng_cuda_devices else torch.random.fork_rng()

            dtype = self._dtype_from_precision(resolved_precision)
            autocast_device = target_device if isinstance(target_device, torch.device) else model.device
            use_autocast = getattr(autocast_device, "type", None) == "cuda" and dtype in (torch.float16, torch.bfloat16)

            with rng_context:
                if "generator" not in gen_params:
                    torch.manual_seed(seed)
                    if rng_cuda_devices:
                        for idx in rng_cuda_devices:
                            with torch.cuda.device(idx):
                                torch.cuda.manual_seed(seed)
                self._log_processing_device("generate_start", target_device, model, inputs)
                with torch.inference_mode():
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            generated_ids = model.generate(**inputs, **gen_params)
                    else:
                        generated_ids = model.generate(**inputs, **gen_params)

            generated_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
        except Exception as e:
            self._logger.error(f"[TS Qwen3 VL V3] Generation error: {e}", exc_info=True)
            output_text = f"ERROR: {e}"

        if unload_after_generation:
            self._unload_model(resolved_model_id, resolved_precision, resolved_attention)

        out_tensor = self._pil_to_tensor(processed_images)
        self._logger.info(f"[TS Qwen3 VL V3] output image shape: {tuple(out_tensor.shape)}")
        return (output_text, out_tensor)

    # ------------------------
    # Utilities
    # ------------------------
    def _log_versions(self):
        try:
            import importlib.metadata as metadata
        except Exception:
            return
        pkgs = ["torch", "transformers", "accelerate", "bitsandbytes", "flash-attn", "tiktoken"]
        self._logger.info("[TS Qwen3 VL V3] Dependency versions:")
        for p in pkgs:
            try:
                v = metadata.version(p)
                self._logger.info(f"[TS Qwen3 VL V3]   {p}: {v}")
            except Exception:
                self._logger.info(f"[TS Qwen3 VL V3]   {p}: not found")

    @classmethod
    def _load_presets(cls):
        presets_path = os.path.join(cls._CURRENT_DIR, "qwen_3_vl_presets.json")
        if not os.path.exists(presets_path):
            return {}, []
        data = {}
        try:
            with open(presets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
        except Exception as e:
            logging.getLogger("TS_Qwen3_VL_V3").warning(f"[TS Qwen3 VL V3] Preset load failed: {e}")
            data = {}
        return data, list(data.keys())

    @staticmethod
    def _tensor_to_pil_list(tensor):
        if tensor is None:
            return []
        images = []
        tensor = tensor.detach().cpu()
        for i in range(tensor.shape[0]):
            arr = np.clip(tensor[i].numpy(), 0.0, 1.0) * 255.0
            img = Image.fromarray(arr.astype(np.uint8))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        return images

    @staticmethod
    def _pil_to_tensor(pil_list):
        if not pil_list:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        tensors = []
        for img in pil_list:
            if img.mode != "RGB":
                img = img.convert("RGB")
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
            tensors.append(t)
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _resize_and_crop_image(image, max_size, multiple_of=32):
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        w, h = image.size
        tw, th = w - (w % multiple_of), h - (h % multiple_of)
        if tw == 0 or th == 0:
            return image
        l, t = (w - tw) / 2, (h - th) / 2
        return image.crop((l, t, l + tw, t + th))

    @classmethod
    def _is_bitsandbytes_available(cls):
        return importlib.util.find_spec("bitsandbytes") is not None

    @classmethod
    def _is_flash_attn_available(cls):
        if importlib.util.find_spec("flash_attn") is None:
            return False
        try:
            import importlib.metadata as metadata
            _ = metadata.version("flash_attn")
        except Exception:
            return False
        return True

    @staticmethod
    def _is_sdpa_available():
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")

    @staticmethod
    def _get_device():
        device = mm.get_torch_device()
        if isinstance(device, str):
            return torch.device(device)
        return device

    def _device_key(self):
        device = self._get_device()
        if device.type == "cuda":
            idx = device.index
            if idx is None:
                try:
                    idx = torch.cuda.current_device()
                except Exception:
                    idx = 0
            return f"cuda:{idx}"
        if device.type == "mps":
            return "mps"
        if device.type == "cpu":
            return "cpu"
        return str(device)

    def _device_from_map(self, device_map):
        cuda_indices = []
        mps_present = False
        for dev in device_map.values():
            if isinstance(dev, torch.device):
                if dev.type == "cuda":
                    cuda_indices.append(dev.index if dev.index is not None else 0)
                elif dev.type == "mps":
                    mps_present = True
                continue
            if isinstance(dev, int):
                cuda_indices.append(int(dev))
                continue
            if not isinstance(dev, str):
                continue
            if dev.startswith("cuda"):
                parts = dev.split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    cuda_indices.append(int(parts[1]))
                else:
                    cuda_indices.append(0)
            elif dev == "mps":
                mps_present = True
        if cuda_indices:
            return torch.device(f"cuda:{min(cuda_indices)}")
        if mps_present:
            return torch.device("mps")
        return None

    def _cuda_indices_from_map(self, device_map):
        indices = set()
        for dev in device_map.values():
            if isinstance(dev, torch.device):
                if dev.type == "cuda":
                    indices.add(dev.index if dev.index is not None else 0)
                continue
            if isinstance(dev, int):
                indices.add(int(dev))
                continue
            if not isinstance(dev, str):
                continue
            if dev.startswith("cuda"):
                parts = dev.split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    indices.add(int(parts[1]))
                else:
                    indices.add(0)
        return sorted(indices)

    def _select_input_device(self, model):
        device = None
        if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
            device = self._device_from_map(model.hf_device_map)
        if device is None:
            try:
                device = model.device
            except Exception:
                device = self._get_device()
        if isinstance(device, torch.device) and device.type == "meta":
            device = self._get_device()
        if torch.cuda.is_available() and getattr(device, "type", None) != "cuda":
            if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
                if any(
                    isinstance(d, (str, torch.device)) and str(d).startswith("cuda")
                    for d in model.hf_device_map.values()
                ):
                    device = self._device_from_map(model.hf_device_map)
        return device

    def _cuda_indices_for_rng(self, model, target_device):
        indices = []
        if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
            indices = self._cuda_indices_from_map(model.hf_device_map)
        if not indices and isinstance(target_device, torch.device) and target_device.type == "cuda":
            idx = target_device.index
            if idx is None:
                try:
                    idx = torch.cuda.current_device()
                except Exception:
                    idx = 0
            indices = [idx]
        return indices

    @staticmethod
    def _select_generator_device(target_device):
        if isinstance(target_device, torch.device) and target_device.type == "cuda":
            return target_device
        if isinstance(target_device, str) and target_device.startswith("cuda"):
            return torch.device(target_device)
        return torch.device("cpu")

    @staticmethod
    def _move_inputs_to_device(inputs, device):
        if device is None:
            return inputs
        if hasattr(inputs, "to"):
            try:
                return inputs.to(device)
            except Exception:
                pass
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(device)
        if isinstance(inputs, (list, tuple)):
            if all(isinstance(x, (int, float, bool)) for x in inputs):
                return torch.tensor(inputs, device=device)
            return type(inputs)(TS_Qwen3_VL_V3._move_inputs_to_device(x, device) for x in inputs)
        if isinstance(inputs, dict):
            return {k: TS_Qwen3_VL_V3._move_inputs_to_device(v, device) for k, v in inputs.items()}
        return inputs

    @staticmethod
    def _supports_generator(model):
        try:
            sig = inspect.signature(model.generate)
            return "generator" in sig.parameters
        except Exception:
            return False

    def _log_processing_device(self, stage, target_device, model, inputs):
        model_device = None
        try:
            model_device = str(model.device)
        except Exception:
            model_device = "unknown"
        input_devices = self._collect_tensor_devices(inputs)
        input_devices_str = ",".join(sorted(input_devices)) if input_devices else "none"
        target_str = str(target_device) if target_device is not None else "none"
        self._logger.info(
            f"[TS Qwen3 VL V3] Device ({stage}): target={target_str} model={model_device} inputs={input_devices_str}"
        )
        if torch.cuda.is_available() and "cpu" in input_devices and str(target_str).startswith("cuda"):
            self._logger.warning("[TS Qwen3 VL V3] Warning: some inputs remain on CPU while target is CUDA.")

    @staticmethod
    def _collect_tensor_devices(obj):
        devices = set()
        if isinstance(obj, torch.Tensor):
            devices.add(str(obj.device))
            return devices
        if hasattr(obj, "data") and isinstance(obj.data, dict):
            for v in obj.data.values():
                devices.update(TS_Qwen3_VL_V3._collect_tensor_devices(v))
            return devices
        if isinstance(obj, dict):
            for v in obj.values():
                devices.update(TS_Qwen3_VL_V3._collect_tensor_devices(v))
            return devices
        if isinstance(obj, (list, tuple)):
            for v in obj:
                devices.update(TS_Qwen3_VL_V3._collect_tensor_devices(v))
            return devices
        return devices

    @staticmethod
    def _model_has_cuda_device(model):
        if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
            for dev in model.hf_device_map.values():
                if isinstance(dev, torch.device) and dev.type == "cuda":
                    return True
                if isinstance(dev, str) and dev.startswith("cuda"):
                    return True
                if isinstance(dev, int):
                    return True
        try:
            return model.device.type == "cuda"
        except Exception:
            return False

    def _resolve_model_id(self, model_name, custom_model_id):
        if model_name == "Custom (manual)":
            custom = (custom_model_id or "").strip()
            if not custom:
                raise ValueError("Custom model id is empty.")
            return custom
        return model_name

    def _resolve_precision(self, precision, model_id):
        if precision != "auto":
            device = self._get_device()
            if precision in ("int4", "int8") and not self._is_bitsandbytes_available():
                self._logger.warning("[TS Qwen3 VL V3] bitsandbytes not found, forcing fp16/fp32.")
                return "fp16" if device.type == "cuda" else "fp32"
            if device.type != "cuda" and precision in ("fp16", "bf16", "int8", "int4"):
                self._logger.warning("[TS Qwen3 VL V3] CPU detected, forcing fp32.")
                return "fp32"
            return precision

        device = self._get_device()
        if device.type != "cuda":
            return "fp32"

        vram_gb = self._get_vram_gb()
        size_b = self._MODEL_SIZES_B.get(model_id, 4)
        bnb_ok = self._is_bitsandbytes_available()
        bf16_ok = torch.cuda.is_bf16_supported()

        if size_b >= 8:
            if vram_gb >= 20 and bf16_ok:
                return "bf16"
            if vram_gb >= 16:
                return "fp16"
            if vram_gb >= 10 and bnb_ok:
                return "int8"
            if bnb_ok:
                return "int4"
            return "fp16"
        if size_b >= 4:
            if vram_gb >= 12 and bf16_ok:
                return "bf16"
            if vram_gb >= 10:
                return "fp16"
            if vram_gb >= 6 and bnb_ok:
                return "int8"
            if bnb_ok:
                return "int4"
            return "fp16"
        if vram_gb >= 8 and bf16_ok:
            return "bf16"
        if vram_gb >= 6:
            return "fp16"
        if bnb_ok:
            return "int8"
        return "fp16"

    def _resolve_attention(self, attention_mode, precision):
        if attention_mode != "auto":
            device = self._get_device()
            if device.type != "cuda":
                self._logger.warning("[TS Qwen3 VL V3] Non-CUDA device, forcing eager attention.")
                return "eager"
            if attention_mode == "flash_attention_2" and (not self._is_flash_attn_available() or precision not in ("fp16", "bf16")):
                self._logger.warning("[TS Qwen3 VL V3] flash_attention_2 unavailable, falling back to sdpa/eager.")
                return "sdpa" if self._is_sdpa_available() else "eager"
            if attention_mode == "sdpa" and not self._is_sdpa_available():
                self._logger.warning("[TS Qwen3 VL V3] SDPA unavailable, forcing eager attention.")
                return "eager"
            return attention_mode

        device = self._get_device()
        if device.type != "cuda":
            return "eager"

        if precision in ("int4", "int8"):
            return "sdpa" if self._is_sdpa_available() else "eager"

        if self._is_flash_attn_available() and precision in ("fp16", "bf16"):
            device = self._get_device()
            major, _minor = torch.cuda.get_device_capability(device)
            if major >= 8:
                return "flash_attention_2"

        return "sdpa" if self._is_sdpa_available() else "eager"

    def _get_vram_gb(self):
        try:
            if torch.cuda.is_available():
                device = self._get_device()
                if device.type == "cuda":
                    props = torch.cuda.get_device_properties(device)
                else:
                    return 0.0
                return float(props.total_memory) / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def _dtype_from_precision(self, precision):
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return torch.float32

    def _get_pad_token_id(self, processor, model):
        pad_id = None
        try:
            pad_id = processor.tokenizer.pad_token_id
        except Exception:
            pad_id = None
        if pad_id is None:
            try:
                pad_id = processor.tokenizer.eos_token_id
            except Exception:
                pad_id = None
        if pad_id is None:
            try:
                pad_id = model.config.eos_token_id
            except Exception:
                pad_id = None
        return pad_id

    # ------------------------
    # Model loading & caching
    # ------------------------
    def _cache_key(self, model_id, precision, attention):
        return f"{model_id}|{precision}|{attention}|{self._device_key()}"

    def _get_cached(self, model_id, precision, attention):
        key = self._cache_key(model_id, precision, attention)
        cached = self._cache.get(key)
        if cached:
            self._touch_cache_key(key)
        return cached, key

    def _set_cached(self, key, model, processor):
        self._cache[key] = (model, processor)
        self._touch_cache_key(key)
        self._evict_cache_if_needed(exclude_key=key)

    def _unload_model(self, model_id, precision, attention):
        _cached, key = self._get_cached(model_id, precision, attention)
        self._unload_cached_key(key)

    def _touch_cache_key(self, key):
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)

    def _evict_cache_if_needed(self, exclude_key=None):
        while len(self._cache_order) > self._cache_max_items:
            oldest = self._cache_order[0]
            if exclude_key and oldest == exclude_key:
                if len(self._cache_order) == 1:
                    break
                oldest = self._cache_order[1]
            self._unload_cached_key(oldest)

    def _unload_cached_key(self, key):
        cached = self._cache.pop(key, None)
        if key in self._cache_order:
            self._cache_order.remove(key)
        if not cached:
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mm.soft_empty_cache()
        self._logger.info(f"[TS Qwen3 VL V3] Model unloaded: {key}")

    def _load_model(self, model_id, precision, attention_mode, offline_mode, hf_token, hf_endpoint):
        cached, key = self._get_cached(model_id, precision, attention_mode)
        if cached:
            return cached

        local_dir = self._ensure_model_available(model_id, offline_mode, hf_token, hf_endpoint)

        model_class, processor_class, bnb_config_cls = self._resolve_transformers_classes()
        device = self._get_device()

        load_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if attention_mode:
            load_kwargs["attn_implementation"] = attention_mode
        if offline_mode:
            load_kwargs["local_files_only"] = True

        if precision in ("int4", "int8"):
            if not self._is_bitsandbytes_available():
                raise RuntimeError(f"{precision} requires bitsandbytes.")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if precision == "int4":
                load_kwargs["quantization_config"] = bnb_config_cls(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True
                )
            else:
                load_kwargs["quantization_config"] = bnb_config_cls(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self._dtype_from_precision(precision)

        self._logger.info(f"[TS Qwen3 VL V3] Loading processor from {local_dir}")
        processor = processor_class.from_pretrained(local_dir, trust_remote_code=True, local_files_only=offline_mode)

        self._logger.info(f"[TS Qwen3 VL V3] Loading model from {local_dir}")
        def _try_load():
            try:
                return model_class.from_pretrained(local_dir, **load_kwargs)
            except TypeError as e:
                if "attn_implementation" in str(e) and "attn_implementation" in load_kwargs:
                    self._logger.warning("[TS Qwen3 VL V3] attn_implementation unsupported, retrying without it.")
                    load_kwargs.pop("attn_implementation", None)
                    return model_class.from_pretrained(local_dir, **load_kwargs)
                raise

        try:
            model = _try_load()
        except RuntimeError as e:
            if self._is_oom_error(e):
                self._logger.warning("[TS Qwen3 VL V3] OOM during load, retrying after cache cleanup.")
                self._prepare_memory(force=True)
                model = _try_load()
            else:
                raise
        if precision not in ("int4", "int8"):
            model.to(device)
        model.eval()

        if torch.cuda.is_available() and not self._model_has_cuda_device(model):
            self._logger.warning("[TS Qwen3 VL V3] CUDA available but model is on CPU; generation will run on CPU.")

        self._set_cached(key, model, processor)
        return model, processor

    def _prepare_memory(self, force=False):
        if not force:
            return
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _is_oom_error(exc):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg

    def _resolve_transformers_classes(self):
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        except Exception:
            try:
                from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
                return Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            except Exception:
                from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM
                return AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    # ------------------------
    # Download & integrity
    # ------------------------
    def _ensure_model_available(self, model_id, offline_mode, hf_token, hf_endpoint):
        models_dir = os.path.join(folder_paths.models_dir, "LLM")
        repo_name = model_id.split("/")[-1]
        local_dir = os.path.join(models_dir, repo_name)

        if offline_mode:
            if not self._check_model_integrity(local_dir):
                raise FileNotFoundError(f"Offline mode: model not found or incomplete at {local_dir}.")
            return local_dir

        if not self._check_model_integrity(local_dir):
            self._download_with_mirrors(model_id, local_dir, hf_token, hf_endpoint)
            if not self._check_model_integrity(local_dir):
                raise RuntimeError("Model download completed but integrity check failed.")

        return local_dir

    def _download_with_mirrors(self, model_id, local_dir, hf_token, hf_endpoint):
        endpoints = [e.strip() for e in (hf_endpoint or "").split(",") if e.strip()]
        if not endpoints:
            endpoints = ["https://huggingface.co"]

        token = hf_token.strip() if hf_token and hf_token.strip() else None

        last_error = None
        for i, endpoint in enumerate(endpoints):
            endpoint_url = endpoint
            if not endpoint_url.startswith("http://") and not endpoint_url.startswith("https://"):
                endpoint_url = "https://" + endpoint_url
            self._logger.info(f"[TS Qwen3 VL V3] Download attempt {i + 1}/{len(endpoints)}: {endpoint_url}")

            try:
                self._snapshot_download(model_id, local_dir, token, endpoint_url)
                self._logger.info("[TS Qwen3 VL V3] Download completed.")
                return
            except Exception as e:
                last_error = e
                self._logger.warning(f"[TS Qwen3 VL V3] Download failed: {e}")
                if self._snapshot_endpoint_supported is False:
                    break

        raise RuntimeError(f"All mirrors failed. Last error: {last_error}")

    def _snapshot_download(self, model_id, local_dir, token, endpoint_url):
        kwargs = {
            "repo_id": model_id,
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "token": token
        }
        if endpoint_url:
            kwargs["endpoint"] = endpoint_url
        while True:
            try:
                snapshot_download(**kwargs)
                if "endpoint" in kwargs:
                    self._snapshot_endpoint_supported = True
                return
            except TypeError as e:
                msg = str(e)
                if "resume_download" in msg and "resume_download" in kwargs:
                    kwargs.pop("resume_download", None)
                    continue
                if "endpoint" in msg and "endpoint" in kwargs:
                    kwargs.pop("endpoint", None)
                    self._snapshot_endpoint_supported = False
                    self._logger.warning("[TS Qwen3 VL V3] snapshot_download endpoint unsupported; using default hub.")
                    continue
                raise

    def _check_model_integrity(self, local_dir):
        if not os.path.isdir(local_dir):
            return False

        required_any = [
            "config.json"
        ]
        for fname in required_any:
            if not os.path.exists(os.path.join(local_dir, fname)):
                return False

        tokenizer_ok = (
            os.path.exists(os.path.join(local_dir, "tokenizer.json")) or
            os.path.exists(os.path.join(local_dir, "tokenizer.model")) or
            os.path.exists(os.path.join(local_dir, "tokenizer_config.json"))
        )
        if not tokenizer_ok:
            return False

        processor_ok = (
            os.path.exists(os.path.join(local_dir, "preprocessor_config.json")) or
            os.path.exists(os.path.join(local_dir, "processor_config.json"))
        )
        if not processor_ok:
            return False

        index_files = [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json"
        ]
        for idx in index_files:
            idx_path = os.path.join(local_dir, idx)
            if os.path.exists(idx_path):
                try:
                    with open(idx_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    weight_map = data.get("weight_map", {})
                    for shard in set(weight_map.values()):
                        shard_path = os.path.join(local_dir, shard)
                        if not os.path.exists(shard_path) or os.path.getsize(shard_path) == 0:
                            return False
                    return True
                except Exception:
                    return False

        single_ok = (
            os.path.exists(os.path.join(local_dir, "model.safetensors")) or
            os.path.exists(os.path.join(local_dir, "pytorch_model.bin"))
        )
        return single_ok

NODE_CLASS_MAPPINGS = {"TS_Qwen3_VL_V3": TS_Qwen3_VL_V3}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Qwen3_VL_V3": "TS Qwen 3 VL V3"}
