"""Shared Qwen runtime used by TS_Qwen3_VL_V3 and TS_SuperPrompt.

Owns the heavy bits both LLM nodes need: capability detection (bitsandbytes /
flash_attn / sdpa / multimodal processor), tensor↔PIL helpers, device + CUDA
selection, VRAM/precision/attention resolution, model + processor loading with
HuggingFace download orchestration, an LRU model cache, and chat-template +
decode helpers. Each node holds a reference to the same process-wide
``QwenEngine`` instance through ``get_qwen_engine()`` so model files and GPU
weights are loaded at most once per process.

Private — loader skips paths with `_`-prefixed components.
"""

from __future__ import annotations

import gc
import importlib.util
import inspect
import json
import logging
import os
import re
from typing import Any

import comfy.model_management as mm
import folder_paths
import numpy as np
import torch
from PIL import Image


_DEFAULT_LOG_PREFIX = "[TS Qwen Engine]"
_DEFAULT_LOGGER_NAME = "comfyui_timesaver.qwen_engine"


# Explicit model sizes (in billions of parameters). Used by ``model_size_b``
# as the authoritative source; anything missing (e.g. user-supplied
# ``Custom (manual)`` ids) falls back to a regex match on the repo name
# (``…-2B``, ``…-4B``, etc.).
_MODEL_SIZES_B: dict[str, float] = {
    "huihui-ai/Huihui-Qwen3.5-2B-abliterated": 2.0,
    "huihui-ai/Huihui-Qwen3.5-4B-abliterated": 4.0,
    "huihui-ai/Huihui-Qwen3.5-9B-abliterated": 9.0,
}


class QwenEngine:
    """Process-wide Qwen runtime: model cache + helpers."""

    def __init__(
        self,
        log_prefix: str = _DEFAULT_LOG_PREFIX,
        logger_name: str = _DEFAULT_LOGGER_NAME,
        cache_max_items: int = 1,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._log_prefix = log_prefix
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._cache_order: list[str] = []
        self._cache_max_items = max(1, int(cache_max_items))
        self._snapshot_endpoint_supported: bool | None = None
        self._optimizations_applied = False

    # ------------------------------------------------------------------
    # Runtime optimisations (Qwen 3.5 inference path)
    # ------------------------------------------------------------------
    def apply_runtime_optimizations(self) -> None:
        """Enable matmul + attention backends that benefit Qwen 3.5 on CUDA.

        Idempotent — paid once per process. Wrapped in best-effort try/except
        so older PyTorch builds that don't expose every backend toggle still
        load the model (the corresponding optimisation is just skipped).

        Tuned for Qwen 3.5 dense-LLM inference under bf16/fp16 + SDPA:

        * TF32 on Ampere+: ``cuda.matmul`` and ``cudnn`` allow_tf32 give a
          measurable speed-up for fp32 ops without affecting bf16/fp16
          accuracy meaningfully.
        * ``set_float32_matmul_precision('high')`` aliases the same TF32
          path for any nn.Linear-style call that PyTorch routes through
          its matmul precision flag.
        * Reduced-precision fp16 reductions: lets the matmul kernels keep
          their accumulator in fp16 for a small bandwidth win on Ampere.
        * SDPA backends: explicitly enable flash + mem-efficient + math
          backends so PyTorch picks the best one (flash for sm_80+ /
          bf16/fp16, mem-efficient otherwise, math as final fallback).
        """
        if self._optimizations_applied:
            return
        self._optimizations_applied = True

        def _try(setter, name):
            try:
                setter()
                self._logger.debug("%s opt: %s enabled", self._log_prefix, name)
            except Exception as exc:  # noqa: BLE001 — best-effort optimisation
                self._logger.debug("%s opt: %s skipped (%s)", self._log_prefix, name, exc)

        if not torch.cuda.is_available():
            self._logger.debug("%s Skipping CUDA opts: no GPU detected.", self._log_prefix)
            return

        _try(lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", True), "cuda.matmul.allow_tf32")
        _try(lambda: setattr(torch.backends.cudnn, "allow_tf32", True), "cudnn.allow_tf32")
        _try(
            lambda: torch.set_float32_matmul_precision("high"),
            "set_float32_matmul_precision",
        )
        _try(
            lambda: setattr(
                torch.backends.cuda.matmul,
                "allow_fp16_reduced_precision_reduction",
                True,
            ),
            "cuda.matmul.allow_fp16_reduced_precision_reduction",
        )
        _try(lambda: torch.backends.cuda.enable_flash_sdp(True), "enable_flash_sdp")
        _try(
            lambda: torch.backends.cuda.enable_mem_efficient_sdp(True),
            "enable_mem_efficient_sdp",
        )
        _try(lambda: torch.backends.cuda.enable_math_sdp(True), "enable_math_sdp")
        self._logger.info("%s Qwen 3.5 runtime optimizations applied (TF32 + SDP).", self._log_prefix)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    @staticmethod
    def is_bitsandbytes_available() -> bool:
        return importlib.util.find_spec("bitsandbytes") is not None

    @staticmethod
    def is_flash_attn_available() -> bool:
        if importlib.util.find_spec("flash_attn") is None:
            return False
        try:
            import importlib.metadata as metadata

            _ = metadata.version("flash_attn")
        except Exception:
            return False
        return True

    @staticmethod
    def is_sdpa_available() -> bool:
        return hasattr(torch.nn.functional, "scaled_dot_product_attention")

    @staticmethod
    def supports_multimodal_inputs(processor) -> bool:
        tokenizer = QwenEngine.get_tokenizer_from_processor(processor)
        if hasattr(processor, "image_processor") or hasattr(processor, "video_processor"):
            return True
        if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "video_processor"):
            return True
        return False

    @staticmethod
    def supports_generator(model) -> bool:
        try:
            sig = inspect.signature(model.generate)
            return "generator" in sig.parameters
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Tensor / PIL conversion
    # ------------------------------------------------------------------
    @staticmethod
    def tensor_to_pil_list(tensor) -> list[Image.Image]:
        if tensor is None:
            return []
        images: list[Image.Image] = []
        tensor = tensor.detach().cpu()
        for i in range(tensor.shape[0]):
            arr = np.clip(tensor[i].numpy(), 0.0, 1.0) * 255.0
            img = Image.fromarray(arr.astype(np.uint8))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        return images

    @staticmethod
    def pil_to_tensor(pil_list) -> torch.Tensor:
        if not pil_list:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        tensors = []
        for img in pil_list:
            if img.mode != "RGB":
                img = img.convert("RGB")
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
            tensors.append(t)
        return torch.cat(tensors, dim=0)

    @classmethod
    def normalize_to_pil_list(cls, image) -> list[Image.Image]:
        """Accept tensor / PIL.Image / iterable of either and return a PIL list.

        ComfyUI ``IMAGE`` inputs arrive as ``torch.Tensor`` (B, H, W, C). The
        TS Super Prompt frontend uploads via ``/upload/image`` and the
        backend resolves the annotated path into a ``PIL.Image`` directly, so
        the same helper has to cope with both shapes.
        """
        if image is None:
            return []
        # PIL.Image: has .convert and .size attrs.
        if hasattr(image, "convert") and hasattr(image, "size") and not hasattr(image, "shape"):
            pil = image if image.mode == "RGB" else image.convert("RGB")
            return [pil]
        # torch.Tensor: has .detach + .shape.
        if hasattr(image, "detach") and hasattr(image, "shape"):
            return cls.tensor_to_pil_list(image)
        # Iterable of either.
        if isinstance(image, (list, tuple)):
            collected: list[Image.Image] = []
            for item in image:
                collected.extend(cls.normalize_to_pil_list(item))
            return collected
        return []

    @staticmethod
    def resize_and_crop_image(image: Image.Image, max_size: int, multiple_of: int = 32) -> Image.Image:
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

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    @staticmethod
    def get_device() -> torch.device:
        device = mm.get_torch_device()
        if isinstance(device, str):
            return torch.device(device)
        return device

    def device_key(self) -> str:
        device = self.get_device()
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

    def device_from_map(self, device_map: dict) -> torch.device | None:
        cuda_indices: list[int] = []
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

    def cuda_indices_from_map(self, device_map: dict) -> list[int]:
        indices: set[int] = set()
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

    def select_input_device(self, model) -> torch.device | None:
        device: torch.device | None = None
        if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
            device = self.device_from_map(model.hf_device_map)
        if device is None:
            try:
                device = model.device
            except Exception:
                device = self.get_device()
        if isinstance(device, torch.device) and device.type == "meta":
            device = self.get_device()
        if torch.cuda.is_available() and getattr(device, "type", None) != "cuda":
            if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
                if any(
                    isinstance(d, (str, torch.device)) and str(d).startswith("cuda")
                    for d in model.hf_device_map.values()
                ):
                    device = self.device_from_map(model.hf_device_map)
        return device

    def cuda_indices_for_rng(self, model, target_device) -> list[int]:
        indices: list[int] = []
        if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
            indices = self.cuda_indices_from_map(model.hf_device_map)
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
    def select_generator_device(target_device) -> torch.device:
        if isinstance(target_device, torch.device) and target_device.type == "cuda":
            return target_device
        if isinstance(target_device, str) and target_device.startswith("cuda"):
            return torch.device(target_device)
        return torch.device("cpu")

    def move_inputs_to_device(self, inputs, device):
        if device is None:
            return inputs
        if hasattr(inputs, "to"):
            try:
                return inputs.to(device)
            except Exception as exc:
                self._logger.debug(
                    "%s inputs.to(%s) failed, falling back to type-specific dispatch: %s",
                    self._log_prefix,
                    device,
                    exc,
                )
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).to(device)
        if isinstance(inputs, (list, tuple)):
            if all(isinstance(x, (int, float, bool)) for x in inputs):
                return torch.tensor(inputs, device=device)
            return type(inputs)(self.move_inputs_to_device(x, device) for x in inputs)
        if isinstance(inputs, dict):
            return {k: self.move_inputs_to_device(v, device) for k, v in inputs.items()}
        return inputs

    @staticmethod
    def collect_tensor_devices(obj) -> set[str]:
        devices: set[str] = set()
        if isinstance(obj, torch.Tensor):
            devices.add(str(obj.device))
            return devices
        if hasattr(obj, "data") and isinstance(obj.data, dict):
            for v in obj.data.values():
                devices.update(QwenEngine.collect_tensor_devices(v))
            return devices
        if isinstance(obj, dict):
            for v in obj.values():
                devices.update(QwenEngine.collect_tensor_devices(v))
            return devices
        if isinstance(obj, (list, tuple)):
            for v in obj:
                devices.update(QwenEngine.collect_tensor_devices(v))
            return devices
        return devices

    def log_processing_device(self, stage: str, target_device, model, inputs) -> None:
        try:
            model_device = str(model.device)
        except Exception:
            model_device = "unknown"
        input_devices = self.collect_tensor_devices(inputs)
        input_devices_str = ",".join(sorted(input_devices)) if input_devices else "none"
        target_str = str(target_device) if target_device is not None else "none"
        self._logger.info(
            "%s Device (%s): target=%s model=%s inputs=%s",
            self._log_prefix,
            stage,
            target_str,
            model_device,
            input_devices_str,
        )

    @staticmethod
    def model_has_cuda_device(model) -> bool:
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

    # ------------------------------------------------------------------
    # Precision / attention / VRAM
    # ------------------------------------------------------------------
    @classmethod
    def model_size_b(cls, model_id: str) -> float:
        explicit = _MODEL_SIZES_B.get(model_id)
        if explicit:
            return float(explicit)
        repo_name = (model_id or "").split("/")[-1]
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[bB](?![a-zA-Z])", repo_name)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                pass
        return 4.0

    def get_vram_gb(self) -> float:
        try:
            if torch.cuda.is_available():
                device = self.get_device()
                if device.type == "cuda":
                    props = torch.cuda.get_device_properties(device)
                    return float(props.total_memory) / (1024 ** 3)
        except Exception as exc:
            self._logger.debug("%s VRAM probe failed: %s", self._log_prefix, exc)
        return 0.0

    def resolve_precision(self, precision: str, model_id: str) -> str:
        if precision != "auto":
            device = self.get_device()
            if precision in ("int4", "int8") and not self.is_bitsandbytes_available():
                self._logger.warning("%s bitsandbytes not found, forcing fp16/fp32.", self._log_prefix)
                return "fp16" if device.type == "cuda" else "fp32"
            if device.type != "cuda" and precision in ("fp16", "bf16", "int8", "int4"):
                self._logger.warning("%s CPU detected, forcing fp32.", self._log_prefix)
                return "fp32"
            return precision

        device = self.get_device()
        if device.type != "cuda":
            return "fp32"

        vram_gb = self.get_vram_gb()
        size_b = self.model_size_b(model_id)
        bnb_ok = self.is_bitsandbytes_available()
        bf16_ok = torch.cuda.is_bf16_supported()

        if size_b >= 8:
            if vram_gb >= 22 and bf16_ok:
                return "bf16"
            if vram_gb >= 20:
                return "fp16"
            if vram_gb >= 12 and bnb_ok:
                return "int8"
            if bnb_ok:
                return "int4"
            return "fp16"
        if size_b >= 4:
            if vram_gb >= 12 and bf16_ok:
                return "bf16"
            if vram_gb >= 10:
                return "fp16"
            if vram_gb >= 8 and bnb_ok:
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

    def resolve_attention(self, attention_mode: str, precision: str) -> str:
        if attention_mode != "auto":
            device = self.get_device()
            if device.type != "cuda":
                return "eager"
            if attention_mode == "flash_attention_2" and (
                not self.is_flash_attn_available() or precision not in ("fp16", "bf16")
            ):
                return "sdpa" if self.is_sdpa_available() else "eager"
            if attention_mode == "sdpa" and not self.is_sdpa_available():
                return "eager"
            return attention_mode

        device = self.get_device()
        if device.type != "cuda":
            return "eager"

        if precision in ("int4", "int8"):
            return "sdpa" if self.is_sdpa_available() else "eager"

        if self.is_flash_attn_available() and precision in ("fp16", "bf16"):
            major, _minor = torch.cuda.get_device_capability(device)
            if major >= 8:
                return "flash_attention_2"

        return "sdpa" if self.is_sdpa_available() else "eager"

    @staticmethod
    def dtype_from_precision(precision: str) -> torch.dtype:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    def estimate_vram_usage(self, model_id: str, precision: str) -> float:
        size_b = self.model_size_b(model_id)
        if precision == "fp32":
            bytes_per_param = 4.0
        elif precision == "auto":
            bytes_per_param = 2.2
        elif precision in ("fp16", "bf16"):
            bytes_per_param = 2.2
        elif precision == "int8":
            bytes_per_param = 1.2
        elif precision == "int4":
            bytes_per_param = 0.8
        else:
            bytes_per_param = 2.2

        weights_gb = size_b * bytes_per_param
        context_overhead_gb = 1.5 if size_b < 8 else 2.5
        return weights_gb + context_overhead_gb

    def ensure_memory_available(self, required_vram_gb: float, force_unload: bool = False) -> None:
        if not torch.cuda.is_available():
            return
        try:
            mm.soft_empty_cache()
            free_mem_gb = mm.get_free_memory() / (1024 ** 3)
            self._logger.info(
                "%s Memory Check: Required=%.2fGB, Free=%.2fGB",
                self._log_prefix,
                required_vram_gb,
                free_mem_gb,
            )
            if force_unload or free_mem_gb < required_vram_gb:
                self._logger.info(
                    "%s Low VRAM detected (or forced). Unloading ComfyUI models...",
                    self._log_prefix,
                )
                mm.unload_all_models()
                mm.soft_empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                free_mem_gb = mm.get_free_memory() / (1024 ** 3)
                self._logger.info("%s Memory Post-Clean: Free=%.2fGB", self._log_prefix, free_mem_gb)
        except Exception as exc:
            self._logger.warning("%s Memory management warning: %s", self._log_prefix, exc)

    def prepare_memory(self, force: bool = False) -> None:
        if not force:
            return
        try:
            mm.soft_empty_cache()
        except Exception as exc:
            self._logger.debug("%s mm.soft_empty_cache() failed: %s", self._log_prefix, exc)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def is_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg

    # ------------------------------------------------------------------
    # Chat template / decode
    # ------------------------------------------------------------------
    @staticmethod
    def get_tokenizer_from_processor(processor):
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        return processor

    def get_pad_token_id(self, processor, model):
        tokenizer = self.get_tokenizer_from_processor(processor)
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return value
        config = getattr(model, "config", None)
        if config is not None:
            value = getattr(config, "eos_token_id", None)
            if value is not None:
                return value
        return None

    def apply_chat_template(self, processor, messages, **overrides):
        template_fn = getattr(processor, "apply_chat_template", None)
        if template_fn is None:
            tokenizer = self.get_tokenizer_from_processor(processor)
            template_fn = getattr(tokenizer, "apply_chat_template", None)
        if template_fn is None:
            raise RuntimeError("Loaded processor/tokenizer does not provide apply_chat_template.")
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        kwargs.update(overrides)
        return template_fn(messages, **kwargs)

    def apply_chat_template_no_thinking(self, processor, messages):
        """``apply_chat_template`` with ``enable_thinking=False`` baked in.

        Qwen 3 / 3.5 ship a built-in "thinking" mode that emits hidden
        ``<think>…</think>`` blocks before the answer. For prompt-engineering
        and prompt-enhance flows we want a direct answer with no chain of
        thought leaking into the output, so this helper probes the various
        kwarg shapes Qwen processors accept (``enable_thinking=False`` on the
        function, or wrapped inside ``chat_template_kwargs``) and falls back
        to a plain call if neither is supported. Also tries a text-flattened
        version of the messages for tokenizers that reject list-content.

        Used by both ``TS_Qwen3_VL_V3`` and the ``TS_SuperPrompt`` enhance
        route so both nodes share the same "no chain-of-thought" guarantee.
        """
        return apply_chat_template_no_thinking(self, processor, messages)

    @staticmethod
    def strip_thinking_block(text: str) -> str:
        """Remove any ``<think>...</think>`` segments from generated text.

        Even with ``enable_thinking=False`` Qwen sometimes emits a partial
        thinking block (e.g. when the chat template overrides our kwarg).
        This is a defensive postprocess so callers never see chain-of-thought.
        """
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            str(text or ""),
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()

    def batch_decode(
        self,
        processor,
        token_ids,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        decode_fn = getattr(processor, "batch_decode", None)
        if decode_fn is None:
            tokenizer = self.get_tokenizer_from_processor(processor)
            decode_fn = getattr(tokenizer, "batch_decode", None)
        if decode_fn is None:
            raise RuntimeError("Loaded processor/tokenizer does not provide batch_decode.")
        return decode_fn(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    # ------------------------------------------------------------------
    # Model loading + LRU cache
    # ------------------------------------------------------------------
    def _cache_key(self, model_id: str, precision: str, attention: str) -> str:
        return f"{model_id}|{precision}|{attention}|{self.device_key()}"

    def _get_cached(self, model_id: str, precision: str, attention: str) -> tuple[tuple[Any, Any] | None, str]:
        key = self._cache_key(model_id, precision, attention)
        cached = self._cache.get(key)
        if cached:
            self._touch_cache_key(key)
        return cached, key

    def _set_cached(self, key: str, model, processor) -> None:
        self._cache[key] = (model, processor)
        self._touch_cache_key(key)
        self._evict_cache_if_needed(exclude_key=key)

    def unload_model(self, model_id: str, precision: str, attention: str) -> None:
        _cached, key = self._get_cached(model_id, precision, attention)
        self._unload_cached_key(key)

    def _touch_cache_key(self, key: str) -> None:
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)

    def _evict_cache_if_needed(self, exclude_key: str | None = None) -> None:
        while len(self._cache_order) > self._cache_max_items:
            oldest = self._cache_order[0]
            if exclude_key and oldest == exclude_key:
                if len(self._cache_order) == 1:
                    break
                oldest = self._cache_order[1]
            self._unload_cached_key(oldest)

    def _unload_cached_key(self, key: str) -> None:
        cached = self._cache.pop(key, None)
        if key in self._cache_order:
            self._cache_order.remove(key)
        if not cached:
            return
        del cached
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mm.soft_empty_cache()
        self._logger.info("%s Model unloaded: %s", self._log_prefix, key)

    def load_model(
        self,
        model_id: str,
        precision: str,
        attention_mode: str,
        offline_mode: bool,
        hf_token: str,
        hf_endpoint: str,
    ):
        cached, key = self._get_cached(model_id, precision, attention_mode)
        if cached:
            return cached

        # Lazy: pay the optimisation toggles cost once, right before the
        # first model actually lands on the GPU.
        self.apply_runtime_optimizations()

        local_dir = self.ensure_model_available(model_id, offline_mode, hf_token, hf_endpoint)
        transformers_module, bnb_config_cls = self._resolve_transformers_runtime()
        device = self.get_device()

        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if attention_mode:
            load_kwargs["attn_implementation"] = attention_mode
        if offline_mode:
            load_kwargs["local_files_only"] = True

        self.prepare_memory(force=True)

        if precision in ("int4", "int8"):
            if not self.is_bitsandbytes_available():
                raise RuntimeError(f"{precision} requires bitsandbytes.")
            if bnb_config_cls is None:
                raise RuntimeError("BitsAndBytesConfig not found in transformers installation.")
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if precision == "int4":
                load_kwargs["quantization_config"] = bnb_config_cls(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["quantization_config"] = bnb_config_cls(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self.dtype_from_precision(precision)
            # Skip the CPU→GPU staging copy for non-quantised loads on CUDA:
            # ``device_map={"": device}`` asks accelerate to materialise every
            # submodule directly on the target GPU. Halves the transient peak
            # RAM usage and shaves the time spent on ``model.to(cuda)``.
            if device.type == "cuda":
                load_kwargs["device_map"] = {"": device}

        self._logger.info("%s Loading processor from %s", self._log_prefix, local_dir)
        processor = self._load_processor_or_tokenizer(transformers_module, local_dir, offline_mode)

        self._logger.info("%s Loading model from %s", self._log_prefix, local_dir)
        model = None
        model_load_errors: list[str] = []
        for loader_name, model_class in self._resolve_model_loader_candidates(transformers_module, local_dir):
            try:
                self._logger.info("%s Trying loader: %s", self._log_prefix, loader_name)
                model = self._load_model_with_loader(model_class, local_dir, load_kwargs)
                self._logger.info("%s Loaded with: %s", self._log_prefix, loader_name)
                break
            except Exception as exc:
                msg = str(exc)
                model_load_errors.append(f"{loader_name}: {msg}")
                self._logger.warning(
                    "%s Loader failed (%s): %s", self._log_prefix, loader_name, msg
                )
                continue

        if model is None:
            tail = "; ".join(model_load_errors[-3:]) if model_load_errors else "no candidates"
            raise RuntimeError(f"Unable to load model {model_id}. Last loader errors: {tail}")

        if precision not in ("int4", "int8") and device.type != "cuda":
            # MPS / CPU path still needs an explicit move; CUDA + non-quant
            # is already placed by device_map above.
            model.to(device)

        model.eval()
        self._set_cached(key, model, processor)
        return model, processor

    def _resolve_transformers_runtime(self):
        import transformers

        bnb_config_cls = getattr(transformers, "BitsAndBytesConfig", None)
        return transformers, bnb_config_cls

    def _resolve_model_loader_candidates(self, transformers_module, local_dir):
        candidates: list[tuple[str, Any]] = []
        seen: set[str] = set()

        def _add(class_name: str) -> None:
            if not class_name or class_name in seen:
                return
            klass = getattr(transformers_module, class_name, None)
            if klass is None or not hasattr(klass, "from_pretrained"):
                return
            seen.add(class_name)
            candidates.append((class_name, klass))

        for architecture_name in self._read_model_architectures(local_dir):
            _add(architecture_name)

        major = self._transformers_major_version(transformers_module)
        if major >= 5:
            ordered = [
                "AutoModelForImageTextToText",
                "AutoModelForCausalLM",
                "AutoModel",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
                "AutoModelForVision2Seq",
            ]
        else:
            ordered = [
                "Qwen3_5ForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
                "AutoModelForCausalLM",
                "AutoModel",
            ]
        for class_name in ordered:
            _add(class_name)
        return candidates

    def _load_model_with_loader(self, model_class, local_dir, load_kwargs):
        kwargs_for_load = dict(load_kwargs)

        def _run_load(kwargs: dict[str, Any]):
            try:
                return model_class.from_pretrained(local_dir, **kwargs)
            except TypeError as exc:
                if "attn_implementation" in str(exc) and "attn_implementation" in kwargs:
                    self._logger.warning(
                        "%s attn_implementation unsupported, retrying without it.",
                        self._log_prefix,
                    )
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("attn_implementation", None)
                    return model_class.from_pretrained(local_dir, **retry_kwargs)
                raise

        try:
            return _run_load(kwargs_for_load)
        except RuntimeError as exc:
            if not self.is_oom_error(exc):
                raise
            self._logger.warning(
                "%s OOM during load, retrying after AGGRESSIVE cleanup.", self._log_prefix
            )
            mm.unload_all_models()
            self.prepare_memory(force=True)
            return _run_load(kwargs_for_load)

    def _load_processor_or_tokenizer(self, transformers_module, local_dir, offline_mode):
        common_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if offline_mode:
            common_kwargs["local_files_only"] = True

        processor_cls = getattr(transformers_module, "AutoProcessor", None)
        if processor_cls is not None:
            try:
                return processor_cls.from_pretrained(local_dir, **common_kwargs)
            except Exception as exc:
                self._logger.warning("%s AutoProcessor load failed: %s", self._log_prefix, exc)

        tokenizer_cls = getattr(transformers_module, "AutoTokenizer", None)
        if tokenizer_cls is not None:
            tokenizer = tokenizer_cls.from_pretrained(local_dir, **common_kwargs)
            self._logger.warning(
                "%s Falling back to AutoTokenizer (text-only features).", self._log_prefix
            )
            return tokenizer

        raise RuntimeError("Neither AutoProcessor nor AutoTokenizer is available in transformers.")

    @staticmethod
    def _read_model_architectures(local_dir: str) -> list[str]:
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(config_path):
            return []
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            architectures = data.get("architectures", [])
            if isinstance(architectures, list):
                return [a for a in architectures if isinstance(a, str) and a]
        except Exception:
            return []
        return []

    @staticmethod
    def _transformers_major_version(transformers_module) -> int:
        version = str(getattr(transformers_module, "__version__", "0"))
        match = re.match(r"\s*(\d+)", version)
        if not match:
            return 0
        try:
            return int(match.group(1))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # HuggingFace download
    # ------------------------------------------------------------------
    def ensure_model_available(
        self,
        model_id: str,
        offline_mode: bool,
        hf_token: str,
        hf_endpoint: str,
    ) -> str:
        models_dir = os.path.join(folder_paths.models_dir, "LLM")
        repo_name = model_id.split("/")[-1]
        local_dir = os.path.join(models_dir, repo_name)

        if offline_mode:
            if not self.check_model_integrity(local_dir):
                raise FileNotFoundError(f"Offline mode: model not found or incomplete at {local_dir}.")
            return local_dir

        if not self.check_model_integrity(local_dir):
            self._download_with_mirrors(model_id, local_dir, hf_token, hf_endpoint)
            if not self.check_model_integrity(local_dir):
                raise RuntimeError("Model download completed but integrity check failed.")
        return local_dir

    def _download_with_mirrors(
        self,
        model_id: str,
        local_dir: str,
        hf_token: str,
        hf_endpoint: str,
    ) -> None:
        """Try each HF endpoint in order; bail fast on hub-incompatibility.

        Failure handling is split deliberately:

        * ``TypeError`` propagated from ``_snapshot_download`` means the
          installed ``huggingface_hub`` does not accept our kwargs at all.
          ``_snapshot_download`` already retries-without-kwarg for the known
          ``resume_download``/``endpoint`` incompatibilities; anything still
          surfacing here is a hub-version mismatch that will fail identically
          on every mirror, so we ``break`` immediately with a clear error.
        * Anything else (network, 4xx, repo-not-found, timeout) is treated as
          transient/mirror-specific and we move on to the next endpoint.
        """

        endpoints = [e.strip() for e in (hf_endpoint or "").split(",") if e.strip()]
        if not endpoints:
            endpoints = ["https://huggingface.co"]
        token = hf_token.strip() if hf_token and hf_token.strip() else None
        last_error: Exception | None = None

        for i, endpoint in enumerate(endpoints):
            endpoint_url = endpoint
            if not endpoint_url.startswith(("http://", "https://")):
                endpoint_url = "https://" + endpoint_url
            self._logger.info(
                "%s Download attempt %d/%d: %s",
                self._log_prefix,
                i + 1,
                len(endpoints),
                endpoint_url,
            )
            try:
                self._snapshot_download(model_id, local_dir, token, endpoint_url)
                self._logger.info("%s Download completed.", self._log_prefix)
                return
            except TypeError as exc:
                # Hub-version mismatch — switching mirrors won't help.
                last_error = exc
                self._logger.warning(
                    "%s huggingface_hub rejected our kwargs; aborting mirror cycle: %s",
                    self._log_prefix,
                    exc,
                )
                break
            except Exception as exc:  # noqa: BLE001 — transient errors → next mirror
                last_error = exc
                self._logger.warning(
                    "%s Download failed on %s: %s. Trying next mirror.",
                    self._log_prefix,
                    endpoint_url,
                    exc,
                )
                continue

        raise RuntimeError(f"All mirrors failed. Last error: {last_error}")

    def _snapshot_download(
        self,
        model_id: str,
        local_dir: str,
        token: str | None,
        endpoint_url: str,
    ) -> None:
        # Lazy: huggingface_hub pulls in HTTP/auth/cache infra. Load only when
        # actually downloading.
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "revision": "main",
            "local_dir": local_dir,
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "token": token,
        }
        if endpoint_url:
            kwargs["endpoint"] = endpoint_url
        while True:
            try:
                snapshot_download(**kwargs)
                if "endpoint" in kwargs:
                    self._snapshot_endpoint_supported = True
                return
            except TypeError as exc:
                msg = str(exc)
                if "resume_download" in msg and "resume_download" in kwargs:
                    kwargs.pop("resume_download", None)
                    continue
                if "endpoint" in msg and "endpoint" in kwargs:
                    kwargs.pop("endpoint", None)
                    self._snapshot_endpoint_supported = False
                    self._logger.warning(
                        "%s snapshot_download endpoint unsupported; using default hub.",
                        self._log_prefix,
                    )
                    continue
                raise

    def check_model_integrity(self, local_dir: str) -> bool:
        if not os.path.isdir(local_dir):
            return False
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            return False
        tokenizer_ok = any(
            os.path.exists(os.path.join(local_dir, name))
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        )
        if not tokenizer_ok:
            return False
        processor_ok = any(
            os.path.exists(os.path.join(local_dir, name))
            for name in ("preprocessor_config.json", "processor_config.json")
        )
        if not processor_ok:
            return False
        for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
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
        return any(
            os.path.exists(os.path.join(local_dir, name))
            for name in ("model.safetensors", "pytorch_model.bin")
        )


# ---------------------------------------------------------------------------
# Module-level chat-template helpers
# ---------------------------------------------------------------------------
# These were originally co-located with the super_prompt enhance path
# (nodes/llm/super_prompt/_qwen.py) — they now live with the rest of the
# engine code so both TS_Qwen3_VL_V3 and TS_SuperPrompt share the same
# "no chain-of-thought" template plumbing without copying it. They're
# module-level (not methods on QwenEngine) so the test stubs can pass any
# duck-typed object as ``engine`` — the only attribute we touch is
# ``get_tokenizer_from_processor``.


def _chat_template_functions(engine, processor) -> list[Any]:
    """Collect every ``apply_chat_template`` callable advertised by the
    processor / tokenizer, deduped and in priority order (processor first,
    then the underlying tokenizer). Raises if neither is available."""
    template_fns: list[Any] = []
    processor_template = getattr(processor, "apply_chat_template", None)
    if processor_template is not None:
        template_fns.append(processor_template)

    tokenizer = engine.get_tokenizer_from_processor(processor)
    tokenizer_template = getattr(tokenizer, "apply_chat_template", None)
    if tokenizer_template is not None and tokenizer_template not in template_fns:
        template_fns.append(tokenizer_template)

    if not template_fns:
        raise RuntimeError("Loaded processor/tokenizer does not provide apply_chat_template.")
    return template_fns


def _template_accepts_kwargs(template_fn, kwargs: dict[str, Any]) -> bool:
    """Return True iff ``template_fn`` declares every key in ``kwargs`` (or
    accepts ``**kwargs``). Best-effort fallback to True on signature errors."""
    try:
        signature = inspect.signature(template_fn)
    except Exception:
        return True
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    return all(key in parameters for key in kwargs.keys())


def _messages_have_visuals(messages: list[dict[str, Any]]) -> bool:
    """True when any message carries an ``image``/``video`` content item."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True
    return False


def _flatten_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert list-content messages to plain ``{role, content: str}`` for
    tokenizer-only chat templates that reject the structured shape."""
    flattened: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "\n\n".join(part for part in text_parts if part)
        else:
            text = str(content or "")
        flattened.append({"role": str(message.get("role") or "user"), "content": text})
    return flattened


def apply_chat_template_no_thinking(engine, processor, messages):
    """Apply Qwen's chat template forcing ``enable_thinking=False``.

    Probes every kwarg shape Qwen processors are known to accept:

    1. ``enable_thinking=False`` directly on the template call;
    2. ``chat_template_kwargs={"enable_thinking": False}`` (newer Qwen);
    3. plain call (template baked-in the choice already).

    For tokenizer-only templates that reject list-content messages, also
    falls back to a flattened text-only representation. Raises the last
    ``TypeError`` if no combination works.
    """
    base_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }

    thinking_variants = (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    )
    message_variants: list[list[dict[str, Any]]] = [messages]
    if not _messages_have_visuals(messages):
        flattened_messages = _flatten_text_messages(messages)
        if flattened_messages != messages:
            message_variants.append(flattened_messages)

    last_error: Exception | None = None
    for template_fn in _chat_template_functions(engine, processor):
        for candidate_messages in message_variants:
            for thinking_kwargs in thinking_variants:
                kwargs = {**base_kwargs, **thinking_kwargs}
                if not _template_accepts_kwargs(template_fn, kwargs):
                    continue
                try:
                    return template_fn(candidate_messages, **kwargs)
                except TypeError as exc:
                    last_error = exc
                    continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Loaded chat template rejected all supported argument variants.")


_ENGINE: QwenEngine | None = None


def get_qwen_engine() -> QwenEngine:
    """Return the process-wide ``QwenEngine`` singleton (lazy-initialised)."""

    global _ENGINE
    if _ENGINE is None:
        _ENGINE = QwenEngine()
    return _ENGINE
