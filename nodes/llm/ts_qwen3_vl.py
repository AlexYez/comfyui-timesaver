"""TS Qwen 3 VL V3 — Qwen prompt + vision-language node.

The heavy runtime lives in :mod:`nodes.llm._qwen_engine` (shared with
``TS_SuperPrompt``). This module owns only the V3 schema, preset loading,
the curated model list, and the ``execute``/``process`` plumbing that
adapts ComfyUI inputs to ``QwenEngine`` calls.

node_id: TS_Qwen3_VL_V3
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from comfy_api.v0_0_2 import IO

from ._qwen_engine import QwenEngine, get_qwen_engine


_LOGGER = logging.getLogger("comfyui_timesaver.ts_qwen3_vl")
_LOG_PREFIX = "[TS Qwen3 VL V3]"
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Canonical defaults used by ``execute`` to keep legacy workflows alive:
# any saved ``model_name`` not in ``_MODEL_LIST`` (i.e. removed from a newer
# release) is silently redirected to ``_DEFAULT_MODEL_ID`` with a warning,
# and presets renamed/removed from ``qwen_3_vl_presets.json`` fall back to
# ``_DEFAULT_PRESET_NAME``. ``validate_inputs`` accepts any value so the
# workflow loads in the first place — the redirect happens at run time.
_DEFAULT_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-2B-abliterated"
_DEFAULT_PRESET_NAME = "Prompts enhance"


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

def _presets_path() -> str:
    """Return the path to ``qwen_3_vl_presets.json``.

    Prefer the file next to this module (``nodes/llm/``); fall back to the
    legacy location one level up (``nodes/``) for backward compatibility.
    """
    preferred = os.path.join(_CURRENT_DIR, "qwen_3_vl_presets.json")
    if os.path.exists(preferred):
        return preferred
    return os.path.join(os.path.dirname(_CURRENT_DIR), "qwen_3_vl_presets.json")


def _load_presets() -> tuple[dict[str, Any], list[str]]:
    """Load ``qwen_3_vl_presets.json`` once on demand. Shared with TS_SuperPrompt."""
    presets_path = _presets_path()
    if not os.path.exists(presets_path):
        return {}, []
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        _LOGGER.warning("%s Preset load failed: %s", _LOG_PREFIX, exc)
        data = {}
    return data, list(data.keys())


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

class _Qwen3VLState:
    """Mutable singleton state. ``lock_class()`` blocks ``cls._x = …`` on
    registered V3 nodes, so anything that needs to mutate lives here."""

    versions_logged: bool = False


_state = _Qwen3VLState()


_MODEL_LIST = [
    "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
    "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
    "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
    "Custom (manual)",
]


def _log_versions_once() -> None:
    if _state.versions_logged:
        return
    _state.versions_logged = True
    try:
        import importlib.metadata as metadata
    except Exception:
        return
    pkgs = ["torch", "transformers", "accelerate", "bitsandbytes", "flash-attn", "tiktoken"]
    _LOGGER.info("%s Dependency versions:", _LOG_PREFIX)
    for pkg in pkgs:
        try:
            v = metadata.version(pkg)
            _LOGGER.info("%s   %s: %s", _LOG_PREFIX, pkg, v)
        except Exception:
            _LOGGER.info("%s   %s: not found", _LOG_PREFIX, pkg)


def _resolve_model_id(model_name: str, custom_model_id: str) -> str:
    if model_name == "Custom (manual)":
        custom = (custom_model_id or "").strip()
        if not custom:
            raise ValueError("Custom model id is empty.")
        return custom
    return model_name


# ---------------------------------------------------------------------------
# Public V3 node
# ---------------------------------------------------------------------------

class TS_Qwen3_VL_V3(IO.ComfyNode):
    """ComfyUI V3 frontend for the shared ``QwenEngine``."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        _presets, preset_keys = _load_presets()
        preset_options = preset_keys + ["Your instruction"]

        precision_options = ["auto", "bf16", "fp16", "fp32"]
        if QwenEngine.is_bitsandbytes_available():
            precision_options.extend(["int8", "int4"])
        attention_options = ["auto", "flash_attention_2", "sdpa", "eager"]
        # Prefer the canonical "Prompts enhance" preset when it exists,
        # otherwise fall back to whatever the JSON lists first.
        if _DEFAULT_PRESET_NAME in preset_options:
            default_preset = _DEFAULT_PRESET_NAME
        elif preset_options:
            default_preset = preset_options[0]
        else:
            default_preset = "Your instruction"

        return IO.Schema(
            node_id="TS_Qwen3_VL_V3",
            display_name="TS Qwen 3 VL V3",
            category="TS/LLM",
            inputs=[
                IO.Combo.Input(
                    "model_name",
                    options=_MODEL_LIST,
                    default=_DEFAULT_MODEL_ID,
                    tooltip="Выберите модель из списка. Для сторонних моделей выберите 'Custom (manual)'.",
                ),
                IO.String.Input(
                    "custom_model_id",
                    default="",
                    multiline=False,
                    tooltip="ID репозитория на HuggingFace (например, 'Qwen/Qwen2-VL-7B-Instruct') или полный локальный путь.",
                ),
                IO.String.Input(
                    "hf_token",
                    default="",
                    multiline=False,
                    tooltip="Ваш токен HuggingFace (Write/Read) для скачивания моделей. Оставьте пустым для публичных моделей.",
                ),
                IO.Combo.Input(
                    "system_preset",
                    options=preset_options,
                    default=default_preset,
                    tooltip="Предустановка системного промпта. Влияет на поведение и стиль ответов модели.",
                ),
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Ваш запрос (промпт) к модели.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip="Сид для воспроизводимости результатов генерации.",
                ),
                IO.Int.Input(
                    "max_new_tokens",
                    default=512,
                    min=64,
                    max=8192,
                    step=64,
                    tooltip="Максимальное количество токенов в ответе (длина текста).",
                ),
                IO.Combo.Input(
                    "precision",
                    options=precision_options,
                    default="auto",
                    tooltip="Точность весов. 'auto' выбирает оптимальную. int4/int8 требуют установленного bitsandbytes.",
                ),
                IO.Combo.Input(
                    "attention_mode",
                    options=attention_options,
                    default="auto",
                    tooltip="Тип внимания. 'flash_attention_2' быстрее и экономичнее, но требует совместимой GPU.",
                ),
                IO.Boolean.Input(
                    "offline_mode",
                    default=False,
                    tooltip="Запретить скачивание. Использовать только файлы, уже находящиеся в папке models/LLM.",
                ),
                IO.Boolean.Input(
                    "unload_after_generation",
                    default=False,
                    tooltip="Выгружать модель из памяти сразу после генерации. Экономит VRAM, но замедляет повторные запуски.",
                ),
                IO.Boolean.Input(
                    "enable",
                    default=True,
                    tooltip="Включить обработку. Если выключено - просто передает изображения на выход без изменений.",
                ),
                IO.Int.Input(
                    "max_image_size",
                    default=1024,
                    min=64,
                    max=4096,
                    step=32,
                    tooltip="Максимальный размер стороны изображения. Большие разрешения требуют больше VRAM.",
                ),
                IO.Int.Input(
                    "video_max_frames",
                    default=16,
                    min=4,
                    max=256,
                    step=4,
                    tooltip="Сколько кадров из видео передавать модели. Больше кадров = лучше понимание контекста, но больше расход памяти.",
                ),
                IO.Image.Input("image", optional=True, tooltip="Входное изображение."),
                IO.Image.Input("video", optional=True, tooltip="Входной видеопоток (батч изображений)."),
                IO.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    force_input=True,
                    optional=True,
                    tooltip="Ваш системный промпт. Работает, если в 'system_preset' выбрано 'Your instruction'.",
                ),
            ],
            outputs=[
                IO.String.Output(display_name="generated_text"),
                IO.Image.Output(display_name="processed_image"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        presets_path = _presets_path()
        mtime = os.path.getmtime(presets_path) if os.path.exists(presets_path) else None
        return (mtime,)

    @classmethod
    def validate_inputs(cls, **_kwargs) -> bool:
        """Accept any combo value so legacy workflows load without errors.

        ``model_name`` and ``system_preset`` values saved by a previous
        release of this pack may no longer be in ``_MODEL_LIST`` /
        ``qwen_3_vl_presets.json``. We deliberately skip the strict combo
        check here — ``_run_qwen_generation`` redirects unknown values to
        the canonical defaults (``_DEFAULT_MODEL_ID`` / ``Prompts enhance``)
        with a logged warning so the user can see what was substituted.
        """
        return True

    @classmethod
    def execute(
        cls,
        model_name: str,
        custom_model_id: str,
        hf_token: str,
        system_preset: str,
        prompt: str,
        seed: int,
        max_new_tokens: int,
        precision: str,
        attention_mode: str,
        offline_mode: bool,
        unload_after_generation: bool,
        enable: bool,
        max_image_size: int,
        video_max_frames: int,
        image: Any = None,
        video: Any = None,
        custom_system_prompt: str | None = None,
    ) -> IO.NodeOutput:
        _log_versions_once()
        engine = get_qwen_engine()
        text, processed_images = _run_qwen_generation(
            engine=engine,
            model_name=model_name,
            custom_model_id=custom_model_id,
            hf_token=hf_token,
            hf_endpoint="huggingface.co, hf-mirror.com",
            system_preset=system_preset,
            prompt=prompt,
            seed=seed,
            max_new_tokens=max_new_tokens,
            precision=precision,
            attention_mode=attention_mode,
            offline_mode=offline_mode,
            unload_after_generation=unload_after_generation,
            enable=enable,
            max_image_size=max_image_size,
            video_max_frames=video_max_frames,
            image=image,
            video=video,
            custom_system_prompt=custom_system_prompt,
        )
        return IO.NodeOutput(text, engine.pil_to_tensor(processed_images))


# ---------------------------------------------------------------------------
# Generation pipeline (kept module-level to avoid `cls._x` writes on the
# locked V3 class — see CLAUDE.md §5).
# ---------------------------------------------------------------------------

def _run_qwen_generation(
    *,
    engine: QwenEngine,
    model_name: str,
    custom_model_id: str,
    hf_token: str,
    hf_endpoint: str,
    system_preset: str,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    precision: str,
    attention_mode: str,
    offline_mode: bool,
    unload_after_generation: bool,
    enable: bool,
    max_image_size: int,
    video_max_frames: int,
    image: Any,
    video: Any,
    custom_system_prompt: str | None,
) -> tuple[str, list]:
    processed_images: list = []

    if not enable:
        _LOGGER.info("%s Processing mode: bypass (CPU)", _LOG_PREFIX)
        if image is not None:
            processed_images.extend(engine.tensor_to_pil_list(image))
        if video is not None:
            processed_images.extend(engine.tensor_to_pil_list(video))
        return (prompt.strip() if prompt else "", processed_images)

    try:
        resolved_model_id = _resolve_model_id(model_name, custom_model_id)
    except ValueError as exc:
        return (f"ERROR: {exc}", processed_images)

    # Legacy-workflow rescue: if the saved combo value isn't in the current
    # ``_MODEL_LIST`` and the user didn't explicitly pick ``Custom (manual)``,
    # silently switch to the canonical default. Without this the next step
    # would try to download a model that may have been removed/renamed and
    # the entire workflow would fail.
    if model_name != "Custom (manual)" and resolved_model_id not in _MODEL_LIST:
        _LOGGER.warning(
            "%s Legacy/unknown model %r — falling back to %r so the workflow keeps running.",
            _LOG_PREFIX,
            resolved_model_id,
            _DEFAULT_MODEL_ID,
        )
        resolved_model_id = _DEFAULT_MODEL_ID

    resolved_precision = engine.resolve_precision(precision, resolved_model_id)
    resolved_attention = engine.resolve_attention(attention_mode, resolved_precision)
    estimated_vram = engine.estimate_vram_usage(resolved_model_id, resolved_precision)
    engine.ensure_memory_available(estimated_vram)

    _LOGGER.info("%s model=%s", _LOG_PREFIX, resolved_model_id)
    _LOGGER.info(
        "%s precision=%s attention=%s",
        _LOG_PREFIX,
        resolved_precision,
        resolved_attention,
    )

    try:
        model, processor = engine.load_model(
            resolved_model_id,
            resolved_precision,
            resolved_attention,
            offline_mode,
            hf_token,
            hf_endpoint,
        )
    except Exception as exc:
        _LOGGER.error("%s Load error: %s", _LOG_PREFIX, exc, exc_info=True)
        return (f"ERROR: {exc}", processed_images)

    target_device = engine.get_device()
    moved_to_gpu = False
    if target_device.type == "cuda" and not engine.model_has_cuda_device(model):
        try:
            _LOGGER.info("%s Moving model to GPU for inference...", _LOG_PREFIX)
            engine.ensure_memory_available(estimated_vram, force_unload=True)
            model.to(target_device)
            moved_to_gpu = True
        except RuntimeError as exc:
            if engine.is_oom_error(exc):
                _LOGGER.error(
                    "%s OOM while moving to GPU. Falling back to CPU/Offload.", _LOG_PREFIX
                )
                try:
                    model.to("cpu")
                except Exception:
                    pass
                engine.prepare_memory(force=True)
                return ("ERROR: Out of Memory during model GPU transfer.", processed_images)
            raise

    preset_configs, _ = _load_presets()
    if system_preset == "Your instruction" and custom_system_prompt:
        system_prompt = custom_system_prompt
        gen_params: dict[str, Any] = {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.0}
    else:
        # Workflow may reference a preset that was renamed/removed since it
        # was saved. Fall back to the canonical ``Prompts enhance`` preset
        # instead of silently dropping all generation params.
        preset_data = preset_configs.get(system_preset) or preset_configs.get(_DEFAULT_PRESET_NAME)
        if isinstance(preset_data, dict):
            system_prompt = preset_data.get("system_prompt", "")
            gen_params = dict(preset_data.get("gen_params", {}))
            if preset_configs.get(system_preset) is None and system_preset:
                _LOGGER.warning(
                    "%s Preset %r not found; using %r.",
                    _LOG_PREFIX,
                    system_preset,
                    _DEFAULT_PRESET_NAME,
                )
        else:
            system_prompt = ""
            gen_params = {}
    if "temperature" not in gen_params:
        gen_params["temperature"] = 0.7

    user_content: list[dict[str, Any]] = []
    if video is not None:
        video_frames = engine.tensor_to_pil_list(video)
        if len(video_frames) > video_max_frames:
            indices = np.linspace(0, len(video_frames) - 1, video_max_frames, dtype=int)
            video_frames = [video_frames[i] for i in indices]
        processed_video = []
        for frame in video_frames:
            frame_proc = engine.resize_and_crop_image(frame, max_image_size)
            processed_images.append(frame_proc)
            processed_video.append(frame_proc)
        user_content.append({"type": "video", "video": processed_video, "fps": 1.0})

    if image is not None:
        for img in engine.tensor_to_pil_list(image):
            img_proc = engine.resize_and_crop_image(img, max_image_size)
            processed_images.append(img_proc)
            user_content.append({"type": "image", "image": img_proc})

    user_content.append({"type": "text", "text": prompt.strip() if prompt else ""})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
        {"role": "user", "content": user_content},
    ]

    output_text = ""
    try:
        if (image is not None or video is not None) and not engine.supports_multimodal_inputs(processor):
            raise RuntimeError(
                "Loaded processor/tokenizer does not support image/video input. "
                "Select a vision-language model or disable image/video inputs."
            )

        # Force ``enable_thinking=False`` so Qwen3 doesn't leak a
        # ``<think>...</think>`` block into ``generated_text``. The engine
        # probes every kwarg shape Qwen processors accept.
        inputs = engine.apply_chat_template_no_thinking(processor, messages)
        input_device = engine.select_input_device(model)
        inputs = engine.move_inputs_to_device(inputs, input_device)
        engine.log_processing_device("inputs_moved", input_device, model, inputs)

        gen_params["max_new_tokens"] = max_new_tokens
        gen_params["use_cache"] = True
        gen_params["pad_token_id"] = engine.get_pad_token_id(processor, model)
        gen_params["do_sample"] = float(gen_params.get("temperature", 0) or 0) > 0
        rng_cuda_devices = engine.cuda_indices_for_rng(model, input_device)

        if engine.supports_generator(model):
            gen_device = engine.select_generator_device(input_device)
            generator = torch.Generator(device=gen_device)
            generator.manual_seed(seed)
            gen_params["generator"] = generator
            rng_context = nullcontext()
        else:
            rng_context = (
                torch.random.fork_rng(devices=rng_cuda_devices)
                if rng_cuda_devices
                else torch.random.fork_rng()
            )

        dtype = engine.dtype_from_precision(resolved_precision)
        autocast_device = (
            input_device
            if isinstance(input_device, torch.device)
            else getattr(model, "device", None)
        )
        use_autocast = (
            getattr(autocast_device, "type", None) == "cuda"
            and dtype in (torch.float16, torch.bfloat16)
        )

        with rng_context:
            if "generator" not in gen_params:
                torch.manual_seed(seed)
                for idx in rng_cuda_devices:
                    with torch.cuda.device(idx):
                        torch.cuda.manual_seed(seed)
            _LOGGER.info("%s Generating...", _LOG_PREFIX)
            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        generated_ids = model.generate(**inputs, **gen_params)
                else:
                    generated_ids = model.generate(**inputs, **gen_params)

        generated_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = engine.batch_decode(
            processor,
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        # Belt-and-braces: drop any ``<think>...</think>`` block that
        # slipped past the ``enable_thinking=False`` kwarg (some chat
        # templates bake the choice in regardless of the kwarg).
        output_text = QwenEngine.strip_thinking_block(output_text)
    except Exception as exc:
        _LOGGER.error("%s Generation error: %s", _LOG_PREFIX, exc, exc_info=True)
        output_text = f"ERROR: {exc}"
    finally:
        if unload_after_generation:
            engine.unload_model(resolved_model_id, resolved_precision, resolved_attention)
        elif moved_to_gpu:
            try:
                _LOGGER.info("%s Soft-offloading model to CPU to free VRAM.", _LOG_PREFIX)
                model.to("cpu")
                engine.prepare_memory(force=True)
            except Exception as cleanup_exc:
                _LOGGER.debug(
                    "%s Soft-offload to CPU failed: %s", _LOG_PREFIX, cleanup_exc
                )

    return output_text, processed_images


NODE_CLASS_MAPPINGS = {"TS_Qwen3_VL_V3": TS_Qwen3_VL_V3}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Qwen3_VL_V3": "TS Qwen 3 VL V3"}
