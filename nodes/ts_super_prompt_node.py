from __future__ import annotations

import asyncio
import gc
import inspect
import json
import logging
import re
import threading
from contextlib import nullcontext
from typing import Any

from aiohttp import web
from comfy_api.latest import IO

try:
    import server
except Exception:
    server = None

try:
    from .ts_qwen3_vl_v3_node import TS_Qwen3_VL_V3
except Exception:
    from ts_qwen3_vl_v3_node import TS_Qwen3_VL_V3


LOGGER = logging.getLogger("comfyui_timesaver.ts_super_prompt")
LOG_PREFIX = "[TS Super Prompt]"
ROUTE_BASE = "/ts_super_prompt"
EVENT_PREFIX = "ts_super_prompt"

# User-configurable settings for the compact TS Super Prompt UI.
# Change these values here instead of exposing extra widgets in ComfyUI.
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_PRESET = "Prompts enhance"
CUSTOM_PRESET = "Your instruction"
SUPER_PROMPT_TARGET = "auto"
SUPER_PROMPT_ENHANCE_ON_EXECUTE = False
SUPER_PROMPT_SEED = 42
SUPER_PROMPT_MAX_NEW_TOKENS = 512
SUPER_PROMPT_PRECISION = "auto"
SUPER_PROMPT_ATTENTION_MODE = "auto"
SUPER_PROMPT_OFFLINE_MODE = False
SUPER_PROMPT_UNLOAD_AFTER_GENERATION = False
SUPER_PROMPT_MAX_IMAGE_SIZE = 1024
SUPER_PROMPT_HF_TOKEN = ""
SUPER_PROMPT_HF_ENDPOINT = "huggingface.co, hf-mirror.com"
SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT = ""

PROMPT_TARGETS = ("auto", "image", "video", "music")

_MODEL_LOCK = threading.Lock()
_QWEN_ENGINE: TS_Qwen3_VL_V3 | None = None


def _log_info(message: str) -> None:
    LOGGER.info("%s %s", LOG_PREFIX, message)


def _log_warning(message: str) -> None:
    LOGGER.warning("%s %s", LOG_PREFIX, message)


def _resolve_prompt_server():
    if server is None:
        _log_warning("PromptServer unavailable. HTTP routes disabled.")
        return None
    try:
        return server.PromptServer.instance
    except Exception as exc:
        _log_warning(f"PromptServer init failed. HTTP routes disabled: {exc}")
        return None


_PROMPT_SERVER = _resolve_prompt_server()


def _register_post(path: str):
    def decorator(func):
        if _PROMPT_SERVER is None:
            return func
        try:
            _PROMPT_SERVER.routes.post(path)(func)
        except Exception as exc:
            _log_warning(f"Failed to register POST route '{path}': {exc}")
        return func

    return decorator


def _send_event(event: str, payload: dict[str, Any]) -> None:
    if _PROMPT_SERVER is None:
        return
    try:
        _PROMPT_SERVER.send_sync(f"{EVENT_PREFIX}.{event}", payload)
    except Exception as exc:
        LOGGER.debug("%s WebSocket event send failed: %s", LOG_PREFIX, exc)


def _send_progress(operation_id: str | None, text: str, percent: float | None = None) -> None:
    payload: dict[str, Any] = {"text": text}
    if operation_id:
        payload["operation_id"] = operation_id
    if percent is not None:
        payload["percent"] = max(0.0, min(100.0, float(percent)))
    _send_event("progress", payload)


def _send_done(operation_id: str | None, text: str = "Ready") -> None:
    payload: dict[str, Any] = {"text": text, "percent": 100.0}
    if operation_id:
        payload["operation_id"] = operation_id
    _send_event("done", payload)


def _send_error(operation_id: str | None, text: str) -> None:
    payload: dict[str, Any] = {"text": text}
    if operation_id:
        payload["operation_id"] = operation_id
    _send_event("error", payload)


def _get_qwen_engine() -> TS_Qwen3_VL_V3:
    global _QWEN_ENGINE
    if _QWEN_ENGINE is None:
        _QWEN_ENGINE = TS_Qwen3_VL_V3()
    return _QWEN_ENGINE


def _load_presets() -> tuple[dict[str, Any], list[str]]:
    presets, keys = TS_Qwen3_VL_V3._load_presets()
    if not isinstance(presets, dict):
        return {}, []
    return presets, [key for key in keys if isinstance(key, str) and key]


def _preset_options() -> list[str]:
    _presets, keys = _load_presets()
    options = list(keys)
    if CUSTOM_PRESET not in options:
        options.append(CUSTOM_PRESET)
    return options or [CUSTOM_PRESET]


def _default_preset(options: list[str]) -> str:
    if DEFAULT_PRESET in options:
        return DEFAULT_PRESET
    return options[0] if options else CUSTOM_PRESET


def _resolve_preset(system_preset: str, custom_system_prompt: str | None) -> tuple[str, dict[str, Any]]:
    presets, _keys = _load_presets()
    preset_name = str(system_preset or "").strip()

    if preset_name == CUSTOM_PRESET:
        prompt = str(custom_system_prompt or "").strip()
        if prompt:
            return prompt, {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}

    preset_data = presets.get(preset_name)
    if not isinstance(preset_data, dict):
        preset_data = presets.get(DEFAULT_PRESET)
    if not isinstance(preset_data, dict) and presets:
        first_key = next(iter(presets.keys()))
        preset_data = presets.get(first_key)

    if isinstance(preset_data, dict):
        system_prompt = str(preset_data.get("system_prompt") or "").strip()
        gen_params = preset_data.get("gen_params") or {}
        if not isinstance(gen_params, dict):
            gen_params = {}
        return system_prompt, dict(gen_params)

    return (
        "You are a senior prompt engineer. Translate the user's idea to English if needed and "
        "return only one polished generation prompt with no commentary.",
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
    )


def _target_instruction(prompt_target: str, has_image: bool) -> str:
    target = str(prompt_target or "auto").strip().lower()
    if target not in PROMPT_TARGETS:
        target = "auto"

    if target == "image":
        return (
            "Target output: image generation prompt. Create one vivid English paragraph focused on "
            "subject, composition, materials, environment, light, lens/camera feel, color palette, "
            "and style. Preserve the user's intent and do not add lists or quality-tag spam."
        )
    if target == "video":
        return (
            "Target output: video generation prompt. Create one cinematic English prompt with a clear "
            "camera move, subject action, motion physics, atmosphere, temporal flow, and visual continuity. "
            "If an image is provided, use it as the visual reference."
        )
    if target == "music":
        return (
            "Target output: music generation prompt. Create one English prompt describing genre, mood, "
            "tempo, rhythm, instrumentation, arrangement, dynamics, production style, and emotional arc. "
            "Do not describe non-audio visuals unless they directly inform the music."
        )

    if has_image:
        return (
            "Target output: infer whether image or video generation is more appropriate from the user's "
            "idea and the visual input, then return one polished English generation prompt."
        )
    return (
        "Target output: infer whether the user needs an image, video, or music generation prompt, then "
        "return one polished English prompt for that medium."
    )


def _build_messages(system_prompt: str, text: str, prompt_target: str, image: Any, max_image_size: int) -> list[dict[str, Any]]:
    engine = _get_qwen_engine()
    user_content: list[dict[str, Any]] = []

    user_text = (
        f"{_target_instruction(prompt_target, image is not None)}\n\n"
        "Hard rules:\n"
        "- Translate the source idea to English when needed.\n"
        "- Keep the user's core meaning, named subjects, and constraints.\n"
        "- Return only the final prompt, with no preface, no analysis, and no markdown.\n"
        "- Do not use thinking mode. Do not output chain-of-thought or hidden reasoning.\n\n"
        f"Source idea:\n{text or ''}"
    )

    system_text = (
        f"{system_prompt.strip()}\n\n"
        "Runtime mode: non-thinking. Produce the answer directly and never include a <think> block."
    )

    if image is not None:
        for pil_image in engine._tensor_to_pil_list(image):
            user_content.append(
                {
                    "type": "image",
                    "image": engine._resize_and_crop_image(pil_image, int(max_image_size)),
                }
            )

    user_content.append({"type": "text", "text": user_text})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {"role": "user", "content": user_content},
    ]


def _messages_have_visuals(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True
    return False


def _flatten_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
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


def _chat_template_functions(engine: TS_Qwen3_VL_V3, processor) -> list[Any]:
    template_fns: list[Any] = []
    processor_template = getattr(processor, "apply_chat_template", None)
    if processor_template is not None:
        template_fns.append(processor_template)

    get_tokenizer = getattr(engine, "_get_tokenizer_from_processor", None)
    tokenizer = get_tokenizer(processor) if callable(get_tokenizer) else getattr(processor, "tokenizer", processor)
    tokenizer_template = getattr(tokenizer, "apply_chat_template", None)
    if tokenizer_template is not None and tokenizer_template not in template_fns:
        template_fns.append(tokenizer_template)

    if not template_fns:
        raise RuntimeError("Loaded processor/tokenizer does not provide apply_chat_template.")
    return template_fns


def _template_accepts_kwargs(template_fn, kwargs: dict[str, Any]) -> bool:
    try:
        signature = inspect.signature(template_fn)
    except Exception:
        return True
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    return all(key in parameters for key in kwargs.keys())


def _apply_chat_template_no_thinking(engine: TS_Qwen3_VL_V3, processor, messages: list[dict[str, Any]]):
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
    message_variants = [messages]
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


def _filter_generation_params(model, params: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.generate)
    except Exception:
        return params
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params

    allowed = set(signature.parameters.keys())
    return {key: value for key, value in params.items() if key in allowed}


_GENERATION_PARAM_ALIASES = {
    "max_tokens": "max_new_tokens",
    "max_completion_tokens": "max_new_tokens",
}
_UNSUPPORTED_GENERATION_PARAMS = {
    "frequency_penalty",
    "n",
    "presence_penalty",
    "response_format",
    "stop",
    "stream",
}


def _normalize_generation_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for alias, target in _GENERATION_PARAM_ALIASES.items():
        value = normalized.pop(alias, None)
        if value is not None and target not in normalized:
            normalized[target] = value
    for key in _UNSUPPORTED_GENERATION_PARAMS:
        normalized.pop(key, None)
    return normalized


def _unused_model_kwargs_from_error(exc: ValueError) -> list[str]:
    match = re.search(r"not used by the model:\s*(\[[^\]]+\])", str(exc))
    if not match:
        return []
    return re.findall(r"'([^']+)'", match.group(1))


def _generate_with_filtered_kwargs(model, inputs: dict[str, Any], gen_params: dict[str, Any]):
    current_params = dict(gen_params)
    for _attempt in range(4):
        try:
            return model.generate(**inputs, **current_params)
        except ValueError as exc:
            unused_keys = set(_unused_model_kwargs_from_error(exc))
            if not unused_keys:
                raise
            next_params = {key: value for key, value in current_params.items() if key not in unused_keys}
            if len(next_params) == len(current_params):
                raise
            _log_warning(f"Dropping unsupported Qwen generation params: {', '.join(sorted(unused_keys))}")
            current_params = next_params
    return model.generate(**inputs, **current_params)


def _clean_model_output(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^\s*```(?:text|markdown|prompt)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    cleaned = re.sub(
        r"^\s*(?:final\s+prompt|prompt|english\s+prompt|enhanced\s+prompt|result)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _generate_with_qwen(
    text: str,
    system_preset: str,
    operation_id: str | None = None,
    image: Any = None,
) -> str:
    if not str(text or "").strip() and image is None:
        return ""

    lock_acquired = _MODEL_LOCK.acquire(blocking=False)
    if not lock_acquired:
        _send_progress(operation_id, "Waiting for Qwen", 2.0)
        _MODEL_LOCK.acquire()

    try:
        _send_progress(operation_id, "Preparing prompt", 5.0)
        engine = _get_qwen_engine()
        system_prompt, gen_params = _resolve_preset(system_preset, SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT)
        resolved_precision = engine._resolve_precision(SUPER_PROMPT_PRECISION, DEFAULT_MODEL_ID)
        resolved_attention = engine._resolve_attention(SUPER_PROMPT_ATTENTION_MODE, resolved_precision)
        estimated_vram = engine._estimate_vram_usage(DEFAULT_MODEL_ID, resolved_precision)

        _log_info(
            f"model={DEFAULT_MODEL_ID} precision={resolved_precision} "
            f"attention={resolved_attention} thinking=disabled"
        )
        _send_progress(operation_id, "Checking memory", 12.0)
        engine._ensure_memory_available(estimated_vram)

        _send_progress(operation_id, "Loading or downloading Qwen model", 22.0)
        model, processor = engine._load_model(
            DEFAULT_MODEL_ID,
            resolved_precision,
            resolved_attention,
            bool(SUPER_PROMPT_OFFLINE_MODE),
            str(SUPER_PROMPT_HF_TOKEN or ""),
            str(SUPER_PROMPT_HF_ENDPOINT or ""),
        )

        target_device = engine._get_device()
        moved_to_gpu = False
        if target_device.type == "cuda" and not engine._model_has_cuda_device(model):
            try:
                _send_progress(operation_id, "Moving Qwen to GPU", 38.0)
                engine._ensure_memory_available(estimated_vram, force_unload=True)
                model.to(target_device)
                moved_to_gpu = True
            except RuntimeError as exc:
                if engine._is_oom_error(exc):
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                    engine._prepare_memory(force=True)
                    raise RuntimeError("Out of memory during Qwen GPU transfer.") from exc
                raise

        try:
            if image is not None and not engine._supports_multimodal_inputs(processor):
                raise RuntimeError(
                    "Loaded processor/tokenizer does not support image input. "
                    "Use a Qwen vision-language model or disconnect image."
                )

            _send_progress(operation_id, "Preparing Qwen input", 52.0)
            messages = _build_messages(
                system_prompt,
                text,
                SUPER_PROMPT_TARGET,
                image,
                int(SUPER_PROMPT_MAX_IMAGE_SIZE),
            )
            inputs = _apply_chat_template_no_thinking(engine, processor, messages)
            input_device = engine._select_input_device(model)
            inputs = engine._move_inputs_to_device(inputs, input_device)
            engine._log_processing_device("super_prompt_inputs", input_device, model, inputs)

            gen_params = dict(gen_params)
            gen_params.setdefault("temperature", 0.7 if image is not None else 1.0)
            gen_params.setdefault("top_p", 0.8 if image is not None else 1.0)
            gen_params.setdefault("top_k", 20)
            gen_params.setdefault("repetition_penalty", 1.0)
            gen_params["max_new_tokens"] = int(SUPER_PROMPT_MAX_NEW_TOKENS)
            gen_params["use_cache"] = True
            gen_params["pad_token_id"] = engine._get_pad_token_id(processor, model)
            gen_params["do_sample"] = float(gen_params.get("temperature", 0.0) or 0.0) > 0.0
            gen_params = _normalize_generation_params(gen_params)

            rng_cuda_devices = engine._cuda_indices_for_rng(model, input_device)
            if engine._supports_generator(model):
                gen_device = engine._select_generator_device(input_device)
                gen = __import__("torch").Generator(device=gen_device)
                gen.manual_seed(int(SUPER_PROMPT_SEED))
                gen_params["generator"] = gen
                rng_context = nullcontext()
            else:
                torch = __import__("torch")
                rng_context = torch.random.fork_rng(devices=rng_cuda_devices) if rng_cuda_devices else torch.random.fork_rng()

            torch = __import__("torch")
            dtype = engine._dtype_from_precision(resolved_precision)
            autocast_device = input_device if hasattr(input_device, "type") else getattr(model, "device", None)
            use_autocast = getattr(autocast_device, "type", None) == "cuda" and dtype in (torch.float16, torch.bfloat16)
            gen_params = _filter_generation_params(model, gen_params)

            with rng_context:
                if "generator" not in gen_params:
                    torch.manual_seed(int(SUPER_PROMPT_SEED))
                    for idx in rng_cuda_devices:
                        with torch.cuda.device(idx):
                            torch.cuda.manual_seed(int(SUPER_PROMPT_SEED))

                _send_progress(operation_id, "Generating AI prompt", 72.0)
                with torch.inference_mode():
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params)
                    else:
                        generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params)

            _send_progress(operation_id, "Decoding prompt", 92.0)
            generated_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = engine._batch_decode(
                processor,
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            return _clean_model_output(output_text)
        finally:
            if SUPER_PROMPT_UNLOAD_AFTER_GENERATION:
                _send_progress(operation_id, "Unloading Qwen", 96.0)
                engine._unload_model(DEFAULT_MODEL_ID, resolved_precision, resolved_attention)
            elif moved_to_gpu:
                try:
                    model.to("cpu")
                    engine._prepare_memory(force=True)
                except Exception:
                    pass
            gc.collect()
    finally:
        _MODEL_LOCK.release()


@_register_post(f"{ROUTE_BASE}/enhance")
async def enhance_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body."}, status=400)
    except Exception:
        data = {}

    operation_id = str(data.get("operation_id") or "")
    try:
        result = await asyncio.to_thread(
            _generate_with_qwen,
            str(data.get("text") or ""),
            str(data.get("system_preset") or DEFAULT_PRESET),
            operation_id,
            None,
        )
        _send_done(operation_id, "AI prompt ready")
        return web.json_response({"ok": True, "text": result, "thinking": False, "model": DEFAULT_MODEL_ID})
    except Exception as exc:
        LOGGER.exception("%s AI prompt enhancement failed", LOG_PREFIX)
        _send_error(operation_id, str(exc))
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


class TS_SuperPrompt(IO.ComfyNode):
    """Compact prompt node: microphone dictation plus Qwen prompt enhancement."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        preset_options = _preset_options()
        return IO.Schema(
            node_id="TS_SuperPrompt",
            display_name="TS Super Prompt",
            category="TS/LLM",
            description=(
                "Voice prompt field with optional Qwen3.5 AI prompt enhancement for image, video, and music prompts."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Поле промпта: сюда попадает распознанная речь, "
                        "а кнопка Ai prompt заменяет текст улучшенным промптом."
                    ),
                ),
                IO.Combo.Input(
                    "system_preset",
                    options=preset_options,
                    default=_default_preset(preset_options),
                    tooltip="Выберите системный пресет из qwen_3_vl_presets.json для улучшения промпта.",
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Опциональное изображение-референс для улучшения промпта, "
                        "если SUPER_PROMPT_ENHANCE_ON_EXECUTE включен в коде."
                    ),
                ),
            ],
            outputs=[IO.String.Output(display_name="text")],
            search_aliases=[
                "super prompt",
                "ai prompt",
                "prompt enhancer",
                "voice recognition",
                "qwen prompt",
                "speech to prompt",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        text: str = "",
        system_preset: str = DEFAULT_PRESET,
        **_: Any,
    ) -> bool | str:
        if not isinstance(text, str):
            return "text must be a string."
        if system_preset not in _preset_options():
            return "system_preset must be one of the presets from qwen_3_vl_presets.json."
        return True

    @classmethod
    def execute(
        cls,
        text: str = "",
        translate_to_english: bool = False,
        system_preset: str = DEFAULT_PRESET,
        image: Any = None,
        **_: Any,
    ) -> IO.NodeOutput:
        _ = translate_to_english
        if not SUPER_PROMPT_ENHANCE_ON_EXECUTE:
            return IO.NodeOutput(text or "")

        enhanced = _generate_with_qwen(
            text=text or "",
            system_preset=system_preset,
            operation_id=None,
            image=image,
        )
        return IO.NodeOutput(enhanced)


NODE_CLASS_MAPPINGS = {"TS_SuperPrompt": TS_SuperPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SuperPrompt": "TS Super Prompt"}
