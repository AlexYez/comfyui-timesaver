"""Qwen-prompt-enhancement pipeline for TS Super Prompt.

Owns: preset loading from `qwen_3_vl_presets.json`, message construction
(system + user + optional image content), `apply_chat_template` invocation
across processor/tokenizer variants with `enable_thinking=False`,
generation-param normalisation/filtering, the public `_generate_with_qwen`
entry point used by the `/ts_super_prompt/enhance` HTTP route, and the
HF-snapshot download progress monitor used while Qwen weights stream in.

`torch` is imported lazily inside `_generate_with_qwen` so contract tests
can stub `nodes.llm.ts_qwen3_vl` without forcing torch.

Private — loader skips paths with `_`-prefixed components.
"""

from __future__ import annotations

import gc
import inspect
import re
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import folder_paths

from ..ts_qwen3_vl import TS_Qwen3_VL_V3
from ._helpers import (
    CUSTOM_PRESET,
    DEFAULT_MODEL_ID,
    DEFAULT_PRESET,
    LOG_PREFIX,
    LOGGER,
    MODEL_LOCK,
    PROMPT_TARGETS,
    SUPER_PROMPT_ATTENTION_MODE,
    SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT,
    SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES,
    SUPER_PROMPT_HF_ENDPOINT,
    SUPER_PROMPT_HF_TOKEN,
    SUPER_PROMPT_MAX_IMAGE_SIZE,
    SUPER_PROMPT_MAX_NEW_TOKENS,
    SUPER_PROMPT_OFFLINE_MODE,
    SUPER_PROMPT_PRECISION,
    SUPER_PROMPT_SEED,
    SUPER_PROMPT_TARGET,
    SUPER_PROMPT_UNLOAD_AFTER_GENERATION,
    directory_size,
    format_bytes,
    log_info,
    log_warning,
    send_progress,
)


_QWEN_ENGINE: TS_Qwen3_VL_V3 | None = None


class QwenDownloadProgressMonitor:
    """Poll local HuggingFace files while snapshot_download runs."""

    def __init__(
        self,
        operation_id: str | None,
        model_id: str,
        local_dir: Path,
        total_bytes: int,
        start_percent: float = 20.0,
        end_percent: float = 44.0,
        enabled: bool = True,
    ):
        self.operation_id = operation_id
        self.model_id = model_id
        self.local_dir = local_dir
        self.total_bytes = max(1, int(total_bytes))
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.enabled = bool(enabled)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_size = -1

    def start(self) -> None:
        if not self.enabled:
            return
        send_progress(self.operation_id, f"Connecting to HuggingFace for {self.model_id}", self.start_percent)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, success: bool) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if success:
            size = directory_size(self.local_dir)
            send_progress(
                self.operation_id,
                f"Qwen model files ready ({format_bytes(size)})",
                self.end_percent,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._emit_progress()
            self._stop.wait(0.7)

    def _emit_progress(self) -> None:
        size = directory_size(self.local_dir)
        if size == self._last_size and size > 0:
            return
        self._last_size = size
        ratio = max(0.0, min(1.0, size / float(self.total_bytes)))
        percent = self.start_percent + (self.end_percent - self.start_percent) * ratio
        if size <= 0:
            text = f"Downloading Qwen model {self.model_id}"
        else:
            text = f"Downloading Qwen model {self.model_id}: {format_bytes(size)}"
        send_progress(self.operation_id, text, percent)


def _qwen_model_dir(model_id: str) -> Path:
    return Path(getattr(folder_paths, "models_dir", Path.cwd() / "models")) / "LLM" / str(model_id).split("/")[-1]


def _qwen_download_estimate(model_id: str) -> int:
    explicit = SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES.get(model_id)
    if explicit:
        return int(explicit)
    try:
        size_b = float(_get_qwen_engine()._model_size_b(model_id))
    except Exception:
        size_b = 2.0
    return int(max(1.0, size_b) * 2_250_000_000)


def _is_qwen_model_available(engine: TS_Qwen3_VL_V3, model_id: str) -> bool:
    checker = getattr(engine, "_check_model_integrity", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(str(_qwen_model_dir(model_id))))
    except Exception:
        return False


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


def preset_options() -> list[str]:
    _presets, keys = _load_presets()
    options = list(keys)
    if CUSTOM_PRESET not in options:
        options.append(CUSTOM_PRESET)
    return options or [CUSTOM_PRESET]


def default_preset(options: list[str]) -> str:
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
            log_warning(f"Dropping unsupported Qwen generation params: {', '.join(sorted(unused_keys))}")
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

    lock_acquired = MODEL_LOCK.acquire(blocking=False)
    if not lock_acquired:
        send_progress(operation_id, "Waiting for Qwen", 2.0)
        MODEL_LOCK.acquire()

    try:
        send_progress(operation_id, "Preparing prompt", 5.0)
        engine = _get_qwen_engine()
        system_prompt, gen_params = _resolve_preset(system_preset, SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT)
        resolved_precision = engine._resolve_precision(SUPER_PROMPT_PRECISION, DEFAULT_MODEL_ID)
        resolved_attention = engine._resolve_attention(SUPER_PROMPT_ATTENTION_MODE, resolved_precision)
        estimated_vram = engine._estimate_vram_usage(DEFAULT_MODEL_ID, resolved_precision)

        log_info(
            f"model={DEFAULT_MODEL_ID} precision={resolved_precision} "
            f"attention={resolved_attention} thinking=disabled"
        )
        send_progress(operation_id, "Checking memory", 12.0)
        engine._ensure_memory_available(estimated_vram)

        send_progress(operation_id, "Checking Qwen model files", 16.0)
        qwen_model_available = _is_qwen_model_available(engine, DEFAULT_MODEL_ID)
        if qwen_model_available:
            send_progress(operation_id, "Qwen model found locally", 22.0)
        elif SUPER_PROMPT_OFFLINE_MODE:
            send_progress(operation_id, "Using offline Qwen model files", 22.0)
        else:
            send_progress(operation_id, "Qwen model download starting", 20.0)

        if not qwen_model_available and not bool(SUPER_PROMPT_OFFLINE_MODE):
            qwen_monitor = QwenDownloadProgressMonitor(
                operation_id=operation_id,
                model_id=DEFAULT_MODEL_ID,
                local_dir=_qwen_model_dir(DEFAULT_MODEL_ID),
                total_bytes=_qwen_download_estimate(DEFAULT_MODEL_ID),
            )
            qwen_monitor.start()
            download_success = False
            try:
                engine._ensure_model_available(
                    DEFAULT_MODEL_ID,
                    bool(SUPER_PROMPT_OFFLINE_MODE),
                    str(SUPER_PROMPT_HF_TOKEN or ""),
                    str(SUPER_PROMPT_HF_ENDPOINT or ""),
                )
                download_success = True
            finally:
                qwen_monitor.stop(download_success)

        send_progress(operation_id, "Loading Qwen model into memory", 46.0)
        model, processor = engine._load_model(
            DEFAULT_MODEL_ID,
            resolved_precision,
            resolved_attention,
            bool(SUPER_PROMPT_OFFLINE_MODE),
            str(SUPER_PROMPT_HF_TOKEN or ""),
            str(SUPER_PROMPT_HF_ENDPOINT or ""),
        )
        send_progress(operation_id, "Qwen model loaded", 50.0)

        target_device = engine._get_device()
        moved_to_gpu = False
        if target_device.type == "cuda" and not engine._model_has_cuda_device(model):
            try:
                send_progress(operation_id, "Moving Qwen to GPU", 54.0)
                engine._ensure_memory_available(estimated_vram, force_unload=True)
                model.to(target_device)
                moved_to_gpu = True
            except RuntimeError as exc:
                if engine._is_oom_error(exc):
                    try:
                        model.to("cpu")
                    except Exception as cleanup_exc:
                        LOGGER.debug("%s OOM cleanup move-to-CPU failed: %s", LOG_PREFIX, cleanup_exc)
                    engine._prepare_memory(force=True)
                    raise RuntimeError("Out of memory during Qwen GPU transfer.") from exc
                raise

        try:
            if image is not None and not engine._supports_multimodal_inputs(processor):
                raise RuntimeError(
                    "Loaded processor/tokenizer does not support image input. "
                    "Use a Qwen vision-language model or disconnect image."
                )

            send_progress(operation_id, "Preparing Qwen input", 62.0)
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

            # Local import keeps torch out of module-level eval so contract
            # tests can stub `nodes.llm.ts_qwen3_vl` without importing torch.
            import torch

            rng_cuda_devices = engine._cuda_indices_for_rng(model, input_device)
            if engine._supports_generator(model):
                gen_device = engine._select_generator_device(input_device)
                gen = torch.Generator(device=gen_device)
                gen.manual_seed(int(SUPER_PROMPT_SEED))
                gen_params["generator"] = gen
                rng_context = nullcontext()
            else:
                rng_context = torch.random.fork_rng(devices=rng_cuda_devices) if rng_cuda_devices else torch.random.fork_rng()

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

                send_progress(operation_id, "Generating AI prompt", 78.0)
                with torch.inference_mode():
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params)
                    else:
                        generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params)

            send_progress(operation_id, "Decoding prompt", 92.0)
            generated_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
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
                send_progress(operation_id, "Unloading Qwen", 96.0)
                engine._unload_model(DEFAULT_MODEL_ID, resolved_precision, resolved_attention)
            elif moved_to_gpu:
                try:
                    model.to("cpu")
                    engine._prepare_memory(force=True)
                except Exception as cleanup_exc:
                    LOGGER.debug("%s Post-generation soft-offload failed: %s", LOG_PREFIX, cleanup_exc)
            gc.collect()
    finally:
        MODEL_LOCK.release()


__all__ = [
    "_generate_with_qwen",
    "preset_options",
    "default_preset",
]
