"""Backward-compatibility re-export shim.

Real code lives in ``nodes/llm/super_prompt/``:

- ``_helpers.py`` — config constants, logger, PromptServer event dispatch.
- ``_voice.py``   — Whisper download/load + ffmpeg decode + VAD + transcribe.
- ``_qwen.py``    — preset loading + chat template + Qwen generation.
- ``ts_super_prompt.py`` — public class + 4 HTTP route handlers.

This shim exists so existing tests that import ``nodes.llm.ts_super_prompt``
keep working. Loader will *also* discover ``nodes/llm/super_prompt/ts_super_prompt.py``
through the recursive scan in __init__.py — that is the only place where the
node and its routes are registered. We deliberately set ``NODE_CLASS_MAPPINGS``
empty here so this file does not double-register the node.

The ``TS_SuperPrompt`` class is re-exported below for tests that introspect
the schema directly. Importing it triggers a one-time import of
``super_prompt/ts_super_prompt.py``, which runs the @register_post / @register_get
decorators exactly once — Python's module cache prevents the loader's later
import from re-running them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports: constants
# ---------------------------------------------------------------------------
from .super_prompt._helpers import (  # noqa: F401
    ACTIVE_MODEL,
    AI_EVENT_PREFIX,
    AI_ROUTE_BASE,
    ALLOWED_AUDIO_SUFFIXES,
    ALL_MODELS,
    AUDIO_EDGE_FADE_MS,
    AUDIO_NORMALIZE_ENABLED,
    AUDIO_NORMALIZE_MAX_GAIN_DB,
    AUDIO_NORMALIZE_TARGET_PEAK,
    AUDIO_SAMPLE_RATE,
    AUDIO_TRIM_ENABLED,
    AUDIO_VAD_ADAPTIVE_MULTIPLIER,
    AUDIO_VAD_ENABLED,
    AUDIO_VAD_FRAME_MS,
    AUDIO_VAD_HOP_MS,
    AUDIO_VAD_LOW_MULTIPLIER,
    AUDIO_VAD_MIN_SPEECH_SEC,
    AUDIO_VAD_PADDING_SEC,
    AUDIO_VAD_RMS_THRESHOLD,
    BEAM_SIZE,
    CUSTOM_PRESET,
    DEFAULT_MODEL_ID,
    DEFAULT_PRESET,
    DEVICE,
    DOWNLOAD_LOCK as _DOWNLOAD_LOCK,
    ENHANCE_MAX_TEXT_LEN as _ENHANCE_MAX_TEXT_LEN,
    GPU_PRECISION,
    INITIAL_PROMPT,
    INITIAL_PROMPT_ENABLED,
    INITIAL_PROMPT_EXTRA,
    LOG_PREFIX,
    LOGGER,
    MODEL_FILE_NAMES,
    MODEL_LOCK as _MODEL_LOCK,
    MODEL_SIZES,
    MODELS_WITHOUT_TRANSLATE,
    PROMPT_TARGETS,
    SOURCE_LANGUAGE,
    SUPER_PROMPT_ATTENTION_MODE,
    SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT,
    SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES,
    SUPER_PROMPT_ENHANCE_ON_EXECUTE,
    SUPER_PROMPT_HF_ENDPOINT,
    SUPER_PROMPT_HF_TOKEN,
    SUPER_PROMPT_MAX_IMAGE_SIZE,
    SUPER_PROMPT_MAX_NEW_TOKENS,
    SUPER_PROMPT_MODEL_HUIHUI_2B,
    SUPER_PROMPT_MODEL_OPTIONS,
    SUPER_PROMPT_MODEL_QWEN_2B,
    SUPER_PROMPT_OFFLINE_MODE,
    SUPER_PROMPT_PRECISION,
    SUPER_PROMPT_SEED,
    SUPER_PROMPT_TARGET,
    SUPER_PROMPT_UNLOAD_AFTER_GENERATION,
    TEMPERATURE,
    TRANSLATE_TO_ENGLISH,
    VOICE_EVENT_PREFIX,
    VOICE_LOG_PREFIX,
    VOICE_MODEL_BASE,
    VOICE_MODEL_CACHE as _VOICE_MODEL_CACHE,
    VOICE_MODEL_HIGH_QUALITY,
    VOICE_ROUTE_BASE,
    VOICE_UPLOAD_MAX_BYTES as _VOICE_UPLOAD_MAX_BYTES,
    WHISPER_DIR,
    directory_size as _directory_size,
    format_bytes as _format_bytes,
    log_info as _log_info,
    log_warning as _log_warning,
    register_get as _register_get,
    register_post as _register_post,
    send_ai_event as _send_ai_event,
    send_done as _send_done,
    send_error as _send_error,
    send_progress as _send_progress,
    send_voice_event as _send_voice_event,
    send_voice_status as _send_voice_status,
    voice_log_info as _voice_log_info,
    voice_log_warning as _voice_log_warning,
)

# ---------------------------------------------------------------------------
# Re-exports: voice pipeline
# ---------------------------------------------------------------------------
from .super_prompt._voice import (  # noqa: F401
    AudioPreprocessResult,
    ProgressBroadcaster,
    _adaptive_vad_thresholds,
    _apply_edge_fade,
    _as_float32_audio,
    _audio_metadata,
    _audio_tmp_dir,
    _clean_transcription_text,
    _configured_initial_prompt,
    _detect_speech_bounds,
    _DUPLICATE_TRANSCRIPTION_WORDS,
    _ensure_runtime_dirs,
    _frame_rms,
    _get_ffmpeg_executable,
    _load_whisper_runtime,
    _missing_runtime_packages,
    _model_file_path,
    _normalize_audio,
    _parse_bool,
    _preprocess_audio,
    _read_audio,
    _resolve_voice_model,
    ensure_model,
    is_model_cached,
    load_model,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Re-exports: Qwen pipeline
# ---------------------------------------------------------------------------
from .super_prompt._qwen import (  # noqa: F401
    QwenDownloadProgressMonitor,
    _apply_chat_template_no_thinking,
    _build_messages,
    _chat_template_functions,
    _clean_model_output,
    _filter_generation_params,
    _flatten_text_messages,
    _generate_with_filtered_kwargs,
    _generate_with_qwen,
    _get_qwen_engine,
    _is_qwen_model_available,
    _load_presets,
    _messages_have_visuals,
    _normalize_generation_params,
    _qwen_download_estimate,
    _qwen_model_dir,
    _resolve_preset,
    _target_instruction,
    _template_accepts_kwargs,
    _unused_model_kwargs_from_error,
    default_preset as _default_preset,
    preset_options as _preset_options,
)

# ---------------------------------------------------------------------------
# Re-exports: public node + HTTP routes
# ---------------------------------------------------------------------------
# Importing this submodule triggers @register_post / @register_get on the
# four routes exactly once. The package loader will re-import the same
# qualified path during discovery, but Python's module cache returns the
# already-imported instance, so decorators do NOT run twice.
from .super_prompt.ts_super_prompt import (  # noqa: F401
    TS_SuperPrompt,
    _read_audio_upload,
    _safe_audio_suffix,
    enhance_endpoint,
    preload_endpoint,
    status_endpoint,
    transcribe_endpoint,
)


# NODE_CLASS_MAPPINGS deliberately left empty here. The real registration lives
# in nodes/llm/super_prompt/ts_super_prompt.py and is discovered by the loader's
# recursive scan. Re-declaring the mapping in this shim made the contract
# snapshot tool (tools/build_node_contracts.py) record this file as the node's
# python_file with api="unknown", because the class is imported (not defined)
# here. Keeping the mappings out makes the snapshot point at the real V3 entry.
NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
