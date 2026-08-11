"""Qwen-prompt-enhancement pipeline for TS Super Prompt.

Owns: preset loading from ``qwen_3_vl_presets.json``, message construction
(system + user + optional image content), ``apply_chat_template`` invocation
across processor/tokenizer variants with ``enable_thinking=False``,
generation-param normalisation/filtering, the public
``_generate_with_qwen`` entry point used by the
``/ts_super_prompt/enhance`` HTTP route, and the HF-snapshot download
progress monitor used while Qwen weights stream in.

Heavy runtime (model load/cache, device/memory/precision, chat-template
defaults) lives in :mod:`nodes.llm._qwen_engine` and is shared with
``TS_Qwen3_VL_V3``.

The ``"Your instruction"`` preset is intentionally **hidden from the
TS_SuperPrompt combo** via ``preset_options()`` — there is no widget to
collect a custom system prompt in this node, so showing the preset only
caused confusion (it would silently fall back to ``DEFAULT_PRESET``).
The constant ``CUSTOM_PRESET`` and the supporting branch in
``_resolve_preset`` are kept so external callers / TS_Qwen3_VL_V3 are
unaffected.

``torch`` is imported lazily inside ``_generate_with_qwen`` so contract
tests can stub the engine without forcing torch.

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

from .._qwen_engine import (
    QwenEngine,
    _chat_template_functions,
    _flatten_text_messages,
    _messages_have_visuals,
    _template_accepts_kwargs,
    get_qwen_engine,
)
from .._qwen_engine import (
    apply_chat_template_no_thinking as _engine_apply_chat_template_no_thinking,
)
from .._qwen_engine import (
    load_presets as _qwen_load_presets,
)
from ._helpers import (
    cancel_operation,
    forget_operation,
    operation_cancelled,
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

# ---------------------------------------------------------------------------
# Engine access
# ---------------------------------------------------------------------------

def _get_qwen_engine() -> QwenEngine:
    """Return the process-wide :class:`QwenEngine` singleton.

    Kept as a function (rather than referencing :func:`get_qwen_engine`
    directly at call sites) so contract tests can monkeypatch this name
    without touching the engine module.
    """
    return get_qwen_engine()


class QwenDownloadProgressMonitor:
    """Poll local HuggingFace files while ``snapshot_download`` runs."""

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
        # ⚠️ `total_bytes` на входе — ОЦЕНКА (см. SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES).
        # Настоящий размер спрашивается у HuggingFace в фоне: пока ответа нет,
        # доля считается от оценки, после — от факта. Показывать проценты от
        # выдуманного числа хуже, чем не показывать вовсе, поэтому источник
        # виден и в тексте.
        self._exact_total = 0
        self._history: list[tuple[float, int]] = []

    def start(self) -> None:
        if not self.enabled:
            return
        send_progress(
            self.operation_id,
            f"Connecting to HuggingFace for {self.model_id}",
            self.start_percent,
        )
        # ⚠️ Настоящий размер спрашивается ДО первого показа, иначе доля
        # прыгает назад: пока отвечает оценка, полоса ушла вперёд, а с приходом
        # факта откатывается (замерено — с 30% на 1%). Ждём недолго: сама
        # загрузка идёт минутами, а без ответа просто останемся на оценке.
        lookup = threading.Thread(target=self._resolve_exact_total, daemon=True)
        lookup.start()
        lookup.join(timeout=2.5)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _resolve_exact_total(self) -> None:
        """Спросить у HuggingFace настоящий размер репозитория."""
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(self.model_id, files_metadata=True)
            total = sum(int(getattr(f, "size", 0) or 0) for f in (info.siblings or []))
            if total <= 0:
                return
            # ⚠️ Ответ, опоздавший к первому показу, НЕ принимаем: подменить
            # знаменатель на ходу значит откатить долю назад (замерено — с 30%
            # на 1%). Оценка, честно названная оценкой, лучше прыгающей полосы.
            if self._last_size > 0:
                log_info(f"{self.model_id}: exact size arrived late, keeping the estimate")
                return
            self._exact_total = total
            log_info(f"{self.model_id}: {format_bytes(total)} to download")
        except Exception as error:              # noqa: BLE001 - без сети остаётся оценка
            log_info(f"could not read the exact size of {self.model_id}: {error}")

    def _speed_and_eta(self, size: int, total: int) -> str:
        """Скорость и остаток по последним замерам.

        Считается по окну, а не от начала: в начале загрузки среднее сильно
        занижено распаковкой и проверками, и обещанное время выходит вдвое
        больше правды.
        """
        import time

        now = time.monotonic()
        self._history.append((now, size))
        self._history = [(when, got) for when, got in self._history if now - when <= 12.0]
        if len(self._history) < 2:
            return ""
        (first_t, first_size), (last_t, last_size) = self._history[0], self._history[-1]
        seconds = last_t - first_t
        gained = last_size - first_size
        if seconds <= 0.5 or gained <= 0:
            return ""
        speed = gained / seconds
        parts = [f"{format_bytes(int(speed))}/s"]
        if total > size:
            left = (total - size) / speed
            if left < 90:
                parts.append(f"~{int(left)} s left")
            else:
                parts.append(f"~{int(left / 60)} min left")
        return " · " + " · ".join(parts)

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
            self._stop.wait(0.5)

    def _emit_progress(self) -> None:
        size = directory_size(self.local_dir)
        if size == self._last_size and size > 0:
            return
        self._last_size = size

        total = self._exact_total or self.total_bytes
        ratio = max(0.0, min(1.0, size / float(total)))
        percent = self.start_percent + (self.end_percent - self.start_percent) * ratio

        name = self.model_id.split("/")[-1]
        if size <= 0:
            text = f"Downloading {name}: starting"
        else:
            share = int(round(ratio * 100))
            about = "" if self._exact_total else "about "
            left = max(0, total - size)
            # Сколько уже пришло И сколько осталось — обе цифры сразу: доля в
            # процентах отвечает «долго ли ещё», а мегабайты остатка — «сколько
            # именно», и одно другое не заменяет.
            text = (f"Downloading {name}: {format_bytes(size)} done, "
                    f"{format_bytes(left)} left of {about}{format_bytes(total)} "
                    f"({share}%){self._speed_and_eta(size, total)}")
        send_progress(self.operation_id, text, percent)


def _qwen_model_dir(model_id: str) -> Path:
    return (
        Path(getattr(folder_paths, "models_dir", Path.cwd() / "models"))
        / "LLM"
        / str(model_id).split("/")[-1]
    )


def _qwen_download_estimate(model_id: str) -> int:
    explicit = SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES.get(model_id)
    if explicit:
        return int(explicit)
    try:
        size_b = float(QwenEngine.model_size_b(model_id))
    except Exception:
        size_b = 2.0
    return int(max(1.0, size_b) * 2_250_000_000)


def _is_qwen_model_available(engine: QwenEngine, model_id: str) -> bool:
    try:
        return bool(engine.check_model_integrity(str(_qwen_model_dir(model_id))))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Preset handling — "Your instruction" hidden from TS_SuperPrompt UI
# ---------------------------------------------------------------------------

def _load_presets() -> tuple[dict[str, Any], list[str]]:
    presets, keys = _qwen_load_presets()
    if not isinstance(presets, dict):
        return {}, []
    return presets, [key for key in keys if isinstance(key, str) and key]


def preset_options() -> list[str]:
    """Return preset names shown in the TS_SuperPrompt combo.

    The ``CUSTOM_PRESET`` ("Your instruction") entry is intentionally
    omitted: there is no widget to collect a custom system prompt in this
    node, so the option would silently fall back to the default preset and
    confuse users. ``TS_Qwen3_VL_V3`` still exposes it via its own schema.
    """
    _presets, keys = _load_presets()
    options = [key for key in keys if isinstance(key, str) and key]
    if not options:
        # Fall back to a single visible option so the combo is never empty.
        options = [DEFAULT_PRESET]
    return options


def default_preset(options: list[str]) -> str:
    if DEFAULT_PRESET in options:
        return DEFAULT_PRESET
    return options[0] if options else DEFAULT_PRESET


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


def _frame_label(index: int, total: int) -> str:
    """What to write next to one reference image.

    A lone image gets nothing: calling it the first frame would promise a shot
    that does not exist. Two keep the FIRST/LAST wording the presets are
    written around. Longer sequences are numbered, with the ends still named —
    which frame the motion starts and finishes on is the part that matters.
    """
    if total < 2:
        return ""
    if total == 2:
        return "FIRST FRAME:" if index == 0 else "LAST FRAME:"
    if index == 0:
        return f"FRAME 1 of {total} (FIRST FRAME):"
    if index == total - 1:
        return f"FRAME {total} of {total} (LAST FRAME):"
    return f"FRAME {index + 1} of {total}:"


def _build_messages(
    system_prompt: str,
    text: str,
    prompt_target: str,
    image: Any,
    max_image_size: int,
) -> list[dict[str, Any]]:
    engine = _get_qwen_engine()
    user_content: list[dict[str, Any]] = []

    # Resolved before the text is written, because how many images there
    # are changes what the model has to be told about them.
    pil_images = engine.normalize_to_pil_list(image) if image is not None else []

    if len(pil_images) == 2:
        # Two frames is what the presets speak about, so keep their wording.
        image_note = (
            "Two images are attached. The FIRST is the first frame of the shot "
            "and the SECOND is the last frame. Describe the change from the "
            "first to the second.\n"
        )
    elif len(pil_images) > 2:
        image_note = (
            f"{len(pil_images)} images are attached, in order from the first "
            "frame of the shot to the last. Describe the change across them.\n"
        )
    elif pil_images:
        image_note = "One reference image is attached. Describe what is in it.\n"
    else:
        image_note = ""

    user_text = (
        f"{_target_instruction(prompt_target, bool(pil_images))}\n\n"
        f"{image_note}"
        "Hard rules:\n"
        "- Translate the source idea to English when needed.\n"
        # Text in quotes is text the generator will draw on screen — a sign,
        # a title, a lyric. An obliging translation turns a Russian shop sign
        # into an English one the person never asked for.
        "- EXCEPTION: text inside quotes is on-screen text. Copy it exactly, "
        "character for character, in its original language, quotes included. "
        "Never translate what is quoted.\n"
        "- Keep the user's core meaning, named subjects, and constraints.\n"
        "- Return only the final prompt, with no preface, no analysis, and no markdown.\n"
        "- Do not use thinking mode. Do not output chain-of-thought or hidden reasoning.\n\n"
        f"Source idea:\n{text or ''}"
    )

    system_text = (
        f"{system_prompt.strip()}\n\n"
        "Runtime mode: non-thinking. Produce the answer directly and never include a <think> block."
    )

    # ``image`` is a torch.Tensor (workflow IMAGE input), a PIL.Image, or a list
    # of either — ``normalize_to_pil_list`` flattens all of them, which is what
    # lets the node's two attachment slots arrive here as one ordered list.
    for index, pil_image in enumerate(pil_images):
        label = _frame_label(index, len(pil_images))
        if label:
            # The label goes NEXT TO its image. Said once further down the
            # message, "the first frame" would leave the model to work out
            # which of several pictures that is.
            user_content.append({"type": "text", "text": label})
        user_content.append(
            {
                "type": "image",
                "image": engine.resize_and_crop_image(pil_image, int(max_image_size)),
            }
        )

    user_content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]


# ``_chat_template_functions`` / ``_template_accepts_kwargs`` /
# ``_messages_have_visuals`` / ``_flatten_text_messages`` /
# ``_apply_chat_template_no_thinking`` previously lived here. They moved
# into :mod:`nodes.llm._qwen_engine` so ``TS_Qwen3_VL_V3`` can share the
# same "no chain-of-thought" template plumbing. The names below stay as
# module-level aliases for backward compatibility with the existing test
# suite (which imports them through this module) and for any external code
# that referenced them by their original location.
_apply_chat_template_no_thinking = _engine_apply_chat_template_no_thinking


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


class _CancelledByUser(RuntimeError):
    """Человек нажал «Отмена». Не ошибка — решение."""


# Границы полосы на время генерации. Стадии до и после уже заняли свои
# проценты (см. вызовы send_progress), поэтому написание текста живёт здесь.
_GENERATION_START_PERCENT = 78.0
_GENERATION_END_PERCENT = 92.0
# Каждый токен слать в сокет незачем: их сотни, а глаз разницы не увидит.
_PROGRESS_EVERY_TOKENS = 8


def _stopping_criteria_for(operation_id: str | None, max_new_tokens: int | None = None):
    """Критерий, который на каждом токене делает две вещи: смотрит на отмену и
    двигает полосу.

    ⚠️ Именно так генерацию и останавливают: ``model.generate`` блокирует поток
    целиком, снаружи его не прервать. И ровно поэтому же прогресс во время
    написания текста больше взять негде — а это самая долгая стадия. Без него
    полоса замирала на 78% и прыгала сразу к готовому промпту.
    """
    if not operation_id:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:                       # noqa: BLE001 - без transformers генерации и нет
        return None

    budget = max(1, int(max_new_tokens or 0)) if max_new_tokens else 0
    span = _GENERATION_END_PERCENT - _GENERATION_START_PERCENT
    state = {"seen": 0}

    class _Watch(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            state["seen"] += 1
            seen = state["seen"]
            if seen % _PROGRESS_EVERY_TOKENS == 0:
                if budget:
                    share = min(1.0, seen / budget)
                    percent = _GENERATION_START_PERCENT + span * share
                    send_progress(operation_id, f"Writing the prompt ({seen} tokens)", percent)
                else:
                    # Потолок неизвестен — говорим сколько написано, без процента:
                    # выдуманная доля хуже честной неизвестности.
                    send_progress(operation_id, f"Writing the prompt ({seen} tokens)", None)
            return operation_cancelled(operation_id)

    return StoppingCriteriaList([_Watch()])


def _generate_with_filtered_kwargs(model, inputs: dict[str, Any], gen_params: dict[str, Any],
                                   operation_id: str | None = None):
    current_params = dict(gen_params)
    stopping = _stopping_criteria_for(operation_id, current_params.get("max_new_tokens"))
    if stopping is not None:
        current_params["stopping_criteria"] = stopping
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
    seed: int | None = None,
    interactive: bool = False,
    model_id: str | None = None,
) -> str:
    """Run the shared Qwen engine and return the cleaned prompt text.

    ``seed`` overrides the fixed module seed so a caller that offers a "generate
    again" button gets a different sample each time; ``None`` keeps the historic
    deterministic behaviour. ``interactive`` marks a call that came from a
    button (an HTTP route) rather than from queue execution, which forbids
    evicting the models of a graph that is running right now.
    """
    if not str(text or "").strip() and image is None:
        return ""

    # Ничего не передали — работаем на прежней модели, как и все вызовы,
    # написанные до появления галочки «крупнее модель».
    model_id = str(model_id or DEFAULT_MODEL_ID)

    lock_acquired = MODEL_LOCK.acquire(blocking=False)
    if not lock_acquired:
        send_progress(operation_id, "Waiting for Qwen", 2.0)
        MODEL_LOCK.acquire()

    try:
        send_progress(operation_id, "Preparing prompt", 5.0)
        engine = _get_qwen_engine()
        system_prompt, gen_params = _resolve_preset(system_preset, SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT)
        resolved_precision = engine.resolve_precision(SUPER_PROMPT_PRECISION, model_id)
        resolved_attention = engine.resolve_attention(SUPER_PROMPT_ATTENTION_MODE, resolved_precision)
        estimated_vram = engine.estimate_vram_usage(model_id, resolved_precision)

        log_info(
            f"model={model_id} precision={resolved_precision} "
            f"attention={resolved_attention} thinking=disabled"
        )
        send_progress(operation_id, "Checking memory", 12.0)
        # A button press must never evict the models a running graph is using.
        allow_unload_others = not (interactive and engine.comfy_prompt_is_running())
        if not allow_unload_others and not engine.has_free_vram_for(estimated_vram):
            raise RuntimeError(
                "ComfyUI is running a prompt and there is not enough free VRAM for Qwen. "
                "Wait for the run to finish and press Generate again."
            )
        engine.ensure_memory_available(estimated_vram, allow_unload_others=allow_unload_others)

        send_progress(operation_id, "Checking Qwen model files", 16.0)
        qwen_model_available = _is_qwen_model_available(engine, model_id)
        if qwen_model_available:
            send_progress(operation_id, "Qwen model found locally", 22.0)
        elif SUPER_PROMPT_OFFLINE_MODE:
            send_progress(operation_id, "Using offline Qwen model files", 22.0)
        else:
            send_progress(operation_id, "Qwen model download starting", 20.0)

        if not qwen_model_available and not bool(SUPER_PROMPT_OFFLINE_MODE):
            qwen_monitor = QwenDownloadProgressMonitor(
                operation_id=operation_id,
                model_id=model_id,
                local_dir=_qwen_model_dir(model_id),
                total_bytes=_qwen_download_estimate(model_id),
            )
            qwen_monitor.start()
            download_success = False
            try:
                engine.ensure_model_available(
                    model_id,
                    bool(SUPER_PROMPT_OFFLINE_MODE),
                    str(SUPER_PROMPT_HF_TOKEN or ""),
                    str(SUPER_PROMPT_HF_ENDPOINT or ""),
                )
                download_success = True
            finally:
                qwen_monitor.stop(download_success)

        send_progress(operation_id, "Loading Qwen model into memory", 46.0)
        model, processor = engine.load_model(
            model_id,
            resolved_precision,
            resolved_attention,
            bool(SUPER_PROMPT_OFFLINE_MODE),
            str(SUPER_PROMPT_HF_TOKEN or ""),
            str(SUPER_PROMPT_HF_ENDPOINT or ""),
        )
        send_progress(operation_id, "Qwen model loaded", 50.0)

        target_device = engine.get_device()
        moved_to_gpu = False
        if target_device.type == "cuda" and not engine.model_has_cuda_device(model):
            try:
                send_progress(operation_id, "Moving Qwen to GPU", 54.0)
                engine.ensure_memory_available(
                    estimated_vram, force_unload=True, allow_unload_others=allow_unload_others
                )
                model.to(target_device)
                moved_to_gpu = True
            except RuntimeError as exc:
                if engine.is_oom_error(exc):
                    try:
                        model.to("cpu")
                    except Exception as cleanup_exc:
                        LOGGER.debug("%s OOM cleanup move-to-CPU failed: %s", LOG_PREFIX, cleanup_exc)
                    engine.prepare_memory(force=True)
                    raise RuntimeError("Out of memory during Qwen GPU transfer.") from exc
                raise

        try:
            if image is not None and not engine.supports_multimodal_inputs(processor):
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
            input_device = engine.select_input_device(model)
            inputs = engine.move_inputs_to_device(inputs, input_device)
            engine.log_processing_device("super_prompt_inputs", input_device, model, inputs)

            gen_params = dict(gen_params)
            gen_params.setdefault("temperature", 0.7 if image is not None else 1.0)
            gen_params.setdefault("top_p", 0.8 if image is not None else 1.0)
            gen_params.setdefault("top_k", 20)
            gen_params.setdefault("repetition_penalty", 1.0)
            # A preset that asks for a bigger budget gets it. This used to
            # overwrite the preset unconditionally, so the Ideogram presets —
            # which ask for 1024 tokens of structured JSON — were cut off at
            # 512: the JSON came back unbalanced, failed to parse, and the raw
            # truncated text was passed downstream as if it were a caption.
            gen_params.setdefault("max_new_tokens", int(SUPER_PROMPT_MAX_NEW_TOKENS))
            gen_params["use_cache"] = True
            gen_params["pad_token_id"] = engine.get_pad_token_id(processor, model)
            gen_params["do_sample"] = float(gen_params.get("temperature", 0.0) or 0.0) > 0.0
            gen_params = _normalize_generation_params(gen_params)

            # Local import keeps torch out of module-level eval so contract
            # tests can stub the engine without importing torch.
            import torch

            # A caller that offers "generate again" passes its own seed; without
            # one the historic fixed seed keeps repeat runs reproducible.
            active_seed = int(SUPER_PROMPT_SEED if seed is None else seed)
            rng_cuda_devices = engine.cuda_indices_for_rng(model, input_device)
            if engine.supports_generator(model):
                gen_device = engine.select_generator_device(input_device)
                generator = torch.Generator(device=gen_device)
                generator.manual_seed(active_seed)
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
                if hasattr(input_device, "type")
                else getattr(model, "device", None)
            )
            use_autocast = (
                getattr(autocast_device, "type", None) == "cuda"
                and dtype in (torch.float16, torch.bfloat16)
            )
            gen_params = _filter_generation_params(model, gen_params)

            # TF32 / high fp32-matmul precision is enabled ONLY for this
            # generation and restored on exit, so it never leaks into the
            # samplers of other nodes (see engine.qwen_matmul_precision).
            with engine.qwen_matmul_precision(), rng_context:
                if "generator" not in gen_params:
                    torch.manual_seed(active_seed)
                    for idx in rng_cuda_devices:
                        with torch.cuda.device(idx):
                            torch.cuda.manual_seed(active_seed)

                send_progress(operation_id, "Generating AI prompt", 78.0)
                with torch.inference_mode():
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params, operation_id)
                    else:
                        generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params, operation_id)

            send_progress(operation_id, "Decoding prompt", 92.0)
            generated_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = engine.batch_decode(
                processor,
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            if operation_cancelled(operation_id):
                # Отменённое не отдаём как результат: обрезанный на середине
                # промпт выглядит готовым, и тем он хуже пустого ответа.
                log_info("enhancement cancelled by the user")
                return ""
            return _clean_model_output(output_text)
        finally:
            forget_operation(operation_id)
            if SUPER_PROMPT_UNLOAD_AFTER_GENERATION:
                send_progress(operation_id, "Unloading Qwen", 96.0)
                # Drop our own references first: the engine's unload cannot free
                # the module while these locals still point at it.
                model = None
                processor = None
                engine.unload_model(model_id, resolved_precision, resolved_attention)
            elif moved_to_gpu:
                try:
                    model.to("cpu")
                    engine.prepare_memory(force=True)
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
