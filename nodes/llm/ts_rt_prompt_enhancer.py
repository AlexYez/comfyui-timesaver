"""TS RT Prompt Enhancer — Gemma 4 on LiteRT-LM as an ordinary graph node.

The workflow sibling of TS Super Prompt RT, the way TS Qwen 3 is the sibling of
TS Super Prompt: no custom interface, no buttons — text, pictures and sound in,
text out, run by the queue like any other node. Everything about the runtime
lives in :mod:`nodes.llm._litert_engine`; the presets and their Gemma-specific
generation settings are the ones TS Super Prompt RT already uses, read from the
same ``qwen_3_vl_presets.json``.

What the engine is, and what was measured on it, shapes the inputs:

* ComfyUI cannot see the memory Gemma takes (WebGPU, not CUDA), so the model is
  unloaded after every run unless ``keep_loaded`` says otherwise.
* Pictures: at most four per message (``max_num_images``). One, two or a whole
  clip's frames go on the same ``images`` input; a longer batch is sampled
  evenly from first to last. There is no VIDEO input on purpose — ComfyUI would
  decode the whole clip into a frame tensor (a minute of 1080p is tens of
  gigabytes) for the four frames the model can take. A clip's sound goes on
  ``audio``.
* Sound: ONE clip per message, and the model is trained on clips of up to 30 s.
  Measured 2026-09-27 on E4B: one clip is heard right, but two or three clips in
  one message came back as the same song described two or three times — so a
  long recording cannot be fed as several clips. Hence two modes: ``listen``
  (the model hears the first 30 s itself — music, sounds, short speech) and
  ``transcribe`` (speech of any length, cut into 30 s segments by TS Super
  Prompt RT's measured transcriber, and the transcript joins the prompt).
* A failure stops the graph with a readable message instead of sending the
  word "ERROR" downstream as a prompt (TS Qwen 3 returns such a string; a
  prompt that says "ERROR: out of memory" only fails later and less clearly).

node_id: TS_RTPromptEnhancer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from comfy_api.v0_0_2 import IO

from ._litert_engine import (
    AUDIO_CLIP_SECONDS,
    CATALOGUE,
    CONTEXT_TOKENS,
    DEFAULT_MODEL,
    cleanup,
    context_error,
    ensure_model,
    estimate_prompt_tokens,
    fits_in_context,
    generate,
    stage_audio,
    stage_images,
    unload_engine,
)
from .super_prompt_rt._helpers import (
    CUSTOM_PRESET,
    DEFAULT_PRESET,
    finish_answer,
    load_presets,
    preset_generation_params,
    presets_path,
    resolve_preset,
)
from .super_prompt_rt._voice import audio_duration
from .super_prompt_rt._voice import transcribe as transcribe_audio

logger = logging.getLogger("comfyui_timesaver.ts_rt_prompt_enhancer")
LOG_PREFIX = "[TS RT Prompt Enhancer]"

#: Frames the runtime takes per message (``max_num_images`` in the engine).
MAX_IMAGES = 4

#: The answer ceiling widget: 0 = the preset's own number.
_MAX_ANSWER = CONTEXT_TOKENS // 2

#: What the node does with a connected recording (see the module docstring).
AUDIO_LISTEN = "listen"
AUDIO_TRANSCRIBE = "transcribe"
AUDIO_MODES = [AUDIO_LISTEN, AUDIO_TRANSCRIBE]

#: Sampling settings for a system prompt typed by hand — the Qwen node's, so a
#: "Your instruction" run behaves the same whichever engine is behind it.
_CUSTOM_GEN_PARAMS = {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.0}


def preset_options() -> list[str]:
    """Every preset in the file, then the hand-typed one — the Qwen node's list."""
    _presets, keys = load_presets()
    options = [key for key in keys if key != CUSTOM_PRESET]
    return (options or [DEFAULT_PRESET]) + [CUSTOM_PRESET]


def sample_frames(image: Any, limit: int = MAX_IMAGES) -> Any:
    """Up to ``limit`` frames, spread evenly over the batch, first and last kept.

    A clip handed in as a batch should be seen from start to end, not as its
    first four frames. Returns the tensor unchanged when it already fits.
    """
    if image is None or getattr(image, "ndim", 0) != 4:
        return image
    count = int(image.shape[0])
    if count <= limit:
        return image
    if limit == 1:
        return image[:1]
    step = (count - 1) / (limit - 1)
    indices = sorted({round(i * step) for i in range(limit)})
    return image[indices]


def resolve_system(system_preset: str, custom_system_prompt: str | None) -> tuple[str, str, dict[str, Any]]:
    """``(label, system_prompt, generation_params)`` for this run.

    "Your instruction" with an empty custom prompt falls back to the default
    preset rather than running the model with no instructions at all — the
    Qwen node does the same.
    """
    custom = str(custom_system_prompt or "").strip()
    if system_preset == CUSTOM_PRESET and custom:
        return CUSTOM_PRESET, custom, preset_generation_params(CUSTOM_PRESET, dict(_CUSTOM_GEN_PARAMS))
    if system_preset == CUSTOM_PRESET:
        logger.warning("%s '%s' is selected but no custom system prompt is connected; using %r.",
                       LOG_PREFIX, CUSTOM_PRESET, DEFAULT_PRESET)
        system_preset = DEFAULT_PRESET
    resolved, system_prompt, gen_params = resolve_preset(system_preset)
    return resolved, system_prompt, preset_generation_params(resolved, gen_params)


def _progress_reporter() -> Any:
    """Node progress bar in ComfyUI, or nothing outside it (tests)."""
    try:
        import comfy.utils

        bar = comfy.utils.ProgressBar(100)
    except Exception:  # noqa: BLE001 - no ComfyUI, no bar
        return None

    def report(_stage: str, percent: float | None = None) -> None:
        # The transcriber sometimes reports a stage without a number.
        if percent is not None:
            bar.update_absolute(int(percent), 100)

    return report


class TS_RTPromptEnhancer(IO.ComfyNode):
    """Gemma 4 prompt/vision node on the LiteRT-LM runtime."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        options = preset_options()
        return IO.Schema(
            node_id="TS_RTPromptEnhancer",
            display_name="TS RT Prompt Enhancer",
            category="TS/LLM",
            description=(
                "Gemma 4 on LiteRT-LM as a graph node: enhance prompts, describe pictures "
                "and sound, write out speech, translate — driven by the same presets as "
                "TS Super Prompt RT."
            ),
            inputs=[
                IO.Combo.Input(
                    "model",
                    options=list(CATALOGUE),
                    default=DEFAULT_MODEL,
                    tooltip=(
                        "E4B writes better, E2B is about twice as fast and lighter. "
                        "Downloaded on first use into models/LLM/litert."
                    ),
                ),
                IO.Combo.Input(
                    "system_preset",
                    options=options,
                    default=DEFAULT_PRESET if DEFAULT_PRESET in options else options[0],
                    tooltip=(
                        "What the model is asked to do. The same presets as TS Super Prompt RT "
                        f"and TS Qwen 3. '{CUSTOM_PRESET}' uses the connected custom_system_prompt."
                    ),
                ),
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Your idea, text or question. May be left empty when pictures or a "
                        "recording are connected."
                    ),
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    # The usual ComfyUI range; the engine folds it into the
                    # runtime's 31 bits (SEED_MODULUS). Declared with
                    # control_after_generate so the saved widget list matches
                    # what the frontend draws next to every seed (TS Qwen 3).
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Same seed, same settings, same answer. Change it for a different wording.",
                ),
                IO.Int.Input(
                    "max_new_tokens",
                    default=0,
                    min=0,
                    max=_MAX_ANSWER,
                    step=64,
                    tooltip=(
                        "Longest answer in tokens. 0 = the preset's own limit. The model stops "
                        "there even mid-sentence — it is a guard against a runaway answer."
                    ),
                ),
                IO.Boolean.Input(
                    "keep_loaded",
                    default=False,
                    tooltip=(
                        "Keep Gemma in memory between runs (faster repeats). ComfyUI CANNOT see "
                        "this memory — it runs on WebGPU, not CUDA — and will plan other models "
                        "as if it were free. Leave off unless the card has room to spare."
                    ),
                ),
                IO.Boolean.Input(
                    "enable",
                    default=True,
                    tooltip="Off: the prompt passes through unchanged and no model is loaded.",
                ),
                IO.Combo.Input(
                    "audio_mode",
                    options=AUDIO_MODES,
                    default=AUDIO_LISTEN,
                    tooltip=(
                        "What to do with a connected recording.\n"
                        "listen — the model hears it itself: music, sounds, a short phrase. "
                        f"Only the first {int(AUDIO_CLIP_SECONDS)} s: that is the longest clip "
                        "the model is trained on.\n"
                        "transcribe — speech of any length (up to 5 min) is written out in "
                        f"{int(AUDIO_CLIP_SECONDS)} s pieces and added to the prompt. Tuned for "
                        "Russian speech with English terms."
                    ),
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        f"One picture, two, or a whole clip's frames. At most {MAX_IMAGES} go "
                        "in: a longer batch is sampled evenly from first to last. Each is "
                        "shrunk to 1024 px — every pixel costs context."
                    ),
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Optional recording — a song, a sound, a voice note, a clip's "
                        "soundtrack. audio_mode decides whether the model listens to it or "
                        "transcribes it."
                    ),
                ),
                IO.String.Input(
                    "custom_system_prompt",
                    multiline=True,
                    force_input=True,
                    optional=True,
                    tooltip=f"Your own system prompt. Used when system_preset is '{CUSTOM_PRESET}'.",
                ),
            ],
            outputs=[IO.String.Output(display_name="text", tooltip="The model's answer.")],
            search_aliases=[
                "gemma", "litert", "prompt enhancer", "rt prompt", "caption", "vlm",
                "audio description", "transcribe",
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **_kwargs: Any) -> Any:
        """The preset file is an input too: edit it and the node runs again."""
        path = presets_path()
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    @classmethod
    def validate_inputs(cls, **_kwargs: Any) -> bool:
        """Accept any saved combo value — a renamed preset or model must not stop
        a workflow from loading; ``execute`` falls back and says so in the log."""
        return True

    @classmethod
    def execute(
        cls,
        model: str = DEFAULT_MODEL,
        system_preset: str = DEFAULT_PRESET,
        prompt: str = "",
        seed: int = 0,
        max_new_tokens: int = 0,
        keep_loaded: bool = False,
        enable: bool = True,
        audio_mode: str = AUDIO_LISTEN,
        images: Any = None,
        audio: Any = None,
        custom_system_prompt: str | None = None,
    ) -> IO.NodeOutput:
        text = str(prompt or "").strip()
        if not enable:
            return IO.NodeOutput(text)

        model_key = model if model in CATALOGUE else DEFAULT_MODEL
        if model_key != model:
            logger.warning("%s Unknown model %r, using %r.", LOG_PREFIX, model, model_key)

        frames = sample_frames(images)
        seconds = audio_duration(audio)
        listen = seconds > 0 and audio_mode != AUDIO_TRANSCRIBE
        transcribe = seconds > 0 and audio_mode == AUDIO_TRANSCRIBE
        if not text and frames is None and seconds <= 0:
            raise RuntimeError(
                f"{LOG_PREFIX} Nothing to work from: the prompt is empty and neither "
                "pictures nor a recording are connected."
            )

        progress = _progress_reporter()
        staged: list[Path] = []
        try:
            if transcribe:
                # The transcriber never downloads by itself (TS Super Prompt
                # RT's record button has its own dialog), but the tooltip here
                # promises a download on first use — so it happens here.
                ensure_model(model_key, allow_download=True)
                # Keeps the model for the enhancement that follows; the unload
                # comes from `generate`, or from the handler below on a failure.
                spoken = transcribe_audio(
                    audio, model_key=model_key, keep_loaded=True, progress=progress,
                )
                if not spoken:
                    logger.warning("%s The recording has no speech the model could write out.", LOG_PREFIX)
                text = "\n\n".join(part for part in (text, spoken) if part)
                if not text and frames is None:
                    raise RuntimeError(
                        f"{LOG_PREFIX} Nothing to work from: the recording has no speech "
                        "and no prompt or pictures are connected."
                    )

            label, system_prompt, params = resolve_system(system_preset, custom_system_prompt)
            if int(max_new_tokens or 0) > 0:
                params["max_new_tokens"] = min(int(max_new_tokens), _MAX_ANSWER)

            heard = min(seconds, AUDIO_CLIP_SECONDS) if listen else 0.0
            if listen and seconds > AUDIO_CLIP_SECONDS:
                logger.warning(
                    "%s The recording is %.0f s long; the model hears the first %.0f s "
                    "(the longest clip it is trained on). For speech, use audio_mode "
                    "'transcribe'.", LOG_PREFIX, seconds, AUDIO_CLIP_SECONDS,
                )
            image_count = int(frames.shape[0]) if frames is not None else 0
            prompt_tokens = estimate_prompt_tokens(
                f"{system_prompt}\n{text}", images=image_count, audio_seconds=heard,
            )
            if not fits_in_context(prompt_tokens, params["max_new_tokens"]):
                raise RuntimeError(context_error(prompt_tokens, params["max_new_tokens"]))

            pictures = stage_images(frames, limit=MAX_IMAGES) if frames is not None else []
            staged.extend(pictures)
            sound = None
            if listen:
                sound, _written = stage_audio(audio, max_seconds=AUDIO_CLIP_SECONDS)
                if sound is not None:
                    staged.append(sound)

            if text:
                request = text
            elif frames is not None:
                # TS Super Prompt RT's wording for a picture with no text, so
                # the two nodes answer the same request the same way.
                request = "Write the prompt for the attached reference."
            else:
                request = "Write the prompt for the attached audio."

            result = generate(
                model_key=model_key,
                system_prompt=system_prompt,
                user_text=request,
                images=pictures,
                audio=sound,
                seed=int(seed),
                keep_loaded=bool(keep_loaded),
                allow_download=True,
                on_progress=progress,
                **params,
            )
        except BaseException:
            # A transcription left the model loaded for a generation that is not
            # coming; `generate` unloads after itself, this covers the gap.
            if transcribe and not keep_loaded:
                unload_engine()
            raise
        finally:
            cleanup(staged)

        answer = finish_answer(label, str(result.get("text", "")).strip(),
                               text if frames is None else None)
        benchmark = result.get("benchmark") or {}
        logger.info(
            "%s %s, %s on %s: %s tokens in %.1f s",
            LOG_PREFIX, label, model_key, result.get("backend"),
            benchmark.get("decode_tokens"), float(result.get("seconds") or 0.0),
        )
        if not answer:
            # Not an error by itself: OCR of a picture without text measured
            # exactly this. Said in the log so an empty prompt is explained.
            logger.warning("%s The model returned an empty answer (%s).", LOG_PREFIX, label)
        return IO.NodeOutput(answer)


NODE_CLASS_MAPPINGS = {"TS_RTPromptEnhancer": TS_RTPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_RTPromptEnhancer": "TS RT Prompt Enhancer"}
