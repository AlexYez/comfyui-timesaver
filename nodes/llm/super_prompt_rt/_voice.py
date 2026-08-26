"""Speech to text through Gemma itself — no Whisper, no numba, no second model.

The same artefact that writes the prompt also hears the recording, which is why
this node needs no separate ASR stack at all. Two measured facts shape the code:

* **Audio costs ~28 prompt tokens per second of sound.** With a 4096-token
  window and a system instruction on top, a single pass is safe up to roughly
  half a minute. Anything longer is cut into segments and stitched — the same
  approach the vendor's own studio takes, for the same reason.
* **The model will summarise if you let it.** A first measurement returned one
  tidy sentence for 25 seconds of speech: 21 decoded tokens where the audio had
  far more to say. The instruction below is written against exactly that
  failure — transcribe, do not summarise, do not translate, do not tidy.

The loader skips ``_``-prefixed modules, so this is never registered as a node.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("comfyui_timesaver.super_prompt_rt.voice")
LOG_PREFIX = "[TS Super Prompt RT voice]"

#: One pass covers this many seconds of sound. 30 s ≈ 840 prompt tokens, which
#: leaves the window comfortable even with the instruction and a long answer.
#:
#: ⚠️ Measured against 20 s and 12 s on the same minute of speech: all three
#: returned the same amount of text (98/100/104 words), but the shorter cuts
#: stuttered at the seams — "можете выбрать вот, можете выбрать вот". Longer
#: segments mean fewer seams, so 30 s wins on quality, not on speed.
SEGMENT_SECONDS = 30.0

#: A transcript this thin for its length suggests the model stopped early rather
#: than the speaker being quiet. Measured over five 30-second stretches of the
#: same recording: 90, 52, 134, 132 and 0 words per minute. The 0 was genuine
#: near-silence (RMS three times lower than the rest), the 52 was the stretch
#: that kept coming back truncated, and everything healthy sat above 90. A
#: threshold here separates the two without touching the quiet passages.
#:
#: ⚠️ The retry is a cheap second chance, NOT a guarantee: on that same stretch
#: it came back the same length as often as not. It is worth a few seconds
#: because the alternative is silently losing half a dictation.
_MIN_WORDS_PER_MINUTE = 55

#: Overlap between segments, so a word split across the cut is heard whole at
#: least once. Cheap insurance: 1 s costs ~28 tokens in one extra segment.
SEGMENT_OVERLAP = 1.0

TRANSCRIBE_SYSTEM = (
    "You are a transcriber. Write out every word that is spoken in the recording, "
    "in the language it was spoken in, from the first word to the last.\n"
    "\n"
    "- Do NOT summarise, shorten, retell or tidy up. A transcript is not a summary: "
    "if the speaker rambles, the transcript rambles.\n"
    "- Do NOT translate. Russian speech is written in Russian.\n"
    "- Do NOT add commentary, headings, speaker labels, timestamps or quotation marks "
    "around the whole thing.\n"
    "- Keep going until the audio ends. Stopping early after one neat sentence is the "
    "single most common way to get this wrong.\n"
    "- If a passage is genuinely unintelligible, write [unintelligible] there and carry on. "
    "Never invent words to fill a gap.\n"
    "\n"
    "Output the transcript and nothing else."
)

TRANSCRIBE_INSTRUCTION = "Transcribe this recording in full."


def audio_duration(audio: dict[str, Any] | None) -> float:
    if not audio:
        return 0.0
    waveform = audio.get("waveform")
    rate = float(audio.get("sample_rate") or 0)
    if waveform is None or rate <= 0:
        return 0.0
    return float(waveform.shape[-1]) / rate


def _slice_audio(audio: dict[str, Any], start: float, seconds: float) -> dict[str, Any]:
    rate = int(audio["sample_rate"])
    waveform = audio["waveform"]
    begin = max(0, int(start * rate))
    end = min(waveform.shape[-1], int((start + seconds) * rate))
    return {"waveform": waveform[..., begin:end], "sample_rate": rate}


def transcribe(
    audio: dict[str, Any],
    *,
    model_key: str,
    backend: str = "GPU",
    keep_loaded: bool = False,
    progress: Callable[[str, float | None], None] | None = None,
) -> str:
    """Full transcript of ``audio``, segmenting when it is longer than a window.

    ``keep_loaded`` is honoured only BETWEEN segments — the model stays resident
    while the segments of one recording are processed, because reloading a 3 GB
    artefact per 30 seconds of speech would dominate the runtime. The final
    unload still happens unless the caller asked to keep it.
    """
    from .._litert_engine import cleanup, generate, stage_audio, unload_engine

    total = audio_duration(audio)
    if total <= 0:
        return ""

    starts: list[float] = [0.0]
    if total > SEGMENT_SECONDS:
        step = SEGMENT_SECONDS - SEGMENT_OVERLAP
        starts = [index * step for index in range(int((total - SEGMENT_OVERLAP) // step) + 1)]

    pieces: list[str] = []
    try:
        for index, start in enumerate(starts):
            if progress is not None:
                progress(
                    f"Transcribing {index + 1}/{len(starts)}",
                    10.0 + 80.0 * index / max(1, len(starts)),
                )
            chunk = _slice_audio(audio, start, SEGMENT_SECONDS) if len(starts) > 1 else audio
            path, seconds = stage_audio(chunk, max_seconds=SEGMENT_SECONDS)
            if path is None or seconds <= 0.1:
                continue
            try:
                text = _transcribe_one(path, model_key=model_key, backend=backend)
                # ⚠️ One retry when the pass came back too thin for the length of
                # audio. Measured: the same 25 seconds returned 17 words once and
                # a full paragraph the next time — the model sometimes decides it
                # is finished after one tidy sentence. Retrying costs a few
                # seconds; leaving half a dictation on the floor costs the user
                # their sentence.
                if text and _looks_truncated(text, seconds):
                    logger.info(
                        "%s Segment %d looked truncated (%d words for %.0f s), retrying.",
                        LOG_PREFIX, index + 1, len(text.split()), seconds,
                    )
                    again = _transcribe_one(
                        path, model_key=model_key, backend=backend, insist=True,
                    )
                    if len(again.split()) > len(text.split()):
                        text = again
            finally:
                cleanup([path])
            if text:
                pieces.append(text)
    finally:
        if not keep_loaded:
            unload_engine()

    return _stitch(pieces)


def _transcribe_one(
    path: Path, *, model_key: str, backend: str, insist: bool = False,
) -> str:
    """One segment through the model. ``insist`` is the second, firmer attempt."""
    from .._litert_engine import generate

    instruction = TRANSCRIBE_INSTRUCTION
    if insist:
        instruction = (
            "Transcribe this recording in full, from the first word to the last. "
            "The previous attempt stopped early — do not stop until the audio ends."
        )
    result = generate(
        model_key=model_key,
        system_prompt=TRANSCRIBE_SYSTEM,
        user_text=instruction,
        audio=path,
        # Low temperature: this is dictation, not writing. The retry lifts it a
        # little, because repeating an identical greedy pass tends to repeat the
        # identical early stop.
        temperature=0.3 if insist else 0.15,
        top_p=0.9,
        top_k=40,
        # Generous ceiling — speech is dense, and the measured failure mode was
        # stopping early, not running long.
        max_new_tokens=1024,
        backend=backend,
        # Held across segments on purpose; released by the caller's `finally`.
        keep_loaded=True,
        allow_download=False,
    )
    return str(result.get("text", "")).strip()


def _looks_truncated(text: str, seconds: float) -> bool:
    if seconds < 5.0:
        return False
    words_per_minute = len(text.split()) / (seconds / 60.0)
    return words_per_minute < _MIN_WORDS_PER_MINUTE


def _stitch(pieces: list[str]) -> str:
    """Join segment transcripts, dropping the overlap when it repeats verbatim.

    The overlap exists so a word cut in half is heard whole; the price is that
    the same few words can appear at the end of one segment and the start of the
    next. Only an exact repeat is removed — anything cleverer would risk eating
    a phrase the speaker really did say twice.
    """
    joined = ""
    for piece in pieces:
        if not joined:
            joined = piece
            continue
        overlap = _longest_overlap(joined, piece)
        joined = f"{joined} {piece[overlap:].lstrip()}".strip() if overlap else f"{joined} {piece}"
    return joined.strip()


def _longest_overlap(left: str, right: str, *, limit: int = 120) -> int:
    tail = left[-limit:]
    for size in range(min(len(tail), len(right)), 10, -1):
        if tail.endswith(right[:size]):
            return size
    return 0
