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

#: One pass covers this many seconds of sound — and the number is not ours to
#: choose freely. Google's Gemma documentation is explicit: "Audio supports a
#: maximum length of 30 seconds", at 25 tokens per second.
#:
#: ⚠️ This runtime does NOT enforce it. Measured: 85 s went through and returned
#: a sensible transcript that carried on past the 30-second mark, and only at
#: 90 s did it stop with "Input token ids are too long: 4688 >= 4096". That is a
#: temptation to resist — past 30 s the clip is outside what the audio encoder
#: was trained on, and "it did not error" is not the same as "it heard all of
#: it". We segment at the documented boundary.
#:
#: Segmenting costs nothing in completeness: the same 60 seconds gave 141 words
#: in two segments against 140 words in a single oversized pass. Shorter cuts
#: are worse — at 20 s and 12 s the text stutters at the seams ("можете выбрать
#: вот, можете выбрать вот"), because every seam is a chance to repeat.
SEGMENT_SECONDS = 30.0

#: Hard ceiling on how much sound one call will transcribe.
#:
#: ⚠️ Not a model limit — segmentation handles length, and a 70-second recording
#: was verified end to end. This is the backstop for a recording nobody meant to
#: make: the node's own UI stops the microphone at three minutes, but a request
#: can arrive from a graph, a script, or a browser tab that froze mid-recording,
#: and an hour of audio would sit there transcribing for a quarter of an hour.
#: Anything longer is cut here, and the caller is told it was.
MAX_TRANSCRIBE_SECONDS = 300.0

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
    "You are a transcriber. Write out every word that is spoken, from the first to "
    "the last, exactly as it was said.\n"
    "\n"
    "LANGUAGE. The speech is usually Russian, and Russian words are written in "
    "Cyrillic. Never translate, never paraphrase, never 'improve' the wording.\n"
    "\n"
    "TECHNICAL TERMS AND NAMES STAY IN LATIN SCRIPT. Russian speech about software "
    "is full of English terms, and they are written the way the industry writes "
    "them - inside the Russian sentence, in Latin letters, with the original "
    "capitalisation:\n"
    "  ComfyUI, Stable Diffusion, LoRA, VAE, ControlNet, checkpoint, workflow, "
    "prompt, latent, upscale, inpaint, seed, sampler, batch.\n"
    "So: 'Мы используем ComfyUI и LoRA' - never 'Мы используем комфиюай и лору'. "
    "The same goes for product and brand names.\n"
    "\n"
    "A NAME YOU DO NOT KNOW IS STILL A NAME. Write what you actually hear, letter "
    "for letter, instead of snapping it to a word you know better: an unfamiliar "
    "product name is far more likely than a familiar one that sounds almost right. "
    "If the speaker says a name syllable by syllable, keep every syllable.\n"
    "\n"
    "DIRECT SPEECH GOES IN QUOTES. When the speaker quotes someone - themselves, "
    "another person, a line to be read out - put the quoted words in quotation "
    "marks: Он сказал: «Это работает».\n"
    "- Quotation marks, never dashes: «Запусти это», not - Запусти это.\n"
    "- Every opening quote gets a closing one. A quote left open is a mistake, and "
    "it is the one the smaller model makes most often.\n"
    "- Everything outside the quotes stays as it was said, and no quotes are "
    "invented where the speaker was merely talking.\n"
    "\n"
    "PUNCTUATION. Write normal sentences with capital letters, commas and full "
    "stops, the way the pauses and intonation suggest. Filler words the speaker "
    "actually said are kept; stuttering repetitions of a single word are written "
    "once.\n"
    "\n"
    "NEVER:\n"
    "- summarise, shorten or retell. A transcript is not a summary: if the speaker "
    "rambles, the transcript rambles;\n"
    "- add commentary, headings, speaker labels or timestamps;\n"
    "- stop early. Finishing after one tidy sentence is the single most common way "
    "to get this wrong - keep going until the audio ends;\n"
    "- invent words to fill a gap. Write [неразборчиво] there and carry on.\n"
    "\n"
    "Output the transcript and nothing else."
)

TRANSCRIBE_INSTRUCTION = (
    "Transcribe this recording in full. Russian in Cyrillic, technical terms and "
    "names in Latin script, direct speech in quotes."
)


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

    if total > MAX_TRANSCRIBE_SECONDS:
        logger.warning(
            "%s Recording is %.0f s long; transcribing the first %.0f s only.",
            LOG_PREFIX, total, MAX_TRANSCRIBE_SECONDS,
        )
        audio = _slice_audio(audio, 0.0, MAX_TRANSCRIBE_SECONDS)
        total = MAX_TRANSCRIBE_SECONDS
        if progress is not None:
            progress(
                f"Recording longer than {int(MAX_TRANSCRIBE_SECONDS / 60)} min — "
                "transcribing the beginning",
                5.0,
            )

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
