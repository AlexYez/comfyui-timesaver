"""Where the pack's ffmpeg comes from — one answer, for every node that needs it.

Four places shell out to ffmpeg (the audio loader, the Whisper engine, the
Super Prompt voice path and the animation preview), and each carried its own
copy of "ask imageio_ffmpeg, else say ffmpeg and hope". The copies had drifted:
one logged the failure, two swallowed it, and none of them could tell the
difference between "the binary is missing" and "the binary is here and the file
is broken" — so a machine without ffmpeg produced four different errors, none
of which said what to do about it.

WHY NOT A PURE-PYTORCH FALLBACK. There is nothing to fall back to. What ffmpeg
does here is demux containers, decode compressed audio, resample, and encode
video — torch has no codecs. torchaudio's own decoding is being retired in
favour of TorchCodec, which links FFmpeg's libraries, and torchvision's video
reader is deprecated and FFmpeg-based too, so "use torch instead" means the
same FFmpeg under a different name plus another dependency. The only genuine
alternative is libsndfile (via soundfile) for wav/flac/ogg, which covers
exactly the containers that were never the problem.

The real guarantee is that the pack does not depend on the machine having
ffmpeg at all: `imageio-ffmpeg` is a hard requirement and ships a static binary
for every platform the pack runs on, macOS arm64 included. PATH is the fallback,
not the plan.
"""

from __future__ import annotations

import logging
import os
import shutil

LOGGER = logging.getLogger("comfyui_timesaver.ffmpeg")
LOG_PREFIX = "[TS ffmpeg]"

# What to say when there is genuinely nothing to run. One message, in one place,
# naming the fix rather than the failure.
MISSING_MESSAGE = (
    f"{LOG_PREFIX} No ffmpeg executable found. It normally ships with this pack "
    "through imageio-ffmpeg: run `pip install --upgrade imageio-ffmpeg` with the "
    "Python that runs ComfyUI, or install ffmpeg yourself and put it on PATH."
)

# Resolution is a filesystem probe, so it is done once. Module level rather than
# on a class: ComfyUI's V3 registration locks node classes (CLAUDE.md 5).
class _State:
    path: str | None = None
    resolved: bool = False


_state = _State()


def _resolve() -> str | None:
    """The ffmpeg to use, or None when there is none. Not cached — see below."""
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
        LOGGER.debug("%s imageio-ffmpeg named %r, which is not a file.", LOG_PREFIX, bundled)
    except Exception as exc:
        LOGGER.debug("%s imageio-ffmpeg could not provide a binary: %s", LOG_PREFIX, exc)

    found = shutil.which("ffmpeg")
    if found:
        LOGGER.debug("%s Using the ffmpeg found on PATH: %s", LOG_PREFIX, found)
        return found
    return None


def ffmpeg_executable() -> str:
    """The command to run, always a string.

    Returns the literal ``"ffmpeg"`` when nothing was found, so a caller that
    already handles a failing subprocess keeps behaving as it did. Callers that
    want to fail early and clearly should use :func:`require_ffmpeg` instead.
    """
    if not _state.resolved:
        _state.path = _resolve()
        _state.resolved = True
        if _state.path is None:
            LOGGER.warning("%s", MISSING_MESSAGE)
    return _state.path or "ffmpeg"


def ffmpeg_available() -> bool:
    """True when an ffmpeg binary was actually located."""
    ffmpeg_executable()
    return _state.path is not None


def require_ffmpeg() -> str:
    """The command to run, or a RuntimeError that says how to get one."""
    executable = ffmpeg_executable()
    if _state.path is None:
        raise RuntimeError(MISSING_MESSAGE)
    return executable


def reset_cache() -> None:
    """Forget the resolved path.

    For tests, and for the rare case of a person installing ffmpeg while
    ComfyUI is running — the alternative is asking them to restart.
    """
    _state.path = None
    _state.resolved = False
