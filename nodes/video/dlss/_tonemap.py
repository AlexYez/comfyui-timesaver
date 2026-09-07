"""Invertible tone map that lets HDR, log and linear pictures through DLSS.

The network was trained on display-referred SDR and the worker takes 8-bit RGBA.
Anything else — PQ/HLG frames, camera log, a linear EXR render — must be turned
into a normal-looking SDR picture first and turned back afterwards, or DLSS
"enhances" a washed-out log image and the result cannot be undone::

    source code -> transfer^-1 -> linear x (1.0 = reference white)
                -> knee curve  -> display linear in [0, 1)
                -> BT.709 OETF -> SDR                      (before the worker)

and the exact inverse after it. Every step is per channel and strictly
monotonic, so the round trip is exact up to 16-bit quantisation. Primaries are
never converted: a gamut change would clip and could not be undone.

The knee is identity up to 0.6 and then a logarithmic squeeze of the excess that
reaches 1.0 exactly at the curve's headroom, with a continuous slope at the knee —
every stop above the knee gets the same share of the remaining codes.

Vendored from the reference application (``src/video/tonemap.py``), trimmed to
what a ComfyUI node needs: no stream metadata, no FFmpeg tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

KNEE = 0.6
HDR_REFERENCE_WHITE_NITS = 203.0        # BT.2408: the HDR level that matches SDR white
HLG_PEAK_NITS = 1000.0
HLG_SYSTEM_GAMMA = 1.2
LINEAR_HEADROOM = 64.0                  # EXR: six stops above white are kept

#: What the node offers, in the order it offers it. "sdr" means "leave alone".
TRANSFER_CHOICES = (
    "sdr", "linear", "pq", "hlg", "slog3", "vlog", "logc3", "logc4", "log3g10",
)
TRANSFER_LABELS = {
    "sdr": "SDR (sRGB / BT.709) — no conversion",
    "linear": "Linear (scene-linear, EXR)",
    "pq": "HDR10 PQ (SMPTE ST 2084)",
    "hlg": "HLG (ARIB STD-B67)",
    "slog3": "Sony S-Log3",
    "vlog": "Panasonic V-Log",
    "logc3": "ARRI LogC3 (EI 800)",
    "logc4": "ARRI LogC4",
    "log3g10": "RED Log3G10",
}


# ----------------------------------------------------------------- curves
def _pq_to_nits(code: np.ndarray) -> np.ndarray:
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
    p = np.power(np.clip(code, 0.0, 1.0), 1.0 / m2)
    return 10000.0 * np.power(np.maximum(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1)


def _nits_to_pq(nits: np.ndarray) -> np.ndarray:
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
    y = np.power(np.clip(nits, 0.0, 10000.0) / 10000.0, m1)
    return np.power((c1 + c2 * y) / (1.0 + c3 * y), m2)


def _hlg_to_scene(code: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    code = np.clip(code, 0.0, 1.0)
    return np.where(code <= 0.5, code * code / 3.0, (np.exp((code - c) / a) + b) / 12.0)


def _scene_to_hlg(scene: np.ndarray) -> np.ndarray:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    scene = np.clip(scene, 0.0, 1.0)
    return np.where(
        scene <= 1.0 / 12.0,
        np.sqrt(3.0 * scene),
        a * np.log(np.maximum(12.0 * scene - b, 1e-12)) + c,
    )


def _slog3_to_linear(code: np.ndarray) -> np.ndarray:
    x = code * 1023.0
    return np.where(
        x >= 171.2102946929,
        np.power(10.0, (x - 420.0) / 261.5) * 0.19 - 0.01,
        (x - 95.0) * 0.01125 / (171.2102946929 - 95.0),
    )


def _linear_to_slog3(lin: np.ndarray) -> np.ndarray:
    x = np.where(
        lin >= 0.01125,
        420.0 + np.log10(np.maximum(lin + 0.01, 1e-12) / 0.19) * 261.5,
        lin * (171.2102946929 - 95.0) / 0.01125 + 95.0,
    )
    return x / 1023.0


def _vlog_to_linear(code: np.ndarray) -> np.ndarray:
    b, c, d = 0.00873, 0.241514, 0.598206
    return np.where(code < 0.181, (code - 0.125) / 5.6, np.power(10.0, (code - d) / c) - b)


def _linear_to_vlog(lin: np.ndarray) -> np.ndarray:
    b, c, d = 0.00873, 0.241514, 0.598206
    return np.where(lin < 0.01, 5.6 * lin + 0.125, c * np.log10(np.maximum(lin + b, 1e-12)) + d)


def _logc3_to_linear(code: np.ndarray) -> np.ndarray:
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    return np.where(
        code > e * cut + f, (np.power(10.0, (code - d) / c) - b) / a, (code - f) / e
    )


def _linear_to_logc3(lin: np.ndarray) -> np.ndarray:
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    return np.where(lin > cut, c * np.log10(np.maximum(a * lin + b, 1e-12)) + d, e * lin + f)


_LOGC4_A = (2.0 ** 18 - 16.0) / 117.45
_LOGC4_B = (1023.0 - 95.0) / 1023.0
_LOGC4_C = 95.0 / 1023.0
_LOGC4_S = (7.0 * np.log(2.0) * 2.0 ** (7.0 - 14.0 * _LOGC4_C / _LOGC4_B)) / (_LOGC4_A * _LOGC4_B)
_LOGC4_T = (2.0 ** (14.0 * (-_LOGC4_C / _LOGC4_B) + 6.0) - 64.0) / _LOGC4_A


def _logc4_to_linear(code: np.ndarray) -> np.ndarray:
    return np.where(
        code < 0.0,
        code * _LOGC4_S + _LOGC4_T,
        (np.power(2.0, 14.0 * (code - _LOGC4_C) / _LOGC4_B + 6.0) - 64.0) / _LOGC4_A,
    )


def _linear_to_logc4(lin: np.ndarray) -> np.ndarray:
    return np.where(
        lin < _LOGC4_T,
        (lin - _LOGC4_T) / _LOGC4_S,
        (np.log2(np.maximum(_LOGC4_A * lin + 64.0, 1e-12)) - 6.0) / 14.0 * _LOGC4_B + _LOGC4_C,
    )


def _log3g10_to_linear(code: np.ndarray) -> np.ndarray:
    a, b, c, g = 0.224282, 155.975327, 0.01, 15.1927
    return np.where(code < 0.0, code / g - c, (np.power(10.0, code / a) - 1.0) / b - c)


def _linear_to_log3g10(lin: np.ndarray) -> np.ndarray:
    a, b, c, g = 0.224282, 155.975327, 0.01, 15.1927
    x = lin + c
    return np.where(x < 0.0, x * g, a * np.log10(x * b + 1.0))


def bt709_oetf(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin < 0.018, 4.5 * lin, 1.099 * np.power(lin, 0.45) - 0.099)


def bt709_eotf(code: np.ndarray) -> np.ndarray:
    code = np.clip(code, 0.0, 1.0)
    return np.where(code < 0.081, code / 4.5, np.power((code + 0.099) / 1.099, 1.0 / 0.45))


def knee_slope(headroom: float, knee: float = KNEE) -> float:
    """Compression rate s chosen so the slope is continuous at the knee."""
    big = max((headroom - knee) / (1.0 - knee), 1e-6)
    s = 1.0
    for _ in range(200):
        s = float(np.log1p(big * s))
        if s < 1e-9:
            break
    return max(s, 1e-9)


def knee_forward(x: np.ndarray, headroom: float = LINEAR_HEADROOM, knee: float = KNEE) -> np.ndarray:
    big = max((headroom - knee) / (1.0 - knee), 1e-6)
    s = knee_slope(headroom, knee)
    x = np.maximum(x, 0.0)
    u = np.clip((x - knee) / (1.0 - knee), 0.0, big)
    return np.where(x <= knee, x, knee + (1.0 - knee) * np.log1p(u * s) / np.log1p(big * s))


def knee_inverse(y: np.ndarray, headroom: float = LINEAR_HEADROOM, knee: float = KNEE) -> np.ndarray:
    big = max((headroom - knee) / (1.0 - knee), 1e-6)
    s = knee_slope(headroom, knee)
    y = np.clip(y, 0.0, 1.0)
    u = np.expm1(np.clip((y - knee) / (1.0 - knee), 0.0, 1.0) * np.log1p(big * s)) / s
    return np.where(y <= knee, y, knee + u * (1.0 - knee))


@dataclass(frozen=True)
class Transfer:
    """One source curve: code <-> linear (1.0 = reference white) plus its headroom."""

    name: str
    decode: Callable[[np.ndarray], np.ndarray]
    encode: Callable[[np.ndarray], np.ndarray]
    headroom: float


def _log_transfer(name: str, decode: Callable, encode: Callable) -> Transfer:
    return Transfer(name, decode, encode, float(decode(np.array([1.0]))[0]))


def transfer_for(name: str) -> Transfer:
    """The curve for a transfer name. ``sdr`` has none — it needs no conversion."""
    if name == "pq":
        white = HDR_REFERENCE_WHITE_NITS
        return Transfer(
            "pq",
            lambda c: _pq_to_nits(c) / white,
            lambda x: _nits_to_pq(x * white),
            10000.0 / white,
        )
    if name == "hlg":
        white = HDR_REFERENCE_WHITE_NITS

        def decode(code: np.ndarray) -> np.ndarray:
            return HLG_PEAK_NITS * np.power(_hlg_to_scene(code), HLG_SYSTEM_GAMMA) / white

        def encode(x: np.ndarray) -> np.ndarray:
            return _scene_to_hlg(
                np.power(np.clip(x * white / HLG_PEAK_NITS, 0.0, 1.0), 1.0 / HLG_SYSTEM_GAMMA)
            )

        return Transfer("hlg", decode, encode, HLG_PEAK_NITS / white)
    if name == "slog3":
        return _log_transfer(name, _slog3_to_linear, _linear_to_slog3)
    if name == "vlog":
        return _log_transfer(name, _vlog_to_linear, _linear_to_vlog)
    if name == "logc3":
        return _log_transfer(name, _logc3_to_linear, _linear_to_logc3)
    if name == "logc4":
        return _log_transfer(name, _logc4_to_linear, _linear_to_logc4)
    if name == "log3g10":
        return _log_transfer(name, _log3g10_to_linear, _linear_to_log3g10)
    if name == "linear":
        return Transfer(
            "linear",
            lambda c: c * LINEAR_HEADROOM,
            lambda x: x / LINEAR_HEADROOM,
            LINEAR_HEADROOM,
        )
    raise ValueError(f"No tone-mapping curve for transfer {name!r}.")


def needs_tone_map(name: str) -> bool:
    return name not in ("", "sdr")


@dataclass
class ToneMap:
    """Forward and inverse tables for one source curve.

    ComfyUI hands over float32 in [0, 1], so the float paths are what this node
    uses: ``forward`` takes the source-coded value and returns display SDR,
    ``inverse`` takes display SDR back to the source coding. Both keep the range
    [0, 1] so the result is still a valid IMAGE.
    """

    transfer: Transfer
    knee: float = KNEE

    def forward(self, coded: np.ndarray) -> np.ndarray:
        """Source-coded float [0, 1] -> display-referred SDR float [0, 1]."""
        linear = self.transfer.decode(np.clip(coded.astype(np.float64), 0.0, 1.0))
        sdr = bt709_oetf(knee_forward(linear, self.transfer.headroom, self.knee))
        return np.clip(sdr, 0.0, 1.0).astype(np.float32)

    def inverse(self, sdr: np.ndarray) -> np.ndarray:
        """Display-referred SDR float [0, 1] -> source-coded float [0, 1]."""
        linear = knee_inverse(
            bt709_eotf(np.clip(sdr.astype(np.float64), 0.0, 1.0)),
            self.transfer.headroom,
            self.knee,
        )
        back = np.nan_to_num(self.transfer.encode(linear), nan=0.0)
        return np.clip(back, 0.0, 1.0).astype(np.float32)


def tone_map_for(name: str) -> ToneMap:
    return ToneMap(transfer_for(name))
