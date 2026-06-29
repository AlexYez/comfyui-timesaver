"""TS Audio Visualizer — render a standard ComfyUI AUDIO input into a stylized waveform image.

node_id: TS_AudioVisualizer

The foreground is a SoundCloud-style bar waveform: the clip is reduced to N peak
amplitudes (one per bar) and each bar is drawn as an antialiased rounded
"capsule" using a vectorized signed-distance field over the full ``[H, W]`` grid.
Bars are tinted with a multi-stop colour gradient (blue→violet by default) and
surrounded by a soft neon glow.

Behind the bars sits an optional **audio-reactive abstract background** driven by
the same loudness envelope — a soft waveform aura ("glow"), layered "mountains"
silhouettes, smoky "plasma", or a "nebula" combo. Everything is rendered purely
on ``torch`` (always present in the ComfyUI runtime) — no third-party imaging
libraries.

Outputs:
- IMAGE ``[1, H, W, 3]`` float32 in ``[0, 1]`` — the rendered visualization.
- MASK  ``[1, H, W]``    float32 in ``[0, 1]`` — bar fill + glow alpha (the
  background is NOT included), handy for compositing the bars over video/footage
  and what the ``transparent`` background relies on.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch
import torch.nn.functional as F

import comfy.model_management as mm

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_audio_visualizer")
LOG_PREFIX = "[TS Audio Visualizer]"


# --------------------------------------------------------------------------- #
# Colour presets — each is a list of RGB stops in [0, 1] interpolated along the
# chosen gradient axis. Default tone is blue→violet; the other presets stay for
# flexibility.
# --------------------------------------------------------------------------- #
_COLOR_PRESETS: dict[str, list[list[float]]] = {
    "Violet": [[0.12, 0.36, 1.00], [0.36, 0.30, 1.00], [0.56, 0.32, 0.98], [0.78, 0.40, 1.00]],
    "Indigo": [[0.05, 0.20, 0.85], [0.18, 0.24, 0.95], [0.36, 0.30, 1.00], [0.55, 0.40, 1.00]],
    "Neon": [[0.00, 0.94, 1.00], [0.48, 0.36, 1.00], [1.00, 0.17, 0.84]],
    "Spectrum": [
        [1.00, 0.00, 0.30], [1.00, 0.48, 0.00], [1.00, 0.90, 0.10],
        [0.00, 0.90, 0.46], [0.00, 0.90, 1.00], [0.16, 0.47, 1.00],
        [0.83, 0.00, 0.98],
    ],
    "Sunset": [[0.16, 0.03, 0.27], [1.00, 0.37, 0.38], [1.00, 0.60, 0.40], [1.00, 0.85, 0.42]],
    "Ocean": [[0.00, 0.12, 0.24], [0.00, 0.40, 1.00], [0.00, 0.83, 1.00], [0.36, 1.00, 0.89]],
    "Aurora": [[0.00, 0.24, 0.18], [0.00, 1.00, 0.62], [0.00, 0.90, 1.00], [0.48, 0.36, 1.00]],
    "Fire": [[0.10, 0.02, 0.00], [1.00, 0.24, 0.00], [1.00, 0.62, 0.00], [1.00, 0.89, 0.29]],
    "Mono": [[1.00, 1.00, 1.00], [0.78, 0.84, 1.00]],
}
_COLOR_PRESET_NAMES = list(_COLOR_PRESETS.keys())

# Base canvas for the ``dark`` background: a deep indigo vertical gradient.
_BG_DARK = [[0.020, 0.024, 0.055], [0.050, 0.040, 0.094]]  # top -> bottom

# Blue→violet palette used to tint every audio-reactive background layer so the
# whole image keeps a coherent indigo/violet character regardless of bar colour.
_BG_PATTERN_COLORS = [[0.04, 0.06, 0.20], [0.10, 0.10, 0.34], [0.22, 0.16, 0.52], [0.40, 0.24, 0.70]]

# Layered "mountains" silhouettes (far -> near). Each tuple:
# (height_scale, floor, smooth_mul, body_rgb, edge_rgb, body_alpha)
_MOUNTAIN_LAYERS = [
    (0.50, 0.10, 4.0, [0.09, 0.10, 0.30], [0.26, 0.32, 0.78], 0.30),  # far, blue, hazy
    (0.74, 0.05, 2.4, [0.17, 0.13, 0.42], [0.44, 0.30, 0.95], 0.42),  # mid
    (1.00, 0.00, 1.3, [0.27, 0.17, 0.55], [0.66, 0.42, 1.00], 0.55),  # near, violet, crisp
]

# Deterministic plasma waves: (freq_x, freq_y, phase, amplitude).
_PLASMA_WAVES = [
    (3.0, 2.0, 0.0, 0.50),
    (7.0, -4.0, 1.3, 0.30),
    (1.5, 6.0, 2.6, 0.20),
    (5.0, 3.0, 4.1, 0.16),
]

_STYLE_OPTIONS = ["mirror", "bottom"]
_GRADIENT_OPTIONS = ["horizontal", "vertical", "amplitude"]
_BACKGROUND_OPTIONS = ["dark", "black", "white", "transparent"]
_BG_PATTERN_OPTIONS = ["nebula", "glow", "mountains", "plasma", "none"]


class TS_AudioVisualizer(IO.ComfyNode):
    """Render an AUDIO clip into a stylized SoundCloud-style waveform image."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_AudioVisualizer",
            display_name="TS Audio Visualizer",
            category="TS/Audio",
            description=(
                "Render a standard ComfyUI audio input into a stylized waveform image: "
                "blue→violet SoundCloud-style gradient bars with a neon glow over an "
                "audio-reactive abstract background. Outputs IMAGE + MASK."
            ),
            inputs=[
                IO.Audio.Input("audio"),
                IO.Int.Input("width", default=1280, min=64, max=8192, step=8, tooltip="Output image width in pixels."),
                IO.Int.Input("height", default=320, min=32, max=8192, step=8, tooltip="Output image height in pixels."),
                IO.Combo.Input("style", options=_STYLE_OPTIONS, default="mirror",
                               tooltip="mirror: bars grow symmetrically from the centre line. bottom: bars grow up from the baseline."),
                IO.Int.Input("bar_width", default=6, min=1, max=128, step=1, display_mode=IO.NumberDisplay.slider,
                             tooltip="Width of each bar in pixels."),
                IO.Int.Input("bar_gap", default=4, min=0, max=128, step=1, display_mode=IO.NumberDisplay.slider,
                             tooltip="Gap between bars in pixels."),
                IO.Combo.Input("color_preset", options=_COLOR_PRESET_NAMES, default="Violet",
                               tooltip="Colour gradient applied to the bars and their glow."),
                IO.Combo.Input("gradient_mode", options=_GRADIENT_OPTIONS, default="horizontal",
                               tooltip="horizontal: along time. vertical: along height. amplitude: colour by bar loudness."),
                IO.Combo.Input("background", options=_BACKGROUND_OPTIONS, default="dark",
                               tooltip="Base canvas. dark: deep indigo gradient. black/white: solid. transparent: black RGB, alpha in MASK (no background pattern)."),
                IO.Combo.Input("bg_pattern", options=_BG_PATTERN_OPTIONS, default="nebula",
                               tooltip="Audio-reactive abstract background behind the bars: nebula (mountains+glow), glow (waveform aura), mountains (layered silhouettes), plasma (smoky field), none."),
                IO.Float.Input("bg_intensity", default=0.6, min=0.0, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider,
                               tooltip="Strength of the abstract background pattern."),
                IO.Float.Input("glow", default=0.40, min=0.0, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider,
                               tooltip="Neon glow / bloom intensity around the bars."),
                IO.Float.Input("sensitivity", default=0.65, min=0.1, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider,
                               tooltip="Loudness curve. Lower values lift quiet parts (amp ** sensitivity)."),
                IO.Float.Input("smoothing", default=0.15, min=0.0, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider,
                               tooltip="Blend each bar with its neighbours for a smoother envelope."),
                IO.Float.Input("height_scale", default=0.90, min=0.1, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider,
                               tooltip="Fraction of the image height the loudest bar may occupy."),
                IO.Boolean.Input("normalize", default=True, tooltip="Scale the loudest bar to full height."),
                IO.Boolean.Input("rounded_caps", default=True, tooltip="Rounded bar ends (SoundCloud look) vs. flat rectangles."),
            ],
            outputs=[
                IO.Image.Output(display_name="IMAGE"),
                IO.Mask.Output(display_name="MASK"),
            ],
            search_aliases=[
                "audio visualizer", "waveform", "soundcloud", "audio to image",
                "music visualizer", "waveform image", "audio bars",
            ],
        )

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _coerce_audio(audio: dict[str, Any]) -> tuple[torch.Tensor, int]:
        """Normalize a ComfyUI AUDIO payload to a mono CPU float tensor [T] and sample_rate."""
        sample_rate = max(1, int((audio or {}).get("sample_rate", 44100)))
        waveform = (audio or {}).get("waveform")
        if waveform is None:
            return torch.zeros(1, dtype=torch.float32), sample_rate
        tensor = torch.as_tensor(waveform).detach().cpu().float()
        # Accept [B, C, T], [C, T] or [T]; collapse to a single mono track [T].
        if tensor.ndim == 3:
            tensor = tensor[0]
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(f"{LOG_PREFIX} unsupported audio waveform shape: {tuple(tensor.shape)}")
        if tensor.numel() == 0 or tensor.shape[-1] == 0:
            return torch.zeros(1, dtype=torch.float32), sample_rate
        mono = tensor.mean(dim=0)
        return mono.contiguous(), sample_rate

    @staticmethod
    def _bar_amplitudes(
        mono: torch.Tensor,
        n_bars: int,
        normalize: bool,
        sensitivity: float,
        smoothing: float,
    ) -> torch.Tensor:
        """Reduce a mono waveform [T] to ``n_bars`` peak amplitudes in [0, 1]."""
        signal = mono.abs().view(1, 1, -1)
        # Adaptive max-pool gives one peak per bar regardless of clip length.
        amps = F.adaptive_max_pool1d(signal, n_bars).view(-1)

        if smoothing > 0.0 and n_bars >= 3:
            smoothed = F.avg_pool1d(amps.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)
            amps = torch.lerp(amps, smoothed, float(smoothing))

        if normalize:
            peak = float(amps.max())
            if peak > 1e-6:
                amps = amps / peak

        amps = amps.clamp(0.0, 1.0) ** float(sensitivity)
        return amps

    @staticmethod
    def _ramp(stops: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate a colour ramp. ``stops`` is [K, 3]; ``t`` is any shape in [0, 1] -> [..., 3]."""
        k = stops.shape[0]
        if k == 1:
            return stops[0].expand(*t.shape, 3)
        pos = t.clamp(0.0, 1.0) * (k - 1)
        i0 = pos.floor().clamp(0, k - 2).long()
        frac = (pos - i0.float()).unsqueeze(-1)
        c0 = stops[i0]
        c1 = stops[i0 + 1]
        return c0 * (1.0 - frac) + c1 * frac

    @staticmethod
    def _gaussian_blur(field: torch.Tensor, sigma: float) -> torch.Tensor:
        """Self-contained separable Gaussian blur of a single-channel [H, W] field."""
        if sigma <= 0.35:
            return field
        h, w = field.shape
        radius = int(min(max(1, round(sigma * 3.0)), max(1, h - 1), max(1, w - 1)))
        coords = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()

        inp = field.view(1, 1, h, w)
        k_h = kernel.view(1, 1, 1, -1)
        inp = F.pad(inp, (radius, radius, 0, 0), mode="replicate")
        inp = F.conv2d(inp, k_h)
        k_v = kernel.view(1, 1, -1, 1)
        inp = F.pad(inp, (0, 0, radius, radius), mode="replicate")
        inp = F.conv2d(inp, k_v)
        return inp.view(h, w)

    @staticmethod
    def _smooth1d(env: torch.Tensor, sigma: float) -> torch.Tensor:
        """Separable 1D Gaussian smoothing of a [W] envelope."""
        n = env.shape[-1]
        if sigma <= 0.5 or n < 3:
            return env
        radius = int(min(max(1, round(sigma * 3.0)), n - 1))
        coords = torch.arange(-radius, radius + 1, device=env.device, dtype=env.dtype)
        kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        inp = F.pad(env.view(1, 1, n), (radius, radius), mode="replicate")
        return F.conv1d(inp, kernel.view(1, 1, -1)).view(-1)

    @classmethod
    def _envelope(cls, amps: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
        """Resample per-bar amplitudes to a smooth per-column envelope [W] in [0, 1]."""
        env = F.interpolate(amps.view(1, 1, -1).to(device), size=width, mode="linear", align_corners=False).view(-1)
        return env.clamp(0.0, 1.0)

    @classmethod
    def _bar_mask(
        cls,
        amps: torch.Tensor,
        width: int,
        height: int,
        bar_width: int,
        bar_gap: int,
        style: str,
        height_scale: float,
        rounded_caps: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an antialiased fill mask [H, W] of bars from per-bar amplitudes."""
        pitch = bar_width + bar_gap
        n_bars = amps.shape[0]
        radius = bar_width / 2.0
        max_bar = max(1.0, height * float(height_scale))

        # Centre the bar block horizontally for a tidy, balanced layout.
        total_w = n_bars * pitch - bar_gap
        left = (width - total_w) / 2.0
        cx0 = left + (bar_width - 1) / 2.0

        xs = torch.arange(width, device=device, dtype=torch.float32)
        # Nearest bar for every column (so rounded caps may bleed into the gaps).
        bar_idx = torch.round((xs - cx0) / pitch).clamp(0, n_bars - 1).long()
        cx = cx0 + bar_idx.float() * pitch
        dx = xs - cx                                   # [W] horizontal distance to bar centre
        bar_len = amps.to(device)[bar_idx] * max_bar   # [W] full bar length in px

        ys = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)  # [H, 1]

        if style == "bottom":
            seg_top = (height - bar_len).unsqueeze(0)              # [1, W]
            if rounded_caps:
                # clamp(ys, seg_top, height) with a per-column tensor bound;
                # torch.clamp rejects mixed tensor/float bounds, so use min/max.
                y_clamped = torch.maximum(ys, seg_top)            # lower bound -> [H, W]
                y_clamped = torch.minimum(y_clamped, torch.full_like(y_clamped, float(height)))
                dy = ys - y_clamped                               # [H, W]
                dist = torch.sqrt(dx.unsqueeze(0) ** 2 + dy ** 2)
                mask = (radius - dist + 0.5).clamp(0.0, 1.0)
            else:
                ax = (radius + 0.5 - dx.abs()).clamp(0.0, 1.0).unsqueeze(0)   # [1, W]
                ay = (ys - seg_top + 0.5).clamp(0.0, 1.0)                     # [H, W]
                mask = ax * ay
        else:  # mirror
            cy = height / 2.0
            half = bar_len / 2.0
            seg_half = (half - radius).clamp(min=0.0).unsqueeze(0)           # [1, W]
            if rounded_caps:
                dy = ((ys - cy).abs() - seg_half).clamp(min=0.0)             # [H, W]
                dist = torch.sqrt(dx.unsqueeze(0) ** 2 + dy ** 2)
                mask = (radius - dist + 0.5).clamp(0.0, 1.0)
            else:
                ax = (radius + 0.5 - dx.abs()).clamp(0.0, 1.0).unsqueeze(0)  # [1, W]
                ay = (half.unsqueeze(0) + 0.5 - (ys - cy).abs()).clamp(0.0, 1.0)  # [H, W]
                mask = ax * ay

        return mask.clamp(0.0, 1.0)

    @classmethod
    def _gradient_rgb(
        cls,
        amps: torch.Tensor,
        width: int,
        height: int,
        bar_width: int,
        bar_gap: int,
        color_preset: str,
        gradient_mode: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the [H, W, 3] colour field for the chosen preset and direction."""
        stops = torch.tensor(_COLOR_PRESETS[color_preset], device=device, dtype=torch.float32)

        if gradient_mode == "vertical":
            t = torch.linspace(0.0, 1.0, height, device=device).view(height, 1).expand(height, width)
        elif gradient_mode == "amplitude":
            pitch = bar_width + bar_gap
            n_bars = amps.shape[0]
            total_w = n_bars * pitch - bar_gap
            cx0 = (width - total_w) / 2.0 + (bar_width - 1) / 2.0
            xs = torch.arange(width, device=device, dtype=torch.float32)
            bar_idx = torch.round((xs - cx0) / pitch).clamp(0, n_bars - 1).long()
            t = amps.to(device)[bar_idx].view(1, width).expand(height, width)
        else:  # horizontal
            t = torch.linspace(0.0, 1.0, width, device=device).view(1, width).expand(height, width)

        return cls._ramp(stops, t)

    # ----------------------------------------------------- audio-reactive bg
    @classmethod
    def _bg_glow(
        cls, env: torch.Tensor, width: int, height: int, style: str, height_scale: float, device: torch.device
    ) -> torch.Tensor:
        """Soft blurred aura in the shape of the (smoothed) waveform -> [H, W, 3]."""
        env_s = cls._smooth1d(env, width * 0.012 + 6.0)
        max_bar = max(1.0, height * float(height_scale)) * 1.25
        ys = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        if style == "bottom":
            top = (height - env_s * max_bar).unsqueeze(0)
            fill = (ys >= top).float()
        else:
            cy = height / 2.0
            half = (env_s * max_bar / 2.0).unsqueeze(0)
            fill = ((ys - cy).abs() <= half).float()
        glow = cls._gaussian_blur(fill, sigma=height * 0.05 + 8.0)
        stops = torch.tensor(_BG_PATTERN_COLORS, device=device, dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, height, device=device).view(height, 1).expand(height, width)
        col = cls._ramp(stops, t)
        return col * glow.unsqueeze(-1) * 1.1

    @classmethod
    def _bg_mountains(
        cls, env: torch.Tensor, width: int, height: int, height_scale: float, device: torch.device
    ) -> torch.Tensor:
        """Layered parallax silhouettes whose ridge lines follow the loudness -> [H, W, 3]."""
        max_bar = max(1.0, height * float(height_scale))
        ys = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        acc = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
        for scale, floor, smooth_mul, body_rgb, edge_rgb, body_alpha in _MOUNTAIN_LAYERS:
            h_line = (cls._smooth1d(env, width * 0.01 * smooth_mul + 4.0) * scale + floor) * max_bar
            top = (height - h_line).unsqueeze(0)                       # [1, W] ridge line
            sil = (ys - top + 0.5).clamp(0.0, 1.0)                     # filled below the ridge
            edge = torch.exp(-((ys - top) ** 2) / (2.0 * 1.6 ** 2))    # thin glowing crest
            body = torch.tensor(body_rgb, device=device, dtype=torch.float32)
            crest = torch.tensor(edge_rgb, device=device, dtype=torch.float32)
            a = (sil * body_alpha).unsqueeze(-1)
            acc = acc * (1.0 - a) + body * a                          # near layers occlude far ones
            acc = acc + crest * edge.unsqueeze(-1)
        return acc

    @classmethod
    def _bg_plasma(cls, env: torch.Tensor, width: int, height: int, device: torch.device) -> torch.Tensor:
        """Deterministic smoky field, brightened where the clip is loud -> [H, W, 3]."""
        xs = torch.linspace(0.0, 1.0, width, device=device).view(1, width)
        ys = torch.linspace(0.0, 1.0, height, device=device).view(height, 1)
        field = torch.zeros(height, width, device=device, dtype=torch.float32)
        for fx, fy, phase, amp in _PLASMA_WAVES:
            field = field + amp * torch.sin(2.0 * torch.pi * (xs * fx + ys * fy) + phase)
        field = (field - field.min()) / (field.max() - field.min() + 1e-6)
        env_s = cls._smooth1d(env, width * 0.015 + 6.0).view(1, width)
        field = field * (0.3 + 0.7 * env_s)                          # louder columns glow brighter
        stops = torch.tensor(_BG_PATTERN_COLORS, device=device, dtype=torch.float32)
        col = cls._ramp(stops, field)
        return col * field.unsqueeze(-1) * 0.9

    @classmethod
    def _bg_pattern_rgb(
        cls,
        pattern: str,
        amps: torch.Tensor,
        width: int,
        height: int,
        style: str,
        height_scale: float,
        intensity: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Dispatch and accumulate the selected abstract background layers -> [H, W, 3]."""
        if pattern == "none" or intensity <= 0.0:
            return torch.zeros(height, width, 3, device=device, dtype=torch.float32)
        env = cls._envelope(amps, width, device)
        out = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
        if pattern in ("mountains", "nebula"):
            out = out + cls._bg_mountains(env, width, height, height_scale, device)
        if pattern in ("glow", "nebula"):
            out = out + cls._bg_glow(env, width, height, style, height_scale, device)
        if pattern == "plasma":
            out = out + cls._bg_plasma(env, width, height, device)
        return (out * float(intensity)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ render
    @classmethod
    def _render(
        cls,
        mono: torch.Tensor,
        width: int,
        height: int,
        style: str,
        bar_width: int,
        bar_gap: int,
        color_preset: str,
        gradient_mode: str,
        background: str,
        bg_pattern: str,
        bg_intensity: float,
        glow: float,
        sensitivity: float,
        smoothing: float,
        height_scale: float,
        normalize: bool,
        rounded_caps: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render the visualization on ``device``; returns (image [1,H,W,3], mask [1,H,W]) on CPU."""
        pitch = bar_width + bar_gap
        n_bars = max(1, (width + bar_gap) // pitch)

        amps = cls._bar_amplitudes(mono.to(device), n_bars, normalize, sensitivity, smoothing)

        fill = cls._bar_mask(amps, width, height, bar_width, bar_gap, style, height_scale, rounded_caps, device)
        viz = cls._gradient_rgb(amps, width, height, bar_width, bar_gap, color_preset, gradient_mode, device)
        bg = cls._background_rgb(background, width, height, device)

        # Audio-reactive abstract background lives behind the bars (skipped when
        # the user wants a transparent output — there is nothing to sit behind).
        if background != "transparent":
            bg = (bg + cls._bg_pattern_rgb(bg_pattern, amps, width, height, style, height_scale, bg_intensity, device)).clamp(0.0, 1.0)

        # Neon bloom: blur the bar fill and tint it with the same gradient.
        glow_sigma = pitch * (0.45 + 1.6 * float(glow))
        bloom = cls._gaussian_blur(fill, glow_sigma) * float(glow) * 1.4 if glow > 0.0 else torch.zeros_like(fill)

        fill3 = fill.unsqueeze(-1)
        bloom3 = bloom.unsqueeze(-1)
        if background == "transparent":
            rgb = viz * fill3 + viz * bloom3
        else:
            rgb = bg * (1.0 - fill3) + viz * fill3 + viz * bloom3
        rgb = rgb.clamp(0.0, 1.0)

        # MASK = bars + their glow only (the background pattern stays out of it).
        alpha = (fill + bloom).clamp(0.0, 1.0)

        image = rgb.unsqueeze(0).to("cpu", dtype=torch.float32).contiguous()
        mask = alpha.unsqueeze(0).to("cpu", dtype=torch.float32).contiguous()
        return image, mask

    @classmethod
    def _background_rgb(cls, background: str, width: int, height: int, device: torch.device) -> torch.Tensor:
        """Build the [H, W, 3] base canvas (before the abstract pattern)."""
        if background == "white":
            return torch.ones(height, width, 3, device=device, dtype=torch.float32)
        if background in ("black", "transparent"):
            return torch.zeros(height, width, 3, device=device, dtype=torch.float32)
        # dark: subtle deep-indigo vertical gradient for a premium look.
        stops = torch.tensor(_BG_DARK, device=device, dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, height, device=device).view(height, 1).expand(height, width)
        return cls._ramp(stops, t)

    # ----------------------------------------------------------------- caching
    @classmethod
    def fingerprint_inputs(
        cls,
        audio: dict[str, Any],
        width: int,
        height: int,
        style: str,
        bar_width: int,
        bar_gap: int,
        color_preset: str,
        gradient_mode: str,
        background: str,
        bg_pattern: str,
        bg_intensity: float,
        glow: float,
        sensitivity: float,
        smoothing: float,
        height_scale: float,
        normalize: bool,
        rounded_caps: bool,
    ) -> str:
        mono, sample_rate = cls._coerce_audio(audio)
        hasher = hashlib.sha256()
        hasher.update(f"{sample_rate}|{int(mono.shape[-1])}".encode("utf-8"))
        # Torch-only signature: coarse 64-bin envelope + global stats (no numpy,
        # no hashing of the full multi-megabyte clip).
        env = F.adaptive_avg_pool1d(mono.abs().view(1, 1, -1), min(64, max(1, mono.shape[-1]))).view(-1)
        sig = [round(float(v), 5) for v in env.tolist()]
        stats = [float(mono.mean()), float(mono.std()), float(mono.abs().max())]
        hasher.update(repr(sig).encode("utf-8"))
        hasher.update(repr([round(s, 6) for s in stats]).encode("utf-8"))
        params = (width, height, style, bar_width, bar_gap, color_preset, gradient_mode,
                  background, bg_pattern, bg_intensity, glow, sensitivity, smoothing,
                  height_scale, normalize, rounded_caps)
        hasher.update(repr(params).encode("utf-8"))
        return hasher.hexdigest()

    # ----------------------------------------------------------------- execute
    @classmethod
    def execute(
        cls,
        audio: dict[str, Any],
        width: int,
        height: int,
        style: str = "mirror",
        bar_width: int = 6,
        bar_gap: int = 4,
        color_preset: str = "Violet",
        gradient_mode: str = "horizontal",
        background: str = "dark",
        bg_pattern: str = "nebula",
        bg_intensity: float = 0.6,
        glow: float = 0.40,
        sensitivity: float = 0.65,
        smoothing: float = 0.15,
        height_scale: float = 0.90,
        normalize: bool = True,
        rounded_caps: bool = True,
    ) -> IO.NodeOutput:
        width = int(width)
        height = int(height)
        bar_width = max(1, int(bar_width))
        bar_gap = max(0, int(bar_gap))

        mono, _ = cls._coerce_audio(audio)

        device = mm.get_torch_device()
        with torch.no_grad():
            try:
                image, mask = cls._render(
                    mono, width, height, style, bar_width, bar_gap, color_preset,
                    gradient_mode, background, bg_pattern, bg_intensity, glow,
                    sensitivity, smoothing, height_scale, normalize, rounded_caps, device,
                )
            except RuntimeError as exc:
                # GPU OOM (a RuntimeError subclass) / unsupported op: fall back
                # to CPU so the render still succeeds.
                if device.type != "cpu":
                    logger.warning("%s GPU render failed (%s); falling back to CPU.", LOG_PREFIX, exc)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    image, mask = cls._render(
                        mono, width, height, style, bar_width, bar_gap, color_preset,
                        gradient_mode, background, bg_pattern, bg_intensity, glow,
                        sensitivity, smoothing, height_scale, normalize, rounded_caps,
                        torch.device("cpu"),
                    )
                else:
                    raise

        return IO.NodeOutput(image, mask)


NODE_CLASS_MAPPINGS = {"TS_AudioVisualizer": TS_AudioVisualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_AudioVisualizer": "TS Audio Visualizer"}
