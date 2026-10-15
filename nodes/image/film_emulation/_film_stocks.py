"""Film stock models for TS Film Emulation — pure torch, no ComfyUI imports.

Until 12.13 a "film preset" was a gamma, a fade and a warm/cool nudge; Ilford
HP5, a black-and-white stock, came out in colour. This module models what makes
stocks look different from each other, in the order light meets them:

1. **Exposure** in linear light.
2. **Characteristic curve** per channel, in stops from middle grey. Each dye
   layer has its own slope and its own offset, so the channels cross over:
   that is where "warm highlights, cool shadows" comes from on real film. The
   toe lifts black to the print's minimum density and the shoulder rolls the
   highlights off softly instead of clipping. Middle grey stays put.
3. **Colour** in six hue bands (red, yellow, green, cyan, blue, magenta): each
   band has its own hue rotation and saturation — Fuji greens lean cyan, Kodak
   greens lean olive, Velvia pushes everything. Neutral greys are untouched by
   construction, and chroma is compressed back into gamut instead of clipped.
4. **Split toning** — separate tints for shadows and highlights.
5. **Halation** — light bouncing off the film base back into the red layer: a
   wide red-orange glow around bright areas, the CineStill signature.
6. **Black and white** stocks take their own spectral weights before the curve
   and may carry a print tone.

Everything is per pixel except halation, which needs a neighbourhood; the blur
runs at low resolution, so the cost stays small on 4K and on clips.

Input and output are display-referred sRGB IMAGE tensors [B, H, W, 3] in 0..1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

#: Rec.709 luma weights (linear light).
_LUMA = (0.2126, 0.7152, 0.0722)
#: Middle grey in linear light: the pivot of every curve.
_MID_GREY = 0.18
#: Input white in stops above middle grey.
_WHITE_STOPS = math.log2(1.0 / _MID_GREY)
#: Hue band centres, degrees: red, yellow, green, cyan, blue, magenta.
_BANDS = ("red", "yellow", "green", "cyan", "blue", "magenta")
_BAND_CENTRES = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)


@dataclass(frozen=True)
class FilmStock:
    """One stock. Defaults are neutral: a stock built from them changes nothing
    but the toe and shoulder."""

    name: str
    family: str  # color_negative | slide | cinema | instant | bw
    note_en: str
    note_ru: str
    exposure: float = 0.0
    #: Curve slope per channel (R, G, B) around middle grey. 1 = unchanged.
    contrast: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Per-channel offset in stops: colour balance that moves with exposure.
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Print black (linear). Higher = milkier blacks.
    black: float = 0.003
    toe: float = 1.6
    #: Stops above input white where the shoulder sits; smaller = earlier roll-off.
    headroom: float = 0.7
    shoulder: float = 1.6
    #: Print white (linear) that input white lands on. The soft shoulder alone
    #: parks white BELOW its asymptote — Portra 400 read 0.87 in sRGB, Polaroid
    #: 0.80: grey, not white. 0.9 linear is ~0.95 sRGB.
    white: float = 0.9
    saturation: float = 1.0
    sat_shadows: float = 1.0
    sat_highlights: float = 1.0
    #: Degrees of hue rotation per band (positive = towards the next band).
    hue: dict = field(default_factory=dict)
    #: Saturation multiplier per band.
    band_sat: dict = field(default_factory=dict)
    #: 0..1 — how much of the hue/saturation change skin tones are spared.
    #: Skin sits at ~20° in the red-yellow zone, so a stock that pushes reds
    #: would otherwise paint faces orange (Velvia) or pink (Kodachrome).
    skin_protect: float = 0.65
    #: Linear gains added in the shadows / highlights (R, G, B), e.g. +0.04 blue.
    shadow_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    halation: float = 0.0
    halation_color: tuple[float, float, float] = (1.0, 0.28, 0.08)
    #: Black-and-white: spectral weights (R, G, B). None = colour stock.
    mono: tuple[float, float, float] | None = None
    #: Print tone for black-and-white, a gain per channel (1 = neutral).
    mono_tone: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _stock(*args, **kwargs) -> FilmStock:
    return FilmStock(*args, **kwargs)


# ── the catalogue ─────────────────────────────────────────────────────────── #
#
# ⚠️ Names are workflow data: a saved graph stores the preset by NAME. The
# first seven are the stocks the node had before 12.13 — never rename or drop
# them. New stocks go anywhere in the gallery, but in the combo they are
# appended after the old ones.
#
# The numbers were tuned by eye on the bundled reference portrait (skin, blonde
# hair, white linen with red embroidery, blue sky, deep forest greens) against
# what these stocks are known for, and checked on a grey ramp for neutrality.

STOCKS: tuple[FilmStock, ...] = (
    # --- the original seven (names frozen) ----------------------------------
    _stock(
        "Kodak Vision3 250D", "cinema",
        "Daylight cinema negative: wide latitude, gentle contrast, natural warm skin.",
        "Дневная кинонегативная: широкая широта, мягкий контраст, естественная тёплая кожа.",
        contrast=(0.92, 0.9, 0.88), offset=(0.04, 0.0, -0.05), black=0.006,
        headroom=0.9, shoulder=1.2, saturation=0.92,
        hue={"green": -6, "cyan": 4}, band_sat={"green": 0.88, "blue": 0.95},
        shadow_tint=(0.0, 0.01, 0.02), highlight_tint=(0.02, 0.01, -0.01), halation=0.2,
    ),
    _stock(
        "Kodak Portra 400", "color_negative",
        "The portrait standard: soft contrast, warm glowing skin, olive greens, airy highlights.",
        "Портретный стандарт: мягкий контраст, тёплая сияющая кожа, оливковая зелень, воздушные света.",
        exposure=0.15, contrast=(0.86, 0.85, 0.82), offset=(0.07, 0.02, -0.06), black=0.005,
        headroom=0.6, shoulder=1.3, saturation=0.9, sat_shadows=0.85, sat_highlights=0.9,
        hue={"red": 4, "yellow": -4, "green": -14, "blue": -6},
        band_sat={"red": 1.02, "yellow": 0.95, "green": 0.75, "cyan": 0.85, "blue": 0.9},
        shadow_tint=(0.0, 0.01, 0.015), highlight_tint=(0.025, 0.012, -0.012),
    ),
    _stock(
        "Fuji Eterna 250T", "cinema",
        "Fuji's cinema stock: muted, low contrast, cool teal shadows — the quiet film look.",
        "Киноплёнка Fuji: приглушённая, низкий контраст, холодные бирюзовые тени — тихий киношный вид.",
        contrast=(0.84, 0.86, 0.9), offset=(-0.03, 0.01, 0.04), black=0.006,
        headroom=0.8, shoulder=1.2, saturation=0.72, sat_shadows=0.7,
        hue={"green": 10, "yellow": -6, "red": -3},
        band_sat={"green": 0.8, "red": 0.85, "yellow": 0.8},
        shadow_tint=(-0.01, 0.015, 0.03), highlight_tint=(0.01, 0.005, -0.005),
    ),
    _stock(
        "Agfa Vista 200", "color_negative",
        "Punchy consumer film: rich reds, magenta-tinted warmth, crisp contrast.",
        "Яркая любительская плёнка: сочные красные, тепло с пурпурным оттенком, чёткий контраст.",
        contrast=(1.08, 1.04, 1.0), offset=(0.06, -0.02, 0.0), black=0.004,
        headroom=0.6, shoulder=1.6, saturation=1.12,
        hue={"red": -4, "green": -8, "blue": 6}, band_sat={"red": 1.2, "magenta": 1.1, "green": 0.95},
        shadow_tint=(0.01, -0.005, 0.015), highlight_tint=(0.02, 0.0, 0.0),
    ),
    _stock(
        "Ilford HP5", "bw",
        "Classic black and white, ISO 400: moderate contrast, open shadows, panchromatic.",
        "Классическая чёрно-белая ISO 400: умеренный контраст, открытые тени, панхроматическая.",
        contrast=(1.02, 1.02, 1.02), black=0.006, headroom=0.7, shoulder=1.4,
        mono=(0.27, 0.6, 0.13),
    ),
    _stock(
        "Kodak Gold 200", "color_negative",
        "Sunny holiday film: golden-yellow warmth, saturated yellows and reds, friendly contrast.",
        "Солнечная отпускная плёнка: золотисто-жёлтое тепло, насыщенные жёлтые и красные, дружелюбный контраст.",
        exposure=0.05, contrast=(0.98, 0.97, 0.92), offset=(0.08, 0.05, -0.1), black=0.006,
        headroom=0.6, shoulder=1.4, saturation=1.06,
        hue={"green": -12, "yellow": -4, "red": 3},
        band_sat={"yellow": 1.2, "red": 1.1, "green": 0.85, "blue": 0.9},
        shadow_tint=(0.01, 0.005, -0.005), highlight_tint=(0.03, 0.02, -0.02),
    ),
    _stock(
        "Fuji Superia 400", "color_negative",
        "Everyday Fuji: cooler, green-leaning palette with magenta shadows and snappy contrast.",
        "Повседневная Fuji: прохладная палитра с уклоном в зелень, пурпурные тени, бодрый контраст.",
        contrast=(1.02, 1.06, 1.04), offset=(-0.03, 0.03, 0.02), black=0.005,
        headroom=0.6, shoulder=1.5, saturation=1.04,
        hue={"green": 12, "yellow": 6, "blue": -6}, band_sat={"green": 1.1, "cyan": 1.05, "red": 0.95},
        shadow_tint=(0.012, -0.005, 0.02), highlight_tint=(-0.005, 0.01, 0.0),
    ),
    # --- colour negative -----------------------------------------------------
    _stock(
        "Kodak Portra 160", "color_negative",
        "The softest Portra: pastel, very low contrast, true-to-life skin and gentle colour.",
        "Самая мягкая Portra: пастель, очень низкий контраст, правдивая кожа и деликатный цвет.",
        exposure=0.2, contrast=(0.8, 0.8, 0.78), offset=(0.05, 0.02, -0.04), black=0.006,
        headroom=0.5, shoulder=1.2, saturation=0.84, sat_shadows=0.8,
        hue={"green": -12, "red": 3}, band_sat={"green": 0.72, "cyan": 0.82, "blue": 0.88},
        shadow_tint=(0.0, 0.008, 0.012), highlight_tint=(0.018, 0.01, -0.008),
    ),
    _stock(
        "Kodak Portra 800", "color_negative",
        "Night-capable Portra: warmer and more saturated than 400, a touch more contrast.",
        "Ночная Portra: теплее и насыщеннее 400-й, чуть больше контраста.",
        exposure=0.1, contrast=(0.95, 0.92, 0.88), offset=(0.1, 0.03, -0.08), black=0.007,
        headroom=0.6, shoulder=1.3, saturation=1.0,
        hue={"red": 4, "green": -12, "blue": -8},
        band_sat={"red": 1.08, "yellow": 1.04, "green": 0.85, "blue": 0.92},
        shadow_tint=(0.005, 0.01, 0.01), highlight_tint=(0.03, 0.015, -0.015), halation=0.06,
    ),
    _stock(
        "Kodak Ektar 100", "color_negative",
        "The most vivid negative: deep saturated reds and blues, fine and contrasty, cool-neutral.",
        "Самый яркий негатив: глубокие насыщенные красные и синие, контрастный, холодно-нейтральный.",
        contrast=(1.12, 1.1, 1.14), offset=(0.0, -0.01, 0.03), black=0.003,
        headroom=0.6, shoulder=1.7, white=0.97, saturation=1.25,
        hue={"red": -3, "blue": -4, "green": 4}, band_sat={"red": 1.2, "blue": 1.25, "cyan": 1.15},
        shadow_tint=(0.0, 0.0, 0.015), highlight_tint=(0.005, 0.0, 0.0),
    ),
    _stock(
        "Kodak Ultramax 400", "color_negative",
        "Kodak's everyday 400: warm, punchy, reddish skin, contrast that pops in sunlight.",
        "Повседневная Kodak 400: тёплая, сочная, красноватая кожа, контраст, играющий на солнце.",
        contrast=(1.06, 1.02, 0.96), offset=(0.08, 0.02, -0.07), black=0.005,
        headroom=0.6, shoulder=1.5, saturation=1.12,
        hue={"red": -4, "green": -10, "yellow": -3}, band_sat={"red": 1.15, "yellow": 1.12, "green": 0.92},
        shadow_tint=(0.01, 0.0, 0.0), highlight_tint=(0.025, 0.012, -0.015),
    ),
    _stock(
        "Fuji Pro 400H", "color_negative",
        "The wedding pastel: airy, minty-cool greens, soft skin, lifted shadows.",
        "Свадебная пастель: воздушная, мятно-прохладная зелень, нежная кожа, приподнятые тени.",
        exposure=0.25, contrast=(0.82, 0.86, 0.86), offset=(-0.03, 0.03, 0.03), black=0.007,
        headroom=0.5, shoulder=1.1, saturation=0.86, sat_shadows=0.78,
        hue={"green": 16, "yellow": 8, "red": -2, "blue": -8},
        band_sat={"green": 0.85, "cyan": 0.95, "red": 0.9, "yellow": 0.82},
        shadow_tint=(-0.005, 0.012, 0.02), highlight_tint=(0.0, 0.01, 0.005),
    ),
    # --- slide / reversal -----------------------------------------------------
    _stock(
        "Fuji Velvia 50", "slide",
        "The landscape slide: extreme saturation, deep contrast, electric greens and blues.",
        "Пейзажный слайд: предельная насыщенность, глубокий контраст, электрическая зелень и синева.",
        exposure=-0.1, contrast=(1.22, 1.2, 1.24), offset=(0.02, 0.0, 0.02), black=0.002,
        toe=1.9, headroom=0.4, shoulder=1.9, white=1.0, saturation=1.32, sat_shadows=1.15,
        hue={"green": 6, "blue": -6, "red": -3}, band_sat={"green": 1.25, "blue": 1.3, "red": 1.15, "magenta": 1.2},
        shadow_tint=(0.01, -0.005, 0.02), skin_protect=0.85,
    ),
    _stock(
        "Fuji Provia 100F", "slide",
        "Neutral professional slide: clean, accurate colour with crisp, generous contrast.",
        "Нейтральный профессиональный слайд: чистый точный цвет, чёткий щедрый контраст.",
        contrast=(1.14, 1.14, 1.14), black=0.002, toe=1.9, headroom=0.5, shoulder=1.8, white=1.0,
        saturation=1.12, band_sat={"blue": 1.08, "green": 1.05},
    ),
    _stock(
        "Kodachrome 64", "slide",
        "The legend: deep crimson reds, rich contrast, warm highlights and dense shadows.",
        "Легенда: глубокие малиновые красные, плотный контраст, тёплые света и густые тени.",
        exposure=-0.12, contrast=(1.18, 1.12, 1.08), offset=(0.05, 0.0, -0.03), black=0.0025,
        toe=1.8, headroom=0.5, shoulder=1.6, white=0.95, saturation=1.16, sat_shadows=1.05,
        hue={"red": -3, "yellow": -6, "green": -10, "blue": -10},
        band_sat={"red": 1.3, "yellow": 1.1, "green": 0.9, "blue": 1.15, "cyan": 0.95},
        shadow_tint=(0.0, 0.0, 0.01), highlight_tint=(0.025, 0.015, -0.01),
    ),
    # --- cinema ----------------------------------------------------------------
    _stock(
        "Kodak Vision3 500T", "cinema",
        "Tungsten cinema negative: cool cast in daylight, soft contrast, faint halation.",
        "Кинонегатив под лампы накаливания: холодный оттенок днём, мягкий контраст, лёгкая халация.",
        contrast=(0.9, 0.9, 0.92), offset=(-0.08, -0.01, 0.1), black=0.007,
        headroom=0.9, shoulder=1.2, saturation=0.9,
        hue={"green": -6}, band_sat={"green": 0.85, "red": 0.95},
        shadow_tint=(-0.005, 0.01, 0.025), highlight_tint=(0.0, 0.005, 0.01), halation=0.35,
    ),
    _stock(
        "Kodak 2383 Print", "cinema",
        "The cinema print look: teal shadows, warm skin, dense blacks — the blockbuster palette.",
        "Вид кинокопии: бирюзовые тени, тёплая кожа, плотный чёрный — палитра блокбастера.",
        exposure=-0.05, contrast=(1.12, 1.1, 1.06), offset=(0.03, 0.0, -0.02), black=0.002,
        toe=1.9, headroom=0.5, shoulder=1.5, saturation=1.0, sat_shadows=0.9,
        hue={"green": 12, "yellow": -8, "red": 4, "blue": -8},
        band_sat={"green": 0.8, "red": 1.08, "yellow": 1.05, "cyan": 1.1},
        shadow_tint=(-0.02, 0.012, 0.03), highlight_tint=(0.03, 0.012, -0.02),
    ),
    _stock(
        "CineStill 800T", "cinema",
        "Night city film: tungsten-cool cast, red halation glowing around every light.",
        "Плёнка ночного города: холодный оттенок под лампы, красная халация вокруг каждого огня.",
        contrast=(0.96, 0.94, 0.96), offset=(-0.09, -0.01, 0.12), black=0.006,
        headroom=0.7, shoulder=1.3, saturation=0.96,
        hue={"green": 8, "red": 4}, band_sat={"green": 0.85, "cyan": 1.05},
        shadow_tint=(-0.01, 0.012, 0.03), highlight_tint=(0.02, 0.0, 0.0),
        halation=1.2, halation_color=(1.0, 0.16, 0.04),
    ),
    _stock(
        "CineStill 50D", "cinema",
        "Clean daylight cinema film: fine, neutral-warm colour with a gentle halation.",
        "Чистая дневная киноплёнка: тонкий нейтрально-тёплый цвет и мягкая халация.",
        contrast=(1.0, 0.98, 0.96), offset=(0.03, 0.0, -0.03), black=0.004,
        headroom=0.7, shoulder=1.4, saturation=1.04,
        hue={"green": -6}, band_sat={"blue": 1.05},
        highlight_tint=(0.012, 0.006, -0.006), halation=0.6, halation_color=(1.0, 0.2, 0.05),
    ),
    # --- instant ---------------------------------------------------------------
    _stock(
        "Polaroid 600", "instant",
        "Instant film: faded blacks, creamy warm highlights, cyan-green shadows, soft colour.",
        "Мгновенная плёнка: выцветший чёрный, кремовые тёплые света, бирюзово-зелёные тени, мягкий цвет.",
        exposure=0.1, contrast=(0.8, 0.84, 0.86), offset=(0.0, 0.03, 0.0), black=0.03,
        toe=1.2, headroom=0.3, shoulder=1.0, white=0.75, saturation=0.8, sat_shadows=0.7,
        hue={"green": 10, "blue": -10, "red": 4}, band_sat={"green": 0.85, "blue": 0.8},
        shadow_tint=(-0.02, 0.02, 0.025), highlight_tint=(0.03, 0.025, -0.01),
    ),
    # --- black and white -------------------------------------------------------
    _stock(
        "Kodak Tri-X 400", "bw",
        "The photojournalist's film: punchy contrast, deep blacks, bright skin.",
        "Плёнка фоторепортёров: бодрый контраст, глубокий чёрный, светлая кожа.",
        contrast=(1.18, 1.18, 1.18), black=0.003, toe=1.8, headroom=0.5, shoulder=1.6, white=1.0,
        mono=(0.32, 0.56, 0.12),
    ),
    _stock(
        "Kodak T-Max 100", "bw",
        "Fine-grain modern B&W: smooth tones, clean highlights, neutral contrast.",
        "Мелкозернистая современная ч/б: плавные тона, чистые света, нейтральный контраст.",
        contrast=(1.08, 1.08, 1.08), black=0.003, toe=1.7, headroom=0.6, shoulder=1.6,
        mono=(0.24, 0.64, 0.12),
    ),
    _stock(
        "Ilford Delta 3200", "bw",
        "Low-light B&W: soft lifted shadows, muted contrast, moody night tone.",
        "Ч/б для низкого света: мягкие приподнятые тени, приглушённый контраст, ночное настроение.",
        exposure=0.15, contrast=(0.88, 0.88, 0.88), black=0.008, toe=1.3, headroom=0.5, shoulder=1.2,
        white=0.85,
        mono=(0.3, 0.58, 0.12),
    ),
    _stock(
        "Sepia Fiber Print", "bw",
        "Toned fibre-paper print: warm sepia, gentle contrast, an old-photograph feel.",
        "Тонированный отпечаток на баритовой бумаге: тёплая сепия, мягкий контраст, старое фото.",
        contrast=(0.98, 0.98, 0.98), black=0.012, toe=1.4, headroom=0.5, shoulder=1.3, white=0.8,
        mono=(0.3, 0.59, 0.11), mono_tone=(1.08, 0.97, 0.8),
    ),
)

STOCK_BY_NAME = {stock.name: stock for stock in STOCKS}

#: Pre-12.13 combo order. Never reorder: the first entry after "External LUT"
#: was never a default, but keep the old seven first all the same.
LEGACY_PRESETS = (
    "Kodak Vision3 250D", "Kodak Portra 400", "Fuji Eterna 250T", "Agfa Vista 200",
    "Ilford HP5", "Kodak Gold 200", "Fuji Superia 400",
)
#: Combo options: "External LUT" first (it is the default), then the old seven,
#: then the stocks added in 12.13.
PRESET_OPTIONS = ("External LUT",) + LEGACY_PRESETS + tuple(
    s.name for s in STOCKS if s.name not in LEGACY_PRESETS
)

FAMILY_LABELS = {
    "color_negative": ("Colour negative", "Цветной негатив"),
    "slide": ("Slide film", "Слайдовая плёнка"),
    "cinema": ("Cinema", "Кино"),
    "instant": ("Instant", "Моментальная"),
    "bw": ("Black and white", "Чёрно-белая"),
}


# ── colour maths ──────────────────────────────────────────────────────────── #

def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(image <= 0.0031308, image * 12.92, 1.055 * image ** (1.0 / 2.4) - 0.055)


def _soft_floor(x: torch.Tensor, floor: float, sharpness: float) -> torch.Tensor:
    """Approaches ``floor`` from above smoothly; equals x well above it."""
    return floor + F.softplus(x - floor, beta=sharpness)


def _soft_ceiling(x: torch.Tensor, ceiling: float, sharpness: float) -> torch.Tensor:
    return ceiling - F.softplus(ceiling - x, beta=sharpness)


def _curve_stops(stops_in, slope, offset, stock: FilmStock):
    black_stops = math.log2(max(stock.black, 1e-5) / _MID_GREY)
    stops = stops_in * slope + offset
    stops = _soft_floor(stops, black_stops, stock.toe)
    return _soft_ceiling(stops, _WHITE_STOPS + stock.headroom, stock.shoulder)


def _white_lift(stock: FilmStock) -> float:
    """Stops to add at input white so neutral white lands on ``stock.white``.

    Measured on the neutral (mean) channel, then applied to all three alike:
    the per-channel cast of the highlights — Portra's warmth — is the look and
    stays.
    """
    slope = sum(stock.contrast) / 3.0
    offset = sum(stock.offset) / 3.0
    reached = _curve_stops(torch.tensor(_WHITE_STOPS, dtype=torch.float64), slope, offset, stock)
    return math.log2(max(stock.white, 1e-5) / _MID_GREY) - float(reached)


def characteristic_curve(lin: torch.Tensor, stock: FilmStock) -> torch.Tensor:
    """Per-channel film curve in log space; returns linear light.

    The lift to print white grows with the square of the input's distance
    above middle grey, so mid grey and everything below it are untouched.
    """
    stops_in = torch.log2(lin.clamp_min(1e-6) / _MID_GREY)
    slope = torch.tensor(stock.contrast, dtype=lin.dtype, device=lin.device)
    offset = torch.tensor(stock.offset, dtype=lin.dtype, device=lin.device)
    stops = _curve_stops(stops_in, slope, offset, stock)
    lift = _white_lift(stock)
    if abs(lift) > 1e-6:
        reach = (stops_in / _WHITE_STOPS).clamp(0.0, 1.0)
        stops = stops + lift * reach * reach
    return (_MID_GREY * torch.exp2(stops)).clamp(0.0, 1.0)


def _band_values(table: dict, default: float) -> list[float]:
    return [float(table.get(band, default)) for band in _BANDS]


def hue_saturation(lin: torch.Tensor, stock: FilmStock) -> torch.Tensor:
    """Hue rotation and saturation per hue band, plus saturation by brightness.

    Works on the chroma of each pixel around its channel mean, so neutral
    greys (no chroma) pass through exactly. Six bands 60° apart with
    triangular weights form a partition of unity — no band boundary seams.
    """
    hue_shift = _band_values(stock.hue, 0.0)
    band_sat = _band_values(stock.band_sat, 1.0)
    neutral = (all(abs(v) < 1e-9 for v in hue_shift) and all(abs(v - 1) < 1e-9 for v in band_sat)
               and stock.saturation == 1.0 and stock.sat_shadows == 1.0 and stock.sat_highlights == 1.0)
    if neutral:
        return lin

    r, g, b = lin.unbind(-1)
    mean = (r + g + b) / 3.0
    u = r - 0.5 * (g + b)
    v = (math.sqrt(3.0) / 2.0) * (g - b)
    hue = torch.rad2deg(torch.atan2(v, u)) % 360.0

    shift = torch.zeros_like(hue)
    sat = torch.zeros_like(hue)
    for centre, d_hue, d_sat in zip(_BAND_CENTRES, hue_shift, band_sat):
        distance = torch.remainder(hue - centre + 180.0, 360.0) - 180.0
        weight = (1.0 - distance.abs() / 60.0).clamp_min(0.0)
        shift = shift + weight * d_hue
        sat = sat + weight * d_sat

    # Saturation by brightness, judged in display terms (what the eye sees).
    luma = r * _LUMA[0] + g * _LUMA[1] + b * _LUMA[2]
    tone = linear_to_srgb(luma).clamp(0.0, 1.0)
    tone = tone * tone * (3.0 - 2.0 * tone)
    sat = sat * stock.saturation * (stock.sat_shadows + (stock.sat_highlights - stock.sat_shadows) * tone)

    # Skin: a bump centred at ~20° (between red and yellow), ±30° wide, and only
    # for moderate chroma — a saturated red dress is not skin and keeps the push.
    # ⚠️ Chroma is measured in LINEAR light, where skin is far more chromatic
    # than its sRGB numbers suggest: sunlit skin measured 1.0-1.2 (chroma over
    # mean). A first cut-off at 0.6 switched the protection off exactly on faces.
    if stock.skin_protect > 0.0:
        skin_distance = (torch.remainder(hue - 20.0 + 180.0, 360.0) - 180.0).abs()
        skin = (1.0 - skin_distance / 30.0).clamp_min(0.0)
        chroma = torch.sqrt(u * u + v * v) / mean.clamp_min(1e-6)
        skin = skin * (1.0 - ((chroma - 1.3) / 0.8).clamp(0.0, 1.0))
        spare = skin * stock.skin_protect
        sat = sat + (1.0 - sat) * spare
        shift = shift * (1.0 - spare)

    angle = torch.deg2rad(shift)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    u2 = (u * cos_a - v * sin_a) * sat
    v2 = (u * sin_a + v * cos_a) * sat

    out = torch.stack((
        mean + (2.0 / 3.0) * u2,
        mean - u2 / 3.0 + v2 / math.sqrt(3.0),
        mean - u2 / 3.0 - v2 / math.sqrt(3.0),
    ), dim=-1)
    return _compress_into_gamut(out, mean)


def _compress_into_gamut(rgb: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Pull chroma back towards the mean until no channel is negative.

    Clamping a negative channel to zero shifts the hue; scaling the chroma
    keeps it and only gives up saturation.
    """
    low = rgb.amin(dim=-1)
    mean = mean.clamp_min(0.0)
    need = low < 0.0
    scale = torch.where(need, mean / (mean - low).clamp_min(1e-8), torch.ones_like(low))
    return mean.unsqueeze(-1) + (rgb - mean.unsqueeze(-1)) * scale.unsqueeze(-1)


def split_tone(lin: torch.Tensor, stock: FilmStock) -> torch.Tensor:
    if not any(stock.shadow_tint) and not any(stock.highlight_tint):
        return lin
    luma = (lin * torch.tensor(_LUMA, dtype=lin.dtype, device=lin.device)).sum(-1, keepdim=True)
    tone = linear_to_srgb(luma)
    shadows = (1.0 - tone) ** 2
    highlights = tone ** 2
    shadow_tint = torch.tensor(stock.shadow_tint, dtype=lin.dtype, device=lin.device)
    highlight_tint = torch.tensor(stock.highlight_tint, dtype=lin.dtype, device=lin.device)
    return (lin * (1.0 + shadows * shadow_tint * 4.0 + highlights * highlight_tint * 2.0)).clamp_min(0.0)


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def halation(lin: torch.Tensor, stock: FilmStock) -> torch.Tensor:
    """Red-orange glow around bright areas, blurred at low resolution."""
    if stock.halation <= 0.0:
        return lin
    batch, height, width, _ = lin.shape
    luma = (lin * torch.tensor(_LUMA, dtype=lin.dtype, device=lin.device)).sum(-1)
    source = ((luma - 0.55) / 0.45).clamp(0.0, 1.0) ** 2
    # Blur radius tied to the frame, not to pixels: the glow must look the
    # same on a preview and on the 4K original.
    scale = max(1, int(round(min(height, width) / 160)))
    small = F.avg_pool2d(source.unsqueeze(1), kernel_size=scale, stride=scale, ceil_mode=True)
    sigma = max(1.0, min(small.shape[-2:]) * 0.035)
    kernel = _gaussian_kernel(sigma, lin.device, lin.dtype)
    pad = kernel.numel() // 2
    small = F.conv2d(F.pad(small, (pad, pad, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    small = F.conv2d(F.pad(small, (0, 0, pad, pad), mode="replicate"), kernel.view(1, 1, -1, 1))
    glow = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    glow = glow.squeeze(1).unsqueeze(-1) * stock.halation
    colour = torch.tensor(stock.halation_color, dtype=lin.dtype, device=lin.device)
    glow = (glow * colour).clamp(0.0, 1.0)
    return 1.0 - (1.0 - lin.clamp(0.0, 1.0)) * (1.0 - glow)  # screen


def render_stock(image: torch.Tensor, stock: FilmStock) -> torch.Tensor:
    """Apply one stock to an sRGB IMAGE batch [B, H, W, 3]; returns a new tensor."""
    lin = srgb_to_linear(image)
    if stock.exposure:
        lin = lin * (2.0 ** stock.exposure)
    if stock.mono is not None:
        weights = torch.tensor(stock.mono, dtype=lin.dtype, device=lin.device)
        mono = (lin * weights).sum(-1, keepdim=True)
        lin = mono.expand_as(lin).contiguous()
    lin = characteristic_curve(lin, stock)
    if stock.mono is None:
        lin = hue_saturation(lin, stock)
        lin = split_tone(lin, stock)
    else:
        tone = torch.tensor(stock.mono_tone, dtype=lin.dtype, device=lin.device)
        lin = lin * tone
    lin = halation(lin, stock)
    return linear_to_srgb(lin)
