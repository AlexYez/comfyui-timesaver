"""TS Film Emulation — film LUT and analog look (Kodak/Fuji/Agfa/Ilford presets + .cube LUTs).

Handles single frames and whole clips: the batch is processed on the GPU in
chunks sized to fit the card, so a long 4K sequence costs the same peak memory
as a short one.

⚠️ THE GRAIN IS THE POINT OF THIS FILE, so it is worth saying what it models.
Real grain is not noise laid on top of a picture. Silver halide crystals develop
or they do not, and the visible speckle is a fluctuation in DENSITY — which is a
logarithmic quantity. Two consequences follow, and both were measured against
the old implementation before this one was written:

* **Grain rides the signal.** Adding a constant-sigma noise gave the same
  deviation everywhere: measured 0.049 / 0.060 / 0.049 in shadow, mid and
  highlight. Film does not behave that way — the same density fluctuation is a
  large linear swing in a bright area and a tiny one in a dark one. Applying the
  noise in log space reproduces that for free.
* **It fades at both ends.** Toward black there is almost nothing to fluctuate;
  toward the shoulder the emulsion saturates. So the modulation peaks in the
  upper mid-tones and falls away on both sides, rather than growing forever.

For clips there is also `grain_speed`, the control professional grain plugins
expose: at 1.0 the pattern is redrawn every frame (lively, digital), lower
values hold one pattern across several frames the way a real scanned negative
does when the projector runs faster than the grain changes.

node_id: TS_Film_Emulation
"""

import logging
import os

import torch
from comfy_api.v0_0_2 import IO

from ._film_stocks import PRESET_OPTIONS, STOCK_BY_NAME, render_stock

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TS Film Emulation]"

#: Bytes of working memory one pixel needs on the heaviest path.
#:
#: ⚠️ Measured, and the first guess was wrong by an order of magnitude. The LUT
#: path materialises eight corner tensors plus the interpolation ladder, so a
#: chunk of 24 1080p frames peaked at **15 GB** of VRAM — almost the whole card,
#: on a node that is supposed to leave room for everything else in the graph.
#: About 180 bytes per pixel is what that works out to; the budget below is
#: derived from it rather than from a frame count.
_BYTES_PER_PIXEL = 180

#: Ceiling on one chunk when the free-memory probe is unavailable, and an upper
#: bound even when it is not: roughly one 4K frame's worth of working set.
_FALLBACK_PIXELS_PER_CHUNK = 4 * 1920 * 1080

#: Never take more than this share of what the card has free — the rest of the
#: graph (the sampler, the model that made these frames) still has to live.
_VRAM_SHARE = 0.25


class TS_Film_Emulation(IO.ComfyNode):
    @classmethod
    def _resolve_luts_dir(cls):
        # The pack's LUTs live in nodes/luts. Since 12.13 this file sits two
        # levels below it (nodes/image/film_emulation/), so the path is spelled
        # out instead of being found by walking up and hoping.
        nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(nodes_dir, "luts")

    @classmethod
    def _scan_luts(cls):
        luts_dir = cls._resolve_luts_dir()
        choices = ["None"]
        if os.path.isdir(luts_dir):
            for fn in sorted(os.listdir(luts_dir)):
                if fn.lower().endswith(".cube"):
                    choices.append(fn)
        return choices

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Film_Emulation",
            display_name="TS Film Emulation",
            category="TS/Image/Color",
            description=(
                "Film looks: 24 modelled stocks (Kodak, Fuji, CineStill, Kodachrome, Polaroid, "
                "black and white) or any .cube LUT from the pack's luts folder. Click the "
                "thumbnail to pick a look from a gallery of previews."
            ),
            inputs=[
                IO.Image.Input("image", tooltip="Image to apply the film look to."),
                IO.Boolean.Input("enable", default=True, tooltip="Kept for older workflows; the node hides it — use Bypass instead. If disabled, the image passes through unchanged."),
                IO.Combo.Input(
                    "film_preset",
                    # "External LUT" first: no default is set, so the first
                    # option IS the default. The seven pre-12.13 stocks follow
                    # in their old order, the new ones after them.
                    options=list(PRESET_OPTIONS),
                    tooltip="Built-in analog film look. Choose 'External LUT' to use a .cube LUT from lut_choice instead.",
                ),
                IO.Combo.Input("lut_choice", options=cls._scan_luts(), tooltip="External .cube LUT file (from the luts folder) applied when film_preset is 'External LUT'. 'None' = no LUT."),
                IO.Float.Input("lut_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Blend amount of the external LUT. 0 = original, 1 = full LUT."),
                IO.Boolean.Input("gamma_correction", default=True, tooltip="Kept for older workflows and ignored: since 12.13 a .cube LUT is always applied to the sRGB values it was made for."),
                IO.Float.Input("film_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Blend amount of the selected film preset. 0 = original, 1 = full preset."),
                IO.Float.Input("contrast_curve", default=1.0, min=0.0, max=3.0, step=0.01, tooltip="S-curve contrast around mid-gray; black and white are never clipped. 1 = unchanged, >1 = more contrast."),
                IO.Float.Input("warmth", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="White balance: positive warms, negative cools. Black stays black."),
                IO.Float.Input("grain_intensity", default=0.02, min=0.0, max=0.5, step=0.01, tooltip="Strength of added film grain. 0 = no grain."),
                IO.Float.Input("grain_size", default=0.5, min=0.5, max=5.0, step=0.1, tooltip="Scale of the grain particles. Higher values give coarser grain."),
                IO.Float.Input("fade", default=0.0, min=0.0, max=0.5, step=0.01, tooltip="Matte film look: lifts the blacks, whites stay white. 0 = off."),
                # ⚠️ Defaults were 0.8 / 0.85 until 12.13: every image lost a
                # fifth of its colour even with no look selected. Saved graphs
                # keep their own values; only new nodes start neutral.
                IO.Float.Input("shadow_saturation", default=1.0, min=0.0, max=2.0, step=0.01, tooltip="Color saturation in the shadows. 1 = unchanged, <1 = desaturated."),
                IO.Float.Input("highlight_saturation", default=1.0, min=0.0, max=2.0, step=0.01, tooltip="Color saturation in the highlights. 1 = unchanged, <1 = desaturated."),
                # ⚠️ Оба входа — ПОСЛЕДНИМИ и optional. `widgets_values`
                # позиционен: вставленный выше вход сдвинул бы все значения в
                # каждом уже сохранённом workflow с этой нодой.
                IO.Float.Input(
                    "grain_speed",
                    default=1.0, min=0.05, max=1.0, step=0.05, optional=True,
                    tooltip=(
                        "For clips: how often the grain pattern is redrawn. 1.0 = a new "
                        "pattern every frame (lively, digital). 0.5 = one pattern held for "
                        "two frames, 0.25 for four — closer to scanned film, where the grain "
                        "does not race the action. No effect on a single image."
                    ),
                ),
                IO.Int.Input(
                    "grain_seed",
                    default=0, min=0, max=0xFFFFFFFF, optional=True,
                    tooltip=(
                        "Seed for the grain. The same seed gives the same grain on the same "
                        "frames, so a re-render matches the take you graded."
                    ),
                ),
            ],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @staticmethod
    def _srgb_to_linear(image):
        return torch.where(
            image <= 0.04045,
            image / 12.92,
            torch.pow((torch.clamp(image, 0, 1) + 0.055) / 1.055, 2.4),
        )

    @staticmethod
    def _linear_to_srgb(image):
        return torch.where(
            image <= 0.0031308,
            image * 12.92,
            1.055 * torch.pow(torch.clamp(image, 0, 1), 1.0 / 2.4) - 0.055,
        )

    @staticmethod
    def load_cube_lut(path):
        if not os.path.isfile(path):
            return None, None
        size = 0
        data = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if "LUT_3D_SIZE" in parts[0].upper():
                    try:
                        size = int(parts[-1])
                    except Exception:
                        size = 0
                elif len(parts) >= 3 and (parts[0][0].isdigit() or parts[0][0] in "-."):
                    try:
                        data.append([float(v) for v in parts[:3]])
                    except ValueError as exc:
                        logger.debug("[TS Film Emulation] Skipping malformed LUT line %r: %s", line, exc)
                        continue
        if size <= 0 or len(data) != size ** 3:
            return None, None
        lut = torch.tensor(data, dtype=torch.float32).view(size, size, size, 3)
        lut = lut.permute(2, 1, 0, 3)
        return lut, size

    @staticmethod
    def _apply_3d_lut_trilinear(img, lut, size):
        device = img.device
        lut = lut.to(device)
        img = torch.clamp(img, 0.0, 1.0)
        coords = img * (size - 1)
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        x0 = torch.floor(x).long().clamp(0, size - 1)
        y0 = torch.floor(y).long().clamp(0, size - 1)
        z0 = torch.floor(z).long().clamp(0, size - 1)
        x1, y1, z1 = (x0 + 1).clamp(max=size - 1), (y0 + 1).clamp(max=size - 1), (z0 + 1).clamp(max=size - 1)
        xd = (x - x0.float()).unsqueeze(-1)
        yd = (y - y0.float()).unsqueeze(-1)
        zd = (z - z0.float()).unsqueeze(-1)
        c000, c001 = lut[x0, y0, z0], lut[x0, y0, z1]
        c010, c011 = lut[x0, y1, z0], lut[x0, y1, z1]
        c100, c101 = lut[x1, y0, z0], lut[x1, y0, z1]
        c110, c111 = lut[x1, y1, z0], lut[x1, y1, z1]
        c00 = c000 * (1 - zd) + c001 * zd
        c01 = c010 * (1 - zd) + c011 * zd
        c10 = c100 * (1 - zd) + c101 * zd
        c11 = c110 * (1 - zd) + c111 * zd
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd
        return c0 * (1 - xd) + c1 * xd

    @staticmethod
    def _apply_contrast_curve(x, contrast=1.0):
        """S-curve around mid-grey whose ends stay pinned at 0 and 1.

        ⚠️ Until 12.13 this was ``0.5 + (x - 0.5) * c`` with a clamp: at 1.3
        everything below 0.115 and above 0.885 was cut flat — a sky lost its
        gradient, a shadow its texture. ``x^p / (x^p + (1-x)^p)`` has slope
        ``p`` at the pivot like the old line, but bends into the ends instead
        of hitting them; 0 still flattens the image to mid-grey.
        """
        if abs(contrast - 1.0) < 1e-6:
            return x
        if contrast <= 1e-6:
            return torch.full_like(x, 0.5)
        x = x.clamp(1e-6, 1.0 - 1e-6)
        a = x ** contrast
        b = (1.0 - x) ** contrast
        return a / (a + b)

    @classmethod
    def _apply_warmth(cls, x, warmth):
        """White balance in linear light: red up, blue down (or back).

        ⚠️ The old version ADDED ±0.05 to red and blue, which tinted pure black
        too. A gain in linear light leaves black black, as a real white
        balance does. ±1 is about ±0.35 stops on each of the two channels.
        """
        lin = cls._srgb_to_linear(x)
        gain = torch.tensor(
            [2.0 ** (0.35 * warmth), 1.0, 2.0 ** (-0.35 * warmth)], dtype=x.dtype, device=x.device,
        )
        return cls._linear_to_srgb((lin * gain).clamp(0.0, 1.0))

    @staticmethod
    def _apply_fade(x, fade):
        """Matte look: lift the blacks, leave the whites.

        The old lerp towards 0.5 also pulled white down to grey. Here black
        rises exactly as far as it did (to ``fade / 2``) and white stays at 1 —
        what a faded print does — so the slider keeps its old scale.
        """
        lift = 0.5 * fade
        return lift + x * (1.0 - lift)

    @staticmethod
    def apply_preset(image, preset_name):
        """Render a built-in film stock (see _film_stocks.py); other names pass through.

        ⚠️ Since 12.13 the stocks are modelled (curve per dye layer, hue bands,
        split toning, halation, true B&W) instead of gamma + fade + warmth. The
        seven old names are kept, so saved workflows open, but they render
        differently — this was asked for, and it is in the CHANGELOG.
        """
        stock = STOCK_BY_NAME.get(preset_name)
        if stock is None:
            return image
        return render_stock(image, stock).clamp(0.0, 1.0)

    @staticmethod
    def _smart_saturation(img, shadows_strength=1.0, highlights_strength=1.0):
        # ⚠️ Переход между тенями и светами — ПЛАВНЫЙ, хотя раньше маска была
        # ступенькой на `gray < 0.5`. На стоп-кадре ступенька почти незаметна, а
        # на видео она видна прекрасно: пиксель, гуляющий вокруг средней яркости
        # (от зерна, от компрессии, от дрожания света), скачет между двумя
        # разными насыщенностями и даёт мерцающую кайму по градиентам.
        gray = img.mean(dim=-1, keepdim=True)
        # smoothstep на половине стопа вокруг средне-серого: достаточно узко,
        # чтобы раздельная подкраска теней и светов сохранилась, и достаточно
        # широко, чтобы граница перестала звенеть.
        edge = torch.clamp((gray - 0.35) / 0.3, 0.0, 1.0)
        weight = edge * edge * (3.0 - 2.0 * edge)
        factor = shadows_strength + (highlights_strength - shadows_strength) * weight
        out = gray + (img - gray) * factor
        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def _grain_response(luma):
        """Насколько сильно зерно проявляется при этой яркости.

        Не константа и не прямая: кривая растёт от чёрного, пикует в верхних
        средних тонах и спадает к белому — так ведёт себя плотность эмульсии.
        Возвращает множитель 0..1.
        """
        x = torch.clamp(luma, 0.0, 1.0)
        # Подъём от чёрного: в тенях флуктуировать почти нечему.
        toe = torch.sqrt(x)
        # Спад у плеча: к белому эмульсия насыщается и зерно вырождается.
        shoulder = torch.clamp(1.0 - torch.pow(x, 3.0), 0.0, 1.0)
        return toe * (0.35 + 0.65 * shoulder)

    @classmethod
    def _apply_grain(cls, out, *, intensity, size, speed, seed, frame_offset):
        """Логарифмическое зерно, одинаковое при любом размере чанка.

        Паттерн привязан к НОМЕРУ КАДРА и к seed, а не к позиции внутри чанка:
        иначе результат зависел бы от того, как нода поделила клип на порции, и
        зерно «перещёлкивало» бы на стыках.
        """
        if intensity <= 0:
            return out

        batch, height, width, _ = out.shape
        size = max(0.5, float(size))
        noise_h = max(1, int(round(height / size)))
        noise_w = max(1, int(round(width / size)))

        # `speed` = 1.0 — новый паттерн каждый кадр; 0.5 — держится два кадра.
        speed = min(1.0, max(0.01, float(speed)))
        frames = []
        for index in range(batch):
            pattern_id = int((frame_offset + index) * speed)
            generator = torch.Generator(device="cpu").manual_seed(
                (int(seed) * 1_000_003 + pattern_id) & 0x7FFF_FFFF
            )
            frames.append(torch.randn((1, 1, noise_h, noise_w), generator=generator))
        noise = torch.cat(frames, dim=0).to(out.device, dtype=out.dtype)

        if (noise_h, noise_w) != (height, width):
            noise = torch.nn.functional.interpolate(
                noise, size=(height, width), mode="bilinear", align_corners=False,
            )
            # Растягивание гасит амплитуду тем сильнее, чем крупнее зерно;
            # без поправки "grain_size" незаметно работал бы регулятором силы.
            noise = noise / noise.std().clamp_min(1e-6)
        noise = noise.permute(0, 2, 3, 1)

        # Плёночная яркость — с весами восприятия, а не среднее по каналам:
        # зерно должно следовать за тем, что глаз считает светом.
        luma = (out[..., 0:1] * 0.2126 + out[..., 1:2] * 0.7152 + out[..., 2:3] * 0.0722)
        response = cls._grain_response(luma)

        # Собственно логарифм: отклонение задаётся в плотности, а в картинку
        # приходит умножением. Отсюда и разная видимость по тонам — она берётся
        # из математики, а не из отдельного «усилителя для светов».
        sigma = float(intensity) * 1.6
        grain = torch.exp(noise * sigma * response)
        return torch.clamp(out * grain, 0.0, 1.0)

    @classmethod
    def execute(cls, image, enable=True, film_preset="External LUT", lut_choice="None", lut_strength=1.0,
                gamma_correction=True, film_strength=1.0, contrast_curve=1.0, warmth=0.0,
                grain_intensity=0.02, grain_size=0.5, fade=0.0,
                shadow_saturation=1.0, highlight_saturation=1.0,
                grain_speed=1.0, grain_seed=0) -> IO.NodeOutput:
        # ⚠️ Эти значения обязаны совпадать с `default=` в схеме выше.
        #
        # Пять из них разошлись: схема обещала `gamma_correction=True` и зерно
        # 0.02, а подпись — `False` и 0.0. На результат это не влияло (входы
        # обязательные, и ComfyUI всегда передаёт их явно — из `widgets_values`
        # либо из схемы), поэтому расхождение и жило незамеченным. Но читается
        # такой код как второй, тайный набор умолчаний: стоит кому-нибудь
        # сделать вход опциональным или позвать `execute` напрямую из теста —
        # и нода тихо начнёт считать по другим числам.
        #
        # Сторож: tests/test_schema_execute_defaults.py — по всему паку.

        if not enable:
            return IO.NodeOutput(image)

        # ⚠️ IMAGE бывает с альфой: VAE Qwen Image 2.1 в ядре ComfyUI объявлен
        # RGBA, и VAEDecode отдаёт 4 канала. Плёнка — это цвет, поэтому
        # считаем три канала, а альфу (и всё, что за ней) отдаём как пришло.
        # Без этого LUT сводил 4 канала к 3, и lerp падал на несовпадении.
        channels = int(image.shape[-1])
        if channels < 3:
            raise ValueError(f"{LOG_PREFIX} Expected an RGB or RGBA image, got {channels} channel(s).")
        extra = image[..., 3:] if channels > 3 else None
        if extra is not None:
            image = image[..., :3]

        lut, lut_size = (None, None)
        if film_preset == "External LUT" and lut_choice != "None":
            lut, lut_size = cls.load_cube_lut(os.path.join(cls._resolve_luts_dir(), lut_choice))
            if lut is None:
                logger.warning("%s Could not read LUT %r — passing the look through without it.",
                               LOG_PREFIX, lut_choice)

        device = cls._work_device(image)
        if lut is not None:
            lut = lut.to(device)

        frames = int(image.shape[0])
        pixels = max(1, int(image.shape[1]) * int(image.shape[2]))
        chunk = max(1, min(frames, cls._pixels_per_chunk(device) // pixels))

        results = []
        # ⚠️ Без `no_grad` torch строит граф на каждом шаге: на клипе это чистый
        # расход памяти, потому что градиенты здесь никому не нужны.
        with torch.no_grad():
            for start in range(0, frames, chunk):
                part = image[start:start + chunk].to(device=device, dtype=torch.float32)
                processed = cls._process_chunk(
                    part, film_preset=film_preset, lut=lut, lut_size=lut_size,
                    lut_strength=lut_strength, gamma_correction=gamma_correction,
                    film_strength=film_strength, contrast_curve=contrast_curve, warmth=warmth,
                    fade=fade, shadow_saturation=shadow_saturation,
                    highlight_saturation=highlight_saturation, grain_intensity=grain_intensity,
                    grain_size=grain_size, grain_speed=grain_speed, grain_seed=grain_seed,
                    frame_offset=start,
                )
                # Возвращаем на CPU сразу: IMAGE в ComfyUI живёт там, и держать
                # весь клип на карте ради одной склейки — верный путь в OOM.
                results.append(processed.to("cpu"))

        out = torch.cat(results, dim=0) if len(results) > 1 else results[0]
        if extra is not None:
            out = torch.cat([out, extra.to(out.device, out.dtype)], dim=-1)
        return IO.NodeOutput(out)

    @staticmethod
    def _pixels_per_chunk(device):
        """Сколько пикселей брать за раз — по тому, что на карте свободно.

        Фиксированное число здесь не годится: на карте с 8 ГБ и на карте с 48
        уместны разные порции, а рядом с нодой обычно живёт ещё и модель,
        которая эти кадры сделала.
        """
        if getattr(device, "type", "cpu") == "cpu":
            return _FALLBACK_PIXELS_PER_CHUNK
        try:
            import comfy.model_management as mm

            free_bytes = float(mm.get_free_memory(device)) * _VRAM_SHARE
            pixels = int(free_bytes // _BYTES_PER_PIXEL)
        except Exception as exc:  # noqa: BLE001 - без замера идём по умолчанию
            logger.debug("%s Could not read free VRAM (%s); using the default chunk.",
                         LOG_PREFIX, exc)
            return _FALLBACK_PIXELS_PER_CHUNK
        # Нижняя граница — один кадр 1080p: меньше резать смысла нет, накладные
        # на перенос съедят выигрыш.
        return max(1920 * 1080, min(pixels, _FALLBACK_PIXELS_PER_CHUNK))

    @staticmethod
    def _work_device(image):
        """Карта, если она есть; иначе — там, где картинка и лежала."""
        try:
            import comfy.model_management as mm

            device = mm.get_torch_device()
            if device is not None and device.type != "cpu":
                return device
        except Exception as exc:  # noqa: BLE001 - вне ComfyUI считаем на CPU
            logger.debug("%s No ComfyUI device manager (%s); staying on the input device.",
                         LOG_PREFIX, exc)
        return image.device

    @classmethod
    def _process_chunk(cls, img, *, film_preset, lut, lut_size, lut_strength, gamma_correction,
                       film_strength, contrast_curve, warmth, fade, shadow_saturation,
                       highlight_saturation, grain_intensity, grain_size, grain_speed,
                       grain_seed, frame_offset):
        out = img.clamp(0, 1)

        # 1. The look: a stock OR a LUT, blended by its strength.
        if film_preset != "External LUT":
            out = torch.lerp(out, cls.apply_preset(out, film_preset), film_strength)
        elif lut is not None and lut_size:
            # ⚠️ LUT читается ОДИН раз в `execute`, а не здесь: файл на 33³
            # точки разбирается построчно, и на клипе это был бы повторный
            # разбор для каждой порции кадров.
            #
            # ⚠️ Applied straight to the sRGB values: a .cube made for display
            # (Rec.709 / sRGB in and out — every creative LUT) expects exactly
            # them. Until 12.13 the default `gamma_correction=True` ran the
            # image through linear->sRGB first — on an image that already WAS
            # sRGB — so the LUT read the wrong tones (Cinematic1: mid-grey 0.41
            # became 0.47). The input is kept for old graphs and ignored.
            looked = cls._apply_3d_lut_trilinear(out, lut, lut_size).clamp(0.0, 1.0)
            out = torch.lerp(out, looked, lut_strength)

        # 2. The user's adjustments, on top of the look — the same order for
        # stocks and LUTs (LUTs used to get them BEFORE the look).
        out = cls._apply_contrast_curve(out, contrast_curve)
        if fade > 0:
            out = cls._apply_fade(out, fade)
        if abs(warmth) > 1e-6:
            out = cls._apply_warmth(out, warmth)
        if abs(shadow_saturation - 1.0) > 1e-6 or abs(highlight_saturation - 1.0) > 1e-6:
            out = cls._smart_saturation(out, shadows_strength=shadow_saturation,
                                        highlights_strength=highlight_saturation)

        # 3. Grain last: it belongs to the film, not to the colour.
        return cls._apply_grain(
            out,
            intensity=grain_intensity,
            size=grain_size,
            speed=grain_speed,
            seed=grain_seed,
            frame_offset=frame_offset,
        )


NODE_CLASS_MAPPINGS = {"TS_Film_Emulation": TS_Film_Emulation}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Film_Emulation": "TS Film Emulation"}

# Gallery routes (/ts_film/looks, /ts_film/preview). Imported for the side effect
# of registering them; the module renders through this file's pipeline.
from . import _film_previews  # noqa: E402,F401
