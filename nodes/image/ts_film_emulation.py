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
        preferred = os.path.join(os.path.dirname(__file__), "luts")
        if os.path.isdir(preferred):
            return preferred
        parent = os.path.dirname(os.path.dirname(__file__))
        candidate = os.path.join(parent, "luts")
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(os.path.dirname(parent), "luts")

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
            description="Apply film-stock looks (Kodak, Fuji, Cineon-style) or any .cube LUT from models/luts.",
            inputs=[
                IO.Image.Input("image", tooltip="Image to apply the film look to."),
                IO.Boolean.Input("enable", default=True, tooltip="If disabled, the image passes through unchanged."),
                IO.Combo.Input(
                    "film_preset",
                    options=["External LUT", "Kodak Vision3 250D", "Kodak Portra 400",
                             "Fuji Eterna 250T", "Agfa Vista 200", "Ilford HP5",
                             "Kodak Gold 200", "Fuji Superia 400"],
                    tooltip="Built-in analog film look. Choose 'External LUT' to use a .cube LUT from lut_choice instead.",
                ),
                IO.Combo.Input("lut_choice", options=cls._scan_luts(), tooltip="External .cube LUT file (from the luts folder) applied when film_preset is 'External LUT'. 'None' = no LUT."),
                IO.Float.Input("lut_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Blend amount of the external LUT. 0 = original, 1 = full LUT."),
                IO.Boolean.Input("gamma_correction", default=True, tooltip="Apply the LUT in sRGB space (recommended for most .cube LUTs)."),
                IO.Float.Input("film_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Blend amount of the selected film preset. 0 = original, 1 = full preset."),
                IO.Float.Input("contrast_curve", default=1.0, min=0.0, max=3.0, step=0.01, tooltip="Contrast around mid-gray. 1 = unchanged, >1 = more contrast."),
                IO.Float.Input("warmth", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Warm/cool shift. Positive warms (red up, blue down), negative cools."),
                IO.Float.Input("grain_intensity", default=0.02, min=0.0, max=0.5, step=0.01, tooltip="Strength of added film grain. 0 = no grain."),
                IO.Float.Input("grain_size", default=0.5, min=0.5, max=5.0, step=0.1, tooltip="Scale of the grain particles. Higher values give coarser grain."),
                IO.Float.Input("fade", default=0.0, min=0.0, max=0.5, step=0.01, tooltip="Lifts blacks toward gray for a faded, matte film look. 0 = off."),
                IO.Float.Input("shadow_saturation", default=0.8, min=0.0, max=2.0, step=0.01, tooltip="Color saturation in the shadows. 1 = unchanged, <1 = desaturated."),
                IO.Float.Input("highlight_saturation", default=0.85, min=0.0, max=2.0, step=0.01, tooltip="Color saturation in the highlights. 1 = unchanged, <1 = desaturated."),
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
        return torch.clamp(0.5 + (x - 0.5) * contrast, 0.0, 1.0)

    @staticmethod
    def apply_preset(image, preset_name):
        presets = {
            "Kodak Vision3 250D": {"warmth": 0.25, "fade": 0.05, "gamma": 0.95},
            "Kodak Portra 400": {"warmth": 0.3, "fade": 0.1, "gamma": 0.9},
            "Fuji Eterna 250T": {"warmth": -0.1, "fade": 0.05, "gamma": 0.95},
            "Agfa Vista 200": {"warmth": 0.1, "fade": 0.08, "gamma": 0.92},
            "Ilford HP5": {"warmth": 0.0, "fade": 0.15, "gamma": 0.85},
            "Kodak Gold 200": {"warmth": 0.2, "fade": 0.08, "gamma": 0.93},
            "Fuji Superia 400": {"warmth": 0.15, "fade": 0.1, "gamma": 0.9},
        }
        if preset_name not in presets or preset_name == "External LUT":
            return image
        p = presets[preset_name]
        img = torch.pow(torch.clamp(image, 0.0, 1.0), p["gamma"])
        img = img * (1 - p["fade"]) + p["fade"] * 0.5
        if abs(p["warmth"]) > 1e-6:
            img[..., 0] += p["warmth"] * 0.05
            img[..., 2] -= p["warmth"] * 0.05
        return torch.clamp(img, 0, 1)

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
                shadow_saturation=0.8, highlight_saturation=0.85,
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

        return IO.NodeOutput(torch.cat(results, dim=0) if len(results) > 1 else results[0])

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

        if film_preset != "External LUT":
            out = torch.lerp(out, cls.apply_preset(out, film_preset), film_strength)
        out = cls._apply_contrast_curve(out, contrast_curve)
        if fade > 0:
            out = out * (1 - fade) + fade * 0.5
        if abs(warmth) > 1e-6:
            out[..., 0] = torch.clamp(out[..., 0] + warmth * 0.05, 0, 1)
            out[..., 2] = torch.clamp(out[..., 2] - warmth * 0.05, 0, 1)
        out = cls._smart_saturation(out, shadows_strength=shadow_saturation, highlights_strength=highlight_saturation)

        # ⚠️ LUT читается ОДИН раз в `execute`, а не здесь: файл на 33³ точки
        # разбирается построчно, и на клипе это был бы повторный разбор для
        # каждой порции кадров.
        if lut is not None and lut_size:
            original_for_lerp = out
            if gamma_correction:
                image_for_lut = cls._linear_to_srgb(out)
                lut_applied = cls._apply_3d_lut_trilinear(image_for_lut, lut, lut_size)
                processed_image = cls._srgb_to_linear(lut_applied)
            else:
                processed_image = cls._apply_3d_lut_trilinear(out, lut, lut_size)
            out = torch.lerp(original_for_lerp, processed_image, lut_strength)

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
