"""TS Film Emulation — film LUT and analog look (Kodak/Fuji/Agfa/Ilford presets + .cube LUTs).

node_id: TS_Film_Emulation
"""

import logging
import os

import torch

from comfy_api.v0_0_2 import IO

logger = logging.getLogger(__name__)


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
            category="TS/Image",
            inputs=[
                IO.Image.Input("image"),
                IO.Boolean.Input("enable", default=True),
                IO.Combo.Input(
                    "film_preset",
                    options=["External LUT", "Kodak Vision3 250D", "Kodak Portra 400",
                             "Fuji Eterna 250T", "Agfa Vista 200", "Ilford HP5",
                             "Kodak Gold 200", "Fuji Superia 400"],
                ),
                IO.Combo.Input("lut_choice", options=cls._scan_luts()),
                IO.Float.Input("lut_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Boolean.Input("gamma_correction", default=True),
                IO.Float.Input("film_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("contrast_curve", default=1.0, min=0.0, max=3.0, step=0.01),
                IO.Float.Input("warmth", default=0.0, min=-1.0, max=1.0, step=0.01),
                IO.Float.Input("grain_intensity", default=0.02, min=0.0, max=0.5, step=0.01),
                IO.Float.Input("grain_size", default=0.5, min=0.5, max=5.0, step=0.1),
                IO.Float.Input("fade", default=0.0, min=0.0, max=0.5, step=0.01),
                IO.Float.Input("shadow_saturation", default=0.8, min=0.0, max=2.0, step=0.01),
                IO.Float.Input("highlight_saturation", default=0.85, min=0.0, max=2.0, step=0.01),
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
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
        gray = img.mean(dim=-1, keepdim=True)
        shadows_mask = (gray < 0.5).float()
        highlights_mask = (gray >= 0.5).float()
        out = img.clone()
        shadow_factor = shadows_mask * shadows_strength + (1 - shadows_mask)
        out = gray + (out - gray) * shadow_factor
        highlight_factor = highlights_mask * highlights_strength + (1 - highlights_mask)
        out = gray + (out - gray) * highlight_factor
        return torch.clamp(out, 0.0, 1.0)

    @classmethod
    def execute(cls, image, enable=True, film_preset="External LUT", lut_choice="None", lut_strength=1.0,
                gamma_correction=False, film_strength=1.0, contrast_curve=1.0, warmth=0.0,
                grain_intensity=0.0, grain_size=1.0, fade=0.0,
                shadow_saturation=1.0, highlight_saturation=1.0) -> IO.NodeOutput:

        if not enable:
            return IO.NodeOutput(image)

        img = image.float().clamp(0, 1)
        out = img.clone()

        if film_preset != "External LUT":
            out = torch.lerp(out, cls.apply_preset(out, film_preset), film_strength)
        out = cls._apply_contrast_curve(out, contrast_curve)
        if fade > 0:
            out = out * (1 - fade) + fade * 0.5
        if abs(warmth) > 1e-6:
            out[..., 0] = torch.clamp(out[..., 0] + warmth * 0.05, 0, 1)
            out[..., 2] = torch.clamp(out[..., 2] - warmth * 0.05, 0, 1)
        out = cls._smart_saturation(out, shadows_strength=shadow_saturation, highlights_strength=highlight_saturation)

        if film_preset == "External LUT" and lut_choice != "None":
            lut_path = os.path.join(cls._resolve_luts_dir(), lut_choice)
            lut, size = cls.load_cube_lut(lut_path)
            if lut is not None and size is not None:
                original_for_lerp = out.clone()
                processed_image = out.clone()

                if gamma_correction:
                    image_for_lut = cls._linear_to_srgb(processed_image)
                    lut_applied = cls._apply_3d_lut_trilinear(image_for_lut, lut, size)
                    processed_image = cls._srgb_to_linear(lut_applied)
                else:
                    processed_image = cls._apply_3d_lut_trilinear(processed_image, lut, size)

                out = torch.lerp(original_for_lerp, processed_image, lut_strength)

        if grain_intensity > 0:
            b, h, w, c = out.shape
            grain_size = max(0.5, grain_size)
            noise_h = max(1, int(h / grain_size))
            noise_w = max(1, int(w / grain_size))
            noise = torch.randn((b, noise_h, noise_w, 1), device=out.device)
            noise = torch.nn.functional.interpolate(
                noise.permute(0, 3, 1, 2),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            out = torch.clamp(out + noise * grain_intensity, 0.0, 1.0)

        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"TS_Film_Emulation": TS_Film_Emulation}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Film_Emulation": "TS Film Emulation"}
