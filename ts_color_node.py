# TS Color & Film Tools for ComfyUI
# Advanced Film Emulation with Tone Compression and new film presets
# Save as: ComfyUI/custom_nodes/TS_Color_And_Film/nodes.py

import torch
import os
import torch.nn.functional as F
import numpy as np
import glob
import folder_paths
from PIL import Image
# ==========================================================
# 1️⃣ TS Color Grade Node
# ==========================================================
class TS_Color_Grade:
    CATEGORY = "Image/Color"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "lift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    @staticmethod
    def adjust_hue(image, hue):
        if abs(hue) < 1e-6:
            return image
        hue = hue / 180.0 * torch.pi
        U = torch.cos(hue)
        W = torch.sin(hue)
        mat = torch.tensor([
            [0.299 + 0.701 * U + 0.168 * W, 0.587 - 0.587 * U + 0.330 * W, 0.114 - 0.114 * U - 0.497 * W],
            [0.299 - 0.299 * U - 0.328 * W, 0.587 + 0.413 * U + 0.035 * W, 0.114 - 0.114 * U + 0.292 * W],
            [0.299 - 0.300 * U + 1.250 * W, 0.587 - 0.588 * U - 1.050 * W, 0.114 + 0.886 * U - 0.203 * W],
        ], device=image.device)
        return torch.clamp(image @ mat.T, 0, 1)

    def process(self, image, hue, saturation, contrast, gain, lift, gamma, brightness):
        img = image.clone().float().clamp(0, 1)

        img = self.adjust_hue(img, hue)

        if saturation != 1.0:
            gray = img.mean(dim=-1, keepdim=True)
            img = torch.lerp(gray, img, saturation)

        if contrast != 1.0:
            img = (img - 0.5) * contrast + 0.5

        if gain != 1.0 or abs(lift) > 1e-6:
            img = img * gain + lift

        if gamma != 1.0:
            img = torch.pow(torch.clamp(img, 0.0, 1.0), gamma)

        if abs(brightness) > 1e-6:
            img = img + brightness

        return (torch.clamp(img, 0.0, 1.0),)


# ==========================================================
# 2️⃣ TS Film Emulation Node (V5.0 - ENABLE + НОВЫЕ ДЕФОЛТЫ)
# ==========================================================
class TS_Film_Emulation:
    CATEGORY = "Image/Color"

    @classmethod
    def _scan_luts(cls):
        base_dir = os.path.dirname(__file__)
        luts_dir = os.path.join(base_dir, "luts")
        choices = ["None"]
        if os.path.isdir(luts_dir):
            for fn in sorted(os.listdir(luts_dir)):
                if fn.lower().endswith(".cube"):
                    choices.append(fn)
        return choices

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # <--- НОВЫЙ ЧЕКБОКС "ENABLE"
                "enable": ("BOOLEAN", {"default": True}),
                "film_preset": (["External LUT", "Kodak Vision3 250D", "Kodak Portra 400",
                                 "Fuji Eterna 250T", "Agfa Vista 200", "Ilford HP5",
                                 "Kodak Gold 200", "Fuji Superia 400"],),
                "lut_choice": (cls._scan_luts(),),
                "lut_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma_correction": ("BOOLEAN", {"default": True}),
                "film_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_curve": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "warmth": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                # <--- НОВЫЕ ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ
                "grain_intensity": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.01}),
                "grain_size": ("FLOAT", {"default": 0.5, "min": 0.5, "max": 5.0, "step": 0.1}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "shadow_saturation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "highlight_saturation": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    # Функции гамма-коррекции, идентичные OlmLUT, на Torch
    @staticmethod
    def _srgb_to_linear(image):
        return torch.where(image <= 0.04045, image / 12.92, torch.pow((torch.clamp(image, 0, 1) + 0.055) / 1.055, 2.4))

    @staticmethod
    def _linear_to_srgb(image):
        return torch.where(image <= 0.0031308, image * 12.92, 1.055 * torch.pow(torch.clamp(image, 0, 1), 1.0/2.4) - 0.055)
    
    # Загрузчик LUT с исправленным порядком осей
    def load_cube_lut(self, path):
        if not os.path.isfile(path): return None, None
        size = 0; data = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if "LUT_3D_SIZE" in parts[0].upper():
                    try: size = int(parts[-1])
                    except: size = 0
                elif len(parts) >= 3 and (parts[0][0].isdigit() or parts[0][0] in "-."):
                    try: data.append([float(v) for v in parts[:3]])
                    except: continue
        if size <= 0 or len(data) != size ** 3: return None, None
        lut = torch.tensor(data, dtype=torch.float32).view(size, size, size, 3)
        lut = lut.permute(2, 1, 0, 3) # <-- Ключевое исправление инверсии
        return lut, size

    # Применение LUT
    def _apply_3d_lut_trilinear(self, img, lut, size):
        device = img.device; lut = lut.to(device)
        img = torch.clamp(img, 0.0, 1.0)
        coords = img * (size - 1)
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        x0, y0, z0 = torch.floor(x).long().clamp(0, size-1), torch.floor(y).long().clamp(0, size-1), torch.floor(z).long().clamp(0, size-1)
        x1, y1, z1 = (x0 + 1).clamp(max=size-1), (y0 + 1).clamp(max=size-1), (z0 + 1).clamp(max=size-1)
        xd, yd, zd = (x - x0.float()).unsqueeze(-1), (y - y0.float()).unsqueeze(-1), (z - z0.float()).unsqueeze(-1)
        c000, c001 = lut[x0, y0, z0], lut[x0, y0, z1]; c010, c011 = lut[x0, y1, z0], lut[x0, y1, z1]
        c100, c101 = lut[x1, y0, z0], lut[x1, y0, z1]; c110, c111 = lut[x1, y1, z0], lut[x1, y1, z1]
        c00 = c000 * (1-zd) + c001 * zd; c01 = c010 * (1-zd) + c011 * zd
        c10 = c100 * (1-zd) + c101 * zd; c11 = c110 * (1-zd) + c111 * zd
        c0 = c00 * (1-yd) + c01 * yd; c1 = c10 * (1-yd) + c11 * yd
        return c0 * (1-xd) + c1 * xd

    # Остальные хелперы без изменений
    @staticmethod
    def _apply_contrast_curve(x, contrast=1.0): return torch.clamp(0.5 + (x - 0.5) * contrast, 0.0, 1.0)
    def apply_preset(self, image, preset_name):
        presets = {"Kodak Vision3 250D": {"warmth": 0.25, "fade": 0.05, "gamma": 0.95},"Kodak Portra 400": {"warmth": 0.3, "fade": 0.1, "gamma": 0.9},"Fuji Eterna 250T": {"warmth": -0.1, "fade": 0.05, "gamma": 0.95},"Agfa Vista 200": {"warmth": 0.1, "fade": 0.08, "gamma": 0.92},"Ilford HP5": {"warmth": 0.0, "fade": 0.15, "gamma": 0.85},"Kodak Gold 200": {"warmth": 0.2, "fade": 0.08, "gamma": 0.93},"Fuji Superia 400": {"warmth": 0.15, "fade": 0.1, "gamma": 0.9},}
        if preset_name not in presets or preset_name == "External LUT": return image
        p = presets[preset_name]
        img = torch.pow(torch.clamp(image, 0.0, 1.0), p["gamma"])
        img = img * (1 - p["fade"]) + p["fade"] * 0.5
        if abs(p["warmth"]) > 1e-6: img[..., 0] += p["warmth"] * 0.05; img[..., 2] -= p["warmth"] * 0.05
        return torch.clamp(img, 0, 1)
    @staticmethod
    def _smart_saturation(img, shadows_strength=1.0, highlights_strength=1.0):
        gray = img.mean(dim=-1, keepdim=True); shadows_mask = (gray < 0.5).float(); highlights_mask = (gray >= 0.5).float()
        out = img.clone()
        shadow_factor = shadows_mask * shadows_strength + (1 - shadows_mask); out = gray + (out - gray) * shadow_factor
        highlight_factor = highlights_mask * highlights_strength + (1 - highlights_mask); out = gray + (out - gray) * highlight_factor
        return torch.clamp(out, 0.0, 1.0)
    
    def process(self, image, enable=True, film_preset="External LUT", lut_choice="None", lut_strength=1.0, gamma_correction=False,
                film_strength=1.0, contrast_curve=1.0, warmth=0.0,
                grain_intensity=0.0, grain_size=1.0, fade=0.0,
                shadow_saturation=1.0, highlight_saturation=1.0):
        
        # <--- ЛОГИКА РАБОТЫ ЧЕКБОКСА ENABLE
        if not enable:
            return (image,)

        img = image.float().clamp(0, 1)
        out = img.clone()
        
        # Базовые эффекты
        if film_preset != "External LUT": out = torch.lerp(out, self.apply_preset(out, film_preset), film_strength)
        out = self._apply_contrast_curve(out, contrast_curve)
        if fade > 0: out = out * (1 - fade) + fade * 0.5
        if abs(warmth) > 1e-6:
            out[..., 0] = torch.clamp(out[..., 0] + warmth * 0.05, 0, 1)
            out[..., 2] = torch.clamp(out[..., 2] - warmth * 0.05, 0, 1)
        out = self._smart_saturation(out, shadows_strength=shadow_saturation, highlights_strength=highlight_saturation)

        # ЛОГИКА ПРИМЕНЕНИЯ LUT - ТОЧНАЯ КОПИЯ OlmLUT
        if film_preset == "External LUT" and lut_choice != "None":
            base_dir = os.path.dirname(__file__)
            lut_path = os.path.join(base_dir, "luts", lut_choice)
            lut, size = self.load_cube_lut(lut_path)
            if lut is not None and size is not None:
                
                original_for_lerp = out.clone()
                processed_image = out.clone()

                if gamma_correction:
                    image_for_lut = self._linear_to_srgb(processed_image)
                    lut_applied = self._apply_3d_lut_trilinear(image_for_lut, lut, size)
                    processed_image = self._srgb_to_linear(lut_applied)
                else:
                    processed_image = self._apply_3d_lut_trilinear(processed_image, lut, size)
                
                out = torch.lerp(original_for_lerp, processed_image, lut_strength)

        # Зерно
        if grain_intensity > 0:
            b, h, w, c = out.shape; grain_size = max(0.5, grain_size)
            noise_h = max(1, int(h / grain_size)); noise_w = max(1, int(w / grain_size))
            noise = torch.randn((b, noise_h, noise_w, 1), device=out.device)
            noise = torch.nn.functional.interpolate(noise.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            out = torch.clamp(out + noise * grain_intensity, 0.0, 1.0)

        return (out,)
# ==========================================================
# 🔧 Node registration
# ==========================================================
NODE_CLASS_MAPPINGS = {
    "TS_Color_Grade": TS_Color_Grade,
    "TS_Film_Emulation": TS_Film_Emulation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Color_Grade": "TS Color Grade",
    "TS_Film_Emulation": "TS Film Emulation",
}
