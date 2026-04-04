import torch
import math
import numpy as np
from PIL import Image

try:
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class TS_ImageResize:
    def __init__(self):
        pass

    LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "bicubic", "area", "lanczos"]

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK",)
    RETURN_NAMES = ("IMAGE", "width", "height", "MASK",)
    FUNCTION = "resize"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "keep_proportion": ("BOOLEAN", {"default": True}),
                "upscale_method": (s.UPSCALE_METHODS, {"default": "bicubic"}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 256.0, "step": 0.01}),
                "dont_enlarge": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, **_):
        return True

    def _pil_resize(self, image_tensor_nchw, size_wh):
        target_w, target_h = size_wh
        pil_images = []
        is_mask = image_tensor_nchw.shape[1] == 1
        
        for i in range(image_tensor_nchw.shape[0]):
            img_tensor = image_tensor_nchw[i]
            if is_mask:
                img_tensor_chw = img_tensor.squeeze(0)
                img_np = img_tensor_chw.cpu().numpy() * 255.0
                pil_image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), 'L')
            else:
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
                pil_image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            
            resample_method = Image.Resampling.NEAREST if is_mask else self.LANCZOS
            resized_pil = pil_image.resize((target_w, target_h), resample_method)
            pil_images.append(resized_pil)
            
        output_tensors = []
        for pil_img in pil_images:
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            if is_mask:
                output_tensors.append(torch.from_numpy(img_np).unsqueeze(0))
            else:
                output_tensors.append(torch.from_numpy(img_np).permute(2, 0, 1))
            
        return torch.stack(output_tensors, dim=0).to(image_tensor_nchw.device)

    def _interp_image(self, image_tensor_nchw, size_wh, method):
        target_w, target_h = size_wh
        if method == "lanczos":
            return self._pil_resize(image_tensor_nchw, (target_w, target_h))
        else:
            interp_kwargs = {"mode": method}
            if TORCHVISION_AVAILABLE and method in ["bilinear", "bicubic"]:
                interp_kwargs["antialias"] = True
            return torch.nn.functional.interpolate(image_tensor_nchw, size=(target_h, target_w), **interp_kwargs)

    def resize(self, pixels, target_width, target_height, smaller_side, larger_side, scale_factor, keep_proportion, upscale_method, divisible_by, megapixels, dont_enlarge, mask=None):
        _B, original_H, original_W, _C = pixels.shape
        ideal_w, ideal_h = float(original_W), float(original_H)

        _megapixels = megapixels if megapixels is not None else 0.0
        _scale_factor = scale_factor if scale_factor is not None else 0.0
        _target_width = target_width if target_width is not None else 0
        _target_height = target_height if target_height is not None else 0
        _smaller_side = smaller_side if smaller_side is not None else 0
        _larger_side = larger_side if larger_side is not None else 0
        _divisible_by = divisible_by if divisible_by is not None else 1
        if _divisible_by < 1: _divisible_by = 1

        chosen_method = None
        if _scale_factor > 0.0: chosen_method = "scale_factor"
        elif _target_width > 0 or _target_height > 0: chosen_method = "target_dims"
        elif _smaller_side > 0 or _larger_side > 0: chosen_method = "side_dims"
        elif _megapixels > 0.0: chosen_method = "megapixels"

        # --- 1. Р Р°СЃС‡РµС‚ 'РёРґРµР°Р»СЊРЅС‹С…' СЂР°Р·РјРµСЂРѕРІ РЅР° РѕСЃРЅРѕРІРµ РІС‹Р±СЂР°РЅРЅРѕРіРѕ РјРµС‚РѕРґР° ---
        if chosen_method == "megapixels":
            if original_H > 0 and original_W > 0:
                aspect_ratio = ideal_w / ideal_h
                total_pixels = _megapixels * 1000000
                ideal_h = math.sqrt(total_pixels / aspect_ratio)
                ideal_w = ideal_h * aspect_ratio
        elif chosen_method == "scale_factor":
            ideal_w *= _scale_factor
            ideal_h *= _scale_factor
        elif chosen_method == "target_dims":
            if keep_proportion:
                 # Р”Р»СЏ 'cover and crop' Рё РѕРґРЅРѕРіРѕ Р·Р°РґР°РЅРЅРѕРіРѕ СЂР°Р·РјРµСЂР°,
                 # РёРґРµР°Р»СЊРЅС‹Рµ СЂР°Р·РјРµСЂС‹ - СЌС‚Рѕ РїСЂРѕСЃС‚Рѕ С†РµР»РµРІС‹Рµ.
                 # Р”Р»СЏ РѕРґРЅРѕРіРѕ СЂР°Р·РјРµСЂР° - РїСЂРѕРїРѕСЂС†РёРѕРЅР°Р»СЊРЅРѕ РјР°СЃС€С‚Р°Р±РёСЂСѓРµРј.
                ratio = ideal_w / ideal_h if ideal_h != 0 else float('inf')
                if _target_width > 0 and _target_height > 0:
                    ideal_w, ideal_h = float(_target_width), float(_target_height)
                elif _target_width > 0:
                    ideal_w = float(_target_width)
                    ideal_h = ideal_w / ratio if ratio != 0 else 0
                elif _target_height > 0:
                    ideal_h = float(_target_height)
                    ideal_w = ideal_h * ratio
            else: # Р‘РµР· СЃРѕС…СЂР°РЅРµРЅРёСЏ РїСЂРѕРїРѕСЂС†РёР№
                if _target_width > 0: ideal_w = float(_target_width)
                if _target_height > 0: ideal_h = float(_target_height)
        elif chosen_method == "side_dims":
            ratio = ideal_w / ideal_h if ideal_h != 0 else float('inf')
            if _smaller_side > 0:
                if ideal_w < ideal_h:
                    ideal_w = float(_smaller_side)
                    ideal_h = ideal_w / ratio if ratio != 0 else 0
                else:
                    ideal_h = float(_smaller_side)
                    ideal_w = ideal_h * ratio
            elif _larger_side > 0:
                if ideal_w > ideal_h:
                    ideal_w = float(_larger_side)
                    ideal_h = ideal_w / ratio if ratio != 0 else 0
                else:
                    ideal_h = float(_larger_side)
                    ideal_w = ideal_h * ratio

        # --- 2. РџСЂРёРјРµРЅРµРЅРёРµ 'dont_enlarge' ---
        if dont_enlarge and chosen_method is not None:
            if ideal_w * ideal_h > original_W * original_H:
                ideal_w, ideal_h = float(original_W), float(original_H)
        
        # --- 3. РћРїСЂРµРґРµР»РµРЅРёРµ С„РёРЅР°Р»СЊРЅС‹С… СЂР°Р·РјРµСЂРѕРІ С…РѕР»СЃС‚Р° ---
        is_cover_crop_mode = (chosen_method == "target_dims" and keep_proportion and _target_width > 0 and _target_height > 0)
        is_free_distort_mode = (chosen_method == "target_dims" and not keep_proportion and _target_width > 0 and _target_height > 0)
        
        final_w, final_h = 0, 0
        if is_cover_crop_mode or is_free_distort_mode:
            # Р’ СЌС‚РёС… РґРІСѓС… СЂРµР¶РёРјР°С… 'divisible_by' РР“РќРћР РР РЈР•РўРЎРЇ,
            # С‚Р°Рє РєР°Рє РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ СЏРІРЅРѕ СѓРєР°Р·Р°Р» РєРѕРЅРµС‡РЅРѕРµ СЂР°Р·СЂРµС€РµРЅРёРµ.
            final_w = int(round(ideal_w))
            final_h = int(round(ideal_h))
        else:
            # Р’Рѕ РІСЃРµС… РѕСЃС‚Р°Р»СЊРЅС‹С… РїСЂРѕРїРѕСЂС†РёРѕРЅР°Р»СЊРЅС‹С… СЂРµР¶РёРјР°С… СЃС‚СЂРѕРіРѕ СЃРѕР±Р»СЋРґР°РµРј РєСЂР°С‚РЅРѕСЃС‚СЊ.
            if _divisible_by > 1:
                final_w = math.floor(ideal_w / _divisible_by) * _divisible_by
                final_h = math.floor(ideal_h / _divisible_by) * _divisible_by
                final_w = max(_divisible_by, final_w)
                final_h = max(_divisible_by, final_h)
            else:
                final_w = int(round(ideal_w))
                final_h = int(round(ideal_h))

        final_w = max(1, final_w)
        final_h = max(1, final_h)

        current_pixels_nchw = pixels.permute(0, 3, 1, 2)
        current_mask_nchw = mask.unsqueeze(1) if mask is not None else None
        
        # --- 4. Р•РґРёРЅР°СЏ Р»РѕРіРёРєР° РјР°СЃС€С‚Р°Р±РёСЂРѕРІР°РЅРёСЏ Рё РѕР±СЂРµР·РєРё ---
        if final_w == original_W and final_h == original_H:
             final_output_pixels_nchw = current_pixels_nchw
        elif is_free_distort_mode:
            # РџСЂРѕСЃС‚РѕРµ РјР°СЃС€С‚Р°Р±РёСЂРѕРІР°РЅРёРµ СЃ РёСЃРєР°Р¶РµРЅРёРµРј
            final_output_pixels_nchw = self._interp_image(current_pixels_nchw, (final_w, final_h), upscale_method)
            if current_mask_nchw is not None:
                current_mask_nchw = self._interp_image(current_mask_nchw, (final_w, final_h), "nearest-exact")
        else:
            # РЈРЅРёРІРµСЂСЃР°Р»СЊРЅР°СЏ Р»РѕРіРёРєР° "РїРѕРєСЂС‹С‚СЊ Рё РѕР±СЂРµР·Р°С‚СЊ" РґР»СЏ РІСЃРµС… РїСЂРѕРїРѕСЂС†РёРѕРЅР°Р»СЊРЅС‹С… СЂРµР¶РёРјРѕРІ
            src_ratio = float(original_W) / float(original_H) if original_H != 0 else float('inf')
            target_canvas_ratio = float(final_w) / float(final_h) if final_h != 0 else float('inf')

            # Р’С‹С‡РёСЃР»СЏРµРј РїСЂРѕРјРµР¶СѓС‚РѕС‡РЅС‹Р№ СЂР°Р·РјРµСЂ С‚Р°Рє, С‡С‚РѕР±С‹ РёР·РѕР±СЂР°Р¶РµРЅРёРµ РџРћРљР Р«Р’РђР›Рћ С†РµР»РµРІРѕР№ С…РѕР»СЃС‚
            if src_ratio > target_canvas_ratio:
                scale_to_h = final_h
                scale_to_w = round(scale_to_h * src_ratio)
            else:
                scale_to_w = final_w
                scale_to_h = round(scale_to_w / src_ratio) if src_ratio != 0 else final_h
            
            scale_to_w, scale_to_h = max(1, int(scale_to_w)), max(1, int(scale_to_h))

            # РњР°СЃС€С‚Р°Р±РёСЂСѓРµРј РґРѕ РїСЂРѕРјРµР¶СѓС‚РѕС‡РЅРѕРіРѕ СЂР°Р·РјРµСЂР°
            scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)
            if current_mask_nchw is not None:
                current_mask_nchw = self._interp_image(current_mask_nchw, (scale_to_w, scale_to_h), "nearest-exact")
            
            # Р’С‹С‡РёСЃР»СЏРµРј РєРѕРѕСЂРґРёРЅР°С‚С‹ РґР»СЏ С†РµРЅС‚СЂР°Р»СЊРЅРѕР№ РѕР±СЂРµР·РєРё
            crop_x = (scaled_pixels_nchw.shape[3] - final_w) // 2
            crop_y = (scaled_pixels_nchw.shape[2] - final_h) // 2
            crop_x, crop_y = max(0, crop_x), max(0, crop_y)
            
            final_output_pixels_nchw = scaled_pixels_nchw[:, :, crop_y : crop_y + final_h, crop_x : crop_x + final_w]
            if current_mask_nchw is not None:
                current_mask_nchw = current_mask_nchw[:, :, crop_y : crop_y + final_h, crop_x : crop_x + final_w]

        final_output_mask = current_mask_nchw.squeeze(1) if current_mask_nchw is not None else mask
        final_output_pixels = final_output_pixels_nchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        
        return (final_output_pixels, final_output_pixels.shape[2], final_output_pixels.shape[1], final_output_mask)


# ... (РѕСЃС‚Р°Р»СЊРЅРѕР№ РєРѕРґ Р±РµР· РёР·РјРµРЅРµРЅРёР№) ...

# рџ”§ РЎРїРёСЃРѕРє РїРѕРґРґРµСЂР¶РёРІР°РµРјС‹С… СЂР°Р·СЂРµС€РµРЅРёР№ Qwen Image
QWEN_IMAGE_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, 1.0),   # 1:1 (Р±Р»РёР¶Р°Р№С€РµРµ Рє 1328x1328)
    (1792, 1008, 1.778), # 16:9 (Р±Р»РёР¶Р°Р№С€РµРµ Рє 1664x928)
    (1008, 1792, 0.562), # 9:16
    (1456, 1088, 1.338), # 4:3 (Р±Р»РёР¶Р°Р№С€РµРµ Рє 1472x1140)
    (1088, 1456, 0.747), # 3:4
    (1568, 1056, 1.484), # 3:2 (Р±Р»РёР¶Р°Р№С€РµРµ Рє 1584x1056)
    (1056, 1568, 0.673), # 2:3
]

def closest_supported_resolution(width, height):
    aspect = width / height
    best_res = None
    best_diff = 999
    for w, h, a in QWEN_IMAGE_SUPPORTED_RESOLUTIONS:
        diff = abs(aspect - a)
        if diff < best_diff:
            best_diff = diff
            best_res = (w, h)
    return best_res

class TS_QwenSafeResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "safe_resize"
    CATEGORY = "image/resize"

    def safe_resize(self, image):
        # image: torch.Tensor, shape (B,H,W,C), dtype float32, range 0..1
        b, h, w, c = image.shape
        assert c in [3, 4], f"Expected 3 or 4 channels, got {c}"

        output_images = []

        for i in range(b):
            # в†’ NumPy (0..255)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # ---- Р’С‹Р±РёСЂР°РµРј Р±Р»РёР¶Р°Р№С€РµРµ СЂР°Р·СЂРµС€РµРЅРёРµ ----
            target_w, target_h = closest_supported_resolution(w, h)

            # ---- РњР°СЃС€С‚Р°Р±РёСЂСѓРµРј ----
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

            # ---- РљСЂРѕРї РїРѕ С†РµРЅС‚СЂСѓ ----
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = resized.crop((left, top, right, bottom))

            # в†’ РѕР±СЂР°С‚РЅРѕ РІ С‚РµРЅР·РѕСЂ float32 (0..1)
            img_out = torch.from_numpy(np.array(cropped)).float() / 255.0
            # (H,W,C) в†’ (1,H,W,C)
            output_images.append(img_out.unsqueeze(0))

        # РЎРѕР±РёСЂР°РµРј Р±Р°С‚С‡
        output = torch.cat(output_images, dim=0)
        return (output,)
    


# Р’СЃРµ СЂР°Р·СЂРµС€РµРЅРёСЏ Qwen: РіРѕСЂРёР·РѕРЅС‚Р°Р»СЊРЅС‹Рµ Рё РІРµСЂС‚РёРєР°Р»СЊРЅС‹Рµ
QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, "1:1"),
    (1792, 1008, "16:9"),
    (1008, 1792, "16:9 Vertical"),
    (1456, 1088, "4:3"),
    (1088, 1456, "4:3 Vertical"),
    (1568, 1056, "3:2"),
    (1056, 1568, "3:2 Vertical"),
]

# РЎРїРёСЃРѕРє РґР»СЏ dropdown РІ UI
ASPECT_OPTIONS = [name for (_, _, name) in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS]

class TS_QwenCanvas:
    """
    TS Qwen Canvas
    Р’РїРёСЃС‹РІР°РµС‚ РёР·РѕР±СЂР°Р¶РµРЅРёРµ РІ РєР°РЅРІР°СЃ РїРѕРґ Р±РµР·РѕРїР°СЃРЅРѕРµ СЂР°Р·СЂРµС€РµРЅРёРµ Qwen.
    РџРѕРґРґРµСЂР¶РёРІР°РµС‚ РјР°СЃРєСѓ ComfyUI.
    Р•СЃР»Рё РёР·РѕР±СЂР°Р¶РµРЅРёРµ РЅРµ РїРѕРґР°РЅРѕ, СЃРѕР·РґР°РµС‚ РїСѓСЃС‚РѕР№ Р±РµР»С‹Р№ РєР°РЅРІР°СЃ РІС‹Р±СЂР°РЅРЅРѕРіРѕ СЂР°Р·СЂРµС€РµРЅРёСЏ.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # image РїРµСЂРµРјРµС‰РµРЅ РІ optional
                "resolution": (ASPECT_OPTIONS, {"default": "1:1"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("canvas_image", "width", "height")
    FUNCTION = "make_canvas"
    CATEGORY = "TS Qwen"

    def make_canvas(self, resolution="1:1", image=None, mask=None):
        # РџРѕР»СѓС‡Р°РµРј target СЂР°Р·РјРµСЂС‹ РїРѕ РІС‹Р±СЂР°РЅРЅРѕРјСѓ РёРјРµРЅРё
        target_w, target_h = None, None
        for w, h, name in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS:
            if name == resolution:
                target_w, target_h = w, h
                break
        if target_w is None:
            raise ValueError(f"Resolution {resolution} not found")

        # РЎРѕР·РґР°С‘Рј Р±РµР»С‹Р№ РєР°РЅРІР°СЃ (Р±Р°Р·Р°)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        # Р•СЃР»Рё РёР·РѕР±СЂР°Р¶РµРЅРёРµ РµСЃС‚СЊ, РѕР±СЂР°Р±Р°С‚С‹РІР°РµРј Рё РЅР°РєР»Р°РґС‹РІР°РµРј РµРіРѕ
        if image is not None:
            # РљРѕРЅРІРµСЂС‚Р°С†РёСЏ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ РІ PIL
            img_tensor = image[0]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            # Р•СЃР»Рё РјР°СЃРєР° РµСЃС‚СЊ Рё РЅРµ РїСѓСЃС‚Р°СЏ, РѕР±СЂРµР·Р°РµРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ РїРѕ bounding box
            if mask is not None:
                mask_tensor = mask[0]
                mask_np = mask_tensor.detach().cpu().numpy()
                if mask_np.ndim == 4:  # (B,1,H,W)
                    mask_np = mask_np[0,0]
                elif mask_np.ndim == 3:  # (H,W,1) РёР»Рё (1,H,W)
                    mask_np = mask_np.squeeze(0).squeeze(-1)
                elif mask_np.ndim != 2:
                    raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

                # РџСЂРѕРІРµСЂРєР° РЅР° РїСѓСЃС‚РѕС‚Сѓ/РѕРґРЅРѕС‚РѕРЅРЅРѕСЃС‚СЊ
                if mask_np.max() > 0 and mask_np.min() < 1:
                    # РќР°С…РѕРґРёРј bounding box
                    ys, xs = np.where(mask_np > 0)
                    if ys.size > 0 and xs.size > 0:
                        top, left = ys.min(), xs.min()
                        bottom, right = ys.max(), xs.max()
                        img = img.crop((left, top, right + 1, bottom + 1))

            # РњР°СЃС€С‚Р°Р±РёСЂСѓРµРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ СЃ СЃРѕС…СЂР°РЅРµРЅРёРµРј РїСЂРѕРїРѕСЂС†РёР№
            img_w, img_h = img.size
            scale_w = target_w / img_w
            scale_h = target_h / img_h

            # РћРїСЂРµРґРµР»СЏРµРј РѕСЂРёРµРЅС‚Р°С†РёСЋ (С…РѕС‚СЏ Р»РѕРіРёРєР° scale РЅРёР¶Рµ РѕРґРёРЅР°РєРѕРІР° РґР»СЏ РѕР±РѕРёС… РІРµС‚РѕРє)
            # scale РІСЃРµРіРґР° Р±РµСЂРµС‚СЃСЏ РїРѕ РјРёРЅРёРјСѓРјСѓ, С‡С‚РѕР±С‹ РІРїРёСЃР°С‚СЊ С†РµР»РёРєРѕРј
            scale = min(scale_w, scale_h)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Р¦РµРЅС‚СЂРёСЂРѕРІР°РЅРёРµ
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(img, (paste_x, paste_y))

        # РљРѕРЅРІРµСЂС‚Р°С†РёСЏ РІ С„РѕСЂРјР°С‚ ComfyUI (B,H,W,C)
        # Р•СЃР»Рё image Р±С‹Р» None, РІРµСЂРЅРµС‚СЃСЏ РїСЂРѕСЃС‚Рѕ Р±РµР»С‹Р№ РїСЂСЏРјРѕСѓРіРѕР»СЊРЅРёРє
        out_img = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)

        return (out_img, target_w, target_h)

class TSAutoTileSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Р’С‹РїР°РґР°СЋС‰РёР№ СЃРїРёСЃРѕРє РґР»СЏ РІС‹Р±РѕСЂР° РѕР±С‰РµРіРѕ РєРѕР»РёС‡РµСЃС‚РІР° С‚Р°Р№Р»РѕРІ
                "tile_count": ([4, 8, 16],),
                "padding": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "divide_by": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("tile_width", "tile_height")

    FUNCTION = "calculate_grid"

    CATEGORY = "utils/Tile Size"

    def find_best_grid(self, total_tiles, image_aspect_ratio):
        """
        РќР°С…РѕРґРёС‚ РЅР°РёР»СѓС‡С€СѓСЋ РїР°СЂСѓ РјРЅРѕР¶РёС‚РµР»РµР№ (СЃРµС‚РєСѓ),
        СЃРѕРѕС‚РЅРѕС€РµРЅРёРµ СЃС‚РѕСЂРѕРЅ РєРѕС‚РѕСЂРѕР№ РЅР°РёР±РѕР»РµРµ Р±Р»РёР·РєРѕ Рє СЃРѕРѕС‚РЅРѕС€РµРЅРёСЋ СЃС‚РѕСЂРѕРЅ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ.
        """
        if total_tiles <= 0:
            return 1, 1

        factors = []
        # РќР°С…РѕРґРёРј РІСЃРµ РїР°СЂС‹ РјРЅРѕР¶РёС‚РµР»РµР№ РґР»СЏ С‡РёСЃР»Р° total_tiles
        for i in range(1, int(math.sqrt(total_tiles)) + 1):
            if total_tiles % i == 0:
                factors.append((total_tiles // i, i))
                if i * i != total_tiles:
                    factors.append((i, total_tiles // i))

        best_pair = (1, total_tiles) # Р—РЅР°С‡РµРЅРёРµ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ
        min_diff = float('inf')

        # РС‰РµРј РїР°СЂСѓ СЃ РЅР°РёРјРµРЅСЊС€РµР№ СЂР°Р·РЅРёС†РµР№ РІ СЃРѕРѕС‚РЅРѕС€РµРЅРёРё СЃС‚РѕСЂРѕРЅ
        for x, y in factors:
            grid_aspect_ratio = x / y
            diff = abs(grid_aspect_ratio - image_aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_pair = (x, y)
        
        return best_pair

    def calculate_grid(self, tile_count, padding, divide_by, image=None, width=512, height=512):
        # РЁР°Рі 1: РћРїСЂРµРґРµР»СЏРµРј РёСЃС…РѕРґРЅС‹Рµ СЂР°Р·РјРµСЂС‹ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ
        if image is not None:
            _, img_height, img_width, _ = image.shape
        else:
            img_width, img_height = width, height

        # РЁР°Рі 2: РќР°С…РѕРґРёРј РЅР°РёР»СѓС‡С€СѓСЋ СЃРµС‚РєСѓ (tiles_x, tiles_y)
        # Р’С‹С‡РёСЃР»СЏРµРј СЃРѕРѕС‚РЅРѕС€РµРЅРёРµ СЃС‚РѕСЂРѕРЅ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ, РёР·Р±РµРіР°СЏ РґРµР»РµРЅРёСЏ РЅР° РЅРѕР»СЊ.
        image_aspect_ratio = img_width / img_height if img_height != 0 else 1.0
        
        # РСЃРїРѕР»СЊР·СѓРµРј РІСЃРїРѕРјРѕРіР°С‚РµР»СЊРЅСѓСЋ С„СѓРЅРєС†РёСЋ РґР»СЏ РїРѕРґР±РѕСЂР° РѕРїС‚РёРјР°Р»СЊРЅРѕР№ СЃРµС‚РєРё
        tiles_x, tiles_y = self.find_best_grid(tile_count, image_aspect_ratio)

        # РЁР°Рі 3: Р Р°СЃСЃС‡РёС‚С‹РІР°РµРј СЂР°Р·РјРµСЂ РѕРґРЅРѕРіРѕ С‚Р°Р№Р»Р° СЃ СѓС‡РµС‚РѕРј РїРµСЂРµРєСЂС‹С‚РёСЏ
        tile_w = (img_width + (tiles_x - 1) * padding) / tiles_x
        tile_h = (img_height + (tiles_y - 1) * padding) / tiles_y

        # РЁР°Рі 4: РћРєСЂСѓРіР»СЏРµРј РґРѕ Р±Р»РёР¶Р°Р№С€РµРіРѕ С‡РёСЃР»Р°, РєСЂР°С‚РЅРѕРіРѕ `divide_by`
        tile_width = round(tile_w / divide_by) * divide_by
        tile_height = round(tile_h / divide_by) * divide_by

        return (tile_width, tile_height)


class TS_WAN_SafeResize:
    WAN_RESOLUTIONS = {
        "high quality": {
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "1:1": (720, 720),
        },
        "standard quality": {
            "16:9": (832, 480),
            "9:16": (480, 832),
            "1:1": (480, 480),
        },
        "low quality": {
            "16:9": (426, 240),
            "9:16": (240, 426),
            "1:1": (240, 240),
        },
    }

    QUALITY_MAP = {
        "Fast quality": "low quality",
        "Standard quality": "standard quality",
        "High quality": "high quality",
    }

    @staticmethod
    def detect_aspect_ratio(width, height):
        aspect = width / height
        if aspect > 1.3:
            return "16:9"
        elif aspect < 0.8:
            return "9:16"
        else:
            return "1:1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "quality": (
                    ["Fast quality", "Standard quality", "High quality"],
                    {"default": "Standard quality"},
                ),
            },
            "optional": {
                "interconnection_in": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "interconnection_out")
    FUNCTION = "safe_resize"
    CATEGORY = "image/resize"

    def safe_resize(self, image, quality, interconnection_in=None):
        # РџСЂРёРѕСЂРёС‚РµС‚ interconnection
        if interconnection_in in self.WAN_RESOLUTIONS:
            internal_quality = interconnection_in
        else:
            internal_quality = self.QUALITY_MAP[quality]

        b, h, w, c = image.shape
        assert c in [3, 4], f"Expected 3 or 4 channels, got {c}"

        aspect_key = self.detect_aspect_ratio(w, h)
        target_w, target_h = self.WAN_RESOLUTIONS[internal_quality][aspect_key]

        output_images = []

        for i in range(b):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            cropped = resized.crop((left, top, left + target_w, top + target_h))

            img_out = torch.from_numpy(np.array(cropped)).float() / 255.0
            output_images.append(img_out.unsqueeze(0))

        output = torch.cat(output_images, dim=0)

        return (output, target_w, target_h, internal_quality)


NODE_CLASS_MAPPINGS = {
    "TS_ImageResize": TS_ImageResize,
    "TS_QwenSafeResize": TS_QwenSafeResize,
    "TS_QwenCanvas": TS_QwenCanvas,
    "TSAutoTileSize": TSAutoTileSize,
    "TS_WAN_SafeResize": TS_WAN_SafeResize
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageResize": "TS Image Resize",
    "TS_QwenSafeResize": "TS Qwen Safe Resize",
    "TS_QwenCanvas": "TS Qwen Canvas",
    "TSAutoTileSize": "TS Auto Tile Size",
    "TS_WAN_SafeResize": "TS WAN Safe Resize"
}
