"""TS Image Resize — multi-mode image resizer (exact / side / scale / megapixels).

node_id: TS_ImageResize
"""

import math

import torch
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


NODE_CLASS_MAPPINGS = {"TS_ImageResize": TS_ImageResize}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageResize": "TS Image Resize"}
