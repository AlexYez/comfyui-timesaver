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

        # --- 1. Расчет 'идеальных' размеров на основе выбранного метода ---
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
                 # Для 'cover and crop' и одного заданного размера,
                 # идеальные размеры - это просто целевые.
                 # Для одного размера - пропорционально масштабируем.
                ratio = ideal_w / ideal_h if ideal_h != 0 else float('inf')
                if _target_width > 0 and _target_height > 0:
                    ideal_w, ideal_h = float(_target_width), float(_target_height)
                elif _target_width > 0:
                    ideal_w = float(_target_width)
                    ideal_h = ideal_w / ratio if ratio != 0 else 0
                elif _target_height > 0:
                    ideal_h = float(_target_height)
                    ideal_w = ideal_h * ratio
            else: # Без сохранения пропорций
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

        # --- 2. Применение 'dont_enlarge' ---
        if dont_enlarge and chosen_method is not None:
            if ideal_w * ideal_h > original_W * original_H:
                ideal_w, ideal_h = float(original_W), float(original_H)
        
        # --- 3. Определение финальных размеров холста ---
        is_cover_crop_mode = (chosen_method == "target_dims" and keep_proportion and _target_width > 0 and _target_height > 0)
        is_free_distort_mode = (chosen_method == "target_dims" and not keep_proportion and _target_width > 0 and _target_height > 0)
        
        final_w, final_h = 0, 0
        if is_cover_crop_mode or is_free_distort_mode:
            # В этих двух режимах 'divisible_by' ИГНОРИРУЕТСЯ,
            # так как пользователь явно указал конечное разрешение.
            final_w = int(round(ideal_w))
            final_h = int(round(ideal_h))
        else:
            # Во всех остальных пропорциональных режимах строго соблюдаем кратность.
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
        
        # --- 4. Единая логика масштабирования и обрезки ---
        if final_w == original_W and final_h == original_H:
             final_output_pixels_nchw = current_pixels_nchw
        elif is_free_distort_mode:
            # Простое масштабирование с искажением
            final_output_pixels_nchw = self._interp_image(current_pixels_nchw, (final_w, final_h), upscale_method)
            if current_mask_nchw is not None:
                current_mask_nchw = self._interp_image(current_mask_nchw, (final_w, final_h), "nearest-exact")
        else:
            # Универсальная логика "покрыть и обрезать" для всех пропорциональных режимов
            src_ratio = float(original_W) / float(original_H) if original_H != 0 else float('inf')
            target_canvas_ratio = float(final_w) / float(final_h) if final_h != 0 else float('inf')

            # Вычисляем промежуточный размер так, чтобы изображение ПОКРЫВАЛО целевой холст
            if src_ratio > target_canvas_ratio:
                scale_to_h = final_h
                scale_to_w = round(scale_to_h * src_ratio)
            else:
                scale_to_w = final_w
                scale_to_h = round(scale_to_w / src_ratio) if src_ratio != 0 else final_h
            
            scale_to_w, scale_to_h = max(1, int(scale_to_w)), max(1, int(scale_to_h))

            # Масштабируем до промежуточного размера
            scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)
            if current_mask_nchw is not None:
                current_mask_nchw = self._interp_image(current_mask_nchw, (scale_to_w, scale_to_h), "nearest-exact")
            
            # Вычисляем координаты для центральной обрезки
            crop_x = (scaled_pixels_nchw.shape[3] - final_w) // 2
            crop_y = (scaled_pixels_nchw.shape[2] - final_h) // 2
            crop_x, crop_y = max(0, crop_x), max(0, crop_y)
            
            final_output_pixels_nchw = scaled_pixels_nchw[:, :, crop_y : crop_y + final_h, crop_x : crop_x + final_w]
            if current_mask_nchw is not None:
                current_mask_nchw = current_mask_nchw[:, :, crop_y : crop_y + final_h, crop_x : crop_x + final_w]

        final_output_mask = current_mask_nchw.squeeze(1) if current_mask_nchw is not None else mask
        final_output_pixels = final_output_pixels_nchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        
        return (final_output_pixels, final_output_pixels.shape[2], final_output_pixels.shape[1], final_output_mask)


# ... (остальной код без изменений) ...

# 🔧 Список поддерживаемых разрешений Qwen Image
QWEN_IMAGE_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, 1.0),   # 1:1 (ближайшее к 1328x1328)
    (1792, 1008, 1.778), # 16:9 (ближайшее к 1664x928)
    (1008, 1792, 0.562), # 9:16
    (1456, 1088, 1.338), # 4:3 (ближайшее к 1472x1140)
    (1088, 1456, 0.747), # 3:4
    (1568, 1056, 1.484), # 3:2 (ближайшее к 1584x1056)
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
            # → NumPy (0..255)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # ---- Выбираем ближайшее разрешение ----
            target_w, target_h = closest_supported_resolution(w, h)

            # ---- Масштабируем ----
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

            # ---- Кроп по центру ----
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = resized.crop((left, top, right, bottom))

            # → обратно в тензор float32 (0..1)
            img_out = torch.from_numpy(np.array(cropped)).float() / 255.0
            # (H,W,C) → (1,H,W,C)
            output_images.append(img_out.unsqueeze(0))

        # Собираем батч
        output = torch.cat(output_images, dim=0)
        return (output,)
    


# Все разрешения Qwen: горизонтальные и вертикальные
QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, "1:1"),
    (1792, 1008, "16:9"),
    (1008, 1792, "16:9 Vertical"),
    (1456, 1088, "4:3"),
    (1088, 1456, "4:3 Vertical"),
    (1568, 1056, "3:2"),
    (1056, 1568, "3:2 Vertical"),
]

# Список для dropdown в UI
ASPECT_OPTIONS = [name for (_, _, name) in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS]

class TS_QwenCanvas:
    """
    TS Qwen Canvas
    Вписывает изображение в канвас под безопасное разрешение Qwen.
    Поддерживает маску ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (ASPECT_OPTIONS, {"default": "1:1"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("canvas_image", "width", "height")
    FUNCTION = "make_canvas"
    CATEGORY = "TS Qwen"

    def make_canvas(self, image, resolution="1:1", mask=None):
        # Получаем target размеры по выбранному имени
        target_w, target_h = None, None
        for w, h, name in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS:
            if name == resolution:
                target_w, target_h = w, h
                break
        if target_w is None:
            raise ValueError(f"Resolution {resolution} not found")

        # Конвертация изображения в PIL
        img_tensor = image[0]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # Если маска есть и не пустая, обрезаем изображение по bounding box
        if mask is not None:
            mask_tensor = mask[0]
            mask_np = mask_tensor.detach().cpu().numpy()
            if mask_np.ndim == 4:  # (B,1,H,W)
                mask_np = mask_np[0,0]
            elif mask_np.ndim == 3:  # (H,W,1) или (1,H,W)
                mask_np = mask_np.squeeze(0).squeeze(-1)
            elif mask_np.ndim != 2:
                raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

            # Проверка на пустоту/однотонность
            if mask_np.max() > 0 and mask_np.min() < 1:
                # Находим bounding box
                ys, xs = np.where(mask_np > 0)
                if ys.size > 0 and xs.size > 0:
                    top, left = ys.min(), xs.min()
                    bottom, right = ys.max(), xs.max()
                    img = img.crop((left, top, right + 1, bottom + 1))

        # Масштабируем изображение с сохранением пропорций
        img_w, img_h = img.size
        scale_w = target_w / img_w
        scale_h = target_h / img_h

        # Определяем ориентацию
        img_is_h = img_w >= img_h
        canvas_is_h = target_w >= target_h
        if img_is_h == canvas_is_h:
            scale = min(scale_w, scale_h)
        else:
            scale = min(scale_w, scale_h)

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Создаём белый канвас
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        # Центрирование
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))

        # Конвертация в формат ComfyUI (B,H,W,C)
        out_img = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)

        return (out_img, target_w, target_h)


NODE_CLASS_MAPPINGS = {
    "TS_ImageResize": TS_ImageResize,
    "TS_QwenSafeResize": TS_QwenSafeResize,
    "TS_QwenCanvas": TS_QwenCanvas
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageResize": "TS Image Resize",
    "TS_QwenSafeResize": "TS Qwen Safe Resize",
    "TS_QwenCanvas": "TS Qwen Canvas"
}