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

    # В PIL.Image.LANCZOS переименовали в Resampling.LANCZOS в Pillow 10.0.0
    # Проверяем наличие нового имени и используем его, если возможно, для совместимости.
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
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "keep_proportion": ("BOOLEAN", {"default": True}),
                "upscale_method": (s.UPSCALE_METHODS, {"default": "bicubic"}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, target_width, target_height, smaller_side, larger_side, scale_factor, upscale_method, **_):
        # Упрощенная валидация, основная логика в самой функции
        return True

    def _pil_resize(self, image_tensor_nchw, size_wh):
        target_w, target_h = size_wh
        pil_images = []
        # Проверяем, является ли тензор маской (один канал) или изображением
        is_mask = image_tensor_nchw.shape[1] == 1
        
        for i in range(image_tensor_nchw.shape[0]):
            img_tensor = image_tensor_nchw[i]
            if is_mask:
                # Для масок убираем измерение канала для PIL
                img_tensor_chw = img_tensor.squeeze(0)
                img_np = img_tensor_chw.cpu().numpy() * 255.0
                pil_image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), 'L')
            else:
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
                pil_image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            
            # Изменяем размер
            resample_method = Image.Resampling.NEAREST if is_mask else self.LANCZOS
            resized_pil = pil_image.resize((target_w, target_h), resample_method)
            pil_images.append(resized_pil)
            
        # Конвертируем обратно в тензор
        output_tensors = []
        for pil_img in pil_images:
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            if is_mask:
                # Добавляем обратно измерение канала для масок
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

    def resize(self, pixels, target_width, target_height, smaller_side, larger_side, scale_factor, keep_proportion, upscale_method, divisible_by, mask=None):
        _B, original_H, original_W, _C = pixels.shape
        
        ideal_w, ideal_h = float(original_W), float(original_H)

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
        
        # *** НАЧАЛО ИЗМЕНЕНИЙ ***
        # Особый случай: жестко заданы оба размера и сохранение пропорций включено.
        # Это активирует режим "масштабировать и обрезать до заполнения" (cover and crop).
        is_cover_crop_mode = (chosen_method == "target_dims" and 
                              keep_proportion and 
                              _target_width > 0 and 
                              _target_height > 0)
        
        if not is_cover_crop_mode:
            # Старая логика для всех остальных случаев
            if chosen_method == "scale_factor":
                ideal_w = original_W * _scale_factor
                ideal_h = original_H * _scale_factor
            elif chosen_method == "target_dims":
                if keep_proportion:
                    if _target_width > 0 and _target_height > 0:
                        ratio_orig = ideal_w / ideal_h if ideal_h != 0 else float('inf')
                        ratio_target = float(_target_width) / float(_target_height) if _target_height != 0 else float('inf')
                        if ideal_h == 0 and _target_height == 0:
                             ideal_w = float(_target_width)
                             ideal_h = 0
                        elif ratio_orig > ratio_target: 
                            ideal_w = float(_target_width)
                            ideal_h = ideal_w / ratio_orig if ratio_orig != 0 else 0
                        else: 
                            ideal_h = float(_target_height)
                            ideal_w = ideal_h * ratio_orig
                    elif _target_width > 0:
                        ideal_h = ideal_h * (float(_target_width) / ideal_w) if ideal_w != 0 else 0
                        ideal_w = float(_target_width)
                    elif _target_height > 0:
                        ideal_w = ideal_w * (float(_target_height) / ideal_h) if ideal_h != 0 else 0
                        ideal_h = float(_target_height)
                else: 
                    if _target_width > 0: ideal_w = float(_target_width)
                    if _target_height > 0: ideal_h = float(_target_height)
            elif chosen_method == "side_dims":
                if _smaller_side > 0:
                    if ideal_w < ideal_h: 
                        ideal_h = ideal_h * (float(_smaller_side) / ideal_w) if ideal_w != 0 else 0
                        ideal_w = float(_smaller_side)
                    else: 
                        ideal_w = ideal_w * (float(_smaller_side) / ideal_h) if ideal_h != 0 else 0
                        ideal_h = float(_smaller_side)
                elif _larger_side > 0:
                    if ideal_w > ideal_h: 
                        ideal_h = ideal_h * (float(_larger_side) / ideal_w) if ideal_w != 0 else 0
                        ideal_w = float(_larger_side)
                    else: 
                        ideal_w = ideal_w * (float(_larger_side) / ideal_h) if ideal_h != 0 else 0
                        ideal_h = float(_larger_side)
        else:
            # Если активен режим cover_crop, то целевые размеры становятся идеальными
            ideal_w = float(_target_width)
            ideal_h = float(_target_height)
        
        ideal_w = max(1.0, ideal_w)
        ideal_h = max(1.0, ideal_h)

        if _divisible_by > 1:
            final_target_w_div = math.floor(ideal_w / _divisible_by) * _divisible_by
            final_target_h_div = math.floor(ideal_h / _divisible_by) * _divisible_by
            final_target_w_div = max(_divisible_by, final_target_w_div)
            final_target_h_div = max(_divisible_by, final_target_h_div)
        else:
            final_target_w_div = round(ideal_w)
            final_target_h_div = round(ideal_h)
        
        final_target_w_div = max(1, int(final_target_w_div))
        final_target_h_div = max(1, int(final_target_h_div))
        
        current_pixels_nchw = pixels.permute(0, 3, 1, 2)
        
        final_output_mask = mask
        current_mask_nchw = None
        if mask is not None:
            current_mask_nchw = mask.unsqueeze(1)

        # Выбираем ветку обработки
        if is_cover_crop_mode:
            # Новая логика "cover and crop"
            src_ratio = float(original_W) / float(original_H) if original_H != 0 else float('inf')
            target_canvas_ratio = float(final_target_w_div) / float(final_target_h_div) if final_target_h_div != 0 else float('inf')

            # Вычисляем промежуточный размер так, чтобы изображение ПОКРЫВАЛО целевую область
            if src_ratio > target_canvas_ratio:
                # Исходник шире цели -> масштабируем по высоте цели
                scale_to_h = final_target_h_div
                scale_to_w = round(scale_to_h * src_ratio)
            else:
                # Исходник выше цели (или такое же соотношение) -> масштабируем по ширине цели
                scale_to_w = final_target_w_div
                scale_to_h = round(scale_to_w / src_ratio) if src_ratio != 0 else final_target_h_div
            
            scale_to_w = max(1, int(scale_to_w))
            scale_to_h = max(1, int(scale_to_h))

            # Масштабируем до промежуточного размера
            scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)

            if mask is not None:
                scaled_mask_nchw = self._interp_image(current_mask_nchw, (scale_to_w, scale_to_h), "nearest-exact")
            
            # Вычисляем координаты для центральной обрезки
            crop_x_start = (scaled_pixels_nchw.shape[3] - final_target_w_div) // 2
            crop_y_start = (scaled_pixels_nchw.shape[2] - final_target_h_div) // 2
            
            crop_x_start = max(0, crop_x_start)
            crop_y_start = max(0, crop_y_start)
            
            final_output_pixels_nchw = scaled_pixels_nchw[:, :, crop_y_start : crop_y_start + final_target_h_div, crop_x_start : crop_x_start + final_target_w_div]
            
            if mask is not None:
                final_output_mask_nchw = scaled_mask_nchw[:, :, crop_y_start : crop_y_start + final_target_h_div, crop_x_start : crop_x_start + final_target_w_div]
                final_output_mask = final_output_mask_nchw.squeeze(1)

        elif keep_proportion or chosen_method == "side_dims" or chosen_method == "scale_factor":
            # Старая логика "вписывания" (contain) для остальных режимов
            src_ratio = float(original_W) / float(original_H) if original_H != 0 else float('inf')
            target_canvas_ratio = float(final_target_w_div) / float(final_target_h_div) if final_target_h_div != 0 else float('inf')

            scale_to_w = float(final_target_w_div)
            scale_to_h = float(final_target_h_div)

            if src_ratio > target_canvas_ratio:
                scale_to_h = final_target_h_div
                scale_to_w = round(scale_to_h * src_ratio)
                if scale_to_w < final_target_w_div : scale_to_w = final_target_w_div
            elif src_ratio < target_canvas_ratio:
                scale_to_w = final_target_w_div
                scale_to_h = round(scale_to_w / src_ratio) if src_ratio != 0 else final_target_h_div
                if scale_to_h < final_target_h_div : scale_to_h = final_target_h_div

            scale_to_w = max(1, int(round(scale_to_w)))
            scale_to_h = max(1, int(round(scale_to_h)))
            
            scaled_pixels_nchw = current_pixels_nchw
            if scale_to_w != original_W or scale_to_h != original_H:
                 scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)

            if mask is not None:
                scaled_mask_nchw = current_mask_nchw
                if scale_to_w != original_W or scale_to_h != original_H:
                    scaled_mask_nchw = self._interp_image(current_mask_nchw, (scale_to_w, scale_to_h), "nearest-exact")
            else:
                scaled_mask_nchw = None

            crop_x_start = (scaled_pixels_nchw.shape[3] - final_target_w_div) // 2
            crop_y_start = (scaled_pixels_nchw.shape[2] - final_target_h_div) // 2
            
            crop_x_start = max(0, crop_x_start)
            crop_y_start = max(0, crop_y_start)
            
            final_output_pixels_nchw = scaled_pixels_nchw[:, :, crop_y_start : crop_y_start + final_target_h_div, crop_x_start : crop_x_start + final_target_w_div]
            
            if mask is not None:
                final_output_mask_nchw = scaled_mask_nchw[:, :, crop_y_start : crop_y_start + final_target_h_div, crop_x_start : crop_x_start + final_target_w_div]
                final_output_mask = final_output_mask_nchw.squeeze(1)
        else: 
            # Логика без сохранения пропорций
            if final_target_w_div != original_W or final_target_h_div != original_H:
                final_output_pixels_nchw = self._interp_image(current_pixels_nchw, (final_target_w_div, final_target_h_div), upscale_method)
                if mask is not None:
                    final_output_mask_nchw = self._interp_image(current_mask_nchw, (final_target_w_div, final_target_h_div), "nearest-exact")
                    final_output_mask = final_output_mask_nchw.squeeze(1)
            else:
                final_output_pixels_nchw = current_pixels_nchw
        
        # *** КОНЕЦ ИЗМЕНЕНИЙ ***
        
        final_output_pixels = final_output_pixels_nchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        
        output_height = final_output_pixels.shape[1]
        output_width = final_output_pixels.shape[2]
            
        return (final_output_pixels, output_width, output_height, final_output_mask)
    

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