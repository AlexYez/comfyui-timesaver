import torch
import math

try:
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class TS_ImageResize:
    def __init__(self):
        pass

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "bicubic", "area", "lanczos"]

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}), # Изменен step
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}), # Изменен step
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}), # Изменен step
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),  # Изменен step
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "keep_proportion": ("BOOLEAN", {"default": True}),
                "upscale_method": (s.UPSCALE_METHODS, {"default": "bicubic"}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, target_width, target_height, smaller_side, larger_side, scale_factor, upscale_method, divisible_by, **_):
        if upscale_method == "lanczos" and not TORCHVISION_AVAILABLE:
            return "Lanczos upscale_method requires torchvision to be installed."
        
        if divisible_by is not None and divisible_by < 1: # Добавлена проверка divisible_by на None
            return "divisible_by must be 1 or greater."

        sf_active = scale_factor is not None and scale_factor > 0.0
        tw_active = target_width is not None and target_width > 0
        th_active = target_height is not None and target_height > 0
        ss_active = smaller_side is not None and smaller_side > 0
        ls_active = larger_side is not None and larger_side > 0
        
        active_method_count = 0
        if sf_active: active_method_count +=1
        if tw_active or th_active: active_method_count +=1
        if ss_active or ls_active: active_method_count +=1

        if active_method_count > 1:
            return "More than one sizing method (scale_factor, target_width/height, smaller/larger_side) is active. Please use only one."
            
        return True

    def _interp_image(self, image_tensor_nchw, size_wh, method):
        target_h, target_w = size_wh[1], size_wh[0]
        if method == "lanczos":
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision is required for Lanczos resampling.")
            return TF.resize(image_tensor_nchw, [target_h, target_w], interpolation=InterpolationMode.LANCZOS, antialias=True)
        else:
            interp_kwargs = {"mode": method}
            if method in ["bilinear", "bicubic"]:
                interp_kwargs["antialias"] = True
            return torch.nn.functional.interpolate(image_tensor_nchw, size=(target_h, target_w), **interp_kwargs)

    def resize(self, pixels, target_width, target_height, smaller_side, larger_side, scale_factor, keep_proportion, upscale_method, divisible_by):
        _B, original_H, original_W, _C = pixels.shape
        device = pixels.device
        
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

        if chosen_method == "scale_factor":
            ideal_w = original_W * _scale_factor
            ideal_h = original_H * _scale_factor
        elif chosen_method == "target_dims":
            if keep_proportion:
                if _target_width > 0 and _target_height > 0:
                    ratio_orig = ideal_w / ideal_h if ideal_h != 0 else float('inf') # Защита от деления на 0
                    ratio_target = float(_target_width) / float(_target_height) if _target_height != 0 else float('inf')
                    if ideal_h == 0 and _target_height == 0: # Обе высоты 0, берем ширину
                         ideal_w = float(_target_width)
                         ideal_h = 0 # или original_H если _target_width был единственным
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
        
        ideal_w = max(1.0, ideal_w)
        ideal_h = max(1.0, ideal_h)

        # --- Коррекция идеальных размеров до кратности divisible_by (округление ВНИЗ) ---
        # Это будут финальные размеры выходного изображения.
        if _divisible_by > 1:
            # Округляем каждую идеальную размерность вниз до ближайшего кратного _divisible_by
            # Это обеспечивает, что мы не создаем пиксели "из ничего" для достижения кратности,
            # а работаем в пределах контента, определенного ideal_w/h.
            final_target_w_div = math.floor(ideal_w / _divisible_by) * _divisible_by
            final_target_h_div = math.floor(ideal_h / _divisible_by) * _divisible_by

            # Если округление дало 0 (например, ideal_w < _divisible_by), ставим _divisible_by (минимально возможное кратное)
            final_target_w_div = max(_divisible_by, final_target_w_div)
            final_target_h_div = max(_divisible_by, final_target_h_div)
        else:
            final_target_w_div = round(ideal_w) # Обычное округление, если кратность не нужна
            final_target_h_div = round(ideal_h)
        
        final_target_w_div = max(1, int(final_target_w_div))
        final_target_h_div = max(1, int(final_target_h_div))

        current_pixels_nchw = pixels.permute(0, 3, 1, 2)

        if keep_proportion or chosen_method == "side_dims" or chosen_method == "scale_factor":
            src_ratio = float(original_W) / float(original_H) if original_H != 0 else float('inf')
            target_canvas_ratio = float(final_target_w_div) / float(final_target_h_div) if final_target_h_div != 0 else float('inf')

            scale_to_w = float(final_target_w_div)
            scale_to_h = float(final_target_h_div)

            if src_ratio > target_canvas_ratio:
                scale_to_h = final_target_h_div # Масштабируем по высоте холста
                scale_to_w = round(scale_to_h * src_ratio) # Ширина будет больше или равна целевой
                if scale_to_w < final_target_w_div : scale_to_w = final_target_w_div # Гарантируем покрытие
            elif src_ratio < target_canvas_ratio:
                scale_to_w = final_target_w_div # Масштабируем по ширине холста
                scale_to_h = round(scale_to_w / src_ratio) if src_ratio != 0 else final_target_h_div # Высота будет больше или равна целевой
                if scale_to_h < final_target_h_div : scale_to_h = final_target_h_div # Гарантируем покрытие
            # Если пропорции совпали, scale_to_w/h уже равны final_target_w/h_div

            scale_to_w = max(1, int(round(scale_to_w)))
            scale_to_h = max(1, int(round(scale_to_h)))
            
            scaled_pixels_nchw = current_pixels_nchw
            if scale_to_w != original_W or scale_to_h != original_H:
                 scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)

            crop_x_start = (scaled_pixels_nchw.shape[3] - final_target_w_div) // 2
            crop_y_start = (scaled_pixels_nchw.shape[2] - final_target_h_div) // 2
            
            crop_x_start = max(0, crop_x_start)
            crop_y_start = max(0, crop_y_start)
            
            final_output_pixels_nchw = scaled_pixels_nchw[:, :, crop_y_start : crop_y_start + final_target_h_div, crop_x_start : crop_x_start + final_target_w_div]
        else: 
            if final_target_w_div != original_W or final_target_h_div != original_H:
                final_output_pixels_nchw = self._interp_image(current_pixels_nchw, (final_target_w_div, final_target_h_div), upscale_method)
            else:
                final_output_pixels_nchw = current_pixels_nchw
        
        final_output_pixels = final_output_pixels_nchw.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        
        output_height = final_output_pixels.shape[1]
        output_width = final_output_pixels.shape[2]
            
        return (final_output_pixels, output_width, output_height)

NODE_CLASS_MAPPINGS = {
    "TS_ImageResize": TS_ImageResize
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageResize": "TS Image Resize"
}