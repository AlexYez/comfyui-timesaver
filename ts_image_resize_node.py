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

    ACTION_TYPE_RESIZE_ONLY = "resize"
    ACTION_TYPE_CROP_RESIZE = "crop_to_fit"
    ACTION_TYPE_PAD_RESIZE = "pad_to_fit"

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
                "action": ([s.ACTION_TYPE_RESIZE_ONLY, s.ACTION_TYPE_CROP_RESIZE, s.ACTION_TYPE_PAD_RESIZE], {"default": s.ACTION_TYPE_RESIZE_ONLY}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "keep_proportion": ("BOOLEAN", {"default": True}),
                "upscale_method": (s.UPSCALE_METHODS, {"default": "bicubic"}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, action, target_width, target_height, smaller_side, larger_side, scale_factor, upscale_method, **_): # Убрал pixels, т.к. он не используется в валидации
        if upscale_method == "lanczos" and not TORCHVISION_AVAILABLE:
            return "Lanczos upscale_method requires torchvision to be installed."

        # Проверка на None перед сравнением
        sf_active = scale_factor is not None and scale_factor > 0.0
        tw_active = target_width is not None and target_width > 0
        th_active = target_height is not None and target_height
        ss_active = smaller_side is not None and smaller_side > 0
        ls_active = larger_side is not None and larger_side > 0

        num_sizing_methods = sum([
            sf_active,
            tw_active or th_active, # Если хотя бы один из target_width/height активен
            ss_active or ls_active  # Если хотя бы один из smaller_side/larger_side активен
        ])
        
        # Логика ниже может быть уточнена: если используется crop/pad, то должен быть хотя бы один метод изменения размера.
        # Если используется просто resize, то num_sizing_methods может быть 0 (тогда изображение не меняется, только divisible_by)
        if action == s.ACTION_TYPE_CROP_RESIZE or action == s.ACTION_TYPE_PAD_RESIZE:
            if num_sizing_methods == 0:
                 return f"For action '{action}', at least one sizing method (scale_factor, target_width/height, or smaller/larger_side) must be set to a non-zero value."
        
        # Дополнительная проверка: не должно быть более одного *активного* метода определения размера
        active_method_count = 0
        if sf_active: active_method_count +=1
        if tw_active or th_active: active_method_count +=1
        if ss_active or ls_active: active_method_count +=1
        
        if active_method_count > 1:
            return "More than one sizing method (scale_factor, target_width/height, smaller/larger_side) is active. Please use only one method."

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

    def resize(self, pixels, action, target_width, target_height, smaller_side, larger_side, scale_factor, keep_proportion, upscale_method, divisible_by):
        _B, original_H, original_W, _C = pixels.shape
        device = pixels.device
        current_pixels_content = pixels

        # --- Шаг 1: Определить операционные размеры (op_w, op_h) ---
        op_w, op_h = float(original_W), float(original_H)

        # Проверка на None и инициализация значениями по умолчанию, если None (для основной логики)
        # Для VALIDATE_INPUTS None означает "не задано", для resize - если дошло до сюда, значит надо работать с числом
        # Но хороший тон - проверять, хотя ComfyUI должен передавать default, если не подключено
        _scale_factor = scale_factor if scale_factor is not None else 0.0
        _target_width = target_width if target_width is not None else 0
        _target_height = target_height if target_height is not None else 0
        _smaller_side = smaller_side if smaller_side is not None else 0
        _larger_side = larger_side if larger_side is not None else 0


        active_methods = []
        if _scale_factor > 0.0: active_methods.append("scale_factor")
        if _target_width > 0 or _target_height > 0: active_methods.append("target_dims")
        if _smaller_side > 0 or _larger_side > 0: active_methods.append("side_dims")

        # Приоритет методов
        chosen_method = None
        if "scale_factor" in active_methods: chosen_method = "scale_factor"
        elif "target_dims" in active_methods: chosen_method = "target_dims"
        elif "side_dims" in active_methods: chosen_method = "side_dims"


        if chosen_method == "scale_factor":
            op_w = original_W * _scale_factor
            op_h = original_H * _scale_factor
        elif chosen_method == "target_dims":
            if keep_proportion:
                # Если заданы оба, вписываем с сохранением пропорций
                if _target_width > 0 and _target_height > 0:
                    ratio_orig = op_w / op_h
                    ratio_target = float(_target_width) / float(_target_height)
                    if ratio_orig > ratio_target: 
                        op_w = float(_target_width)
                        op_h = op_w / ratio_orig
                    else: 
                        op_h = float(_target_height)
                        op_w = op_h * ratio_orig
                elif _target_width > 0:
                    op_h = op_h * (float(_target_width) / op_w)
                    op_w = float(_target_width)
                elif _target_height > 0:
                    op_w = op_w * (float(_target_height) / op_h)
                    op_h = float(_target_height)
            else: # keep_proportion == False
                if _target_width > 0: op_w = float(_target_width)
                if _target_height > 0: op_h = float(_target_height)
        elif chosen_method == "side_dims":
            if _smaller_side > 0:
                if op_w < op_h: 
                    op_h = op_h * (float(_smaller_side) / op_w)
                    op_w = float(_smaller_side)
                else: 
                    op_w = op_w * (float(_smaller_side) / op_h)
                    op_h = float(_smaller_side)
            elif _larger_side > 0: # smaller_side имеет приоритет, если оба заданы
                if op_w > op_h: 
                    op_h = op_h * (float(_larger_side) / op_w)
                    op_w = float(_larger_side)
                else: 
                    op_w = op_w * (float(_larger_side) / op_h)
                    op_h = float(_larger_side)
        
        op_w = max(1, round(op_w))
        op_h = max(1, round(op_h))

        # --- Шаг 2: Применить `action` ---
        canvas_w_effective, canvas_h_effective = op_w, op_h
        
        if action == self.ACTION_TYPE_RESIZE_ONLY:
            if op_w != original_W or op_h != original_H:
                pixels_nchw = current_pixels_content.permute(0, 3, 1, 2)
                current_pixels_content = self._interp_image(pixels_nchw, (op_w, op_h), upscale_method).permute(0, 2, 3, 1).clamp(0.0, 1.0)
            final_canvas_pixels = current_pixels_content

        elif action == self.ACTION_TYPE_CROP_RESIZE:
            src_w_float, src_h_float = float(original_W), float(original_H)
            src_ratio = src_w_float / src_h_float
            canvas_ratio = float(op_w) / float(op_h)
            
            temp_pixels_to_scale = current_pixels_content
            if abs(src_ratio - canvas_ratio) > 1e-5:
                crop_x_val, crop_y_val = 0.0, 0.0
                if src_ratio > canvas_ratio:
                    new_w_content = src_h_float * canvas_ratio
                    crop_x_val = src_w_float - new_w_content
                else:
                    new_h_content = src_w_float / canvas_ratio
                    crop_y_val = src_h_float - new_h_content
                crop_x_half = round(crop_x_val / 2.0)
                crop_y_half = round(crop_y_val / 2.0)
                temp_pixels_to_scale = current_pixels_content[:, crop_y_half : original_H - crop_y_half, crop_x_half : original_W - crop_x_half, :]
            
            if temp_pixels_to_scale.shape[2] != op_w or temp_pixels_to_scale.shape[1] != op_h:
                pixels_nchw = temp_pixels_to_scale.permute(0, 3, 1, 2)
                final_canvas_pixels = self._interp_image(pixels_nchw, (op_w, op_h), upscale_method).permute(0, 2, 3, 1).clamp(0.0, 1.0)
            else:
                final_canvas_pixels = temp_pixels_to_scale

        elif action == self.ACTION_TYPE_PAD_RESIZE:
            src_w_float, src_h_float = float(original_W), float(original_H)
            scale_ratio_w = float(op_w) / src_w_float
            scale_ratio_h = float(op_h) / src_h_float
            fit_scale = min(scale_ratio_w, scale_ratio_h)
            content_w_scaled = max(1, round(src_w_float * fit_scale))
            content_h_scaled = max(1, round(src_h_float * fit_scale))

            scaled_content_pixels = current_pixels_content
            if content_w_scaled != original_W or content_h_scaled != original_H:
                pixels_nchw = current_pixels_content.permute(0, 3, 1, 2)
                scaled_content_pixels = self._interp_image(pixels_nchw, (content_w_scaled, content_h_scaled), upscale_method).permute(0, 2, 3, 1).clamp(0.0, 1.0)

            final_canvas_pixels = torch.zeros((_B, op_h, op_w, _C), dtype=pixels.dtype, device=device)
            pad_x_offset = (op_w - content_w_scaled) // 2
            pad_y_offset = (op_h - content_h_scaled) // 2
            final_canvas_pixels[:, pad_y_offset : pad_y_offset + content_h_scaled, pad_x_offset : pad_x_offset + content_w_scaled, :] = scaled_content_pixels
        else:
            final_canvas_pixels = current_pixels_content

        # --- Шаг 3: `divisible_by` ---
        h_before_div, w_before_div = final_canvas_pixels.shape[1:3]
        _divisible_by = divisible_by if divisible_by is not None else 1
        
        if _divisible_by > 1:
            final_H_div = ((h_before_div + _divisible_by - 1) // _divisible_by) * _divisible_by
            final_W_div = ((w_before_div + _divisible_by - 1) // _divisible_by) * _divisible_by

            if final_H_div != h_before_div or final_W_div != w_before_div:
                padded_for_div_pixels = torch.zeros((_B, final_H_div, final_W_div, _C), dtype=final_canvas_pixels.dtype, device=device)
                pad_h_total = final_H_div - h_before_div
                pad_w_total = final_W_div - w_before_div
                pad_top = pad_h_total // 2
                pad_left = pad_w_total // 2
                padded_for_div_pixels[:, pad_top : pad_top + h_before_div, pad_left : pad_left + w_before_div, :] = final_canvas_pixels
                final_output_pixels = padded_for_div_pixels
            else:
                final_output_pixels = final_canvas_pixels
        else:
            final_output_pixels = final_canvas_pixels
        
        output_height = final_output_pixels.shape[1]
        output_width = final_output_pixels.shape[2]
            
        return (final_output_pixels, output_width, output_height)

NODE_CLASS_MAPPINGS = {
    "TS_ImageResize": TS_ImageResize
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageResize": "TS Image Resize"
}