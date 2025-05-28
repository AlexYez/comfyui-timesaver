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

    # ACTION_TYPE_RESIZE_ONLY = "resize" # Убрано
    # ACTION_TYPE_CROP_RESIZE = "crop_to_fit" # Убрано
    # ACTION_TYPE_PAD_RESIZE = "pad_to_fit" # Убрано

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
                # "action": ([s.ACTION_TYPE_RESIZE_ONLY, s.ACTION_TYPE_CROP_RESIZE, s.ACTION_TYPE_PAD_RESIZE], {"default": s.ACTION_TYPE_RESIZE_ONLY}), # Убрано
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
    def VALIDATE_INPUTS(s, target_width, target_height, smaller_side, larger_side, scale_factor, upscale_method, divisible_by, **_):
        if upscale_method == "lanczos" and not TORCHVISION_AVAILABLE:
            return "Lanczos upscale_method requires torchvision to be installed."
        
        if divisible_by is not None and divisible_by < 1:
            return "divisible_by must be 1 or greater."

        sf_active = scale_factor is not None and scale_factor > 0.0
        tw_active = target_width is not None and target_width > 0
        th_active = target_height is not None and target_height > 0
        ss_active = smaller_side is not None and smaller_side > 0
        ls_active = larger_side is not None and larger_side > 0
        
        active_method_count = 0
        if sf_active: active_method_count +=1
        if tw_active or th_active: active_method_count +=1 # Считаем как один метод, если хотя бы один из target_width/height активен
        if ss_active or ls_active: active_method_count +=1 # Аналогично для smaller/larger_side

        if active_method_count > 1:
            return "More than one sizing method (scale_factor, target_width/height, smaller/larger_side) is active. Please use only one."
        
        # if active_method_count == 0 and (divisible_by is None or divisible_by == 1) : # Если нет методов изменения и нет кратности
        #     return "No resizing operation specified. Set a sizing method or divisible_by > 1."
            
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
        
        # --- Шаг 1: Определить "идеальные" целевые размеры (ideal_w, ideal_h) до учета divisible_by ---
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
                    ratio_orig = ideal_w / ideal_h
                    ratio_target = float(_target_width) / float(_target_height)
                    if ratio_orig > ratio_target: 
                        ideal_w = float(_target_width)
                        ideal_h = ideal_w / ratio_orig
                    else: 
                        ideal_h = float(_target_height)
                        ideal_w = ideal_h * ratio_orig
                elif _target_width > 0:
                    ideal_h = ideal_h * (float(_target_width) / ideal_w)
                    ideal_w = float(_target_width)
                elif _target_height > 0:
                    ideal_w = ideal_w * (float(_target_height) / ideal_h)
                    ideal_h = float(_target_height)
            else: # keep_proportion == False
                if _target_width > 0: ideal_w = float(_target_width)
                if _target_height > 0: ideal_h = float(_target_height)
        elif chosen_method == "side_dims":
            # Эта логика всегда сохраняет пропорции
            if _smaller_side > 0:
                if ideal_w < ideal_h: 
                    ideal_h = ideal_h * (float(_smaller_side) / ideal_w)
                    ideal_w = float(_smaller_side)
                else: 
                    ideal_w = ideal_w * (float(_smaller_side) / ideal_h)
                    ideal_h = float(_smaller_side)
            elif _larger_side > 0:
                if ideal_w > ideal_h: 
                    ideal_h = ideal_h * (float(_larger_side) / ideal_w)
                    ideal_w = float(_larger_side)
                else: 
                    ideal_w = ideal_w * (float(_larger_side) / ideal_h)
                    ideal_h = float(_larger_side)
        
        ideal_w = max(1.0, ideal_w)
        ideal_h = max(1.0, ideal_h)

        # --- Шаг 2: Скорректировать идеальные размеры до кратности divisible_by (округление ВНИЗ) ---
        # Это будут финальные размеры выходного изображения.
        if _divisible_by > 1:
            if keep_proportion or chosen_method == "side_dims" or chosen_method == "scale_factor":
                # При сохранении пропорций или если метод уже подразумевает пропорции,
                # нужно найти такой масштаб, чтобы обе стороны после округления вниз до кратности _divisible_by
                # оставались как можно ближе к идеальным пропорциям.
                # Это сложнее, чем просто округлить каждую сторону.
                # Простой подход: округлить меньшую из идеальных сторон вниз до кратности,
                # а вторую вычислить, сохранив идеальные пропорции, и затем ее тоже округлить вниз.
                # Либо, сначала масштабировать до кратного, а потом кропать.
                # Выберем путь: вычислить финальные кратные размеры, потом к ним масштабировать с кропом.

                # Округляем каждую идеальную размерность вниз до ближайшего кратного _divisible_by
                final_target_w_div = math.floor(ideal_w / _divisible_by) * _divisible_by
                final_target_h_div = math.floor(ideal_h / _divisible_by) * _divisible_by

                # Важно: если округление дало 0, ставим _divisible_by (минимально возможное)
                final_target_w_div = max(_divisible_by, final_target_w_div)
                final_target_h_div = max(_divisible_by, final_target_h_div)
            else: # keep_proportion == False и метод target_dims
                final_target_w_div = math.floor(ideal_w / _divisible_by) * _divisible_by if _target_width > 0 else ideal_w
                final_target_h_div = math.floor(ideal_h / _divisible_by) * _divisible_by if _target_height > 0 else ideal_h
                final_target_w_div = max(_divisible_by if _target_width > 0 else 1, final_target_w_div)
                final_target_h_div = max(_divisible_by if _target_height > 0 else 1, final_target_h_div)
        else:
            final_target_w_div = round(ideal_w)
            final_target_h_div = round(ideal_h)
        
        final_target_w_div = max(1, int(final_target_w_div))
        final_target_h_div = max(1, int(final_target_h_div))


        # --- Шаг 3: Масштабирование и кроп (или простое масштабирование) ---
        current_pixels_nchw = pixels.permute(0, 3, 1, 2)

        if keep_proportion or chosen_method == "side_dims" or chosen_method == "scale_factor":
            # Масштабируем так, чтобы изображение ПОЛНОСТЬЮ ПОКРЫЛО final_target_w_div x final_target_h_div
            # с сохранением исходных пропорций. Затем кропаем.
            
            src_ratio = float(original_W) / float(original_H)
            target_canvas_ratio = float(final_target_w_div) / float(final_target_h_div)

            if src_ratio > target_canvas_ratio:
                # Исходное шире, чем целевой холст -> масштабируем по высоте целевого холста
                # Ширина будет больше, чем нужно, ее потом обрежем
                scale_to_h = final_target_h_div
                scale_to_w = round(scale_to_h * src_ratio)
            else:
                # Исходное выше (или той же пропорции), чем целевой холст -> масштабируем по ширине целевого холста
                # Высота будет больше, чем нужно, ее потом обрежем
                scale_to_w = final_target_w_div
                scale_to_h = round(scale_to_w / src_ratio)
            
            scale_to_w = max(1, scale_to_w)
            scale_to_h = max(1, scale_to_h)

            # Если после вычисления масштаба размеры все еще равны оригинальным и целевым,
            # И при этом целевые размеры не изменились от оригинальных - можно пропустить интерполяцию
            needs_interp = True
            if scale_to_w == original_W and scale_to_h == original_H and \
               final_target_w_div == original_W and final_target_h_div == original_H:
                 needs_interp = False

            if needs_interp or (scale_to_w != original_W or scale_to_h != original_H):
                 scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)
            else:
                 scaled_pixels_nchw = current_pixels_nchw


            # Кроп до final_target_w_div, final_target_h_div
            crop_x_start = (scaled_pixels_nchw.shape[3] - final_target_w_div) // 2
            crop_y_start = (scaled_pixels_nchw.shape[2] - final_target_h_div) // 2
            
            # Защита от выхода за пределы, если что-то пошло не так с расчетами выше
            crop_x_start = max(0, crop_x_start)
            crop_y_start = max(0, crop_y_start)
            
            # Убедимся, что не пытаемся обрезать больше, чем есть
            crop_w = min(final_target_w_div, scaled_pixels_nchw.shape[3] - crop_x_start)
            crop_h = min(final_target_h_div, scaled_pixels_nchw.shape[2] - crop_y_start)


            if crop_w != final_target_w_div or crop_h != final_target_h_div:
                # Если после кропа размеры не совпали с целевыми (что маловероятно при правильных расчетах, но для безопасности)
                # создаем холст нужного размера и вставляем обрезанное, или просто берем обрезанное.
                # Это может случиться, если scale_to_w/h оказались МЕНЬШЕ final_target_w/h_div
                # В этом случае предыдущая логика масштабирования "покрыть холст" не идеальна.
                # Проще сначала убедиться, что scale_to_w/h >= final_target_w/h_div.
                # Исправим scale_to_w/h на предыдущем шаге, если они меньше целевых после округления
                if scale_to_w < final_target_w_div : scale_to_w = final_target_w_div
                if scale_to_h < final_target_h_div : scale_to_h = final_target_h_div
                # И ПЕРЕМАСШТАБИРОВАТЬ, если изменились
                if scaled_pixels_nchw.shape[3] != scale_to_w or scaled_pixels_nchw.shape[2] != scale_to_h:
                     scaled_pixels_nchw = self._interp_image(current_pixels_nchw, (scale_to_w, scale_to_h), upscale_method)
                # Пересчитываем кроп
                crop_x_start = (scale_to_w - final_target_w_div) // 2
                crop_y_start = (scale_to_h - final_target_h_div) // 2
                crop_x_start = max(0, crop_x_start)
                crop_y_start = max(0, crop_y_start)
                crop_w = final_target_w_div
                crop_h = final_target_h_div


            final_output_pixels_nchw = scaled_pixels_nchw[:, :, crop_y_start : crop_y_start + crop_h, crop_x_start : crop_x_start + crop_w]

        else: # keep_proportion == False (и метод был target_dims)
            # Простое масштабирование (искажение) до final_target_w_div, final_target_h_div
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