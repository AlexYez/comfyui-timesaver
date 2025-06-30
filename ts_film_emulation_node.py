import torch

# Вспомогательная функция для плавного перехода (smoothstep)
# Используется для создания плавных масок для теней/светов
def smoothstep(x):
    # Clamping x to [0, 1] for robustness, though usually handled by calling code
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

# Пресеты для различных видов пленки
FILM_PRESETS = {
    "None": {
        "brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0,
        "red_mult": 1.0, "green_mult": 1.0, "blue_mult": 1.0,
        "black_point": 0.0, "white_point": 1.0,
        "grain_strength": 0.0, "vignette_strength": 0.0, "vignette_midpoint": 0.6,
        "shadow_tint_rgb": (0.0, 0.0, 0.0), "shadow_tint_strength": 0.0,
        "highlight_tint_rgb": (0.0, 0.0, 0.0), "highlight_tint_strength": 0.0,
        "shadow_luminance_threshold": 0.2, "highlight_luminance_threshold": 0.8,
        "shadow_feather": 0.1, "highlight_feather": 0.1,
    },
    "Kodachrome 64 (Vibrant)": {
        "brightness": 1.0, "contrast": 1.15, "gamma": 0.9, "saturation": 1.25,
        "red_mult": 1.05, "green_mult": 0.95, "blue_mult": 0.9,
        "black_point": 0.02, "white_point": 0.98,
        "grain_strength": 0.015, "vignette_strength": 0.1, "vignette_midpoint": 0.7,
        "shadow_tint_rgb": (0.05, 0.0, 0.0), "shadow_tint_strength": 0.05, # Slight warm shadows
        "highlight_tint_rgb": (0.0, 0.02, 0.05), "highlight_tint_strength": 0.03, # Slight cool highlights
        "shadow_luminance_threshold": 0.25, "highlight_luminance_threshold": 0.75,
        "shadow_feather": 0.1, "highlight_feather": 0.1,
    },
    "Fujifilm Velvia 50 (High Sat)": {
        "brightness": 1.0, "contrast": 1.2, "gamma": 0.85, "saturation": 1.35,
        "red_mult": 1.0, "green_mult": 1.05, "blue_mult": 1.05,
        "black_point": 0.03, "white_point": 0.97,
        "grain_strength": 0.01, "vignette_strength": 0.05, "vignette_midpoint": 0.65,
        "shadow_tint_rgb": (0.02, 0.05, 0.0), "shadow_tint_strength": 0.07, # Greenish shadows
        "highlight_tint_rgb": (0.0, 0.0, 0.02), "highlight_tint_strength": 0.04, # Slight blue highlights
        "shadow_luminance_threshold": 0.2, "highlight_luminance_threshold": 0.8,
        "shadow_feather": 0.12, "highlight_feather": 0.12,
    },
    "Portra 400 (Soft & Natural)": {
        "brightness": 1.0, "contrast": 1.05, "gamma": (1.02, 1.0, 1.05), "saturation": 1.1, # R slightly up, B more up
        "red_mult": 1.03, "green_mult": 0.98, "blue_mult": 1.02,
        "black_point": 0.01, "white_point": 0.99,
        "grain_strength": 0.02, "vignette_strength": 0.15, "vignette_midpoint": 0.75,
        "shadow_tint_rgb": (0.0, 0.0, 0.05), "shadow_tint_strength": 0.03, # Slight cool shadows
        "highlight_tint_rgb": (0.05, 0.02, 0.0), "highlight_tint_strength": 0.03, # Slight warm highlights
        "shadow_luminance_threshold": 0.3, "highlight_luminance_threshold": 0.7,
        "shadow_feather": 0.15, "highlight_feather": 0.15,
    },
    "CineStill 800T (Tungsten Look)": {
        "brightness": 0.98, "contrast": 1.1, "gamma": (0.9, 0.95, 1.0), "saturation": 1.15, # Shift green/red down relative to blue
        "red_mult": 1.15, "green_mult": 0.9, "blue_mult": 0.85, # Tungsten shift
        "black_point": 0.02, "white_point": 0.95,
        "grain_strength": 0.03, "vignette_strength": 0.2, "vignette_midpoint": 0.6,
        "shadow_tint_rgb": (0.0, 0.05, 0.05), "shadow_tint_strength": 0.08, # Cyan/green shadows
        "highlight_tint_rgb": (0.1, 0.0, 0.0), "highlight_tint_strength": 0.05, # Reddish highlights (halation imitation)
        "shadow_luminance_threshold": 0.2, "highlight_luminance_threshold": 0.8,
        "shadow_feather": 0.1, "highlight_feather": 0.1,
    },
    "Ilford HP5 Plus (B&W)": {
        "brightness": 1.0, "contrast": 1.3, "gamma": 0.7, "saturation": 0.0, # Completely desaturated
        "red_mult": 1.0, "green_mult": 1.0, "blue_mult": 1.0, # No color shift for B&W
        "black_point": 0.05, "white_point": 0.95,
        "grain_strength": 0.04, "vignette_strength": 0.25, "vignette_midpoint": 0.7,
        "shadow_tint_rgb": (0.0, 0.0, 0.0), "shadow_tint_strength": 0.0,
        "highlight_tint_rgb": (0.0, 0.0, 0.0), "highlight_tint_strength": 0.0,
        "shadow_luminance_threshold": 0.2, "highlight_luminance_threshold": 0.8,
        "shadow_feather": 0.1, "highlight_feather": 0.1,
    },
    "Lomography 800 (Gritty)": {
        "brightness": 1.02, "contrast": 1.1, "gamma": (0.95, 1.0, 0.9), "saturation": 1.2, # Shift blue down
        "red_mult": 1.08, "green_mult": 0.95, "blue_mult": 0.95,
        "black_point": 0.03, "white_point": 0.97,
        "grain_strength": 0.035, "vignette_strength": 0.2, "vignette_midpoint": 0.65,
        "shadow_tint_rgb": (0.05, 0.0, 0.05), "shadow_tint_strength": 0.05, # Magenta shadows
        "highlight_tint_rgb": (0.0, 0.05, 0.0), "highlight_tint_strength": 0.03, # Greenish highlights
        "shadow_luminance_threshold": 0.25, "highlight_luminance_threshold": 0.75,
        "shadow_feather": 0.1, "highlight_feather": 0.1,
    },
    "Cross-Process (Cyan/Green Cast)": {
        "brightness": 1.0, "contrast": 1.2, "gamma": (0.8, 1.1, 1.1), "saturation": 1.3, # Strong channel shifts
        "red_mult": 0.9, "green_mult": 1.15, "blue_mult": 1.1,
        "black_point": 0.05, "white_point": 0.95,
        "grain_strength": 0.025, "vignette_strength": 0.18, "vignette_midpoint": 0.55,
        "shadow_tint_rgb": (0.0, 0.08, 0.08), "shadow_tint_strength": 0.15, # Strong cyan/blue shadows
        "highlight_tint_rgb": (0.08, 0.08, 0.0), "highlight_tint_strength": 0.1, # Strong yellow/green highlights
        "shadow_luminance_threshold": 0.15, "highlight_luminance_threshold": 0.85,
        "shadow_feather": 0.1, "highlight_feather": 0.1,
    },
}

class TS_FilmEmulation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "film_type": (list(FILM_PRESETS.keys()),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "force_gpu": ("BOOLEAN", {"default": True}), # Возвращаем в интерфейс
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 64, "step": 1}), # Новый параметр для размера батча
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_emulation_batched" # Новая функция-обертка для батчинга
    CATEGORY = "Image/Postprocessing"

    # Основная логика обработки одного или мини-батча изображений
    def _apply_emulation_single_batch(self, images: torch.Tensor, film_type: str, strength: float, target_device: torch.device):
        """
        Применяет эффект эмуляции пленки к батчу изображений.
        Оптимизирована для использования in-place операций и работы на GPU.
        """
        
        # Перемещаем изображение на целевое устройство, если оно еще не там
        original_images = images.to(target_device)
        processed_images = original_images.clone() # Здесь создается копия, которая будет изменяться in-place

        params = FILM_PRESETS.get(film_type, FILM_PRESETS["None"])

        # Если выбран "None" и сила эффекта 0, возвращаем оригинальные изображения
        if film_type == "None" and strength == 0.0:
            return original_images

        # Извлекаем все параметры из выбранного пресета
        brightness = params["brightness"]
        contrast = params["contrast"]
        gamma_val = params["gamma"] # Can be float or (r,g,b) tuple
        saturation = params["saturation"]
        red_mult = params["red_mult"]
        green_mult = params["green_mult"]
        blue_mult = params["blue_mult"]
        black_point = params["black_point"]
        white_point = params["white_point"]
        grain_strength = params["grain_strength"]
        vignette_strength = params["vignette_strength"]
        vignette_midpoint = params["vignette_midpoint"]
        shadow_tint_rgb = params["shadow_tint_rgb"]
        shadow_tint_strength = params["shadow_tint_strength"]
        highlight_tint_rgb = params["highlight_tint_rgb"]
        highlight_tint_strength = params["highlight_tint_strength"]
        shadow_luminance_threshold = params["shadow_luminance_threshold"]
        highlight_luminance_threshold = params["highlight_luminance_threshold"]
        shadow_feather = params["shadow_feather"]
        highlight_feather = params["highlight_feather"]

        with torch.no_grad():
            # 1. Коррекция точек черного и белого
            processed_images.sub_(black_point).div_(white_point - black_point + 1e-6).clamp_(0.0, 1.0)

            # 2. Яркость
            processed_images.mul_(brightness)

            # 3. Контраст
            processed_images.sub_(0.5).mul_(contrast).add_(0.5)

            # 4. Канальная Гамма
            if isinstance(gamma_val, (list, tuple)):
                gamma_tensor = torch.tensor(gamma_val, device=target_device, dtype=processed_images.dtype).view(1, 1, 1, 3)
                processed_images.pow_(1.0 / gamma_tensor)
                del gamma_tensor
            else:
                processed_images.pow_(1.0 / gamma_val)

            # 5. Насыщенность
            if saturation != 1.0: # Apply only if saturation changes
                grayscale_images = processed_images.mean(dim=-1, keepdim=True)
                temp_grayscale_tint = grayscale_images.mul(1.0 - saturation)
                processed_images.mul_(saturation).add_(temp_grayscale_tint)
                del temp_grayscale_tint, grayscale_images

            # 6. Цветовые множители
            color_multipliers = torch.tensor(
                [red_mult, green_mult, blue_mult],
                device=target_device, dtype=processed_images.dtype
            ).view(1, 1, 1, 3)
            processed_images.mul_(color_multipliers)
            del color_multipliers

            processed_images.clamp_(0.0, 1.0) # Обрезаем после основных коррекций

            # 7. Сплит-Тонирование (Color Grading by Luminance)
            if shadow_tint_strength > 0 or highlight_tint_strength > 0:
                luminance = (processed_images[..., 0] * 0.2126 +
                             processed_images[..., 1] * 0.7152 +
                             processed_images[..., 2] * 0.0722).unsqueeze(-1)
                
                shadow_mask_raw = (shadow_luminance_threshold + shadow_feather - luminance) / shadow_feather
                shadow_mask = smoothstep(shadow_mask_raw)
                del shadow_mask_raw

                highlight_mask_raw = (luminance - (highlight_luminance_threshold - highlight_feather)) / highlight_feather
                highlight_mask = smoothstep(highlight_mask_raw)
                del highlight_mask_raw, luminance

                shadow_tint_color_tensor = torch.tensor(shadow_tint_rgb, device=target_device, dtype=processed_images.dtype).view(1, 1, 1, 3)
                highlight_tint_color_tensor = torch.tensor(highlight_tint_rgb, device=target_device, dtype=processed_images.dtype).view(1, 1, 1, 3)
                
                processed_images.add_(shadow_tint_color_tensor * shadow_tint_strength * shadow_mask)
                processed_images.add_(highlight_tint_color_tensor * highlight_tint_strength * highlight_mask)
                
                del shadow_tint_color_tensor, highlight_tint_color_tensor, shadow_mask, highlight_mask

                processed_images.clamp_(0.0, 1.0)

            # 8. Виньетирование (Vignette)
            if vignette_strength > 0:
                B, H, W, C = processed_images.shape
                
                x = torch.linspace(-1, 1, W, device=target_device)
                y = torch.linspace(-1, 1, H, device=target_device)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
                
                distance = torch.sqrt(grid_x**2 + grid_y**2)
                max_dist = torch.sqrt(torch.tensor(2.0, device=target_device))
                normalized_distance = distance / max_dist
                
                mask_region = normalized_distance - vignette_midpoint
                mask_region.clamp_(0.0, 1.0).div_(1.0 - vignette_midpoint + 1e-6)
                
                vignette_factor = 1.0 - (mask_region ** 1.5) * vignette_strength
                del mask_region, normalized_distance, distance, grid_x, grid_y, x, y, max_dist
                
                processed_images.mul_(vignette_factor.unsqueeze(-1))
                del vignette_factor

            # 9. Зерно (Grain)
            if grain_strength > 0:
                noise = (torch.randn_like(processed_images) * grain_strength).to(target_device)
                processed_images.add_(noise)
                processed_images.clamp_(0.0, 1.0)
                del noise

            # 10. Смешивание с оригинальным изображением по параметру strength
            # Вместо создания нового final_image, мы можем модифицировать processed_images
            # и original_images, а затем вернуть processed_images.
            # Если strength == 1.0, processed_images уже является финальным результатом.
            # Если strength == 0.0, original_images уже является финальным результатом,
            # и мы его вернули в самом начале.
            # Поэтому эта ветка выполняется только для 0 < strength < 1.0
            if strength > 0 and strength < 1.0:
                processed_images.mul_(strength).add_(original_images.mul(1.0 - strength))
            elif strength == 0.0: # Should have been caught earlier, but for safety
                processed_images = original_images # Reassign to avoid issues
            # If strength == 1.0, processed_images already holds the full effect, no further blend needed.
            
            # Explicitly clear CUDA cache for better memory management between batches
            if target_device.type == 'cuda':
                torch.cuda.empty_cache()

            return processed_images

    # Новая функция-обертка для обработки батчей
    def apply_emulation_batched(self, image: torch.Tensor, film_type: str, strength: float, force_gpu: bool, batch_size: int):
        total_images = image.shape[0] # Общее количество изображений в батче

        # Определяем целевое устройство
        if force_gpu and torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")
        
        # Если батч меньше или равен batch_size, обрабатываем все сразу
        if total_images <= batch_size:
            final_images = self._apply_emulation_single_batch(image, film_type, strength, target_device)
            return (final_images,)

        # Иначе, делим на мини-батчи
        output_images = []
        for i in range(0, total_images, batch_size):
            current_batch = image[i:i + batch_size]
            processed_batch = self._apply_emulation_single_batch(current_batch, film_type, strength, target_device)
            output_images.append(processed_batch.cpu()) # Перемещаем на CPU, чтобы освободить GPU память между мини-батчами

        # Объединяем все обработанные мини-батчи обратно в один большой тензор
        final_images = torch.cat(output_images, dim=0)

        # Перемещаем финальный тензор на целевое устройство, если ComfyUI ожидает его там
        # В большинстве случаев ComfyUI ожидает тензор на CPU для сохранения или дальнейшей обработки
        # но если следующая нода ожидает GPU, это может быть не оптимально.
        # Для безопасности возвращаем на CPU, как это делает большинство нод.
        return (final_images,)

# Этот словарь обязателен для ComfyUI для обнаружения вашей ноды
NODE_CLASS_MAPPINGS = {
    "TS_FilmEmulation": TS_FilmEmulation
}

# Этот словарь также обязателен для ComfyUI для отображения названий нод в UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FilmEmulation": "TS Film Emulation"
}