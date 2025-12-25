import torch
import comfy
import torch.nn.functional as F
import math

# Custom Gaussian blur function with separable convolution for performance on GPU
def gaussian_blur_mask(mask_tensor_batch, blur_amount, device):
    """
    Применяет Гауссово размытие к БАТЧУ тензоров маски, используя разделяемую свертку для эффективности.
    Args:
        mask_tensor_batch (torch.Tensor): Тензор масок (B, H, W), должен быть float32.
        blur_amount (int): Степень размытия. Интерпретируется как примерное значение сигмы.
                           Также влияет на размер ядра.
        device (torch.device): Устройство (например, 'cpu' или 'cuda'), на котором выполняются операции.
    Returns:
        torch.Tensor: Размытый тензор масок (B, H, W).
    """
    if blur_amount <= 0:
        return mask_tensor_batch # Размытие не требуется, возвращаем исходные маски

    # Преобразуем blur_amount в значение сигмы для функции Гаусса
    sigma = float(blur_amount) / 3.0
    
    # Вычисляем размер ядра на основе сигмы. Оно должно быть нечетным.
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1 # Гарантируем нечетный размер ядра

    # Убедимся, что размер ядра не менее 3 для значимого размытия
    kernel_size = max(3, kernel_size)

    # Создаем 1D ядро Гаусса на указанном устройстве
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=torch.float32)
    gauss = torch.exp((-ax ** 2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum() # Нормализуем, чтобы сумма была равна 1

    # Вход mask_tensor_batch имеет форму B, H, W. Нужно преобразовать в [B, C, H, W] для F.conv2d, где C=1.
    input_tensor_for_conv = mask_tensor_batch.unsqueeze(1) # [B, 1, H, W]

    # --- Применяем разделяемую свертку ---
    # 1. Горизонтальное размытие
    # Ядро для горизонтальной свертки: [1, 1, 1, K] (out_channels, in_channels/groups, kernel_height, kernel_width)
    horizontal_kernel = kernel_1d.view(1, 1, 1, -1)
    
    blurred_h = F.conv2d(input_tensor_for_conv, 
                         horizontal_kernel, 
                         padding=(0, kernel_size // 2), # Дополнение только по ширине
                         groups=1) # группы=1 для стандартной свертки
    
    # 2. Вертикальное размытие
    # Ядро для вертикальной свертки: [1, 1, K, 1]
    vertical_kernel = kernel_1d.view(1, 1, -1, 1)

    blurred_v = F.conv2d(blurred_h, 
                         vertical_kernel, 
                         padding=(kernel_size // 2, 0), # Дополнение только по высоте
                         groups=1)

    return blurred_v.squeeze(1) # Возвращаем к форме [B, H, W]


def box_blur_mask(mask_tensor_batch, blur_amount, device):
    """
    Применяет Box размытие к БАТЧУ тензоров маски, используя разделяемую свертку.
    Args:
        mask_tensor_batch (torch.Tensor): Тензор масок (B, H, W), должен быть float32.
        blur_amount (int): Степень размытия (примерный радиус).
        device (torch.device): Устройство (например, 'cpu' или 'cuda').
    Returns:
        torch.Tensor: Размытый тензор масок (B, H, W).
    """
    if blur_amount <= 0:
        return mask_tensor_batch

    # Размер ядра для box blur: 2 * radius + 1
    kernel_size = int(blur_amount * 2) + 1
    
    # Убедимся, что размер ядра не менее 3 для значимого размытия, если blur_amount > 0
    if blur_amount > 0:
        kernel_size = max(3, kernel_size)
    else: # Если blur_amount был 0, но почему-то не вернулся выше, то kernel_size=1
        return mask_tensor_batch


    # Создаем 1D ядро "коробки" (все единицы, нормализованные)
    kernel_1d = torch.ones(kernel_size, device=device, dtype=torch.float32) / kernel_size

    input_tensor_for_conv = mask_tensor_batch.unsqueeze(1) # [B, 1, H, W]

    # --- Применяем разделяемую свертку ---
    # 1. Горизонтальное размытие
    horizontal_kernel = kernel_1d.view(1, 1, 1, -1)
    
    blurred_h = F.conv2d(input_tensor_for_conv, 
                         horizontal_kernel, 
                         padding=(0, kernel_size // 2), 
                         groups=1)
    
    # 2. Вертикальное размытие
    vertical_kernel = kernel_1d.view(1, 1, -1, 1)

    blurred_v = F.conv2d(blurred_h, 
                         vertical_kernel, 
                         padding=(kernel_size // 2, 0), 
                         groups=1)

    return blurred_v.squeeze(1) # Возвращаем к форме [B, H, W]


class TSCropToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 64, "min": 0, "max": 320}),
                "divide_by": ("INT", {"default": 32, "min": 1, "max": 64}),
                "max_resolution": ("INT", {"default": 720, "min": 320, "max": 2048, "step": 1, "description": "Max resolution for the smaller side of the crop."}),
                "fixed_mask_frame_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "description": "If > 0, use mask from this 1-indexed frame for all crops. (Overrides interpolation)"}),
                "interpolation_window_size": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "description": "Number of frames for interpolation (0 for none). Affects frames within the batch."}),
                "force_gpu": ("BOOLEAN", {"default": True, "description": "If true, force processing on GPU if available."}),
                # Добавление параметров в конец для совместимости
                "fixed_crop_size": ("BOOLEAN", {"default": False}),
                "fixed_width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "fixed_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "INT", "INT")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "crop_data", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def crop(self, images, mask, padding, divide_by, max_resolution, fixed_mask_frame_index, interpolation_window_size, force_gpu, fixed_crop_size=False, fixed_width=1024, fixed_height=1024):
        # Определение целевого устройства
        target_device = comfy.model_management.get_torch_device() if force_gpu and torch.cuda.is_available() else torch.device("cpu")
        print(f"TSCropToMask: Using device {target_device}")

        # Перемещение входных тензоров на целевое устройство
        images = images.to(target_device)
        mask = mask.to(target_device)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        batch_size = images.shape[0]
        original_img_h, original_img_w = images.shape[1], images.shape[2]
        
        if mask.shape[0] == batch_size:
            pass
        elif mask.shape[0] == 1 and batch_size > 1:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1)
            print(f"Info: Single mask repeated for {batch_size} images.")
        elif batch_size == 1 and mask.shape[0] > 1:
            print(f"Info: Single image provided with batch mask. Will use specified or first mask frame for crop calculation.")
            pass
        else:
            print(f"Warning: Mask batch size ({mask.shape[0]}) and image batch size ({batch_size}) mismatch. Falling back to using first mask for all images in the batch.")
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        # Проверяем, являются ли все маски в батче пустыми или однотонными.
        all_masks_are_solid = True
        for i in range(mask.shape[0]):
            m = mask[i]
            if m.numel() > 0: # Убеждаемся, что в срезе маски есть элементы
                if m.min() != m.max():
                    all_masks_are_solid = False
                    break
        
        if all_masks_are_solid:
            print("TSCropToMask: Все входные маски пустые или однотонные. Пропускаем обрезку и возвращаем исходные изображения.")
            crop_data = []
            for i in range(batch_size):
                crop_data.append({
                    "original_x": 0, "original_y": 0,
                    "original_width": original_img_w, "original_height": original_img_h,
                    "crop_x": 0, "crop_y": 0,
                    "initial_crop_width": original_img_w, "initial_crop_height": original_img_h,
                    "final_crop_width": original_img_w, "final_crop_height": original_img_h,
                })
            return (images, mask, crop_data, original_img_w, original_img_h)

        raw_regions = []
        if fixed_mask_frame_index > 0:
            mask_idx_to_use = min(fixed_mask_frame_index - 1, mask.shape[0] - 1)
            fixed_m = mask[mask_idx_to_use]
            nonzero = torch.nonzero(fixed_m, as_tuple=False)
            if len(nonzero) == 0:
                y_min, x_min, y_max, x_max = 0, 0, original_img_h, original_img_w
            else:
                y_min, x_min = nonzero.min(dim=0).values
                y_max, x_max = nonzero.max(dim=0).values
                y_min, x_min, y_max, x_max = y_min.item(), x_min.item(), y_max.item() + 1, x_max.item() + 1
            
            y_min, x_min = max(0, y_min - padding), max(0, x_min - padding)
            y_max, x_max = min(original_img_h, y_max + padding), min(original_img_w, x_max + padding)
            for i in range(batch_size):
                raw_regions.append({"x_min": x_min, "y_min": y_min, "width": x_max - x_min, "height": y_max - y_min})
        else:
            for i in range(batch_size):
                m = mask[i]
                nonzero = torch.nonzero(m, as_tuple=False)
                if len(nonzero) == 0:
                    y_min, x_min, y_max, x_max = 0, 0, original_img_h, original_img_w
                else:
                    y_min, x_min = nonzero.min(dim=0).values
                    y_max, x_max = nonzero.max(dim=0).values
                    y_min, x_min, y_max, x_max = y_min.item(), x_min.item(), y_max.item() + 1, x_max.item() + 1
                
                y_min, x_min = max(0, y_min - padding), max(0, x_min - padding)
                y_max, x_max = min(original_img_h, y_max + padding), min(original_img_w, x_max + padding)
                raw_regions.append({"x_min": x_min, "y_min": y_min, "width": x_max - x_min, "height": y_max - y_min})
        
        smoothed_regions = []
        if fixed_mask_frame_index == 0 and interpolation_window_size > 0:
            for i in range(batch_size):
                window_start, window_end = max(0, i - interpolation_window_size // 2), min(batch_size, i + interpolation_window_size // 2 + 1)
                win = raw_regions[window_start:window_end]
                avg_x = sum(r["x_min"] for r in win) / len(win)
                avg_y = sum(r["y_min"] for r in win) / len(win)
                avg_w = sum(r["width"] for r in win) / len(win)
                avg_h = sum(r["height"] for r in win) / len(win)
                smoothed_regions.append({"x_min": int(round(avg_x)), "y_min": int(round(avg_y)), "width": int(round(max(1, avg_w))), "height": int(round(max(1, avg_h)))})
        else:
            smoothed_regions = raw_regions

        # --- Логика Fixed Crop Size (сохранение пропорций) ---
        if fixed_crop_size:
            target_aspect = fixed_width / fixed_height
            for r in smoothed_regions:
                cur_w, cur_h = r["width"], r["height"]
                cur_aspect = cur_w / cur_h
                if cur_aspect > target_aspect:
                    new_h = cur_w / target_aspect
                    r["y_min"] = int(r["y_min"] - (new_h - cur_h) / 2)
                    r["height"] = int(new_h)
                else:
                    new_w = cur_h * target_aspect
                    r["x_min"] = int(r["x_min"] - (new_w - cur_w) / 2)
                    r["width"] = int(new_w)
                
                # Зажим в границы оригинального изображения
                r["x_min"] = max(0, min(r["x_min"], original_img_w - 1))
                r["y_min"] = max(0, min(r["y_min"], original_img_h - 1))
                r["width"] = int(min(r["width"], original_img_w - r["x_min"]))
                r["height"] = int(min(r["height"], original_img_h - r["y_min"]))

        final_regions = []
        for r in smoothed_regions:
            target_width = ((r["width"] + divide_by - 1) // divide_by) * divide_by
            target_height = ((r["height"] + divide_by - 1) // divide_by) * divide_by
            final_regions.append({
                "x_min": r["x_min"], "y_min": r["y_min"],
                "width": r["width"], "height": r["height"],
                "target_width": target_width, "target_height": target_height
            })

        max_target_width = max(r["target_width"] for r in final_regions)
        max_target_height = max(r["target_height"] for r in final_regions)
        
        cropped_images, cropped_masks, crop_data = [], [], []
        for i in range(batch_size):
            img = images[i]
            r = final_regions[i]
            center_x, center_y = r["x_min"] + r["width"] // 2, r["y_min"] + r["height"] // 2
            crop_x, crop_y = center_x - max_target_width // 2, center_y - max_target_height // 2
            crop_x = max(0, min(crop_x, original_img_w - max_target_width))
            crop_y = max(0, min(crop_y, original_img_h - max_target_height))

            if original_img_w < max_target_width: crop_x = max(0, (original_img_w - max_target_width) // 2)
            if original_img_h < max_target_height: crop_y = max(0, (original_img_h - max_target_height) // 2)

            crop_x_end, crop_y_end = crop_x + max_target_width, crop_y + max_target_height
            cropped_img = img[crop_y:crop_y_end, crop_x:crop_x_end, :]
            cropped_msk = mask[min(i, mask.shape[0]-1)][crop_y:crop_y_end, crop_x:crop_x_end]
            
            if cropped_img.shape[0] < max_target_height or cropped_img.shape[1] < max_target_width:
                pb, pr = max_target_height - cropped_img.shape[0], max_target_width - cropped_img.shape[1]
                cropped_img = F.pad(cropped_img.unsqueeze(0).permute(0, 3, 1, 2), (0, pr, 0, pb), value=0).permute(0, 2, 3, 1).squeeze(0)
                cropped_msk = F.pad(cropped_msk.unsqueeze(0).unsqueeze(0), (0, pr, 0, pb), value=0).squeeze(0).squeeze(0)
            
            cropped_images.append(cropped_img)
            cropped_masks.append(cropped_msk)
            crop_data.append({
                "original_x": r["x_min"], "original_y": r["y_min"],
                "original_width": r["width"], "original_height": r["height"],
                "crop_x": crop_x, "crop_y": crop_y,
                "initial_crop_width": max_target_width, "initial_crop_height": max_target_height,
                "final_crop_width": max_target_width, "final_crop_height": max_target_height
            })
        
        cropped_images, cropped_masks = torch.stack(cropped_images), torch.stack(cropped_masks)
        
        curr_w, curr_h = max_target_width, max_target_height
        if fixed_crop_size:
            new_width, new_height = fixed_width, fixed_height
        else:
            min_dim = min(curr_w, curr_h)
            new_width, new_height = curr_w, curr_h
            if min_dim > max_resolution:
                scale = max_resolution / min_dim
                new_width = max(divide_by, (int(curr_w * scale) // divide_by) * divide_by)
                new_height = max(divide_by, (int(curr_h * scale) // divide_by) * divide_by)

        if new_width != curr_w or new_height != curr_h:
            cropped_images = F.interpolate(cropped_images.permute(0, 3, 1, 2), size=(new_height, new_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            cropped_masks = F.interpolate(cropped_masks.unsqueeze(1), size=(new_height, new_width), mode='nearest').squeeze(1)
            for d in crop_data: d["final_crop_width"], d["final_crop_height"] = new_width, new_height
        
        return (cropped_images, cropped_masks, crop_data, new_width, new_height)


class TSRestoreFromCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "blur": ("INT", {"default": 64, "min": 0, "max": 256, "step": 1}),
                "blur_type": (["Gaussian", "Box"], {"default": "Gaussian"}),
                "force_gpu": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "image/processing"

    def restore(self, original_images, cropped_images, crop_data, blur, blur_type, force_gpu):
        target_device = comfy.model_management.get_torch_device() if force_gpu and torch.cuda.is_available() else torch.device("cpu")
        print(f"TSRestoreFromCrop: Using device {target_device}")
        original_images, cropped_images = original_images.to(target_device), cropped_images.to(target_device)
        restored_images = []
        
        orig_img_h, orig_img_w = original_images.shape[1], original_images.shape[2]

        for i in range(original_images.shape[0]):
            orig_img = original_images[i].clone()
            c_img, data = cropped_images[i], crop_data[i]
            
            if c_img.shape[0] != data["initial_crop_height"] or c_img.shape[1] != data["initial_crop_width"]:
                c_img = F.interpolate(c_img.permute(2, 0, 1).unsqueeze(0), size=(data["initial_crop_height"], data["initial_crop_width"]), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            
            p_x, p_y = data["original_x"], data["original_y"]
            p_w, p_h = data["original_width"], data["original_height"]
            
            paste_x_in_crop = p_x - data["crop_x"]
            paste_y_in_crop = p_y - data["crop_y"]

            temp_canvas = torch.zeros((p_h, p_w, 3), device=target_device, dtype=c_img.dtype)
            src_y_s, src_x_s = max(0, paste_y_in_crop), max(0, paste_x_in_crop)
            src_y_e, src_x_e = min(c_img.shape[0], paste_y_in_crop + p_h), min(c_img.shape[1], paste_x_in_crop + p_w)
            dst_y_s, dst_x_s = max(0, -paste_y_in_crop), max(0, -paste_x_in_crop)
            
            if src_y_e > src_y_s and src_x_e > src_x_s:
                temp_canvas[dst_y_s : dst_y_s + (src_y_e - src_y_s), dst_x_s : dst_x_s + (src_x_e - src_x_s), :] = c_img[src_y_s:src_y_e, src_x_s:src_x_e, :]
            
            region_to_paste = temp_canvas 

            y_start_orig, y_end_orig = max(0, p_y), min(orig_img_h, p_y + p_h)
            x_start_orig, x_end_orig = max(0, p_x), min(orig_img_w, p_x + p_w)

            if y_end_orig > y_start_orig and x_end_orig > x_start_orig:
                local_y_s, local_y_e = y_start_orig - p_y, y_end_orig - p_y
                local_x_s, local_x_e = x_start_orig - p_x, x_end_orig - p_x
                final_paste_region = region_to_paste[local_y_s:local_y_e, local_x_s:local_x_e, :]

                if blur > 0:
                    # --- УМНАЯ МАСКА С УЧЕТОМ ГРАНИЦ ИЗОБРАЖЕНИЯ ---
                    # Если область кропа примыкает к краю кадра ближе чем на blur, 
                    # мы не сужаем маску с этой стороны.
                    
                    left_edge = 0 if p_x <= blur else blur
                    top_edge = 0 if p_y <= blur else blur
                    right_edge = 0 if (p_x + p_w) >= (orig_img_w - blur) else blur
                    bottom_edge = 0 if (p_y + p_h) >= (orig_img_h - blur) else blur

                    shrunk_mask_canvas = torch.zeros((p_h, p_w), device=target_device)
                    
                    # Сужаем только те стороны, которые НЕ у края кадра
                    m_y_s, m_x_s = top_edge, left_edge
                    m_y_e, m_x_e = max(m_y_s, p_h - bottom_edge), max(m_x_s, p_w - right_edge)
                    
                    if m_y_e > m_y_s and m_x_e > m_x_s:
                        shrunk_mask_canvas[m_y_s:m_y_e, m_x_s:m_x_e] = 1.0
                    
                    if blur_type == "Gaussian":
                        blurred_mask = gaussian_blur_mask(shrunk_mask_canvas.unsqueeze(0), blur, target_device).squeeze(0)
                    else:
                        blurred_mask = box_blur_mask(shrunk_mask_canvas.unsqueeze(0), blur, target_device).squeeze(0)
                    
                    sliced_mask = blurred_mask[local_y_s:local_y_e, local_x_s:local_x_e].unsqueeze(-1)
                    original_part = orig_img[y_start_orig:y_end_orig, x_start_orig:x_end_orig, :]
                    
                    orig_img[y_start_orig:y_end_orig, x_start_orig:x_end_orig, :] = (final_paste_region * sliced_mask) + (original_part * (1.0 - sliced_mask))
                else:
                    orig_img[y_start_orig:y_end_orig, x_start_orig:x_end_orig, :] = final_paste_region
            
            restored_images.append(orig_img)
        
        return (torch.stack(restored_images),)


NODE_CLASS_MAPPINGS = {
    "TSCropToMask": TSCropToMask,
    "TSRestoreFromCrop": TSRestoreFromCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSCropToMask": "TS Crop To Mask",
    "TSRestoreFromCrop": "TS Restore From Crop",
}