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
                # Параметр interpolation_strength удален, так как всегда 1.0
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "INT", "INT")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "crop_data", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def crop(self, images, mask, padding, divide_by, max_resolution, fixed_mask_frame_index, interpolation_window_size):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        batch_size = images.shape[0]
        
        if mask.shape[0] != batch_size:
            if mask.shape[0] == 1:
                mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                print(f"Warning: Mask batch size ({mask.shape[0]}) does not match image batch size ({batch_size}). Using first mask for all images.")
                mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        raw_regions = []
        
        if fixed_mask_frame_index > 0:
            mask_idx_to_use = min(fixed_mask_frame_index - 1, mask.shape[0] - 1)
            fixed_m = mask[mask_idx_to_use]
            
            nonzero = torch.nonzero(fixed_m, as_tuple=False)
            if len(nonzero) == 0:
                y_min, x_min = 0, 0
                y_max, x_max = fixed_m.shape
            else:
                y_min, x_min = nonzero.min(dim=0).values
                y_max, x_max = nonzero.max(dim=0).values
                y_min, x_min = y_min.item(), x_min.item()
                y_max, x_max = y_max.item() + 1, x_max.item() + 1
            
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(fixed_m.shape[0], y_max + padding)
            x_max = min(fixed_m.shape[1], x_max + padding)
            
            width = x_max - x_min
            height = y_max - y_min

            for i in range(batch_size):
                raw_regions.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "width": width,
                    "height": height
                })
        else:
            for i in range(batch_size):
                m = mask[i]
                nonzero = torch.nonzero(m, as_tuple=False)
                
                if len(nonzero) == 0:
                    y_min, x_min = 0, 0
                    y_max, x_max = m.shape
                else:
                    y_min, x_min = nonzero.min(dim=0).values
                    y_max, x_max = nonzero.max(dim=0).values
                    y_min, x_min = y_min.item(), x_min.item()
                    y_max, x_max = y_max.item() + 1, x_max.item() + 1
                
                y_min = max(0, y_min - padding)
                x_min = max(0, x_min - padding)
                y_max = min(m.shape[0], y_max + padding)
                x_max = min(m.shape[1], x_max + padding)
                
                width = x_max - x_min
                height = y_max - y_min
                
                raw_regions.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "width": width,
                    "height": height
                })
        
        smoothed_regions = []
        if fixed_mask_frame_index == 0 and interpolation_window_size > 0:
            for i in range(batch_size):
                window_start = max(0, i - interpolation_window_size // 2)
                window_end = min(batch_size, i + interpolation_window_size // 2 + 1)
                
                window_x_mins = [r["x_min"] for r in raw_regions[window_start:window_end]]
                window_y_mins = [r["y_min"] for r in raw_regions[window_start:window_end]]
                window_widths = [r["width"] for r in raw_regions[window_start:window_end]]
                window_heights = [r["height"] for r in raw_regions[window_start:window_end]]

                avg_x_min = sum(window_x_mins) / len(window_x_mins)
                avg_y_min = sum(window_y_mins) / len(window_y_mins)
                avg_width = sum(window_widths) / len(window_widths)
                avg_height = sum(window_heights) / len(window_heights)

                blended_x_min = round(avg_x_min)
                blended_y_min = round(avg_y_min)
                blended_width = round(avg_width)
                blended_height = round(avg_height)

                blended_width = max(1, blended_width)
                blended_height = max(1, blended_height)

                smoothed_regions.append({
                    "x_min": int(blended_x_min),
                    "y_min": int(blended_y_min),
                    "width": int(blended_width),
                    "height": int(blended_height)
                })
        else:
            smoothed_regions = raw_regions

        final_regions = []
        for r in smoothed_regions:
            target_width = ((r["width"] + divide_by - 1) // divide_by) * divide_by
            target_height = ((r["height"] + divide_by - 1) // divide_by) * divide_by
            final_regions.append({
                "x_min": r["x_min"],
                "y_min": r["y_min"],
                "width": r["width"],
                "height": r["height"],
                "target_width": target_width, 
                "target_height": target_height
            })

        max_target_width = max(r["target_width"] for r in final_regions)
        max_target_height = max(r["target_height"] for r in final_regions)
        
        # Step 3: Perform cropping with uniform size
        cropped_images = []
        cropped_masks = []
        crop_data = [] # Инициализация здесь
        
        for i in range(batch_size):
            img = images[i]
            msk = mask[i] 
            r = final_regions[i]

            center_x = r["x_min"] + r["width"] // 2
            center_y = r["y_min"] + r["height"] // 2
            
            crop_x = center_x - max_target_width // 2
            crop_y = center_y - max_target_height // 2

            crop_x = max(0, min(crop_x, img.shape[1] - max_target_width))
            crop_y = max(0, min(crop_y, img.shape[0] - max_target_height))

            if img.shape[1] < max_target_width:
                crop_x = (img.shape[1] - max_target_width) // 2
                crop_x = max(0, crop_x)
            if img.shape[0] < max_target_height:
                crop_y = (img.shape[0] - max_target_height) // 2
                crop_y = max(0, crop_y)

            crop_x_end = crop_x + max_target_width
            crop_y_end = crop_y + max_target_height

            cropped_img = img[crop_y:crop_y_end, crop_x:crop_x_end, :]
            cropped_msk = msk[crop_y:crop_y_end, crop_x:crop_x_end]
            
            if cropped_img.shape[0] < max_target_height or cropped_img.shape[1] < max_target_width:
                pad_bottom = max_target_height - cropped_img.shape[0]
                pad_right = max_target_width - cropped_img.shape[1]
                
                cropped_img = torch.nn.functional.pad(
                    cropped_img.unsqueeze(0).permute(0, 3, 1, 2),
                    (0, pad_right, 0, pad_bottom),
                    mode='constant',
                    value=0
                ).permute(0, 2, 3, 1).squeeze(0)
                
                cropped_msk = torch.nn.functional.pad(
                    cropped_msk.unsqueeze(0).unsqueeze(0),
                    (0, pad_right, 0, pad_bottom),
                    mode='constant',
                    value=0
                ).squeeze(0).squeeze(0)
            
            cropped_images.append(cropped_img)
            cropped_masks.append(cropped_msk)
            
            initial_crop_width = max_target_width
            initial_crop_height = max_target_height

            crop_data.append({
                "original_x": r["x_min"],
                "original_y": r["y_min"],
                "original_width": r["width"],
                "original_height": r["height"],
                "crop_x": crop_x,
                "crop_y": crop_y,
                "initial_crop_width": initial_crop_width,
                "initial_crop_height": initial_crop_height,
                "final_crop_width": initial_crop_width, 
                "final_crop_height": initial_crop_height
            })
        
        cropped_images = torch.stack(cropped_images)
        cropped_masks = torch.stack(cropped_masks)
        
        # Step 4: Apply resolution scaling based on max_resolution (for the smaller side) and divide_by
        current_width, current_height = max_target_width, max_target_height
        
        min_dim_current = min(current_width, current_height)
        
        new_width, new_height = current_width, current_height

        if min_dim_current > max_resolution:
            scale_ratio = max_resolution / min_dim_current
            
            new_width = int(current_width * scale_ratio)
            new_height = int(current_height * scale_ratio)

            new_width = (new_width // divide_by) * divide_by
            new_height = (new_height // divide_by) * divide_by
            
            new_width = max(divide_by, new_width)
            new_height = max(divide_by, new_height)

            if new_width != current_width or new_height != current_height:
                cropped_images = cropped_images.permute(0, 3, 1, 2)
                cropped_images = F.interpolate(
                    cropped_images,
                    size=(new_height, new_width),
                    mode='bilinear',
                    align_corners=False
                )
                cropped_images = cropped_images.permute(0, 2, 3, 1)
                
                cropped_masks = cropped_masks.unsqueeze(1)
                cropped_masks = F.interpolate(
                    cropped_masks,
                    size=(new_height, new_width),
                    mode='nearest'
                )
                cropped_masks = cropped_masks.squeeze(1)
                
                for data_item in crop_data:
                    data_item["final_crop_width"] = new_width
                    data_item["final_crop_height"] = new_height
            else:
                for data_item in crop_data:
                    data_item["final_crop_width"] = new_width
                    data_item["final_crop_height"] = new_height
        else:
            for data_item in crop_data:
                data_item["final_crop_width"] = current_width
                data_item["final_crop_height"] = current_height

        return (
            cropped_images, 
            cropped_masks, 
            crop_data,
            new_width,
            new_height
        )


class TSRestoreFromCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "blur": ("INT", {"default": 64, "min": 0, "max": 256, "step": 1}), # Параметр размытия
                "blur_type": (["Gaussian", "Box"], {"default": "Gaussian"}), # Новый параметр
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "image/processing"

    def restore(self, original_images, cropped_images, crop_data, blur, blur_type): # Добавили blur_type
        restored_images = []
        batch_size = original_images.shape[0]
        
        current_device = cropped_images.device 

        for i in range(batch_size):
            orig_img = original_images[i].clone()
            cropped_img = cropped_images[i]
            data = crop_data[i]
            
            # Масштабируем cropped_img обратно до размеров, которые были ДО масштабирования max_resolution.
            if cropped_img.shape[0] != data["initial_crop_height"] or cropped_img.shape[1] != data["initial_crop_width"]:
                cropped_img = cropped_img.permute(2, 0, 1).unsqueeze(0)
                cropped_img = F.interpolate(
                    cropped_img,
                    size=(data["initial_crop_height"], data["initial_crop_width"]),
                    mode='bilinear',
                    align_corners=False
                )
                cropped_img = cropped_img.squeeze(0).permute(1, 2, 0)
            
            paste_x_in_crop = data["original_x"] - data["crop_x"]
            paste_y_in_crop = data["original_y"] - data["crop_y"]

            paste_width = data["original_width"]
            paste_height = data["original_height"]

            cropped_img_h, cropped_img_w, _ = cropped_img.shape
            
            temp_canvas = torch.zeros((paste_height, paste_width, 3), device=current_device, dtype=cropped_img.dtype)
            
            src_y_start = max(0, paste_y_in_crop)
            src_x_start = max(0, paste_x_in_crop)
            src_y_end = min(cropped_img_h, paste_y_in_crop + paste_height)
            src_x_end = min(cropped_img_w, paste_x_in_crop + paste_width)
            
            dest_y_start = max(0, -paste_y_in_crop)
            dest_x_start = max(0, -paste_x_in_crop)
            
            if src_y_end > src_y_start and src_x_end > src_x_start:
                temp_canvas[
                    dest_y_start : dest_y_start + (src_y_end - src_y_start),
                    dest_x_start : dest_x_start + (src_x_end - src_x_start),
                    :
                ] = cropped_img[src_y_start:src_y_end, src_x_start:src_x_end, :]
            
            region_to_paste = temp_canvas 
            
            if region_to_paste.shape[0] != paste_height or region_to_paste.shape[1] != paste_width:
                print(f"Warning: region_to_paste size mismatch after re-cropping. Expected ({paste_height}, {paste_width}), got ({region_to_paste.shape[0]}, {region_to_paste.shape[1]}). Resizing...")
                region_to_paste = region_to_paste.permute(2,0,1).unsqueeze(0)
                region_to_paste = F.interpolate(
                    region_to_paste,
                    size=(paste_height, paste_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).permute(1,2,0)
            
            # --- Логика смешивания (блендинга) ---
            if blur > 0:
                base_mask_for_blending = torch.ones((paste_height, paste_width), 
                                                    device=current_device, dtype=torch.float32)

                shrink_amount = blur 
                
                shrunk_mask_canvas = torch.zeros_like(base_mask_for_blending, device=current_device)
                
                inner_y_start = max(0, shrink_amount)
                inner_x_start = max(0, shrink_amount)
                inner_y_end = max(inner_y_start, min(paste_height, paste_height - shrink_amount))
                inner_x_end = max(inner_x_start, min(paste_width, paste_width - shrink_amount))

                if inner_y_end > inner_y_start and inner_x_end > inner_x_start:
                    shrunk_mask_canvas[inner_y_start:inner_y_end, inner_x_start:inner_x_end] = 1.0
                
                # Вызываем нужную функцию размытия в зависимости от blur_type
                if blur_type == "Gaussian":
                    blurred_mask = gaussian_blur_mask(shrunk_mask_canvas.unsqueeze(0), blur, current_device).squeeze(0)
                elif blur_type == "Box":
                    blurred_mask = box_blur_mask(shrunk_mask_canvas.unsqueeze(0), blur, current_device).squeeze(0)
                else:
                    # На случай, если каким-то образом передан неизвестный тип, не размываем.
                    print(f"Warning: Unknown blur_type '{blur_type}'. No blur applied for this frame.")
                    blurred_mask = shrunk_mask_canvas # Нет размытия
                
                original_region_for_blend = orig_img[
                    data["original_y"] : data["original_y"] + data["original_height"],
                    data["original_x"] : data["original_x"] + data["original_width"],
                    :
                ]
                
                blurred_mask_expanded = blurred_mask.unsqueeze(-1)
                
                blended_region = (region_to_paste * blurred_mask_expanded) + \
                                 (original_region_for_blend * (1.0 - blurred_mask_expanded))
                
                orig_img[
                    data["original_y"] : data["original_y"] + data["original_height"],
                    data["original_x"] : data["original_x"] + data["original_width"],
                    :
                ] = blended_region
            else: # blur равно 0, просто жесткая вставка
                orig_img[
                    data["original_y"] : data["original_y"] + data["original_height"],
                    data["original_x"] : data["original_x"] + data["original_width"],
                    :
                ] = region_to_paste
            
            restored_images.append(orig_img)
        
        return (torch.stack(restored_images),)


NODE_CLASS_MAPPINGS = {
    "TSCropToMask": TSCropToMask,
    "TSRestoreFromCrop": TSRestoreFromCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSCropToMask": "TS Crop To Mask (Interpolated)",
    "TSRestoreFromCrop": "TS Restore From Crop",
}