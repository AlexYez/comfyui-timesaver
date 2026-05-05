"""TS Restore From Crop — paste a processed crop back into the original frame with edge blur.

node_id: TSRestoreFromCrop
"""

import logging

import torch
import torch.nn.functional as F

import comfy

logger = logging.getLogger("comfyui_timesaver.ts_restore_from_crop")
LOG_PREFIX = "[TS Restore From Crop]"


def _gaussian_blur_mask(mask_tensor_batch, blur_amount, device):
    """Apply a separable Gaussian blur to a batch of [B, H, W] masks."""
    if blur_amount <= 0:
        return mask_tensor_batch

    sigma = float(blur_amount) / 3.0
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device, dtype=torch.float32)
    gauss = torch.exp((-ax ** 2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()

    input_tensor_for_conv = mask_tensor_batch.unsqueeze(1)

    horizontal_kernel = kernel_1d.view(1, 1, 1, -1)
    blurred_h = F.conv2d(input_tensor_for_conv, horizontal_kernel, padding=(0, kernel_size // 2), groups=1)

    vertical_kernel = kernel_1d.view(1, 1, -1, 1)
    blurred_v = F.conv2d(blurred_h, vertical_kernel, padding=(kernel_size // 2, 0), groups=1)

    return blurred_v.squeeze(1)


def _box_blur_mask(mask_tensor_batch, blur_amount, device):
    """Apply a separable box blur to a batch of [B, H, W] masks."""
    if blur_amount <= 0:
        return mask_tensor_batch

    kernel_size = int(blur_amount * 2) + 1
    if blur_amount > 0:
        kernel_size = max(3, kernel_size)
    else:
        return mask_tensor_batch

    kernel_1d = torch.ones(kernel_size, device=device, dtype=torch.float32) / kernel_size

    input_tensor_for_conv = mask_tensor_batch.unsqueeze(1)

    horizontal_kernel = kernel_1d.view(1, 1, 1, -1)
    blurred_h = F.conv2d(input_tensor_for_conv, horizontal_kernel, padding=(0, kernel_size // 2), groups=1)

    vertical_kernel = kernel_1d.view(1, 1, -1, 1)
    blurred_v = F.conv2d(blurred_h, vertical_kernel, padding=(kernel_size // 2, 0), groups=1)

    return blurred_v.squeeze(1)


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
    CATEGORY = "TS/Image"

    def restore(self, original_images, cropped_images, crop_data, blur, blur_type, force_gpu):
        target_device = comfy.model_management.get_torch_device() if force_gpu and torch.cuda.is_available() else torch.device("cpu")
        logger.info("%s Using device %s", LOG_PREFIX, target_device)
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
                    # Edge-aware blur: don't soften edges that touch the image border.
                    left_edge = 0 if p_x <= blur else blur
                    top_edge = 0 if p_y <= blur else blur
                    right_edge = 0 if (p_x + p_w) >= (orig_img_w - blur) else blur
                    bottom_edge = 0 if (p_y + p_h) >= (orig_img_h - blur) else blur

                    shrunk_mask_canvas = torch.zeros((p_h, p_w), device=target_device)

                    m_y_s, m_x_s = top_edge, left_edge
                    m_y_e, m_x_e = max(m_y_s, p_h - bottom_edge), max(m_x_s, p_w - right_edge)

                    if m_y_e > m_y_s and m_x_e > m_x_s:
                        shrunk_mask_canvas[m_y_s:m_y_e, m_x_s:m_x_e] = 1.0

                    if blur_type == "Gaussian":
                        blurred_mask = _gaussian_blur_mask(shrunk_mask_canvas.unsqueeze(0), blur, target_device).squeeze(0)
                    else:
                        blurred_mask = _box_blur_mask(shrunk_mask_canvas.unsqueeze(0), blur, target_device).squeeze(0)

                    sliced_mask = blurred_mask[local_y_s:local_y_e, local_x_s:local_x_e].unsqueeze(-1)
                    original_part = orig_img[y_start_orig:y_end_orig, x_start_orig:x_end_orig, :]

                    orig_img[y_start_orig:y_end_orig, x_start_orig:x_end_orig, :] = (final_paste_region * sliced_mask) + (original_part * (1.0 - sliced_mask))
                else:
                    orig_img[y_start_orig:y_end_orig, x_start_orig:x_end_orig, :] = final_paste_region

            restored_images.append(orig_img)

        return (torch.stack(restored_images),)


NODE_CLASS_MAPPINGS = {"TSRestoreFromCrop": TSRestoreFromCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"TSRestoreFromCrop": "TS Restore From Crop"}
