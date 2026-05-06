"""TS Crop To Mask — crop a batch of images around a mask region with optional smoothing.

node_id: TSCropToMask
"""

import logging

import torch
import torch.nn.functional as F

import comfy

from comfy_api.latest import IO

logger = logging.getLogger("comfyui_timesaver.ts_crop_to_mask")
LOG_PREFIX = "[TS Crop To Mask]"


_CropData = IO.Custom("CROP_DATA")


class TSCropToMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TSCropToMask",
            display_name="TS Crop To Mask",
            category="TS/Image",
            inputs=[
                IO.Image.Input("images"),
                IO.Mask.Input("mask"),
                IO.Int.Input("padding", default=64, min=0, max=320),
                IO.Int.Input("divide_by", default=32, min=1, max=64),
                IO.Int.Input("max_resolution", default=720, min=320, max=2048, step=1),
                IO.Int.Input("fixed_mask_frame_index", default=0, min=0, max=9999, step=1),
                IO.Int.Input("interpolation_window_size", default=0, min=0, max=100, step=1),
                IO.Boolean.Input("force_gpu", default=True),
                IO.Boolean.Input("fixed_crop_size", default=False),
                IO.Int.Input("fixed_width", default=1024, min=64, max=4096, step=8),
                IO.Int.Input("fixed_height", default=1024, min=64, max=4096, step=8),
            ],
            outputs=[
                IO.Image.Output(display_name="cropped_images"),
                IO.Mask.Output(display_name="cropped_masks"),
                _CropData.Output(display_name="crop_data"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, images, mask, padding, divide_by, max_resolution, fixed_mask_frame_index, interpolation_window_size, force_gpu, fixed_crop_size=False, fixed_width=1024, fixed_height=1024) -> IO.NodeOutput:
        target_device = comfy.model_management.get_torch_device() if force_gpu and torch.cuda.is_available() else torch.device("cpu")
        logger.info("%s Using device %s", LOG_PREFIX, target_device)

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
            logger.info("%s Single mask repeated for %d images.", LOG_PREFIX, batch_size)
        elif batch_size == 1 and mask.shape[0] > 1:
            logger.info(
                "%s Single image provided with batch mask. Will use specified or first mask frame for crop calculation.",
                LOG_PREFIX,
            )
        else:
            logger.warning(
                "%s Mask batch size (%d) and image batch size (%d) mismatch. Falling back to first mask for all images.",
                LOG_PREFIX,
                mask.shape[0],
                batch_size,
            )
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        all_masks_are_solid = True
        for i in range(mask.shape[0]):
            m = mask[i]
            if m.numel() > 0:
                if m.min() != m.max():
                    all_masks_are_solid = False
                    break

        if all_masks_are_solid:
            logger.info("%s All input masks are empty/solid. Skipping crop.", LOG_PREFIX)
            crop_data = []
            for i in range(batch_size):
                crop_data.append({
                    "original_x": 0, "original_y": 0,
                    "original_width": original_img_w, "original_height": original_img_h,
                    "crop_x": 0, "crop_y": 0,
                    "initial_crop_width": original_img_w, "initial_crop_height": original_img_h,
                    "final_crop_width": original_img_w, "final_crop_height": original_img_h,
                })
            return IO.NodeOutput(images, mask, crop_data, original_img_w, original_img_h)

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

            if original_img_w < max_target_width:
                crop_x = max(0, (original_img_w - max_target_width) // 2)
            if original_img_h < max_target_height:
                crop_y = max(0, (original_img_h - max_target_height) // 2)

            crop_x_end, crop_y_end = crop_x + max_target_width, crop_y + max_target_height
            cropped_img = img[crop_y:crop_y_end, crop_x:crop_x_end, :]
            cropped_msk = mask[min(i, mask.shape[0] - 1)][crop_y:crop_y_end, crop_x:crop_x_end]

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
                "final_crop_width": max_target_width, "final_crop_height": max_target_height,
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
            for d in crop_data:
                d["final_crop_width"], d["final_crop_height"] = new_width, new_height

        return IO.NodeOutput(cropped_images, cropped_masks, crop_data, new_width, new_height)


NODE_CLASS_MAPPINGS = {"TSCropToMask": TSCropToMask}
NODE_DISPLAY_NAME_MAPPINGS = {"TSCropToMask": "TS Crop To Mask"}
