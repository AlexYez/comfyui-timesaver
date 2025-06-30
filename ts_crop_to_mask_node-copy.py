import torch
import comfy
import torch.nn.functional as F

class TSCropToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 64, "min": 0, "max": 320}),
                "divide_by": ("INT", {"default": 32, "min": 1, "max": 64}),
                "max_resolution": ("INT", {"default": 720, "min": 32, "max": 2048, "step": 32}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "CROP_DATA", "INT", "INT")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "crop_data", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def crop(self, images, mask, padding, divide_by, max_resolution):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        if images.shape[0] != mask.shape[0]:
            mask = mask[0].unsqueeze(0).repeat(images.shape[0], 1, 1)
        
        # Step 1: Find all crop regions
        regions = []
        for i in range(images.shape[0]):
            m = mask[i]
            nonzero = torch.nonzero(m, as_tuple=False)
            
            if len(nonzero) == 0:
                # Full image if mask is empty
                y_min, x_min = 0, 0
                y_max, x_max = m.shape
            else:
                y_min, x_min = nonzero.min(dim=0).values
                y_max, x_max = nonzero.max(dim=0).values
                y_min, x_min = y_min.item(), x_min.item()
                y_max, x_max = y_max.item() + 1, x_max.item() + 1
            
            # Apply padding
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(m.shape[0], y_max + padding)
            x_max = min(m.shape[1], x_max + padding)
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Calculate divisible size
            target_width = ((width + divide_by - 1) // divide_by) * divide_by
            target_height = ((height + divide_by - 1) // divide_by) * divide_by
            
            regions.append({
                "x_min": x_min,
                "y_min": y_min,
                "width": width,
                "height": height,
                "target_width": target_width,
                "target_height": target_height
            })
        
        # Step 2: Find max dimensions across batch
        max_width = max(r["target_width"] for r in regions)
        max_height = max(r["target_height"] for r in regions)
        
        # Step 3: Perform cropping with uniform size
        cropped_images = []
        cropped_masks = []
        crop_data = []
        
        for i in range(images.shape[0]):
            img = images[i]
            msk = mask[i]
            r = regions[i]
            
            # Calculate center of original region
            center_x = r["x_min"] + r["width"] // 2
            center_y = r["y_min"] + r["height"] // 2
            
            # Calculate crop coordinates centered on original region
            crop_x = max(0, center_x - max_width // 2)
            crop_y = max(0, center_y - max_height // 2)
            crop_x_end = min(img.shape[1], crop_x + max_width)
            crop_y_end = min(img.shape[0], crop_y + max_height)
            
            # Adjust if near edges
            if crop_x_end - crop_x < max_width:
                crop_x = max(0, crop_x_end - max_width)
            if crop_y_end - crop_y < max_height:
                crop_y = max(0, crop_y_end - max_height)
            
            # Crop image
            cropped_img = img[crop_y:crop_y_end, crop_x:crop_x_end, :]
            
            # Crop mask
            cropped_msk = msk[crop_y:crop_y_end, crop_x:crop_x_end]
            
            # Pad if necessary
            if cropped_img.shape[0] < max_height or cropped_img.shape[1] < max_width:
                pad_bottom = max_height - cropped_img.shape[0]
                pad_right = max_width - cropped_img.shape[1]
                
                # Pad image
                cropped_img = torch.nn.functional.pad(
                    cropped_img.unsqueeze(0).permute(0, 3, 1, 2),
                    (0, pad_right, 0, pad_bottom),
                    mode='constant',
                    value=0
                ).permute(0, 2, 3, 1).squeeze(0)
                
                # Pad mask
                cropped_msk = torch.nn.functional.pad(
                    cropped_msk.unsqueeze(0).unsqueeze(0),
                    (0, pad_right, 0, pad_bottom),
                    mode='constant',
                    value=0
                ).squeeze(0).squeeze(0)
            
            cropped_images.append(cropped_img)
            cropped_masks.append(cropped_msk)
            
            # Store data for restoration
            crop_data.append({
                "original_x": r["x_min"],
                "original_y": r["y_min"],
                "original_width": r["width"],
                "original_height": r["height"],
                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": max_width,
                "crop_height": max_height
            })
        
        cropped_images = torch.stack(cropped_images)
        cropped_masks = torch.stack(cropped_masks)
        
        # Step 4: Apply max resolution downscaling if needed
        min_side = min(max_width, max_height)
        final_width, final_height = max_width, max_height
        
        if min_side > max_resolution:
            # Calculate scale ratio
            ratio = max_resolution / min_side
            new_width = int(round(max_width * ratio))
            new_height = int(round(max_height * ratio))
            
            # Align dimensions to divide_by
            new_width = max(divide_by, (new_width // divide_by) * divide_by)
            new_height = max(divide_by, (new_height // divide_by) * divide_by)
            
            # Resize images
            cropped_images = cropped_images.permute(0, 3, 1, 2)  # [B, C, H, W]
            cropped_images = F.interpolate(
                cropped_images,
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            )
            cropped_images = cropped_images.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            # Resize masks
            cropped_masks = cropped_masks.unsqueeze(1)
            cropped_masks = F.interpolate(
                cropped_masks,
                size=(new_height, new_width),
                mode='nearest'
            )
            cropped_masks = cropped_masks.squeeze(1)
            
            final_width, final_height = new_width, new_height
        
        return (
            cropped_images, 
            cropped_masks, 
            crop_data,
            final_width,
            final_height
        )


class TSRestoreFromCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "image/processing"

    def restore(self, original_images, cropped_images, crop_data):
        restored_images = []
        
        for i in range(original_images.shape[0]):
            orig = original_images[i].clone()
            crop = cropped_images[i]
            data = crop_data[i]
            
            # Upscale if needed (to original crop size)
            if crop.shape[0] != data["crop_height"] or crop.shape[1] != data["crop_width"]:
                crop = crop.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                crop = F.interpolate(
                    crop,
                    size=(data["crop_height"], data["crop_width"]),
                    mode='bilinear',
                    align_corners=False
                )
                crop = crop.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            
            # Calculate position within the crop
            paste_x = data["original_x"] - data["crop_x"]
            paste_y = data["original_y"] - data["crop_y"]
            
            # Calculate actual region size (may be smaller near edges)
            paste_width = min(data["original_width"], crop.shape[1] - paste_x)
            paste_height = min(data["original_height"], crop.shape[0] - paste_y)
            
            # Extract the relevant region from the crop
            region_to_paste = crop[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width, :]
            
            # Paste back to original position
            orig[data["original_y"]:data["original_y"]+paste_height, 
                 data["original_x"]:data["original_x"]+paste_width, :] = region_to_paste
            
            restored_images.append(orig)
        
        return (torch.stack(restored_images),)


NODE_CLASS_MAPPINGS = {
    "TSCropToMask": TSCropToMask,
    "TSRestoreFromCrop": TSRestoreFromCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSCropToMask": "TS Crop To Mask",
    "TSRestoreFromCrop": "TS Restore From Crop",
}