import torch
import torch.nn.functional as F
import sys
import traceback

class TS_SmartImageBatch:
    """
    TS Smart Image Batch Node
    Intelligently batches two images. 
    - Takes image1 (required) and image2 (optional).
    - If image2 is missing or invalid, passes image1.
    - If image2 exists, matches its dimensions (H, W) to image1 and creates a batch.
    """

    def __init__(self):
        self.device = torch.device("cpu") # Default, will update based on input

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "smart_batch"
    CATEGORY = "TS Nodes/Image"
    OUTPUT_NODE = False

    def log_step(self, message, is_error=False):
        """
        Custom logging with ANSI colors for better visibility in ComfyUI console.
        """
        prefix = "\033[96m[TS_SmartImageBatch]\033[0m" # Cyan
        if is_error:
            prefix = "\033[91m[TS_SmartImageBatch ERROR]\033[0m" # Red
        
        print(f"{prefix} {message}")

    def ensure_tensor(self, data):
        """Validates if data is a torch Tensor."""
        return isinstance(data, torch.Tensor)

    def smart_batch(self, image1, image2=None):
        try:
            self.log_step("Execution started.")
            
            # 1. Validate Primary Input
            if not self.ensure_tensor(image1):
                raise ValueError(f"image1 must be a torch.Tensor, got {type(image1)}")

            self.log_step(f"Input image1 shape: {image1.shape} | Device: {image1.device} | Dtype: {image1.dtype}")

            # 2. Check Secondary Input (Optional)
            # Conditions to ignore image2: None, not a tensor, or empty tensor
            if image2 is None:
                self.log_step("image2 is None. Passing through image1.")
                return (image1,)
            
            if not self.ensure_tensor(image2):
                self.log_step(f"image2 received but is not a Tensor ({type(image2)}). Ignoring and passing image1.")
                return (image1,)

            if image2.numel() == 0:
                 self.log_step("image2 is an empty tensor. Passing through image1.")
                 return (image1,)

            self.log_step(f"Input image2 shape: {image2.shape} | Device: {image2.device}")

            # 3. Smart Processing & Batching
            # ComfyUI Image Tensor Format: [Batch, Height, Width, Channels] (BHWC)
            
            target_h = image1.shape[1]
            target_w = image1.shape[2]
            target_c = image1.shape[3]
            
            processed_image2 = image2

            # Check Channels match
            if image2.shape[3] != target_c:
                self.log_step(f"Channel mismatch! Img1: {target_c}, Img2: {image2.shape[3]}. Cannot batch safely.", is_error=True)
                # Fallback to returning image1 to prevent crash, or could implement channel conversion. 
                # For safety/strictness, we return image1.
                return (image1,)

            # Check Spatial Dimensions (H, W)
            if image2.shape[1] != target_h or image2.shape[2] != target_w:
                self.log_step(f"Dimension mismatch detected. Resizing image2 to match image1 ({target_w}x{target_h})...")
                
                # Permute to BCHW for PyTorch Interpolation
                # image2: [B, H, W, C] -> [B, C, H, W]
                img2_bchw = image2.permute(0, 3, 1, 2)
                
                # Resize using bilinear interpolation
                img2_resized = F.interpolate(
                    img2_bchw, 
                    size=(target_h, target_w), 
                    mode="bilinear", 
                    align_corners=False
                )
                
                # Permute back to BHWC
                # [B, C, H, W] -> [B, H, W, C]
                processed_image2 = img2_resized.permute(0, 2, 3, 1)
                self.log_step(f"image2 resized to: {processed_image2.shape}")

            # 4. Concatenation
            # Ensure devices match
            if processed_image2.device != image1.device:
                processed_image2 = processed_image2.to(image1.device)

            output_image = torch.cat((image1, processed_image2), dim=0)
            
            self.log_step(f"Batching complete. Final Output Shape: {output_image.shape}")
            
            return (output_image,)

        except Exception as e:
            self.log_step(f"CRITICAL ERROR: {str(e)}", is_error=True)
            traceback.print_exc()
            # Emergency fallback: return input to not break the workflow graph completely
            return (image1,)

# Registration
NODE_CLASS_MAPPINGS = {
    "TS_SmartImageBatch": TS_SmartImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_SmartImageBatch": "TS Smart Image Batch"
}