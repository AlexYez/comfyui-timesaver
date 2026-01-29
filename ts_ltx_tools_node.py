import torch
import comfy.utils
import comfy.model_management
import sys
import traceback

class TS_LTX_FirstLastFrame:
    """
    Control the first and last frames of an LTX video generation context.
    
    Changes:
    - Added 'enable_last_frame' boolean switch.
    - Removed 'middle_frames' support.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "first_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_last_frame": ("BOOLEAN", {"default": True, "label_on": "Enable Last Frame", "label_off": "Disable Last Frame"}),
            },
            "optional": {
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def log_step(self, message: str):
        """Custom logging with color for visibility."""
        print(f"\033[96m[TS_LTX_FirstLastFrame]\033[0m {message}")

    def execute(
        self, 
        vae, 
        latent: dict, 
        first_strength: float, 
        last_strength: float, 
        enable_last_frame: bool,
        first_image: torch.Tensor = None, 
        last_image: torch.Tensor = None
    ):
        self.log_step("Execution started.")
        
        try:
            # 1. Setup and device handling
            device = comfy.model_management.get_torch_device()
            
            # Deep copy latent to avoid mutating upstream data
            samples = latent["samples"].clone()
            
            # Latent Shape: [Batch, Channels, Frames, Height, Width]
            batch_size, channels, latent_frames, latent_height, latent_width = samples.shape
            self.log_step(f"Input Latent Shape: {samples.shape}")

            # 2. Calculate pixel dimensions based on VAE formula
            # formula is typically (offset, h_scale, w_scale)
            _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
            target_px_width = latent_width * width_scale_factor
            target_px_height = latent_height * height_scale_factor
            
            self.log_step(f"Target Pixel Resolution: {target_px_width}x{target_px_height}")

            # 3. Initialize Noise Mask
            # 1.0 = Allow Noise (Generation), 0.0 = Keep Source (Conditioning)
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"].clone()
            else:
                noise_mask = torch.ones(
                    (batch_size, 1, latent_frames, 1, 1),
                    dtype=torch.float32,
                    device=samples.device,
                )

            # 4. Helper for encoding and embedding
            def embed_frames(encoded_latent, start_idx, strength):
                # Calculate valid range
                enc_frames = encoded_latent.shape[2]
                end_idx = min(latent_frames, start_idx + enc_frames)
                src_end = end_idx - start_idx
                
                if src_end <= 0:
                    return

                # Embed samples
                samples[:, :, start_idx:end_idx] = encoded_latent[:, :, :src_end]
                
                # Apply Mask (min logic to preserve strongest constraint)
                mask_val = 1.0 - strength
                current_mask_slice = noise_mask[:, :, start_idx:end_idx, :, :]
                new_mask_val = torch.full_like(current_mask_slice, mask_val)
                
                noise_mask[:, :, start_idx:end_idx, :, :] = torch.minimum(current_mask_slice, new_mask_val)
                self.log_step(f"Embedded frame(s) at index {start_idx}:{end_idx} with strength {strength}")

            with torch.no_grad():
                # 5. Process First Frame
                if first_image is not None and first_strength > 0.0:
                    self.log_step("Processing First Frame...")
                    # Prepare image on device
                    img_first = first_image.to(device)
                    # Encode
                    enc_first = self._encode_image(vae, img_first, target_px_height, target_px_width)
                    # Embed at start (index 0)
                    embed_frames(enc_first, 0, first_strength)

                # 6. Process Last Frame
                if enable_last_frame and last_image is not None and last_strength > 0.0:
                    self.log_step("Processing Last Frame...")
                    # Prepare image on device
                    img_last = last_image.to(device)
                    # Encode
                    enc_last = self._encode_image(vae, img_last, target_px_height, target_px_width)
                    
                    # Calculate position: start at (total_frames - encoded_frames)
                    last_enc_frames = enc_last.shape[2]
                    start_idx = max(0, latent_frames - last_enc_frames)
                    
                    # Embed at end
                    embed_frames(enc_last, start_idx, last_strength)
                elif not enable_last_frame:
                    self.log_step("Last Frame processing disabled by user.")

            self.log_step("Execution finished successfully.")
            return ({"samples": samples, "noise_mask": noise_mask},)

        except Exception as e:
            print(f"\033[91m[Error in TS_LTX_FirstLastFrame]\033[0m {str(e)}")
            traceback.print_exc()
            return (latent,) # Return original on error to prevent crash

    def _encode_image(self, vae, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resizes and encodes an image into latent space.
        Args:
            image: Tensor [B, H, W, C] (ComfyUI Format)
        Returns:
            Latent: Tensor [B, C, T, H, W]
        """
        # Ensure BCHW for resizing
        samples = image.movedim(-1, 1) 
        
        # Resize if dimensions don't match
        if samples.shape[2] != target_h or samples.shape[3] != target_w:
            samples = comfy.utils.common_upscale(
                samples, 
                target_w, 
                target_h, 
                "bilinear", 
                "center"
            )
        
        # Convert back to BHWC for VAE encoding (Standard Comfy VAE behavior)
        samples = samples.movedim(1, -1)
        
        # Ensure only RGB (strip alpha if exists)
        samples = samples[:, :, :, :3]
        
        # Encode
        return vae.encode(samples)

# ----------------------------------------------------------------------------
# NODE REGISTRATION
# ----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "TS_LTX_FirstLastFrame": TS_LTX_FirstLastFrame
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_LTX_FirstLastFrame": "LTX First/Last Frame"
}