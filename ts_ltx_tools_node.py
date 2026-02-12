import torch
import comfy.utils
import comfy.model_management
import sys
import traceback

class TS_LTX_FirstLastFrame:
    """
    Control the first and last frames of an LTX video generation context.
    
    Features:
    - Independent First/Last frame control.
    - Robust handling of optional inputs: if 'last_image' is missing or invalid, 
      it gracefully falls back to single-frame (first-frame only) mode, 
      ignoring the 'enable_last_frame' toggle.
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
            _, height_scale_factor, width_scale_factor = vae.downscale_index_formula
            target_px_width = latent_width * width_scale_factor
            target_px_height = latent_height * height_scale_factor
            
            self.log_step(f"Target Pixel Resolution: {target_px_width}x{target_px_height}")

            # 3. Initialize Noise Mask
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"].clone()
            else:
                noise_mask = torch.ones(
                    (batch_size, 1, latent_frames, 1, 1),
                    dtype=torch.float32,
                    device=samples.device,
                )

            # 4. Helper for embedding
            def embed_frames(encoded_latent, start_idx, strength):
                enc_frames = encoded_latent.shape[2]
                end_idx = min(latent_frames, start_idx + enc_frames)
                src_end = end_idx - start_idx
                
                if src_end <= 0:
                    return

                # Embed samples
                samples[:, :, start_idx:end_idx] = encoded_latent[:, :, :src_end]
                
                # Apply Mask
                mask_val = 1.0 - strength
                current_mask_slice = noise_mask[:, :, start_idx:end_idx, :, :]
                new_mask_val = torch.full_like(current_mask_slice, mask_val)
                
                # Use minimum to preserve the strongest constraint (lowest value)
                noise_mask[:, :, start_idx:end_idx, :, :] = torch.minimum(current_mask_slice, new_mask_val)
                self.log_step(f"Embedded frame(s) at index {start_idx}:{end_idx} with strength {strength}")

            with torch.no_grad():
                # ----------------------------------------------------------------
                # PROCESS FIRST FRAME
                # ----------------------------------------------------------------
                # Check validation strictly
                is_first_valid = first_image is not None and isinstance(first_image, torch.Tensor)
                
                if is_first_valid and first_strength > 0.0:
                    self.log_step("Processing First Frame...")
                    img_first = first_image.to(device)
                    enc_first = self._encode_image(vae, img_first, target_px_height, target_px_width)
                    embed_frames(enc_first, 0, first_strength)
                elif first_image is None:
                    self.log_step("No First Frame input provided. Skipping.")

                # ----------------------------------------------------------------
                # PROCESS LAST FRAME
                # ----------------------------------------------------------------
                # Validate inputs for Last Frame
                # It is valid ONLY if input is provided AND it is a Tensor
                is_last_valid = last_image is not None and isinstance(last_image, torch.Tensor)
                
                if enable_last_frame:
                    if is_last_valid and last_strength > 0.0:
                        self.log_step("Processing Last Frame...")
                        img_last = last_image.to(device)
                        enc_last = self._encode_image(vae, img_last, target_px_height, target_px_width)
                        
                        # Calculate position: start at (total_frames - encoded_frames)
                        last_enc_frames = enc_last.shape[2]
                        start_idx = max(0, latent_frames - last_enc_frames)
                        
                        embed_frames(enc_last, start_idx, last_strength)
                    else:
                        # Logic for why it was skipped despite being enabled
                        if last_image is None:
                            self.log_step("Last Frame enabled but NO input image provided. Skipping.")
                        elif not isinstance(last_image, torch.Tensor):
                            self.log_step("Last Frame enabled but input is NOT a valid tensor. Skipping.")
                        elif last_strength <= 0.0:
                            self.log_step("Last Frame enabled but strength is 0. Skipping.")
                else:
                    self.log_step("Last Frame processing disabled by user switch.")

            self.log_step("Execution finished successfully.")
            return ({"samples": samples, "noise_mask": noise_mask},)

        except Exception as e:
            print(f"\033[91m[Error in TS_LTX_FirstLastFrame]\033[0m {str(e)}")
            traceback.print_exc()
            return (latent,)

    def _encode_image(self, vae, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        Resizes and encodes an image into latent space.
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
        
        # Convert back to BHWC for VAE encoding
        samples = samples.movedim(1, -1)
        
        # Ensure only RGB
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
    "TS_LTX_FirstLastFrame": "TS LTX First/Last Frame"
}