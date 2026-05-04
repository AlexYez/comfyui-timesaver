import torch
import torch.nn.functional as F
import kornia.filters # РўСЂРµР±СѓРµС‚СЃСЏ СѓСЃС‚Р°РЅРѕРІРєР°: pip install kornia

class TS_FilmGrain: 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "force_gpu": ("BOOLEAN", {
                    "default": True,
                    "display": "toggle"
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0,  # РћСЃС‚Р°РІР»СЏРµРј, С…РѕСЂРѕС€РёР№ Р±Р°Р·РѕРІС‹Р№ СЂР°Р·РјРµСЂ
                    "min": 0.1, 
                    "max": 5.0, 
                    "step": 0.1, 
                    "display": "slider"
                }),
                "grain_intensity": ("FLOAT", {
                    "default": 0.065, # РЈРІРµР»РёС‡РµРЅРѕ РЅР° 30% (Р±С‹Р»Рѕ 0.05 * 1.3 = 0.065)
                    "min": 0.0, 
                    "max": 0.5, 
                    "step": 0.005, 
                    "display": "slider"
                }),
                "grain_speed": ("FLOAT", {
                    "default": 0.5, # РћСЃС‚Р°РІР»СЏРµРј, С…РѕСЂРѕС€РёР№ Р±Р°Р»Р°РЅСЃ РґР»СЏ РІРёРґРµРѕ
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "display": "slider"
                }),
                "grain_softness": ("FLOAT", { # РђРєС‚РёРІРёСЂСѓРµРј РґР»СЏ Р±РѕР»РµРµ РѕСЂРіР°РЅРёС‡РЅРѕРіРѕ Р·РµСЂРЅР°
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1, 
                    "display": "slider"
                }),
                "color_grain_strength": ("FLOAT", { # РђРєС‚РёРІРёСЂСѓРµРј РґР»СЏ Р»РµРіРєРѕРіРѕ С†РІРµС‚РѕРІРѕРіРѕ РѕС‚С‚РµРЅРєР°
                    "default": 0.15, # РЈРјРµСЂРµРЅРЅС‹Р№ С†РІРµС‚РЅРѕР№ СЌС„С„РµРєС‚
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "display": "slider"
                }),
                "mid_tone_grain_bias": ("FLOAT", { 
                    "default": 0.5, # РћСЃС‚Р°РІР»СЏРµРј, С…РѕСЂРѕС€Рѕ РґР»СЏ Kodak-РїРѕРґРѕР±РЅРѕРіРѕ Р·РµСЂРЅР°
                    "min": 0.01, 
                    "max": 0.99, 
                    "step": 0.01, 
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xFFFFFFFFFFFFFFFF 
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "Image Adjustments/Grain"

    def _generate_octave_noise(self, batch_dim, target_h, target_w, channels, scale_factor, current_seed, device, dtype):
        """
        Р“РµРЅРµСЂРёСЂСѓРµС‚ РѕРґРЅСѓ РѕРєС‚Р°РІСѓ РіР°СѓСЃСЃРѕРІР° С€СѓРјР° РґР»СЏ Р±Р°С‚С‡Р° (batch_dim=B) РёР»Рё РѕРґРЅРѕРіРѕ СЌРєР·РµРјРїР»СЏСЂР° (batch_dim=1)
        Рё СѓРІРµР»РёС‡РёРІР°РµС‚ РµРµ РґРѕ С†РµР»РµРІРѕРіРѕ СЂР°Р·СЂРµС€РµРЅРёСЏ.
        """
        noise_h_octave = max(1, int(target_h / scale_factor))
        noise_w_octave = max(1, int(target_w / scale_factor))
        
        torch.manual_seed(current_seed) 
        base_noise = torch.randn(batch_dim, noise_h_octave, noise_w_octave, channels, device=device, dtype=dtype)
        
        upscaled_noise = F.interpolate(
            base_noise.permute(0, 3, 1, 2), 
            size=(target_h, target_w), 
            mode='bicubic', 
            align_corners=False
        )
        del base_noise 

        upscaled_noise = upscaled_noise.permute(0, 2, 3, 1)

        return upscaled_noise

    def apply_grain(self, images, force_gpu, grain_size, grain_intensity, grain_speed, 
                    grain_softness, color_grain_strength, mid_tone_grain_bias, seed):
        
        B, H, W, C = images.shape 
        
        target_device = images.device
        target_dtype = images.dtype

        if force_gpu:
            if torch.cuda.is_available():
                target_device = 'cuda'
                target_dtype = torch.float16 
            else:
                target_device = 'cpu'
                target_dtype = torch.float32 
                print("Warning: 'force_gpu' is enabled but CUDA is not available. Falling back to CPU/float32.")
        else: 
            if images.device.type == 'cpu':
                target_device = 'cpu'
                target_dtype = torch.float32 

        if images.device != target_device or images.dtype != target_dtype:
            images = images.to(target_device, dtype=target_dtype, non_blocking=True)
            print(f"Info: Images moved to {images.device} and converted to {images.dtype}")
        
        device = images.device 
        compute_dtype = images.dtype 
        
        print(f"--- TS_FilmGrain Debug Info ---")
        print(f"Final compute device: {device}")
        print(f"Final compute dtype: {compute_dtype}")
        print(f"Input images shape: {images.shape}")
        if device.type == 'cuda':
            print(f"CUDA current device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        print(f"--- End Debug Info ---")


        noise_octaves = [
            (grain_size * 1.0, 0.6), 
            (grain_size * 2.0, 0.3), 
            (grain_size * 4.0, 0.1)  
        ]
        
        octave_norm_factor = sum(w for _, w in noise_octaves)

        static_seed_offset = 0 
        static_mono_noise_sum = torch.zeros(1, H, W, 1, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            static_mono_noise_sum.add_(weight * self._generate_octave_noise(
                1, H, W, 1, scale, seed + static_seed_offset + j * 100, device, compute_dtype
            ))
        static_mono_noise_sum.div_(octave_norm_factor) 

        dynamic_seed_offset = 12345 
        dynamic_mono_noise_sum = torch.zeros(B, H, W, 1, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            dynamic_mono_noise_sum.add_(weight * self._generate_octave_noise(
                B, H, W, 1, scale, seed + dynamic_seed_offset + j * 100, device, compute_dtype
            ))
        dynamic_mono_noise_sum.div_(octave_norm_factor)

        mixed_mono_noise = (1.0 - grain_speed) * static_mono_noise_sum + grain_speed * dynamic_mono_noise_sum
        del static_mono_noise_sum, dynamic_mono_noise_sum 

        static_color_noise_sum = torch.zeros(1, H, W, C, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            static_color_noise_sum.add_(weight * self._generate_octave_noise(
                1, H, W, C, scale, seed + static_seed_offset + j * 1000 + 1, device, compute_dtype
            ))
        static_color_noise_sum.div_(octave_norm_factor)

        dynamic_color_noise_sum = torch.zeros(B, H, W, C, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            dynamic_color_noise_sum.add_(weight * self._generate_octave_noise(
                B, H, W, C, scale, seed + dynamic_seed_offset + j * 1000 + 1, device, compute_dtype
            ))
        dynamic_color_noise_sum.div_(octave_norm_factor)
        
        mixed_color_noise = (1.0 - grain_speed) * static_color_noise_sum + grain_speed * dynamic_color_noise_sum
        del static_color_noise_sum, dynamic_color_noise_sum 

        luminosity_map = images.permute(0, 3, 1, 2).mean(dim=1, keepdim=True) 
        
        clamped_bias = torch.tensor(mid_tone_grain_bias, device=device, dtype=compute_dtype)
        
        shadow_exponent = 2.0 / clamped_bias 
        highlight_exponent = 2.0 / (1.0 - clamped_bias) 
        
        raw_mod_factor = (luminosity_map.pow(shadow_exponent)) * ((1.0 - luminosity_map).pow(highlight_exponent))
        del luminosity_map
        
        max_val_approx = (torch.tensor(mid_tone_grain_bias, dtype=torch.float32).pow(shadow_exponent.to(torch.float32))) * \
                         ((torch.tensor(1.0 - mid_tone_grain_bias, dtype=torch.float32)).pow(highlight_exponent.to(torch.float32)))
        max_val_approx = max_val_approx.clamp(min=1e-6).to(device=device, dtype=compute_dtype)
        
        luminosity_modulated_factor = raw_mod_factor.div_(max_val_approx) 
        del raw_mod_factor 

        luminosity_modulated_factor = luminosity_modulated_factor.permute(0, 2, 3, 1)

        mono_grain_effect = mixed_mono_noise.mul_(grain_intensity * 0.5).mul_(luminosity_modulated_factor)
        del mixed_mono_noise, luminosity_modulated_factor 

        color_grain_effect = mixed_color_noise.mul_(grain_intensity * color_grain_strength * 0.25)
        del mixed_color_noise 

        final_grain = mono_grain_effect + color_grain_effect 
        del mono_grain_effect, color_grain_effect 
        
        if grain_softness > 0:
            kernel_size = int(grain_softness * 2) * 2 + 1 
            sigma = grain_softness 
            
            final_grain = kornia.filters.gaussian_blur2d(
                final_grain.permute(0, 3, 1, 2), 
                (kernel_size, kernel_size), 
                (sigma, sigma)
            ).permute(0, 2, 3, 1) 

        output_images = images.add_(final_grain) 
        output_images.clamp_(0.0, 1.0) 
        
        del final_grain 

        return (output_images,)

# --- ComfyUI Integration ---
NODE_CLASS_MAPPINGS = {
    "TS_FilmGrain": TS_FilmGrain 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FilmGrain": "TS Film Grain" 
}
