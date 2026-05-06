import logging

import torch
import torch.nn.functional as F
import kornia.filters

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_film_grain")
LOG_PREFIX = "[TS Film Grain]"


class TS_FilmGrain(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_FilmGrain",
            display_name="TS Film Grain",
            category="TS/Image",
            inputs=[
                IO.Image.Input("images"),
                IO.Boolean.Input("force_gpu", default=True),
                IO.Float.Input("grain_size", default=1.0, min=0.1, max=5.0, step=0.1, display_mode=IO.NumberDisplay.slider),
                IO.Float.Input("grain_intensity", default=0.065, min=0.0, max=0.5, step=0.005, display_mode=IO.NumberDisplay.slider),
                IO.Float.Input("grain_speed", default=0.5, min=0.0, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider),
                IO.Float.Input("grain_softness", default=0.5, min=0.0, max=2.0, step=0.1, display_mode=IO.NumberDisplay.slider),
                IO.Float.Input("color_grain_strength", default=0.15, min=0.0, max=1.0, step=0.01, display_mode=IO.NumberDisplay.slider),
                IO.Float.Input("mid_tone_grain_bias", default=0.5, min=0.01, max=0.99, step=0.01, display_mode=IO.NumberDisplay.slider),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @staticmethod
    def _generate_octave_noise(batch_dim, target_h, target_w, channels, scale_factor, current_seed, device, dtype):
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

    @classmethod
    def execute(cls, images, force_gpu, grain_size, grain_intensity, grain_speed,
                grain_softness, color_grain_strength, mid_tone_grain_bias, seed) -> IO.NodeOutput:
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
                logger.warning("%s force_gpu enabled but CUDA is not available; falling back to CPU/float32.", LOG_PREFIX)
        else:
            if images.device.type == 'cpu':
                target_device = 'cpu'
                target_dtype = torch.float32

        if images.device != target_device or images.dtype != target_dtype:
            images = images.to(target_device, dtype=target_dtype, non_blocking=True)

        device = images.device
        compute_dtype = images.dtype

        noise_octaves = [
            (grain_size * 1.0, 0.6),
            (grain_size * 2.0, 0.3),
            (grain_size * 4.0, 0.1)
        ]

        octave_norm_factor = sum(w for _, w in noise_octaves)

        static_seed_offset = 0
        static_mono_noise_sum = torch.zeros(1, H, W, 1, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            static_mono_noise_sum.add_(weight * cls._generate_octave_noise(
                1, H, W, 1, scale, seed + static_seed_offset + j * 100, device, compute_dtype
            ))
        static_mono_noise_sum.div_(octave_norm_factor)

        dynamic_seed_offset = 12345
        dynamic_mono_noise_sum = torch.zeros(B, H, W, 1, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            dynamic_mono_noise_sum.add_(weight * cls._generate_octave_noise(
                B, H, W, 1, scale, seed + dynamic_seed_offset + j * 100, device, compute_dtype
            ))
        dynamic_mono_noise_sum.div_(octave_norm_factor)

        mixed_mono_noise = (1.0 - grain_speed) * static_mono_noise_sum + grain_speed * dynamic_mono_noise_sum
        del static_mono_noise_sum, dynamic_mono_noise_sum

        static_color_noise_sum = torch.zeros(1, H, W, C, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            static_color_noise_sum.add_(weight * cls._generate_octave_noise(
                1, H, W, C, scale, seed + static_seed_offset + j * 1000 + 1, device, compute_dtype
            ))
        static_color_noise_sum.div_(octave_norm_factor)

        dynamic_color_noise_sum = torch.zeros(B, H, W, C, device=device, dtype=compute_dtype)
        for j, (scale, weight) in enumerate(noise_octaves):
            dynamic_color_noise_sum.add_(weight * cls._generate_octave_noise(
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

        # Non-mutating add+clamp: when `images` is the caller's input tensor
        # (no device/dtype conversion happened above), an in-place add_ would
        # poison the workflow cache and re-apply grain on every cache hit.
        output_images = (images + final_grain).clamp(0.0, 1.0)

        del final_grain

        return IO.NodeOutput(output_images)


NODE_CLASS_MAPPINGS = {
    "TS_FilmGrain": TS_FilmGrain
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FilmGrain": "TS Film Grain"
}
