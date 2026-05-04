import traceback

import torch

from comfy_extras.nodes_lt import LTXVAddGuide


class TS_LTX_FirstLastFrame:
    """
    Apply native LTX guide conditioning for the first and optional last frame.

    The node intentionally mirrors chaining one or two LTXVAddGuide nodes:
    - first image -> frame_idx = 0
    - last image  -> frame_idx = -1
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "first_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    @staticmethod
    def _is_valid_image(value) -> bool:
        return value is not None and isinstance(value, torch.Tensor)

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS_LTX_FirstLastFrame] {message}")

    @staticmethod
    def _clone_latent(latent: dict) -> dict:
        cloned = {"samples": latent["samples"].clone()}
        if "noise_mask" in latent and latent["noise_mask"] is not None:
            cloned["noise_mask"] = latent["noise_mask"].clone()
        for key, value in latent.items():
            if key in cloned:
                continue
            cloned[key] = value
        return cloned

    @staticmethod
    def _unpack_node_output(node_output):
        if hasattr(node_output, "result"):
            return node_output.result
        return node_output

    def execute(
        self,
        positive,
        negative,
        vae,
        latent: dict,
        first_strength: float,
        last_strength: float,
        first_image: torch.Tensor = None,
        last_image: torch.Tensor = None,
    ):
        try:
            positive_out = positive
            negative_out = negative
            latent_out = self._clone_latent(latent)

            has_first = self._is_valid_image(first_image)
            has_last = self._is_valid_image(last_image)

            if has_first and has_last:
                self._log("First frame to last frame")
            elif has_first:
                self._log("First frame only")
            elif has_last:
                self._log("Last frame only")
            else:
                self._log("Text to video (no frames)")

            if not has_first and not has_last:
                return (positive_out, negative_out, latent_out)

            if has_first and first_strength > 0.0:
                positive_out, negative_out, latent_out = self._unpack_node_output(
                    LTXVAddGuide.execute(
                        positive=positive_out,
                        negative=negative_out,
                        vae=vae,
                        latent=latent_out,
                        image=first_image,
                        frame_idx=0,
                        strength=first_strength,
                    )
                )
            elif first_image is not None and not has_first:
                self._log("First frame input is not a valid image tensor. Skipping first-frame guide.")

            if has_last and last_strength > 0.0:
                positive_out, negative_out, latent_out = self._unpack_node_output(
                    LTXVAddGuide.execute(
                        positive=positive_out,
                        negative=negative_out,
                        vae=vae,
                        latent=latent_out,
                        image=last_image,
                        frame_idx=-1,
                        strength=last_strength,
                    )
                )
            elif last_image is not None and not has_last:
                self._log("Last frame input is not a valid image tensor. Skipping last-frame guide.")

            return (positive_out, negative_out, latent_out)
        except Exception as exc:
            print(f"[TS_LTX_FirstLastFrame] {exc}")
            traceback.print_exc()
            return (positive, negative, latent)


NODE_CLASS_MAPPINGS = {
    "TS_LTX_FirstLastFrame": TS_LTX_FirstLastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_LTX_FirstLastFrame": "TS LTX First/Last Frame",
}
