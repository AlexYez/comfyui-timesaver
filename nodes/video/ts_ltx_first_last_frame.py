import logging
import traceback

import torch

from comfy_api.latest import IO
from comfy_extras.nodes_lt import LTXVAddGuide

logger = logging.getLogger("comfyui_timesaver.ts_ltx_first_last_frame")
LOG_PREFIX = "[TS LTX First/Last Frame]"


class TS_LTX_FirstLastFrame(IO.ComfyNode):
    """
    Apply native LTX guide conditioning for the first and optional last frame.

    The node intentionally mirrors chaining one or two LTXVAddGuide nodes:
    - first image -> frame_idx = 0
    - last image  -> frame_idx = -1
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTX_FirstLastFrame",
            display_name="TS LTX First/Last Frame",
            category="TS/Video",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Vae.Input("vae"),
                IO.Latent.Input("latent"),
                IO.Float.Input("first_strength", default=0.7, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("last_strength", default=0.7, min=0.0, max=1.0, step=0.01),
                IO.Image.Input("first_image", optional=True),
                IO.Image.Input("last_image", optional=True),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                IO.Latent.Output(display_name="latent"),
            ],
        )

    @staticmethod
    def _is_valid_image(value) -> bool:
        return value is not None and isinstance(value, torch.Tensor)

    @staticmethod
    def _log(message: str) -> None:
        logger.info("%s %s", LOG_PREFIX, message)

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

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent: dict,
        first_strength: float,
        last_strength: float,
        first_image: torch.Tensor = None,
        last_image: torch.Tensor = None,
    ) -> IO.NodeOutput:
        try:
            positive_out = positive
            negative_out = negative
            latent_out = cls._clone_latent(latent)

            has_first = cls._is_valid_image(first_image)
            has_last = cls._is_valid_image(last_image)

            if has_first and has_last:
                cls._log("First frame to last frame")
            elif has_first:
                cls._log("First frame only")
            elif has_last:
                cls._log("Last frame only")
            else:
                cls._log("Text to video (no frames)")

            if not has_first and not has_last:
                return IO.NodeOutput(positive_out, negative_out, latent_out)

            if has_first and first_strength > 0.0:
                positive_out, negative_out, latent_out = cls._unpack_node_output(
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
                cls._log("First frame input is not a valid image tensor. Skipping first-frame guide.")

            if has_last and last_strength > 0.0:
                positive_out, negative_out, latent_out = cls._unpack_node_output(
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
                cls._log("Last frame input is not a valid image tensor. Skipping last-frame guide.")

            return IO.NodeOutput(positive_out, negative_out, latent_out)
        except Exception as exc:
            logger.error("%s %s\n%s", LOG_PREFIX, exc, traceback.format_exc())
            return IO.NodeOutput(positive, negative, latent)


NODE_CLASS_MAPPINGS = {
    "TS_LTX_FirstLastFrame": TS_LTX_FirstLastFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_LTX_FirstLastFrame": "TS LTX First/Last Frame",
}
