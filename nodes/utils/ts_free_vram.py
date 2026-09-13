"""TS Free VRAM — take a model off the card at a chosen point in the graph.

Some steps refuse to share the card. The clearest case is the LTX 2.5 video VAE:
it declares `disable_offload = True`, so ComfyUI must hold it fully resident, and
its decode asks for a very large reservation. With the diffusion model still
staged next to it the two together exceed the card — and on Windows the driver
does not raise an OOM, it spills into shared system memory. Nothing fails; the
decode simply crawls at 100% GPU while VRAM sits at the ceiling, and because
there is no OOM ComfyUI never falls back to tiled decoding either.

This node solves that by position rather than by patching anything: it passes its
input through untouched, and while it runs it frees the card. Placed between the
sampler and the decode, "while it runs" is exactly the gap between the two.

⚠️ THE WIRE TYPE IS FREE, THE POSITION IS NOT. The passthrough is a wildcard so
the node fits on any link, but ComfyUI runs a node when its output is needed —
so only a link that is consumed AFTER the heavy step frees anything useful. On a
MODEL or CONDITIONING link feeding a sampler the node runs BEFORE sampling, where
there is nothing to free yet.

The same trick is used inside TS Latent Upscale, which cannot hold the upscaler
and the diffusion model at once either.
"""

from __future__ import annotations

import logging

import comfy.model_management
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_free_vram")
LOG_PREFIX = "[TS Free VRAM]"

_MB = 1024 * 1024


def _free_megabytes() -> float:
    """Free VRAM on the sampling device, in MB (0 when there is no such device)."""
    try:
        device = comfy.model_management.get_torch_device()
        if comfy.model_management.is_device_cpu(device):
            return 0.0
        return comfy.model_management.get_free_memory(device) / _MB
    except Exception:  # noqa: BLE001 - reporting must never break a graph
        return 0.0


class TS_FreeVRAM(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_FreeVRAM",
            display_name="TS Free VRAM",
            category="TS/Utils",
            description=(
                "Take models off the GPU at a chosen point in the graph. Passes its input "
                "through untouched, so put it on a link that is used right before a step "
                "that needs the whole card — a heavy VAE decode, an upscale, a second sampler."
            ),
            inputs=[
                IO.AnyType.Input(
                    "passthrough",
                    tooltip=(
                        "Anything at all — it comes back out unchanged. Its only job is to "
                        "place this node in time: ComfyUI runs a node when its output is "
                        "needed, so put this on a link consumed AFTER the heavy step you "
                        "want to free memory for (a latent going into VAE Decode is the "
                        "usual spot). On a link feeding a sampler it would run too early."
                    ),
                ),
                IO.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Which model to unload. Connect the same MODEL the sampler used and "
                        "only that one is taken off the card. Leave it empty to unload "
                        "everything currently loaded."
                    ),
                ),
            ],
            outputs=[
                IO.AnyType.Output(
                    display_name="passthrough",
                    tooltip="Exactly what came in, unchanged.",
                )
            ],
            search_aliases=[
                "free vram", "unload model", "vram", "out of memory", "offload", "purge vram",
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, passthrough=None, model=None) -> float:
        """Never cacheable.

        ⚠️ This node's whole value is a SIDE EFFECT, and its output equals its
        input. With an honest fingerprint ComfyUI would serve the passthrough
        from cache on every later run and skip execution entirely — the first
        run would be fast and every one after it slow again, with nothing in the
        log to explain why. NaN never equals itself, so the cache never hits.
        """
        return float("nan")

    @classmethod
    def execute(cls, passthrough=None, model=None) -> IO.NodeOutput:
        before = _free_megabytes()

        if model is not None:
            comfy.model_management.unload_model_and_clones(model)
            what = "the connected model"
        else:
            comfy.model_management.unload_all_models()
            what = "every loaded model"
        comfy.model_management.soft_empty_cache()

        after = _free_megabytes()
        logger.info(
            "%s Unloaded %s. Free VRAM %.0f MB -> %.0f MB (%+.0f MB).",
            LOG_PREFIX, what, before, after, after - before,
        )
        return IO.NodeOutput(passthrough)


NODE_CLASS_MAPPINGS = {"TS_FreeVRAM": TS_FreeVRAM}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_FreeVRAM": "TS Free VRAM"}
