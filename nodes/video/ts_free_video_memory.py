"""TS Free Video Memory — explicit GC + CUDA cache flush between heavy video steps.

node_id: TS_Free_Video_Memory
"""

import gc
import logging

import torch

from comfy_api.latest import IO

logger = logging.getLogger("comfyui_timesaver.ts_free_video_memory")
LOG_PREFIX = "[TS Free Video Memory]"


class TS_Free_Video_Memory(IO.ComfyNode):
    """Explicit memory cleanup pass-through node."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Free_Video_Memory",
            display_name="TS Free Video Memory",
            category="TS/Video",
            inputs=[
                IO.Image.Input("images"),
                IO.Combo.Input("aggressive_cleanup", options=["disable", "enable"], default="disable"),
                IO.Combo.Input("report_memory", options=["disable", "enable"], default="enable"),
            ],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(cls, images, aggressive_cleanup="disable", report_memory="enable") -> IO.NodeOutput:
        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info("%s Before cleanup: %.2fGB allocated, %.2fGB reserved", LOG_PREFIX, allocated, reserved)

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive_cleanup == "enable":
                torch.cuda.synchronize()
                if hasattr(torch.cuda, 'caching_allocator_delete_caches'):
                    try:
                        torch.cuda.caching_allocator_delete_caches()
                    except Exception as e:
                        logger.warning(
                            "%s torch.cuda.caching_allocator_delete_caches() failed (older PyTorch?): %s",
                            LOG_PREFIX,
                            e,
                        )

        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info("%s After cleanup: %.2fGB allocated, %.2fGB reserved", LOG_PREFIX, allocated, reserved)

        return IO.NodeOutput(images)


NODE_CLASS_MAPPINGS = {"TS_Free_Video_Memory": TS_Free_Video_Memory}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Free_Video_Memory": "TS Free Video Memory"}
