"""TS Free Video Memory — explicit GC + CUDA cache flush between heavy video steps.

node_id: TS_Free_Video_Memory
"""

import gc
import logging

import torch

logger = logging.getLogger("comfyui_timesaver.ts_free_video_memory")
LOG_PREFIX = "[TS Free Video Memory]"


class TS_Free_Video_Memory:
    """Explicit memory cleanup pass-through node."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "aggressive_cleanup": (["disable", "enable"], {"default": "disable"}),
            "report_memory": (["disable", "enable"], {"default": "enable"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cleanup_memory"
    CATEGORY = "TS/Video"

    def cleanup_memory(self, images, aggressive_cleanup="disable", report_memory="enable"):
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

        return (images,)


NODE_CLASS_MAPPINGS = {"TS_Free_Video_Memory": TS_Free_Video_Memory}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Free_Video_Memory": "TS Free Video Memory"}
