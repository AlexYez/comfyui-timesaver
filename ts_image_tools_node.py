from typing import Optional
import time

import torch
import comfy.utils


class TS_ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image Batch to Image List] {message}")

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    def execute(self, image: torch.Tensor):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None, returning empty list.")
            return ([],)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        images = [image[i : i + 1, ...] for i in range(image.shape[0])]

        if images:
            self._log(f"Output list length={len(images)}")
            self._log_tensor("Output[0]", images[0])

        return (images,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return "none"
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}"


class TS_ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image List to Image Batch] {message}")

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    @staticmethod
    def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 3 or 4 dims, got {image.ndim}")
        return image

    def execute(self, images):
        if images is None:
            self._log("Input list is None.")
            return ()

        if not isinstance(images, list):
            images = [images]

        self._log(f"Input list length={len(images)}")

        if len(images) == 0:
            self._log("Input list is empty.")
            return ()

        valid_images = [img for img in images if img is not None]
        if len(valid_images) == 0:
            self._log("All input images are None.")
            return ()

        normalized = []
        for idx, img in enumerate(valid_images):
            if not isinstance(img, torch.Tensor):
                raise ValueError(f"Image {idx} is not a torch.Tensor: {type(img)}")
            norm = self._ensure_bhwc(img)
            normalized.append(norm)

        base = normalized[0]
        target_h, target_w = base.shape[1], base.shape[2]
        target_c = min(img.shape[3] for img in normalized)
        target_dtype = base.dtype
        target_device = base.device

        self._log_tensor("Input[0]", base)
        self._log(f"Target size={target_w}x{target_h} channels={target_c}")

        resized = []
        for idx, img in enumerate(normalized):
            if img.device != target_device:
                self._log(f"Image {idx} moved to {target_device}")
                img = img.to(target_device)
            if img.dtype != target_dtype:
                img = img.to(target_dtype)
            if img.shape[1] != target_h or img.shape[2] != target_w:
                self._log(
                    f"Image {idx} resized from {img.shape[2]}x{img.shape[1]} to {target_w}x{target_h}"
                )
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), target_w, target_h, "lanczos", "center"
                ).movedim(1, -1)
            if img.shape[3] != target_c:
                self._log(f"Image {idx} channels trimmed to {target_c}")
                img = img[..., :target_c]
            resized.append(img)

        batch = torch.cat(resized, dim=0)
        self._log_tensor("Output", batch)
        return (batch,)

    @classmethod
    def IS_CHANGED(cls, images) -> str:
        if images is None:
            return "none"
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return "empty"
        shapes = []
        for img in images:
            if isinstance(img, torch.Tensor):
                shapes.append(tuple(img.shape))
        try:
            sums = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    sums.append(float(img.mean()))
            return f"{shapes}_{sums}"
        except Exception:
            return f"{shapes}"


class TS_ImageBatchCut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "first_cut": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "last_cut": ("INT", {"default": 0, "min": 0, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image Batch Cut] {message}")

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    @staticmethod
    def _normalize_cut(value: int) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

    def execute(self, image: torch.Tensor, first_cut: int, last_cut: int):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None, returning as-is.")
            return (image,)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        total = int(image.shape[0])
        cut_start = self._normalize_cut(first_cut)
        cut_end = self._normalize_cut(last_cut)

        self._log(f"Total frames={total} first_cut={cut_start} last_cut={cut_end}")

        if cut_start == 0 and cut_end == 0:
            self._log("No cut applied.")
            return (image,)

        if cut_start + cut_end >= total:
            self._log("Cut exceeds batch length, returning empty batch.")
            empty = image[:0, ...]
            self._log_tensor("Output", empty)
            return (empty,)

        trimmed = image[cut_start : total - cut_end, ...]
        self._log_tensor("Output", trimmed)
        return (trimmed,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor, first_cut: int, last_cut: int) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return f"none_{first_cut}_{last_cut}"
        try:
            return (
                f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}_{first_cut}_{last_cut}"
            )
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}_{first_cut}_{last_cut}"


class TS_GetImageMegapixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("megapixels",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Get Image Megapixels] {message}")

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    def execute(self, image: torch.Tensor):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None.")
            return (0.0,)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        megapixels = float(width * height) / 1_000_000.0

        self._log(f"Computed megapixels={megapixels}")
        return (megapixels,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return "none"
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}"


class TS_GetImageSizeSide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "large_side": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Large Side",
                        "label_off": "Small Side",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Get Image Size Side] {message}")

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    def execute(self, image: torch.Tensor, large_side: bool):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None.")
            return (0,)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        size = max(height, width) if large_side else min(height, width)

        self._log(f"Computed size={size} (large_side={large_side})")
        return (size,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor, large_side: bool) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return f"none_{large_side}"
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}_{large_side}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}_{large_side}"


class TS_ImagePromptInjector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt_graph": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image Prompt Injector] {message}")

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    @staticmethod
    def _normalize_prompt(prompt: Optional[str]) -> str:
        if prompt is None:
            return ""
        if not isinstance(prompt, str):
            return str(prompt)
        return prompt

    @staticmethod
    def _is_link(value) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 2

    @staticmethod
    def _get_meta_title(node: dict) -> str:
        meta = node.get("_meta", {})
        if isinstance(meta, dict):
            title = meta.get("title", "")
            if isinstance(title, str):
                return title
        return ""

    @classmethod
    def _is_positive_prompt_node(cls, node: dict) -> bool:
        title = cls._get_meta_title(node).lower()
        if "negative" in title:
            return False
        return "positive" in title

    @staticmethod
    def _is_text_encoder_node(node: dict) -> bool:
        class_type = node.get("class_type", "")
        if not isinstance(class_type, str):
            return False
        return class_type.startswith("CLIPTextEncode")

    @classmethod
    def _select_target_nodes(cls, prompt_graph: dict) -> list[str]:
        candidates = []
        positives = []
        for node_id, node in prompt_graph.items():
            if not isinstance(node, dict):
                continue
            if not cls._is_text_encoder_node(node):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if "text" not in inputs:
                continue
            candidates.append(str(node_id))
            if cls._is_positive_prompt_node(node):
                positives.append(str(node_id))
        return positives if positives else candidates

    @staticmethod
    def _get_node(prompt_graph: dict, node_id):
        if prompt_graph is None:
            return None
        key = str(node_id)
        if key in prompt_graph:
            return prompt_graph[key]
        if node_id in prompt_graph:
            return prompt_graph[node_id]
        return None

    @staticmethod
    def _iter_linked_inputs(inputs: dict):
        for value in inputs.values():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                yield value[0]

    @classmethod
    def _find_text_encoders_from(cls, prompt_graph: dict, start_ids: list) -> set[str]:
        results: set[str] = set()
        stack = list(start_ids)
        visited = set()
        while stack:
            node_id = stack.pop()
            key = str(node_id)
            if key in visited:
                continue
            visited.add(key)
            node = cls._get_node(prompt_graph, node_id)
            if not isinstance(node, dict):
                continue
            if cls._is_text_encoder_node(node) and isinstance(node.get("inputs", {}), dict):
                if "text" in node["inputs"]:
                    results.add(key)
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict):
                for linked_id in cls._iter_linked_inputs(inputs):
                    stack.append(linked_id)
        return results

    @classmethod
    def _find_sampler_roots(cls, prompt_graph: dict) -> tuple[list, list]:
        positives = []
        negatives = []
        for node in prompt_graph.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if "positive" in inputs and cls._is_link(inputs.get("positive")):
                positives.append(inputs["positive"][0])
            if "negative" in inputs and cls._is_link(inputs.get("negative")):
                negatives.append(inputs["negative"][0])
        return positives, negatives

    @classmethod
    def _inject_prompt_into_graph(cls, prompt_graph, prompt_text: str) -> None:
        if prompt_graph is None:
            cls._log("prompt_graph is None, metadata not updated.")
            return
        if isinstance(prompt_graph, list):
            prompt_graph = prompt_graph[0] if prompt_graph else None
        if not isinstance(prompt_graph, dict):
            cls._log(f"prompt_graph invalid type={type(prompt_graph)}, metadata not updated.")
            return

        positive_roots, negative_roots = cls._find_sampler_roots(prompt_graph)
        positive_ids = cls._find_text_encoders_from(prompt_graph, positive_roots)
        negative_ids = cls._find_text_encoders_from(prompt_graph, negative_roots)

        if positive_ids:
            target_ids = positive_ids - negative_ids
            if not target_ids:
                target_ids = positive_ids
        else:
            target_ids = set(cls._select_target_nodes(prompt_graph))

        if not target_ids:
            cls._log("No text encoder nodes found in prompt_graph.")
            return

        updated = 0
        for node_id in target_ids:
            node = cls._get_node(prompt_graph, node_id)
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            text_value = inputs.get("text")
            if text_value == "" or cls._is_link(text_value):
                inputs["text"] = prompt_text
                updated += 1

        if updated == 0:
            cls._log("No linked text inputs updated.")
        else:
            cls._log(f"Updated prompt text in {updated} node(s).")

    def execute(self, image: torch.Tensor, prompt: str, prompt_graph=None):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None, returning as-is.")
            return (image,)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim not in (3, 4):
            raise ValueError(
                f"Expected IMAGE with 3 or 4 dims [H,W,C] or [B,H,W,C], got {image.ndim}"
            )

        prompt_text = self._normalize_prompt(prompt)
        if prompt_text == "":
            self._log("Prompt is empty, metadata not updated.")
            return (image,)

        self._log(f"Injecting prompt length={len(prompt_text)}")
        self._inject_prompt_into_graph(prompt_graph, prompt_text)
        return (image,)

    @classmethod
    def IS_CHANGED(
        cls, image: torch.Tensor, prompt: str, prompt_graph=None
    ) -> str:
        prompt_text = cls._normalize_prompt(prompt)
        if isinstance(image, torch.Tensor):
            shape = tuple(image.shape)
        else:
            shape = "none"
        return f"ts_prompt_injector_{shape}_{len(prompt_text)}_{time.time_ns()}"


NODE_CLASS_MAPPINGS = {
    "TS_ImageBatchToImageList": TS_ImageBatchToImageList,
    "TS_ImageListToImageBatch": TS_ImageListToImageBatch,
    "TS_ImageBatchCut": TS_ImageBatchCut,
    "TS_GetImageMegapixels": TS_GetImageMegapixels,
    "TS_GetImageSizeSide": TS_GetImageSizeSide,
    "TS_ImagePromptInjector": TS_ImagePromptInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageBatchToImageList": "TS Image Batch to Image List",
    "TS_ImageListToImageBatch": "TS Image List to Image Batch",
    "TS_ImageBatchCut": "TS Image Batch Cut",
    "TS_GetImageMegapixels": "TS Get Image Megapixels",
    "TS_GetImageSizeSide": "TS Get Image Size",
    "TS_ImagePromptInjector": "TS Image Prompt Injector",
}
