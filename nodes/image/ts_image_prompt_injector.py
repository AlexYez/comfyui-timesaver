"""TS Image Prompt Injector.

node_id: TS_ImagePromptInjector
"""

from typing import Optional
import time
import logging

import torch

from comfy_api.latest import IO


logger = logging.getLogger("comfyui_timesaver.ts_image_prompt_injector")
LOG_PREFIX = "[TS Image Prompt Injector]"


class TS_ImagePromptInjector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImagePromptInjector",
            display_name="TS Image Prompt Injector",
            category="TS/Image",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("prompt", default="", multiline=True),
            ],
            outputs=[IO.Image.Output(display_name="image")],
            hidden=[IO.Hidden.prompt],
        )

    @staticmethod
    def _log(message: str) -> None:
        logger.info("%s %s", LOG_PREFIX, message)

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

    @classmethod
    def execute(cls, image: torch.Tensor, prompt: str) -> IO.NodeOutput:
        cls._log_tensor("Input", image)

        if image is None:
            cls._log("Input is None, returning as-is.")
            return IO.NodeOutput(image)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim not in (3, 4):
            raise ValueError(
                f"Expected IMAGE with 3 or 4 dims [H,W,C] or [B,H,W,C], got {image.ndim}"
            )

        prompt_text = cls._normalize_prompt(prompt)
        if prompt_text == "":
            cls._log("Prompt is empty, metadata not updated.")
            return IO.NodeOutput(image)

        cls._log(f"Injecting prompt length={len(prompt_text)}")
        prompt_graph = cls.hidden.prompt
        cls._inject_prompt_into_graph(prompt_graph, prompt_text)
        return IO.NodeOutput(image)

    @classmethod
    def fingerprint_inputs(cls, image: torch.Tensor, prompt: str) -> str:
        prompt_text = cls._normalize_prompt(prompt)
        if isinstance(image, torch.Tensor):
            shape = tuple(image.shape)
        else:
            shape = "none"
        return f"ts_prompt_injector_{shape}_{len(prompt_text)}_{time.time_ns()}"


NODE_CLASS_MAPPINGS = {"TS_ImagePromptInjector": TS_ImagePromptInjector}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImagePromptInjector": "TS Image Prompt Injector"}
