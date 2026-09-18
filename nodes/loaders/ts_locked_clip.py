"""TS Load CLIP (locked) — текстовый энкодер из `.tsmodel`, до четырёх файлов.

Заменяет сразу CLIPLoader, DualCLIPLoader, TripleCLIPLoader и QuadrupleCLIPLoader:
лишние входы просто оставляют на `none`.

⚠️ Список типов берётся из `comfy.sd.CLIPType` НА ЛЕТУ. Захардкоженный список
устарел бы в тот день, когда в ComfyUI появится очередная модель.
"""

from __future__ import annotations

import torch
from comfy_api.v0_0_2 import IO

from ._locked import load, options

CATEGORY_FOLDER = "text_encoders"
NONE = "none"


def clip_types() -> list[str]:
    import comfy.sd

    return [entry.name.lower() for entry in comfy.sd.CLIPType]


class TS_LockedCLIP(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        names = options(CATEGORY_FOLDER)
        extra = [NONE, *names]
        return IO.Schema(
            node_id="TS_LockedCLIP",
            display_name="TS Load CLIP",
            category="TS/Loaders",
            description=(
                "Load a locked .tsmodel text encoder — up to four files at once, so it "
                "stands in for CLIPLoader, DualCLIPLoader, TripleCLIPLoader and "
                "QuadrupleCLIPLoader alike. The type list comes from ComfyUI itself."
            ),
            inputs=[
                IO.Combo.Input(
                    "clip_name1",
                    options=names,
                    tooltip="A .tsmodel file from models/text_encoders.",
                ),
                IO.Combo.Input(
                    "type",
                    options=clip_types(),
                    default="stable_diffusion",
                    tooltip="Which text encoder this is. The list is read from comfy.sd.CLIPType.",
                ),
                IO.Combo.Input(
                    "clip_name2", options=extra, default=NONE, optional=True,
                    tooltip="Second encoder, for models that need a pair. 'none' to skip.",
                ),
                IO.Combo.Input(
                    "clip_name3", options=extra, default=NONE, optional=True,
                    tooltip="Third encoder. 'none' to skip.",
                ),
                IO.Combo.Input(
                    "clip_name4", options=extra, default=NONE, optional=True,
                    tooltip="Fourth encoder. 'none' to skip.",
                ),
                IO.Combo.Input(
                    "device", options=["default", "cpu"], default="default",
                    optional=True, advanced=True,
                    tooltip="Keep the encoder on the CPU to leave the card to the model.",
                ),
            ],
            outputs=[IO.Clip.Output(display_name="clip")],
            search_aliases=["locked", "tsmodel", "clip loader", "text encoder",
                            "dual clip", "triple clip"],
        )

    @classmethod
    def execute(cls, clip_name1: str, type: str = "stable_diffusion",  # noqa: A002 - имя входа
                clip_name2: str = NONE, clip_name3: str = NONE, clip_name4: str = NONE,
                device: str = "default") -> IO.NodeOutput:
        import comfy.sd
        import comfy.utils
        import folder_paths

        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        model_options: dict = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        payload = []
        for relative in (clip_name1, clip_name2, clip_name3, clip_name4):
            if not relative or relative == NONE:
                continue
            state, metadata, _path = load(CATEGORY_FOLDER, relative)
            # Тот же шаг, что делает `comfy.sd.load_clip` со старыми квантами.
            if hasattr(comfy.utils, "convert_old_quants"):
                state, metadata = comfy.utils.convert_old_quants(
                    state, model_prefix="", metadata=metadata)
            payload.append(state)

        clip = comfy.sd.load_text_encoder_state_dicts(
            payload,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        return IO.NodeOutput(clip)


NODE_CLASS_MAPPINGS = {"TS_LockedCLIP": TS_LockedCLIP}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedCLIP": "TS Load CLIP"}
