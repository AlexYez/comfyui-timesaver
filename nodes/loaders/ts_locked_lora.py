"""TS Load LoRA (locked) — штатный LoraLoader для файлов `.tsmodel`."""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._locked import load, options

CATEGORY_FOLDER = "loras"


def apply_locked_lora(model, clip, relative: str, strength_model: float, strength_clip: float):
    """Общее для обеих LoRA-нод: открыть запертый файл и применить его.

    ⚠️ Нулевые силы — не «применить ноль», а «не трогать»: так же ведёт себя
    штатная нода, и на этом держатся графы, где LoRA выключают силой.
    """
    import comfy.sd

    if strength_model == 0 and strength_clip == 0:
        return model, clip

    lora, metadata, _path = load(CATEGORY_FOLDER, relative)
    try:
        return comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata=metadata)
    except TypeError:                   # ComfyUI до появления lora_metadata
        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)


class TS_LockedLora(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LockedLora",
            display_name="TS Load LoRA",
            category="TS/Loaders",
            description=(
                "Apply a locked .tsmodel LoRA to a model and its CLIP, the way LoraLoader "
                "does. Both strengths at zero leave everything untouched."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="The model the LoRA is applied to."),
                IO.Clip.Input("clip", tooltip="The CLIP the LoRA is applied to."),
                IO.Combo.Input(
                    "lora_name",
                    options=options(CATEGORY_FOLDER),
                    tooltip="A .tsmodel file from models/loras.",
                ),
                IO.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="How strongly the LoRA affects the model.",
                ),
                IO.Float.Input(
                    "strength_clip", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="How strongly it affects the text encoder.",
                ),
            ],
            outputs=[
                IO.Model.Output(display_name="model"),
                IO.Clip.Output(display_name="clip"),
            ],
            search_aliases=["locked", "tsmodel", "lora loader"],
        )

    @classmethod
    def execute(cls, model, clip, lora_name: str,
                strength_model: float, strength_clip: float) -> IO.NodeOutput:
        patched_model, patched_clip = apply_locked_lora(
            model, clip, lora_name, strength_model, strength_clip)
        return IO.NodeOutput(patched_model, patched_clip)


NODE_CLASS_MAPPINGS = {"TS_LockedLora": TS_LockedLora}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedLora": "TS Load LoRA"}
