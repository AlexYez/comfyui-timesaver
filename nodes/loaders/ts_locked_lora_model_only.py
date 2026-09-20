"""TS Load LoRA, model only — штатный LoraLoaderModelOnly для файлов `.tsmodel`.

⚠️ Только «model only», и это не упрощение ради экономии. Модели, ради которых
замок и заводился, — видео и дистилляты: LTX, Wan, Krea 2. CLIP в таком графе
отдельным проводом не ходит вовсе, и вход, который нечем заполнить, там только
мешает. Парная нода «model + clip» в паке была ровно один день и снята по просьбе
владельца: вернуть её можно из истории git (релиз 12.11.4).
"""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._locked import load, options

CATEGORY_FOLDER = "loras"


def apply_locked_lora(model, relative: str, strength_model: float):
    """Открыть запертый файл и применить его к модели.

    ⚠️ Нулевая сила — не «применить ноль», а «не трогать»: так же ведёт себя
    штатная нода, и на этом держатся графы, где LoRA выключают силой.
    """
    import comfy.sd

    if strength_model == 0:
        return model

    lora, metadata, _path = load(CATEGORY_FOLDER, relative)
    try:
        patched, _clip = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0, lora_metadata=metadata)
    except TypeError:                   # ComfyUI до появления lora_metadata
        patched, _clip = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
    return patched


class TS_LockedLoraModelOnly(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LockedLoraModelOnly",
            display_name="TS Load LoRA, model only",
            category="TS/Loaders",
            description=(
                "Apply a locked .tsmodel LoRA to a model, the way LoraLoaderModelOnly "
                "does — for graphs that carry no CLIP at all, which is every video model. "
                "Strength 0 leaves the model untouched."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="The model the LoRA is applied to."),
                IO.Combo.Input(
                    "lora_name",
                    options=options(CATEGORY_FOLDER),
                    tooltip="A .tsmodel file from models/loras.",
                ),
                IO.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="How strongly the LoRA affects the model. 0 leaves it untouched.",
                ),
            ],
            outputs=[IO.Model.Output(display_name="model")],
            search_aliases=["locked", "tsmodel", "lora loader", "lora model only"],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength_model: float) -> IO.NodeOutput:
        return IO.NodeOutput(apply_locked_lora(model, lora_name, strength_model))


NODE_CLASS_MAPPINGS = {"TS_LockedLoraModelOnly": TS_LockedLoraModelOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedLoraModelOnly": "TS Load LoRA, model only"}
