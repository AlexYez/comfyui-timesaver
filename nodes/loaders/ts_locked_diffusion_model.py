"""TS Load Diffusion Model (locked) — штатный UNETLoader для файлов `.tsmodel`.

⚠️ Нода НИЧЕГО не знает об архитектурах. Она разбирает заголовок в памяти и
отдаёт state dict в ту же `comfy.sd.load_diffusion_model_state_dict`, что и
штатный загрузчик. Поэтому кванты и новые архитектуры подхватываются сами, без
правок здесь.
"""

from __future__ import annotations

import torch
from comfy_api.v0_0_2 import IO

from ._locked import options, resolve
from ._mclock import LOG_PREFIX

CATEGORY_FOLDER = "diffusion_models"

WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _model_options(weight_dtype: str) -> dict:
    """То же соответствие, что у штатного UNETLoader."""
    if weight_dtype == "fp8_e4m3fn":
        return {"dtype": torch.float8_e4m3fn}
    if weight_dtype == "fp8_e4m3fn_fast":
        return {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True}
    if weight_dtype == "fp8_e5m2":
        return {"dtype": torch.float8_e5m2}
    return {}


def load_locked_diffusion_model(path: str, model_options: dict | None = None):
    """Загрузить модель и оставить рецепт, как загрузить её снова.

    ⚠️ `cached_patcher_init` — то, чем ComfyUI повторяет загрузку при глубоком
    клонировании и на нескольких картах. Без него такой клон остался бы без
    весов, и это выяснилось бы только на второй карте.
    """
    import comfy.sd

    from ._mclock import load_state_dict_comfy

    options_dict = dict(model_options or {})
    state, metadata = load_state_dict_comfy(path)
    model = comfy.sd.load_diffusion_model_state_dict(
        state, model_options=options_dict, metadata=metadata)
    if model is None:
        raise RuntimeError(
            f"{LOG_PREFIX} Could not detect the model type of {path}. The file opened "
            "fine, so this is the model itself, not the lock."
        )
    model.cached_patcher_init = (load_locked_diffusion_model, (path, options_dict))
    return model


class TS_LockedDiffusionModel(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LockedDiffusionModel",
            display_name="TS Load Diffusion Model",
            category="TS/Loaders",
            description=(
                "Load a locked .tsmodel diffusion model — the stock UNETLoader for files "
                "that no other reader opens. Any architecture and any quantisation the "
                "stock loader handles works here too, because the tensors go into the "
                "very same loading function."
            ),
            inputs=[
                IO.Combo.Input(
                    "unet_name",
                    options=options(CATEGORY_FOLDER),
                    tooltip=(
                        "A .tsmodel file from models/diffusion_models. The stock loaders "
                        "do not list these — that is the whole point of the lock."
                    ),
                ),
                IO.Combo.Input(
                    "weight_dtype",
                    options=WEIGHT_DTYPES,
                    default="default",
                    advanced=True,
                    tooltip="Same weight casting the stock UNETLoader offers.",
                ),
            ],
            outputs=[IO.Model.Output(display_name="model")],
            search_aliases=["locked", "tsmodel", "unet loader", "load diffusion model"],
        )

    @classmethod
    def execute(cls, unet_name: str, weight_dtype: str = "default") -> IO.NodeOutput:
        path = resolve(CATEGORY_FOLDER, unet_name)
        return IO.NodeOutput(load_locked_diffusion_model(path, _model_options(weight_dtype)))


NODE_CLASS_MAPPINGS = {"TS_LockedDiffusionModel": TS_LockedDiffusionModel}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedDiffusionModel": "TS Load Diffusion Model"}
