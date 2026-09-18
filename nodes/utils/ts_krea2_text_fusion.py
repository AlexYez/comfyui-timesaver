"""TS Krea 2 Text Fusion — one dial on how loudly the prompt reaches Krea 2.

Krea 2 does not condition on a single text embedding. Its encoder is
Qwen3-VL-4B, and the model keeps **twelve hidden layers of it at once**
(``hidden_states[2, 5, 8, … 35]``, see ``comfy/text_encoders/krea2.py``). Those
twelve taps are collapsed into the one embedding the DiT attends to by a single
linear layer inside the model:

```
self.projector = operations.Linear(num_txt_layers, 1, bias=False)   # 12 -> 1
```

— ``comfy/ldm/krea2/model.py``. Its weight is therefore **twelve numbers**, one
per tap, and they are a learned weighted sum, not a projection into another
space. Measured in the owner's checkpoints, ``txtfusion.projector.weight`` is
indeed ``[1, 12]``.

Scaling those twelve numbers scales the whole fused text embedding, and with it
how hard the prompt pushes against the image: up for prompt adherence, down for
the model's own ideas. The trick comes from
`Extraltodeus/ComfyUI-Krea2-attention-tweak
<https://github.com/Extraltodeus/ComfyUI-Krea2-attention-tweak>`_, who spotted
it in Beinsezii's ``Krea-2-Turbo-Projector-Scale-LoRA`` — a LoRA whose values
were close enough to the originals to be a multiplier rather than a change of
direction. He settles on 3.0.

⚠️ **The zeros below are not a mistake.** ComfyUI has no "multiply this weight"
patch, but ``ModelPatcher.add_patches(patches, strength_patch, strength_model)``
applies ``weight *= strength_model`` **before** it adds ``strength_patch *
patch`` (``comfy/lora.py``, ``calculate_weight``). So a patch of zeros at
``strength_patch = 0`` is a pure multiply: the addition is skipped entirely
(``if strength != 0.0``), and even if that guard ever went away, adding zeros
changes nothing. The multiply lands on a **copy** — ``patch_weight_to_device``
casts with ``copy=True`` — so the checkpoint on disk and in memory is untouched
and the patch is undone with the model.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_krea2_text_fusion")
LOG_PREFIX = "[TS Krea 2 Text Fusion]"

#: Тот самый линейный слой 12 -> 1. Имя одинаково у обычного Krea 2, у Turbo и у
#: квантованных int8/convrot сборок — проверено по заголовкам safetensors.
PROJECTOR_KEY = "diffusion_model.txtfusion.projector.weight"


class TS_Krea2TextFusion(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Krea2TextFusion",
            display_name="TS Krea 2 Text Fusion",
            category="TS/Utils",
            description=(
                "Scale how loudly the prompt reaches a Krea 2 model. Krea 2 fuses twelve "
                "hidden layers of its text encoder into one embedding through a single "
                "12-to-1 linear layer; this multiplies that layer, so the whole text "
                "signal gets louder or quieter. Higher follows the prompt more, lower "
                "leaves the model more room. Krea 2 only — any other model is refused."
            ),
            inputs=[
                IO.Model.Input(
                    "model",
                    tooltip="A Krea 2 or Krea 2 Turbo model. Nothing else carries this layer.",
                ),
                IO.Float.Input(
                    "strength",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.05,
                    tooltip=(
                        "Multiplier for the text-fusion projector. 1.0 is the model as "
                        "trained and changes nothing; around 3.0 is what the author of "
                        "the original tweak settles on — more prompt adherence while "
                        "still inventive. 0.0 silences the text path entirely, and "
                        "negative values invert it."
                    ),
                ),
            ],
            outputs=[
                IO.Model.Output(
                    display_name="model",
                    tooltip="The same model with the projector scaled. The original is untouched.",
                ),
            ],
            search_aliases=[
                "krea", "krea2", "krea 2 attention", "krea attention tweak",
                "txtfusion", "projector scale", "prompt adherence",
            ],
        )

    @classmethod
    def execute(cls, model, strength: float) -> IO.NodeOutput:
        patcher = model.clone()

        # 1.0 — это «как обучено». Патч с таким множителем ничего не изменил бы,
        # но остался бы в списке и заставил ядро пересчитывать вес впустую.
        if float(strength) == 1.0:
            logger.debug("%s strength is 1.0 — nothing to scale", LOG_PREFIX)
            return IO.NodeOutput(patcher)

        weight = patcher.model_state_dict().get(PROJECTOR_KEY)
        if weight is None:
            raise RuntimeError(
                f"{LOG_PREFIX} This model has no '{PROJECTOR_KEY}', so it is not a Krea 2. "
                "The twelve-layer text fusion this node scales exists only in Krea 2 and "
                "Krea 2 Turbo; connect one of those."
            )

        # ⚠️ Нули здесь — носитель, а не значение: работу делает третий аргумент
        # (`strength_model`), который ядро применяет как умножение. Подробности —
        # в docstring модуля.
        applied = patcher.add_patches(
            {PROJECTOR_KEY: (torch.zeros_like(weight),)}, 0.0, float(strength),
        )
        if not applied:
            # add_patches молча пропускает ключ, которого нет в state_dict самой
            # модели. Молчать вслед за ним нельзя: человек увидел бы обычную
            # картинку и решил, что твик не работает.
            raise RuntimeError(
                f"{LOG_PREFIX} The model reports '{PROJECTOR_KEY}' but refused the patch "
                "for it. The checkpoint looks like Krea 2 but is put together differently "
                "than this node expects."
            )

        logger.info("%s text fusion scaled by %.3f (%d values)",
                    LOG_PREFIX, float(strength), int(weight.numel()))
        return IO.NodeOutput(patcher)


NODE_CLASS_MAPPINGS = {"TS_Krea2TextFusion": TS_Krea2TextFusion}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Krea2TextFusion": "TS Krea 2 Text Fusion"}
