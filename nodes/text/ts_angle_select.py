"""TS Angle Select — ракурс камеры мышкой, промпт на выходе.

Три положения (поворот, высота, крупность) превращаются в промпт по правилам
выбранного пресета. Пресет — JSON в `nodes/text/angle_presets/`, поэтому новая
модель со своим словарём не требует правки кода.

⚠️ Словарь пресета — это то, на чём модель обучалась. У Qwen Multi-Angle это
буквально триггерная фраза LoRA (`<sks> …`): переписать её «по-человечески»
значит сломать обусловливание. Правила чтения и сборки — в `_angle_presets.py`.

Визуальный редактор живёт в `js/text/ts-angle-select.js`; сама нода работает и
без него — три виджета остаются обычными и настраиваются руками.
"""

from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO

from ._angle_presets import (
    AZIMUTHS,
    FRAMINGS,
    HEIGHTS,
    build_prompt,
    load_presets,
    preset_names,
    snap_azimuth,
)

logger = logging.getLogger("comfyui_timesaver.ts_angle_select")
LOG_PREFIX = "[TS Angle Select]"


class TS_AngleSelect(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_AngleSelect",
            display_name="TS Angle Select",
            category="TS/Text",
            essentials_category="Text",
            description=(
                "Point the camera at the subject and get the prompt that asks a model for "
                "exactly that view.\n"
                "Drag the camera around the stage, pick a height and a framing; the phrasing "
                "comes from the chosen preset, so the same three settings can drive different "
                "models."
            ),
            inputs=[
                IO.Combo.Input(
                    "preset",
                    options=preset_names(),
                    default=preset_names()[0],
                    tooltip=(
                        "Which model's vocabulary to speak.\n"
                        "• Qwen Multi-Angle — Qwen-Image-Edit with the Multiple-Angles LoRA. "
                        "The wording is the LoRA's trigger phrase, not a description: it was "
                        "trained on these exact words, and the '<sks>' token has to be there.\n"
                        "Presets are plain JSON in nodes/text/angle_presets — a new model is a "
                        "new file, not a code change."
                    ),
                ),
                IO.Int.Input(
                    "azimuth",
                    default=0,
                    min=0,
                    max=315,
                    step=45,
                    tooltip=(
                        "Where the camera stands, in degrees around the subject. 0 is in "
                        "front, 90 is the subject's right, 180 is behind.\n"
                        "Eight positions, 45 degrees apart — the same eight the model was "
                        "trained on. Anything in between is snapped to the nearest one, "
                        "because a phrase for it does not exist."
                    ),
                ),
                IO.Combo.Input(
                    "height",
                    options=list(HEIGHTS),
                    default="eye-level",
                    tooltip=(
                        "How high the camera sits.\n"
                        "• low — below the subject, looking up.\n"
                        "• eye-level (default) — level with the subject.\n"
                        "• elevated — a little above, looking slightly down.\n"
                        "• high — well above the subject, looking down."
                    ),
                ),
                IO.Combo.Input(
                    "framing",
                    options=list(FRAMINGS),
                    default="medium",
                    tooltip=(
                        "How much of the subject is in frame.\n"
                        "• wide — the whole subject and its surroundings.\n"
                        "• medium (default) — roughly waist up.\n"
                        "• close-up — head and shoulders."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="prompt",
                    tooltip=(
                        "The camera-angle prompt. Feed it to the text encoder — on its own "
                        "for a pure re-angle, or joined with your own description."
                    ),
                ),
            ],
            search_aliases=["angle", "camera", "view", "multiangle", "reangle", "ракурс"],
        )

    @classmethod
    def execute(
        cls,
        preset: str,
        azimuth: int,
        height: str,
        framing: str,
    ) -> IO.NodeOutput:
        presets = load_presets()
        chosen = presets.get(preset)
        if chosen is None:
            # ⚠️ Пресет мог уехать вместе с файлом (переименовали, удалили). Не
            # молчим и не подставляем пустоту: берём первый и говорим об этом.
            if not presets:
                raise RuntimeError(
                    f"{LOG_PREFIX} No usable presets in nodes/text/angle_presets. "
                    f"Each preset is a JSON file with 'template', 'horizontal', "
                    f"'height' and 'framing'."
                )
            fallback = next(iter(presets))
            logger.warning("%s Unknown preset %r; using %r instead.",
                           LOG_PREFIX, preset, fallback)
            chosen = presets[fallback]

        snapped = snap_azimuth(azimuth)
        if snapped != azimuth:
            logger.info("%s azimuth %s -> %s (nearest of %s).",
                        LOG_PREFIX, azimuth, snapped,
                        ", ".join(str(a) for a in AZIMUTHS))
        prompt = build_prompt(chosen, snapped, height, framing)
        logger.info("%s %s | %s deg, %s, %s -> %s",
                    LOG_PREFIX, chosen.get("name"), snapped, height, framing, prompt)
        return IO.NodeOutput(prompt)


NODE_CLASS_MAPPINGS = {"TS_AngleSelect": TS_AngleSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_AngleSelect": "TS Angle Select"}
