"""TS Prompt Library — библиотека готовых промптов, выход STRING.

Выбираешь раздел, модель и пресет — нода показывает промпт с кнопкой
«Копировать» и отдаёт его дальше по графу. Поля вида ``{TARGET}`` заполняются
прямо в ноде: в выход уходит готовый текст, а не заготовка со скобками.

Библиотека — данные в `nodes/text/prompt_library/` (раздел → коллекция под
модель → пресеты); правила чтения, проверки и подстановки — в
`_prompt_library.py`. Новый раздел или модель добавляется папкой с JSON, без
правки кода. Интерфейс — `js/text/ts-prompt-library.js`; без него нода
работает на двух обычных виджетах.
"""

from __future__ import annotations

import hashlib
import json
import logging

from comfy_api.v0_0_2 import IO

from ._prompt_library import fill, find_preset, is_text_value, library_version, preset_keys

logger = logging.getLogger("comfyui_timesaver.ts_prompt_library")
LOG_PREFIX = "[TS Prompt Library]"

# Вариант выпадашки, когда библиотека пуста или не читается: схема обязана
# собраться, иначе ComfyUI не покажет ноду вовсе и не скажет почему.
EMPTY_OPTION = "(library is empty)"


def _options() -> list[str]:
    return preset_keys() or [EMPTY_OPTION]


def parse_fields(text) -> dict:
    """JSON значений полей -> {ИМЯ: текст}. Мусор — пустой словарь."""
    if isinstance(text, dict):
        data = text
    else:
        try:
            data = json.loads(str(text or "") or "{}")
        except ValueError:
            return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if is_text_value(v)}


class TS_PromptLibrary(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        options = _options()
        return IO.Schema(
            node_id="TS_PromptLibrary",
            display_name="TS Prompt Library",
            category="TS/Text",
            essentials_category="Text",
            description=(
                "A library of ready-made prompts. Pick a section, a model and a preset: the "
                "node shows the prompt with a Copy button and sends it on as text.\n"
                "Fields in braces such as {TARGET} are filled in right on the node, so the "
                "output is a finished prompt. First section: context editing for Qwen Image "
                "2.1 and FLUX.2 Klein 9B."
            ),
            inputs=[
                IO.Combo.Input(
                    "preset",
                    options=options,
                    default=options[0],
                    tooltip=(
                        "Which prompt to use, as section/model/code.\n"
                        "The library is plain JSON in nodes/text/prompt_library — a new "
                        "section or model is a new folder, not a code change."
                    ),
                ),
                IO.String.Input(
                    "fields",
                    default="{}",
                    multiline=False,
                    socketless=True,
                    tooltip=(
                        "Values for the fields in braces, as JSON: {\"TARGET\": \"the red "
                        "suitcase\"}. The node's own panel fills this in; an empty field "
                        "stays in braces."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="prompt",
                    tooltip="The finished prompt, fields filled in. Feed it to the text encoder.",
                ),
            ],
            search_aliases=[
                "prompt library", "prompt presets", "prompts", "edit prompts",
                "qwen image edit", "flux klein", "библиотека промптов", "промпты",
            ],
        )

    @classmethod
    def validate_inputs(cls, preset: str | None, fields: str | None = "{}"):
        # Подключённый проводом вход на этапе проверки приходит как None:
        # значение появится только при выполнении, судить о нём рано.
        if preset is None:
            return True
        if find_preset(preset)[0] is None:
            return (
                f"{LOG_PREFIX} Preset '{preset}' is not in the library. It may have been "
                f"renamed or removed from nodes/text/prompt_library."
            )
        return True

    @classmethod
    def fingerprint_inputs(cls, preset: str, fields: str = "{}") -> str:
        # Правка JSON библиотеки меняет текст пресета — прогон обязан это увидеть.
        digest = hashlib.blake2b(digest_size=16)
        digest.update(f"{library_version()}|{preset}|{fields}".encode("utf-8", "replace"))
        return digest.hexdigest()

    @classmethod
    def execute(cls, preset: str, fields: str = "{}") -> IO.NodeOutput:
        chosen, collection = find_preset(preset)
        if chosen is None:
            raise RuntimeError(
                f"{LOG_PREFIX} Preset '{preset}' is not in the library "
                f"(nodes/text/prompt_library). Pick another one on the node."
            )
        values = parse_fields(fields)
        prompt = fill(chosen["prompt"], values)
        empty = [name for name in chosen["fields"] if not values.get(name, "").strip()]
        if empty:
            # Не ошибка: заготовку со скобками бывает удобно дописать дальше по
            # графу. Но модель, получившая «{TARGET}» буквально, ответит мусором.
            logger.warning(
                "%s %s: field(s) %s left empty — they stay in braces in the prompt.",
                LOG_PREFIX, chosen["key"], ", ".join(empty),
            )
        logger.info("%s %s (%s)", LOG_PREFIX, chosen["key"], collection["title"])
        return IO.NodeOutput(prompt)


NODE_CLASS_MAPPINGS = {"TS_PromptLibrary": TS_PromptLibrary}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_PromptLibrary": "TS Prompt Library"}
