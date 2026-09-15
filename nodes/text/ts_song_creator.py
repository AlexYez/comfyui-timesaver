"""TS Song Creator — текст песни и стиль, двумя строками на выходе.

Модели генерации песен (YuE 2, ACE-Step) принимают ДВА поля: текст с
сегментными тегами и отдельный промпт стиля. Раньше это были две безымянные
`STRING`-ноды, и всё неудобство ложилось на человека: теги набирались руками,
стиль вспоминался по памяти, а ударение в русской строке ставилось копированием
символа из таблицы символов.

Нода — тонкая: она ничего не переписывает и не «улучшает», а отдаёт ровно то,
что человек набрал. Вся работа — в интерфейсе (`js/text/song_creator/`):
кнопки сегментных тегов, библиотека стилей из пресетов, ударение по правой
кнопке мыши и полноэкранный режим для длинного текста.

⚠️ Нода работает и без интерфейса: оба поля — обычные многострочные виджеты.
Пресет влияет только на то, какие стили и теги показывает интерфейс; на выход
он не влияет, и это сказано в подсказке ко входу.
"""

from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO

from ._song_presets import preset_names

logger = logging.getLogger("comfyui_timesaver.ts_song_creator")
LOG_PREFIX = "[TS Song Creator]"


def _normalize(text: str) -> str:
    """Убрать возврат каретки и хвостовые пробелы строк.

    ⚠️ Правится ТОЛЬКО то, что портит вход модели независимо от замысла автора:
    `\\r\\n` из буфера обмена Windows и пробелы в конце строк. Пустые строки,
    регистр и знаки препинания не трогаются — в песне они значат ровно то, что
    написал человек.

    Args:
        text: содержимое поля.

    Returns:
        Нормализованный текст.
    """
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


class TS_SongCreator(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        presets = preset_names()
        return IO.Schema(
            node_id="TS_SongCreator",
            display_name="TS Song Creator",
            category="TS/Text",
            essentials_category="Text",
            description=(
                "Write the song and its style in one place, and hand a song model both "
                "strings it needs.\n"
                "Section tags go in by button, the style comes from a library of presets, "
                "and a stressed vowel is one right-click away — which is what Russian "
                "lyrics need and what no plain text box offers. Open the editor full "
                "screen when the lyrics get long."
            ),
            inputs=[
                IO.String.Input(
                    "lyrics",
                    multiline=True,
                    default="",
                    tooltip=(
                        "The song itself. Section tags such as [verse] and [chorus] tell the "
                        "model where a part starts — the buttons above insert the ones this "
                        "preset's model reads.\n"
                        "Right-click a vowel to mark it stressed: the mark goes into the text "
                        "as a combining accent, exactly what a model needs to sing a Russian "
                        "word the way you mean it."
                    ),
                ),
                IO.String.Input(
                    "style",
                    multiline=True,
                    default="",
                    tooltip=(
                        "How it should sound: genre, tempo, voice, instruments, mood. A "
                        "comma-separated list works best.\n"
                        "Pick a preset from the library to fill this in, then edit it — the "
                        "presets are a starting point, not a cage."
                    ),
                ),
                IO.Combo.Input(
                    "preset",
                    options=presets,
                    default=presets[0],
                    tooltip=(
                        "Whose vocabulary the editor offers: the style library and the "
                        "section-tag buttons come from this preset.\n"
                        "It does NOT change the output — what leaves this node is exactly "
                        "what is in the two fields. Presets are plain JSON in "
                        "nodes/text/song_presets, so another model is a file, not a code "
                        "change."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="lyrics",
                    tooltip="The lyrics, as typed. Carriage returns and trailing spaces removed.",
                ),
                IO.String.Output(
                    display_name="style",
                    tooltip="The style prompt, as typed.",
                ),
            ],
            search_aliases=[
                "song", "lyrics", "music", "yue", "ace step", "acestep",
                "песня", "текст песни", "стиль",
            ],
        )

    @classmethod
    def execute(cls, lyrics: str, style: str, preset: str) -> IO.NodeOutput:
        """Отдать оба поля. `preset` читает интерфейс, выход от него не зависит."""
        return IO.NodeOutput(_normalize(lyrics), _normalize(style))


NODE_CLASS_MAPPINGS = {"TS_SongCreator": TS_SongCreator}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SongCreator": "TS Song Creator"}
