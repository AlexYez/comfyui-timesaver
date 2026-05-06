from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_EN = ROOT / "README.md"
README_RU = ROOT / "README.ru.md"


@dataclass(frozen=True)
class NodeCard:
    node_id: str
    en_description: str
    image_path: str
    required: str
    optional: str
    outputs: str
    internal_id: str
    class_name: str
    file_name: str
    category: str
    function_name: str
    dependency_note: str | None


CARD_RE = re.compile(
    r"<details>\n<summary><strong>(?P<node_id>.*?)</strong> - (?P<description>.*?)</summary>"
    r"(?P<body>.*?)</details>",
    re.S,
)


def _extract_line(body: str, label: str) -> str:
    match = re.search(rf"- {re.escape(label)}: `(.*?)`", body)
    return match.group(1).strip() if match else ""


def _extract_controls(body: str, title: str) -> str:
    match = re.search(rf"- {re.escape(title)}: (.*)", body)
    return match.group(1).strip() if match else "`-`"


def _extract_image_path(body: str) -> str:
    match = re.search(r"!\[Screenshot placeholder for .*?\]\((.*?)\)", body)
    return match.group(1).strip() if match else "docs/img/placeholders/placeholder.png"


def _extract_outputs(body: str) -> str:
    match = re.search(r"\*\*Outputs\*\*\n- `(.*?)`", body)
    return match.group(1).strip() if match else "-"


def _extract_dependency_note(body: str) -> str | None:
    match = re.search(r"- Dependency note: (.*)", body)
    if not match:
        return None
    return match.group(1).strip()


def parse_cards(readme_text: str) -> list[NodeCard]:
    cards: list[NodeCard] = []
    for match in CARD_RE.finditer(readme_text):
        body = match.group("body")
        cards.append(
            NodeCard(
                node_id=match.group("node_id").strip(),
                en_description=match.group("description").strip(),
                image_path=_extract_image_path(body),
                required=_extract_controls(body, "Required"),
                optional=_extract_controls(body, "Optional"),
                outputs=_extract_outputs(body),
                internal_id=_extract_line(body, "Internal id"),
                class_name=_extract_line(body, "Class"),
                file_name=_extract_line(body, "File"),
                category=_extract_line(body, "Category"),
                function_name=_extract_line(body, "Function"),
                dependency_note=_extract_dependency_note(body),
            )
        )
    return cards


RU_DESCRIPTION_BY_NODE_ID: dict[str, str] = {
    "TS_Qwen3_VL_V3": "Основная мультимодальная нода Qwen (текст + изображение/видео) с пресетами, управлением precision и offline-режимом.",
    "TSWhisper": "Нода Whisper для транскрибации и перевода аудио с выводом SRT и обычного текста.",
    "TS_SileroTTS": "Русская TTS-нода на базе Silero с чанкингом и выходом AUDIO.",
    "TS_MusicStems": "Разделяет музыку на стемы (vocals, bass, drums, others, instrumental).",
    "TS_PromptBuilder": "Собирает структурированные промпты из JSON-конфига и seed для воспроизводимых вариаций.",
    "TS_BatchPromptLoader": "Читает многострочные промпты и выдаёт один промпт по индексу/шагу.",
    "TS_StylePromptSelector": "Загружает текст style-промпта из библиотеки по ID или имени.",
    "TS_ImageResize": "Гибкая нода resize: точный размер, масштаб по стороне, scale factor или целевые мегапиксели.",
    "TS_QwenSafeResize": "Безопасный resize-пресет под ограничения препроцессинга Qwen.",
    "TS_WAN_SafeResize": "Safe-resize для WAN-пайплайнов с размером, дружелюбным к моделям.",
    "TS_QwenCanvas": "Создаёт canvas с разрешением, удобным для Qwen, и при необходимости размещает image/mask.",
    "TS_ResolutionSelector": "Выбирает целевое разрешение по aspect-пресетам или custom ratio и может вернуть подготовленный canvas.",
    "TS_Color_Grade": "Быстрый первичный color grading: hue, temperature, saturation, contrast, gamma и tone-контроли.",
    "TS_Film_Emulation": "Нода для киношной стилизации: пресеты, LUT, warmth, fade и контроль зерна.",
    "TS_FilmGrain": "Добавляет управляемое плёночное зерно с настройкой размера, интенсивности, цвета и движения.",
    "TS_Color_Match": "Переносит цветовое настроение с референса на целевое изображение, сохраняя структуру сцены.",
    "TS_BGRM_BiRefNet": "AI-удаление фона через BiRefNet: быстро, чисто и удобно для прозрачных композиций.",
    "TSCropToMask": "Кадрирует область вокруг маски, чтобы ускорить локальные правки и сэкономить память.",
    "TSRestoreFromCrop": "Возвращает обработанный crop обратно в исходный кадр с опциональным смешиванием.",
    "TS_ImageBatchToImageList": "Преобразует batched IMAGE tensor в покадровый list-поток.",
    "TS_ImageListToImageBatch": "Собирает list-поток изображений обратно в batched IMAGE tensor.",
    "TS_ImageBatchCut": "Обрезает кадры с начала и/или конца image batch.",
    "TS_GetImageMegapixels": "Возвращает значение мегапикселей для быстрой оценки качества и производительности.",
    "TS_GetImageSizeSide": "Возвращает размер выбранной стороны изображения для логики графа и автоконфигурации.",
    "TS_ImagePromptInjector": "Встраивает текст промпта в image-flow, чтобы контекст шёл вместе с веткой изображения.",
    "TS_ImageTileSplitter": "Делит изображение на перекрывающиеся тайлы для тяжёлой обработки в высоком качестве.",
    "TS_ImageTileMerger": "Склеивает тайлы обратно в целое изображение по tile metadata и blending.",
    "TSAutoTileSize": "Автоматически рассчитывает размер тайлов (width/height) под вашу целевую сетку.",
    "TS Cube to Equirectangular": "Конвертирует шесть граней куба в одну equirectangular 360-панораму.",
    "TS Equirectangular to Cube": "Конвертирует 360-панораму в шесть граней куба для редактирования и проекций.",
    "TS_VideoDepthNode": "Строит depth maps по последовательности кадров для композитинга, relighting и depth-эффектов.",
    "TS_Video_Upscale_With_Model": "Апскейлит последовательность кадров с выбранной моделью апскейла и memory-стратегиями.",
    "TS_RTX_Upscaler": "Нода NVIDIA RTX Upscaler для быстрого и качественного апскейла на поддерживаемых системах.",
    "TS_DeflickerNode": "Снижает временное мерцание яркости и цвета в видеопоследовательностях.",
    "TS_Free_Video_Memory": "Pass-through нода, которая агрессивно освобождает RAM/VRAM между тяжёлыми видео-шагами.",
    "TS_LTX_FirstLastFrame": "Добавляет guidance первого и последнего кадра в latent-пайплайн (полезно для LTX video control).",
    "TS_Animation_Preview": "Создаёт быстрый превью-ролик из кадров с опциональным объединением аудио.",
    "TS_MultiReference": "Передаёт до трёх reference-изображений в conditioning для qwen-image-edit-multi-reference и совместимых моделей. 3 опциональных IMAGE-входа + 3 опциональных IMAGE-выхода с resized-версиями (пустые слоты блокируются ExecutionBlocker).",
    "TS_FilePathLoader": "Возвращает путь к файлу и имя файла по индексу из списка папки.",
    "TS Files Downloader": "Массово скачивает модели и ассеты: resume, mirrors, proxy и опциональная распаковка.",
    "TS Youtube Chapters": "Конвертирует EDL-тайминги в готовые timestamp-главы для YouTube.",
    "TS_ModelScanner": "Сканирует файлы моделей и возвращает читаемую сводку структуры и метаданных.",
    "TS_ModelConverter": "Конвертер модели в один клик для сценариев смены precision.",
    "TS_ModelConverterAdvanced": "Расширенный конвертер моделей с детальным контролем формата, пресета и результата.",
    "TS_ModelConverterAdvancedDirect": "Расширенный конвертер, работающий напрямую от подключённого входа MODEL.",
    "TS_CPULoraMerger": "?????????? ?? ??????? LoRA ? ??????? ?????? ?? CPU ? ????????? ????? safetensors-????.",
    "TS_FloatSlider": "Простой UI float slider для аккуратного управления параметрами графа.",
    "TS_Int_Slider": "Простой UI integer slider для детерминированных целочисленных параметров.",
    "TS_Smart_Switch": "Переключает между двумя входами по режиму и помогает держать граф компактным.",
    "TS_Math_Int": "Нода целочисленной математики для счётчиков, смещений и простой логики графа.",
    "TS_Keyer": "???????????????? color-difference keyer ??? green/blue/red screen ? ?????? ?????? ? ?????????? despill.",
    "TS_Despill": "????????? ???? ?????????? spill ? ??????????? classic, balanced, adaptive ? hue_preserve.",
}


def dependency_note_to_ru(note: str | None) -> str | None:
    if not note:
        return None

    text = note
    replacements = {
        "Uses ": "Использует ",
        "plus optional": "и опционально",
        "Optional dependency": "Опциональная зависимость",
        "required": "требуется",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def build_ru_readme(cards: list[NodeCard]) -> str:
    lines: list[str] = []
    lines.append("# ComfyUI Timesaver Nodes")
    lines.append("")
    lines.append("[English](README.md) | [Русский](README.ru.md)")
    lines.append("")
    lines.append(
        "Полное и дружелюбное описание **всех нод текущего пака**. "
        "Каждая нода оформлена отдельной раскрывающейся карточкой, чтобы README было удобно читать даже новичку."
    )
    lines.append("")
    lines.append("Репозиторий: https://github.com/AlexYez/comfyui-timesaver")
    lines.append("")
    lines.append("## Установка")
    lines.append("")
    lines.append("1. Поместите папку в `ComfyUI/custom_nodes/comfyui-timesaver`.")
    lines.append("2. При необходимости установите зависимости из `requirements.txt`.")
    lines.append("3. Перезапустите ComfyUI.")
    lines.append("")
    lines.append("## ????????? ???????")
    lines.append("")
    lines.append("```text")
    lines.append("comfyui-timesaver/")
    lines.append("?? nodes/                     # ??? ???? ? node-related ???????")
    lines.append("?  ?? *.py                    # ?????????? ???")
    lines.append("?  ?? luts/                   # LUT-?????")
    lines.append("?  ?? prompts/                # ????? Prompt Builder")
    lines.append("?  ?? styles/                 # ?????? style-????????")
    lines.append("?  ?? video_depth_anything/   # ????? depth-??????")
    lines.append("?  ?? qwen_3_vl_presets.json  # ??????? ??? Qwen-????")
    lines.append("?? js/                        # Frontend-??????????")
    lines.append("?? doc/                       # ?????????? ????????????")
    lines.append("?? requirements.txt")
    lines.append("?? pyproject.toml")
    lines.append("?? __init__.py                # ????????? + startup audit")
    lines.append("```")
    lines.append("")
    lines.append("")
    lines.append("## Что есть в этом README")
    lines.append("")
    lines.append(f"- Задокументировано нод: **{len(cards)}**")
    lines.append("- Для каждой ноды есть раскрывающаяся карточка `<details>`")
    lines.append("- Для каждой ноды добавлен плейсхолдер скриншота")
    lines.append("- Описание написано простым языком, без перегруза терминами")
    lines.append("")
    lines.append("## Каталог нод")
    lines.append("")
    lines.append("| Node ID | Для чего нужна | Категория | Типы выходов |")
    lines.append("| --- | --- | --- | --- |")
    for card in cards:
        ru_desc = RU_DESCRIPTION_BY_NODE_ID.get(card.node_id, card.en_description)
        lines.append(
            f"| `{card.node_id}` | {ru_desc} | `{card.category}` | `{card.outputs}` |"
        )

    lines.append("")
    lines.append("## Подробные карточки нод")
    lines.append("")
    lines.append("### Все ноды")
    lines.append("")

    for card in cards:
        ru_desc = RU_DESCRIPTION_BY_NODE_ID.get(card.node_id, card.en_description)
        ru_dep_note = dependency_note_to_ru(card.dependency_note)

        lines.append("<details>")
        lines.append(f"<summary><strong>{card.node_id}</strong> - {ru_desc}</summary>")
        lines.append("")
        lines.append(
            f"![Плейсхолдер скриншота для {card.node_id}]({card.image_path})"
        )
        lines.append("")
        lines.append("> Плейсхолдер: замените этот блок вашим скриншотом ноды.")
        lines.append("")
        lines.append("**Что делает эта нода**")
        lines.append(ru_desc)
        lines.append("")
        lines.append("**Быстрый старт**")
        lines.append("1. Добавьте ноду в граф и подключите обязательные входы.")
        lines.append("2. Сначала оставьте дефолты и меняйте параметры постепенно.")
        lines.append("3. Подключите выход к следующей ноде и сравните результат.")
        lines.append("")
        lines.append("**Основные параметры**")
        lines.append(f"- Обязательные: {card.required}")
        lines.append(f"- Опциональные: {card.optional}")
        lines.append("")
        lines.append("**Выходы**")
        lines.append(f"- `{card.outputs}`")
        lines.append("")
        lines.append("**Техническая информация**")
        lines.append(f"- Internal id: `{card.internal_id}`")
        lines.append(f"- Class: `{card.class_name}`")
        lines.append(f"- File: `{card.file_name}`")
        lines.append(f"- Category: `{card.category}`")
        lines.append(f"- Function: `{card.function_name}`")
        if ru_dep_note:
            lines.append(f"- Примечание по зависимостям: {ru_dep_note}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    en_text = README_EN.read_text(encoding="utf-8")
    en_text = en_text.replace(
        "[English](README.md) | [???????](README.ru.md)",
        "[English](README.md) | [Russian](README.ru.md)",
    )
    README_EN.write_text(en_text, encoding="utf-8")

    cards = parse_cards(en_text)
    if not cards:
        raise RuntimeError("No node cards found in README.md")

    missing = [card.node_id for card in cards if card.node_id not in RU_DESCRIPTION_BY_NODE_ID]
    if missing:
        raise RuntimeError(f"Missing RU descriptions for nodes: {missing}")

    ru_text = build_ru_readme(cards)
    README_RU.write_text(ru_text, encoding="utf-8")

    print(f"[TS README] EN cards: {len(cards)}")
    print(f"[TS README] RU file generated: {README_RU}")


if __name__ == "__main__":
    main()
