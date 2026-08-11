"""TS LoRA Loader — стопка LoRA для модели, одной нодой.

Зачем нода, если в ComfyUI уже есть загрузчик. Одна LoRA — одна нода, и цепочка
из шести штук занимает пол-экрана, а чтобы поменять их порядок, надо
перекладывать провода. Здесь список живёт внутри одной ноды: плюс добавляет,
перетаскивание меняет порядок, ползунок — силу.

⚠️ САМА ЭТА НОДА LoRA НЕ ГРУЗИТ. Она разворачивается в цепочку РОДНЫХ
`LoraLoaderModelOnly` (`enable_expand` + `GraphBuilder`) — тех самых, что
ComfyUI поставляет и поддерживает. Отсюда два следствия, ради которых всё и
затевалось: поведение один в один совпадает с ручной цепочкой, а кэш ComfyUI
работает по каждому звену отдельно — правка силы последней LoRA не заставляет
пересчитывать предыдущие.

Только модель, без CLIP: это осознанное сужение под `LoraLoaderModelOnly`.
Текстовый энкодер у современных семейств живёт отдельно, и половина LoRA в
обиходе — чисто модельные.

Список хранится ОДНОЙ строкой JSON (`loras_json`), а не набором виджетов.
`widgets_values` в ComfyUI позиционный: виджеты, появляющиеся и исчезающие
вместе со строками списка, сдвигали бы значения всех соседей при каждой
загрузке чужого workflow. Одна строка — одна позиция, сколько бы LoRA в ней ни
лежало.
"""

from __future__ import annotations

import json
import logging

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_lora_loader")
LOG_PREFIX = "[TS LoRA Loader]"

NATIVE_LOADER = "LoraLoaderModelOnly"

# ⚠️ Запасные границы, а НЕ источник истины. Настоящие спрашиваются у родной
# ноды (`strength_bounds`): нода разворачивается именно в неё, и значение,
# принятое здесь, но отвергнутое там, было бы обманом. Сюда доходит только
# случай «родной ноды не нашлось» — например, тест без ComfyUI.
# Отрицательная сила законна: так гасят влияние LoRA, зашитой в чекпоинт.
FALLBACK_STRENGTH_MIN = -100.0
FALLBACK_STRENGTH_MAX = 100.0

# Ответ ядра за один запуск не меняется, а INPUT_TYPES у некоторых нод недёшев.
# Мутация словаря, а не переприсваивание: у V3-нод класс заперт (§5 CLAUDE.md),
# и модульный словарь — санкционированный способ хранить такой кэш.
_bounds_cache: dict[str, tuple[float, float]] = {}


def _native_strength_bounds() -> tuple[float, float] | None:
    """Границы силы из схемы родной ноды, если её удалось прочитать.

    Понимает обе формы схемы: сегодняшнюю V1 (``INPUT_TYPES``) и V3
    (``define_schema``) — на случай, если ядро переведёт загрузчик.

    Returns:
        Пара ``(min, max)`` или ``None``, если схему прочитать не вышло.
    """
    try:
        import nodes

        native = nodes.NODE_CLASS_MAPPINGS.get(NATIVE_LOADER)
        if native is None:
            return None

        low = high = None
        if hasattr(native, "define_schema"):
            for entry in getattr(native.define_schema(), "inputs", []):
                if getattr(entry, "id", None) == "strength_model":
                    low, high = getattr(entry, "min", None), getattr(entry, "max", None)
                    break
        if low is None and hasattr(native, "INPUT_TYPES"):
            spec = native.INPUT_TYPES().get("required", {}).get("strength_model")
            if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
                low, high = spec[1].get("min"), spec[1].get("max")

        low, high = float(low), float(high)
        return (low, high) if low < high else None
    except Exception as error:              # noqa: BLE001 - вне ComfyUI и в тестах
        logger.debug("%s could not read %s bounds: %s", LOG_PREFIX, NATIVE_LOADER, error)
        return None


def strength_bounds() -> tuple[float, float]:
    """Допустимая сила — ровно та, что принимает родная нода.

    Своих чисел здесь нет намеренно: зашитая копия разошлась бы с ComfyUI при
    первом же его обновлении, и нода начала бы резать значения, которые ядро
    приняло бы спокойно.
    """
    cached = _bounds_cache.get(NATIVE_LOADER)
    if cached is None:
        cached = _native_strength_bounds() or (FALLBACK_STRENGTH_MIN, FALLBACK_STRENGTH_MAX)
        _bounds_cache[NATIVE_LOADER] = cached
    return cached


def parse_stack(raw: str) -> list[dict]:
    """Разобрать список LoRA из строки JSON.

    Формат: ``[{"name": "x.safetensors", "strength": 0.8, "on": true}, ...]``.

    Битую строку не считаем ошибкой прогона: workflow мог прийти от чужой
    сборки или из более новой версии пака. Пустой список означает «просто
    пропусти модель дальше», и это честнее, чем упасть.

    Args:
        raw: содержимое виджета ``loras_json``.

    Returns:
        Список записей с приведёнными типами; выключенные и безымянные
        отброшены.
    """
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):
        logger.warning("%s could not read the LoRA list, treating it as empty", LOG_PREFIX)
        return []
    if not isinstance(data, list):
        return []

    low, high = strength_bounds()
    stack = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        # Выключенная строка остаётся в списке — человек её отложил, а не
        # удалил, — но в граф не попадает.
        if entry.get("on") is False:
            continue
        try:
            strength = float(entry.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        strength = max(low, min(high, strength))
        # Нулевая сила — это отсутствие влияния. Пропускать её значит не
        # загружать файл вовсе: заметно быстрее и ровно то же самое на выходе.
        if strength == 0.0:
            continue
        stack.append({"name": name, "strength": strength})
    return stack


def known_loras() -> list[str]:
    """Имена LoRA, которые видит эта сборка ComfyUI."""
    try:
        import folder_paths

        return list(folder_paths.get_filename_list("loras"))
    except Exception as error:              # noqa: BLE001 - вне ComfyUI и в тестах
        logger.debug("%s could not list loras: %s", LOG_PREFIX, error)
        return []


class TS_LoraLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LoraLoader",
            display_name="TS LoRA Loader",
            category="TS/utils",
            description=(
                "A stack of model-only LoRAs in one node: add with the plus "
                "button, drag to reorder, set strength (negative allowed). "
                "Expands into a chain of native LoraLoaderModelOnly nodes, so "
                "behaviour and caching match a hand-built chain exactly."
            ),
            # Разворачивается в цепочку родных загрузчиков — см. модуль сверху.
            enable_expand=True,
            inputs=[
                IO.Model.Input("model"),
                IO.String.Input(
                    "loras_json",
                    default="[]",
                    socketless=True,
                    tooltip=(
                        "Internal field: the LoRA stack as JSON. The node's own "
                        "interface writes it; there is nothing to type here."
                    ),
                ),
            ],
            outputs=[IO.Model.Output(display_name="MODEL")],
            search_aliases=["lora", "lora stack", "loras", "lora loader"],
        )

    @classmethod
    def execute(cls, model, loras_json: str = "[]") -> IO.NodeOutput:
        from comfy_execution.graph_utils import GraphBuilder

        stack = parse_stack(loras_json)
        if not stack:
            # Пустая стопка — не ошибка: нода в графе может стоять заранее.
            return IO.NodeOutput(model)

        available = set(known_loras())
        graph = GraphBuilder()
        current = model
        used = 0
        for entry in stack:
            if available and entry["name"] not in available:
                # Файла нет на этой машине. Пропускаем именно эту LoRA, а не
                # валим прогон: остальные восемь работать могут.
                logger.warning("%s '%s' is not in the loras folder — skipped",
                               LOG_PREFIX, entry["name"])
                continue
            loader = graph.node(
                NATIVE_LOADER,
                model=current,
                lora_name=entry["name"],
                strength_model=entry["strength"],
            )
            current = loader.out(0)
            used += 1

        if not used:
            return IO.NodeOutput(model)
        logger.info("%s %d LoRA(s) applied", LOG_PREFIX, used)
        return IO.NodeOutput(current, expand=graph.finalize())


NODE_CLASS_MAPPINGS = {"TS_LoraLoader": TS_LoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LoraLoader": "TS LoRA Loader"}
