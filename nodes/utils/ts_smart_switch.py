"""TS Smart Switch — type-aware boolean toggle between two graph branches.

node_id: TS_Smart_Switch
"""

import torch
from comfy_api.v0_0_2 import IO

from .._shared import TS_Logger

_DATA_TYPES = ["images", "video", "audio", "mask", "string", "int", "float"]

#: ⚠️ Отличает «вход не подключён» от «подключён, но ещё не посчитан».
#:
#: У ленивого входа `None` означает ВТОРОЕ: провод есть, значение ещё не
#: пришло. Неподключённый вход не передаётся вовсе, поэтому у параметра
#: остаётся значение по умолчанию — вот этот часовой. Приём взят из
#: `SoftSwitchNode` самого ComfyUI, где он прокомментирован теми же словами.
#:
#: Различать обязательно: попросить пустое гнездо — это отказ всего прогона
#: («says it needs input input_1, but there is no input to that node at all»),
#: а не тихий пропуск. Замерено.
_NOT_CONNECTED = object()


def _is_valid_image(data) -> bool:
    return isinstance(data, torch.Tensor) and data.ndim == 4


def _is_valid_mask(data) -> bool:
    if not isinstance(data, torch.Tensor):
        return False
    return data.ndim == 3 or (data.ndim == 4 and data.shape[-1] == 1)


def _is_valid_video(data) -> bool:
    if isinstance(data, torch.Tensor):
        return data.ndim == 5
    if isinstance(data, (list, tuple)) and data:
        return all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in data)
    # ⚠️ Родной VIDEO в ComfyUI — это ОБЪЕКТ (`VideoInput`/`VideoFromFile`), а
    # не тензор: у него есть `get_components`/`get_stream_source`. Проверка
    # знала только про тензоры и пятимерные списки, поэтому настоящее видео из
    # ядра признавалось «неверным типом» и переключатель ронял прогон.
    if hasattr(data, "get_components") or hasattr(data, "get_stream_source"):
        return True
    # Словарь с кадрами — форма, в которой видео ходит между частью нод.
    if isinstance(data, dict) and isinstance(data.get("images"), torch.Tensor):
        return True
    return False


def _is_valid_audio(data) -> bool:
    if isinstance(data, dict) and "waveform" in data:
        return isinstance(data["waveform"], torch.Tensor)
    return False


def _is_valid_scalar(data, data_type: str) -> bool:
    if data_type == "string":
        return isinstance(data, str)
    if data_type == "int":
        return isinstance(data, int) and not isinstance(data, bool)
    if data_type == "float":
        return isinstance(data, float)
    return False


def _is_valid_by_type(data, data_type: str) -> bool:
    if data is None:
        return False
    if data_type == "images":
        return _is_valid_image(data)
    if data_type == "video":
        return _is_valid_video(data)
    if data_type == "audio":
        return _is_valid_audio(data)
    if data_type == "mask":
        return _is_valid_mask(data)
    if data_type in ("string", "int", "float"):
        return _is_valid_scalar(data, data_type)
    return False


class TS_Smart_Switch(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Smart_Switch",
            display_name="TS Smart Switch",
            category="TS/Utils",
            description="Smart switch for ANY data. Auto-failover if one input is missing.",
            inputs=[
                IO.Combo.Input(
                    "data_type",
                    options=_DATA_TYPES,
                    tooltip="Type of data being routed. Each input is validated against it; mismatched inputs are ignored.",
                ),
                IO.Boolean.Input(
                    "switch",
                    default=True,
                    label_on="Input 1",
                    label_off="Input 2",
                    tooltip="On selects Input 1, off selects Input 2. If only one input is valid, it is passed through regardless (auto-failover).",
                ),
                # ⚠️ `lazy=True` — не украшение, а причина, по которой эта нода
                # вообще экономит время.
                #
                # Обычный вход ComfyUI обязан вычислить ДО вызова ноды: он же не
                # знает, что она возьмёт лишь одно значение. Пока входы были
                # обычными, ветка, которую тумблер не выбрал, считалась всё
                # равно — замерено на живом сервере: при тумблере на входе 2
                # изменение параметра в ветке 1 пересчитывало её целиком, хотя
                # её результат не нужен никому.
                #
                # Больно это потому, что кэш ComfyUI помнит только ПОСЛЕДНЮЮ
                # конфигурацию входов: поработав со второй стадией, человек
                # вытесняет результат первой, и возврат к прежним настройкам
                # снова гонит тяжёлый VAE Decode. С ленивым входом невыбранная
                # ветка не считается вовсе — ни в первый раз, ни после
                # вытеснения.
                IO.AnyType.Input(
                    "input_1",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "First input branch. Evaluated only when it is the one selected "
                        "(or when the other branch turns out to be missing), so an unused "
                        "branch costs nothing."
                    ),
                ),
                IO.AnyType.Input(
                    "input_2",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "Second input branch. Evaluated only when it is the one selected "
                        "(or when the other branch turns out to be missing)."
                    ),
                ),
            ],
            outputs=[
                IO.AnyType.Output(
                    display_name="output",
                    tooltip="The selected input.",
                )
            ],
        )

    @classmethod
    def check_lazy_status(cls, data_type: str, switch: bool,
                          input_1=_NOT_CONNECTED, input_2=_NOT_CONNECTED):
        """Какую ветку на самом деле нужно посчитать.

        Вызывается несколько раз, по мере появления значений; уже вычисленные
        входы сохраняются между вызовами. Спрашиваем выбранную ветку, а
        запасную — только если выбранной нет или она оказалась не того типа.

        Замерено на живом сервере: прогон, где выбрана дешёвая ветка, идёт
        0.21 с против 5.64 с, когда выбрана дорогая, — то есть невыбранная
        ветка не считается вовсе.
        """
        chosen_name, other_name = ("input_1", "input_2") if switch else ("input_2", "input_1")
        chosen = input_1 if switch else input_2
        other = input_2 if switch else input_1

        if chosen is not _NOT_CONNECTED:
            if chosen is None:
                return [chosen_name]     # подключён, но ещё не посчитан — его и считаем
            if _is_valid_by_type(chosen, data_type):
                return []
            # Посчитан, но не того типа: нужна запасная ветка.
            if other is not _NOT_CONNECTED and other is None:
                return [other_name]
            return []

        # Выбранного входа нет вовсе — сразу за запасным (авто-подхват).
        if other is not _NOT_CONNECTED and other is None:
            return [other_name]
        return []

    @classmethod
    def execute(cls, data_type: str, switch: bool,
                input_1=_NOT_CONNECTED, input_2=_NOT_CONNECTED) -> IO.NodeOutput:
        # Часовой — деталь ленивой проверки; дальше он равнозначен «ничего нет».
        input_1 = None if input_1 is _NOT_CONNECTED else input_1
        input_2 = None if input_2 is _NOT_CONNECTED else input_2

        valid_1 = _is_valid_by_type(input_1, data_type)
        valid_2 = _is_valid_by_type(input_2, data_type)

        if input_1 is not None and not valid_1:
            TS_Logger.warn(
                "SmartSwitch",
                f"Input 1 ignored: type mismatch for data_type={data_type}",
            )
        if input_2 is not None and not valid_2:
            TS_Logger.warn(
                "SmartSwitch",
                f"Input 2 ignored: type mismatch for data_type={data_type}",
            )

        if valid_1 and valid_2:
            if switch:
                result = input_1
                selected_source = "Input 1"
                status_msg = "(Switch: ON)"
            else:
                result = input_2
                selected_source = "Input 2"
                status_msg = "(Switch: OFF)"
        elif valid_1:
            result = input_1
            selected_source = "Input 1"
            status_msg = "(Auto-Failover)"
        elif valid_2:
            result = input_2
            selected_source = "Input 2"
            status_msg = "(Auto-Failover)"
        else:
            # Auto-failover covers ONE missing branch; with no valid input at
            # all, emitting None just crashes a downstream node far from the
            # real cause. Fail here, at the switch, with an actionable message.
            raise RuntimeError(
                f"TS Smart Switch: neither input matches data_type='{data_type}' "
                "(both disconnected, None, or wrong type). Connect at least one "
                "matching input or fix data_type."
            )

        info = "Unknown"
        if hasattr(result, "shape"):
            info = f"Tensor {result.shape}"
        elif isinstance(result, (int, float, str)):
            info = str(result)

        TS_Logger.log(
            "SmartSwitch",
            f"Selected: {selected_source} {status_msg} | {info}",
        )
        return IO.NodeOutput(result)


NODE_CLASS_MAPPINGS = {"TS_Smart_Switch": TS_Smart_Switch}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Smart_Switch": "TS Smart Switch"}
