"""TS Smart Batch — batch two images, tolerating a missing or disabled input.

node_id: TS_SmartBatch

Почему не хватает встроенной ``Batch Images``. У неё оба входа ОБЯЗАТЕЛЬНЫЕ:
отключил один — и весь граф падает с ошибкой ещё до запуска. А типичная работа
устроена ровно наоборот: собираешь пару «первый кадр / последний кадр» и хочешь
свободно выключать то одну сторону, то другую, не перекладывая связи. Здесь оба
входа необязательные, и результат зависит от того, что реально пришло:

* пришли оба — батч из двух (или из их батчей: 3 + 2 дадут 5);
* пришёл один — он и уходит на выход, как есть;
* не пришло ничего — понятная ошибка, а не пустая картинка, которая
  притворится результатом.

Согласование пары сделано ТАК ЖЕ, как в ядре (``nodes.ImageBatch``): не хватает
альфы — канал добавляется единицей, размеры разошлись — второй кадр
подгоняется под первый билинейно. Это делает ноду заменой встроенной без
сюрпризов.
"""

import logging
from typing import Any, Optional

import comfy.utils
import torch
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_smart_batch")
LOG_PREFIX = "[TS Smart Batch]"


class TS_SmartBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_SmartBatch",
            display_name="TS Smart Batch",
            category="TS/Image/Batch",
            description=(
                "Batch two images without breaking when one of them is missing. "
                "Both inputs are optional: connect both and you get a batch of two, "
                "connect one and it passes through untouched. Made for first/last "
                "frame pairs you want to switch on and off without rewiring."
            ),
            inputs=[
                # Входы РАСТУТ: занял последний — появляется следующий, как у
                # родной Batch Images. Два показываются сразу, дальше по мере
                # подключения, потолок — тридцать два.
                IO.Autogrow.Input(
                    "images",
                    optional=True,
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.Image.Input(
                            "image",
                            # ⚠️ optional=True НА ШАБЛОНЕ — не украшение.
                            #
                            # Ядро раздаёт слоты так: первые `min` штук попадают
                            # в required, если шаблон объявлен обязательным
                            # (`_expand_schema_for_dynamic`). С обязательным
                            # шаблоном `image0` снова становится обязательным —
                            # то есть возвращается ровно та беда встроенной
                            # Batch Images, ради которой нода и написана:
                            # «Required input is missing: image0» ещё до запуска.
                            # Поймано живым прогоном, статикой не видно.
                            optional=True,
                            tooltip=(
                                "An image for the batch. Every slot is optional: "
                                "leave one empty (or bypass the node feeding it) and "
                                "the rest are batched without it. Sizes are matched "
                                "to the first image that actually arrives."
                            ),
                        ),
                        prefix="image",
                        min=2,
                        max=32,
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="image",
                    tooltip=(
                        "Batch of whatever actually arrived: both images stacked, "
                        "or the single one that was connected."
                    ),
                )
            ],
            search_aliases=[
                "smart batch",
                "optional batch images",
                "batch images optional",
                "combine images",
                "first last frame",
            ],
        )

    @staticmethod
    def _log(message: str) -> None:
        logger.info("%s %s", LOG_PREFIX, message)

    @classmethod
    def _usable_image(cls, value: Any, label: str) -> Optional[torch.Tensor]:
        """Картинка, пригодная к работе, либо ``None``.

        Отсутствующий вход — штатный случай, а не ошибка: ради него нода и
        написана. Но сюда может прийти и мусор — например, если вход накормлен
        через wildcard-переходник или нода-источник вернула ``None`` вместо
        картинки. Такое тоже пропускаем молча (с записью в журнал), иначе
        «не падать при отключённом входе» превращается в «падать при странном».
        """
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            cls._log(f"{label}: ignored, expected an IMAGE tensor but got {type(value).__name__}")
            return None
        if value.ndim == 3:
            # Одиночный кадр без батч-измерения — обычная форма у части нод.
            value = value.unsqueeze(0)
        if value.ndim != 4:
            cls._log(f"{label}: ignored, expected [B,H,W,C] but got {tuple(value.shape)}")
            return None
        if value.shape[0] == 0:
            cls._log(f"{label}: ignored, the batch is empty")
            return None
        return value

    @staticmethod
    def _match_channels(first: torch.Tensor, second: torch.Tensor):
        """Довести пару до общего числа каналов.

        Тот же приём, что в ядре: недостающая альфа дописывается единицей —
        «непрозрачно». Так RGB и RGBA складываются в один батч без отказа.
        """
        if first.shape[-1] == second.shape[-1]:
            return first, second
        if first.shape[-1] > second.shape[-1]:
            second = torch.nn.functional.pad(second, (0, 1), mode="constant", value=1.0)
        else:
            first = torch.nn.functional.pad(first, (0, 1), mode="constant", value=1.0)
        return first, second

    @staticmethod
    def _slot_order(name: str) -> tuple:
        """Ключ сортировки входов ПО НОМЕРУ, а не по алфавиту.

        ⚠️ Растущие входы зовутся `image0`, `image1`, ... `image10`, и обычный
        `sorted()` ставит `image10` ПЕРЕД `image2`. Для батча это не мелочь:
        порядок кадров и есть результат — первый кадр обязан быть первым.
        (В ядре так и написано — `sorted(reference_images)`, — но там потолок
        восемь входов, и до десятого дело не доходит.)
        """
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        return (0, int(digits)) if digits else (1, 0), str(name)

    @classmethod
    def execute(cls, images=None) -> IO.NodeOutput:
        slots = dict(images or {})
        usable = []
        for name in sorted(slots, key=cls._slot_order):
            image = cls._usable_image(slots[name], name)
            if image is not None:
                usable.append(image)

        if not usable:
            raise RuntimeError(
                f"{LOG_PREFIX} No images arrived — there is nothing to batch. "
                f"Connect at least one image."
            )

        if len(usable) == 1:
            cls._log(f"one image connected -> passing {tuple(usable[0].shape)} through")
            return IO.NodeOutput(usable[0])

        # Первый пришедший задаёт форму; остальные подгоняются под него — так же,
        # как это делает встроенная Batch Images.
        batch = usable[0]
        for nxt in usable[1:]:
            batch, nxt = cls._match_channels(batch, nxt)
            if batch.shape[1:] != nxt.shape[1:]:
                cls._log(
                    f"resizing {tuple(nxt.shape[1:3])} -> {tuple(batch.shape[1:3])}"
                )
                nxt = comfy.utils.common_upscale(
                    nxt.movedim(-1, 1), batch.shape[2], batch.shape[1],
                    "bilinear", "center",
                ).movedim(1, -1)
            batch = torch.cat(
                (batch, nxt.to(dtype=batch.dtype, device=batch.device)), dim=0)

        cls._log(f"batched {len(usable)} input(s) -> {batch.shape[0]} frame(s)")
        return IO.NodeOutput(batch)


NODE_CLASS_MAPPINGS = {"TS_SmartBatch": TS_SmartBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SmartBatch": "TS Smart Batch"}
