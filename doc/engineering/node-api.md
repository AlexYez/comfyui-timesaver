# Ноды и контракт V3

Публичная нода описывает схему и execute в одном основном файле; тяжёлый движок может быть приватным помощником.

## API и регистрация

Ноды используют `from comfy_api.v0_0_2 import IO` (при необходимости UI),
`IO.ComfyNode`, classmethod define_schema и classmethod execute → IO.NodeOutput.
Версионированный namespace не является обещанием стабильности upstream API.
Не заменяйте его подвижным comfy_api.latest в самих нодах. Отдельное действующее
исключение: корневой загрузчик импортирует ComfyExtension из latest для
определения поддержки V3-регистрации; это не V1-нода.

Существующие публичные классы зарегистрированы через NODE_CLASS_MAPPINGS;
корень собирает их и в современном ComfyUI отдаёт через comfy_entrypoint.
Подробности — [загрузчик](../architecture/runtime.md).

```python
from comfy_api.v0_0_2 import IO

class TS_ExampleNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ExampleNode", display_name="TS Example Node",
            category="TS/Image/Retouch",
            inputs=[IO.Image.Input("image")],
            outputs=[IO.Image.Output("image")],
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        return IO.NodeOutput(image.clone())

NODE_CLASS_MAPPINGS = {"TS_ExampleNode": TS_ExampleNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ExampleNode": "TS Example Node"}
```

Пример показывает форму API, не разрешает добавлять ещё одну простую clone-ноду.
Custom-типы: IO.Custom; wildcard: IO.AnyType. Для output-ноды — outputs=[] и
is_output_node=True. Для списков входов — is_input_list=True в схеме;
для отдельного списочного выхода — is_output_list=True на IO.Output.

## State и кэш

ComfyUI клонирует и блокирует классы нод: присваивание cls.model = ... может
упасть на lock_class. Обычный вариант — private _NodeState на уровне модуля,
с отдельным экземпляром _state, блокировкой и явным unload.
Это разрешённое состояние, а не запрещённый «глобальный скрытый singleton».
Примеры: [BiRefNet](../../nodes/image/ts_bgrm_birefnet.py),
[Qwen](../../nodes/llm/_qwen_engine.py).

Существующий второй вариант — in-place изменение объявленного class-dict:
cls._model_cache[key] = value. Это не rebinding атрибута; поверхностный клон
делит dict с исходным классом. Такие кэши у [Silero Stress](../../nodes/text/ts_silero_stress.py)
и [Silero TTS](../../nodes/audio/ts_silero_tts.py) не нужно переписывать без причины.
Для памяти, живущей вне ModelPatcher, зарегистрировать функцию освобождения
через register_memory_release в [общем модуле](../../nodes/_shared.py) и соблюдать
блокировку владельца: закрытие LiteRT во время генерации уже приводило к access violation.

## Naming и раскладка

Новый основной файл: ts_<имя>.py; класс/node_id: TS_<Name>; display_name:
TS <Name>; JS extension ID стабилен и уникален. Публичный файл с `_` в начале
или в `_`-папке загрузчик пропустит. Существующие идентификаторы без подчёркивания
или с пробелами — совместимые исторические имена, не материал для переименования.

Нода выбирает существующую папку категории: image, video, audio, llm, text,
files, utils, loaders, conditioning или ideogram. Новую полку не заводят молча.
Большая нода может жить в подпапке feature/: основной ts_*.py хранит **весь
публичный класс**, рядом private _helpers/_routes/_engine, а не schema.py и execute.py.
Shared-логика для нескольких нод живёт рядом с владельцами; общая для разных
категорий — в nodes/. Связанная frontend-нода имеет entry JS и private ES helpers.

**Существующее исключение:**
[ts_h3_regenerate.py](../../nodes/video/latent_upscale/ts_h3_regenerate.py)
регистрирует TS_H3RegenerateModel и TS_H3Regenerate в одном файле. Это факт
нынешнего кода и снимка контрактов. Документационная уборка его не меняет;
общий принцип one-node-one-file не является поручением немедленно дробить этот файл.

## Validation, fingerprint, lazy

validate_inputs возвращает True или текст ошибки. Именованный параметр в этом
методе означает, что его проверку берёт на себя нода; **kwargs может отключить
проверку ядром всех входов. При проверке провода значения могут быть None:
нельзя делать width % 32 до проверки типа и наличия значения.

Для нового кода — явные параметры и только реально реализуемые проверки.
В существующих Qwen/RT нодах permissive-validation сохраняет старые имена
пресетов и моделей: её ужесточение может сломать старый граф. Сначала прочитать
причину в соответствующем методе, а не исправлять сигнатуру по общему правилу.

fingerprint_inputs включает все влияющие на результат входы и внешнее состояние
(для файла — путь, размер/mtime либо содержимое по нужному контракту).
Новый вход, отсутствующий в явной сигнатуре fingerprint_inputs, может лишить
ноду правильного cache key. Постоянный fingerprint подходит только действительно
неизменному результату. check_lazy_status не загружает модель и не делает inference.
Скрытые входы явно декларируются в схеме и читаются через cls.hidden.

## Данные, которые являются списками

ComfyUI исполняет список «вширь»: все элементы одной ноды раньше следующей.
Поэтому batch-list не гарантирует инкрементальный save в конце каждой задачи.
У [TS Batch Write](../../nodes/files/ts_batch_write.py) нельзя добавлять is_input_list:
это заставит принять весь список разом. Для пошаговых заданий у источника
есть one_per_run. Конкретная семантика — в коде
[источника](../../nodes/utils/ts_batch_source.py) и [индексе памяти](../../project_memory/MEMORY.md).
