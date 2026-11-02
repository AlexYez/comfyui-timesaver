# Архитектура comfyui-timesaver

Пак ComfyUI собирает ноды V3 и браузерные расширения без собственного сервера или frontend-сборки.

| Раздел | Что найдёте |
| --- | --- |
| [Runtime и регистрация](runtime.md) | Автообнаружение, V3 entrypoint, ошибки импорта, guard и разрешённые адаптации. |
| [Карта подсистем](subsystems.md) | Владельцы моделей, медиа, LLM, загрузок, batch и ассетов. |
| [Frontend и контракты](frontend.md) | JS entry/helpers, HTTP/events, rehydrate, тема и границы состояния. |
| [Правила реализации](../engineering/README.md) | Неподвижные workflow-контракты, V3, тензоры и DOM pitfalls. |
| [Эксплуатация](../operations/README.md) | Установка, проверки, релиз, конфигурация и диагностика. |
| [Память проекта](../../project_memory/MEMORY.md) | Причины решений и история конкретных подсистем. |

Карта сверена 2026-11-02. Актуальная версия — в
[pyproject.toml](../../pyproject.toml); публичный состав — в
[снимке контрактов](../../tests/contracts/node_contracts.json), где сейчас 88 node_id,
все api=v3. Считать нужно node_id, а не ts_*.py: часть модулей — инжекторы,
маршруты или compatibility shim, а один файл H3 регистрирует две ноды.

Основные слои:

```text
ComfyUI host
  ├─ __init__.py → discovery → node classes → comfy_entrypoint
  ├─ nodes/ → публичные схемы + приватные model/media helpers
  ├─ js/ → ComfyUI автоматически импортирует ES modules
  └─ locales/, js/docs/, example_workflows/ → пользовательские данные и справка
```

Это описание структуры, не обещание наличия всех optional-моделей на любой
машине. Реально зарегистрированный набор проверяется у текущего ComfyUI.
Снятые TS Image Studio и TS Prompt Designer отсутствуют в активном runtime;
оставшиеся js/_studio helpers используются другими нодами и не подлежат удалению
по одному имени папки.
