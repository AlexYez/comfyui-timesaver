---
name: runtime_dependencies
description: "LiteRT API вместо CLI, общий SDK NVIDIA VFX и проверка бинарного ABI после смены Torch."
metadata:
  type: reference
---

LiteRT API вместо CLI, общий SDK NVIDIA VFX и проверка бинарного ABI после смены Torch.

## Правила установки

Источники версий: [optional extras](../../pyproject.toml), [requirements](../../requirements.txt), [подсказки runtime guard](../../ts_dependency_manager.py). Нативные LiteRT и NVIDIA VFX остаются необязательными, вне core и extra `all`: они нужны отдельным нодам и доступны не на всех платформах. Нельзя исправлять локальный конфликт добавлением этих SDK всем пользователям.

- LiteRT-ноды импортируют `litert_lm`, который предоставляет **`litert-lm-api`**. Одноимённый CLI-пакет `litert-lm` и `litert-lm-builder` им не нужны. Старый CLI может закреплять несовместимый API; если он не нужен другим задачам, удаляют CLI/builder, а не откатывают рабочий API. Команды — в [справке ноды](../../js/docs/TS_SuperPromptRT/ru.md).
- NVIDIA VFX устанавливают с индекса NVIDIA. Новый extra RTX согласован с [официальными requirements Nvidia VFX Nodes](https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes/blob/main/requirements.txt), которым нужен SDK не ниже 0.2. TS-адаптер понимает и старый API, но это не повод устанавливать его рядом с официальным паком. Команда и диапазон — в [справке RTX](../../js/docs/TS_RTX_Upscaler/ru.md).
- Для `cv2` подсказка предлагает `opencv-python-headless`, как core requirements; не нужно добавлять вторую GUI-сборку OpenCV ради отсутствующего импорта.
- `pip check` проверяет объявленные зависимости, но не бинарный ABI. После смены Torch проверяют **реальный вызов** CUDA-расширения. В wheel должны совпадать версия Torch, CUDA, Python ABI и платформа. Простого `import` оболочки недостаточно: `_C` бывает ленивым. Не закрепляйте локальные версии Torch или чужих расширений в требованиях TS-пака, который их не использует.

## Проверенная локальная сборка — 2026-11-02

Это снимок окружения владельца, а не минимальные требования пака. Использован Python portable ComfyUI из [описания окружения](../../doc/operations/environment.md).

| Пакет | Обнаружено | Исправление и причина |
|---|---|---|
| LiteRT | CLI 0.16.1 требовал API 0.16.1; API уже 0.19.0 | Неиспользуемые CLI и builder 0.16.1 сняты после поиска потребителей. API 0.19.0 и SHA его файлов сохранены. |
| NVIDIA VFX | 0.2.0.0, оба варианта `QualityLevel` доступны | Пакет не менялся; нижняя граница extra TS поднята с 0.1.0.1 до 0.2.0.0. |
| Torch / vision / audio | 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130 | Не менялись. CUDA Torch и xFormers 0.0.35 прошли короткое вычисление. |
| `cc_torch` | 0.2+cu130torch2.10, ошибка DLL при вызове `_C` | Заменён на 0.2+cu130torch2.11, cp312 Windows AMD64. |
| `torch_generic_nms` | 0.1+cu130torch2.10, та же ошибка DLL | Заменён на 0.1+cu130torch2.11, cp312 Windows AMD64. |

CUDA wheels взяты из того же [репозитория поставщика](https://github.com/PozzettiAndrea/cuda-wheels), что и прежние сборки; SHA256 проверены по GitHub release assets. Установка с `--no-deps` сохранила основной Torch. Старые файлы и SHA-манифесты сохранены локально в игнорируемом Git `archive/2026-11-02/` (партии `python-litert-cli-0.16.1` и `cuda-wheels-torch2.10-before`). Это резервные копии конкретной установки, а не переносимые зависимости проекта.

После замены: `pip check` чистый, CUDA connected components вернул площадь 16, NMS оставил индексы `[0, 2]`. В работающем ComfyUI TS RTX Upscaler создал новый PNG 256×256 из тестового 128×128; LiteRT E2B улучшил русский промпт и выгрузил модель при `keep_loaded=false`. Полная проверка всех сторонних нод не проводилась.
