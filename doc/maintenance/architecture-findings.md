# Консолидация архитектуры и правил

Архитектура пересверена по текущим исходникам, а подробные правила вынесены из корневого CLAUDE в тематические модули.

Проверка выполнена 2026-11-02. Это документационная работа: исходники,
схемы, зависимости и поведение runtime не изменялись этой частью задачи.

## Что перенесено

| Источник | Итог |
| --- | --- |
| Прежний doc/ARCHITECTURE.md, 83 653 байта | Короткий стабильный [указатель](../ARCHITECTURE.md) и [architecture hub](../architecture/README.md) с runtime, subsystems и frontend. |
| Подробные правила прежнего CLAUDE.md | [Engineering hub](../engineering/README.md): workflow/совместимость, node API, backend и DOM patterns. Корневые инструкции собираются отдельно. |
| Прежний doc/THEME.md, 27 788 байт | Стабильный [указатель](../THEME.md), [theme hub](../theme/README.md), components, localization и overlays. |
| Локальные абсолютные DEV/runtime пути | Ссылка на [environment](../operations/environment.md); они больше не объявлены универсальной структурой клона. |
| Команды тестов, установки, релиза и чтения логов | Указатели на [operations](../operations/README.md) вместо второй копии инструкций. Упоминания локальных test/tool файлов явно не обещают их наличие в чистом клоне. |

Оригиналы трёх крупных документов до записи сохранены в локальном
`tests/.cache/memory-consolidation/architecture-originals.zip` только для
сравнения сохранности. Этот кэш не является источником действующих правил.

## Исправленные утверждения

| Раньше | Теперь | Доказательство |
| --- | --- | --- |
| Архитектура называла 12.5.0 и одновременно 79/71/80 нод; CLAUDE — 12.13.2 и 85/84 ноды. | Версия берётся из pyproject, публичные node_id — из snapshot; на момент сверки 88, все V3. | [pyproject](../../pyproject.toml), [контракты](../../tests/contracts/node_contracts.json). |
| Любой ImportError → SKIPPED. | Только распознанная отсутствующая внешняя зависимость → SKIPPED; сломанный внутренний import → ERROR. | _load_module и _is_internal_module_name в [loader](../../__init__.py). |
| Аудит импортов — поиск строк import. | AST-разбор, включая lazy imports; docstrings не превращаются в пакеты. | _extract_import_roots_from_source в [loader](../../__init__.py). |
| Корневые mappings всегда доступны как итог регистрации. | При доступном V3 host корень удаляет mappings и отдаёт comfy_entrypoint; GET_SCHEMA проверяется для каждой ноды отдельно. | TimesaverExtension/get_node_list и del NODE_CLASS_MAPPINGS в [loader](../../__init__.py). |
| Runtime guard возвращает typed fallback для V1. | Действующий guard оборачивает classmethod execute, сохраняет async и бросает RuntimeError; typed fallback отсутствует. | [TSDependencyManager](../../ts_dependency_manager.py). |
| Любой frontend JS — IIFE; общая структура студии существует. | Публичные entry — ES modules с registerExtension, рядом private ES helpers. Из js/_studio остались _dnd и _compare, активной студии нет. | [frontend tree](../../js/), [DnD](../../js/_studio/_dnd.js), [Compare](../../js/_studio/_compare.js). |
| Для DOM нельзя задавать computeSize вообще. | Его нельзя вручную задавать для Vue. Общий helper задаёт computeSize в legacy и height bounds в Vue. | addResizableDomWidget в [DOM helper](../../js/_dom_widget.js). |
| Все данные темы читаются только из четырёх переменных host. | Основная палитра использует четыре значения; статический muted fallback также читает descrip-text. | themeCss в [теме](../../js/_theme.js). |
| Запрещён module-level mutable state, но одновременно предписан _NodeState. | Явный private state/cache с владельцем и lock допустим; class-dict in-place тоже остаётся действующим паттерном. | [Qwen](../../nodes/llm/_qwen_engine.py), [Silero TTS](../../nodes/audio/ts_silero_tts.py), [Silero Stress](../../nodes/text/ts_silero_stress.py). |
| Music Stems в nodes/audio/ts_music_stems.py приведена как class-dict cache. | Нынешняя нода в music_stems/ с module-level _model_cache; пример class-dict заменён Silero Stress. | [Music Stems](../../nodes/audio/music_stems/ts_music_stems.py), [Silero Stress](../../nodes/text/ts_silero_stress.py). |
| Prompt Builder находится в prompt_builder/ и сканирует семантические папки через _packs/_scene. | Нынешний ts_prompt_builder.py сканирует непосредственные *.txt из nodes/prompts и читает порядок блоков из config_json/config файла. | ts_list_prompt_files, ts_parse_config_data, ts_prompt_builder_config_path в [текущем файле](../../nodes/text/ts_prompt_builder.py). |
| Каждая публичная нода всегда в отдельном файле. | Это правило нового кода; уже есть исключение: две H3 Regenerate-ноды в одном файле. Уборка документации его не меняет. | [H3 mappings](../../nodes/video/latent_upscale/ts_h3_regenerate.py), [snapshot](../../tests/contracts/node_contracts.json). |
| LATENT всегда samples [B,C,H,W]. | Это типично для изображения; видео может быть 5D, H3 аудио+видео — NestedTensor. | [H3 regenerate](../../nodes/video/latent_upscale/ts_h3_regenerate.py), [H3 core](../../nodes/video/latent_upscale/_h3_core.py). |
| Единственная существующая адаптация чужого кода — pasted media fix. | _apply_core_patches также подключает Windows asyncio proactor guard. Новые патчи всё равно требуют отдельной задачи. | [loader](../../__init__.py), [proactor guard](../../compat/proactor_guard.py). |

## Границы проверки

Проверены файловые пути и relative links созданных модулей, размеры UTF-8 и
указанные определения в исходниках. Ссылки на новые operations, node hub и
root AGENTS зависят от завершения параллельных частей; окончательный проход
проводится по всему дереву после их объединения.

Упомянутые исторические замеры UI/памяти сохранены как объяснения инцидентов,
не повторены и не объявлены универсальными характеристиками. GPU inference,
браузерные сценарии и тестовый набор для документационной правки не запускались.
