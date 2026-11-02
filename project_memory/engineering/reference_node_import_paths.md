---
name: reference_node_import_paths
description: "`from ...ts_dependency_manager` резолвится в ComfyUI, но ломает тесты; использовать nodes/_deps.py. Патчить надо место определения, а не re-export"
metadata:
  type: reference
---

`from ...ts_dependency_manager` резолвится в ComfyUI, но ломает тесты; использовать nodes/_deps.py. Патчить надо место определения, а не re-export

Исходники темы: [_deps.py](../../nodes/_deps.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

⚠️ **Устарело в части состава нод.** 11.08.2026 с пака сняты 16 нод (см. [reference_removed_nodes_2026_08.md](../history/reference_removed_nodes_2026_08.md)); упомянутые ниже снятые ноды и их файлы больше не существуют. Остальное в записи верно.

# Импорты в нодах: ComfyUI и тесты резолвят по-разному

**ComfyUI** грузит пак как пакет → модуль ноды называется `<pack>.nodes.<категория>.<файл>`, поэтому `from ...ts_dependency_manager import` (3 точки) поднимается до корня пака и работает.

**Тесты** (`tests/test_*.py`) кладут корень пака в `sys.path` и делают `importlib.import_module("nodes.image.ts_film_grain")` → top-level пакет = `nodes`, третья точка уходит за него → `ImportError: attempted relative import beyond top-level package`.

Это всплыло на v9.77: `...`-импорт жил в 3 нодах без тестов (`ts_music_stems`, `ts_cube_to_equirect`, `ts_equirect_to_cube`), и как только его добавили в `ts_film_grain`/`ts_rtx_upscaler`/`ts_video_upscale_with_model` (у которых тесты есть) — 128 падений.

## How to apply

- **Не пиши `from ...ts_dependency_manager import` в нодах.** Используй `from .._deps import TSDependencyManager` — [`nodes/_deps.py`](../../nodes/_deps.py) держит try/except-fallback в одном месте.
- Тот же приём подойдёт любому будущему модулю в корне пака, который нужен нодам.
- Внутрипакетные импорты (`from .._hf_download import`, `from ._image_utils import`) резолвятся одинаково под обоими загрузчиками — с ними проблемы нет.

# Патчить надо место определения

`ts_bgrm_birefnet` **re-export**-ит хелперы из `_image_utils`. `monkeypatch.setattr(birefnet_module, ...)` больше не влияет на поведение — патчить надо `nodes.image._image_utils`. Так же со стабами: `tests/_engine_stubs.py` подменяет `nodes.llm._qwen_engine` целиком, поэтому **любой новый публичный символ engine, который импортируют ноды на module level, надо добавить и в стаб** — иначе весь super_prompt-набор падает с `ImportError: cannot import name ...`.

Связано: [reference_tests](reference_tests.md), [reference_pack_review_2026_07](../history/reference_pack_review_2026_07.md).
