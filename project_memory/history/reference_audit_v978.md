---
name: reference_audit_v978
description: "История: Whole-codebase quality audit → v9.78 fixes; torch-flag scoping decision; fingerprint-mean fix intentionally SKIPPED (caching semantics = workflow breakage)"
metadata:
  type: reference
  status: historical
---

История: Whole-codebase quality audit → v9.78 fixes; torch-flag scoping decision; fingerprint-mean fix intentionally SKIPPED (caching semantics = workflow breakage)

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

⚠️ **Устарело в части состава нод.** 11.08.2026 с пака сняты 16 нод (см. [reference_removed_nodes_2026_08.md](reference_removed_nodes_2026_08.md)); упомянутые ниже снятые ноды и их файлы больше не существуют. Остальное в записи верно.

# Whole-codebase quality audit → v9.78 (2026-07)

Read-only quality audit всего пака (bugs / dead code / architecture / fragility / stack-pitfalls), затем автономная реализация ВСЕХ правок с главным ограничением пользователя: **старые workflow не должны сломаться**. Продолжает [reference_pack_review_2026_07](reference_pack_review_2026_07.md) (та вела к v9.77). Commit `8d6c16b`, push → registry на bump `pyproject` 9.77→9.78.

## Что исправлено (по группам)

- **llm/`_qwen_engine.py`**: `apply_runtime_optimizations` теперь включает ТОЛЬКО SDPA-бэкенды (безвредный глобал). TF32/matmul-precision/fp16-reduce вынесены в `@contextlib.contextmanager qwen_matmul_precision()` (save prev → set → restore в finally, no-op на non-CUDA). Генерация в `ts_qwen3_vl.py` и `super_prompt/_qwen.py` обёрнута `with engine.qwen_matmul_precision(), rng_context:`.
- **video/`ts_video_depth.py`**: `cudnn.benchmark` теперь в try/finally (сохранение prev, restore), убраны неиспользуемые поля.
- **image/`ts_smart_inpaint.py`**: удалён unreachable guider/sampler/sigmas-код (нода не имеет таких входов); `_do_sample` = только `comfy.sample.sample(...)`.
- **image/`_image_utils.py`**: `_resolve_dtype` + `_temporal_smooth_alphas` сведены сюда из дублей bgrm/vitmatte (AST-identical проверено). `ts_bgrm_birefnet` re-export'ит для back-compat.
- **image/`ts_matting_vitmatte.py`**: drop лишних каналов RGBA (`image[..., :3]`) после ndim-валидации.
- **image/`ts_wan_safe_resize.py` + `ts_qwen_safe_resize.py`**: `assert c in [3,4]` → `raise ValueError`; `int(w*scale)` → `max(target, int(round(...)))` (нет -1 crop / 1px чёрного шва).
- **image/`ts_qwen_canvas.py`**: logger + warning при batch>1 + tooltips «only FIRST frame used».
- **image/`ts_color_match.py`**: убраны dead sinkhorn-availability fallbacks (`_SUPPORTED_MODES = ["mkl","sinkhorn"]`).
- **audio/loader/`_audio_helpers.py`**: `_hash_audio_tensor` bounded (strided subsample, `_HASH_MAX_ELEMENTS=1_000_000`); TTL-cleanup stale cache-файлов (7 дней).
- **`nodes/_shared.py`**: `resolve_prompt_server(warn)` + `make_route_registrars(...)` → `(register_get, register_post)`. Переиспользуют lama_cleanup / sam_media_loader / audio_loader / super_prompt/_helpers / prompt_builder / style_prompt_selector (дедуп повторявшейся aiohttp-регистрации). См. [reference_extension_ids](../engineering/reference_extension_ids.md).
- **text/`ts_prompt_builder.py` + `ts_style_prompt_selector.py`**: pack-root через `Path(__file__).resolve().parents[2]` (переживает переименованный клон), legacy custom_nodes-scan как fallback.
- **files/`ts_downloader.py` + llm**: secrets-warning tooltips на hf_token / proxy_url / modelscope_token.

## Ключевое решение №1 — torch-флаги: scoping, НЕ удаление

Топ-находка аудита: `apply_runtime_optimizations` оставлял TF32/`set_float32_matmul_precision('high')` глобально включёнными → деградация fp32-точности/детерминизма ВСЕГО процесса (сэмплеры включительно). Не удалил оптимизацию (сохранил Qwen-ускорение), а **обернул в save/restore-скоуп**. Инвариант закреплён тестами в `test_static_invariants.py`: `test_global_torch_flags_only_written_in_scoped_files` (allowlist `_qwen_engine.py` + `ts_video_depth.py`) + `test_scoped_torch_flag_files_still_restore`. Любой новый unscoped write этих флагов упадёт. См. §17 ARCHITECTURE. Связано с [reference_device_policy](../engineering/reference_device_policy.md).

## Ключевое решение №2 — что СОЗНАТЕЛЬНО НЕ трогал (primum non nocere)

Находка «убрать `float(image.mean())` из `fingerprint_inputs` в tile/batch/size-нодах» **ОТКЛОНЕНА**. Изменение fingerprint меняет cache-ключ → ComfyUI может отдать stale-вывод на сохранённых графах = поломка workflow, что нарушает главное ограничение пользователя. Не трогать без явного запроса. (wan/qwen resize уже корректно обрабатывают RGBA через PIL-mode — там только assert→raise + round.)

## Verification (под ComfyUI Python)

- `python -m compileall .` — ok
- `python -m pytest tests` (offline) — 611 passed / 12 skipped / 1 known-fail (`test_memory_load_status_is_short_for_button_label`, numba требует NumPy ≤2.1 при env 2.5 — env-only)
- `tools/build_node_contracts.py --check` — 61 нода, snapshot без drift (контракты не менялись → workflow целы)
- live+browser против запущенного ComfyUI: `test_comfyui_live_api` + `test_browser_smoke` + `test_ideogram_editor_e2e` = **22 passed** (все ноды регистрируются, display_name/category/widget-defaults совпадают со snapshot, console чистая)
